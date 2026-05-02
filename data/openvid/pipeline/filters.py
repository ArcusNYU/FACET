"""
Dataset Pipeline Step 1: caption-level single-person filter.
Reference: https://huggingface.co/cross-encoder/nli-deberta-v3-small

Stage 1 (regex, ~free):
    Reject captions hitting MULTI patterns (over-reject is preferred).
    Only "single"/"alone"/"solo" is treated as a confident single-person signal;
    every other potentially-single phrase falls through to NLI because
    "a man, ..., with a woman" can fool simple keyword rules.

Stage 2 (NLI, ~ms on A100 with big batch):
    Run a batched ONNX NLI model (cross-encoder/nli-DeBerta-v3-small).
    Hypotheses and thresholds are LOCKED, do not edit.

Output:
    `<stem>.single.csv` next to the input csv (or under --out-dir),
    with an extra `single` column (True/False) and the original columns preserved. 
    Trailing `Unnamed:` padding columns are dropped.

This script is meant to be run ONCE per CSV before prepare.py.
"""

from __future__ import annotations
import argparse
import glob as _glob
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from data.utils import load_cfg


# ---- Stage 1: regex pre-filter ----
MULTI_PATTERNS = [
    r"\btwo\b",
    r"\bthree\b",
    r"\bfour\b",
    r"\bseveral\b",
    r"\bmultiple\b",
    r"\bmany\b",
    r"\bgroup\b",
    r"\bcrowd\b",
    r"\bpeople\b",
    r"\bindividuals\b",
    r"\bpersons\b",
    r"\ba man and\b",
    r"\ba woman and\b",
    r"\bman and woman\b",
    r"\btwo men\b",
    r"\btwo women\b",
    r"\bcouple\b",
]

# Only "single"/"alone"/"solo" is reliable: e.g. "a single woman ..." or "single individual".
SINGLE_PATTERNS = [
    r"\bsingle\b",
    r"\balone\b",
    r"\bsolo\b",
]

_MULTI_RE = [re.compile(p) for p in MULTI_PATTERNS]
_SINGLE_RE = [re.compile(p) for p in SINGLE_PATTERNS]


def caption_rule(caption: str) -> Optional[bool]:
    """Fast regex-based pre-filter.
    Returns:
        True  -> confidently single person (mentions "single ...")
        False -> confidently multi person  (matched MULTI_PATTERNS)
        None  -> undecided, send to NLI
    MULTI takes precedence over SINGLE (prefer over-reject over letting multi in).
    """
    c = caption.lower()
    for r in _MULTI_RE:
        if r.search(c):
            return False
    for r in _SINGLE_RE:
        if r.search(c):
            return True
    return None


# ---- Stage 2: NLI batch inference ----
class SinglePersonNLI:
    """
    Both Positive and Negative Hypotheses, and threshold values must NOT be changed in any condition if specified.
    """

    POSITIVE_HYPOTHESES = [
        "The caption describes one person.",
        "The caption is about a single person.",
        "The caption focuses on one individual.",
    ]

    NEGATIVE_HYPOTHESES = [
        "The caption describes multiple people.",
        "The caption says there is a second person.",
        "The caption says the person is accompanied by someone else.",
    ]

    POSITIVE_THRESHOLD = 0.65
    NEGATIVE_THRESHOLD = 0.40
    MARGIN_THRESHOLD = 0.25

    LABELS = ["contradiction", "entailment", "neutral"]
    ENTAILMENT_ID = 1

    def __init__(
        self,
        model_dir: str,
        onnx_filename: str = "model.onnx",
        provider: str = "cuda",
        max_length: int = 256,
    ):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if provider == "cuda"
            else ["CPUExecutionProvider"]
        )
        path = Path(model_dir) / "onnx" / onnx_filename
        onnx_path = str(path) if path.exists() else str(Path(model_dir) / onnx_filename)
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        # Enquiry standard input names of the specified ONNX model:
        self.input_names = [x.name for x in self.session.get_inputs()]  

    @staticmethod
    def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
        logits = logits - np.max(logits, axis=axis, keepdims=True)
        exp = np.exp(logits)
        return exp / np.sum(exp, axis=axis, keepdims=True)

    def _predict_pairs(self, premises: List[str], hypotheses: List[str]) -> np.ndarray:
        """Batched (premise, hypothesis) -> probs [B, 3]."""
        encoded = self.tokenizer(
            premises, hypotheses,
            padding=True, truncation=True,
            max_length=self.max_length, return_tensors="np",
            # ONNX runtime requires numpy
        )
        ort_inputs = {n: encoded[n] for n in self.input_names if n in encoded}
        logits = self.session.run(None, ort_inputs)[0]
        return self._softmax(logits, axis=-1)  # p_contradict + p_entail + p_neutral = 1

    def is_single_batch(self, captions: List[str]) -> List[bool]:
        """Batched equivalent of is_single_person_video_caption in nli.py."""
        N = len(captions)
        if N == 0:
            return []
        P = len(self.POSITIVE_HYPOTHESES)
        G = len(self.NEGATIVE_HYPOTHESES)

        # Construct N*P positive pairs and N*G negative pairs
        pos_prem, pos_hypo = [], []
        neg_prem, neg_hypo = [], []
        # primises are always captions
        for c in captions:
            for h in self.POSITIVE_HYPOTHESES:
                pos_prem.append(c); pos_hypo.append(h)
            for h in self.NEGATIVE_HYPOTHESES:
                neg_prem.append(c); neg_hypo.append(h)

        pos_probs = self._predict_pairs(pos_prem, pos_hypo).reshape(N, P, 3)
        neg_probs = self._predict_pairs(neg_prem, neg_hypo).reshape(N, G, 3)

        pos_ent = pos_probs[..., self.ENTAILMENT_ID]   # [N, P]
        neg_ent = neg_probs[..., self.ENTAILMENT_ID]   # [N, G]
        # max probability of the entailment label for each caption:
        # as long as there is any wording that makes the model strongly believe the hypothesis, it is given a high score.
        pos_score = pos_ent.max(axis=-1)               # [N]
        neg_score = neg_ent.max(axis=-1)               # [N]

        single = (
            (pos_score >= self.POSITIVE_THRESHOLD) &
            (neg_score <= self.NEGATIVE_THRESHOLD) &
            ((pos_score - neg_score) >= self.MARGIN_THRESHOLD)
        )
        return single.tolist()


# ---- top-level annotate_csv ----
def _drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    """HQ-OpenHumanVid csvs have trailing empty-header columns that pandas
    renames to Unnamed: N. We drop them to keep the output tidy."""
    return df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")]


def annotate_csv(
    csv_path: str,
    out_path: str,
    nli: SinglePersonNLI,
    batch_size: int = 128,
    caption_col: str = "caption",
    out_col: str = "single",
    use_regex: bool = True,
) -> dict:
    """Read a CSV, append a `single` bool column, write it out.
    Returns a stats dict."""
    df = _drop_unnamed(pd.read_csv(csv_path))
    if caption_col not in df.columns:
        raise KeyError(f"caption column '{caption_col}' not in {csv_path}; got {list(df.columns)}")

    captions = df[caption_col].astype(str).tolist()
    N = len(captions)
    results: List[Optional[bool]] = [None] * N
    pending_idx: List[int] = []

    if use_regex:
        for i, c in enumerate(captions):
            r = caption_rule(c)
            if r is not None:
                results[i] = r
            else:
                pending_idx.append(i)
    else:
        pending_idx = list(range(N))

    if pending_idx:
        pending = [captions[i] for i in pending_idx]
        out_vals: List[bool] = []
        for j in tqdm(range(0, len(pending), batch_size),  # batch processing in case of OOM
                      desc=f"NLI {Path(csv_path).name}"):
            out_vals.extend(nli.is_single_batch(pending[j:j + batch_size]))
        for i, r in zip(pending_idx, out_vals):
            results[i] = r

    df[out_col] = results
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    n_true = int(sum(1 for r in results if r))
    return {
        "csv": str(csv_path),
        "out": str(out_path),
        "total": N,
        "regex_decided": N - len(pending_idx),
        "nli_decided": len(pending_idx),
        "single_true": n_true,
        "single_false": N - n_true,
    }


# ---- CLI ----
def _expand(pattern: str) -> List[str]:
    if any(c in pattern for c in "*?["):
        return sorted(_glob.glob(pattern))
    return [pattern]


def main():
    p = argparse.ArgumentParser("Annotate HQ-OpenHumanVid CSVs with a `single` column")
    p.add_argument("--config", default="data/openvid/config.yaml",
                   help="dataset config; uses cfg.prepare.{csv_glob,nli_model_dir,nli_onnx,nli_batch_size}")
    p.add_argument("--out-dir", default=None, #FIXME: 改为放置到/mnt/highspeed/users/Arcus/openvid 也就是整理后的数据集中 也就是 config.yaml 中的 out_root
                   help="output directory; default: same as each input csv. filename: <stem>.single.csv")
    p.add_argument("--provider", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--no-regex", action="store_true",
                   help="disable regex pre-filter, send all captions to NLI")
    args = p.parse_args()

    cfg = load_cfg(args.config).prepare
    paths = _expand(cfg.csv_glob)
    if not paths:
        raise FileNotFoundError(f"No csv matched: {cfg.csv_glob}")

    print(f"[filters] loading NLI from {cfg.nli_model_dir} ({args.provider})")
    nli = SinglePersonNLI(
        model_dir=cfg.nli_model_dir,
        onnx_filename=cfg.nli_onnx,
        provider=args.provider,
    )

    all_stats = []
    for csv_path in paths:
        stem = Path(csv_path).stem
        out_dir = Path(args.out_dir) if args.out_dir else Path(csv_path).parent
        out_path = out_dir / f"{stem}.single.csv"
        stats = annotate_csv(
            csv_path=csv_path,
            out_path=str(out_path),
            nli=nli,
            batch_size=int(cfg.nli_batch_size),
            use_regex=not args.no_regex,
        )
        all_stats.append(stats)
        print(f"[{stem}] total={stats['total']} "
              f"regex={stats['regex_decided']} nli={stats['nli_decided']} "
              f"pass={stats['single_true']} reject={stats['single_false']}")

    total = sum(s["total"] for s in all_stats)
    passed = sum(s["single_true"] for s in all_stats)
    print(f"\n[filters] {len(all_stats)} file(s), {total} captions, "
          f"{passed} passed ({passed / max(total, 1):.2%})")


if __name__ == "__main__":
    main()
