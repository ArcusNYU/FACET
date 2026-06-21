"""
test.py

FACET / WAN2.2-TI2V-5B + OmniControl LoRA inference entry point.

Edits the hairstyle of the person in user-provided material (NOT a held-out
test-set scorer -- trainer/valid.py already does that). It is the offline twin
of the future gradio demo, so its structure mirrors train.py (step 0/1/2 ...)
minus everything training-specific.

Material layout (paths.src_dir, default ./test, scanned RECURSIVELY):
    <src_dir>/<sample_name>/
        *.mp4   source / mask / (optional) ground-truth video(s)
        *.png|*.jpg   reference image (the target hairstyle)
        [caption.txt | prompt.txt]   optional text prompt
  - >=1 subfolder  -> one sample each  -> result named <sample_name>_pred.mp4
  - no subfolder   -> the material lives directly in <src_dir> (single sample)

Per-sample material resolution
    videos:
      * ground-truth video: stem has gt / target / ground  (enables metrics)
      * mask video:         stem has "mask"                 (binary mask sequence)
      * source video:       the remaining video
          - 1 video, no mask video  -> SCHP-parse it -> hair mask + masked src
          - source stem has source/original/raw/src + a mask video -> apply mask
          - otherwise (a mask video present, source not raw-tagged) -> the source
            is treated as an ALREADY-masked src video; the mask video supplies the
            binary mask the model needs.
    reference image:
      * stem has "mask" -> already a masked ref (bg=127); used as-is
      * otherwise       -> SCHP-parse it -> hair bbox crop + 127-gray background

All preprocessing matches the training pipeline (data/transform.py fit-pad +
[-1,1] norm, masked region painted gray=127) and applies NO mask perturbation /
geometric augmentation.

Layout:
    0.  argparse + HF offline lock
    1.  load test.yaml (+ CLI overrides) -> resolved paths
    2.  setup: accelerator + seed + dirs + logger (mlflow)
    3.  FACETWanModel construction + LoRA load + freeze + eval (T5 kept on device)
    4.  discover samples -> (rank-sharded) inference loop -> save results
    5.  GT-conditional metrics (light incl. *_mask + heavy FID/FVD) -> mlflow
    6.  tear-down
"""

from __future__ import annotations

# =============================================================================
# 0. HF offline lock + env hygiene  (must precede any HF import)
# =============================================================================
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")
warnings.filterwarnings("ignore", message=".*timm.models.layers is deprecated.*", category=FutureWarning)

# =============================================================================
# 1. Imports
# =============================================================================
import argparse
import datetime as dt
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from accelerate import Accelerator
from accelerate.utils import set_seed

import metrics
from data.utils import load_cfg
from data.transform import VideoTfm, MaskPerturb, RefTfm
from facet.config import FACETConfig
from facet.model import FACETWanModel, FACETPipeline
from trainer.valid import heavy_eval
from utils import _resolve_dtype, video_to_uint8, mask_to_uint8, write_mp4, write_png

logger = logging.getLogger("facet.test")


# Keyword routing for the per-sample material files.
_SRC_KEYS = ("source", "original", "raw", "src")
_GT_KEYS = ("gt", "target", "ground", "groundtruth", "ground_truth")
_MASK_KEY = "mask"
_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")

_LIGHT_KEYS = ("psnr", "ssim", "lpips")
_MASK_KEYS = ("psnr_mask", "ssim_mask", "lpips_mask")
_ALL_LIGHT_KEYS = _LIGHT_KEYS + _MASK_KEYS

_DTYPE_TO_ACC = {"no": "no", "fp32": "no", "fp16": "fp16", "bf16": "bf16"}


# =============================================================================
# 2. argparse
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FACET inference / hairstyle-editing entry point.")
    p.add_argument("--test_yaml", type=str, default=str(_PROJECT_ROOT / "test.yaml"),
                   help="Path to test.yaml.")
    # Overrides (None -> use test.yaml). ckpt_name + facet_config are the two the
    # user changes most often (which checkpoint, which model geometry/rank).
    p.add_argument("--ckpt_name", type=str, default=None,
                   help="LoRA checkpoint under paths.ckpt_root (w/ or w/o .safetensors).")
    p.add_argument("--facet_config", type=str, default=None,
                   help="Model config yaml (default facet/config.yaml; use a run's "
                        "config_snapshot.yaml to inherit its exact lora rank).")
    p.add_argument("--src_dir", type=str, default=None, help="Input material root.")
    p.add_argument("--save_dir", type=str, default=None, help="Results root.")
    return p.parse_args()


def _to_abs(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (_PROJECT_ROOT / p).resolve()


# =============================================================================
# 3. Setup (accelerator + seed + dirs + logger)
# =============================================================================
def seed_everything(seed: int, rank: int) -> None:
    """Seed every RNG layer (python / numpy / torch / accelerate, per-rank).

    The per-sample init noise itself is drawn from a seed-derived generator built
    inside FACETPipeline (seed=cfg.test.seed), so inference is reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed, device_specific=True)


def setup_logger(accelerator, cfg, output_root: Path, track_root: Path):
    """Minimal mlflow + console + jsonl logger (simplified trainer.logger)."""
    backend = str(cfg.log.get("backend", "none")).lower()
    state = {"accelerator": accelerator, "tracking": False,
             "jsonl": None, "is_main": accelerator.is_main_process}

    if accelerator.is_main_process:
        output_root.mkdir(parents=True, exist_ok=True)
        state["jsonl"] = output_root / "metrics.jsonl"
        # Dump the effective test config for reproducibility.
        (output_root / "test_config_snapshot.yaml").write_text(
            json.dumps(dict(cfg), ensure_ascii=False, indent=2), encoding="utf-8",
        )
        if backend == "mlflow":
            track_root.mkdir(parents=True, exist_ok=True)
            os.environ["MLFLOW_TRACKING_URI"] = track_root.resolve().as_uri()
            logger.info("[test] mlflow store=%s  (mlflow ui --backend-store-uri %s)",
                        track_root.resolve(), track_root.resolve().as_uri())

    if backend in ("mlflow", "tensorboard", "wandb"):
        try:
            run_name = cfg.log.get("cloud_run_name") or output_root.name
            init_kwargs = {"mlflow": {"run_name": run_name}} if backend == "mlflow" else {}
            accelerator.init_trackers(
                project_name=str(cfg.log.get("project_name", "facet_test")),
                config=_flat_cfg(cfg),
                init_kwargs=init_kwargs,
            )
            state["tracking"] = True
        except Exception as e:  # noqa: BLE001 - logging must never crash inference
            logger.warning("[test] init_trackers failed (%s); cloud logging off.", e)
    return state


def _flat_cfg(cfg) -> Dict[str, Any]:
    """Flatten the (small) test DotDict to scalar params for mlflow."""
    out: Dict[str, Any] = {}

    def walk(prefix: str, obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(f"{prefix}.{k}" if prefix else k, v)
        elif isinstance(obj, (list, tuple)):
            out[prefix] = ",".join(map(str, obj))
        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            out[prefix] = obj

    walk("", cfg)
    return out


def log_scalars(state, scalars: Dict[str, float], step: int = 0, phase: str = "test") -> None:
    if not state["is_main"]:
        return
    pretty = ", ".join(f"{k}={v:.5f}" for k, v in scalars.items())
    logger.info("[test] %s", pretty)
    if state["tracking"]:
        try:
            state["accelerator"].log({f"{phase}/{k}": float(v) for k, v in scalars.items()}, step=step)
        except Exception as e:  # noqa: BLE001
            logger.warning("[test] accelerator.log failed: %s", e)
    if state["jsonl"] is not None:
        with open(state["jsonl"], "a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, "phase": phase, **scalars}, ensure_ascii=False) + "\n")


# =============================================================================
# 4. SCHP parser (lazy, one per process)
# =============================================================================
class _Schp:
    """Lazy SchpParser holder so SCHP weights load only if a sample needs them."""

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self._parser = None

    def _ensure(self):
        if self._parser is None:
            from data.openvid.pipeline.parse import SchpParser
            weight = _to_abs(self.cfg.schp.weight)
            if not Path(weight).exists():
                raise FileNotFoundError(
                    f"SCHP weights not found: {weight}. Provide pre-masked material "
                    "(mask video + masked ref) or fix schp.weight in test.yaml."
                )
            logger.info("[test] loading SCHP from %s", weight)
            self._parser = SchpParser(
                weight_path=str(weight), device=self.device,
                batch_size=int(self.cfg.schp.get("batch_size", 32)),
            )
        return self._parser

    def hair_mask(self, video_rgb: np.ndarray) -> np.ndarray:
        """[T,H,W,3] uint8 RGB -> [T,H,W] uint8 in {0,1} (hair, temporally smoothed)."""
        from data.openvid.pipeline.parse import SchpParser
        parser = self._ensure()
        parsing = parser.parse_video(video_rgb)
        ids = [int(x) for x in self.cfg.schp.get("hair_ids", [2])]
        mask = SchpParser.select(parsing, ids)
        return SchpParser.smooth(mask, k=int(self.cfg.schp.get("smooth_k", 3)))


# =============================================================================
# 5. Material discovery + IO helpers
# =============================================================================
def _stem_has(name: str, keys) -> bool:
    s = name.lower()
    return any(k in s for k in keys)


def _has_mp4(d: Path) -> bool:
    return any(d.glob("*.mp4"))


def discover_samples(src_dir: Path, save_dir: Path) -> List[Tuple[Path, str]]:
    """Recursively find sample dirs (any dir holding >=1 mp4), excluding save_dir."""
    def excluded(p: Path) -> bool:
        try:
            p.resolve().relative_to(save_dir.resolve())
            return True
        except ValueError:
            return False

    sample_dirs = [
        d for d in sorted(x for x in src_dir.rglob("*") if x.is_dir())
        if not excluded(d) and _has_mp4(d)
    ]
    if sample_dirs:
        out = []
        for d in sample_dirs:
            rel = d.relative_to(src_dir)
            out.append((d, "_".join(rel.parts) if rel.parts else d.name))
        return out
    if _has_mp4(src_dir):
        return [(src_dir, src_dir.name)]
    return []


def read_rgb_video(path: Path, num_frames: int) -> np.ndarray:
    """mp4 -> [num_frames, H, W, 3] uint8 RGB (first num_frames, repeat-last pad)."""
    from decord import VideoReader, cpu
    vr = VideoReader(str(path), ctx=cpu(0))
    n = len(vr)
    idx = list(range(num_frames)) if n >= num_frames else list(range(n)) + [n - 1] * (num_frames - n)
    return vr.get_batch(idx).asnumpy()


def read_mask_video(path: Path, num_frames: int) -> np.ndarray:
    """mp4 -> [num_frames, H, W] uint8 {0,1} (luminance threshold @127)."""
    v = read_rgb_video(path, num_frames)                 # [T,H,W,3]
    gray = v.mean(axis=-1)                                # [T,H,W]
    return (gray > 127).astype(np.uint8)


def read_image_rgb(path: Path) -> np.ndarray:
    """png/jpg -> [H,W,3] uint8 RGB."""
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))


# ---- ref crop helpers (inlined from data/openvid/pipeline/main.py to avoid
#      importing its heavy IQA/VLM deps) ----
def _mask_bbox(m2d: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(m2d > 0)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def build_masked_ref(frame_rgb: np.ndarray, mask_2d: np.ndarray, ref_size: int,
                     pad_ratio: float = 0.20, gray: int = 127) -> Optional[np.ndarray]:
    """Bbox-crop the hair, square-pad, resize to ref_size, fill bg with `gray`.

    Mirrors main.build_ref_rgba + ref_sampler 127-gray fill so the test ref looks
    exactly like a training ref. Returns [ref_size,ref_size,3] uint8 or None.
    """
    import cv2
    H, W = mask_2d.shape
    bb = _mask_bbox(mask_2d)
    if bb is None:
        return None
    y0, y1, x0, x1 = bb
    bh, bw = y1 - y0, x1 - x0
    py, px = int(round(bh * pad_ratio)), int(round(bw * pad_ratio))
    y0, x0 = max(0, y0 - py), max(0, x0 - px)
    y1, x1 = min(H, y1 + py), min(W, x1 + px)
    bh, bw = y1 - y0, x1 - x0

    crop_rgb = frame_rgb[y0:y1, x0:x1]
    crop_m = (mask_2d[y0:y1, x0:x1] > 0)
    side = max(bh, bw)
    sq = np.full((side, side, 3), gray, dtype=np.uint8)   # background pre-filled gray
    top, left = (side - bh) // 2, (side - bw) // 2
    region = sq[top:top + bh, left:left + bw]
    region[crop_m] = crop_rgb[crop_m]                      # paste only the hair pixels
    sq[top:top + bh, left:left + bw] = region
    if side != ref_size:
        sq = cv2.resize(sq, (ref_size, ref_size), interpolation=cv2.INTER_AREA)
    return sq


# =============================================================================
# 6. Per-sample preprocessing
# =============================================================================
class Prepared:
    def __init__(self, masked_src, src_mask, ref, mask_metric, gt, prompt):
        self.masked_src = masked_src   # [3, F, H, W] in [-1,1]
        self.src_mask = src_mask       # [1, F, H, W] in {0,1}
        self.ref = ref                 # [3, rs, rs] in [-1,1]
        self.mask_metric = mask_metric # [F, 1, H, W] in {0,1} (for *_mask metrics)
        self.gt = gt                   # [F, 3, H, W] in [-1,1] or None
        self.prompt = prompt           # str


def preprocess_sample(
    sample_dir: Path, cfg, facet_cfg: FACETConfig, schp: _Schp,
) -> Optional[Prepared]:
    """Resolve material files, run SCHP where needed, build model-ready tensors."""
    H, W = int(facet_cfg.target.height), int(facet_cfg.target.width)
    F = int(facet_cfg.target.num_frames)
    rs = int(facet_cfg.reference.image_size)

    video_tfm = VideoTfm(H, W)
    mask_tfm = MaskPerturb(H, W)        # perturb=False below -> resize + binarize only
    ref_tfm = RefTfm(rs)

    # ---- videos: split into gt / mask / source ----
    mp4s = sorted(sample_dir.glob("*.mp4"))
    if not mp4s:
        logger.warning("[test] %s: no mp4 found; skipping.", sample_dir.name)
        return None
    gt_vids = [m for m in mp4s if _stem_has(m.stem, _GT_KEYS) and not _stem_has(m.stem, (_MASK_KEY,))]
    rest = [m for m in mp4s if m not in gt_vids]
    mask_vids = [m for m in rest if _stem_has(m.stem, (_MASK_KEY,))]
    src_vids = [m for m in rest if m not in mask_vids]
    if not src_vids:
        logger.warning("[test] %s: no source video (only mask/gt); skipping.", sample_dir.name)
        return None
    src_path = src_vids[0]

    src_raw = read_rgb_video(src_path, F)                  # [F,Hraw,Wraw,3]

    if mask_vids:
        # Mask provided -> no SCHP for the source branch.
        mask_raw = read_mask_video(mask_vids[0], F)        # [F,Hraw,Wraw] {0,1}
    else:
        # No mask video -> SCHP-parse the source for the hair mask.
        mask_raw = schp.hair_mask(src_raw)                 # [F,Hraw,Wraw] {0,1}

    src_video = video_tfm(src_raw)                         # [F,3,H,W] in [-1,1]
    src_mask = mask_tfm(mask_raw, perturb=False)           # [F,1,H,W] in {0,1}

    # Paint the masked region gray=127 (normalized 0.0), matching data/base.py.
    GRAY_127_NORM = 0.0
    masked_src = torch.where(src_mask > 0.5, torch.full_like(src_video, GRAY_127_NORM), src_video)

    # ---- reference image ----
    imgs = sorted([p for p in sample_dir.iterdir() if p.suffix.lower() in _IMG_EXTS])
    if not imgs:
        logger.warning("[test] %s: no reference image (png/jpg); skipping.", sample_dir.name)
        return None
    ref_masked_imgs = [p for p in imgs if _stem_has(p.stem, (_MASK_KEY,))]
    if ref_masked_imgs:
        ref_rgb = read_image_rgb(ref_masked_imgs[0])       # already masked (bg=127)
    else:
        raw_ref = read_image_rgb(imgs[0])                  # [Hr,Wr,3]
        ref_mask = schp.hair_mask(raw_ref[None, ...])[0]   # [Hr,Wr] {0,1}
        ref_rgb = build_masked_ref(raw_ref, ref_mask, rs)
        if ref_rgb is None:
            logger.warning("[test] %s: SCHP found no hair in ref; skipping.", sample_dir.name)
            return None
    ref = ref_tfm(ref_rgb)                                 # [3,rs,rs] in [-1,1]

    # ---- optional ground truth (enables metrics) ----
    gt = video_tfm(read_rgb_video(gt_vids[0], F)) if gt_vids else None  # [F,3,H,W] | None

    # ---- optional text prompt ----
    prompt = ""
    for cap in ("caption.txt", "prompt.txt"):
        cp = sample_dir / cap
        if cp.exists():
            prompt = cp.read_text(encoding="utf-8").strip()
            break

    return Prepared(
        masked_src=masked_src.permute(1, 0, 2, 3).contiguous(),  # [3,F,H,W]
        src_mask=src_mask.permute(1, 0, 2, 3).contiguous(),      # [1,F,H,W]
        ref=ref,
        mask_metric=src_mask,                                    # [F,1,H,W]
        gt=gt,
        prompt=prompt,
    )


# =============================================================================
# 7. Main
# =============================================================================
def main() -> None:
    args = parse_args()

    # -------- 1. Load test config + resolve paths + CLI overrides ----------
    cfg = load_cfg(args.test_yaml)
    if args.ckpt_name is not None:
        cfg.paths.ckpt_name = args.ckpt_name
    if args.facet_config is not None:
        cfg.paths.facet_config = args.facet_config
    if args.src_dir is not None:
        cfg.paths.src_dir = args.src_dir
    if args.save_dir is not None:
        cfg.paths.save_dir = args.save_dir

    src_dir = _to_abs(cfg.paths.src_dir)
    save_dir = _to_abs(cfg.paths.save_dir)
    facet_config_path = _to_abs(cfg.paths.facet_config)
    ckpt_root = _to_abs(cfg.paths.ckpt_root)
    fvd_dir = _to_abs(cfg.paths.fvd_dir) if cfg.paths.get("fvd_dir") else None
    inception_dir = _to_abs(cfg.paths.inception_dir) if cfg.paths.get("inception_dir") else None

    # -------- 2. Setup: accelerator + seed + dirs + logger ----------------
    precision = str(cfg.accel.get("precision", "bf16")).lower()
    accelerator = Accelerator(
        mixed_precision=_DTYPE_TO_ACC.get(precision, "bf16"),
        log_with=("mlflow" if str(cfg.log.get("backend", "none")).lower() == "mlflow" else None),
    )
    device = accelerator.device
    seed = int(cfg.test.get("seed", 42))
    seed_everything(seed, accelerator.process_index)

    run_name = f"{dt.datetime.now():%m%d_%H%M}_{cfg.test.get('suffix', 'test')}"
    output_root = _to_abs(cfg.paths.run_root) / run_name if cfg.paths.get("run_root") else (_PROJECT_ROOT / "runs" / run_name)
    eval_root = output_root / "eval"   # pred/gt pairs for heavy_eval (GT samples)
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)
        output_root.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    log_state = setup_logger(accelerator, cfg, output_root, _PROJECT_ROOT / "mlflow")

    # -------- 3. Build FACET model + load LoRA + freeze --------------------
    facet_cfg = FACETConfig.from_yaml(str(facet_config_path))
    facet_cfg.device = str(device)                 # per-rank cuda:i
    dtype = _resolve_dtype(facet_cfg.dtype)
    if accelerator.is_main_process:
        logger.info("[test] constructing FACETWanModel (config=%s) ...", facet_config_path)
    model = FACETWanModel(facet_cfg)

    # Resolve + load the LoRA checkpoint (accept name with/without suffix).
    ckpt_name = str(cfg.paths.ckpt_name)
    if not ckpt_name.endswith(".safetensors"):
        ckpt_name += ".safetensors"
    ckpt_path = ckpt_root / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"LoRA checkpoint not found: {ckpt_path}. Set paths.ckpt_name in test.yaml "
            "or pass --ckpt_name (the *_best.safetensors under paths.ckpt_root)."
        )
    if accelerator.is_main_process:
        logger.info("[test] loading LoRA -> %s", ckpt_path)
    model.load_lora(str(ckpt_path))
    model.set_lora(trainable=False)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    model.to(device)                               # incl. T5 text_encoder (kept on device)
    pipeline = FACETPipeline(model)

    # CFG is not wired into training yet -> force scale 1.0 (no guidance).
    cfg_scale = float(cfg.test.get("cfg_scale", 1.0))
    if cfg_scale != 1.0:
        logger.warning("[test] cfg_scale=%.2f but CFG is unsupported; forcing 1.0.", cfg_scale)
        cfg_scale = 1.0

    # -------- 4. Discover samples + rank-sharded inference loop ------------
    samples = discover_samples(src_dir, save_dir)
    if accelerator.is_main_process:
        logger.info("[test] discovered %d sample(s) under %s", len(samples), src_dir)
    if not samples:
        logger.warning("[test] nothing to do; exiting.")
        _finish(accelerator, log_state)
        return

    my_samples = samples[accelerator.process_index :: accelerator.num_processes]

    # Per-key metric accumulators (summed across ranks at the end).
    light_sum = {k: torch.zeros((), device=device) for k in _ALL_LIGHT_KEYS}
    light_cnt = {k: torch.zeros((), device=device) for k in _ALL_LIGHT_KEYS}
    n_done = torch.zeros((), device=device)

    num_steps = int(cfg.test.get("num_inference_steps", 50))
    sigma_shift = float(cfg.test.get("sigma_shift", 5.0))
    fps = int(cfg.test.get("fps", 24))

    for sample_dir, name in my_samples:
        try:
            prep = preprocess_sample(sample_dir, cfg, facet_cfg, schp=_get_schp(cfg, device))
        except Exception as e:  # noqa: BLE001 - one bad sample must not kill the batch
            logger.warning("[test] preprocess failed for %s: %s", name, e)
            continue
        if prep is None:
            continue

        try:
            video = pipeline(
                src_video=prep.masked_src.to(device=device, dtype=dtype),
                src_mask=prep.src_mask.to(device=device),
                reference_images=prep.ref.to(device=device, dtype=dtype),
                prompt=prep.prompt,
                num_inference_steps=num_steps,
                cfg_scale=cfg_scale,
                sigma_shift=sigma_shift,
                seed=seed,
                output_type="video",
            )  # [1, 3, F, H, W] in [-1, 1]
        except Exception as e:  # noqa: BLE001
            logger.warning("[test] inference failed for %s: %s", name, e)
            continue

        pred = video[0].permute(1, 0, 2, 3).contiguous()   # [F,3,H,W]

        # ---- save results (single-sample -> bare names; subfolder -> name_*) ----
        stem = "pred" if (sample_dir == src_dir) else f"{name}_pred"
        write_mp4(video_to_uint8(pred), save_dir / f"{stem}.mp4", fps=fps, allow_fallback=True)
        write_mp4(mask_to_uint8(prep.mask_metric), save_dir / f"{stem.replace('pred', 'mask')}.mp4",
                  fps=fps, allow_fallback=True)
        write_png(((prep.ref.permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy(),
                  save_dir / f"{stem.replace('pred', 'ref')}.png")
        n_done += 1
        logger.info("[test] %s -> %s.mp4", name, stem)

        # ---- GT-conditional metrics ----
        if prep.gt is not None:
            gt = prep.gt.to(device=device)
            mask = prep.mask_metric.to(device=device)
            n = min(pred.shape[0], gt.shape[0])
            p_, g_, m_ = pred[:n], gt[:n], mask[:n]
            lm = metrics.light_metrics(p_, g_)
            lm.update(metrics.light_metrics_mask(p_, g_, m_))
            for k, v in lm.items():
                light_sum[k] += float(v)
                light_cnt[k] += 1.0
            # persist pred/gt for the heavy (FID/FVD) distribution metrics
            pair = eval_root / name
            pair.mkdir(parents=True, exist_ok=True)
            write_mp4(video_to_uint8(p_), pair / "pred.mp4", fps=fps, allow_fallback=True)
            write_mp4(video_to_uint8(g_), pair / "gt.mp4", fps=fps, allow_fallback=True)

    # -------- 5. Reduce + report metrics ----------------------------------
    accelerator.wait_for_everyone()
    results: Dict[str, float] = {}
    for k in _ALL_LIGHT_KEYS:
        ksum = accelerator.reduce(light_sum[k].clone(), reduction="sum")
        kcnt = accelerator.reduce(light_cnt[k].clone(), reduction="sum")
        if float(kcnt) > 0:
            results[k] = float(ksum / kcnt)
    total_done = int(accelerator.reduce(n_done.clone(), reduction="sum"))

    if results:
        log_scalars(log_state, results, step=0, phase="test")
    elif accelerator.is_main_process:
        logger.info("[test] no ground-truth videos found -> light metrics skipped "
                    "(pure inference/demo mode).")

    # Heavy metrics (distribution-level) on main process over the saved GT pairs.
    if accelerator.is_main_process:
        if eval_root.is_dir() and any(eval_root.iterdir()):
            heavy = heavy_eval(eval_root, fvd_dir=str(fvd_dir) if fvd_dir else None,
                               fid_dir=str(inception_dir) if inception_dir else None)
            if heavy:
                log_scalars(log_state, {f"heavy_{k}": v for k, v in heavy.items()}, step=0, phase="test")
        else:
            logger.info("[test] no GT pairs -> heavy metrics (FID/FVD) skipped.")
        logger.info("[test] done. %d sample(s) -> %s", total_done, save_dir)

    # -------- 6. Tear-down ------------------------------------------------
    _finish(accelerator, log_state)


# A single SCHP holder per process (created lazily on first use).
_SCHP_SINGLETON: Optional[_Schp] = None


def _get_schp(cfg, device) -> _Schp:
    global _SCHP_SINGLETON
    if _SCHP_SINGLETON is None:
        _SCHP_SINGLETON = _Schp(cfg, device)
    return _SCHP_SINGLETON


def _finish(accelerator, log_state) -> None:
    if log_state.get("tracking"):
        try:
            accelerator.end_training()
        except Exception as e:  # noqa: BLE001
            logger.warning("[test] end_training failed: %s", e)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    logging.basicConfig(
        level=os.environ.get("FACET_LOGLEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
