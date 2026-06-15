"""
Smoke-test variant of data/celebv/pipeline/main.py with verbose ref-frame diagnostics.

Differences vs celebv main.py:
  - Replaces _pick_refs with _pick_refs_verbose: prints each candidate's
    bbox / mask ratio / IQA score / VLM judgement, AND saves an annotated
    overlay PNG so you can see exactly which stage rejected a candidate.
  - Annotated images go to {out_root}/_debug/{cid}/{stage}_{idx:04d}.png
    e.g. cv_fail_0123.png / iqa_fail_0123.png / vlm_fail_0123.png / accept_0123.png
  - If $DISPLAY is set (ssh -X / X11 forwarding), each annotated frame is also
    shown via cv2.imshow (~600ms). Otherwise it just saves to disk.
  - prepare_clip_test does NOT write final outputs (mp4/masks/refs/index/failed);
    it only runs the diagnostic, so even no_ref clips leave a debug folder.

Usage:
    python -m data.celebv.pipeline.main_test --config data/celebv/config.yaml --limit 10

All other helpers (read_video, fit_pad, ref crop, downloaded.json reader) are
imported from main.py to avoid drift.
"""
from __future__ import annotations
import argparse
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "8")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
import warnings
warnings.filterwarnings(
    'ignore',
    message='.*timm.models.layers is deprecated.*',
    category=FutureWarning,
)

import random
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from data.utils import load_cfg
from data.celebv.pipeline.parse import SchpParser
from data.celebv.pipeline.score import IqaScorer, VlmFilter, cv_check
from data.celebv.pipeline.main import (
    read_video,
    fit_pad_mask,
    mask_bbox,
    build_ref_rgba,
    raw_clip_path,
    read_downloaded,
)


_HAS_DISPLAY = bool(os.environ.get("DISPLAY"))


# ============================================================
#                     Visualization helper
# ============================================================
def _annotate(
    frame_rgb: np.ndarray,                 # [H, W, 3] uint8
    mask_2d: np.ndarray,                   # [H, W]    uint8 in {0, 1}
    bbox: Optional[Tuple[int, int, int, int]],
    lines: List[str],                      # text lines to overlay top-left
    color_bbox: Tuple[int, int, int],      # BGR
) -> np.ndarray:
    """Return BGR uint8 image with mask overlay + bbox + text annotations."""
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR).copy()
    # Translucent green tint on mask region for visibility.
    if mask_2d is not None and mask_2d.any():
        overlay = bgr.copy()
        overlay[mask_2d > 0] = (0, 255, 0)
        bgr = cv2.addWeighted(overlay, 0.35, bgr, 0.65, 0)
    if bbox is not None:
        y0, y1, x0, x1 = bbox
        cv2.rectangle(bgr, (x0, y0), (x1, y1), color_bbox, 3)
    # text panel
    y = 28
    for line in lines:
        cv2.putText(bgr, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 4, cv2.LINE_AA)  # outline
        cv2.putText(bgr, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 1, cv2.LINE_AA)
        y += 26
    return bgr


def _show_or_save(bgr: np.ndarray, save_path: Path, window_name: str = "ref candidate", wait_ms: int = 600) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), bgr)
    if _HAS_DISPLAY:
        try:
            cv2.imshow(window_name, bgr)
            cv2.waitKey(wait_ms)
        except cv2.error:
            pass


# ============================================================
#                  Verbose ref picker
# ============================================================
def _pick_refs_verbose(
    cid: str,
    video_raw: np.ndarray,
    mask_raw: np.ndarray,
    candidate_idx: List[int],
    cfg_prepare,
    iqa: IqaScorer,
    vlm: VlmFilter,
    max_tries: int,
    debug_dir: Path,
) -> List[Dict[str, Any]]:
    cat = cfg_prepare.category
    ref_size = int(cfg_prepare.get("ref_size", 480))
    pad_ratio = float(cfg_prepare.ref_pad_ratio)
    k_keep = int(cfg_prepare.ref_candidates_k)
    iqa_thresh = float(cfg_prepare.iqa_thresh)
    cv_min = {k: int(v) for k, v in dict(cfg_prepare.cv_min_size).items()}
    cv_ratio = {k: list(v) for k, v in dict(cfg_prepare.cv_mask_ratio).items()}

    print(f"  [pick_refs] cid={cid} category={cat}")
    print(f"  [pick_refs] cv_min_size[{cat}]={cv_min.get(cat)}, "
          f"cv_mask_ratio[{cat}]={cv_ratio.get(cat)}, iqa_thresh={iqa_thresh}")
    print(f"  [pick_refs] candidates={len(candidate_idx)}, max_tries={max_tries}, k_keep={k_keep}")

    accepted: List[Dict[str, Any]] = []
    pool = list(candidate_idx)
    random.shuffle(pool)
    tries = 0

    for idx in pool:
        if len(accepted) >= k_keep:
            break
        if tries >= max_tries:
            print(f"  [pick_refs] hit max_tries={max_tries}, stop")
            break
        tries += 1

        m = mask_raw[idx]
        bb = mask_bbox(m)
        if bb is None:
            print(f"  [try {tries:02d}] frame={idx:04d} REJECT@bbox: empty mask")
            _show_or_save(
                _annotate(video_raw[idx], m, None,
                          [f"frame={idx} REJECT@bbox: empty mask"], (0, 0, 255)),
                debug_dir / f"bbox_fail_{idx:04d}.png",
            )
            continue
        y0, y1, x0, x1 = bb
        bh, bw = y1 - y0, x1 - x0
        ratio = int(m[y0:y1, x0:x1].sum()) / max(bh * bw, 1)

        # --- L1: cv_check ---
        cv_min_v = cv_min.get(cat, 0)
        lo, hi = cv_ratio.get(cat, [0.0, 1.0])
        cv_pass = cv_check((bh, bw), ratio, cat, cv_min, cv_ratio)
        cv_msg = (f"CV: bh={bh} bw={bw} min(bh,bw)={min(bh,bw)} vs cv_min={cv_min_v}",
                  f"ratio={ratio:.3f} vs [{lo},{hi}] -> {'PASS' if cv_pass else 'FAIL'}")
        if not cv_pass:
            print(f"  [try {tries:02d}] frame={idx:04d} REJECT@CV  {cv_msg[0]} | {cv_msg[1]}")
            _show_or_save(
                _annotate(video_raw[idx], m, bb,
                          [f"frame={idx} REJECT@CV", *cv_msg], (0, 0, 255)),
                debug_dir / f"cv_fail_{idx:04d}.png",
            )
            continue

        rgba = build_ref_rgba(video_raw[idx], m, pad_ratio, ref_size)
        if rgba is None:
            print(f"  [try {tries:02d}] frame={idx:04d} REJECT@build_rgba")
            continue
        rgb_only = rgba[..., :3]

        # --- L2: IQA ---
        try:
            iqa_score = float(iqa.score(rgb_only))
        except Exception as e:
            iqa_score = -1.0
            print(f"  [try {tries:02d}] frame={idx:04d} IQA exception: {type(e).__name__}: {e}")
        iqa_msg = f"IQA: score={iqa_score:.4f} vs thresh={iqa_thresh} -> {'PASS' if iqa_score >= iqa_thresh else 'FAIL'}"

        if iqa_score < iqa_thresh:
            print(f"  [try {tries:02d}] frame={idx:04d} REJECT@IQA {iqa_msg}")
            _show_or_save(
                _annotate(video_raw[idx], m, bb,
                          [f"frame={idx} REJECT@IQA", *cv_msg, iqa_msg], (0, 165, 255)),
                debug_dir / f"iqa_fail_{idx:04d}.png",
            )
            continue

        # --- L3: VLM ---
        # Accept criterion: match=True AND occlusion=False
        # (no "multiple" check; CelebV bbox crops are reliably single-person).
        try:
            v = vlm.judge(rgb_only, category=cat)
        except Exception as e:
            v = {"match": False, "occlusion": True}
            print(f"  [try {tries:02d}] frame={idx:04d} VLM exception: {type(e).__name__}: {e}")
        vlm_msg = f"VLM: match={v['match']} occlusion={v['occlusion']}"
        vlm_pass = v["match"] and not v["occlusion"]

        if not vlm_pass:
            print(f"  [try {tries:02d}] frame={idx:04d} REJECT@VLM {cv_msg[0]} | {iqa_msg} | {vlm_msg}")
            _show_or_save(
                _annotate(video_raw[idx], m, bb,
                          [f"frame={idx} REJECT@VLM", *cv_msg, iqa_msg, vlm_msg], (0, 255, 255)),
                debug_dir / f"vlm_fail_{idx:04d}.png",
            )
            continue

        # accepted
        print(f"  [try {tries:02d}] frame={idx:04d} ACCEPT  {cv_msg[0]} | {iqa_msg} | {vlm_msg}")
        _show_or_save(
            _annotate(video_raw[idx], m, bb,
                      [f"frame={idx} ACCEPT", *cv_msg, iqa_msg, vlm_msg], (0, 255, 0)),
            debug_dir / f"accept_{idx:04d}.png",
        )
        accepted.append({
            "frame_idx": int(idx),
            "iqa":       float(iqa_score),
            "bbox_hw":   [int(bh), int(bw)],
            "rgba":      rgba,
        })

    return accepted


# ============================================================
#                  Per-clip pipeline (verbose)
# ============================================================
def prepare_clip_test(
    cid: str,
    ytb_id: str,
    cfg,
    cfg_prepare,
    schp: SchpParser,
    iqa: IqaScorer,
    vlm: VlmFilter,
    raw_root: Path,
    out_root: Path,
) -> dict:
    debug_dir = out_root / "_debug" / cid
    debug_dir.mkdir(parents=True, exist_ok=True)

    src = raw_clip_path(raw_root, cid)
    if not src.exists():
        print(f"\n[clip {cid}] missing src: {src}")
        return {"_status": "missing_src", "clip_id": cid}

    H, W = int(cfg.height), int(cfg.width)
    NF = int(cfg.num_frames)
    max_tries = int(cfg_prepare.get("ref_max_tries", 15))

    print(f"\n[clip {cid}] (ytb={ytb_id}) reading {src}")
    video_raw, reason = read_video(src, min_frames=NF, max_frames=NF * 2)
    if video_raw is None:
        kind = "short" if reason.startswith("short:") else "unreadable"
        print(f"[clip {cid}] {kind}: {reason}")
        return {"_status": kind, "clip_id": cid, "reason": reason}
    T_total = video_raw.shape[0]
    print(f"[clip {cid}] decoded {T_total} frames at raw shape "
          f"{video_raw.shape[1]}x{video_raw.shape[2]} (HxW)")

    print(f"[clip {cid}] running SCHP...")
    parsing_raw = schp.parse_video(video_raw)
    keep_ids = list(map(int, cfg_prepare.lip_label_ids))
    mask_raw = SchpParser.select(parsing_raw, keep_ids)
    mask_raw = SchpParser.smooth(mask_raw, k=int(cfg_prepare.temporal_smooth_k))
    n_nonzero = int((mask_raw.reshape(T_total, -1).sum(axis=1) > 0).sum())
    print(f"[clip {cid}] mask coverage: {n_nonzero}/{T_total} frames have non-empty mask")

    tgt_mask = fit_pad_mask(mask_raw[:NF], H, W)
    if tgt_mask.sum() == 0:
        print(f"[clip {cid}] empty_mask after fit_pad")
        return {"_status": "empty_mask", "clip_id": cid}

    if T_total > NF:
        ref_candidate_idx = list(range(NF, T_total))
    else:
        ref_candidate_idx = list(range(NF))

    refs = _pick_refs_verbose(
        cid, video_raw, mask_raw, ref_candidate_idx,
        cfg_prepare, iqa, vlm, max_tries=max_tries, debug_dir=debug_dir,
    )
    if not refs:
        print(f"[clip {cid}] no_ref")
        return {"_status": "no_ref", "clip_id": cid}

    print(f"[clip {cid}] OK: {len(refs)} refs accepted")
    return {"_status": "ok", "clip_id": cid, "n_refs": len(refs)}


# ============================================================
#                            CLI
# ============================================================
def main():
    p = argparse.ArgumentParser("CelebV pipeline smoke test (verbose)")
    p.add_argument("--config", default=str(Path(__file__).parent.parent / "config.yaml"))
    p.add_argument("--limit", type=int, default=10,
                   help="cap on total clips to inspect (from downloaded.json order)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--schp-batch", type=int, default=None)
    args = p.parse_args()

    cfg_top = load_cfg(args.config)
    cfg = cfg_top.prepare
    raw_root = Path(cfg.raw_video_root)
    out_root = Path(cfg.out_root)
    weight_dir = Path(cfg.weight_dir)

    clips = read_downloaded(raw_root / "downloaded.json")
    if args.limit > 0:
        clips = clips[:args.limit]
    print(f"[test] inspecting {len(clips)} clips from {raw_root / 'downloaded.json'}")

    schp_path = str(weight_dir / cfg.schp_model)
    iqa_path = str(weight_dir / cfg.iqa_model)
    vlm_path = str(weight_dir / cfg.vlm_dir)

    print(f"[test] DISPLAY={'SET' if _HAS_DISPLAY else 'NOT SET (will only save pngs)'}")
    print(f"[test] debug images -> {out_root}/_debug/<clip_id>/")

    print(f"[test] loading SCHP from {schp_path}")
    schp = SchpParser(
        weight_path=schp_path, device=args.device,
        batch_size=int(args.schp_batch or cfg.get("schp_batch", 32)),
    )
    print(f"[test] loading IQA from {iqa_path}")
    iqa = IqaScorer(weight_path=iqa_path, metric=cfg.iqa_metric, device=args.device)
    print(f"[test] loading VLM from {vlm_path}")
    vlm = VlmFilter(model_dir=vlm_path, prompt_file=cfg.vlm_prompt)

    counters: Dict[str, int] = {}
    for cid, ytb in clips:
        try:
            res = prepare_clip_test(cid, ytb, cfg_top, cfg, schp, iqa, vlm, raw_root, out_root)
        except Exception as e:
            res = {"_status": "error", "clip_id": cid,
                   "error": f"{type(e).__name__}: {e}",
                   "trace": traceback.format_exc(limit=4)}
            print(f"[error] {res['error']}\n{res['trace']}", flush=True)
            # OOM / cuda errors leave fragmented memory; release before next clip.
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        counters[res["_status"]] = counters.get(res["_status"], 0) + 1

    if _HAS_DISPLAY:
        cv2.destroyAllWindows()
    print(f"\n[test] done. counters: {counters}")


if __name__ == "__main__":
    main()
