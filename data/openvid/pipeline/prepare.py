"""
Dataset Pipeline Step 4: per-clip materialization.

Reads `<stem>.single.csv` produced by filters.py, keeps rows with `single == True`,
For each clip:
  1. resolve raw mp4 path
       csv path = "clips/{part}/{ab}/{cd}/{cid}"
       file     = {raw_video_root}/{part}/{ab}/{cd}/{cid}.mp4
  2. read >=81 frames with decord, take the first num_frames
  3. resize each frame to (height, width) with aspect-preserving + black pad
  4. SCHP parse + temporal majority smoothing -> binary mask [T,H,W]
  5. pick top-K reference frames via cv_check -> IQA -> VLM cascade
  6. for each kept ref: bbox crop with `ref_pad_ratio` pad, resize and save as RGBA png
  7. write {cid}.mp4 (raw normalized, NOT masked) + masks.npz + meta.json
  8. atomic append to index.jsonl, supports resume via existing entries.
"""
# TODO: 检查csv读取的位置  检查转出的文件结构是否正确
# FIXME: 不是对top-k reference frames进行筛选 而是先crop出来然后再进行判断
# TODO: 可能涉及fps重采样
# TODO: 运行后书写脚本进行数据集特征统计 因为数据分布可能直接训练模型性能
 
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from data.utils import load_cfg
from data.openvid.pipeline.parse import SchpParser
from data.openvid.pipeline.score import IqaScorer, VlmFilter, cv_check


# ============================================================
#                    Image / video helpers
# ============================================================
def fit_pad(img: np.ndarray, th: int, tw: int, interp: int = cv2.INTER_AREA) -> np.ndarray:
    """Aspect-preserving resize to fit inside (th, tw), then center-pad with 0.
    Works on [H,W,C] uint8 OR [H,W] uint8.
    """
    h, w = img.shape[:2]
    if h == th and w == tw:
        return img
    s = min(th / h, tw / w)
    nh, nw = max(int(round(h * s)), 1), max(int(round(w * s)), 1)
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    pad_h = th - nh
    pad_w = tw - nw
    top = pad_h // 2
    left = pad_w // 2
    if resized.ndim == 3:
        out = np.zeros((th, tw, resized.shape[2]), dtype=resized.dtype)
        out[top:top + nh, left:left + nw] = resized
    else:
        out = np.zeros((th, tw), dtype=resized.dtype)
        out[top:top + nh, left:left + nw] = resized
    return out


def fit_pad_video(video: np.ndarray, th: int, tw: int) -> np.ndarray:
    """Apply fit_pad to each frame, returns [T, th, tw, 3]."""
    out = np.empty((video.shape[0], th, tw, 3), dtype=video.dtype)
    for i in range(video.shape[0]):
        out[i] = fit_pad(video[i], th, tw, interp=cv2.INTER_AREA)
    return out


def fit_pad_mask(mask: np.ndarray, th: int, tw: int) -> np.ndarray:
    """fit_pad each frame of [T,H,W] uint8 mask using NEAREST so binary stays binary."""
    out = np.empty((mask.shape[0], th, tw), dtype=mask.dtype)
    for i in range(mask.shape[0]):
        out[i] = fit_pad(mask[i], th, tw, interp=cv2.INTER_NEAREST)
    return out


def write_mp4(frames_rgb: np.ndarray, out_path: Path, fps: int) -> None:
    """Pipe RGB frames to ffmpeg libx264 yuv420p crf18. Raises on failure."""
    T, H, W, _ = frames_rgb.shape
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        "-an",
        str(out_path),
    ] # -i -: read from stdin;  
    # -c:v encoder H.264; -pix_fmt pixel format YUV 4:2:0
    # -crf: constant rate factor; -an: audio none
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    _, err = proc.communicate(frames_rgb.tobytes())
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {err.decode(errors='ignore')[:400]}")


def read_video(path: Path, num_frames: int, fps: float) -> Optional[np.ndarray]:
    """Read first num_frames frames as RGB uint8 [T,H,W,3]. Returns None if short / unreadable.

    fps mismatch: DO NOT resample. The caller is expected to filter on fps
    via metadata later if needed; HQ-OpenHumanVid is mostly already 24fps-ish.
    """
    from decord import VideoReader, cpu
    try:
        vr = VideoReader(str(path), ctx=cpu(0))
    except Exception:
        return None
    n = len(vr)
    if n < num_frames:
        return None
    return vr.get_batch(list(range(num_frames))).asnumpy()


def laplacian_sharpness(rgb: np.ndarray) -> float:
    """Variance of Laplacian on the V channel of HSV; higher = sharper."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


# ============================================================
#                    Ref crop (bbox + square pad + RGBA)
# ============================================================
def mask_bbox(mask_2d: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Tight bounding box of a 2D binary mask. Returns (y0, y1, x0, x1) inclusive-exclusive."""
    ys, xs = np.where(mask_2d > 0)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def expand_bbox(bbox: Tuple[int, int, int, int], H: int, W: int, pad_ratio: float) -> Tuple[int, int, int, int]:
    """Expand bbox by pad_ratio on each side then clip to image.
       [Important!] Avoiding losing information in subsequent steps.
    """
    y0, y1, x0, x1 = bbox
    bh, bw = y1 - y0, x1 - x0
    py = int(round(bh * pad_ratio))
    px = int(round(bw * pad_ratio))
    y0 = max(0, y0 - py); x0 = max(0, x0 - px)
    y1 = min(H, y1 + py); x1 = min(W, x1 + px)
    return y0, y1, x0, x1


def build_ref_rgba(
    frame_rgb: np.ndarray,         # [H,W,3] uint8
    mask_2d: np.ndarray,           # [H,W]  uint8 in {0,1}
    pad_ratio: float,
    ref_size: int,
) -> Optional[Tuple[np.ndarray, Tuple[int, int]]]:
    """
    Build the 480x480 RGBA ref image:
        - tight bbox on mask + outer pad_ratio
        - crop frame and mask
        - square pad (short side -> long side, center, value=0)
        - resize to (ref_size, ref_size)
        - RGB = resized frame; A = resized mask (255 fg, 0 bg)
    Returns (rgba [H,W,4] uint8, bbox_hw_after_expand) or None when mask is empty.
    """
    H, W = mask_2d.shape
    bb = mask_bbox(mask_2d)
    if bb is None:
        return None
    y0, y1, x0, x1 = expand_bbox(bb, H, W, pad_ratio)
    bh, bw = y1 - y0, x1 - x0
    if bh < 4 or bw < 4:
        return None

    crop_rgb  = frame_rgb[y0:y1, x0:x1]                # [bh, bw, 3]
    crop_mask = mask_2d[y0:y1, x0:x1]                  # [bh, bw]

    # square pad: short side -> long side
    side = max(bh, bw)
    sq_rgb  = np.zeros((side, side, 3), dtype=np.uint8)
    sq_alpha = np.zeros((side, side),    dtype=np.uint8)
    top  = (side - bh) // 2
    left = (side - bw) // 2
    sq_rgb[top:top + bh, left:left + bw]   = crop_rgb
    sq_alpha[top:top + bh, left:left + bw] = (crop_mask > 0).astype(np.uint8) * 255

    # resize to ref_size
    if side != ref_size:
        sq_rgb   = cv2.resize(sq_rgb,   (ref_size, ref_size), interpolation=cv2.INTER_AREA)
        sq_alpha = cv2.resize(sq_alpha, (ref_size, ref_size), interpolation=cv2.INTER_NEAREST)

    rgba = np.dstack([sq_rgb, sq_alpha])               # [ref_size, ref_size, 4]
    return rgba, (bh, bw)


# ============================================================
#                       Path resolution
# ============================================================
def resolve_raw_mp4(raw_root: Path, csv_path_field: str) -> Path:
    """
    csv `path` field: "clips/part_001/f6/05/f605b8e9..."
    raw layout      : {raw_root}/part_001/f6/05/f605b8e9.mp4 (no `clips/` prefix)
    """
    rel = csv_path_field
    if rel.startswith("clips/"):
        rel = rel[len("clips/"):]
    return raw_root / (rel + ".mp4")


def part_of_csv_path(csv_path_field: str) -> str:
    """Extract `part_xxx` from csv path."""
    parts = csv_path_field.split("/")
    for p in parts:
        if p.startswith("part_"):
            return p
    raise ValueError(f"no part_xxx in csv path: {csv_path_field}")


def out_clip_dir(out_root: Path, part: str, cid: str) -> Path:
    return out_root / "clips" / part / cid[:2] / cid[2:4] / cid


# ============================================================
#                       index.jsonl helpers
# ============================================================
def read_done_clips(index_path: Path) -> set:
    if not index_path.exists():
        return set()
    done = set()
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["clip_id"])
            except Exception:
                pass
    return done


def append_jsonl(path: Path, obj: dict) -> None:
    """Atomic append to a jsonl file: write to a temp adjacent, then rename via os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


# ============================================================
#                    Per-clip pipeline
# ============================================================
def _ref_pool_for_clip(
    parsing: np.ndarray,                # [T,H,W] uint8 class ids
    mask_seq: np.ndarray,               # [T,H,W] uint8 in {0,1}
    video: np.ndarray,                  # [T,H,W,3] uint8
    cfg_prepare,
    iqa: IqaScorer,
    vlm: VlmFilter,
) -> List[Dict[str, Any]]:
    """Run the L1->L2->L3 cascade across all T frames; return up to ref_candidates_k accepted refs.

    Each accepted entry is a dict with bbox metadata + an in-memory RGBA crop.
    """
    T, H, W = mask_seq.shape
    cat = cfg_prepare.category
    ref_size = int(cfg_prepare.ref_size if hasattr(cfg_prepare, "ref_size") else 480)
    pad_ratio = float(cfg_prepare.ref_pad_ratio)
    k_keep = int(cfg_prepare.ref_candidates_k)
    iqa_thresh = float(cfg_prepare.iqa_thresh)

    cv_min = dict(cfg_prepare.cv_min_size)
    cv_ratio = {k: list(v) for k, v in dict(cfg_prepare.cv_mask_ratio).items()}

    # --- L1: cv_check + sharpness sort ---
    pre: List[Tuple[int, float, Tuple[int, int, int, int]]] = []  # (idx, sharpness, bbox)
    for t in range(T):
        m = mask_seq[t]
        bb = mask_bbox(m)
        if bb is None:
            continue
        y0, y1, x0, x1 = bb
        bh, bw = y1 - y0, x1 - x0
        bbox_pixels = max(bh * bw, 1)
        mask_pixels = int(m[y0:y1, x0:x1].sum())
        ratio = mask_pixels / bbox_pixels
        if not cv_check((bh, bw), ratio, cat, cv_min, cv_ratio):
            continue
        sharp = laplacian_sharpness(video[t])
        pre.append((t, sharp, (y0, y1, x0, x1)))

    if not pre:
        return []
    pre.sort(key=lambda x: x[1], reverse=True)
    # cap pre-pool to ~3x final K to bound IQA + VLM cost
    pre = pre[:max(k_keep * 3, k_keep)]

    accepted: List[Dict[str, Any]] = []
    for idx, sharp, _bb in pre:
        rgba_meta = build_ref_rgba(video[idx], mask_seq[idx], pad_ratio, ref_size)
        if rgba_meta is None:
            continue
        rgba, bbox_hw = rgba_meta

        # L2 IQA on RGB only
        rgb_only = rgba[..., :3]
        try:
            iqa_score = iqa.score(rgb_only)
        except Exception:
            iqa_score = -1.0
        if iqa_score < iqa_thresh:
            continue

        # L3 VLM judge
        try:
            v = vlm.judge(rgb_only, category=cat)
        except Exception:
            v = {"category_match": False, "occlusion": True, "truncation": True}
        if not v["category_match"] or v["occlusion"] or v["truncation"]:
            continue

        accepted.append({
            "frame_idx": int(idx),
            "sharpness": float(sharp),
            "iqa": float(iqa_score),
            "bbox_hw": [int(bbox_hw[0]), int(bbox_hw[1])],
            "vlm": v,
            "rgba": rgba,
        })
        if len(accepted) >= k_keep:
            break

    return accepted


def prepare_one_clip(
    row: pd.Series,
    cfg,                             # full top-level cfg (for height/width/num_frames/fps)
    cfg_prepare,                     # cfg.prepare
    schp: SchpParser,
    iqa: IqaScorer,
    vlm: VlmFilter,
    raw_root: Path,
    out_root: Path,
) -> Optional[dict]:
    cid = str(row["clip_id"])
    csv_path_field = str(row["path"])
    part = part_of_csv_path(csv_path_field)

    src = resolve_raw_mp4(raw_root, csv_path_field)
    if not src.exists():
        return {"_status": "missing_src", "clip_id": cid, "path": csv_path_field}

    H, W = int(cfg.height), int(cfg.width)
    NF, FPS = int(cfg.num_frames), int(cfg.fps)

    # 1. read raw video, first NF frames
    video_raw = read_video(src, NF, fps=FPS)
    if video_raw is None:
        return {"_status": "short_or_unreadable", "clip_id": cid, "path": csv_path_field}

    # 2. resize + pad to (H, W)
    video = fit_pad_video(video_raw, H, W)

    # 3. SCHP parse + binary mask + temporal smooth
    parsing = schp.parse_video(video)
    keep_ids = list(map(int, cfg_prepare.lip_label_ids))
    mask_seq = SchpParser.select(parsing, keep_ids)
    mask_seq = SchpParser.smooth(mask_seq, k=int(cfg_prepare.temporal_smooth_k))

    if mask_seq.sum() == 0:
        return {"_status": "empty_mask", "clip_id": cid, "path": csv_path_field}

    # 4. pick refs via cascade
    refs = _ref_pool_for_clip(parsing, mask_seq, video, cfg_prepare, iqa, vlm)
    if not refs:
        return {"_status": "no_ref", "clip_id": cid, "path": csv_path_field}

    # 5. write outputs
    cdir = out_clip_dir(out_root, part, cid)
    cdir.mkdir(parents=True, exist_ok=True)

    # 5.1 video.mp4 (raw normalized, NOT masked)
    write_mp4(video, cdir / f"{cid}.mp4", fps=FPS)

    # 5.2 masks.npz
    np.savez_compressed(cdir / "masks.npz", mask=mask_seq.astype(np.uint8))

    # 5.3 ref_imgs/{idx:04d}.png (RGBA)
    rdir = cdir / "ref_imgs"
    rdir.mkdir(exist_ok=True)
    ref_meta = []
    for r in refs:
        Image.fromarray(r["rgba"], mode="RGBA").save(rdir / f"{r['frame_idx']:04d}.png")
        ref_meta.append({
            "frame_idx": r["frame_idx"],
            "sharpness": r["sharpness"],
            "iqa":       r["iqa"],
            "bbox_hw":   r["bbox_hw"],
            "vlm":       r["vlm"],
        })

    # 5.4 meta.json
    meta = {
        "clip_id":   cid,
        "source":    "openvid",
        "part":      part,
        "src_path":  str(src),
        "csv_path":  csv_path_field,
        "n_frames":  NF,
        "fps":       FPS,
        "height":    H,
        "width":     W,
        "caption":   str(row.get("caption", "")),
        "category":  cfg_prepare.category,
        "lip_label_ids": keep_ids,
        "ref_candidates": ref_meta,
        "scores": {                                              # carried over from csv if present
            k: float(row[k]) for k in (
                "luminance_min", "luminance_max",
                "blur_min", "blur_max",
                "aesthetic", "technical_score", "global_motion",
            ) if k in row and pd.notna(row[k])
        },
    }
    with open(cdir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {
        "_status": "ok",
        "clip_id": cid,
        "part":    part,
        "n_refs":  len(refs),
        "path":    f"clips/{part}/{cid[:2]}/{cid[2:4]}/{cid}",
    }


# ============================================================
#                            CLI
# ============================================================
def _expand_csv_glob(pattern: str) -> List[str]:
    import glob as _g
    if any(c in pattern for c in "*?["):
        return sorted(_g.glob(pattern))
    return [pattern]


def main():
    p = argparse.ArgumentParser("OpenVid per-clip preprocessing")
    p.add_argument("--config", default="data/openvid/config.yaml")
    p.add_argument("--limit", type=int, default=-1,
                   help="optional cap on total clips (debug); -1 = no cap")
    p.add_argument("--device", default="cuda")
    p.add_argument("--schp-batch", type=int, default=None,
                   help="override cfg.prepare.schp_batch if you hit OOM")
    args = p.parse_args()

    cfg_top = load_cfg(args.config)
    cfg = cfg_top.prepare
    raw_root = Path(cfg.raw_video_root)
    out_root = Path(cfg.out_root)
    index_path = out_root / cfg.index_file

    # find single.csv files based on csv_glob
    csv_pattern = cfg.csv_glob
    if csv_pattern.endswith(".csv"):
        single_pattern = csv_pattern[:-len(".csv")] + ".single.csv"
    else:
        single_pattern = csv_pattern + ".single.csv"
    csv_paths = _expand_csv_glob(single_pattern)
    if not csv_paths:
        print(f"[prepare] no .single.csv found for pattern {single_pattern}; "
              f"run filters.py first.", file=sys.stderr)
        sys.exit(1)

    # load models once
    print(f"[prepare] loading SCHP from {cfg.schp_model}")
    schp = SchpParser(
        weight_path=cfg.schp_model,
        device=args.device,
        batch_size=int(args.schp_batch or cfg.get("schp_batch", 32)),
    )
    print(f"[prepare] loading IQA ({cfg.iqa_metric}) from {cfg.iqa_model}")
    iqa = IqaScorer(weight_path=cfg.iqa_model, metric=cfg.iqa_metric, device=args.device)
    print(f"[prepare] loading VLM from {cfg.vlm_dir}")
    vlm = VlmFilter(model_dir=cfg.vlm_dir, prompt_file=cfg.vlm_prompt)

    # resume support
    done = read_done_clips(index_path)
    print(f"[prepare] resume: {len(done)} clips already in {index_path}")

    total_processed = 0
    counters: Dict[str, int] = {}

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        df = df[df["single"].astype(str).str.lower() == "true"]
        df = df[~df["clip_id"].astype(str).isin(done)]
        if 0 <= args.limit <= total_processed:
            break
        if args.limit > 0:
            remain = args.limit - total_processed
            df = df.head(remain)

        for _, row in tqdm(df.iterrows(), total=len(df), desc=Path(csv_path).name):
            try:
                res = prepare_one_clip(row, cfg_top, cfg, schp, iqa, vlm, raw_root, out_root)
            except Exception as e:
                res = {
                    "_status": "error",
                    "clip_id": str(row.get("clip_id", "")),
                    "error": f"{type(e).__name__}: {e}",
                    "trace": traceback.format_exc(limit=4),
                }
            status = res.get("_status", "unknown")
            counters[status] = counters.get(status, 0) + 1
            if status == "ok":
                # only success entries land in index.jsonl
                entry = {k: v for k, v in res.items() if not k.startswith("_")}
                append_jsonl(index_path, entry)
                done.add(entry["clip_id"])
            total_processed += 1

    print(f"\n[prepare] done. counters: {counters}")


if __name__ == "__main__":
    main()
