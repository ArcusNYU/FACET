"""
Dataset Pipeline Stage 2 - main entry-point.

Reads `<stem>.single.csv` produced by filters.py from cfg.out_root,
keeps rows with `single == True`. For each clip:

  1. Resolve raw mp4 path:
         csv path = "clips/{part}/{ab}/{cd}/{cid}"     (no extension)
         file     = {raw_video_root}/{part}/{ab}/{cd}/{cid}.mp4
  2. Read up to `num_frames * 2` raw frames with decord; first `num_frames`
     form the tgt segment, the remainder is the ref candidate pool.
     If the clip only has exactly `num_frames` frames, fall back to sampling
     ref from the tgt segment itself (noted risk: ref appears in tgt).
  3. Run SCHP on the RAW frames (not fit-padded), so the parser sees the
     original content without the black pad region. Then derive binary mask
     via lip_label_ids + temporal majority smoothing.
  4. Pick up to `ref_candidates_k` reference frames via cascade:
         L1 cv_check        (bbox size + mask ratio range)
         L2 IQA >= iqa_thresh
         L3 VLM judge       (match=True, occlusion=False).
                            `truncation` is still requested from the VLM but
                            currently NOT used as a reject criterion.
     Rejection sampling on random post-tgt indices, cap at ref_max_tries.
  5. fit_pad the first num_frames of raw video AND raw mask to (height, width)
     -> tgt video.mp4 (raw normalized, NOT masked) + masks.npz
  6. Save ref_imgs/{frame_idx:04d}.png (RGBA).
  7. Write minimal meta.json.
  8. Atomic append to index.jsonl (resume-friendly).

Resume: clip_ids already in index.jsonl are skipped on restart.
Failure classes (missing_src / short / empty_mask / no_ref / error) are
counted on stdout but not recorded; they will be re-attempted on restart.
Model loading cost is amortized over the whole run, so retrying is cheap.
"""

# TODO: 可能涉及fps重采样
# TODO: 运行后书写脚本进行数据集特征统计 因为数据分布可能直接训练模型性能
# TODO: 其他数据集在构建pipeline的时候可能也需要下面某些通用性强的helper函数 到时候需要将它们移动至 /data/helpers.py


from __future__ import annotations
import argparse
import json
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0") # for A100
import warnings
warnings.filterwarnings(
    'ignore',
    message='.*timm.models.layers is deprecated.*',
    category=FutureWarning,
)

import random
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
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


def read_video(path: Path, min_frames: int, max_frames: int) -> "tuple[Optional[np.ndarray], str]":
    """Read up to max_frames frames as RGB uint8 [T,H,W,3] from the head of the video.

    Returns (frames, reason) where:
        frames  : np.ndarray [T,H,W,3] on success, None on failure.
        reason  : "" on success | "unreadable:<exc>" if decord fails | "short:<n>/<min>" if too few frames.

    TODO fps resample note:
        We currently pull frames at the source fps, not the target cfg.fps.
        HQ-OpenHumanVid clips are mostly ~24fps so the bias is small; a precise
        fix would either (a) stride-sample int(round(src_fps / target_fps)) or
        (b) pre-pipe through `ffmpeg -vf fps=target`. Deferred to a later pass.
    """
    # TODO: 在此添加fps重采样逻辑
    from decord import VideoReader, cpu
    try:
        vr = VideoReader(str(path), ctx=cpu(0))
    except Exception as e:
        return None, f"unreadable:{type(e).__name__}: {e}"
    n = len(vr)
    if n < min_frames:
        return None, f"short:{n}/{min_frames}"
    take = min(n, max_frames)
    return vr.get_batch(list(range(take))).asnumpy(), ""


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
) -> Optional[np.ndarray]:
    """
    Build the 480x480 RGBA ref image on [RAW] video:
    Using raw video and raw mask:
        - tight bbox on mask + outer pad_ratio
        - crop frame and mask
        - square pad (short side -> long side, center, value=0)
        - resize to (ref_size, ref_size)
        - RGB = resized frame; A = resized mask (255 fg, 0 bg)
    Returns rgba [ref_size,ref_size,4] uint8, or None when the bbox is empty.
    """
    H, W = mask_2d.shape
    bb = mask_bbox(mask_2d)
    if bb is None:
        return None
    y0, y1, x0, x1 = expand_bbox(bb, H, W, pad_ratio)
    bh, bw = y1 - y0, x1 - x0

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

    return np.dstack([sq_rgb, sq_alpha])               # [ref_size, ref_size, 4]


# ============================================================
#                       Path resolution
# ============================================================
def resolve_raw_mp4(raw_root: Path, csv_path_field: str) -> Path:
    """
    csv `path` field: "clips/part_001/f6/05/f605b8e9xxx"
    raw layout      : {raw_root}/part_001/f6/05/f605b8e9xxx.mp4 (no `clips/` prefix)
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
    """For breakpoint resume support."""
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
    """Append a jsonl line with fsync. Single-process, no locking needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line) # write to python buffer
        f.flush()     # to OS buffer
        os.fsync(f.fileno()) # write to disk


# ============================================================
#                    Ref frame rejection sampling
# ============================================================
def _pick_refs(
    video_raw: np.ndarray,          # [T_total, H_raw, W_raw, 3] uint8
    mask_raw: np.ndarray,           # [T_total, H_raw, W_raw]    uint8 in {0,1}
    candidate_idx: List[int],       # frame indices eligible for ref sampling
    cfg_prepare,
    iqa: IqaScorer,
    vlm: VlmFilter,
    max_tries: int,
) -> List[Dict[str, Any]]:
    """Random-sample from candidate_idx until we accept k_keep refs or exhaust max_tries
    using the L1 (cv_check) -> L2 (IQA) -> L3 (VLM) cascade.

    Returns list of dicts {"frame_idx", "iqa", "bbox_hw", "rgba"}.
    """
    cat = cfg_prepare.category
    ref_size = int(cfg_prepare.get("ref_size", 480))
    pad_ratio = float(cfg_prepare.ref_pad_ratio)
    k_keep = int(cfg_prepare.ref_candidates_k)
    iqa_thresh = float(cfg_prepare.iqa_thresh)

    # cv_min_size in config is calibrated against raw video resolution
    # (HQ-OpenHumanVid is mostly 720p / occasionally 1080p), so it's used as-is.
    # cv_mask_ratio is dimensionless and resolution-invariant, also used as-is.
    cv_min = {k: int(v) for k, v in dict(cfg_prepare.cv_min_size).items()}
    cv_ratio = {k: list(v) for k, v in dict(cfg_prepare.cv_mask_ratio).items()}

    accepted: List[Dict[str, Any]] = []
    pool = list(candidate_idx)
    random.shuffle(pool)
    tries = 0
    for idx in pool:
        if len(accepted) >= k_keep:
            break
        if tries >= max_tries:
            break
        tries += 1

        m = mask_raw[idx]
        bb = mask_bbox(m)
        if bb is None:
            continue
        y0, y1, x0, x1 = bb
        bh, bw = y1 - y0, x1 - x0
        ratio = int(m[y0:y1, x0:x1].sum()) / max(bh * bw, 1)
        if not cv_check((bh, bw), ratio, cat, cv_min, cv_ratio):
            continue

        rgba = build_ref_rgba(video_raw[idx], m, pad_ratio, ref_size)
        if rgba is None:
            continue
        # L2 IQA on RGB only
        rgb_only = rgba[..., :3]

        try:
            iqa_score = iqa.score(rgb_only)  # NOTE: iqa依然能够看到背景区域 因为原始背景区域的rgb像素并未被置0
        except Exception:
            iqa_score = -1.0
        if iqa_score < iqa_thresh:
            continue

        # L3 VLM judge
        # NOTE: truncation 字段仍由 VLM 返回, 但当前不参与拒绝, 只看 match & occlusion.
        # 经验上 当前openhumanvid数据集的特性, 许多镜头为人物上半身, 使得VLM判断truncation为true, 导致 no_ref 过多.
        try:
            v = vlm.judge(rgb_only, category=cat)
        except Exception:
            v = {"match": False, "occlusion": True, "truncation": True}
        if not v["match"] or v["occlusion"]:
            continue

        accepted.append({
            "frame_idx": int(idx),
            "iqa":       float(iqa_score),
            "bbox_hw":   [int(bh), int(bw)],
            "rgba":      rgba,
        })

    return accepted


# ============================================================
#                    Per-clip pipeline
# ============================================================
def prepare_clip(
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
    max_tries = int(cfg_prepare.get("ref_max_tries", 15))

    # 1. read raw video -> up to NF*2 frames; tgt = first NF, ref pool = the rest
    video_raw, read_reason = read_video(src, min_frames=NF, max_frames=NF * 2)
    if video_raw is None:
        # read_reason is either "short:<n>/<min>" or "unreadable:<exc>"
        kind = "short" if read_reason.startswith("short:") else "unreadable"
        return {"_status": kind, "clip_id": cid, "path": csv_path_field, "reason": read_reason}
    T_total = video_raw.shape[0]

    # 2. SCHP on RAW frames (no black pad interference).
    #    Then fit_pad derived masks so downstream tgt stays at target resolution.
    parsing_raw = schp.parse_video(video_raw)                            # [T,H_raw,W_raw]
    keep_ids = list(map(int, cfg_prepare.lip_label_ids))
    mask_raw = SchpParser.select(parsing_raw, keep_ids)
    mask_raw = SchpParser.smooth(mask_raw, k=int(cfg_prepare.temporal_smooth_k))

    # 3. fit_pad tgt video + tgt mask (ref crops still come from raw for fidelity)
    tgt_video = fit_pad_video(video_raw[:NF], H, W)                      # [NF,H,W,3]
    tgt_mask  = fit_pad_mask(mask_raw[:NF], H, W)                        # [NF,H,W]
    if tgt_mask.sum() == 0:
        return {"_status": "empty_mask", "clip_id": cid, "path": csv_path_field}

    # 4. pick refs via random sampling on post-tgt frames;
    #    fall back to tgt frames if total length == NF (ref appears in tgt risk).
    if T_total > NF:
        ref_candidate_idx = list(range(NF, T_total))
    else:
        ref_candidate_idx = list(range(NF))
    refs = _pick_refs(
        video_raw, mask_raw, ref_candidate_idx,
        cfg_prepare, iqa, vlm, max_tries=max_tries,
    )
    if not refs:
        return {"_status": "no_ref", "clip_id": cid, "path": csv_path_field}

    # 5. write outputs
    cdir = out_clip_dir(out_root, part, cid)
    cdir.mkdir(parents=True, exist_ok=True)

    # 5.1 video.mp4 (raw normalized, NOT masked)
    write_mp4(tgt_video, cdir / f"{cid}.mp4", fps=FPS)

    # 5.2 masks.npz (target space, zlib)
    np.savez_compressed(cdir / "masks.npz", mask=tgt_mask.astype(np.uint8))

    # 5.3 ref_imgs/{idx:04d}.png (RGBA, 480x480)
    rdir = cdir / "ref_imgs"
    rdir.mkdir(exist_ok=True)
    ref_meta = []
    for r in refs:
        Image.fromarray(r["rgba"], mode="RGBA").save(rdir / f"{r['frame_idx']:04d}.png")
        ref_meta.append({
            "frame_idx": r["frame_idx"],
            "iqa":       r["iqa"],
            "bbox_hw":   r["bbox_hw"],
        })

    # 5.4 meta.json 
    rel_path = f"clips/{part}/{cid[:2]}/{cid[2:4]}/{cid}"
    meta = {
        "clip_id":         cid,
        "source":          "openvid",
        "part":            part,
        "path":            rel_path,        # relative to cfg.data_root
        "fps":             FPS,
        "height":          H,
        "width":           W,
        "caption":         str(row.get("caption", "")),
        "category":        cfg_prepare.category,
        "ref_candidates":  ref_meta,
        "scores": {  # carried over from csv if present
            k: float(row[k]) for k in (
                "luminance_min", "luminance_max",
                "blur_min", "blur_max",
                "aesthetic", "technical_score", "global_motion",
            ) if k in row and pd.notna(row[k])}
    }
    with open(cdir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {
        "_status": "ok",
        "clip_id": cid,
        "part":    part,
        "n_refs":  len(refs),
        "path":    rel_path,
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
    p.add_argument("--config", default=str(Path(__file__).parent.parent / "config.yaml"))
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
    weight_dir = Path(cfg.weight_dir)
    index_path = out_root / cfg.index_file

    # filters.py writes `<stem>.single.csv` under cfg.out_root.
    csv_pattern_name = Path(cfg.csv_glob).name
    if csv_pattern_name.endswith(".csv"):
        single_name = csv_pattern_name[:-4] + ".single.csv"
    else:
        single_name = csv_pattern_name + ".single.csv"
    single_pattern = str(out_root / single_name)
    csv_paths = _expand_csv_glob(single_pattern)
    if not csv_paths:
        print(f"[prepare] no .single.csv found for pattern {single_pattern}; "
              f"run filters.py first.", file=sys.stderr)
        sys.exit(1)

    # ---- load models once (resolve all paths against weight_dir) ----
    schp_path = str(weight_dir / cfg.schp_model)
    iqa_path  = str(weight_dir / cfg.iqa_model)
    vlm_path  = str(weight_dir / cfg.vlm_dir)

    print(f"[prepare] loading SCHP from {schp_path}")
    schp = SchpParser(
        weight_path=schp_path,
        device=args.device,
        batch_size=int(args.schp_batch or cfg.get("schp_batch", 32)),
    )

    print(f"[prepare] loading IQA ({cfg.iqa_metric}) from {iqa_path}")
    iqa = IqaScorer(weight_path=iqa_path, metric=cfg.iqa_metric, device=args.device)

    print(f"[prepare] loading VLM from {vlm_path}")
    vlm = VlmFilter(model_dir=vlm_path, prompt_file=cfg.vlm_prompt)

    # ---- resume ----
    done = read_done_clips(index_path)
    print(f"[prepare] resume: {len(done)} clips already in {index_path}")

    total_processed = 0
    counters: Dict[str, int] = {}

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path, low_memory=False)
        df = df[df["single"].astype(str).str.lower() == "true"]
        df = df[~df["clip_id"].astype(str).isin(done)]
        if 0 <= args.limit <= total_processed:
            break
        if args.limit > 0:
            remain = args.limit - total_processed
            df = df.head(remain)

        for _, row in tqdm(df.iterrows(), total=len(df), desc=Path(csv_path).name):
            try:
                res = prepare_clip(row, cfg_top, cfg, schp, iqa, vlm, raw_root, out_root)
            except Exception as e:
                res = {
                    "_status": "error",
                    "clip_id": str(row.get("clip_id", "")),
                    "error": f"{type(e).__name__}: {e}",
                    "trace": traceback.format_exc(limit=4),
                }
            status = res.get("_status", "unknown")
            counters[status] = counters.get(status, 0) + 1
            if status != "ok" and counters[status] <= 3:
                cid_log = res.get("clip_id", "")
                if status == "error":
                    print(f"\n[{status}] clip={cid_log} {res.get('error','')}\n{res.get('trace','')}", flush=True)
                else:
                    reason = res.get("reason", res.get("path", ""))
                    print(f"\n[{status}] clip={cid_log} {reason}", flush=True)
            if status == "ok":
                # only success entries land in index.jsonl
                entry = {k: v for k, v in res.items() if not k.startswith("_")}
                append_jsonl(index_path, entry)
                done.add(entry["clip_id"])
            total_processed += 1

    print(f"\n[prepare] done. counters: {counters}")


if __name__ == "__main__":
    main()
