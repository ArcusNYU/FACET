"""
CelebV-HQ Dataset Pipeline Stage 3 - main entry-point.

Per clip:
  1. Resolve raw mp4: {raw_video_root}/clip/{cid}.mp4
  2. Read up to num_frames*2 frames; first num_frames = tgt segment, rest = ref pool.
  3. SCHP on RAW frames -> lip_label_ids mask -> temporal smooth.
  4. fit_pad first num_frames of raw video + mask to (height, width).
  5. Pick up to ref_candidates_k refs via L1 cv_check -> L2 IQA -> L3 VLM cascade.
  6. Write {cid}.mp4 (raw normalized, NOT masked) + masks.npz + ref_imgs/*.png + meta.json.
  7. Atomic, flock-guarded append to index.jsonl (ok) or failed.jsonl (rejected).

Concurrency: `--shard i/N` splits the downloaded.json clip set across N workers
by md5(clip_id) % N (mirrors cache.py). Both jsonl ledgers are guarded by
fcntl.flock so concurrent shard appends never interleave (safe on NFS/Lustre).
"""

# TODO: 可能涉及fps重采样
# TODO: 后期把openvid/celebv重复的 fit_pad / ref-crop / jsonl helpers 抽到 data/utils.py 共享
# TODO: 后期使用Qwen2.5-plus来增加caption prompt生成的多样化 

# NOTE: appearance + action attributes (carried from candidate.json via downloaded.json):
#       (a) synthesized into a structured caption -> meta.json["caption"]
#       (b) stored raw -> meta.json["appearance"] + meta.json["action"] + meta.json["hair_color"]
#           for later per-attribute / per-action evaluation tables.


from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "8")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")  # for A100
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
from PIL import Image
from tqdm import tqdm

from data.utils import load_cfg
from data.celebv.pipeline.parse import SchpParser
from data.celebv.pipeline.score import IqaScorer, VlmFilter, cv_check
from data.celebv.pipeline.attributes import build_caption
from utils import write_mp4


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


# write_mp4 imported from utils (shared with loader_visual.py)


def read_video(path: Path, min_frames: int, max_frames: int) -> "tuple[Optional[np.ndarray], str]":
    """Read up to max_frames frames as RGB uint8 [T,H,W,3] from the head of the video.

    Returns (frames, reason) where:
        frames : np.ndarray [T,H,W,3] on success, None on failure.
        reason : "" on success | "unreadable:<exc>" | "short:<n>/<min>".
    """
    # TODO: fps重采样逻辑 (源clip来自youtube, fps不保证==cfg.fps)
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
    """Expand bbox by pad_ratio on each side then clip to image."""
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
def raw_clip_path(raw_root: Path, cid: str) -> Path:
    """acquire.py writes the trimmed+cropped clip to {raw_root}/clip/{cid}.mp4."""
    return raw_root / "clip" / f"{cid}.mp4"


def out_clip_dir(out_root: Path, cid: str) -> Path:
    """Flat per-clip output dir (no hash bucketing): {out_root}/clip/{cid}/."""
    return out_root / "clip" / cid


# ============================================================
#                    jsonl ledger helpers
# ============================================================
def _read_cids(path: Path) -> set:
    """Collect clip_ids from a jsonl ledger (index.jsonl or failed.jsonl)."""
    if not path.exists():
        return set()
    done = set()
    with open(path, "r", encoding="utf-8") as f:
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
    """Append a jsonl line with fsync, guarded by an advisory file lock.

    Concurrent `--shard` workers append to the SAME index.jsonl / failed.jsonl.
    fcntl.flock(LOCK_EX) serializes the write+fsync window across processes so
    lines never interleave (POSIX O_APPEND atomicity does not hold on NFS/Lustre).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def read_downloaded(path: Path) -> Dict[str, dict]:
    """Read acquire.py's downloaded.json -> {clip_id: {ytb_id, appearance, action,
    hair_color}} in file order (Python dicts preserve insertion order).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"downloaded.json not found: {path}; run data/celebv/pipeline/acquire.py first"
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[str, dict] = {}
    for cid, v in data.items():
        v = v or {}
        out[cid] = {
            "ytb_id":     v.get("ytb_id", ""),
            "appearance": list(v.get("appearance", [])),
            "action":     list(v.get("action", [])),
            "hair_color": v.get("hair_color", ""),
        }
    return out


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

    # FIXME: cv thresholds are calibrated at target resolution and resolution-invariant
    # for the ratio; sizes are used as-is (CelebV crops are already head-region).
    # FIXME: cv min size的阈值需要升高 
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
        rgb_only = rgba[..., :3]   # IQA/VLM see RGB (bg pixels not zeroed)

        # L2 IQA on RGB only
        try:
            iqa_score = iqa.score(rgb_only)
        except Exception:
            iqa_score = -1.0
        if iqa_score < iqa_thresh:
            continue

        # L3 VLM judges on RGB only
        try:
            v = vlm.judge(rgb_only, category=cat)
        except Exception:
            v = {"match": False, "occlusion": True}
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
    cid: str,
    ytb_id: str,
    cfg,                             # full top-level cfg (for height/width/num_frames/fps)
    cfg_prepare,                     # cfg.prepare
    schp: SchpParser,
    iqa: IqaScorer,
    vlm: VlmFilter,
    raw_root: Path,
    out_root: Path,
    attrs: Optional[dict] = None,    # {appearance, hair_color} carried from downloaded.json
) -> dict:
    src = raw_clip_path(raw_root, cid)
    if not src.exists():
        return {"_status": "missing_src", "clip_id": cid}

    H, W = int(cfg.height), int(cfg.width)
    NF, FPS = int(cfg.num_frames), int(cfg.fps)
    max_tries = int(cfg_prepare.get("ref_max_tries", 15))

    # 1. read raw video -> up to NF*2 frames; tgt = first NF, ref pool = the rest
    video_raw, read_reason = read_video(src, min_frames=NF, max_frames=NF * 2)
    if video_raw is None:
        # read_reason is either "short:<n>/<min>" or "unreadable:<exc>"
        kind = "short" if read_reason.startswith("short:") else "unreadable"
        return {"_status": kind, "clip_id": cid, "reason": read_reason}
    T_total = video_raw.shape[0]

    # 2. SCHP on RAW frames -> binary mask (hair) -> temporal smooth
    parsing_raw = schp.parse_video(video_raw)                            # [T,H_raw,W_raw]
    keep_ids = list(map(int, cfg_prepare.lip_label_ids))
    mask_raw = SchpParser.select(parsing_raw, keep_ids)
    mask_raw = SchpParser.smooth(mask_raw, k=int(cfg_prepare.temporal_smooth_k))

    # 3. fit_pad tgt video + tgt mask (ref crops still come from raw for fidelity)
    tgt_video = fit_pad_video(video_raw[:NF], H, W)                      # [NF,H,W,3]
    tgt_mask  = fit_pad_mask(mask_raw[:NF], H, W)                        # [NF,H,W]
    if tgt_mask.sum() == 0:
        return {"_status": "empty_mask", "clip_id": cid}

    # 4. pick refs from post-tgt frames (fallback to tgt frames if clip == NF long)
    if T_total > NF:
        ref_candidate_idx = list(range(NF, T_total))
    else:
        ref_candidate_idx = list(range(NF))
    refs = _pick_refs(
        video_raw, mask_raw, ref_candidate_idx,
        cfg_prepare, iqa, vlm, max_tries=max_tries,
    )
    if not refs:
        return {"_status": "no_ref", "clip_id": cid}

    # 5. write outputs (flat layout: out_root/clip/{cid}/...)
    cdir = out_clip_dir(out_root, cid)
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

    # 5.4 meta.json. captions are synthesized from appearance + action attributes
    appearance = list(attrs.get("appearance", [])) if attrs else []
    action = list(attrs.get("action", [])) if attrs else []
    hair_color = (attrs.get("hair_color", "") if attrs else "") or ""
    caption = build_caption(appearance, action) if appearance else ""

    rel_path = f"clip/{cid}"
    meta = {
        "clip_id":        cid,
        "source":         "celebv",
        "ytb_id":         ytb_id,
        "path":           rel_path,         # relative to cfg.data_root
        "fps":            FPS,
        "height":         H,
        "width":          W,
        "caption":        caption,
        "category":       cfg_prepare.category,
        "hair_color":     hair_color,
        "appearance":     appearance,       # raw 40-d 0/1 vector (per-attribute tables)
        "action":         action,           # raw 35-d 0/1 vector (per-action tables / Qwen prompts)
        "ref_candidates": ref_meta,
    }
    with open(cdir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {
        "_status":    "ok",
        "clip_id":    cid,
        "ytb_id":     ytb_id,
        "n_refs":     len(refs),
        "path":       rel_path,
        "hair_color": hair_color,           
        # hair_color lands in index.jsonl for downstream use, for example, ablation test on hair_color
    }


# ============================================================
#                            CLI
# ============================================================
def main():
    p = argparse.ArgumentParser("CelebV-HQ per-clip preprocessing")
    p.add_argument("--config", default=str(Path(__file__).parent.parent / "config.yaml"))
    p.add_argument("--limit", type=int, default=-1,
                   help="optional cap on total clips (debug); -1 = no cap")
    p.add_argument("--device", default="cuda")
    p.add_argument("--schp-batch", type=int, default=None,
                   help="override cfg.prepare.schp_batch if you hit OOM")
    p.add_argument("--shard", default="0/1",
                   help="i/N: handle only clips whose md5(cid)%%N == i. Lets you "
                        "co-locate multiple workers on the same GPU when VRAM allows.")
    p.add_argument("--retry-failed", action="store_true",
                   help="re-attempt clips previously recorded in failed.jsonl")
    args = p.parse_args()

    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    assert 0 <= shard_i < shard_n, f"bad --shard {args.shard}"

    cfg_top = load_cfg(args.config)
    cfg = cfg_top.prepare
    raw_root = Path(cfg.raw_video_root)
    out_root = Path(cfg.out_root)
    weight_dir = Path(cfg.weight_dir)
    index_path = out_root / cfg.index_file
    failed_path = out_root / cfg.get("failed_file", "failed.jsonl")
    downloaded_path = raw_root / "downloaded.json"

    downloaded = read_downloaded(downloaded_path)   # {cid: {ytb_id, appearance, action, hair_color}}
    clips = [(cid, v["ytb_id"]) for cid, v in downloaded.items()]
    print(f"[prepare] downloaded.json clips: {len(clips)} from {downloaded_path}")

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

    # ---- resume: skip clips already in index.jsonl OR failed.jsonl ----
    done = _read_cids(index_path)
    failed_done = set() if args.retry_failed else _read_cids(failed_path)
    print(f"[prepare] resume: {len(done)} ok in {index_path.name}, "
          f"{len(failed_done)} failed in {failed_path.name} "
          f"(retry_failed={args.retry_failed})")

    # filter -> shard -> limit
    todo = [(cid, ytb) for cid, ytb in clips
            if cid not in done and cid not in failed_done]
    if shard_n > 1:
        def _in_shard(cid: str) -> bool:
            return (int(hashlib.md5(cid.encode()).hexdigest(), 16) % shard_n) == shard_i
        before = len(todo)
        todo = [(cid, ytb) for cid, ytb in todo if _in_shard(cid)]
        print(f"[prepare] shard={args.shard}: kept {len(todo)}/{before} clips")
    if args.limit > 0:
        todo = todo[:args.limit]

    counters: Dict[str, int] = {}
    for cid, ytb in tqdm(todo, total=len(todo), desc="celebv"):
        try:
            res = prepare_clip(cid, ytb, cfg_top, cfg, schp, iqa, vlm, raw_root, out_root,
                               attrs=downloaded.get(cid))
        except Exception as e:
            res = {
                "_status": "error",
                "clip_id": cid,
                "ytb_id":  ytb,
                "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc(limit=4),
            }
        status = res.get("_status", "unknown")
        counters[status] = counters.get(status, 0) + 1
        if status != "ok" and counters[status] <= 3:
            if status == "error":
                print(f"\n[{status}] clip={cid} {res.get('error','')}\n{res.get('trace','')}", flush=True)
            else:
                print(f"\n[{status}] clip={cid} {res.get('reason','')}", flush=True)

        if status == "ok":
            entry = {k: v for k, v in res.items() if not k.startswith("_")}
            append_jsonl(index_path, entry)
            done.add(cid)
        else:
            # record rejection so resume does not re-analyze it (waste of SCHP+IQA+VLM)
            fail_entry = {
                "clip_id": cid,
                "ytb_id":  ytb,
                "status":  status,
                "reason":  res.get("reason") or res.get("error") or "",
            }
            append_jsonl(failed_path, fail_entry)
            failed_done.add(cid)

    print(f"\n[prepare] done. shard={args.shard} counters: {counters}")


if __name__ == "__main__":
    main()
