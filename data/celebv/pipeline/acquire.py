"""
CelebV-HQ downloader + raw clip extractor.
Dataset Pipeline Stage 1 - Acquisition.
Reference: https://github.com/celebv-text/CelebV-Text/blob/main/download_and_process.py

It produces the raw_root layout:
    {raw_root}/
        downloaded.json          # {clip_id: {...}} of successfully extracted clips (resume ledger)
        failed.json              # {clip_id: {...}} clips whose YouTube source was unavailable
        clip/
            {clip_id}.mp4        # time-trimmed + bbox-cropped clip (NOT resized; main.py fit_pads later)
        _raw_tmp/
            {ytb_id}.mp4         # full YouTube download, deleted once all its clips are extracted

Identity model:
    CelebV-HQ's celebvhq_info.json `clips` keys look like "M2Ohb0FAaJU_1", i.e.
    "{ytb_id}_{n}". ONE youtube video (ytb_id) can yield MULTIPLE clips (different
    time windows / bboxes). So:
      - clip_id  = the clips-dict KEY (unique)        -> names clip/{clip_id}.mp4
      - ytb_id   = the youtube id (shared)            -> names _raw_tmp/{ytb_id}.mp4
    We download each ytb_id ONCE and extract all of its clips from that single file.

bbox crop strategy (deliberate choice, see download.py PR discussion):
    We do NOT resize to the 832x480 training canvas here. We crop a single-person-safe
    region (face bbox expanded with extra headroom for hair) at native resolution and
    leave the aspect-ratio-preserving resize+pad to data/celebv/pipeline/main.py
    (same `fit_pad` path as openvid). This keeps every dataset on one geometric code
    path and avoids baking in an irreversible distortion at acquisition time.

Concurrency:
    yt-dlp (network) and ffmpeg (cpu) both run as external subprocesses, so a
    ThreadPoolExecutor gives real parallelism (GIL is released while waiting on the
    subprocess). Downloads and processing run in pools: download `--pool` clips
    worth of raw videos, extract them, persist the ledger, free the raw files, repeat.

Example:
    python data/celebv/download.py \
        --info data/celebv/celebvhq_info.json \
        --raw-root /mnt/highspeed/users/Arcus/CELEBV \
        --limit 500 --pool 100 --workers 6 --proc-workers 12
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from tqdm import tqdm


# ============================================================
#                     info.json parsing
# ============================================================
Clip = Tuple[str, str, Tuple[float, float], Tuple[float, float, float, float]]
# (clip_id, ytb_id, (start_sec, end_sec), (top, bottom, left, right) normalized)


def load_clips(info_path: Path) -> List[Clip]:
    """Flatten celebvhq_info.json['clips'] into an ordered list of Clip tuples.
    Order follows json insertion order, so clips sharing a ytb_id stay adjacent.
    """
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    out: List[Clip] = []
    for clip_id, val in info["clips"].items():
        ytb_id = val["ytb_id"]
        time = (float(val["duration"]["start_sec"]), float(val["duration"]["end_sec"]))
        b = val["bbox"]
        bbox = (float(b["top"]), float(b["bottom"]), float(b["left"]), float(b["right"]))
        out.append((clip_id, ytb_id, time, bbox))
    return out


# ============================================================
#                     resume ledger (json)
# ============================================================
def load_ledger(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_ledger(path: Path, data: Dict[str, dict]) -> None:
    """Atomic json dump (tmp -> replace) so a kill never corrupts the ledger."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ============================================================
#                     download (yt-dlp)
# ============================================================
def download_raw(
    ytb_id: str,
    raw_path: Path,
    proxy: Optional[str],
    use_aria2c: bool,
    timeout: int,
) -> Tuple[bool, str]:
    """Download a full youtube video to raw_path. Returns (ok, reason).
    Skips the network call if raw_path already exists (cross-pool reuse).
    """
    if raw_path.exists() and raw_path.stat().st_size > 0:
        return True, "cached"
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = ["yt-dlp"]
    if proxy:
        cmd += ["--proxy", proxy]
    cmd += [
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio",
        "--skip-unavailable-fragments",
        "--merge-output-format", "mp4",
        "-o", str(raw_path),
    ]
    if use_aria2c:
        cmd += [
            "--external-downloader", "aria2c",
            "--external-downloader-args", "aria2c:-x 16 -k 1M",
        ]
    cmd += [f"https://www.youtube.com/watch?v={ytb_id}"]

    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except FileNotFoundError:
        return False, "yt-dlp-not-installed"

    if proc.returncode != 0 or not raw_path.exists():
        tail = proc.stderr.decode(errors="ignore")[-200:] if proc.stderr else ""
        return False, f"yt-dlp-fail:{tail}"
    return True, "ok"


# ============================================================
#                     crop geometry
# ============================================================
def _secs_to_timestr(secs: float) -> str:
    hrs = int(secs // 3600)
    mins = int((secs - hrs * 3600) // 60)
    sec = int(secs % 60)
    frac = int(round((secs - int(secs)) * 100))
    return f"{hrs:02d}:{mins:02d}:{sec:02d}.{frac:02d}"


def compute_crop_px(
    bbox: Tuple[float, float, float, float],
    H: int,
    W: int,
    pad: float,
    top_extra: float,
) -> Tuple[int, int, int, int]:
    """Expand the normalized face bbox (top,bottom,left,right) into a pixel crop.

    - `pad`       : symmetric outer padding as a fraction of bbox h/w (all sides).
    - `top_extra` : EXTRA upward padding (fraction of bbox height) so hair above the
                    forehead is included -- important for the hair-editing task.
    Returns (x, y, w, h) with even w/h (yuv420p requires even dims), clipped to frame.
    """
    top, bottom, left, right = bbox
    bh = bottom - top
    bw = right - left

    t = top - bh * (pad + top_extra)
    b = bottom + bh * pad
    l = left - bw * pad
    r = right + bw * pad

    # clip to normalized [0, 1]
    t = min(max(t, 0.0), 1.0)
    b = min(max(b, 0.0), 1.0)
    l = min(max(l, 0.0), 1.0)
    r = min(max(r, 0.0), 1.0)

    y0 = int(round(t * H))
    y1 = int(round(b * H))
    x0 = int(round(l * W))
    x1 = int(round(r * W))

    w = max(x1 - x0, 2)
    h = max(y1 - y0, 2)
    # even dims, and keep within frame
    w -= w % 2
    h -= h % 2
    x0 = min(x0, W - w)
    y0 = min(y0, H - h)
    x0 = max(x0, 0)
    y0 = max(y0, 0)
    return x0, y0, w, h


def probe_size(path: Path) -> Optional[Tuple[int, int]]:
    """Return (W, H) of a video via cv2, or None if unreadable."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if w <= 0 or h <= 0:
        return None
    return w, h


def process_clip(
    raw_path: Path,
    out_path: Path,
    bbox: Tuple[float, float, float, float],
    time: Tuple[float, float],
    pad: float,
    top_extra: float,
) -> Tuple[bool, str]:
    """Time-trim + bbox-crop a single clip out of its raw youtube video.
    No resize: data/celebv/pipeline/main.py does the fit_pad to 832x480 later.
    """
    if out_path.exists() and out_path.stat().st_size > 0:
        return True, "cached"
    size = probe_size(raw_path)
    if size is None:
        return False, "probe-fail"
    W, H = size
    x, y, w, h = compute_crop_px(bbox, H, W, pad, top_extra)
    start, end = time

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(raw_path),
        "-vf", f"crop={w}:{h}:{x}:{y}",
        "-ss", _secs_to_timestr(start),
        "-to", _secs_to_timestr(end),
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-an",
        str(out_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=600)
    except subprocess.TimeoutExpired:
        return False, "ffmpeg-timeout"
    except FileNotFoundError:
        return False, "ffmpeg-not-installed"
    if proc.returncode != 0 or not out_path.exists():
        tail = proc.stderr.decode(errors="ignore")[-200:] if proc.stderr else ""
        return False, f"ffmpeg-fail:{tail}"
    return True, "ok"


# ============================================================
#                     selection / pooling
# ============================================================
def select_pending(
    clips: List[Clip],
    downloaded: Dict[str, dict],
    failed: Dict[str, dict],
    clip_dir: Path,
    retry_failed: bool,
    limit: int,
) -> List[Clip]:
    """Pick up to `limit` clips not yet done. A clip is done if it's in the
    downloaded ledger or already has clip/{clip_id}.mp4. Failed clips are skipped
    unless --retry-failed."""
    pending: List[Clip] = []
    for c in clips:
        cid = c[0]
        if cid in downloaded:
            continue
        if (clip_dir / f"{cid}.mp4").exists():
            continue
        if (not retry_failed) and cid in failed:
            continue
        pending.append(c)
        if limit > 0 and len(pending) >= limit:
            break
    return pending


def pending_per_ytb(selected: List[Clip]) -> Dict[str, int]:
    """Count how many selected clips each ytb_id still owes, for raw cleanup."""
    counts: Dict[str, int] = {}
    for _, ytb_id, _, _ in selected:
        counts[ytb_id] = counts.get(ytb_id, 0) + 1
    return counts


# ============================================================
#                            main
# ============================================================
def main():
    p = argparse.ArgumentParser("CelebV-HQ downloader + raw clip extractor")
    p.add_argument("--info", default="data/celebv/pipeline/celebvhq_info.json")
    p.add_argument("--raw-root", default="/mnt/highspeed/users/Arcus/CELEBV")
    p.add_argument("--limit", type=int, default=200,
                   help="clips to acquire this run (-1 = all remaining)")
    p.add_argument("--pool", type=int, default=100,
                   help="download this many clips' raw videos, then extract, then repeat")
    p.add_argument("--workers", type=int, default=6, help="concurrent yt-dlp downloads")
    p.add_argument("--proc-workers", type=int, default=12, help="concurrent ffmpeg crops")
    p.add_argument("--proxy", default=None)
    p.add_argument("--no-aria2c", action="store_true", help="disable aria2c external downloader")
    p.add_argument("--dl-timeout", type=int, default=900, help="per-video yt-dlp timeout (s)")
    p.add_argument("--crop-pad", type=float, default=0.10,
                   help="symmetric bbox outer padding (fraction of bbox side)")
    p.add_argument("--crop-top-extra", type=float, default=0.15,
                   help="extra upward padding for hair (fraction of bbox height)")
    p.add_argument("--keep-raw", action="store_true",
                   help="keep _raw_tmp youtube downloads instead of deleting them")
    p.add_argument("--retry-failed", action="store_true",
                   help="re-attempt clips previously recorded in failed.json")
    args = p.parse_args()

    raw_root = Path(args.raw_root)
    clip_dir = raw_root / "clip"
    tmp_dir = raw_root / "_raw_tmp"
    downloaded_path = raw_root / "downloaded.json"
    failed_path = raw_root / "failed.json"
    clip_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    clips = load_clips(Path(args.info))
    downloaded = load_ledger(downloaded_path)
    failed = load_ledger(failed_path)
    print(f"[celebv] total clips in info.json : {len(clips)}")
    print(f"[celebv] already downloaded       : {len(downloaded)}")
    print(f"[celebv] previously failed        : {len(failed)}")

    selected = select_pending(clips, downloaded, failed, clip_dir,
                              args.retry_failed, args.limit)
    if not selected:
        print("[celebv] nothing to do (all requested clips already present).")
        return
    print(f"[celebv] acquiring this run        : {len(selected)} clips "
          f"(pool={args.pool}, workers={args.workers}, proc-workers={args.proc_workers})")

    owe = pending_per_ytb(selected)   # ytb_id -> remaining clip count (for raw cleanup)
    n_ok, n_fail = 0, 0
    pbar = tqdm(total=len(selected), desc="celebv", unit="clip")

    # iterate in pools of clips
    for start in range(0, len(selected), args.pool):
        batch = selected[start:start + args.pool]

        # ---- phase 1: download unique raw videos for this batch ----
        batch_ytbs = sorted({c[1] for c in batch})
        dl_status: Dict[str, Tuple[bool, str]] = {}
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            fut = {
                ex.submit(download_raw, ytb, tmp_dir / f"{ytb}.mp4",
                          args.proxy, not args.no_aria2c, args.dl_timeout): ytb
                for ytb in batch_ytbs
            }
            for f in as_completed(fut):
                ytb = fut[f]
                try:
                    dl_status[ytb] = f.result()
                except Exception as e:  # noqa: BLE001
                    dl_status[ytb] = (False, f"exc:{type(e).__name__}")

        # ---- phase 2: extract clips whose raw video downloaded ok ----
        proc_jobs = [c for c in batch if dl_status.get(c[1], (False, ""))[0]]
        # clips whose source failed -> record as failed, advance pbar
        for cid, ytb, _, _ in batch:
            if not dl_status.get(ytb, (False, ""))[0]:
                failed[cid] = {"ytb_id": ytb, "reason": dl_status.get(ytb, (False, "?"))[1]}
                owe[ytb] -= 1
                n_fail += 1
                pbar.update(1)

        with ThreadPoolExecutor(max_workers=args.proc_workers) as ex:
            fut = {
                ex.submit(process_clip, tmp_dir / f"{c[1]}.mp4",
                          clip_dir / f"{c[0]}.mp4", c[3], c[2],
                          args.crop_pad, args.crop_top_extra): c
                for c in proc_jobs
            }
            for f in as_completed(fut):
                cid, ytb, _, _ = fut[f]
                try:
                    ok, reason = f.result()
                except Exception as e:  # noqa: BLE001
                    ok, reason = False, f"exc:{type(e).__name__}"
                if ok:
                    downloaded[cid] = {"ytb_id": ytb}
                    failed.pop(cid, None)
                    n_ok += 1
                else:
                    failed[cid] = {"ytb_id": ytb, "reason": reason}
                    n_fail += 1
                owe[ytb] -= 1
                pbar.update(1)

        # ---- phase 3: persist ledgers + free raw videos no longer owed ----
        save_ledger(downloaded_path, downloaded)
        save_ledger(failed_path, failed)
        if not args.keep_raw:
            for ytb in batch_ytbs:
                if owe.get(ytb, 0) <= 0:
                    rp = tmp_dir / f"{ytb}.mp4"
                    try:
                        rp.unlink(missing_ok=True)
                    except OSError:
                        pass

    pbar.close()
    print(f"[celebv] done. ok={n_ok}  fail={n_fail}  "
          f"ledger={len(downloaded)} total in {downloaded_path}")


if __name__ == "__main__":
    main()
