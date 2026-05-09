"""
Dataloader smoke test + visual dump.

Iterates the train loader for --n samples (default 20) and:
  - Prints clip_id / source / path / caption / category to terminal.
  - Dumps ref_img.png / masked_video.mp4 / mask.mp4 to ./visual_exampler/{cid}/.
  - Checks tgt_latent / t5_emb shape (no value stats -- they are encoded).

build_loaders and collate_batch live in loader.py (import from there in train.py).

Run:
    python loader_visual.py [--cfg data/config.yaml] [--n 20] [--out ./visual_exampler]
"""

from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from loader import build_loaders       # canonical builder, reused by train.py


# ============================================================
#                   Tensor -> uint8 helpers
# ============================================================
def video_to_uint8(v: torch.Tensor) -> np.ndarray:
    """[T,3,H,W] in [-1,1] -> [T,H,W,3] uint8 RGB."""
    v = v.detach().cpu().clamp(-1.0, 1.0)
    v = v.add(1.0).mul(127.5).clamp(0, 255).byte()
    return v.permute(0, 2, 3, 1).contiguous().numpy()


def image_to_uint8(img: torch.Tensor) -> np.ndarray:
    """[3,H,W] in [-1,1] -> [H,W,3] uint8 RGB."""
    img = img.detach().cpu().clamp(-1.0, 1.0)
    img = img.add(1.0).mul(127.5).clamp(0, 255).byte()
    return img.permute(1, 2, 0).contiguous().numpy()


def mask_to_uint8(m: torch.Tensor) -> np.ndarray:
    """[T,1,H,W] in {0,1} -> [T,H,W,3] uint8 grayscale (fg=255)."""
    m = m.detach().cpu().squeeze(1).clamp(0, 1).mul(255).byte()
    return m.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous().numpy()


# ============================================================
#                    MP4 / PNG writers
# ============================================================
def write_mp4(frames_rgb: np.ndarray, out_path: Path, fps: int = 24) -> None:
    """Pipe RGB frames to ffmpeg. Falls back to per-frame PNG dump if ffmpeg
    is unavailable."""
    T, H, W, _ = frames_rgb.shape
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-an",
        str(out_path),
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        from PIL import Image
        dump_dir = out_path.with_suffix(".frames")
        dump_dir.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames_rgb):
            Image.fromarray(f).save(dump_dir / f"{i:03d}.png")
        print(f"  [warn] ffmpeg not found -> {T} frames -> {dump_dir}")
        return
    _, err = proc.communicate(frames_rgb.tobytes())
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg: {err.decode(errors='ignore')[:300]}")


def write_png(arr: np.ndarray, out_path: Path) -> None:
    from PIL import Image
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(out_path)


# ============================================================
#                  Shape-only tensor reporter
# ============================================================
def report_shape(name: str, t: Optional[torch.Tensor]) -> str:
    """For encoded tensors (latent / t5_emb): report shape only."""
    if t is None:
        return f"{name}: None  (cache miss -- run data/openvid/pipeline/cache.py)"
    if not torch.is_tensor(t):
        return f"{name}: not a tensor ({type(t).__name__})"
    return f"{name}: shape={tuple(t.shape)}  dtype={t.dtype}"


# ============================================================
#                            Main
# ============================================================
def main():
    p = argparse.ArgumentParser("dataloader smoke test + visual dump")
    p.add_argument("--cfg", default="data/config.yaml")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--out", default="./visual_exampler")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, train_sampler, val_sampler = build_loaders(
        cfg_path=args.cfg,
        batch_size=1,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    train_sampler.set_epoch(0)

    print(f"[loader] train concat={len(train_loader.dataset)}  quota/epoch={len(train_sampler)}")
    print(f"[loader] val   concat={len(val_loader.dataset)}    quota/epoch={len(val_sampler)}")
    print(f"[loader] dumping {args.n} samples -> {out_root.resolve()}\n")

    seen = 0
    for batch in train_loader:
        if seen >= args.n:
            break

        # collate keeps everything as List; batch_size=1 so [0] unwraps the single sample
        cid      = batch["clip_id"][0]
        source   = batch["source"][0]
        path     = batch["path"][0]
        caption  = batch["caption"][0]
        cat      = batch["category"][0]
        masked   = batch["masked_video"][0]   # [T,3,H,W] in [-1,1]
        mask     = batch["mask"][0]           # [T,1,H,W] in {0,1}
        ref_img  = batch["ref_img"][0]        # [3,H,W]   in [-1,1]
        tgt_lat  = batch["tgt_latent"][0]     # Tensor or None
        t5_emb   = batch["t5_emb"][0]         # Tensor or None

        # ---- terminal output ----
        print(f"[{seen+1:02d}/{args.n}] clip_id  = {cid}")
        print(f"        source   = {source}")
        print(f"        path     = {path}")
        print(f"        category = {cat}")
        print(f"        caption  = {caption!r}")
        print(f"        masked_video : {tuple(masked.shape)}  "
              f"min={masked.min().item():+.3f} max={masked.max().item():+.3f}")
        print(f"        mask         : {tuple(mask.shape)}  "
              f"fg_ratio={mask.mean().item():.3f}")
        print(f"        ref_img      : {tuple(ref_img.shape)}  "
              f"min={ref_img.min().item():+.3f} max={ref_img.max().item():+.3f}")
        print(f"        {report_shape('tgt_latent', tgt_lat)}")
        print(f"        {report_shape('t5_emb    ', t5_emb)}")

        # ---- file dump ----
        cdir = out_root / cid
        cdir.mkdir(parents=True, exist_ok=True)

        write_png(image_to_uint8(ref_img),  cdir / "ref_img.png")
        write_mp4(video_to_uint8(masked),   cdir / "masked_video.mp4", fps=args.fps)
        write_mp4(mask_to_uint8(mask),      cdir / "mask.mp4",         fps=args.fps)

        summary = {
            "clip_id":  cid,
            "source":   source,
            "path":     path,
            "category": cat,
            "caption":  caption,
            "shapes": {
                "masked_video": list(masked.shape),
                "mask":         list(mask.shape),
                "ref_img":      list(ref_img.shape),
                "tgt_latent":   list(tgt_lat.shape) if torch.is_tensor(tgt_lat) else None,
                "t5_emb":       list(t5_emb.shape)  if torch.is_tensor(t5_emb)  else None,
            },
        }
        with open(cdir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"        -> {cdir}\n")
        seen += 1

    print(f"[loader] done. {seen} samples written to {out_root.resolve()}")


if __name__ == "__main__":
    main()
