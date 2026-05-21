"""
Project-level shared utilities.

Shared by:
  - loader_visual.py         (visualization)
  - data/openvid/pipeline/main.py  (dataset preprocessing)
  - train.py / eval.py       
"""

from __future__ import annotations
import subprocess
from pathlib import Path

import numpy as np
import torch
from typing import Tuple
from torch import nn


# ============================================================
#                  Tensor -> uint8 converters
# ============================================================
def video_to_uint8(v: torch.Tensor) -> np.ndarray:
    """[T,3,H,W] float in [-1,1] -> [T,H,W,3] uint8 RGB."""
    v = v.detach().cpu().clamp(-1.0, 1.0)
    v = v.add(1.0).mul(127.5).clamp(0, 255).byte()
    return v.permute(0, 2, 3, 1).contiguous().numpy()


def image_to_uint8(img: torch.Tensor) -> np.ndarray:
    """[3,H,W] float in [-1,1] -> [H,W,3] uint8 RGB."""
    img = img.detach().cpu().clamp(-1.0, 1.0)
    img = img.add(1.0).mul(127.5).clamp(0, 255).byte()
    return img.permute(1, 2, 0).contiguous().numpy()


def mask_to_uint8(m: torch.Tensor) -> np.ndarray:
    """[T,1,H,W] float in {0,1} -> [T,H,W,3] uint8 grayscale (fg=255)."""
    m = m.detach().cpu().squeeze(1).clamp(0, 1).mul(255).byte()
    return m.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous().numpy()


# ============================================================
#                       I/O writers
# ============================================================
def write_mp4(
    frames_rgb: np.ndarray,
    out_path: Path,
    fps: int = 24,
    allow_fallback: bool = False,
) -> None:
    """Pipe [T,H,W,3] uint8 RGB frames to ffmpeg (libx264 yuv420p crf18).

    allow_fallback=False (default): raises RuntimeError when ffmpeg is absent
      -- the right behaviour for preprocessing pipelines.
    allow_fallback=True: falls back to a per-frame PNG dump under
      {out_path}.frames/ -- handy for visualization on machines without ffmpeg.
    """
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
        if not allow_fallback:
            raise RuntimeError("ffmpeg not found; install it or pass allow_fallback=True")
        from PIL import Image
        dump_dir = out_path.with_suffix(".frames")
        dump_dir.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames_rgb):
            Image.fromarray(f).save(dump_dir / f"{i:03d}.png")
        print(f"  [warn] ffmpeg missing -> {T} frames dumped to {dump_dir}")
        return
    _, err = proc.communicate(frames_rgb.tobytes())
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {err.decode(errors='ignore')[:400]}")


def write_png(arr: np.ndarray, out_path: Path) -> None:
    """Save [H,W,3] or [H,W,4] uint8 array as PNG."""
    from PIL import Image
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(out_path)


# ============================================================
#                       Model utilities
# ============================================================

def resolve_dtype(dtype: str) -> torch.dtype:
    if dtype in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype in ("fp16", "float16"):
        return torch.float16
    if dtype in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def _get_parent_module(root: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    """
    Given 'blocks.0.self_attn.q', return:
      parent = root.blocks[0].self_attn
      child_name = 'q'
    """
    parts = module_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]