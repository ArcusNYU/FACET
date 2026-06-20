"""
Project-wide utilities for the FACET workspace.

Shared by:
  - facet/model.py                 (model)
  - loader_visual.py               (visualization)
  - data/openvid/pipeline/main.py  (dataset preprocessing)
  - train.py / eval.py             (training & evaluation)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import subprocess

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def image_to_uint8(img: torch.Tensor) -> np.ndarray:
    """[3,H,W] float in [-1,1] -> [H,W,3] uint8 RGB."""
    img = img.detach().cpu().clamp(-1.0, 1.0)
    img = img.add(1.0).mul(127.5).clamp(0, 255).byte()
    return img.permute(1, 2, 0).contiguous().numpy()

def mask_to_uint8(m: torch.Tensor) -> np.ndarray:
    """[T,1,H,W] float in {0,1} -> [T,H,W,3] uint8 grayscale (fg=255)."""
    m = m.detach().cpu().squeeze(1).clamp(0, 1).mul(255).byte()
    return m.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous().numpy()

def video_to_uint8(v: torch.Tensor) -> np.ndarray:
    """[T,3,H,W] float in [-1,1] -> [T,H,W,3] uint8 RGB."""
    v = v.detach().cpu().clamp(-1.0, 1.0)
    v = v.add(1.0).mul(127.5).clamp(0, 255).byte()
    return v.permute(0, 2, 3, 1).contiguous().numpy()


def read_mp4(path: Union[str, Path]) -> Optional[torch.Tensor]:
    """Read an mp4 -> [T, 3, H, W] float in [-1, 1]; None if missing/unreadable.
    """
    if not Path(path).exists():
        return None
    try:
        from decord import VideoReader, cpu
    except Exception as e:  # noqa: BLE001
        logger.warning("[utils] decord unavailable (%s); cannot read %s", e, path)
        return None
    try:
        vr = VideoReader(str(path), ctx=cpu(0))
        arr = vr.get_batch(range(len(vr))).asnumpy()        # [T, H, W, 3] uint8
    except Exception as e:  # noqa: BLE001
        logger.warning("[utils] failed reading %s: %s", path, e)
        return None
    return torch.from_numpy(arr).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)


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


# ------------------------------------------------------------
# Project root
# ------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent
# obtain a fixed project root path: /nvmedata/workspace2/users/Arcus/FACET

def _resolve_project_root(p: Union[str, Path]) -> Path:
    """
    Obtain absolute path under the FACET project root.
    推断p在服务器中的绝对路径 (通过组合项目绝对路径+相对于项目根目录的相对路径)
    """
    p = Path(p)
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    return p


# ------------------------------------------------------------
# dtype helpers
# ------------------------------------------------------------

_DTYPE_TABLE = {
    "fp32": torch.float32, "float32": torch.float32, "f32": torch.float32,
    "fp16": torch.float16, "float16": torch.float16, "f16": torch.float16, "half": torch.float16,
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
}


def _resolve_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    """Map a string like 'bf16' to torch.bfloat16. Passes torch.dtype through."""
    if isinstance(dtype, torch.dtype):
        return dtype
    key = str(dtype).lower()
    if key not in _DTYPE_TABLE:
        raise ValueError(
            f"Unknown dtype string: {dtype!r}. "
            f"Expected one of {sorted(_DTYPE_TABLE)}."
        )
    return _DTYPE_TABLE[key]


# ------------------------------------------------------------
# Module-tree helpers
# ------------------------------------------------------------


def _get_parent_module(root: nn.Module, dotted: str) -> Tuple[nn.Module, str]:
    """
    Resolve a dotted module path to (parent_module, child_attr_name).

    Supports numeric segments for ModuleList / Sequential.

    Example:
        parent, child = _get_parent_module(dit, "blocks.0.self_attn.q")
        # parent: dit.blocks[0].self_attn
        # child:  "q"
    """
    parts = dotted.split(".")
    parent: nn.Module = root
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]  # type: ignore[index]
        else:
            parent = getattr(parent, p)
    return parent, parts[-1]


# ------------------------------------------------------------
# Path helpers
# ------------------------------------------------------------


def _has_glob_wildcard(pattern: str) -> bool:
    """
    Check if the pattern contains glob wildcards ('*', '?', '[').
    """
    return any(c in pattern for c in "*?[")


def _resolve_local_path(dir_or_file: Union[str, Path], pattern: str) -> str:
    """
    Resolve `pattern` under `dir_or_file` to a single absolute path.

    - If `pattern` contains glob wildcards ('*', '?', '['), the function globs
      and [asserts exactly ONE match].
    - Otherwise it joins `dir_or_file/pattern` directly. 
      Works for files AND directories (!used by the tokenizer path resolution).
    """
    base = _resolve_project_root(dir_or_file)
    if not base.exists():
        raise FileNotFoundError(f"Base directory not found: {base}")

    if _has_glob_wildcard(pattern):
        matches = sorted(base.glob(pattern))
        if len(matches) == 0:
            raise FileNotFoundError(
                f"No file matched pattern {pattern!r} under {base}."
            )
        if len(matches) > 1:
            raise ValueError(
                f"Pattern {pattern!r} under {base} matched {len(matches)} files. "
                f"Use _resolve_local_paths for multi-shard checkpoints."
            )
        return str(matches[0])

    full = base / pattern
    if not full.exists():
        raise FileNotFoundError(f"Path not found: {full}")
    return str(full)


def _resolve_local_paths(dir_or_file: Union[str, Path], pattern: str) -> List[str]:
    """Same as `_resolve_local_path` but returns a sorted list of all matches."""
    base = _resolve_project_root(dir_or_file)
    if not base.exists():
        raise FileNotFoundError(f"Base directory not found: {base}")

    if _has_glob_wildcard(pattern):
        matches = sorted(base.glob(pattern))
        if len(matches) == 0:
            raise FileNotFoundError(
                f"No file matched pattern {pattern!r} under {base}."
            )
        return [str(m) for m in matches]

    full = base / pattern
    if not full.exists():
        raise FileNotFoundError(f"Path not found: {full}")
    return [str(full)]
