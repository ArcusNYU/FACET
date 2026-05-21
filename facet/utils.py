from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import torch
import torch.nn as nn


# ============================================================
# dtype helpers
# ============================================================


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


# ============================================================
# Module tree helpers
# ============================================================


def _get_parent_module(root: nn.Module, dotted: str) -> Tuple[nn.Module, str]:
    """
    Resolve a dotted module path to (parent_module, child_attr_name).

    Supports numeric segments for ModuleList / Sequential (e.g. "blocks.0.self_attn.q").

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


# ============================================================
# Path helpers
# ============================================================


def _resolve_local_path(dir_or_file: Union[str, Path], pattern: str) -> str:
    """
    Glob `pattern` inside `dir_or_file` and return a single matching path.

    Used by the local checkpoint loader so that yaml entries like
        dir: "./weights/WAN2.1"
        dit: "diffusion_pytorch_model*.safetensors"
    can be resolved without going online.
    """
    base = Path(dir_or_file)
    if not base.exists():
        raise FileNotFoundError(f"Base directory not found: {base}")

    matches = sorted(base.glob(pattern))
    if len(matches) == 0:
        raise FileNotFoundError(
            f"No file matched pattern {pattern!r} under {base}."
        )
    if len(matches) == 1:
        return str(matches[0])
    # Multiple shards (e.g. 14B model). Return list-like joined as comma string is unsafe;
    # callers that expect multiple shards should call `_resolve_local_paths` instead.
    raise ValueError(
        f"Pattern {pattern!r} under {base} matched {len(matches)} files. "
        f"Use _resolve_local_paths for multi-shard checkpoints."
    )


def _resolve_local_paths(dir_or_file: Union[str, Path], pattern: str) -> list[str]:
    """Same as `_resolve_local_path` but returns a sorted list of all matches."""
    base = Path(dir_or_file)
    if not base.exists():
        raise FileNotFoundError(f"Base directory not found: {base}")
    matches = sorted(base.glob(pattern))
    if len(matches) == 0:
        raise FileNotFoundError(
            f"No file matched pattern {pattern!r} under {base}."
        )
    return [str(m) for m in matches]
