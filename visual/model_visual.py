"""
Loading example + structure visualization for FACETWanModel.

What this script does:
    1. Instantiates `FACETWanModel` from `facet/config.yaml`
       (which loads WAN2.1-VACE-1.3B + freezes base + injects LoRA).
    2. Prints a compact module tree with per-submodule parameter counts
       (total / trainable) so you can eyeball which subtrees actually carry
       gradients after LoRA injection.
    3. Prints a final summary table:
         - total params
         - trainable params (= LoRA params)
         - frozen params (= base model)
         - trainable ratio
         - per-target-module breakdown (q / k / v / o / ffn.0 / ffn.2)

Usage (from project root):
    python visual/model_visual.py
    python visual/model_visual.py --config facet/config.yaml --max-depth 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0") # for A100

import torch
import torch.nn as nn

# Make the FACET project root importable when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from facet.model import FACETWanModel  # noqa: E402
from facet.lora import LoRALinear  # noqa: E402


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _human(n: int) -> str:
    """Format an int like 1234567 -> '1.23M'."""
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 1_000:
        return f"{n / 1e3:.2f}K"
    return str(n)


def _count_params(module: nn.Module) -> Tuple[int, int]:
    """Return (total_params, trainable_params) of a module (recursive)."""
    total = 0
    trainable = 0
    for p in module.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable


def _build_tree_lines(
    module: nn.Module,
    name: str = "model",
    prefix: str = "",
    is_last: bool = True,
    depth: int = 0,
    max_depth: int = 3,
    collapse_repeated: bool = True,
) -> list:
    """
    Render a torch.nn module tree with parameter counts at each level.

    - max_depth: stop recursing past this depth (deeper subtrees are still
      counted, just rolled up).
    - collapse_repeated: when an nn.ModuleList holds many homogeneous blocks
      (e.g. dit.blocks.0 ... dit.blocks.29), show only block[0] expanded and
      a "...x29 more" summary instead of dumping all 30.
    """
    lines = []
    total, trainable = _count_params(module)
    cls_name = module.__class__.__name__
    tag = ""
    if isinstance(module, LoRALinear):
        tag = " [LoRA]"

    branch = "" if depth == 0 else ("└── " if is_last else "├── ")
    head = (
        f"{prefix}{branch}{name} ({cls_name}){tag}"
        f"  total={_human(total)} trainable={_human(trainable)}"
    )
    lines.append(head)

    if depth >= max_depth:
        return lines

    child_prefix = "" if depth == 0 else (prefix + ("    " if is_last else "│   "))

    children = list(module.named_children())
    if len(children) == 0:
        return lines

    # Detect repeated homogeneous blocks (typically nn.ModuleList children).
    if collapse_repeated and len(children) > 4:
        cls_seq = [type(c).__name__ for _, c in children]
        if len(set(cls_seq)) == 1:
            # All children share the same class: collapse.
            first_name, first_child = children[0]
            lines += _build_tree_lines(
                first_child,
                name=first_name,
                prefix=child_prefix,
                is_last=False,
                depth=depth + 1,
                max_depth=max_depth,
                collapse_repeated=collapse_repeated,
            )
            rest = children[1:]
            rest_total = sum(_count_params(c)[0] for _, c in rest)
            rest_train = sum(_count_params(c)[1] for _, c in rest)
            lines.append(
                f"{child_prefix}└── ...x{len(rest)} more {cls_seq[0]}  "
                f"total={_human(rest_total)} trainable={_human(rest_train)}"
            )
            return lines

    for i, (cn, child) in enumerate(children):
        lines += _build_tree_lines(
            child,
            name=cn,
            prefix=child_prefix,
            is_last=(i == len(children) - 1),
            depth=depth + 1,
            max_depth=max_depth,
            collapse_repeated=collapse_repeated,
        )
    return lines


def _lora_breakdown(model: nn.Module) -> dict:
    """
    Bucket trainable LoRA parameters by (a) which subtree (dit.blocks /
    dit.vace_blocks / other) and (b) which target module suffix
    (q / k / v / o / ffn.0 / ffn.2 / before_proj / after_proj).
    """
    by_subtree: dict = {}
    by_target: dict = {}

    for full_name, mod in model.named_modules():
        if not isinstance(mod, LoRALinear):
            continue
        n_train = sum(p.numel() for p in mod.parameters() if p.requires_grad)

        if ".vace_blocks." in full_name or full_name.startswith("vace."):
            subtree = "vace_blocks"
        elif ".blocks." in full_name or full_name.startswith("dit.blocks"):
            subtree = "base_blocks"
        else:
            subtree = "other"
        by_subtree[subtree] = by_subtree.get(subtree, 0) + n_train

        # Target = last 1 or 2 dotted parts that match a typical LoRA target.
        parts = full_name.split(".")
        for target in ("ffn.0", "ffn.2"):
            if ".".join(parts[-2:]) == target:
                by_target[target] = by_target.get(target, 0) + n_train
                break
        else:
            tail = parts[-1]
            if tail in {"q", "k", "v", "o", "before_proj", "after_proj"}:
                by_target[tail] = by_target.get(tail, 0) + n_train
            else:
                by_target[tail] = by_target.get(tail, 0) + n_train

    return {"by_subtree": by_subtree, "by_target": by_target}


def _print_separator(title: str = "", width: int = 88) -> None:
    if title:
        title = f" {title} "
        pad = (width - len(title)) // 2
        print("=" * pad + title + "=" * (width - pad - len(title)))
    else:
        print("=" * width)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "facet" / "config.yaml"),
        help="Path to FACET config yaml.",
    )
    ap.add_argument(
        "--max-depth", type=int, default=4,
        help="Tree depth cap for the structure dump.",
    )
    ap.add_argument(
        "--no-collapse", action="store_true",
        help="Don't collapse repeated homogeneous blocks (full dump).",
    )
    args = ap.parse_args()

    _print_separator("FACET model load")
    print(f"config: {args.config}")
    print(f"torch:  {torch.__version__}   cuda available: {torch.cuda.is_available()}")
    print()

    model = FACETWanModel.from_config(args.config)

    # ---- 1. Top-level summary ----
    total, trainable = _count_params(model)
    frozen = total - trainable
    print(
        f"dtype: {model.dtype}    device: {model.device}    "
        f"gradient_checkpointing: {model.cfg.gradient_checkpointing}"
    )
    print(
        f"total={_human(total)}  trainable={_human(trainable)}  "
        f"frozen={_human(frozen)}  "
        f"trainable_ratio={(trainable / max(total, 1)) * 100:.3f}%"
    )

    # ---- 2. Module tree ----
    _print_separator("Module tree (collapsed)")
    for line in _build_tree_lines(
        model,
        name="FACETWanModel",
        max_depth=args.max_depth,
        collapse_repeated=not args.no_collapse,
    ):
        print(line)

    # ---- 3. LoRA breakdown ----
    _print_separator("LoRA trainable breakdown")
    br = _lora_breakdown(model)
    print("by subtree:")
    for k, v in sorted(br["by_subtree"].items(), key=lambda x: -x[1]):
        print(f"  {k:14s}  {_human(v):>10s}  ({v:,})")
    print("by target module:")
    for k, v in sorted(br["by_target"].items(), key=lambda x: -x[1]):
        print(f"  {k:14s}  {_human(v):>10s}  ({v:,})")

    # ---- 4. Per-component (dit / vace / vae / text_encoder) ----
    _print_separator("Per-component params")
    for name in ("dit", "vace", "vae", "text_encoder"):
        sub = getattr(model, name, None)
        if sub is None:
            print(f"  {name:14s}  (not loaded)")
            continue
        t, tr = _count_params(sub)
        print(
            f"  {name:14s}  total={_human(t):>9s}  trainable={_human(tr):>9s}  "
            f"({(tr / max(t, 1)) * 100:6.3f}%)"
        )

    # ---- 5. Sanity: list a few injected LoRA module names ----
    _print_separator("Sample LoRA module paths (first 12)")
    lora_paths = [n for n, m in model.named_modules() if isinstance(m, LoRALinear)]
    print(f"total LoRA modules: {len(lora_paths)}")
    for p in lora_paths[:12]:
        print(f"  {p}")
    if len(lora_paths) > 12:
        print(f"  ... ({len(lora_paths) - 12} more)")

    _print_separator()


if __name__ == "__main__":
    main()
