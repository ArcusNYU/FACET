from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import FACETLoRAConfig
from typing import Optional, Sequence, List
from utils import _get_parent_module


# ============================================================
# LoRA Implementation
# ============================================================


class LoRALinear(nn.Module):
    """
    Minimal LoRA wrapper for nn.Linear.

    y = base(x) + scale * B(A(dropout(x)))

    The base linear layer is frozen.
    """
    # TODO: 32 -> 128
    def __init__(
        self,
        base: nn.Linear,
        rank: int = 32,
        alpha: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_features = base.in_features
        out_features = base.out_features

        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)

        # Common LoRA init: down random, up zero.
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.lora_up(self.lora_down(self.dropout(x))) * self.scale
        # Safety cast: under accelerate's autocast policy lora_* params may live
        # in fp32 while the base linear stays in bf16. Without this match the
        # outer + would upcast everything and silently break dtype invariants.
        if lora_out.dtype != base_out.dtype:
            lora_out = lora_out.to(base_out.dtype)
        return base_out + lora_out


# ============================================================
# B. LoRA targeting and injection
# ============================================================


def _suffix_match(name: str, target_modules: Sequence[str]) -> Optional[str]:
    """Return the matched suffix or None. Suffixes can be 'q' or 'ffn.0'."""
    for tm in target_modules:
        if name == tm or name.endswith("." + tm):
            return tm
    return None


def lora_targets(
    name: str,
    target_modules: Sequence[str],
    in_base_block: bool,
    lora_cfg: FACETLoRAConfig,
) -> bool:
    """
    Decide whether the nn.Linear at module path `name` should be wrapped.

    Rules:
      - Must be under dit.blocks.*.
      - `lora_cfg.on_base_blocks` toggles whole regions.
      - For q/k/v/o, cross_attn is excluded unless on_cross_attn=True.
    """
    if in_base_block and not lora_cfg.on_base_blocks:
        return False

    matched = _suffix_match(name, target_modules)

    if matched is None:
        return False

    # q/k/v/o: skip cross_attn unless explicitly allowed.
    if matched in ("q", "k", "v", "o"):
        if ".cross_attn." in name and not lora_cfg.on_cross_attn:
            return False
        # Belt-and-suspenders: ensure self_attn membership unless cross_attn opt-in.
        if ".self_attn." not in name and ".cross_attn." not in name:
            return False

    return True


def inject_lora(
    root: nn.Module,
    lora_cfg: FACETLoRAConfig,
) -> List[str]:
    """
    Replace target nn.Linear modules with LoRALinear in-place.

    Returns the list of replaced module paths (for logging / debugging).

    NOTE: `root` must be the DiT (transformer) module.
    Deliberately do NOT iterate over vae / text_encoder.
    """
    replaced: List[str] = []

    for name, module in list(root.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        in_base_block = name.startswith("blocks.")

        if not lora_targets(
            name=name,
            target_modules=lora_cfg.target_modules,
            in_base_block=in_base_block,
            lora_cfg=lora_cfg,
        ):
            continue

        parent, child_name = _get_parent_module(root, name)
        setattr(
            parent,
            child_name,
            LoRALinear(
                base=module,
                rank=lora_cfg.rank,
                alpha=lora_cfg.alpha,
                dropout=lora_cfg.dropout,
            ),
        )
        replaced.append(name)

    return replaced

