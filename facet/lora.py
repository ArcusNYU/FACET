from __future__ import annotations

import math

import torch
import torch.nn as nn


# ============================================================
# LoRA
# ============================================================


class LoRALinear(nn.Module):
    """
    Minimal LoRA wrapper for nn.Linear.

    y = base(x) + scale * B(A(dropout(x)))

    The base linear layer is frozen.
    """

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
