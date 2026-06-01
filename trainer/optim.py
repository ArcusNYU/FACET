"""
trainer/optim.py

Optimizer + LR scheduler factories.

Responsibilities:
  1. build_optimizer(model, cfg_train)
       Collects ONLY the LoRA parameters (lora_down / lora_up) that the
       trainer is allowed to update. Frozen base / VAE / T5 params are
       filtered out via requires_grad.
  2. build_lr_scheduler(optimizer, cfg_train, total_steps)
       Hand-rolled LambdaLR. Supports:
         - "constant"
         - "constant_with_warmup"
       (No transformers dependency; we already have heavy enough deps.)

Both factories follow DiffSynth's defaults (see DiffSynth runner.py L27-28):
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

DiffSynth uses a plain ConstantLR; we add an optional warmup ramp (250 steps
typical) because LoRA is more sensitive to lr spike in the first hundred steps
than full fine-tuning.

Public API:
    trainable_parameters(model) -> List[Parameter]
    count_trainable(model)      -> (n_trainable, n_total)
    build_optimizer(model, cfg_train) -> torch.optim.Optimizer
    build_lr_scheduler(optimizer, cfg_train, total_steps) -> _LRScheduler
"""

from __future__ import annotations

# 1. Imports ------------------------------------------------------------------
import logging
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from trainer.config import TrainConfig

logger = logging.getLogger(__name__)


# 2. Param filtering ----------------------------------------------------------


def trainable_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    Return the list of parameters with requires_grad=True.

    For FACET, this is exactly the lora_down / lora_up tensors injected by
    facet.lora.inject_lora (everything else was frozen by
    FACETWanModel._freeze_base before LoRA injection).

    We DO NOT filter by name string ("lora_down" / "lora_up") on purpose:
    requires_grad is the single source of truth, and any future module the
    user marks trainable (e.g. a new adapter) will be picked up automatically.
    """
    return [p for p in model.parameters() if p.requires_grad]


def count_trainable(model: nn.Module) -> Tuple[int, int]:
    """(n_trainable, n_total) parameter counts."""
    n_train = 0
    n_total = 0
    for p in model.parameters():
        n = p.numel()
        n_total += n
        if p.requires_grad:
            n_train += n
    return n_train, n_total


# 3. Optimizer ----------------------------------------------------------------


def build_optimizer(
    model: nn.Module,
    cfg_train: TrainConfig,
) -> torch.optim.Optimizer:
    """
    Build the optimizer over LoRA-only parameters.

    Supported (cfg_train.optimizer.name):
        - "adamw"  (DiffSynth default)

    Raises if the trainable parameter list is empty (would silently train
    nothing otherwise).
    """
    params = trainable_parameters(model)
    if len(params) == 0:
        raise RuntimeError(
            "[trainer.optim] No trainable parameters found. "
            "Did FACETWanModel._init_lora run? Did _freeze_base get called "
            "BEFORE LoRA injection? Check facet/model.py."
        )

    n_train, n_total = count_trainable(model)
    logger.info(
        "[trainer.optim] Trainable params: %s / %s (%.4f%%)",
        f"{n_train:,}", f"{n_total:,}",
        100.0 * n_train / max(1, n_total),
    )

    name = cfg_train.optimizer.name.lower()
    if name == "adamw":
        # AdamW matches DiffSynth's runner.py L27.
        return torch.optim.AdamW(
            params,
            lr=float(cfg_train.optimizer.lr),
            betas=tuple(cfg_train.optimizer.betas),
            weight_decay=float(cfg_train.optimizer.weight_decay),
            eps=float(cfg_train.optimizer.eps),
        )

    raise ValueError(
        f"[trainer.optim] Unsupported optimizer.name={name!r}; "
        "expected 'adamw'."
    )


# 4. LR scheduler -------------------------------------------------------------


def _lr_lambda_constant(_step: int) -> float:
    """No-op lambda: keep base_lr forever."""
    return 1.0


def _make_lr_lambda_constant_with_warmup(warmup_steps: int):
    """
    Linear warmup from 0 -> 1 over `warmup_steps`, then constant.

    Defensive: if warmup_steps <= 0, behave like 'constant'.
    """
    warmup_steps = max(0, int(warmup_steps))

    def _lr_lambda(step: int) -> float:
        if warmup_steps == 0:
            return 1.0
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        return 1.0

    return _lr_lambda


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg_train: TrainConfig,
    total_steps: int,
) -> LambdaLR:
    """
    Build the LR scheduler. Counted in OPTIMIZER STEPS (post-accumulation).

    Supported (cfg_train.scheduler.name):
        - "constant"
        - "constant_with_warmup"

    `total_steps` is passed in to allow future schedulers (cosine / linear
    decay) without changing the public signature.
    """
    name = cfg_train.scheduler.name.lower()
    if name == "constant":
        lr_lambda = _lr_lambda_constant
    elif name == "constant_with_warmup":
        warmup = int(cfg_train.scheduler.warmup_steps)
        if warmup > total_steps:
            logger.warning(
                "[trainer.optim] warmup_steps=%d > total_steps=%d; "
                "scheduler will warm up across the entire run.",
                warmup, total_steps,
            )
        lr_lambda = _make_lr_lambda_constant_with_warmup(warmup)
    else:
        raise ValueError(
            f"[trainer.optim] Unsupported scheduler.name={name!r}; "
            "expected 'constant' or 'constant_with_warmup'."
        )

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
