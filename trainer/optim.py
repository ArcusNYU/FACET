"""
trainer/optim.py
reference: https://github.com/modelscope/DiffSynth-Studio/blob/main/diffsynth/diffusion/runner.py

FACET training pipeline Step 4 & 6.
Optimizer + LR scheduler factories + trainable-parameter stats.

Responsibilities:
  1. build_optimizer(model, cfg_train)
       Collects ONLY the LoRA parameters (lora_down / lora_up) that the
       trainer is allowed to update. Frozen base / VAE / T5 params excluded.
  2. build_lr_scheduler(optimizer, cfg_train, total_steps)
       Hand-rolled LambdaLR. (No transformers dependency.) Supports:
         - "constant"
         - "constant_with_warmup"
  3. model_stats(model, output_root)
       Trainable-parameter table (total / trainable / per-target q,k,v,o,
       ffn.0,ffn.2). Logs to console + dumps params.txt.

NOTE: 
    DiffSynth itself uses a plain ConstantLR (no warmup); the warmup is a FACET
    addition because LoRA is more sensitive to an early lr spike than full FT.

Public API:
    trainable_params(model)     -> List[Parameter]
    count_trainable(model)      -> (n_trainable, n_total)
    model_stats(model, ...)     -> dict
    build_optimizer(model, cfg_train) -> torch.optim.Optimizer
    build_lr_scheduler(optimizer, cfg_train, total_steps) -> LambdaLR
"""

from __future__ import annotations

# 1. Imports ------------------------------------------------------------------
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from trainer.config import TrainConfig

logger = logging.getLogger(__name__)


# 2. Param filtering ----------------------------------------------------------


def trainable_params(model: nn.Module) -> List[nn.Parameter]:
    """
    Return the list of parameters with requires_grad=True.

    For FACET, this is exactly the lora_down / lora_up tensors injected by
    facet.lora.inject_lora (everything else was frozen by
    FACETWanModel._freeze_base before LoRA injection).
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


def _match_target(param_name: str, lora_targets: Tuple[str, ...]) -> str:
    """
    Map a trainable LoRA param name to its target module suffix.

    e.g. "dit.blocks.0.self_attn.q.lora_down.weight" -> "q"
         "dit.blocks.0.ffn.0.lora_up.weight"         -> "ffn.0"
    Anything unrecognized falls into "other".
    """
    base = param_name
    for marker in (".lora_down", ".lora_up"):
        if marker in base:
            base = base.split(marker)[0]
            break
    for t in lora_targets:
        if base == t or base.endswith("." + t):
            return t
    return "other"


def _reformat(n: int) -> str:
    """1234567 -> '1.23M'."""
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 1_000:
        return f"{n / 1e3:.2f}K"
    return str(n)


def model_stats(
    model: nn.Module,
    lora_targets: Tuple[str, ...],
    output_root: Optional[Path] = None,
) -> Dict[str, object]:
    """
    Build + log a trainable-parameter table; optionally dump params.txt.

    Reports:
      - total / trainable / frozen counts + trainable ratio
      - per-target breakdown over q / k / v / o / ffn.0 / ffn.2
      - LoRA module count

    NOTE: call on the UNWRAPPED model (before accelerator.prepare) so parameter
    names are clean (dit.blocks.* rather than module.dit.blocks.*).
    """
    n_train, n_total = count_trainable(model)
    n_frozen = n_total - n_train
    ratio = n_train / max(1, n_total)

    by_target: Dict[str, int] = {t: 0 for t in lora_targets}
    by_target["other"] = 0
    n_lora_modules = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ("lora_down" in name) or ("lora_up" in name):
            by_target[_match_target(name, lora_targets)] += p.numel()
            if "lora_down" in name:        # count each LoRALinear once
                n_lora_modules += 1

    # ---- console table ----
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("[trainer.optim] trainable parameter stats")
    lines.append(f"  total     : {_reformat(n_total)}  ({n_total:,})")
    lines.append(f"  trainable : {_reformat(n_train)}  ({n_train:,})   {100.0 * ratio:.4f}%")
    lines.append(f"  frozen    : {_reformat(n_frozen)}  ({n_frozen:,})")
    lines.append(f"  LoRA modules: {n_lora_modules}")
    lines.append("  by target:")
    for t in (*lora_targets, "other"):
        v = by_target[t]
        if v == 0 and t == "other":
            continue
        pct = 100.0 * v / max(1, n_train)
        lines.append(f"    {t:>7s} : {_reformat(v):>9s}  ({pct:5.2f}% of trainable)")
    lines.append("=" * 60)
    table = "\n".join(lines)
    for ln in lines:
        logger.info(ln)

    # ---- params.txt dump ----
    if output_root is not None:
        params_txt = Path(output_root) / "params.txt"
        params_txt.parent.mkdir(parents=True, exist_ok=True)
        params_txt.write_text(table + "\n", encoding="utf-8")
        logger.info("[trainer.optim] params table -> %s", params_txt)

    return {
        "total": n_total,
        "trainable": n_train,
        "frozen": n_frozen,
        "trainable_ratio": ratio,
        "lora_modules": n_lora_modules,
        "by_target": by_target,
    }


# 3. Optimizer ----------------------------------------------------------------


def build_optimizer(
    model: nn.Module,
    cfg_train: TrainConfig,
) -> torch.optim.Optimizer:
    """
    Build the optimizer over LoRA-only parameters.
    """
    params = trainable_params(model)
    if len(params) == 0:
        raise RuntimeError(
            "[trainer.optim] No trainable parameters found. "
        )
    n_train, n_total = count_trainable(model)
    logger.info(
        "[trainer.optim] Trainable params: %s / %s (%.4f%%)",
        f"{n_train:,}", f"{n_total:,}",
        100.0 * n_train / max(1, n_total),
    )

    name = cfg_train.optimizer.name.lower()
    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=float(cfg_train.optimizer.lr),
            betas=tuple(cfg_train.optimizer.betas),
            weight_decay=float(cfg_train.optimizer.weight_decay),
            eps=float(cfg_train.optimizer.eps),
        )

    raise ValueError(
        f"[trainer.optim] Unsupported optimizer.name={name!r}; "
        "expected 'adamw'." #NOTE: 目前只支持AdamW
    )


# 4. LR scheduler -------------------------------------------------------------


def _lr_lambda_constant(_step: int) -> float:
    """No-op lambda: keep base_lr forever."""
    return 1.0


def _lr_lambda_constant_with_warmup(warmup_steps: int):
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
    # NOTE: '_lr_lambda' is a customized lambda function that is used to schedule the learning rate during training


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
        lr_lambda = _lr_lambda_constant_with_warmup(warmup)
    else:
        raise ValueError(
            f"[trainer.optim] Unsupported scheduler.name={name!r}; "
            "expected 'constant' or 'constant_with_warmup'."
        )

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
