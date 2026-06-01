"""
trainer/sum.py

Phase 1.5 stub: print/log trainable parameter table on launch.

Final form will be a per-target breakdown (q / k / v / o / ffn.0 / ffn.2)
mirroring visual/model_visual.py::_lora_breakdown, but WITHOUT the module
tree (per trainer.txt note: tree is too noisy).

Current minimal contract: trainable_stats(model, output_root)
  - logs (n_trainable / n_total / ratio) to console
  - returns the same triple so train.py can write into mlflow later.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import torch.nn as nn

logger = logging.getLogger(__name__)


def trainable_stats(model: nn.Module, output_root: Path | None = None) -> Tuple[int, int, float]:
    """
    Phase 1 minimal: return (n_trainable, n_total, ratio).

    Phase 1.5 will:
      - bucket by target module (q/k/v/o/ffn.0/ffn.2)
      - write a Markdown table to {output_root}/params.txt
    """
    n_train = 0
    n_total = 0
    for p in model.parameters():
        n = p.numel()
        n_total += n
        if p.requires_grad:
            n_train += n
    ratio = n_train / max(1, n_total)
    logger.info(
        "[trainer.sum] trainable=%s / total=%s  (%.4f%%)",
        f"{n_train:,}", f"{n_total:,}", 100.0 * ratio,
    )
    # TODO Phase 1.5: per-target breakdown + params.txt dump.
    return n_train, n_total, ratio
