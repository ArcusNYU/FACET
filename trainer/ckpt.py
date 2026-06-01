"""
trainer/ckpt.py

Phase 2+ stub: checkpoint save / load / top-K bookkeeping.

Final form will:
  - save_topk(...) : write LoRA safetensors under runs/<run>/ckpts/ AND
                     symlink/copy the top-K to ckpt/<run_name>_best.safetensors
  - save_last(...) : always overwrite ckpt/<run_name>_last.safetensors
  - update manifest.json (path, primary_metric value, source run_dir).
  - maybe_resume(...) : reload model + optimizer + lr_scheduler from a runs/
                        snapshot (Phase 2.x; not in trainer.txt's Phase 1).
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def save_topk(
    accelerator,
    model,
    optimizer,
    lr_scheduler,
    metrics: Dict[str, Any],
    epoch: int,
    global_step: int,
    output_root,
    cfg_train,
) -> None:
    """Stub for Phase 2+. Does nothing yet."""
    if accelerator.is_main_process:
        logger.info(
            "[trainer.ckpt] save_topk stub: epoch=%d step=%d metrics=%s",
            epoch, global_step, metrics,
        )
    # TODO Phase 2+: write LoRA safetensors + top-K bookkeeping.


def save_last(accelerator, model, output_root, cfg_train) -> None:
    """Stub for Phase 2+."""
    pass


def maybe_resume(accelerator, model, optimizer, lr_scheduler, resume_from) -> int:
    """Stub. Returns the resumed global_step (0 if no resume)."""
    if resume_from:
        logger.warning(
            "[trainer.ckpt] resume_from=%s requested but maybe_resume is "
            "a Phase 2+ stub; starting from step 0.",
            resume_from,
        )
    return 0
