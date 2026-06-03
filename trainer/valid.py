"""
trainer/valid.py

Phase 3 stub: in-loop validation.

Final form:
  - every cfg.train.val_every_steps -> compute val loss + PSNR/SSIM/LPIPS/CLIPSim
  - every 1000 steps -> generate N=cfg.validate.num_samples videos, save under
        runs/<run>/samples/steps_<global_step>/
  - return metrics dict so trainer.ckpt.save_topk can select top-K (cfg.validate.topk).
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def run(
    model,
    val_loader,
    val_sampler,
    cfg,
    epoch: int,
    global_step: int,
) -> Dict[str, Any]:
    """
    Phase 3 will fill this in. Returns an empty dict so train.py's downstream
    `trainer.logger.log_metrics(metrics, ...)` + `save_topk(metrics, ...)`
    calls don't crash.
    """
    logger.info(
        "[trainer.valid] stub: skipping validation at epoch=%d step=%d",
        epoch, global_step,
    )
    # TODO: 准备 validation pbar
    return {}
