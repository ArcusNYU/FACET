"""
trainer/logger.py

Phase 1.5 stub: cloud + console logger.

Final form will:
  - call `accelerator.init_trackers(cfg.log.project_name, config=cfg.flat(), ...)`
  - write {output_root}/config_snapshot.yaml via cfg.dump_snapshot
  - write {output_root}/metrics.jsonl (one row per logged step)
  - mirror scalars to mlflow / tensorboard backend per cfg.log.backend.

Current minimal contract:
  - setup(accelerator, cfg, output_root)   : dump config snapshot
  - log_step(...)                          : prints to console; jsonl later
  - log_metrics(...)                       : prints to console; jsonl later
  - finish()                               : no-op
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def setup(accelerator, cfg, output_root: Path) -> None:
    """Dump config_snapshot.yaml on rank 0. mlflow / TB hookup pending."""
    if accelerator.is_main_process:
        snap_path = Path(output_root) / "config_snapshot.yaml"
        cfg.dump_snapshot(snap_path)
        logger.info("[trainer.logger] config snapshot -> %s", snap_path)
    # TODO Phase 1.5: accelerator.init_trackers(...)


def log_step(loss: float, lr: float, grad_norm: float | None, global_step: int) -> None:
    """Console-only step log; mlflow scalar later."""
    gn = "n/a" if grad_norm is None else f"{grad_norm:.4f}"
    logger.info(
        "[step %d] loss=%.5f lr=%.3e grad_norm=%s",
        global_step, loss, lr, gn,
    )
    # TODO Phase 1.5: accelerator.log({...}, step=global_step) + jsonl append


def log_metrics(metrics: Dict[str, Any], global_step: int) -> None:
    """Console-only metrics log."""
    pretty = ", ".join(f"{k}={v}" for k, v in metrics.items())
    logger.info("[val @ step %d] %s", global_step, pretty)
    # TODO Phase 1.5: same as log_step.


def finish() -> None:
    """Tear-down stub. Will call accelerator.end_training()."""
    pass
