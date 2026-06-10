"""
trainer/logger.py
FACET training pipeline Step 9(logging) & 12.

Console + cloud (mlflow / tensorboard) + local metrics.jsonl logger.

Module-level singleton (one logger per process). Wired once via setup(); the
log_* helpers are then called from the training loop.

logger.setup():
  - dump config_snapshot.yaml via cfg.dump_snapshot
  - point metrics.jsonl at output_root/metrics.jsonl
  - for mlflow: route the tracking store under output_root/logs/mlruns
  - accelerator.init_trackers(project_name, config=cfg.flat(), ...)

NOTE: The accelerator MUST have been built with `log_with=<backend>` (done in
trainer.setup.init_env), otherwise accelerator.log is a no-op.

Tracked scalars:
  train/loss, train/lr, train/grad_norm   (every cfg.train.log_every_steps)
  val/<metric>                            (every cfg.train.val_every_steps)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# Module-level logger state, populated by setup().
_STATE: Dict[str, Any] = {
    "accelerator": None,
    "backend": "none",
    "tracking": False,          # True once init_trackers succeeded
    "jsonl_path": None,         # Path | None (main process only)
    "is_main": False,
} # NOTE: 全局变量


def setup(accelerator, cfg, output_root: Path, track_root: Path) -> None:
    """Dump snapshot, route the tracking store, and init cloud trackers."""
    _STATE["accelerator"] = accelerator
    _STATE["backend"] = (cfg.log.backend or "none").lower()
    _STATE["is_main"] = bool(accelerator.is_main_process)

    if accelerator.is_main_process:
        output_root = Path(output_root)
        track_root = Path(track_root)
        # 1. config snapshot
        snap_path = output_root / "config_snapshot.yaml"
        cfg.dump_snapshot(snap_path)
        logger.info("[trainer.logger] config snapshot -> %s", snap_path)
        # 2. metrics.jsonl target (created lazily on first append)
        _STATE["jsonl_path"] = output_root / "metrics.jsonl"
        # 3. mlflow: pin the tracking store to a FIXED path (track_root) so the
        #    SSH->local mlflow UI mapping never has to change between runs.
        if _STATE["backend"] == "mlflow":
            track_root.mkdir(parents=True, exist_ok=True)
            os.environ["MLFLOW_TRACKING_URI"] = track_root.resolve().as_uri()

    # 4. init cloud trackers (no-op if accelerator built with log_with=None)
    if _STATE["backend"] in ("mlflow", "tensorboard", "wandb"):
        try:
            run_name = cfg.log.cloud_run_name or output_root.name
            init_kwargs = {}
            if _STATE["backend"] == "mlflow":
                init_kwargs = {"mlflow": {"run_name": run_name}}
            elif _STATE["backend"] == "wandb":
                init_kwargs = {"wandb": {"name": run_name}}
            accelerator.init_trackers(
                project_name=cfg.log.project_name,
                config=cfg.flat(),
                init_kwargs=init_kwargs,
            )
            _STATE["tracking"] = True
            if accelerator.is_main_process:
                logger.info("[trainer.logger] trackers initialized (%s).", _STATE["backend"])
        except Exception as e:  # noqa: BLE001 - tracking must never crash training
            _STATE["tracking"] = False
            logger.warning("[trainer.logger] init_trackers failed (%s); cloud logging off.", e)


def _append_jsonl(record: Dict[str, Any]) -> None:
    """Append one json record to metrics.jsonl (main process only)."""
    path: Optional[Path] = _STATE.get("jsonl_path")
    if path is None or not _STATE.get("is_main"):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _track(scalars: Dict[str, float], step: int) -> None:
    """Forward scalars to the cloud tracker (no-op if tracking disabled)."""
    if not _STATE.get("tracking"):
        return
    acc = _STATE.get("accelerator")
    if acc is None:
        return
    try:
        acc.log(scalars, step=step)
    except Exception as e:  # noqa: BLE001
        logger.warning("[trainer.logger] accelerator.log failed: %s", e)


def log_step(loss: float, lr: float, grad_norm: Optional[float], global_step: int) -> None:
    """Per-step training log: console + tracker + jsonl (main process)."""
    if not _STATE.get("is_main"):
        return
    gn = "n/a" if grad_norm is None else f"{grad_norm:.4f}"
    logger.info("[step %d] loss=%.5f lr=%.3e grad_norm=%s", global_step, loss, lr, gn)

    scalars: Dict[str, float] = {"train/loss": float(loss), "train/lr": float(lr)}
    if grad_norm is not None:
        scalars["train/grad_norm"] = float(grad_norm)
    _track(scalars, global_step)  #NOTE: 后续要log新的内容 就在这下面直接添加
    _append_jsonl({"step": global_step, "phase": "train", **scalars})


def log_metrics(metrics: Dict[str, Any], global_step: int) -> None:
    """Validation metrics log: console + tracker + jsonl (main process)."""
    if not _STATE.get("is_main"):
        return
    if not metrics:
        return
    pretty = ", ".join(f"{k}={v}" for k, v in metrics.items())
    logger.info("[val @ step %d] %s", global_step, pretty)

    scalars = {f"val/{k}": float(v) for k, v in metrics.items()
               if isinstance(v, (int, float))}
    _track(scalars, global_step)
    _append_jsonl({"step": global_step, "phase": "val", **metrics})


def finish() -> None:
    """End cloud tracking cleanly."""
    acc = _STATE.get("accelerator")
    if acc is not None and _STATE.get("tracking"):
        try:
            acc.end_training()
        except Exception as e:  # noqa: BLE001
            logger.warning("[trainer.logger] end_training failed: %s", e)
