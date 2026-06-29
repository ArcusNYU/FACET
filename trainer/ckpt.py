"""
trainer/ckpt.py
FACET training pipeline Step (checkpointing).

A stateful CheckpointManager that the training loop owns. It maintains the
top-K bookkeeping in-instance (the "global variable" lives on the manager held
by train.py), so when a (K+1)-th better checkpoint arrives the worst one is
evicted from disk and only K survive.

Output locations:
  runs/<run>/ckpts/<run_name>_last.safetensors : always-overwritten latest +
       last.json sidecar {global_step, epoch}  -> RESUME source (per-run only).
  runs/<run>/ckpts/step_*.safetensors          : per-run top-K snapshots.
  <ckpt_root>/<run_name>_best.safetensors      : the single best, exported for test.py.
  <ckpt_root>/manifest.json                    : best/topk pointers + source run dir.

Saving uses FACETWanModel.save_lora (facet/model.py), which filters lora_down /
lora_up out of state_dict. DDP-safe: only the main process writes, and the model
is unwrapped via accelerator.unwrap_model before save_lora.

Resume: find_resume(run_root, run_name) locates the _last weights + sidecar so
train.py can reload LoRA + restore (global_step, epoch). Optimizer state is NOT
persisted (lr is constant post-warmup; Adam moments re-warm quickly for LoRA).
"""

# further possible TODO: 按mask-ratio分bucket 每个bucket内先平均 再对bucket做macro-average

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Metric-name -> optimization direction. trainer.config.ValidateConfig stores
# only `primary_metric`; the min/max direction is inferred here.
_LOWER = {"lpips", "fvd", "loss", "mse", "l1"}
_HIGHER = {"psnr", "ssim", "clipsim", "clip", "clipsim_var", "clip_sim"}


def _direction(metric: str) -> str:
    """Return 'min' or 'max' for a metric name (defaults to 'min' if unknown)."""
    m = metric.lower()
    if m.endswith("_mask"):
        m = m[: -len("_mask")]
    if m in _HIGHER:
        return "max"
    if m in _LOWER:
        return "min"
    logger.warning("[trainer.ckpt] unknown primary_metric=%r; assuming lower-is-better.", metric)
    return "min"


def find_resume(run_root, run_name: str) -> Optional[Dict[str, Any]]:
    """
    Locate a resumable checkpoint for `run_name`.

    Searches runs/<run_name>/ckpts/ for <run_name>_last.safetensors and 
    last.json sidecar. Returns {"path", "global_step", "epoch"} or
    None when the weights are absent (-> caller starts fresh).
    """
    ckpt_dir = Path(run_root) / run_name / "ckpts"
    weights = ckpt_dir / f"{run_name}_last.safetensors"
    sidecar = ckpt_dir / f"last.json"
    if not weights.exists():
        logger.warning("[trainer.ckpt] resume: %s not found; starting fresh.", weights)
        return None
    step, epoch = 0, 0
    if sidecar.exists():
        try:
            d = json.loads(sidecar.read_text(encoding="utf-8"))
            step = int(d.get("global_step", 0))
            epoch = int(d.get("epoch", 0))
        except (OSError, json.JSONDecodeError, ValueError):
            logger.warning("[trainer.ckpt] resume: cannot parse %s; step/epoch=0.", sidecar)
    else:
        logger.warning("[trainer.ckpt] resume: sidecar %s missing; step/epoch=0.", sidecar)
    return {"path": str(weights), "global_step": step, "epoch": epoch}


# FIXME: 目前已经在 train.yaml中设置了针对各项指标的权重 需要传入 checkpointmanager管理器中用于计算score
# 由于LPIPS/PSNR/SSIM的范围分布不同 在计算score之前还需要对这些指标进行归一化
class CheckpointManager:
    """
    Owns top-K state for one run.

    Args:
        accelerator    : accelerate.Accelerator (for unwrap + rank guard + barrier).
        run_name       : <MMDD>_s<steps>_<suffix>.
        output_root    : runs/<run_name>/   (per-run snapshots go to ckpts/).
        ckpt_root      : <root>/ckpts/      (exported best/last + manifest).
        topk           : how many best snapshots to keep.
        primary_metric : metric driving the ranking (e.g. "lpips").
    """
    # NOTE: LPIPS is more close to human perception than PSNR/SSIM
    # while PSNR and SSIM can constrain the structure and color consistency respectively

    def __init__(
        self,
        accelerator,
        run_name: str,
        output_root: Path,
        ckpt_root: Path,
        topk: int = 3,
        primary_metric: str = "lpips",
    ):
        self.accelerator = accelerator
        self.run_name = run_name
        self.output_root = Path(output_root)
        self.run_ckpt_dir = self.output_root / "ckpts"
        self.ckpt_root = Path(ckpt_root)
        self.topk = int(topk)
        self.primary_metric = primary_metric
        self.direction = _direction(primary_metric)

        # top-K registry: list of dicts {score, step, path, metrics}, kept sorted
        # best-first. Held in-instance (this is the "global" the loop maintains).
        self._registry: List[Dict[str, Any]] = []

    # ---- helpers ------------------------------------------------------------
    def _compare(self, a: float, b: float) -> bool:
        """Is score a strictly better than score b under self.direction?"""
        return a < b if self.direction == "min" else a > b

    def best(self) -> Optional[Dict[str, Any]]:
        """
        Return the current best registry entry {score, step, path, metrics}, or
        None if no checkpoint has been ranked yet.

        NOTE: Only the main process maintains the registry
        """
        return self._registry[0] if self._registry else None

    def _unwrap(self, model):
        return self.accelerator.unwrap_model(model)

    # ---- last (rolling) -----------------------------------------------------
    def save_last(self, model, global_step: int, epoch: int = 0) -> Optional[Path]:
        """
        Overwrite runs/<run>/ckpts/<run_name>_last.safetensors with current LoRA,
        plus a last.json sidecar {global_step, epoch} for resume.

        Always called on the cadence; cheap rolling snapshot for inspection /
        resume. Stays in the per-run ckpts/ dir (never exported to ckpt_root).
        Main process only.
        """
        self.accelerator.wait_for_everyone()
        out_path: Optional[Path] = None
        if self.accelerator.is_main_process:
            self.run_ckpt_dir.mkdir(parents=True, exist_ok=True)
            out_path = self.run_ckpt_dir / f"{self.run_name}_last.safetensors"
            self._unwrap(model).save_lora(str(out_path))
            # overwrite (not append)
            sidecar = self.run_ckpt_dir / "last.json"
            sidecar.write_text(
                json.dumps(
                    {"global_step": int(global_step), "epoch": int(epoch),
                     "run_name": self.run_name,
                     "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")},
                    ensure_ascii=False, indent=2,
                ),
                encoding="utf-8",
            )
            logger.info("[trainer.ckpt] last -> %s (step %d, epoch %d)", out_path, global_step, epoch)
        self.accelerator.wait_for_everyone()
        return out_path

    # ---- top-K (metric-driven) ---------------------------------------------
    def save_topk(self, model, metrics: Dict[str, Any], global_step: int) -> None:
        """
        Consider the current weights for the top-K set, keyed by primary_metric.
        Main process only; other ranks just barrier.
        """
        self.accelerator.wait_for_everyone()
        if not self.accelerator.is_main_process:
            self.accelerator.wait_for_everyone()
            return

        # FIXME: score的计算方式调整为 以 lpips_mask 作为primary_metric 并且赋予最大的权重值 
        # 并且score计算的时候采用 robust Z-score 并且需要扩大测试样本的数量maybe
        # NOTE: 暂时不针对score的计算设置EMA 因为validation阶段的样本选择都通过了hash进行锁定
        score = None if not metrics else metrics.get(self.primary_metric)
        if score is None:
            logger.info(
                "[trainer.ckpt] save_topk: no '%s' in metrics at step %d; skipping top-K.",
                self.primary_metric, global_step,
            )
            self.accelerator.wait_for_everyone()
            return
        score = float(score)

        # Decide whether this checkpoint enters the top-K.
        full = len(self._registry) >= self.topk
        worst = self._registry[-1] if self._registry else None
        enters = (not full) or (worst is not None and self._compare(score, worst["score"]))
        # enters为True的条件: 1. 当前topk未满时直接计入 2. 当前topk已满时 且当前score比worst的score更优时计入
        if not enters:
            logger.info(
                "[trainer.ckpt] step %d %s=%.5f not in top-%d; skip.",
                global_step, self.primary_metric, score, self.topk,
            )
            self.accelerator.wait_for_everyone()
            return

        # Save the per-run snapshot (: current best checkpoint).
        self.run_ckpt_dir.mkdir(parents=True, exist_ok=True)
        snap = self.run_ckpt_dir / f"step_{global_step:07d}_{self.primary_metric}{score:.4f}.safetensors"
        self._unwrap(model).save_lora(str(snap))

        self._registry.append({
            "score": score, "step": global_step, "path": str(snap), "metrics": dict(metrics),
        })
        self._registry.sort(key=lambda e: e["score"], reverse=(self.direction == "max"))

        # Evict the worst checkpoint (delete files).
        while len(self._registry) > self.topk:
            dropped = self._registry.pop()
            try:
                Path(dropped["path"]).unlink(missing_ok=True)
                logger.info("[trainer.ckpt] evicted %s (%s=%.5f)",
                            dropped["path"], self.primary_metric, dropped["score"])
            except OSError as e:
                logger.warning("[trainer.ckpt] could not delete %s: %s", dropped["path"], e)

        # If the current best changed, export it + refresh manifest.
        best = self._registry[0]
        if best["step"] == global_step:
            self.ckpt_root.mkdir(parents=True, exist_ok=True)
            best_export = self.ckpt_root / f"{self.run_name}_best.safetensors"
            shutil.copyfile(snap, best_export) # overwrite the previous best checkpoint directly
            logger.info("[trainer.ckpt] new best %s=%.5f -> %s",
                        self.primary_metric, score, best_export)

        self._write_manifest() # only used by main process
        self.accelerator.wait_for_everyone()

    # ---- manifest -----------------------------------------------------------
    def _write_manifest(self) -> None:
        """Write <ckpt_root>/manifest.json describing best + top-K for this run."""
        best = self._registry[0] if self._registry else None
        manifest_path = self.ckpt_root / "manifest.json"
        # Preserve other runs' entries if the manifest already exists.
        data: Dict[str, Any] = {}
        if manifest_path.exists():
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}

        data[self.run_name] = {
            "primary_metric": self.primary_metric,
            "direction": self.direction,
            "source_run_dir": str(self.output_root.resolve()),
            "best_export": str((self.ckpt_root / f"{self.run_name}_best.safetensors").resolve()),
            "last_ckpt": str((self.run_ckpt_dir / f"{self.run_name}_last.safetensors").resolve()),
            "best": best,
            "topk": self._registry,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        } # update the current run's entry in manifest.json
        manifest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
