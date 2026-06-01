"""
trainer/setup.py

Process-level setup for the FACET training pipeline.

Responsibilities:
  1. Lock HuggingFace to OFFLINE mode (no implicit network fetches).
  2. Seed every RNG (Python / NumPy / torch CPU+GPU / accelerate / per-rank).
  3. Wire cudnn / tf32 / matmul_precision knobs.
  4. Construct accelerate.Accelerator with the right mixed_precision dtype
     and DDP kwargs.
  5. Build the output_root directory (runs/<run_name>/...) on rank 0 only,
     then barrier so workers see the same path.
  6. Mint per-rank torch.Generator objects (cpu_gen + gpu_gen) for explicit
     RNG plumbing (FlowMatch noise sampling, DataLoader workers, etc.).

Entry point: init_env(cfg, args) -> SetupContext
"""

from __future__ import annotations

# ---- 0. HF offline lock (must run BEFORE any HF import) --------------------
# Belt-and-suspenders: also set in train.py before any heavy import. Repeating
# here in case someone imports trainer.setup directly from a notebook.
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")   # DiffSynth pattern

# 1. Imports ------------------------------------------------------------------
import argparse
import dataclasses
import datetime as dt
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

import accelerate
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed

from trainer.config import MergedConfig

logger = logging.getLogger(__name__)


# 2. Setup result --------------------------------------------------------------


@dataclass
class SetupContext:
    """
    Everything init_env produces, in one bag, passed around train.py.

    Attributes:
        accelerator   : the accelerate.Accelerator (mixed precision wrapped).
        device        : torch.device for THIS rank (alias of accelerator.device).
        dtype         : torch.dtype matching cfg.accel.precision (bf16/fp16/fp32).
        run_name      : <YYYYMMDD_HHMMSS>_s<total_steps>_<suffix>  (final at step 11).
        output_root   : runs/<run_name>/   (created on rank 0).
        ckpt_dir      : output_root/ckpts/ (created lazily).
        logs_dir      : output_root/logs/  (created lazily).
        samples_dir   : output_root/samples/ (created lazily).
        cpu_gen       : torch.Generator on CPU, seeded with cfg.train.seed.
        gpu_gen       : torch.Generator on accelerator.device, seeded with
                        seed + rank (so each rank draws DIFFERENT noise).
        seed          : the seed (echoed for downstream use).
    """
    accelerator: Accelerator
    device: torch.device
    dtype: torch.dtype
    run_name: str
    output_root: Path
    ckpt_dir: Path
    logs_dir: Path
    samples_dir: Path
    cpu_gen: torch.Generator
    gpu_gen: torch.Generator
    seed: int


# 3. Helpers ------------------------------------------------------------------


_DTYPE_MAP = {
    "no":   torch.float32,
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _resolve_mixed_precision(precision: str) -> Tuple[torch.dtype, str]:
    """Map cfg.accel.precision to (torch.dtype, accelerator-string)."""
    key = precision.lower()
    if key not in _DTYPE_MAP:
        raise ValueError(
            f"Unknown accel.precision={precision!r}; "
            f"expected one of {sorted(_DTYPE_MAP)}."
        )
    dtype = _DTYPE_MAP[key]
    acc_str = "no" if key in ("no", "fp32") else key
    return dtype, acc_str


def _apply_backend_switches(cfg_accel) -> None:
    """
    cudnn / tf32 / matmul_precision wiring.

    NOTE: cudnn.benchmark and cudnn.deterministic are mutually exclusive;
    we let benchmark win when both are true and log a warning.
    """
    bench = bool(cfg_accel.cudnn_benchmark)
    det = bool(cfg_accel.cudnn_deterministic)
    if bench and det:
        logger.warning(
            "[setup] both cudnn_benchmark and cudnn_deterministic are true; "
            "favoring benchmark for throughput."
        )
        det = False

    torch.backends.cudnn.benchmark = bench
    torch.backends.cudnn.deterministic = det
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg_accel.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(cfg_accel.allow_tf32)

    mp = cfg_accel.matmul_precision
    if mp not in ("highest", "high", "medium"):
        raise ValueError(
            f"accel.matmul_precision must be highest|high|medium, got {mp!r}"
        )
    torch.set_float32_matmul_precision(mp)


def _seed_everything(seed: int, rank: int) -> Tuple[torch.Generator, torch.Generator]:
    """
    Seed every layer of RNG and mint explicit generators.

    Layers covered:
      - Python's random / NumPy / torch CPU+GPU (set_seed)
      - accelerate's global RNG (device_specific=True spreads per-rank)
      - explicit torch.Generator on CPU (seed) and GPU (seed + rank).

    Why explicit gpu_gen with rank offset:
      `accelerate.set_seed(seed, device_specific=True)` already spreads the
      torch RNG state per rank. But for `torch.randn(shape, generator=...)`
      calls (e.g. in trainer.loss.sample_timesteps + FlowMatch noise), we want
      a stable, dedicated generator that is NOT advanced by other code paths
      (validation generation, dropout, etc.). Hence dedicated gpu_gen.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # device_specific=True -> each rank gets (seed + process_index).
    set_seed(seed, device_specific=True)

    cpu_gen = torch.Generator(device="cpu").manual_seed(int(seed))
    if torch.cuda.is_available():
        gpu_gen = torch.Generator(device="cuda").manual_seed(int(seed) + int(rank))
    else:
        gpu_gen = torch.Generator(device="cpu").manual_seed(int(seed) + int(rank))
    return cpu_gen, gpu_gen


def _build_run_name(cfg: MergedConfig, total_steps: int) -> str:
    """
    run_name = <YYYYMMDD_HHMMSS>_s<total_steps>_<suffix>

    suffix is `cfg.run.suffix` (sanitized to filesystem-safe chars). When
    empty, the trailing underscore is dropped.
    """
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = (cfg.run.suffix or "").strip()
    suffix = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in suffix)
    base = f"{stamp}_s{total_steps}"
    return f"{base}_{suffix}" if suffix else base


def _make_run_dirs(output_root: Path) -> Tuple[Path, Path, Path]:
    """
    Create the per-run subdirectory tree on rank 0.

    Sub-dirs (ckpts / logs / samples) are created eagerly so that any callsite
    can drop a file without first checking existence.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_root / "ckpts"
    logs_dir = output_root / "logs"
    samples_dir = output_root / "samples"
    for d in (ckpt_dir, logs_dir, samples_dir):
        d.mkdir(parents=True, exist_ok=True)
    return ckpt_dir, logs_dir, samples_dir


# 4. Public API ---------------------------------------------------------------


def init_env(
    cfg: MergedConfig,
    args: argparse.Namespace,
    total_steps: int,
) -> SetupContext:
    """
    Build the Accelerator, seed everything, lay out the run directory.

    Args:
        cfg          : MergedConfig from trainer.config.load_merge.
        args         : argparse.Namespace from train.py (currently unused but
                       kept in the signature for forward compatibility).
        total_steps  : output of trainer.config.estimate_total_steps.
                       Used to build run_name and ALSO recorded on the context
                       so trainer.optim can size the warmup schedule.

    Returns:
        SetupContext with accelerator, device, dtype, run paths, generators.

    NOTE on gradient checkpointing:
        The model already wraps each DiT block with torch.utils.checkpoint
        inside FACETWanModel.forward (see facet/model.py). Do NOT call
        accelerator._set_gradient_checkpointing or similar helpers - that
        double-wraps and can deadlock NCCL.
    """
    # 4.1  Mixed precision
    dtype, precision_str = _resolve_mixed_precision(cfg.accel.precision)

    # 4.2  Accelerator
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=bool(cfg.accel.find_unused_parameters),
    )
    accelerator = Accelerator(
        mixed_precision=precision_str,
        gradient_accumulation_steps=int(cfg.train.gradient_accumulation_steps),
        kwargs_handlers=[ddp_kwargs],
        # log_with is set by trainer.logger later (Phase 1.5) via init_trackers.
    )

    # 4.3  Backend switches AFTER Accelerator (so its own probing has run)
    _apply_backend_switches(cfg.accel)

    # 4.4  Seed every RNG layer
    seed = int(cfg.train.seed)
    cpu_gen, gpu_gen = _seed_everything(seed, accelerator.process_index)

    # 4.5  Override cfg.facet.device per rank.
    #      facet/config.yaml ships with device="cuda"; under DDP every rank
    #      must own its own cuda:i. accelerator.device already encodes that.
    cfg.facet.device = str(accelerator.device)

    # 4.6  run_name + output_root (rank 0 creates, all wait at barrier)
    run_name = _build_run_name(cfg, total_steps)
    output_root = Path(cfg.paths.runs_root) / run_name

    if accelerator.is_main_process:
        ckpt_dir, logs_dir, samples_dir = _make_run_dirs(output_root)
        # Also ensure the shared ckpt_root exists (lazily) so trainer.ckpt
        # can drop manifest.json + per-run *.safetensors files later.
        Path(cfg.paths.ckpt_root).mkdir(parents=True, exist_ok=True)
    else:
        # Best-effort: workers compute the same paths from cfg; rank 0
        # creates them. Wait until rank 0 is done.
        ckpt_dir = output_root / "ckpts"
        logs_dir = output_root / "logs"
        samples_dir = output_root / "samples"
    accelerator.wait_for_everyone()

    # 4.7  Boot banner on rank 0
    if accelerator.is_main_process:
        logger.info("=" * 78)
        logger.info("[setup] FACET trainer init")
        logger.info("  run_name      : %s", run_name)
        logger.info("  output_root   : %s", output_root.resolve())
        logger.info("  num_processes : %d", accelerator.num_processes)
        logger.info("  device        : %s", accelerator.device)
        logger.info("  precision     : %s (torch dtype = %s)", precision_str, dtype)
        logger.info("  seed          : %d (per-rank gpu_gen seed = %d)",
                    seed, seed + accelerator.process_index)
        logger.info("  cudnn.benchmark=%s  deterministic=%s  tf32=%s  matmul=%s",
                    torch.backends.cudnn.benchmark,
                    torch.backends.cudnn.deterministic,
                    torch.backends.cuda.matmul.allow_tf32,
                    cfg.accel.matmul_precision)
        logger.info("=" * 78)

    return SetupContext(
        accelerator=accelerator,
        device=accelerator.device,
        dtype=dtype,
        run_name=run_name,
        output_root=output_root,
        ckpt_dir=ckpt_dir,
        logs_dir=logs_dir,
        samples_dir=samples_dir,
        cpu_gen=cpu_gen,
        gpu_gen=gpu_gen,
        seed=seed,
    )
