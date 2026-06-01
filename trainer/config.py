"""
trainer/config.py

Load + merge + snapshot the FACET training config.

Responsibilities:
  - Parse train.yaml into typed dataclasses (TrainConfig, AccelConfig, ...).
  - Build a FACETConfig from `paths.facet_config` and apply `facet_overrides`.
  - Land train.yaml's `training:` block onto cfg.facet.training (the only block
    that the model itself reads at training time; see trainer.txt L194).
  - Expose a single MergedConfig object with .facet / .train / .accel / .run
    / .log / .validate / .paths sub-configs.
  - .flat()         : dict of "ns.key" -> scalar, suitable for mlflow logging.
  - .dump_snapshot(): write merged yaml into runs/<run>/config_snapshot.yaml.

Authoritative for: optimizer / scheduler / loop / cached / cfg-training flags.
NOT authoritative for: model dims (those still live in facet/config.yaml).
"""

from __future__ import annotations

# 1. Imports ------------------------------------------------------------------
import argparse
import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from facet.config import FACETConfig


# 2. Sub-config dataclasses ---------------------------------------------------


@dataclass
class PathsConfig:
    """Filesystem roots referenced across the trainer."""
    facet_config: str = "facet/config.yaml"
    data_config: str = "data/config.yaml"
    weight_dir: str = "weights/WAN2.2"
    runs_root: str = "runs"
    ckpt_root: str = "ckpt"


@dataclass
class AccelConfig:
    """Compute-stack switches consumed by trainer/setup.py."""
    precision: str = "bf16"                      # "no" | "fp16" | "bf16"
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False
    allow_tf32: bool = True
    matmul_precision: str = "high"               # "highest" | "high" | "medium"
    find_unused_parameters: bool = False
    launcher: str = "auto"                       # "auto" | "python" | "accelerate"
    num_gpus: int = 1


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 1.0e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1.0e-2
    eps: float = 1.0e-8


@dataclass
class SchedulerConfig:
    name: str = "constant_with_warmup"           # "constant" | "constant_with_warmup"
    warmup_steps: int = 200


@dataclass
class TrainConfig:
    """Loop-level training knobs. Step counts here are OPTIMIZER STEPS."""
    seed: int = 42
    epochs: int = 5
    max_steps: Optional[int] = None              # null in yaml -> None
    batch_size: int = 1
    num_workers: int = 4
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    log_every_steps: int = 10
    val_every_steps: int = 500
    save_every_steps: int = 1000


@dataclass
class RunConfig:
    """Output_root naming. Final run_name resolved by trainer.setup."""
    suffix: str = "facet"
    resume_from: Optional[str] = None


@dataclass
class LogConfig:
    """Cloud + console logging (consumed by trainer.logger in Phase 1.5)."""
    backend: str = "mlflow"                      # "mlflow" | "tensorboard" | "none"
    project_name: str = "facet"
    cloud_run_name: Optional[str] = None         # None -> use run_name
    log_every_steps: int = 10
    log_grad_norm: bool = True
    log_timestep_hist: bool = True


@dataclass
class ValidateConfig:
    """Validation pass knobs (consumed by trainer.valid in Phase 3)."""
    num_samples: int = 10
    num_inference_steps: int = 25
    cfg_scale: float = 5.0
    metrics: List[str] = field(default_factory=lambda: ["psnr", "ssim", "lpips", "clipsim"])
    primary_metric: str = "lpips"
    primary_metric_direction: str = "min"        # "min" | "max"
    topk: int = 3


# 3. Top-level merged config --------------------------------------------------


@dataclass
class MergedConfig:
    """
    Bundle of everything `train.py` consumes.

    Layout:
      cfg.facet      -> FACETConfig            (model side, from facet/config.yaml + overrides)
      cfg.train      -> TrainConfig            (loop knobs)
      cfg.accel      -> AccelConfig            (compute stack)
      cfg.run        -> RunConfig              (run_name suffix, resume)
      cfg.log        -> LogConfig              (cloud / console)
      cfg.validate   -> ValidateConfig         (in-loop val)
      cfg.paths      -> PathsConfig            (roots)
      cfg._raw       -> dict                   (verbatim train.yaml; for snapshot)
    """
    facet: FACETConfig
    train: TrainConfig
    accel: AccelConfig
    run: RunConfig
    log: LogConfig
    validate: ValidateConfig
    paths: PathsConfig
    _raw: Dict[str, Any] = field(default_factory=dict)

    # 3.1  Flat view for mlflow / tensorboard.
    def flat(self) -> Dict[str, Any]:
        """
        Return a flat dict of scalar-ish values keyed by "ns.key".

        Skips nested dataclasses, lists, dicts (mlflow doesn't render those well).
        Used by trainer.logger to upload config to the run tracker.
        """
        out: Dict[str, Any] = {}

        def walk(prefix: str, obj: Any) -> None:
            if dataclasses.is_dataclass(obj):
                for f in dataclasses.fields(obj):
                    walk(f"{prefix}.{f.name}" if prefix else f.name, getattr(obj, f.name))
                return
            if isinstance(obj, (str, int, float, bool)) or obj is None:
                out[prefix] = obj
                return
            if isinstance(obj, (list, tuple)):
                out[prefix] = ",".join(map(str, obj))
                return
            # other types ignored on purpose; snapshot yaml has the full picture.

        for name in ("facet", "train", "accel", "run", "log", "validate", "paths"):
            walk(name, getattr(self, name))
        return out

    # 3.2  Snapshot dump.
    def dump_snapshot(self, out_path: str | Path) -> None:
        """
        Write a single yaml capturing the merged effective config.

        We dump as plain dict (not dataclass repr) so the file is human-readable
        and can be re-loaded with safe_load.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        def to_plain(obj: Any) -> Any:
            if dataclasses.is_dataclass(obj):
                return {f.name: to_plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
            if isinstance(obj, dict):
                return {k: to_plain(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [to_plain(v) for v in obj]
            return obj

        payload = {
            "facet":    to_plain(self.facet),
            "train":    to_plain(self.train),
            "accel":    to_plain(self.accel),
            "run":      to_plain(self.run),
            "log":      to_plain(self.log),
            "validate": to_plain(self.validate),
            "paths":    to_plain(self.paths),
            "_train_yaml_raw": self._raw,    # round-trip the user's literal yaml
        }
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


# 4. Helpers ------------------------------------------------------------------


def _override_dataclass(dc: Any, overrides: Dict[str, Any]) -> None:
    """
    In-place overlay a dict of overrides onto a dataclass.

    - Unknown keys are warned and ignored (won't silently typo).
    - Nested dicts recurse into nested dataclasses.
    - List-to-tuple coercion is applied for fields typed as tuple (betas, ...).
    """
    if not overrides:
        return
    field_names = {f.name for f in dataclasses.fields(dc)}
    for k, v in overrides.items():
        if k not in field_names:
            print(f"[trainer.config] Unknown key {type(dc).__name__}.{k} in train.yaml, ignored.")
            continue
        cur = getattr(dc, k)
        if dataclasses.is_dataclass(cur) and isinstance(v, dict):
            _override_dataclass(cur, v)
        else:
            # Coerce list to tuple if field is declared as tuple.
            if isinstance(cur, tuple) and isinstance(v, list):
                v = tuple(v)
            setattr(dc, k, v)


def _override_facet(facet_cfg: FACETConfig, overrides: Dict[str, Any]) -> None:
    """
    Apply train.yaml's `facet_overrides:` block onto a FACETConfig.

    Reuses FACETConfig's own _SUB_CONFIGS / _FLAT_FIELDS schema so we never
    drift from facet/config.py's notion of which keys are scalars vs blocks.
    """
    if not overrides:
        return
    for k, v in overrides.items():
        if k in FACETConfig._FLAT_FIELDS:
            setattr(facet_cfg, k, v)
            continue
        if k in FACETConfig._SUB_CONFIGS:
            attr_name, _ = FACETConfig._SUB_CONFIGS[k]
            sub = getattr(facet_cfg, attr_name)
            for sk, sv in (v or {}).items():
                if not hasattr(sub, sk):
                    print(f"[trainer.config] Unknown key facet_overrides.{k}.{sk}, ignored.")
                    continue
                if sk == "patch_size" and isinstance(sv, list):
                    sv = tuple(sv)
                if sk == "target_modules" and isinstance(sv, list):
                    sv = tuple(sv)
                setattr(sub, sk, sv)
            continue
        print(f"[trainer.config] Unknown top-level key facet_overrides.{k}, ignored.")


def _apply_training_block(facet_cfg: FACETConfig, training_block: Dict[str, Any]) -> None:
    """
    Land train.yaml's `training:` block onto cfg.facet.training.

    This is the ONE place where train.yaml feeds into the model-side config.
    Done explicitly (not via facet_overrides) because the user requested this
    block to live in train.yaml, NOT facet/config.yaml (see trainer.txt L194).
    """
    if not training_block:
        return
    # FACETTrainingConfig in facet/config.py is the schema.
    sub = facet_cfg.training
    for sk, sv in training_block.items():
        if not hasattr(sub, sk):
            print(f"[trainer.config] Unknown key training.{sk}, ignored.")
            continue
        setattr(sub, sk, sv)


# 5. Public API ---------------------------------------------------------------


def load_merge(args: argparse.Namespace) -> MergedConfig:
    """
    Load train.yaml + facet/config.yaml and merge them.

    args.train_yaml: path to train.yaml (set by launch_train.py / parse_args).

    Merge order (later wins):
      1. FACETConfig defaults
      2. facet/config.yaml
      3. train.yaml: facet_overrides:
      4. train.yaml: training:                   -> cfg.facet.training
    """
    train_yaml_path = Path(getattr(args, "train_yaml", "train.yaml")).resolve()
    if not train_yaml_path.is_file():
        raise FileNotFoundError(f"train.yaml not found: {train_yaml_path}")

    with open(train_yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # 5.1  paths
    paths = PathsConfig()
    _override_dataclass(paths, raw.get("paths", {}))

    # 5.2  facet (model) config
    facet_cfg = FACETConfig.from_yaml(paths.facet_config)
    _override_facet(facet_cfg, raw.get("facet_overrides", {}))
    _apply_training_block(facet_cfg, raw.get("training", {}))

    # 5.3  trainer-side blocks
    train = TrainConfig()
    _override_dataclass(train, raw.get("train", {}))

    accel = AccelConfig()
    _override_dataclass(accel, raw.get("accel", {}))

    run = RunConfig()
    _override_dataclass(run, raw.get("run", {}))

    log_cfg = LogConfig()
    _override_dataclass(log_cfg, raw.get("log", {}))

    validate_cfg = ValidateConfig()
    _override_dataclass(validate_cfg, raw.get("validate", {}))

    return MergedConfig(
        facet=facet_cfg,
        train=train,
        accel=accel,
        run=run,
        log=log_cfg,
        validate=validate_cfg,
        paths=paths,
        _raw=copy.deepcopy(raw),
    )


def estimate_total_steps(
    train_cfg: TrainConfig,
    len_train_loader: int,
) -> int:
    """
    Compute the total optimizer-step count for this run.

    Priority:
      - cfg.train.max_steps if set    (cap)
      - else epochs * len(loader) / grad_accum_steps  (ceil)

    Used by trainer.setup to:
      a) build run_name = "...s<total_steps>..."
      b) build lr_scheduler (so warmup ratio is well-defined)
    """
    if train_cfg.max_steps is not None and train_cfg.max_steps > 0:
        return int(train_cfg.max_steps)
    micro_per_epoch = max(1, len_train_loader)
    grad_accum = max(1, int(train_cfg.gradient_accumulation_steps))
    return -((-(train_cfg.epochs * micro_per_epoch)) // grad_accum)   # ceil div
