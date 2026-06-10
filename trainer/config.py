"""
trainer/config.py
FACET training pipeline Step 1.

Load + record the FACET training config.

Responsibilities:
  - Parse train.yaml into typed dataclasses (TrainingConfig, TrainConfig,
    AccelConfig, RunConfig, LogConfig, ValidateConfig, PathsConfig).
  - Load the model config separately via FACETConfig.from_yaml(paths.facet_config).
  - Resolve every entry in PathsConfig to an ABSOLUTE path rooted at the project
    directory, so downstream code never has to re-resolve cwd-relative paths.
  - Expose a single MergedConfig object with
    .facet / .training / .train / .accel / .run / .log / .validate / .paths.
  - .flat()         : dict of "ns.key" -> scalar, suitable for mlflow logging.
  - .dump_snapshot(): write the effective config to runs/<run>/config_snapshot.yaml.

facet/config.yaml and train.yaml are managed INDEPENDENTLY. There is no cross-yaml
override anymore: model knobs are edited in facet/config.yaml, training knobs in
train.yaml. load_merge only LOADS + RECORDS; it does not reconcile the two.
NOTE: when a new config block is added, update this docstring + MergedConfig.

Difference between 'training' and 'train':
- training is for entire strategy, while train is for controlling specific loop-level hyperparameters.
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


# Project root = parent of the `trainer/` package (i.e. the dir that holds
# train.py). All relative paths in train.yaml are resolved against this so the
# trainer behaves identically regardless of the shell's cwd.
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


def _to_abs(p: str | Path) -> str:
    """Resolve a (possibly relative) path against the project root -> absolute str."""
    p = Path(p)
    if not p.is_absolute():
        p = (_PROJECT_ROOT / p).resolve()
    return str(p)


# 2. Sub-config dataclasses ---------------------------------------------------


@dataclass
class PathsConfig:
    """
    Filesystem roots referenced across the trainer.

    All fields are rewritten to ABSOLUTE paths inside load_merge (rooted at the
    project directory), so callers can use them directly.
    """
    facet_config: str = "facet/config.yaml"
    data_config: str = "data/config.yaml"
    weight_dir: str = "weights/WAN2.2"
    run_root: str = "runs"
    ckpt_root: str = "ckpts"


@dataclass
class AccelConfig:
    """Compute-stack switches consumed by trainer/setup.py + launch_train.py."""
    precision: str = "bf16"                      # "no" | "fp16" | "bf16"
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False
    allow_tf32: bool = True
    matmul_precision: str = "high"               # "highest" | "high" | "medium"
    find_unused_parameters: bool = False
    launcher: str = "auto"                       # "auto" | "python" | "accelerate"
    # (sets CUDA_VISIBLE_DEVICES in launch_train.py). Single id -> python launcher; multiple -> accelerate.
    gpu_ids: List[int] = field(default_factory=lambda: [0])

    @property
    def num_gpus(self) -> int:
        """Inferred from gpu_ids (launch_train.py picks the launcher off this)."""
        return len(self.gpu_ids)


@dataclass
class TrainingConfig:
    """
    FlowMatch / objective knobs.

    Combination presets (see trainer/loss.py header for full rationale):
      A (SD3)       : timestep_sampling=logit_normal,      sigma_shift=1.0, loss_weighting=none
      B (DiffSynth) : timestep_sampling=uniform(discrete), sigma_shift=5.0, loss_weighting=bsmnt   <- FACET default
      C             : timestep_sampling=logit_normal,      sigma_shift=5.0, loss_weighting=none    <- stage2 ablation

    Fields:
        prediction_type    : FlowMatch target ("velocity" | "noise").
        timestep_sampling  : timestep density ("uniform" | "logit_normal").
        sigma_shift        : flow-matching shift; 1.0 = linear, 5.0 = shift (WAN/DiffSynth default).
        loss_weighting     : per-timestep loss weight ("none" | "bsmnt").
        loss_type          : "mse" only (extend in trainer/loss.py if needed).
        cfg_training       : drop text/ref to train unconditional branches.
        text_dropout_prob  : consulted only when cfg_training=True.
        ref_dropout_prob   : consulted only when cfg_training=True.
        cached_t5          : True -> batch["t5_emb"] used directly.
        cached_tgt_latent  : True -> batch["tgt_latent"] used directly.
                             (src_video / ref_img are ALWAYS encoded online
                             because of per-epoch mask perturbation.)
    """
    prediction_type: str = "velocity"            # "noise" | "velocity"
    timestep_sampling: str = "uniform"           # "uniform" | "logit_normal"
    sigma_shift: float = 5.0                     # 1.0 = linear, 5.0 = shift
    loss_weighting: str = "bsmnt"                # "none" | "bsmnt"
    loss_type: str = "mse"
    cfg_training: bool = False
    text_dropout_prob: float = 0.0
    ref_dropout_prob: float = 0.0
    cached_t5: bool = True
    cached_tgt_latent: bool = True


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
    warmup_steps: int = 1000


@dataclass
class TrainConfig:
    """Loop-level training knobs. Step counts here are OPTIMIZER STEPS."""
    seed: int = 42
    epochs: int = 5
    max_steps: int = 20000    #TODO: 后期考虑调整此参数
    batch_size: int = 1
    num_workers: int = 4
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    log_every_steps: int = 10
    val_every_steps: int = 500
    save_every_steps: int = 1000
    start_eval_steps: int = 10000


@dataclass
class RunConfig:
    """Output_root naming. Final run_name resolved by trainer.setup."""
    suffix: str = "facet"
    resume_from: Optional[str] = None


@dataclass
class LogConfig:
    """Cloud + console logging."""
    backend: str = "mlflow"                      # "mlflow" | "tensorboard" | "none"
    project_name: str = "facet"
    cloud_run_name: Optional[str] = None         # None -> use run_name
    log_every_steps: int = 10
    log_grad_norm: bool = True
    log_timestep_hist: bool = True


@dataclass
class ValidateConfig:
    """
    Validation pass knobs.

    primary_metric drives top-K selection.
    """
    num_samples: int = 10
    num_inference_steps: int = 25
    cfg_scale: float = 5.0
    metrics: List[str] = field(default_factory=lambda: ["psnr", "ssim", "lpips", "clipsim"])
    primary_metric: str = "lpips"
    topk: int = 3


# 3. Top-level merged config --------------------------------------------------


@dataclass
class MergedConfig:
    """
    Bundle of everything `train.py` consumes.

    Layout:
      cfg.facet      -> FACETConfig            (model; from facet/config.yaml)
      cfg.training   -> TrainingConfig         (objective / FlowMatch knobs)
      cfg.train      -> TrainConfig            (loop)
      cfg.accel      -> AccelConfig            (compute)
      cfg.run        -> RunConfig              (name)
      cfg.log        -> LogConfig              (cloud)
      cfg.validate   -> ValidateConfig         (valid)
      cfg.paths      -> PathsConfig            (absolute paths)
      cfg._raw       -> dict                   (verbatim train.yaml; for snapshot)
    """
    facet: FACETConfig
    training: TrainingConfig
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

        for name in ("facet", "training", "train", "accel", "run", "log", "validate", "paths"):
            walk(name, getattr(self, name))
        return out

    # 3.2  Snapshot dump.
    def dump_snapshot(self, out_path: str | Path) -> None:
        """Write a single yaml capturing the effective (loaded) config."""
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
            "training": to_plain(self.training),
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


def _load_into_dataclass(dc: Any, block: Dict[str, Any]) -> None:
    """
    Populate a dataclass IN PLACE from a yaml block.

    - Unknown keys are warned and ignored (so typos surface instead of silently
      vanishing).
    - Nested dicts recurse into nested dataclasses (e.g. train.optimizer).
    - List values are coerced to tuple when the target field is typed as tuple
      (e.g. optimizer.betas).
    """
    if not block:
        return
    field_types = {f.name: f.type for f in dataclasses.fields(dc)}
    for k, v in block.items():
        if k not in field_types:
            print(f"[trainer.config] Unknown key {type(dc).__name__}.{k} in train.yaml, ignored.")
            continue
        cur = getattr(dc, k)
        if dataclasses.is_dataclass(cur) and isinstance(v, dict):
            _load_into_dataclass(cur, v)
        else:
            # Coerce list to tuple if field is declared as tuple.
            if isinstance(cur, tuple) and isinstance(v, list):
                v = tuple(v)
            setattr(dc, k, v)


# 5. Public API ---------------------------------------------------------------


def load_merge(args: argparse.Namespace) -> MergedConfig:
    """
    Load train.yaml + facet/config.yaml into a single MergedConfig.
    Paths are resolved to absolute.

    args.train_yaml: path to train.yaml (set by launch_train.py / parse_args).
    """
    train_yaml_path = Path(getattr(args, "train_yaml", "train.yaml")).resolve()
    if not train_yaml_path.is_file():
        raise FileNotFoundError(f"train.yaml not found: {train_yaml_path}")

    with open(train_yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # 5.1  paths  (load, then resolve every entry to an absolute path)
    paths = PathsConfig()
    _load_into_dataclass(paths, raw.get("paths", {}))
    for fld in dataclasses.fields(paths):
        setattr(paths, fld.name, _to_abs(getattr(paths, fld.name)))

    # 5.2  facet (model) config -- loaded independently from facet/config.yaml
    facet_cfg = FACETConfig.from_yaml(paths.facet_config)

    # 5.3  trainer-side blocks
    training = TrainingConfig()
    _load_into_dataclass(training, raw.get("training", {}))

    train = TrainConfig()
    _load_into_dataclass(train, raw.get("train", {}))

    accel = AccelConfig()
    _load_into_dataclass(accel, raw.get("accel", {}))

    run = RunConfig()
    _load_into_dataclass(run, raw.get("run", {}))

    log_cfg = LogConfig()
    _load_into_dataclass(log_cfg, raw.get("log", {}))

    validate_cfg = ValidateConfig()
    _load_into_dataclass(validate_cfg, raw.get("validate", {}))

    return MergedConfig(
        facet=facet_cfg,
        training=training,
        train=train,
        accel=accel,
        run=run,
        log=log_cfg,
        validate=validate_cfg,
        paths=paths,
        _raw=copy.deepcopy(raw),
    )
