"""
Multi-dataset builder.

Reads `data/config.yaml` and produces train / val `ConcatDataset` pairs
plus per-dataset epoch quotas that `data/sampler.py::MultiSampler` consumes.

Registration pattern
--------------------
Each dataset module at `data/{name}/{name}.py` exposes its dataset class
decorated with `@register("{name}")`. `build_datasets()` imports
`data.{name}.{name}` lazily on first reference, so unused datasets do not pull
heavy deps (decord, SCHP, etc.).

Quota semantics
---------------
train_per_epoch / valid_per_epoch in data/config.yaml:
    - float in (0, 1]  -> ratio of the split size
    - int   >= 1       -> absolute sample count (clipped to split size)
The result is a fixed-per-epoch budget per sub-dataset. When quota > size,
we cycle through the dataset once and top up the remainder with replacement.
"""

from __future__ import annotations
import importlib
from typing import Any, Callable, Dict, List, Tuple, Type

from torch.utils.data import ConcatDataset, Dataset

from data.utils import load_cfg
from data.transform import TfmBundle
from data.ref_sampler import RefSampler


# ---- registry ---------------------------------------------------------------
_REGISTRY: Dict[str, Type[Dataset]] = {}


def register(name: str) -> Callable[[Type[Dataset]], Type[Dataset]]:
    """Class decorator that records the dataset class under `name`."""
    def wrap(cls: Type[Dataset]) -> Type[Dataset]:
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            # silent override is confusing -> raise
            raise RuntimeError(f"dataset '{name}' already registered by {_REGISTRY[name]!r}")
        _REGISTRY[name] = cls
        return cls
    return wrap


def _lazy_import(name: str) -> None:
    """Import `data/{name}/{name}.py` if needed to trigger the @register."""
    if name in _REGISTRY:
        return
    mod = f"data.{name}.{name}"
    importlib.import_module(mod)
    if name not in _REGISTRY:
        raise KeyError(
            f"module {mod} imported but '{name}' not registered. "
            f"Did you add `@register(\"{name}\")` to the dataset class?"
        )


# ---- quota resolution --------------------------------------------------------
def _resolve_quota(v: Any, total: int) -> int:
    """float in (0, 1] -> ratio * total; int >= 1 -> min(v, total)."""
    if isinstance(v, bool):
        raise TypeError(f"per_epoch must be int or float, got bool: {v!r}")
    if isinstance(v, float):
        if v < 0:
            raise ValueError(f"per_epoch ratio must be >= 0, got {v}")
        return max(0, int(round(v * total)))
    if isinstance(v, int):
        if v < 0:
            raise ValueError(f"per_epoch count must be >= 0, got {v}")
        return min(v, total)
    raise TypeError(f"per_epoch must be int or float, got {type(v).__name__}: {v!r}")


# ---- public API --------------------------------------------------------------
def build_datasets(
    cfg_path: str = "data/config.yaml",
) -> Tuple[ConcatDataset, ConcatDataset, List[int], List[int]]:
    """Instantiate every registered dataset listed in `cfg_path`.

    Returns:
        train_concat : ConcatDataset of train splits, in the order of cfg.datasets
        val_concat   : ConcatDataset of val splits,   same order
        train_quotas : List[int], per-epoch absolute quota per sub-dataset
        val_quotas   : List[int], same for val
    """
    cfg = load_cfg(cfg_path)  # top-level cfg (data/config.yaml)

    # Build them ONCE and share across every sub-dataset.
    tfm = TfmBundle.from_cfg(shared=cfg)
    refs = RefSampler.from_cfg(shared=cfg)

    train_dss: List[Dataset] = []
    val_dss: List[Dataset] = []
    train_quotas: List[int] = []
    val_quotas: List[int] = []

    for entry in cfg.datasets:
        name = entry["name"]
        sub_cfg_path = entry.get("config", f"data/{name}/config.yaml")
        sub_cfg = load_cfg(sub_cfg_path)

        _lazy_import(name)
        cls = _REGISTRY[name]

        ds_train = cls(sub_cfg, tfm, refs, split="train")
        ds_val   = cls(sub_cfg, tfm, refs, split="val")

        train_dss.append(ds_train)
        val_dss.append(ds_val)

        train_quotas.append(_resolve_quota(entry.get("train_per_epoch", 1.0), len(ds_train)))
        val_quotas.append(  _resolve_quota(entry.get("valid_per_epoch", 1.0), len(ds_val)))

    return (
        ConcatDataset(train_dss),
        ConcatDataset(val_dss),
        train_quotas,
        val_quotas,
    )
