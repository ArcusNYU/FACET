"""
Dataset-side utilities: yaml loading with dot-access (avoid dragging omegaconf as a hard dep).

Used by data/{dataset_name}/pipeline/{filters, parse, score, prepare}.py and later by train.py.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any

import yaml


class DotDict(dict):
    """dict subclass that exposes keys as attributes recursively.
       Mutation goes through __setitem__ so the underlying dict stays canonical.
    """

    def __getattr__(self, key: str) -> Any:
        try:
            v = self[key]
        except KeyError:
            raise AttributeError(key)
        if isinstance(v, dict) and not isinstance(v, DotDict):
            v = DotDict(v)
            self[key] = v
        if isinstance(v, list):
            v = [DotDict(x) if isinstance(x, dict) and not isinstance(x, DotDict) else x for x in v]
            self[key] = v
        return v

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def load_cfg(path: str | Path) -> DotDict:
    """Load a yaml file as a DotDict (recursive)."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        return DotDict()
    return DotDict(raw)
