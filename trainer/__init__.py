"""
FACET training pipeline.

Convention: train.py imports `trainer` and calls into submodules through dot
access:
    trainer.config.load_merge(args)
    trainer.setup.init_env(cfg, args, total_steps)
    trainer.optim.build_optimizer(model, cfg.train)
    trainer.loss.sample_timesteps(B, sampling=..., generator=...)

Re-exporting submodules here lets `import trainer` cover every callsite
without per-submodule imports in train.py.

Phase 1: config / setup / optim / loss are implemented.
Phase 1.5 (TODO): sum / logger / ckpt skeletons -> minimal stubs only.
Phase 3 (TODO):  valid -- full validation loop + metrics.
"""

from __future__ import annotations

from . import config       # noqa: F401
from . import setup        # noqa: F401
from . import optim        # noqa: F401
from . import loss         # noqa: F401

# Phase 1.5 placeholders (implemented as stubs in their own files):
from . import sum          # noqa: F401
from . import logger       # noqa: F401
from . import ckpt         # noqa: F401
from . import valid        # noqa: F401
