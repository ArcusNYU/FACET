"""
FACET training pipeline.

train.py imports `trainer` and calls into submodules through dot access:
    trainer.config.load_merge(args)
    trainer.setup.init_env(cfg, args, total_steps)
    trainer.optim.build_optimizer(model, cfg.train)
    trainer.optim.model_stats(model, output_root)
    trainer.loss.FlowMatch(cfg.training)
    trainer.logger.setup(...) / log_step(...) / log_metrics(...)
    trainer.ckpt.CheckpointManager(...)
    trainer.valid.run(...)

Re-exporting submodules here lets `import trainer` cover every callsite.
"""

from __future__ import annotations

from . import config       # noqa: F401
from . import setup        # noqa: F401
from . import optim        # noqa: F401
from . import loader       # noqa: F401
from . import loss         # noqa: F401
from . import logger       # noqa: F401
from . import ckpt         # noqa: F401
from . import valid        # noqa: F401
