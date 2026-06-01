"""
train.py

FACET / WAN2.2-TI2V-5B + OmniControl LoRA training entry point.

Layout (mirrors trainer.txt L88-185):
    0.  argparse + HF offline lock                        (Phase 1)
    1.  trainer.config.load_merge                          (Phase 1)
    2.  trainer.setup.init_env                             (Phase 1)
    3.  FACETWanModel construction                         (Phase 1)
    4.  trainer.sum.trainable_stats                        (Phase 1 stub)
    5.  loader.build_loaders                               (Phase 1)
    6.  trainer.optim.build_optimizer                      (Phase 1)
    7.  trainer.optim.build_lr_scheduler                   (Phase 1)
    8.  accelerator.prepare                                (Phase 1)
    9.  trainer.ckpt.maybe_resume                          (Phase 2+ stub)
    10. trainer.logger.setup                               (Phase 1 stub)
    11. global_step / start_epoch init                     (Phase 1)
    12. for epoch in range(...):                           (Phase 2: actual loop)
            ... batch.prepare / loss / backward / step
            ... validate / log_metrics / save_topk
    13. trainer.logger.finish()                            (Phase 1 stub)

In Phase 1 the script ends right before step 12. The loop is sketched with
`pass` so the smoke run (single GPU, 1 step planned) exercises every setup
path. Phase 2 will fill in steps 12-13.
"""

from __future__ import annotations

# =============================================================================
# 0. HF offline lock + env hygiene  (MUST run before any heavy import)
# =============================================================================
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="deepspeed")

# =============================================================================
# 1. Imports
# =============================================================================
import argparse
import logging
import sys
from pathlib import Path

# Make project root importable when running `python train.py` directly.
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch

from facet.model import FACETWanModel
from loader import build_loaders
import trainer  # re-exports trainer.{config,setup,optim,loss,sum,logger,ckpt,valid}


logger = logging.getLogger("facet.train")


# =============================================================================
# 2. argparse
# =============================================================================
def parse_args() -> argparse.Namespace:
    """
    CLI surface.

    Currently we expose just --train_yaml; everything else flows through the
    yaml. launch_train.py will pass this (plus any future opt-out flags like
    --dry_run / --no_lora_inject for debugging).
    """
    p = argparse.ArgumentParser(description="FACET training entry point.")
    p.add_argument(
        "--train_yaml",
        type=str,
        default=str(_PROJECT_ROOT / "train.yaml"),
        help="Path to train.yaml.",
    )
    return p.parse_args()


# =============================================================================
# 3. Main
# =============================================================================
def main() -> None:
    args = parse_args()

    # -------- 1. Load + merge config ----------------------------------------
    cfg = trainer.config.load_merge(args)

    # -------- 2. Build loaders BEFORE Accelerator init? --------------------
    # We need len(train_loader) to estimate total_steps, which feeds into
    # run_name. DistributedSampler-style metadata needs num_replicas / rank,
    # which we don't know until Accelerator exists.
    #
    # Resolution: build a TEMPORARY single-rank loader just to measure
    # len(train_loader); discard, then rebuild with the real rank info after
    # Accelerator init. This avoids loading any dataset content twice (only
    # the index lists are computed).
    #
    # NOTE: for Phase 1 we accept this small wart. Phase 2 may switch to
    # `max_steps` to skip the estimate entirely.
    tmp_train_loader, _tmp_val_loader, _tmp_ts, _tmp_vs = build_loaders(
        cfg_path=cfg.paths.data_config,
        batch_size=int(cfg.train.batch_size),
        num_workers=0,                    # cheap: just measure len
        seed=int(cfg.train.seed),
        rank=0,
        num_replicas=1,
    )
    total_steps = trainer.config.estimate_total_steps(cfg.train, len(tmp_train_loader))
    del tmp_train_loader, _tmp_val_loader, _tmp_ts, _tmp_vs

    # -------- 3. Setup env (accelerator + seed + dirs + generators) --------
    ctx = trainer.setup.init_env(cfg, args, total_steps=total_steps)
    accelerator = ctx.accelerator

    # -------- 4. Build the FACET model --------------------------------------
    # FACETWanModel.__init__ runs (in order):
    #   _load_base_components -> _freeze_base -> _init_lora
    # cfg.facet.device was overridden by trainer.setup.init_env to match
    # accelerator.device for the current rank, so weights load to the right
    # cuda:i on every worker.
    if accelerator.is_main_process:
        logger.info("[train] constructing FACETWanModel ...")
    model = FACETWanModel(cfg.facet)

    # -------- 5. Trainable-parameter summary -------------------------------
    trainer.sum.trainable_stats(model, output_root=ctx.output_root)

    # -------- 6. Real loaders (with rank-aware MultiSampler) ----------------
    train_loader, val_loader, train_sampler, val_sampler = build_loaders(
        cfg_path=cfg.paths.data_config,
        batch_size=int(cfg.train.batch_size),
        num_workers=int(cfg.train.num_workers),
        seed=int(cfg.train.seed),
        rank=accelerator.process_index,
        num_replicas=accelerator.num_processes,
    )

    # -------- 7. Optimizer (LoRA-only) + LR scheduler ----------------------
    optimizer = trainer.optim.build_optimizer(model, cfg.train)
    lr_scheduler = trainer.optim.build_lr_scheduler(
        optimizer, cfg.train, total_steps=total_steps,
    )

    # -------- 8. Accelerator.prepare ---------------------------------------
    # val_loader is intentionally NOT prepared. Validation runs as
    # rank-0 inference (avoids DDP all-gather inside scheduler step).
    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler,
    )

    # -------- 9. (Optional) resume ------------------------------------------
    # Phase 2+ : `trainer.ckpt.maybe_resume` returns 0 in the current stub.
    resumed_step = trainer.ckpt.maybe_resume(
        accelerator, model, optimizer, lr_scheduler, cfg.run.resume_from,
    )

    # -------- 10. Logger setup (config snapshot + mlflow hookup) -----------
    trainer.logger.setup(accelerator, cfg, output_root=ctx.output_root)

    # -------- 11. Loop counters ---------------------------------------------
    global_step = int(resumed_step or 0)
    start_epoch = 0   # Phase 2+: derive from resumed_step / len(train_loader)

    # =========================================================================
    # 12. Training loop  -- Phase 2 (NOT YET IMPLEMENTED)
    # =========================================================================
    if accelerator.is_main_process:
        logger.info(
            "[train] Phase 1 setup complete.\n"
            "  total_steps    = %d\n"
            "  epochs         = %d\n"
            "  len(train_loader) = %d  (per-rank)\n"
            "  run_name       = %s\n"
            "  output_root    = %s",
            total_steps,
            int(cfg.train.epochs),
            len(train_loader),
            ctx.run_name,
            ctx.output_root,
        )
        logger.info(
            "[train] training loop body is a Phase 2 TODO. Exiting cleanly."
        )

    for epoch in range(start_epoch, int(cfg.train.epochs)):
        train_sampler.set_epoch(epoch)
        model.train()
        # TODO Phase 2: per-batch step
        #   for batch in train_loader:
        #       with accelerator.accumulate(model):
        #           batch_prep = trainer.batch.prepare(batch, model, cfg)   # Phase 2
        #           t = trainer.loss.sample_timesteps(B, sampling=cfg.facet.training.timestep_sampling,
        #                                             generator=ctx.gpu_gen, device=ctx.device)
        #           noise = torch.randn(tgt_latents.shape, generator=ctx.gpu_gen,
        #                               device=tgt_latents.device, dtype=tgt_latents.dtype)
        #           noisy, target = trainer.loss.add_noise(tgt_latents, noise, t,
        #                                                  prediction_type=cfg.facet.training.prediction_type)
        #           out = model(noisy_latents=noisy, timesteps=t, prompt_embeds=embeds,
        #                       ref_latents=ref_lat, src_latents=src_lat, src_mask=src_mask)
        #           loss = trainer.loss.compute_loss(out["pred"], target, loss_type=cfg.facet.training.loss_type)
        #           accelerator.backward(loss)
        #           if accelerator.sync_gradients:
        #               grad_norm = accelerator.clip_grad_norm_(
        #                   (p for p in model.parameters() if p.requires_grad),
        #                   float(cfg.train.max_grad_norm))
        #           optimizer.step(); lr_scheduler.step(); optimizer.zero_grad()
        #       if accelerator.sync_gradients:
        #           global_step += 1
        #           if global_step % cfg.train.log_every_steps == 0:
        #               trainer.logger.log_step(loss.detach().float().item(),
        #                                       lr_scheduler.get_last_lr()[0],
        #                                       grad_norm, global_step)
        #           if global_step % cfg.train.val_every_steps == 0:
        #               metrics = trainer.valid.run(model, val_loader, val_sampler,
        #                                           cfg, epoch, global_step)
        #               trainer.logger.log_metrics(metrics, global_step)
        #               trainer.ckpt.save_topk(accelerator, model, optimizer, lr_scheduler,
        #                                      metrics, epoch, global_step, ctx.output_root, cfg.train)
        #           if global_step >= total_steps:
        #               break
        break   # Phase 1: do not enter the loop. Remove in Phase 2.

    # -------- 13. Tear-down --------------------------------------------------
    trainer.logger.finish()


if __name__ == "__main__":
    # Console logging on rank 0 (other ranks stay quiet by default).
    logging.basicConfig(
        level=os.environ.get("FACET_LOGLEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
