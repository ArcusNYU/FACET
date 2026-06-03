"""
train.py

FACET / WAN2.2-TI2V-5B + OmniControl LoRA training entry point.

Layout:
    0.  argparse + HF offline lock
    1.  trainer.config.load_merge
    2.  trainer.setup.init_env
    3.  FACETWanModel construction + set_lora(trainable)
    4.  trainer.optim.model_stats
    5.  loader.build_loaders
    6.  trainer.optim.build_optimizer / build_lr_scheduler
    7.  accelerator.prepare
    8.  FlowMatch objective + CheckpointManager + grad-clip param view
    9.  trainer.logger.setup
    10. loop counters
    11. training loop (validation + metrics land in Phase 3)
    12. trainer.logger.finish()
"""

from __future__ import annotations

# =============================================================================
# 0. HF offline lock + env hygiene
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
from tqdm.auto import tqdm

from facet.model import FACETWanModel
from loader import build_loaders
import trainer  # re-exports trainer.{config,setup,optim,loss,logger,ckpt,valid}


logger = logging.getLogger("facet.train")


# =============================================================================
# 2. argparse
# =============================================================================
def parse_args() -> argparse.Namespace:
    """
    CLI surface.

    plus any future opt-out flags for debugging.
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
# 3. batch preparation (OmniControl-specific data path)
# =============================================================================
def prepare_batch(raw_model, batch, cfg, device, dtype):
    # FIXME: loader给出的内容目前除了prompt embeds 都不是list
    # tgt_latent似乎也是一个list 这部分目前逻辑混乱需要进行溯源
    """
    Turn a raw dataloader batch into model.forward kwargs.
    
    Loader layout:
        src_video  : [T, 3, H, W] in [-1, 1]   (masked source video, pixel)
        src_mask   : [T, 1, H, W] in {0, 1}
        ref_img    : [3, H, W]    in [-1, 1]
        tgt_video  : [T, 3, H, W] in [-1, 1]
        tgt_latent : [16, F_lat, h, w] or None  (cached)
        t5_emb     : [L, 4096]    or None       (cached)

    NOTE on layout: the loader is FRAME-first ([T, C, H, W]); the WAN VAE /
    model is CHANNEL-first ([B, C, T, H, W]). Videos/masks are permuted here.

    src_video / ref_img are ALWAYS VAE-encoded online because of real-time mask perturbation
    tgt_latent / t5_emb use the cache when cfg.training.cached_* is set.

    Returns a dict of forward kwargs + the target latents 
    (for the FlowMatch target) on `device`/`dtype`.
    """
    src_video = batch["src_video"]    # List[[T,3,H,W]]
    src_mask  = batch["src_mask"]     # List[[T,1,H,W]]
    ref_img   = batch["ref_img"]      # List[[3,H,W]]

    # a. src branch: permute frame-first -> channel-first, stack, VAE-encode.
    src_video = torch.stack([v.permute(1, 0, 2, 3) for v in src_video], dim=0)  # [B,3,T,H,W]
    # FIXME: 改变处理方式 
    src_latents = raw_model.encode_src_video(src_video)                          # [B,16,F_lat,h,w]

    # b. ref branch: stack [3,H,W] -> [B,3,H,W], VAE-encode to 1-frame latents.
    ref_img = torch.stack(ref_img, dim=0)                                        # [B,3,H,W]
    # FIXME: 改变处理方式 
    ref_latents = raw_model.encode_reference_image(ref_img)                      # [B,16,1,hr,wr]

    # c. src_mask: [T,1,H,W] -> [1,T,H,W] -> stack -> [B,1,F,H,W] (pixel space).
    src_mask = torch.stack([m.permute(1, 0, 2, 3) for m in src_mask], dim=0).to(device)
    # FIXME: 改变处理方式 

    # d. target latents: cached or encode tgt_video online.
    if cfg.training.cached_tgt_latent:
        tgt_lat = batch["tgt_latent"]
        if any(t is None for t in tgt_lat):
            raise RuntimeError(
                "cached_tgt_latent=True but batch['tgt_latent'] has None entries "
                "(cache miss). Run the latent cache step or set cached_tgt_latent=false."
            )
        tgt_latents = torch.stack(tgt_lat, dim=0).to(device=device, dtype=dtype)
    else:
        tgt_video = torch.stack([v.permute(1, 0, 2, 3) for v in batch["tgt_video"]], dim=0)
        tgt_latents = raw_model.encode_src_video(tgt_video)
    # FIXME: 如果tgt latent的确是list的话 那么可以使用 torch.stack 但是 tgt_video不是

    # e. prompt embeds: cached t5 or encode captions. forward wants List[Tensor].
    if cfg.training.cached_t5:
        t5 = batch["t5_emb"]
        if any(e is None for e in t5):
            raise RuntimeError(
                "cached_t5=True but batch['t5_emb'] has None entries (cache miss)."
            )
        prompt_embeds = [e.to(device=device, dtype=dtype) for e in t5]
    else:
        prompt_embeds = raw_model.encode_prompt(batch["caption"])

    # f. (Phase 2+) CFG dropout fork: when cfg.training.cfg_training, randomly
    #    blank prompt_embeds / ref_latents here. Disabled for now.

    return {
        "noisy_inputs": {
            "prompt_embeds": prompt_embeds,
            "ref_latents": ref_latents,
            "src_latents": src_latents,
            "src_mask": src_mask,
        }, #FIXME: 为啥非要noisy inputs这个加一层?? 
        "tgt_latents": tgt_latents,
    } #TODO: 标注出来return中内容的形状


# =============================================================================
# 4. Main
# =============================================================================
def main() -> None:
    args = parse_args()

    # -------- 1. Load + record config ---------------------------------------
    cfg = trainer.config.load_merge(args)

    # -------- 2. Setup env (accelerator + seed + dirs + generators) --------
    total_steps = int(cfg.train.max_steps)
    ctx = trainer.setup.init_env(cfg, args, total_steps=total_steps)
    accelerator = ctx.accelerator

    # -------- 3. Build the FACET model --------------------------------------
    # FACETWanModel.__init__ runs: _load_base_components -> _freeze_base -> _init_lora.
    # cfg.facet.device was set to accelerator.device per rank by init_env, so the
    # weights load onto the right cuda:i and self.device / self.dtype are correct.
    if accelerator.is_main_process:
        logger.info("[train] constructing FACETWanModel ...")
    model = FACETWanModel(cfg.facet)

    # 3.1 Mark LoRA params trainable (base already frozen in __init__).
    model.set_lora(trainable=True)

    # -------- 4. Trainable-parameter stats (main process; clean names pre-prepare)
    if accelerator.is_main_process:
        trainer.optim.model_stats(model, output_root=ctx.output_root)

    # -------- 5. Data loaders (rank-aware MultiSampler) ---------------------
    train_loader, val_loader, train_sampler, val_sampler = build_loaders(
        cfg_path=cfg.paths.data_config,
        batch_size=int(cfg.train.batch_size),
        num_workers=int(cfg.train.num_workers),
        seed=int(cfg.train.seed),
        rank=accelerator.process_index,
        num_replicas=accelerator.num_processes,
    )

    # -------- 6. Optimizer (LoRA-only) + LR scheduler ---
    optimizer = trainer.optim.build_optimizer(model, cfg.train)
    lr_scheduler = trainer.optim.build_lr_scheduler(optimizer, cfg.train, total_steps=total_steps)

    # -------- 7. accelerator.prepare ----------------------------------------
    # val_loader is intentionally NOT prepared: validation runs as rank-0
    # inference (avoids DDP all-gather inside the scheduler step).
    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler,
    )

    # -------- 8. Objective + checkpoint manager + grad-clip view -----------
    raw_model = accelerator.unwrap_model(model)             # for VAE/T5 encode calls
    flow = trainer.loss.FlowMatch(cfg.training)             # builds BSMNT table once
    ckpt_mgr = trainer.ckpt.CheckpointManager(
        accelerator=accelerator,
        run_name=ctx.run_name,
        output_root=ctx.output_root,
        ckpt_root=Path(cfg.paths.ckpt_root),
        topk=int(cfg.validate.topk),
        primary_metric=cfg.validate.primary_metric,
    )
    params_to_clip = [p for p in model.parameters() if p.requires_grad]

    # -------- 9. Logger setup (snapshot + trackers) ------------------------
    trainer.logger.setup(accelerator, cfg, output_root=ctx.output_root)

    # -------- 10. Loop counters ---------------------------------------------
    global_step = 0
    start_epoch = 0

    if accelerator.is_main_process:
        logger.info(
            "[train] setup complete.\n"
            "  total_steps      = %d\n"
            "  epochs           = %d\n"
            "  len(train_loader)= %d  (per-rank)\n"
            "  log_every        = %d   val_every = %d   save_every = %d\n"
            "  run_name         = %s\n"
            "  output_root      = %s",
            total_steps, int(cfg.train.epochs), len(train_loader),
            int(cfg.train.log_every_steps), int(cfg.train.val_every_steps),
            int(cfg.train.save_every_steps), ctx.run_name, ctx.output_root,
        )

    pbar = tqdm(
        total=total_steps,
        disable=not accelerator.is_main_process,
        desc="train",
        dynamic_ncols=True,
    )

    # =========================================================================
    # 11. Training loop
    # =========================================================================
    done = False
    for epoch in range(start_epoch, int(cfg.train.epochs)):
        if done:
            break
        train_sampler.set_epoch(epoch)   # refresh per-epoch shuffle deterministically
        model.train()

        for batch in train_loader:
            with accelerator.accumulate(model):
                # a. data prep (encode src/ref online; cached tgt latent + t5)
                prep = prepare_batch(raw_model, batch, cfg, ctx.device, ctx.dtype)
                tgt_latents = prep["tgt_latents"]
                fwd = prep["noisy_inputs"]  # FIXME: 不需要fwd 改成直接从prep中索引
                B = tgt_latents.shape[0]

                # b. FlowMatch noise + target
                t = flow.sample_timesteps(B, generator=ctx.gpu_gen, device=ctx.device)
                noise = torch.randn(
                    tgt_latents.shape, generator=ctx.gpu_gen,
                    device=tgt_latents.device, dtype=tgt_latents.dtype,
                )
                noisy, target = flow.add_noise(tgt_latents, noise, t)

                # c. forward (3-branch OmniControl attention) -> velocity pred
                out = model(
                    noisy_latents=noisy,
                    timesteps=t,
                    prompt_embeds=fwd["prompt_embeds"],
                    ref_latents=fwd["ref_latents"],
                    src_latents=fwd["src_latents"],
                    src_mask=fwd["src_mask"],
                )
                loss = flow.compute_loss(out["pred"], target, t)

                # d. backward + (sync-gated) grad clip + step
                accelerator.backward(loss)
                grad_norm = None
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, float(cfg.train.max_grad_norm))
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # e. step-gated bookkeeping (only on real optimizer steps)
            if accelerator.sync_gradients:
                global_step += 1
                pbar.update(1)

                if global_step % int(cfg.train.log_every_steps) == 0:
                    gn = float(grad_norm) if grad_norm is not None else None
                    trainer.logger.log_step(
                        loss.detach().float().item(),
                        lr_scheduler.get_last_lr()[0],
                        gn, global_step,
                    )

                if global_step % int(cfg.train.save_every_steps) == 0:
                    ckpt_mgr.save_last(model, global_step)

                # Validation + metric-driven top-K land in Phase 3. valid.run is
                # a stub returning {} for now, so log_metrics / save_topk no-op.
                if global_step % int(cfg.train.val_every_steps) == 0:
                    metrics = trainer.valid.run(model, val_loader, val_sampler, cfg, epoch, global_step)
                    trainer.logger.log_metrics(metrics, global_step)
                    ckpt_mgr.save_topk(model, metrics, global_step)

                if global_step >= total_steps:
                    done = True
                    break

    pbar.close()

    # -------- 12. Tear-down --------------------------------------------------
    ckpt_mgr.save_last(model, global_step)   # final rolling checkpoint
    trainer.logger.finish()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("[train] done. global_step=%d", global_step)


if __name__ == "__main__":
    # Console logging on rank 0 (other ranks stay quiet by default).
    logging.basicConfig(
        level=os.environ.get("FACET_LOGLEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
