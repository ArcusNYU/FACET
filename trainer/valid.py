"""
trainer/valid.py
FACET training pipeline — in-loop validation + end-of-run heavy evaluation.

Two public entry points
-----------------------
run(...)        : called every cfg.train.val_every_steps (gated by start_eval).
                  ALWAYS computes validation loss over the (sharded) val set.
                  On cfg.train.save_every_steps it ALSO generates a FIXED set of
                  videos, scores light metrics (psnr/ssim/lpips), and dumps the
                  generated clips + their ground truth under
                      runs/<run>/samples/step_<global_step>/<clip_id>/{pred,gt}.mp4
                  Returns an averaged-across-ranks metrics dict; train.py forwards
                  it to logger.log_metrics + ckpt.save_topk (top-K by lpips).

heavy_eval(...) : called once at end-of-run on the BEST checkpoint's already
                  saved samples (no re-generation). Computes FID (+ FVD when an
                  I3D extractor is wired). Reused by test.py on held-out data.

Design notes
------------
* Distributed: every rank participates (no is_main_process gate). The val
  loader is sharded by val_sampler, so each rank scores a disjoint slice; the
  loss + light metrics are summed and reduced across ranks at the end. There is
  NO collective inside the per-batch loop (forward through the UNWRAPPED model
  so DDP never tries to all-reduce on unequal per-rank batch counts).
* Fixed sample set: generation clips are chosen deterministically by hashing
  clip_id with the run seed, then cached in a module global so the SAME clips
  are scored at every validation pass -> metric deltas reflect weight changes,
  not a moving sample set.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

import metrics
from utils import read_mp4, video_to_uint8, write_mp4

logger = logging.getLogger(__name__)


_LIGHT_KEYS: Tuple[str, ...] = ("psnr", "ssim", "lpips")

# resued on light metrics evaluation
_GEN_IDS: Optional[set] = None

# fps for the saved preview mp4s (metrics re-read frames regardless of fps).
_SAVE_FPS = 24  # FIXME: 暂时先设置为24 因为openhumanvid的数据集为24fps


# =============================================================================
# helpers
# =============================================================================
def _stable_hash(seed: int, key: str) -> int:
    """Deterministic cross-process hash (built-in hash() is per-process salted)."""
    # NOTE: md5: same input offers the same output
    return int(hashlib.md5(f"{seed}:{key}".encode("utf-8")).hexdigest(), 16)


def _global_index_clip_ids(concat) -> List[str]:
    """
    Map ConcatDataset global index -> clip_id WITHOUT loading any sample.

    Walks each sub-dataset's `.items` (BaseVideoDataset exposes the split index
    as a list of {"clip_id", "part", ...}); the global order matches
    ConcatDataset indexing (sub-dataset 0 first, then 1, ...), which is the same
    space MultiSampler emits. Returns [] if a sub-dataset can't be introspected.
    """
    ids: List[str] = []
    for d in getattr(concat, "datasets", []):
        items = getattr(d, "items", None)
        if items is None:
            return []
        ids.extend(str(it.get("clip_id", "")) for it in items)
    return ids


def _select_gen_ids(val_loader, val_sampler, num_samples: int, seed: int) -> set:
    """
    Pick this rank's fixed generation clips and cache them module-globally.

    The choice is deterministic: take this rank's (fixed-epoch) sampler indices,
    map them to clip_ids, sort by stable hash, keep the first `num_samples`.
    """
    global _GEN_IDS
    if _GEN_IDS is not None:
        return _GEN_IDS

    id_map = _global_index_clip_ids(val_loader.dataset)
    cand: List[str] = []
    if id_map:
        for gi in list(val_sampler):  # obtain clips prepared for current rank
            if 0 <= gi < len(id_map):
                cand.append(id_map[gi])
    cand = list(dict.fromkeys(cand))  # dedupe, keep order, prevent repeating samples due to padding
    cand.sort(key=lambda c: _stable_hash(seed, c)) # hash(md5) sorting
    _GEN_IDS = set(cand[: max(0, int(num_samples))])
    logger.info(
        "[trainer.valid] generation set: %d clip(s) chosen (per-rank, seed=%d).",
        len(_GEN_IDS), seed,
    )
    return _GEN_IDS


def _save_pair(step_dir: Path, clip_id: str, pred: torch.Tensor, gt: torch.Tensor) -> None:
    """Dump pred + gt as mp4 under step_dir/<clip_id>/ (pred/gt both [F,3,H,W] in [-1,1])."""
    out = step_dir / clip_id
    out.mkdir(parents=True, exist_ok=True)
    # allow_fallback=True: dump PNGs instead of crashing training if ffmpeg is absent.
    write_mp4(video_to_uint8(pred), out / "pred.mp4", fps=_SAVE_FPS, allow_fallback=True)
    write_mp4(video_to_uint8(gt), out / "gt.mp4", fps=_SAVE_FPS, allow_fallback=True)


# =============================================================================
# in-loop validation (light metrics)
# =============================================================================
@torch.no_grad()
def run(
    prepare_batch,
    raw_model,
    flow,
    val_loader,
    val_sampler,
    cfg,
    ctx,
    global_step: int,
) -> Dict[str, Any]:
    """
    In-loop Validation.

    Args:
        prepare_batch : train.prepare_batch, passed in by the caller.
        raw_model : UNWRAPPED FACETWanModel (accelerator.unwrap_model(model));
                    used for both the loss forward and .generate()/.decode().
        flow      : trainer.loss.FlowMatch (shared objective; reused, not rebuilt).
        val_loader / val_sampler : rank-sharded validation loader + sampler.
        cfg       : MergedConfig.
        ctx       : trainer.setup.SetupContext (accelerator/device/dtype/seed/dirs).
        global_step : current optimizer step.

    Returns:
        metrics dict averaged across ranks, e.g. light metrics:
            {"loss": ..., "psnr": ..., "ssim": ..., "lpips": ...}.
    """
    acc = ctx.accelerator
    device = ctx.device
    dtype = ctx.dtype

    raw_model.eval()

    do_gen = (
        int(cfg.train.save_every_steps) > 0
        and global_step % int(cfg.train.save_every_steps) == 0
    )
    chosen = (
        _select_gen_ids(
            val_loader, val_sampler,
            num_samples=int(cfg.validate.num_samples), seed=int(ctx.seed),
        )
        if do_gen else set()
    )
    step_dir = ctx.samples_dir / f"step_{global_step:07d}"

    # Reset every pass so the val loss is measured under IDENTICAL noise/timesteps
    val_gen = torch.Generator(device=device).manual_seed(int(ctx.seed) + acc.process_index)

    loss_sum = torch.zeros((), device=device, dtype=torch.float32)
    loss_cnt = torch.zeros((), device=device, dtype=torch.float32)
    light_sum = {k: torch.zeros((), device=device, dtype=torch.float32) for k in _LIGHT_KEYS}
    light_cnt = torch.zeros((), device=device, dtype=torch.float32)

    for batch in val_loader:
        prep = prepare_batch(raw_model, batch, cfg, device, dtype)
        # NOTE: though valid loader is not put on acc.device(normally CUDA),
        # prepare_batch will move the batch to the correct device and dtype
        tgt_latents = prep["tgt_latents"]
        B = tgt_latents.shape[0]

        # ---- a. validation loss (same FlowMatch objective as training) ----
        t = flow.sample_timesteps(B, generator=val_gen, device=device)
        noise = torch.randn(
            tgt_latents.shape, generator=val_gen,
            device=tgt_latents.device, dtype=tgt_latents.dtype,
        )
        noisy, target = flow.add_noise(tgt_latents, noise, t)
        out = raw_model(
            noisy_latents=noisy,
            timesteps=t,
            prompt_embeds=prep["prompt_embeds"],
            ref_latents=prep["ref_latents"],
            src_latents=prep["src_latents"],
            src_mask=prep["src_mask"],
        )
        loss = flow.compute_loss(out["pred"], target, t)
        loss_sum += loss.float() * B
        loss_cnt += B

        # ---- b. full generation + light metrics (gen step, chosen clips only) ----
        if not do_gen:
            continue
        clip_ids = batch.get("clip_id", [None] * B)
        for b in range(B):
            cid = clip_ids[b]
            if cid is None or cid not in chosen:
                continue
            try:
                _eval_gen(
                    raw_model, prep, batch, b, cid, cfg, device,
                    seed=int(ctx.seed), step_dir=step_dir,
                    light_sum=light_sum,
                )
                light_cnt += 1
            except Exception as e:  # noqa: BLE001 - never let one clip kill a long run
                logger.warning("[trainer.valid] generation failed for clip %s: %s", cid, e)

    # ---- c. reduce across ranks ----
    results: Dict[str, Any] = {}
    total_loss = acc.reduce(loss_sum.clone(), reduction="sum")
    total_lcnt = acc.reduce(loss_cnt.clone(), reduction="sum")
    if float(total_lcnt) > 0:
        results["loss"] = float(total_loss / total_lcnt)

    if do_gen:
        total_gcnt = acc.reduce(light_cnt.clone(), reduction="sum")
        for k in _LIGHT_KEYS:
            ksum = acc.reduce(light_sum[k].clone(), reduction="sum")
            if float(total_gcnt) > 0:
                results[k] = float(ksum / total_gcnt)
        if acc.is_main_process:
            logger.info(
                "[trainer.valid] step %d: %d generated sample(s) -> %s",
                global_step, int(total_gcnt), step_dir,
            )

    raw_model.train()
    return results


def _eval_gen(
    raw_model, prep, batch, b: int, cid: str, cfg, device,
    *, seed: int, step_dir: Path, light_sum: Dict[str, torch.Tensor],
) -> None:
    """Generate one clip, accumulate its light metrics, and save pred + gt."""
    # Per-clip deterministic init noise so improvement reflects weights, not noise.
    g = torch.Generator(device=device).manual_seed(_stable_hash(seed, str(cid)) & 0x7FFFFFFF)

    video = raw_model.generate(
        prompt_embeds=[prep["prompt_embeds"][b]],
        ref_latents=prep["ref_latents"][b : b + 1],
        src_latents=prep["src_latents"][b : b + 1],
        # [b: b+1]: preserve the batch dimension for the single-frame latents
        src_mask=prep["src_mask"][b : b + 1],
        num_inference_steps=int(cfg.validate.num_inference_steps),
        cfg_scale=1.0,  # TODO: CFG validation (wire cfg.validate.cfg_scale once CFG lands)
        sigma_shift=float(cfg.training.sigma_shift),
        generator=g,
    )  # [1, 3, F, H, W] in [-1, 1]

    pred = video[0].permute(1, 0, 2, 3).contiguous()  # [F, 3, H, W]
    gt = batch["tgt_video"][b].to(device=pred.device) # [T, 3, H, W] in [-1, 1]
    if abs(pred.shape[0] - gt.shape[0]) > 2:  # tolerate a 1~2 frame boundary diff
        logger.warning(
            "[trainer.valid] frame count mismatch for clip %s: pred=%d gt=%d",
            cid, pred.shape[0], gt.shape[0],
        )
    n = min(pred.shape[0], gt.shape[0])
    pred, gt = pred[:n], gt[:n]

    # Save FIRST so the clip persists for end-of-run heavy_eval / inspection even
    # if a light-metric backbone (e.g. lpips) is unavailable in this environment.
    _save_pair(step_dir, str(cid), pred, gt)

    lm = metrics.light_metrics(pred, gt)  # {"psnr","ssim","lpips"}
    for k in _LIGHT_KEYS:
        if k in lm:
            light_sum[k] += float(lm[k])


# =============================================================================
# end-of-run / test.py validation (heavy metrics)
# =============================================================================
def heavy_eval(
    step_dir,
    fvd_dir=None,
    fid_dir=None,
    value_range: Tuple[float, float] = (-1.0, 1.0),
) -> Dict[str, float]:
    """
    Heavy metrics over the pred/gt pairs saved under step_dir/<clip_id>/{pred,gt}.mp4.

    `fvd_dir` / `fid_dir` are the local weight dirs (cfg.paths.fvd_dir /
    cfg.paths.inception_dir); they are forwarded to metrics.heavy_metrics, which
    resolves + caches the I3D / InceptionV3 backbones on demand. Either may be
    None (FVD is then skipped; FID falls back to the torchmetrics default).

    NOTE: clips are stacked into one [N, T, 3, H, W] tensor (kept on CPU so FID's
    InceptionV3 never OOMs the GPU). For large N this should be batched/streamed.
    TODO: incremental feature accumulation to bound memory.
    """
    step_dir = Path(step_dir)
    if not step_dir.is_dir():
        logger.warning("[trainer.valid] heavy_eval: directory not found: %s", step_dir)
        return {}

    preds: List[torch.Tensor] = []
    gts: List[torch.Tensor] = []
    for clip_dir in sorted(p for p in step_dir.iterdir() if p.is_dir()):
        pv = read_mp4(clip_dir / "pred.mp4")
        gv = read_mp4(clip_dir / "gt.mp4")
        if pv is None or gv is None:
            continue
        preds.append(pv)
        gts.append(gv)

    if not preds:
        logger.warning("[trainer.valid] heavy_eval: no readable pred/gt pairs under %s", step_dir)
        return {}

    # Crop every clip to a common frame count so they stack cleanly.
    t_min = min(min(p.shape[0] for p in preds), min(g.shape[0] for g in gts))
    pred = torch.stack([p[:t_min] for p in preds], dim=0)   # [N, T, 3, H, W]
    gt = torch.stack([g[:t_min] for g in gts], dim=0)

    logger.info("[trainer.valid] heavy_eval: %d clip pair(s), %d frames each.", pred.shape[0], t_min)
    return metrics.heavy_metrics(
        pred, gt, value_range=value_range,
        fvd_dir=fvd_dir,
        fid_dir=fid_dir,
    )
