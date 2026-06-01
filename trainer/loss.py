"""
trainer/loss.py

FlowMatch training utilities for the WAN2.2-TI2V-5B base.

Three public functions consumed inside the training step:
  1. sample_timesteps(B, sampling, generator, device, dtype)
       Returns a batch of timesteps in [0, 1000].
  2. add_noise(latents, noise, timesteps, prediction_type)
       Returns (noisy_latents, training_target).
  3. compute_loss(pred, target, loss_type)
       Returns the scalar loss.

DiffSynth reference (DiffSynth/diffsynth/diffusion/flow_match.py and
diffusion/loss.py::FlowMatchSFTLoss):

    sigma          = scheduler.sigmas[timestep_id]
    noisy_latents  = (1 - sigma) * latents + sigma * noise
    training_target = noise - latents                                  # velocity
    loss = mse_loss(pred.float(), target.float())
    loss = loss * scheduler.training_weight(timestep)                  # BSMNT weighting

Differences vs DiffSynth that I deliberately chose:

    [A] Continuous timesteps.
        DiffSynth picks `timestep = scheduler.timesteps[timestep_id]` (i.e.
        from a discretized 1000-step grid). We use CONTINUOUS t in [0, 1000]
        for two reasons:
          - SD3 / Wan logit_normal works in continuous sigma; quantizing
            collapses density in mid-range and defeats the point.
          - Easier debugging (timestep_dist histogram is meaningful).
        For "uniform" sampling we still draw continuously over [0, 1000).

    [B] sigma = t / 1000   (no shift).
        WAN's FlowMatchScheduler uses sigma_shift=5 at inference. At training,
        DiffSynth's loss reads sigmas from the SHIFTED grid; we use the linear
        identity sigma=t/1000 to keep the training objective shift-agnostic.
        This matches the SD3 paper's recipe and avoids coupling train-time
        loss shape with inference-time sigma schedule.
        => TODO Phase 2.x: re-evaluate if convergence suffers; we can swap
        in `scheduler.sigmas[argmin(|timesteps - t|)]` to align with inference.

    [C] No BSMNT training_weight.
        DiffSynth multiplies loss by a Gaussian-shaped weight along t. With
        logit_normal sampling, the SAMPLING density already concentrates on
        mid-t, so the extra weighting double-emphasizes the same region.
        We leave a placeholder `apply_training_weight` flag; default off.

    [D] Loss math in fp32 (matches DiffSynth: `pred.float(), target.float()`).
"""

from __future__ import annotations

# 1. Imports ------------------------------------------------------------------
import logging
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# 2. Constants ----------------------------------------------------------------

# WAN's FlowMatchScheduler.num_train_timesteps. Hard-coded here because the
# scheduler is a runtime object; for the loss math we only need this scalar.
_NUM_TRAIN_TIMESTEPS: float = 1000.0


# 3. Timestep sampling --------------------------------------------------------


def sample_timesteps(
    batch_size: int,
    sampling: str = "logit_normal",
    generator: Optional[torch.Generator] = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample a batch of training timesteps in [0, num_train_timesteps).

    Args:
        batch_size : B.
        sampling   : "uniform" | "logit_normal".
        generator  : torch.Generator. Must live on the same device as the
                     output for cuda tensors. For determinism, ALWAYS pass
                     trainer.setup.SetupContext.gpu_gen here.
        device     : output device.
        dtype      : output dtype. Default fp32 (model.compute_time_features
                     will recast to bf16 internally).

    Returns:
        timesteps : [B] in [0, 1000) (fp32 by default).

    Sampling regimes:
        "uniform"      : t ~ U[0, 1000).
                         Each timestep is equally likely. Matches DiffSynth
                         pre-Wan defaults; baseline.
        "logit_normal" : u ~ N(0, 1), sigma = sigmoid(u), t = sigma * 1000.
                         The SD3 / Wan recommendation; concentrates mass on
                         mid-sigma which is where the model learns most.

    NOTE on `torch.randn_like`:
        torch.randn_like does NOT accept a `generator=` kwarg. trainer.txt L149
        contained a typo on this. Use torch.randn(shape, generator=..., ...)
        instead -- which is what we do below.
    """
    g_device = generator.device if generator is not None else torch.device(device)

    if sampling == "uniform":
        # U[0, 1) -> scale to [0, 1000).
        u = torch.rand((batch_size,), generator=generator, device=g_device, dtype=torch.float32)
        t = u * _NUM_TRAIN_TIMESTEPS

    elif sampling == "logit_normal":
        # SD3 / Wan recipe: sigma = sigmoid(N(0,1)), t = sigma * 1000.
        u = torch.randn((batch_size,), generator=generator, device=g_device, dtype=torch.float32)
        sigma = torch.sigmoid(u)
        t = sigma * _NUM_TRAIN_TIMESTEPS

    else:
        raise ValueError(
            f"[trainer.loss] Unknown timestep_sampling={sampling!r}; "
            "expected 'uniform' or 'logit_normal'."
        )

    return t.to(device=device, dtype=dtype)


# 4. Forward noising + target -------------------------------------------------


def add_noise(
    latents: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    prediction_type: str = "velocity",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Linear FlowMatch forward process + training target.

    Forward (with sigma = t / 1000):
        x_t  = (1 - sigma) * x_0 + sigma * noise

    Targets:
        prediction_type = "velocity": target = noise - x_0                (FlowMatch velocity)
        prediction_type = "noise"   : target = noise                       (eps-pred, legacy)

    Args:
        latents     : x_0,   [B, ...]   any rank >= 1; broadcasting over t.
        noise       : eps,   same shape as latents.
        timesteps   : [B]    fp32, in [0, 1000].
        prediction_type : see above.

    Returns:
        noisy_latents : x_t, same shape & dtype as latents.
        target        : training target, same shape & dtype as latents.

    NOTE:
        - sigma is computed in fp32 to avoid catastrophic precision loss in
          bf16 when t is near 0 or 1000 (epsilon * eps still matters at the
          tail). Then cast back to latents.dtype for the actual blend.
        - We broadcast sigma to latents.ndim by trailing-unsqueeze (sigma is
          shared over channels / spatial dims).
    """
    if latents.shape != noise.shape:
        raise ValueError(
            f"latents.shape ({tuple(latents.shape)}) != "
            f"noise.shape ({tuple(noise.shape)})"
        )

    sigma_fp32 = (timesteps.to(torch.float32) / _NUM_TRAIN_TIMESTEPS).to(latents.device)
    while sigma_fp32.ndim < latents.ndim:
        sigma_fp32 = sigma_fp32.unsqueeze(-1)
    sigma = sigma_fp32.to(latents.dtype)

    noisy_latents = (1.0 - sigma) * latents + sigma * noise

    if prediction_type == "velocity":
        target = noise - latents
    elif prediction_type == "noise":
        target = noise
    else:
        raise ValueError(
            f"[trainer.loss] Unknown prediction_type={prediction_type!r}; "
            "expected 'velocity' or 'noise'."
        )

    return noisy_latents, target


# 5. Loss reduction -----------------------------------------------------------


def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "mse",
    timestep_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Reduce (pred, target) to a scalar training loss.

    Args:
        pred              : [B, ...] model output (same shape as target).
        target            : [B, ...] training target from add_noise.
        loss_type         : "mse" only (placeholder for future "l1" / "huber").
        timestep_weights  : optional [B] weights to scale per-sample loss
                            BEFORE reduction. Used by future BSMNT weighting
                            (matches DiffSynth's training_weight). Default None
                            (uniform weight 1.0 per sample).

    Returns:
        loss : scalar fp32 tensor.

    Implementation notes:
        - We compute MSE in fp32 (cast both inputs) to match DiffSynth's
          `mse_loss(pred.float(), target.float())` pattern. bf16 + small
          residuals can otherwise mask gradient signal in early training.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred.shape ({tuple(pred.shape)}) != "
            f"target.shape ({tuple(target.shape)})"
        )

    if loss_type != "mse":
        raise ValueError(
            f"[trainer.loss] Unsupported loss_type={loss_type!r}; "
            "expected 'mse' (extend here for l1 / huber later)."
        )

    if timestep_weights is None:
        return F.mse_loss(pred.float(), target.float())

    # Per-sample MSE (mean over non-batch dims), then weight & mean over batch.
    if timestep_weights.shape[0] != pred.shape[0]:
        raise ValueError(
            f"timestep_weights.shape ({tuple(timestep_weights.shape)}) "
            f"does not match batch size {pred.shape[0]}."
        )
    se = (pred.float() - target.float()).pow(2)
    per_sample = se.flatten(1).mean(dim=1)                 # [B]
    w = timestep_weights.to(per_sample.device, dtype=torch.float32)
    return (per_sample * w).mean()
