"""
trainer/loss.py
FACET training pipeline Step 8 & 11(loss / objective).

FlowMatch training objective for the WAN2.2-TI2V-5B base, packaged as a single
stateful `FlowMatch` object so the (shift-dependent) BSMNT weight table is built
once and reused every step:

    flow = FlowMatch(cfg.training)                       # build once
    t            = flow.sample_timesteps(B, generator=gpu_gen, device=dev)
    noisy, tgt   = flow.add_noise(latents, noise, t)
    loss         = flow.compute_loss(pred, tgt, t)

================================================================================
CONFIGURABLE OBJECTIVE  (3 presets)
================================================================================
The objective is the product of three orthogonal choices:

    timestep_sampling   sigma_shift   loss_weighting
    -----------------   -----------   --------------
 A  logit_normal        1.0 (linear)  none            <- SD3 recipe
 B  uniform(discrete)   5.0 (shifted) bsmnt           <- DiffSynth / WAN default  (FACET)
 C  logit_normal        5.0 (shifted) none            <- ablation candidate

FACET baseline uses B. Reasons (verified against source, not guessed):
  * DiffSynth's training (diffusion/training_module.py: `set_timesteps(1000,
    training=True)`) defaults set_timesteps_wan to shift=5 -> it trains LoRA on
    the SHIFTED sigma grid with discrete-uniform timesteps + BSMNT weighting.
  * WAN2.2-TI2V-5B ships sample_shift=5.0 (official wan_ti2v_5B config), and the
    shift map `sigma = shift*s / (1 + (shift-1)*s)` is WAN's own. Inference uses
    shift=5, so training at shift=5 keeps the LoRA on the SAME timestep
    distribution the base is conditioned on.
  * A LoRA trained on a DIFFERENT sigma law (e.g. linear) would first have to
    learn to remap "linear-sigma world" back into "shift-5 world" before it can
    even start learning the masked-reference editing task -- a real waste of a
    small-rank adapter. (The base's exact PRETRAIN-time sampling is not publicly
    documented; what is certain is that inference + every major WAN trainer
    (DiffSynth, AI-Toolkit, musubi) operate at shift~5, which is what matters
    for matching train to inference.)
  * The masked src-video branch is a strong conditioning signal, so even at high
    sigma the model is not "blindly guessing" -- high-t predictions stay
    meaningful, which makes the high-t-heavy shifted schedule a good fit.
  * BSMNT's gaussian weight also leans toward mid-t but decays MORE GENTLY than
    logit_normal, keeping non-trivial weight near the ends -- better for a task
    that must also get the structural high-t phase right.

================================================================================
sigma_shift  (high-noise budgeting)
================================================================================
Flow-matching denoises high->low sigma. The high-noise phase (sigma->1) fixes
GLOBAL structure (composition, motion direction, large color blocks); the
low-noise phase (sigma->0) only refines texture. For video / high resolution,
getting global structure wrong cannot be repaired by good texture, so spending
more of the (training + inference) budget on the high-noise segment pays off.
The shift map pushes sigma mass upward; WAN picks shift=5 (more aggressive than
SD3's 3) precisely because video demands stronger global coherence.

  shift = 1.0  -> identity (linear sigma)
  shift = 5.0  -> sigma = 5*s / (1 + 4*s)   (WAN default)

Decoupling note (kept): using a DIFFERENT shift at train vs inference is a valid
ablation (musubi explicitly notes train shift need not equal inference shift),
but it risks a train/inference mismatch for a small LoRA, so FACET keeps them
equal (5.0) for stable convergence rather than theoretical tidiness.

================================================================================
Continuous vs discrete timesteps
================================================================================
logit_normal MUST be continuous: sigmoid(Normal(x)) is a continuous sampling
function, so t must be a float. If t were discretized to integer steps, large
numbers of samples get rounded to the same integer t and the carefully-shaped
density information is lost -- quantizing collapses density in mid-range and
defeats the point.

uniform uses continuous purely for consistency + debugging friendliness.
(DiffSynth itself samples uniform DISCRETELY: `timestep_id = randint(0,1000)`
then indexes the grid. The continuous-uniform used here is the >=1000-point
limit of that and differs negligibly, while sharing one code path with
logit_normal.)

================================================================================
BSMNT loss weighting
================================================================================
Per-timestep loss scale, gaussian-centered at t=500:

    very low t (sigma->0): signal is almost the clean latent; velocity is easy
                           to predict; loss small.
    very high t (sigma->1): signal is almost pure noise; the model is basically
                           guessing; loss large but little is actually learned.
    mid t:                 mixed signal+noise; this is what truly decides
                           generation quality.

Without weighting, the extreme-t losses dominate the gradient. A gaussian weight
forces the model to focus on mid-t. We replicate DiffSynth's exact formula
(FlowMatchScheduler.set_training_weight) so behavior matches the reference,
evaluated CONTINUOUSLY on t (no grid snapping):

    y(t)   = exp(-2 * ((t - 500)/1000)^2)   # Gaussian function centered at t=500
    w(t)   = (y(t) - y_min) * (1000 / sum_grid(y - y_min))   # Normalized to sum to 1

where y_min and the normalizer are precomputed once from the shift-dependent
1000-point grid (so the mean weight stays ~1 and loss magnitude is comparable
to the unweighted case).

================================================================================
Other notes
================================================================================
[velocity]  prediction_type="velocity": target = noise - latents. This is
            d x_t / d sigma for the linear path x_t=(1-sigma)x0+sigma*noise, so
            it is consistent with FlowMatchScheduler.step's Euler update.
[fp32 loss] MSE is computed in fp32 (cast both inputs), matching DiffSynth's
            `mse_loss(pred.float(), target.float())`.
"""

from __future__ import annotations

# 1. Imports ------------------------------------------------------------------
import logging
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from trainer.config import TrainingConfig

logger = logging.getLogger(__name__)


# 2. Constants ----------------------------------------------------------------
_NUM_TRAIN_TIMESTEPS: float = 1000.0
_BSMNT_GRID_STEPS: int = 1000          # grid resolution used to precompute weights


# 3. FlowMatch objective ------------------------------------------------------
class FlowMatch:
    """
    Stateful FlowMatch objective. Built once per run from cfg.training.

    Holds: sampling regime, sigma shift, prediction/loss types, and a precomputed
    BSMNT normalization (y_min + norm) derived from the shift-dependent grid.
    """

    def __init__(self, cfg_training: TrainingConfig):
        self.sampling = cfg_training.timestep_sampling
        self.sigma_shift = float(cfg_training.sigma_shift)
        self.prediction_type = cfg_training.prediction_type
        self.loss_weighting = cfg_training.loss_weighting
        self.loss_type = cfg_training.loss_type

        if self.sampling not in ("uniform", "logit_normal"):
            raise ValueError(
                f"[trainer.loss] timestep_sampling={self.sampling!r}; "
                "expected 'uniform' or 'logit_normal'."
            )
        if self.prediction_type not in ("velocity", "noise"):
            raise ValueError(
                f"[trainer.loss] prediction_type={self.prediction_type!r}; "
                "expected 'velocity' or 'noise'."
            )
        if self.loss_weighting not in ("none", "bsmnt"):
            raise ValueError(
                f"[trainer.loss] loss_weighting={self.loss_weighting!r}; "
                "expected 'none' or 'bsmnt'."
            )
        if self.loss_type != "mse":
            raise ValueError(
                f"[trainer.loss] loss_type={self.loss_type!r}; expected 'mse'."
            )

        self._bsmnt_y_min, self._bsmnt_norm = self._precompute_bsmnt()

        logger.info(
            "[trainer.loss] FlowMatch: sampling=%s shift=%.3f weighting=%s pred=%s",
            self.sampling, self.sigma_shift, self.loss_weighting, self.prediction_type,
        )

    # ---- 3.1 shift map ------------------------------------------------------
    def _apply_shift(self, s: torch.Tensor) -> torch.Tensor:
        """sigma = shift * s / (1 + (shift-1) * s). Identity when shift == 1."""
        # NOTE: shift设置为1的时候 等同于线性sigma 不产生任何shit
        if self.sigma_shift == 1.0:
            return s
        sh = self.sigma_shift
        return sh * s / (1.0 + (sh - 1.0) * s)

    # ---- 3.2 BSMNT precompute ----------------------------------------------
    def _precompute_bsmnt(self) -> Tuple[float, float]:
        """
        Replicate FlowMatchScheduler.set_training_weight constants on the
        shift-dependent 1000-point grid: returns (y_min, norm) so that

            w(t) = clamp(exp(-2*((t-500)/1000)^2) - y_min, min=0) * norm

        has the same shape + normalization as DiffSynth's discrete table.
        Only meaningful for loss_weighting="bsmnt"; harmless otherwise.
        """
        s = torch.linspace(1.0, 0.0, _BSMNT_GRID_STEPS + 1)[:-1]   # [1000]
        sigma = self._apply_shift(s)
        grid_t = sigma * _NUM_TRAIN_TIMESTEPS
        y = torch.exp(-2.0 * ((grid_t - _NUM_TRAIN_TIMESTEPS / 2) / _NUM_TRAIN_TIMESTEPS) ** 2)
        y_min = float(y.min())
        denom = float((y - y_min).sum()) # NOTE: y - y_min: 使得两端的权重接近于0
        # Guard against degenerate denom (e.g. shift extremes); fall back to 1.0.
        norm = (_BSMNT_GRID_STEPS / denom) if denom > 1e-8 else 1.0
        return y_min, norm

    # ---- 3.3 timestep sampling ---------------------------------------------
    def sample_timesteps(
        self,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Sample [B] timesteps in [0, 1000), continuous, with sigma_shift applied.

        generator: pass SetupContext.gpu_gen for deterministic, isolated noise.
        """
        g_device = generator.device if generator is not None else torch.device(device)

        if self.sampling == "uniform":
            u = torch.rand((batch_size,), generator=generator, device=g_device, dtype=torch.float32)
        else:  # logit_normal
            z = torch.randn((batch_size,), generator=generator, device=g_device, dtype=torch.float32)
            u = torch.sigmoid(z)

        sigma = self._apply_shift(u)
        t = sigma * _NUM_TRAIN_TIMESTEPS
        return t.to(device=device, dtype=dtype)

    # ---- 3.4 forward noising + target --------------------------------------
    def add_noise(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Linear FlowMatch forward process + training target.s
            sigma = timesteps / 1000            (timesteps already shift-applied)
            x_t   = (1 - sigma) * x0 + sigma * noise
            target= noise - x0    (velocity)  |  noise  (eps-pred)
        Returns:
            noisy_latents : x_t, same shape & dtype as latents.
            target        : training target, same shape & dtype as latents.

        sigma is computed in fp32 (tail precision), then cast to latents.dtype.
        """
        if latents.shape != noise.shape:
            raise ValueError(
                f"latents.shape ({tuple(latents.shape)}) != noise.shape ({tuple(noise.shape)})"
            )

        sigma = (timesteps.to(torch.float32) / _NUM_TRAIN_TIMESTEPS).to(latents.device)
        while sigma.ndim < latents.ndim:
            sigma = sigma.unsqueeze(-1)
        sigma = sigma.to(latents.dtype)

        noisy_latents = (1.0 - sigma) * latents + sigma * noise

        if self.prediction_type == "velocity":
            target = noise - latents
        else:  # noise
            target = noise

        return noisy_latents, target

    # ---- 3.5 per-timestep weight -------------------------------------------
    def loss_weights(self, timesteps: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Return [B] BSMNT weights, or None when loss_weighting == "none".

        Evaluated continuously on t (no grid snapping) using the precomputed
        y_min + norm constants.
        """
        if self.loss_weighting == "none":
            return None
        t = timesteps.to(torch.float32)
        y = torch.exp(-2.0 * ((t - _NUM_TRAIN_TIMESTEPS / 2) / _NUM_TRAIN_TIMESTEPS) ** 2)
        w = (y - self._bsmnt_y_min).clamp_min(0.0) * self._bsmnt_norm
        return w

    # ---- 3.6 loss reduction -------------------------------------------------
    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        MSE (fp32) of (pred, target), optionally BSMNT-weighted per sample.

        Unweighted   -> mean over all elements.
        BSMNT-weighted-> per-sample MSE (mean over non-batch dims) * w(t),
                         then mean over the batch.
        """
        if pred.shape != target.shape:
            raise ValueError(
                f"pred.shape ({tuple(pred.shape)}) != target.shape ({tuple(target.shape)})"
            )

        weights = self.loss_weights(timesteps)
        if weights is None:
            return F.mse_loss(pred.float(), target.float()) # fp32

        if weights.shape[0] != pred.shape[0]:
            raise ValueError(
                f"weights.shape ({tuple(weights.shape)}) does not match batch size {pred.shape[0]}."
            )
        se = (pred.float() - target.float()).pow(2)
        per_sample = se.flatten(1).mean(dim=1)                  # [B]
        w = weights.to(per_sample.device, dtype=torch.float32)
        return (per_sample * w).mean()
