"""
metrics.py — video quality metrics shared by train.py & test.py.

Input convention
----------------
Every metric compares a prediction against an ALIGNED ground truth:

    pred, gt : float tensors shaped [T, C, H, W] or [B, T, C, H, W]
               C = 3 (RGB), values in `value_range` (default (-1, 1), the
               project's pixel range; per utils.video_to_uint8). A leading
               batch dim is optional and is flattened together with the frame
               dim before the per-frame metrics run.

Two tiers
---------
  * light_metrics()  — cheap, per-frame reconstruction metrics (run in-loop):
        psnr  (higher = better)
        ssim  (higher = better)
        lpips (lower  = better)
  * heavy_metrics()  — distribution metrics over many frames/clips (run at the
                       end on the best checkpoint):
        fid   (lower = better)  — frame-level Fréchet Inception distance
        fvd   (lower = better)  — clip-level Fréchet Video distance (I3D)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "psnr", "ssim", "lpips", "fid", "fvd",
    "build_fvd_extractor", "light_metrics", "heavy_metrics",
]

ValueRange = Tuple[float, float]
_DEFAULT_RANGE: ValueRange = (-1.0, 1.0)


# =============================================================================
# shape / range normalisation helpers
# =============================================================================
def _as_5d(x: torch.Tensor) -> torch.Tensor:
    """[T,C,H,W] -> [1,T,C,H,W]; [B,T,C,H,W] passes through."""
    # NOTE: 因为是视频任务 所以4维输入默认第1维是帧数量
    if x.dim() == 4:
        x = x.unsqueeze(0)
    if x.dim() != 5:
        raise ValueError(
            f"expected a video tensor of rank 4 [T,C,H,W] or 5 [B,T,C,H,W], "
            f"got shape {tuple(x.shape)}"
        )
    return x


def _flatten_frames(x: torch.Tensor) -> torch.Tensor:
    """[B,T,C,H,W] -> [B*T,C,H,W]."""
    b, t, c, h, w = x.shape
    return x.reshape(b * t, c, h, w)


def _to_01(x: torch.Tensor, value_range: ValueRange) -> torch.Tensor:
    """Rescale from `value_range` to [0, 1] and clamp."""
    lo, hi = value_range
    return ((x.float() - lo) / (hi - lo)).clamp_(0.0, 1.0)


def _to_pm1(x: torch.Tensor, value_range: ValueRange) -> torch.Tensor:
    """Rescale from `value_range` to [-1, 1] and clamp (LPIPS input convention)."""
    return _to_01(x, value_range).mul_(2.0).sub_(1.0)


def _prep_pair(
    pred: torch.Tensor,
    gt: torch.Tensor,
    value_range: ValueRange,
    *,
    to_pm1: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Validate + flatten both videos to [N,C,H,W] in [0,1] (or [-1,1])."""
    pred = _as_5d(pred)
    gt = _as_5d(gt)
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch: {tuple(pred.shape)} vs {tuple(gt.shape)}")
    conv = _to_pm1 if to_pm1 else _to_01
    return _flatten_frames(conv(pred, value_range)), _flatten_frames(conv(gt, value_range))


# =============================================================================
# light metrics — PSNR / SSIM / LPIPS  (per-frame, averaged)
# =============================================================================
# NOTE: both PSNR and SSIM require input in range [0, 1]
# ---- PSNR ----
def psnr(pred: torch.Tensor, gt: torch.Tensor, value_range: ValueRange = _DEFAULT_RANGE) -> float:
    """Mean per-frame PSNR (dB), higher is better."""
    from torchmetrics.functional.image import peak_signal_noise_ratio

    p, g = _prep_pair(pred, gt, value_range)
    val = peak_signal_noise_ratio(p, g, data_range=1.0, dim=(1, 2, 3), reduction="elementwise_mean")
    return float(val)


# ---- SSIM ----
def ssim(pred: torch.Tensor, gt: torch.Tensor, value_range: ValueRange = _DEFAULT_RANGE) -> float:
    """Mean per-frame SSIM, higher is better."""
    from torchmetrics.functional.image import structural_similarity_index_measure

    p, g = _prep_pair(pred, gt, value_range)
    val = structural_similarity_index_measure(p, g, data_range=1.0)
    return float(val)


# ---- LPIPS ----
# NOTE: LPIPS requires input in range [-1, 1]
# lpips models are expensive to build; cache one per (net, device).
_LPIPS_CACHE: Dict[Tuple[str, str], "torch.nn.Module"] = {}

def _get_lpips(net: str, device: torch.device) -> "torch.nn.Module":
    import lpips as lpips_lib

    key = (net, str(device))
    if key not in _LPIPS_CACHE:
        model = lpips_lib.LPIPS(net=net).to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _LPIPS_CACHE[key] = model
    return _LPIPS_CACHE[key]

def lpips(
    pred: torch.Tensor,
    gt: torch.Tensor,
    value_range: ValueRange = _DEFAULT_RANGE,
    net: str = "alex",
) -> float:
    """Mean per-frame LPIPS (AlexNet backbone by default), lower is better."""
    p, g = _prep_pair(pred, gt, value_range, to_pm1=True)  # LPIPS wants [-1, 1]
    model = _get_lpips(net, p.device)
    with torch.no_grad():
        d = model(p, g)  # [N, 1, 1, 1]
    return float(d.mean())


# ---- Light metrics ----
def light_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    value_range: ValueRange = _DEFAULT_RANGE,
    lpips_net: str = "alex",
) -> Dict[str, float]:
    """Compute the cheap per-frame reconstruction metrics in one shot."""
    return {
        "psnr": psnr(pred, gt, value_range),
        "ssim": ssim(pred, gt, value_range),
        "lpips": lpips(pred, gt, value_range, net=lpips_net),
    }


# =============================================================================
# heavy metrics — FID / FVD  (distribution distance over many frames/clips)
# =============================================================================
def _frechet_distance(feat_real: torch.Tensor, feat_fake: torch.Tensor, eps: float = 1e-6) -> float:
    """
    Fréchet distance between two sets of feature vectors (the FID/FVD core).

    Args:
        feat_real / feat_fake: [N, D] feature matrices (N samples, D dims).

    Returns the scalar Fréchet distance assuming both feature sets are Gaussian.
    """
    import numpy as np
    from scipy import linalg

    r = feat_real.detach().cpu().double().numpy()
    f = feat_fake.detach().cpu().double().numpy()
    # double(): use fp64 to avoid numerical instability
    # mean(axis=0): compute the mean of each Gaussian distribution
    mu1, mu2 = r.mean(axis=0), f.mean(axis=0)
    # covariance matrix: a measure of how two variables vary in tandem from their means
    sigma1, sigma2 = np.cov(r, rowvar=False), np.cov(f, rowvar=False)

    # difference between the means of the two Gaussian distributions:
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():  # numerical fallback (singular product)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    # offset: add a small value to the covariance matrix to make it diagonalizable
    if np.iscomplexobj(covmean):
        covmean = covmean.real # remove complex part
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))
    # d^2 = (mu1 - mu2)^2 + trace(sigma1 + sigma2 - 2 * covmean)


# ---- FID ----
def fid(pred: torch.Tensor, gt: torch.Tensor, value_range: ValueRange = _DEFAULT_RANGE) -> float:
    """
    Frame-level Fréchet Inception Distance (treats every frame as an image).

    Lower is better. Backed by torchmetrics' InceptionV3 feature extractor.
    """
    from torchmetrics.image.fid import FrechetInceptionDistance

    p, g = _prep_pair(pred, gt, value_range)  # [N, 3, H, W] in [0, 1]
    metric = FrechetInceptionDistance(feature=2048, normalize=True).to(p.device)
    metric.update(g, real=True)
    metric.update(p, real=False)
    # real: whether the data is ground truth or generated
    return float(metric.compute())


# ---- FVD ----
def fvd(
    pred: torch.Tensor,
    gt: torch.Tensor,
    value_range: ValueRange = _DEFAULT_RANGE,
    feature_extractor: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> float:
    """
    Clip-level Fréchet Video Distance, lower is better.

    FVD = Fréchet distance between I3D features of the real vs generated clips,
        feature_extractor(video) -> [B, D] features
            video : [B, T, C, H, W] in `value_range`
    """
    pred = _as_5d(pred)
    gt = _as_5d(gt)
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch: {tuple(pred.shape)} vs {tuple(gt.shape)}")

    if feature_extractor is None:
        raise NotImplementedError(
            "fvd() needs an I3D feature_extractor(video)->[B,D]. Build one with "
            "metrics.build_fvd_extractor(weights_dir). The Fréchet math is ready "
            "in _frechet_distance()."
        )

    feat_real = feature_extractor(_to_pm1(gt, value_range))
    feat_fake = feature_extractor(_to_pm1(pred, value_range))
    return _frechet_distance(feat_real, feat_fake)


# ---- I3D feature extractor (FVD backbone) ----
_I3D_CACHE: Dict[Tuple[str, str], Optional[Callable[[torch.Tensor], torch.Tensor]]] = {}

def build_fvd_extractor(
    weights_dir: str | Path,
    device: str | torch.device = "cpu",
):
    """
    Build a Kinetics-400 I3D feature extractor for FVD, loaded LOCALLY (offline).
    Args:
        weights_dir : local dir (or file) holding the I3D checkpoint.
        device      : where to place the backbone.

    Returns:
        extractor(video[B,T,C,H,W] in [-1,1]) -> features[B, D], or None when no
        usable checkpoint is found / the interface doesn't match (FVD is then
        skipped and only FID is reported, instead of crashing the run).

    Expected checkpoint: the de-facto FVD I3D TorchScript module (the one used by
    StyleGAN-V / common video-quality repos), callable as
        model(x[B,C,T,H,W], rescale=False, resize=True, return_features=True).
    A TF/Keras I3D (e.g. huggingface Mouwiya/kinetics-400) would need converting
    to this TorchScript form first.
    """
    # cache -> return
    key = (str(weights_dir), str(device))
    if key in _I3D_CACHE:
        return _I3D_CACHE[key]

    # search for checkpoint
    wdir = Path(weights_dir)
    ckpt: Optional[Path] = None
    if wdir.is_file():
        ckpt = wdir
    elif wdir.is_dir():
        for name in ("i3d_torchscript.pt", "i3d_pretrained_400.pt"):
            if (wdir / name).exists():
                ckpt = wdir / name
                break
        if ckpt is None:
            cands = sorted([*wdir.glob("*.pt"), *wdir.glob("*.pth"), *wdir.glob("*.ts")])
            ckpt = cands[0] if cands else None

    if ckpt is None:
        logger.warning("[metrics] no I3D checkpoint under %s; FVD will be skipped.", weights_dir)
        _I3D_CACHE[key] = None
        return None

    # load checkpoint
    try:
        model = torch.jit.load(str(ckpt), map_location=device).eval().to(device)
    except Exception as e:  # noqa: BLE001 - a bad/incompatible file must not crash training
        logger.warning("[metrics] failed to load I3D '%s' (%s); FVD skipped.", ckpt, e)
        _I3D_CACHE[key] = None
        return None

    @torch.no_grad()
    def extractor(video: torch.Tensor) -> torch.Tensor:
        # video: [B, T, C, H, W] in [-1, 1] -> I3D wants [B, C, T, H, W].
        x = video.to(device=device, dtype=torch.float32).permute(0, 2, 1, 3, 4).contiguous()
        feats = model(x, rescale=False, resize=True, return_features=True)
        return feats.flatten(1)

    # Smoke-test the interface so an interface mismatch disables FVD cleanly.
    # try:
    #     _ = extractor(torch.zeros(1, 16, 3, 64, 64, device=device))
    # except Exception as e:  # noqa: BLE001
    #     logger.warning("[metrics] I3D interface mismatch for '%s' (%s); FVD skipped.", ckpt, e)
    #     _I3D_CACHE[key] = None
    #     return None
    
    logger.info("[metrics] I3D FVD extractor ready (%s).", ckpt)
    _I3D_CACHE[key] = extractor
    return extractor


# ---- Heavy metrics ----
def heavy_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    value_range: ValueRange = _DEFAULT_RANGE,
    fvd_feature_extractor: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Dict[str, float]:
    """Compute the distribution-level metrics (FID always; FVD if a backbone is given)."""
    out: Dict[str, float] = {"fid": fid(pred, gt, value_range)}
    if fvd_feature_extractor is not None:
        out["fvd"] = fvd(pred, gt, value_range, feature_extractor=fvd_feature_extractor)
    return out
