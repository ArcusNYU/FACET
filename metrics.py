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
    "light_metrics", "heavy_metrics",
    "psnr_mask", "ssim_mask", "lpips_mask", "light_metrics_mask",
]

ValueRange = Tuple[float, float]
_DEFAULT_RANGE: ValueRange = (-1.0, 1.0)

# FIXME: 添加 light_metrics_bg? 做法依然是套壳psnr/ssim/lpips 但是在valid.py&ckpt.py中将light_metrics改为使用light_metrics_bg
# TODO: 在light_metrics中加入temporal proxy - tLPIPS / boundary proxy (/ subject consistency proxy)
# 即视频任务必须添加 目前还不确定针对 temporal proxy 是选择 rLPIPS_err还是选择 temporal_l1 需要选择两者中相对更好的
# boundary proxy目前还不清楚具体做法 但可用的为 LPIPS_boundary / L1_boundary / gradient_boundary_err
# TODO: 在heavy_metrics中加入VBench / VACE-like metrics


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
# masked light metrics — PSNR / SSIM / LPIPS restricted to the EDIT region
# =============================================================================
# Convention: pred/gt are [T, C, H, W] (a single clip); mask is [T, 1, H, W] or
# [T, H, W] in {0, 1}, PIXEL-aligned with pred/gt. A leading batch dim of size 1
# is accepted and squeezed.
# NOTE: 在针对mask区域的轻量指标测评中 由于mask及其bbox区域不断变动 所以无法组成batch tensor stack
_SSIM_MIN_SIDE = 16   # SSIM gaussian window is 11x11 -> crop must exceed it
_LPIPS_MIN_SIDE = 32  # AlexNet has several 2x downsamples -> keep crops sane
_MASK_MIN_SIDE = max(_SSIM_MIN_SIDE, _LPIPS_MIN_SIDE)  # one shared crop for all 3


def _as_clip_tchw(x: torch.Tensor) -> torch.Tensor:
    """[T,C,H,W] passes through (4D == single clip, batch 1); [1,T,C,H,W] -> [T,C,H,W].
    """
    if x.dim() == 5 and x.shape[0] == 1:
        x = x[0]
    if x.dim() != 4:
        raise ValueError(f"expected [T,C,H,W] or [1,T,C,H,W]; got {tuple(x.shape)}")
    return x


def _as_mask_thw(mask: torch.Tensor) -> torch.Tensor:
    """[T,1,H,W] / [1,T,H,W] / [T,H,W] -> [T,H,W] float in {0,1}."""
    m = mask
    if m.dim() == 5 and m.shape[0] == 1:
        m = m[0]                      # [T,1,H,W] or [1,T,H,W]?  -> handled below
    if m.dim() == 4:
        if m.shape[1] == 1:           # [T,1,H,W]
            m = m[:, 0]
        elif m.shape[0] == 1:         # [1,T,H,W]
            m = m[0]
        else:
            raise ValueError(f"ambiguous mask shape {tuple(mask.shape)}")
    if m.dim() != 3:
        raise ValueError(f"expected mask [T,1,H,W] or [T,H,W]; got {tuple(mask.shape)}")
    return (m.float() > 0.5).float()


def _frame_bbox(
    m2d: torch.Tensor, h: int, w: int, min_side: int, pad: int,
) -> Optional[Tuple[int, int, int, int]]:
    """Tight bbox (y0,y1,x0,x1) of a 2D binary mask, padded + grown to min_side.

    Returns None for an empty mask. The box is symmetric-grown to at least
    `min_side` on each axis and clamped to the frame so the downstream SSIM
    window / LPIPS backbone always receive a large-enough crop.
    """
    ys, xs = torch.where(m2d > 0.5)
    if ys.numel() == 0:
        return None
    y0 = int(ys.min().item()) - pad
    y1 = int(ys.max().item()) + 1 + pad
    x0 = int(xs.min().item()) - pad
    x1 = int(xs.max().item()) + 1 + pad

    def _grow(a: int, b: int, hi: int) -> Tuple[int, int]:
        if b - a < min_side:
            c = (a + b) / 2.0
            a = int(round(c - min_side / 2.0))
            b = a + min_side
        a = max(0, a)
        b = min(hi, b)
        if b - a < min_side:          # frame smaller than min_side: take full extent
            a, b = 0, hi
        return a, b

    y0, y1 = _grow(y0, y1, h)
    x0, x1 = _grow(x0, x1, w)
    return y0, y1, x0, x1


def _masked_metric(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    metric_fn: Callable[..., float],
    value_range: ValueRange,
    **fn_kwargs,
) -> Optional[float]:
    """Average `metric_fn` over per-frame mask-bbox crops. None if no valid frame."""
    p = _as_clip_tchw(pred)
    g = _as_clip_tchw(gt)
    if p.shape != g.shape:
        raise ValueError(f"pred/gt shape mismatch: {tuple(p.shape)} vs {tuple(g.shape)}")
    m = _as_mask_thw(mask)
    T = min(p.shape[0], m.shape[0]) # frame dim match
    H, W = p.shape[-2], p.shape[-1]

    vals: list[float] = []
    for t in range(T):
        bb = _frame_bbox(m[t], H, W, _MASK_MIN_SIDE, pad=0)
        if bb is None:
            continue
        y0, y1, x0, x1 = bb
        cp = p[t : t + 1, :, y0:y1, x0:x1].contiguous()   # [1,C,h,w]
        cg = g[t : t + 1, :, y0:y1, x0:x1].contiguous()
        vals.append(metric_fn(cp, cg, value_range, **fn_kwargs))
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def psnr_mask(pred, gt, mask, value_range: ValueRange = _DEFAULT_RANGE) -> Optional[float]:
    """Mean per-frame PSNR over the mask bbox (higher = better)."""
    return _masked_metric(pred, gt, mask, psnr, value_range)


def ssim_mask(pred, gt, mask, value_range: ValueRange = _DEFAULT_RANGE) -> Optional[float]:
    """Mean per-frame SSIM over the mask bbox (higher = better)."""
    return _masked_metric(pred, gt, mask, ssim, value_range)


def lpips_mask(
    pred, gt, mask, value_range: ValueRange = _DEFAULT_RANGE, net: str = "alex",
) -> Optional[float]:
    """Mean per-frame LPIPS over the mask bbox (lower = better)."""
    return _masked_metric(pred, gt, mask, lpips, value_range, net=net)


def light_metrics_mask(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    value_range: ValueRange = _DEFAULT_RANGE,
    lpips_net: str = "alex",
) -> Dict[str, float]:
    """Edit-region reconstruction metrics (bbox-cropped); drops any that are None."""
    out = {
        "psnr_mask": psnr_mask(pred, gt, mask, value_range),
        "ssim_mask": ssim_mask(pred, gt, mask, value_range),
        "lpips_mask": lpips_mask(pred, gt, mask, value_range, net=lpips_net),
    }
    return {k: v for k, v in out.items() if v is not None}


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


# ---- InceptionV3 feature extractor (FID backbone) ----
_FID_INCEPTION_CACHE: Dict[Tuple[str, str], Optional["torch.nn.Module"]] = {}


class _FidInception(torch.nn.Module):
    """
    Adapter so a LOCAL torch_fidelity InceptionV3 survives torchmetrics' FID init.

    torch_fidelity's net strictly requires a *uint8* tensor on its own device and
    raises otherwise. This wrapper coerces device+dtype, so the probe succeeds.
    During the real FID.update() the input is already uint8 on-device, so this is
    a no-op there.
    """

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dev = next(self.inner.parameters()).device
        if x.device != dev:
            x = x.to(dev)
        if x.dtype != torch.uint8:
            x = x.clamp(0, 255).to(torch.uint8)
        return self.inner(x)


def _build_fid_inception(fid_dir: str | Path, device: str) -> Optional["torch.nn.Module"]:
    """
    Build (and cache) a NoTrainInceptionV3 feature extractor from local weights.
    """
    key = (str(fid_dir), str(device))
    if key in _FID_INCEPTION_CACHE:
        return _FID_INCEPTION_CACHE[key]

    # resolve the checkpoint file:
    p = Path(fid_dir)
    ckpt: Optional[Path] = None
    if p.is_file():
        ckpt = p
    elif p.is_dir():
        preferred = p / "weights-inception-2015-12-05-6726825d.pth"
        if preferred.exists():
            ckpt = preferred
        else:
            cands = sorted([*p.glob("*.pth"), *p.glob("*.pt")])
            ckpt = cands[0] if cands else None

    if ckpt is None:
        logger.warning(
            "[metrics] no InceptionV3 checkpoint under %s; FID falls back to the "
            "torchmetrics default (online download).", fid_dir,
        )
        _FID_INCEPTION_CACHE[key] = None
        return None

    try:
        from torchmetrics.image.fid import NoTrainInceptionV3

        inner = NoTrainInceptionV3(
            name="inception-v3-compat",
            features_list=["2048"],
            feature_extractor_weights_path=str(ckpt),
        ).to(device).eval()
        inception = _FidInception(inner).to(device).eval()
    except Exception as e:  # noqa: BLE001 - a bad file must not crash the run
        logger.warning("[metrics] failed to load InceptionV3 '%s' (%s); using default.", ckpt, e)
        _FID_INCEPTION_CACHE[key] = None
        return None

    logger.info("[metrics] InceptionV3 FID extractor ready (%s).", ckpt)
    _FID_INCEPTION_CACHE[key] = inception
    return inception


# ---- FID ----
def fid(
    pred: torch.Tensor,
    gt: torch.Tensor,
    value_range: ValueRange = _DEFAULT_RANGE,
    fid_dir: str | Path | None = None,
) -> float:
    """
    Frame-level Fréchet Inception Distance (treats every frame as an image).

    Lower is better. Backed by torchmetrics' InceptionV3 feature extractor.

    `fid_dir` points at the local InceptionV3-FID checkpoint (file or directory,
    e.g. cfg.paths.inception_dir). When None, torchmetrics' default feature
    extractor is used (which would download weights online).
    """
    from torchmetrics.image.fid import FrechetInceptionDistance

    p, g = _prep_pair(pred, gt, value_range)  # [N, 3, H, W] in [0, 1]

    feature: object = 2048
    if fid_dir is not None:
        inception = _build_fid_inception(fid_dir, str(p.device))
        if inception is not None:
            feature = inception

    metric = FrechetInceptionDistance(feature=feature, normalize=True).to(p.device)
    metric.update(g, real=True)
    metric.update(p, real=False)
    # real: whether the data is ground truth or generated
    return float(metric.compute())


# ---- FVD ----
def fvd(
    pred: torch.Tensor,
    gt: torch.Tensor,
    value_range: ValueRange = _DEFAULT_RANGE,
    fvd_dir: str | Path | None = None,
) -> Optional[float]:
    """
    Clip-level Fréchet Video Distance, lower is better.

    `fvd_dir` points at the local I3D checkpoint (file or directory,
    e.g. cfg.paths.fvd_dir). Returns None when fvd_dir is None or no usable I3D
    backbone is found, so callers can simply skip reporting FVD.

    FVD = Fréchet distance between I3D features of the real vs generated clips
    (each clip: [B, T, C, H, W] in `value_range`).
    """
    if fvd_dir is None:
        return None

    pred = _as_5d(pred)
    gt = _as_5d(gt)
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch: {tuple(pred.shape)} vs {tuple(gt.shape)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = _build_fvd_extractor(fvd_dir, device=device)
    if extractor is None:
        return None

    feat_real = extractor(_to_pm1(gt, value_range))
    feat_fake = extractor(_to_pm1(pred, value_range))
    return _frechet_distance(feat_real, feat_fake)


# ---- I3D feature extractor (FVD backbone) ----
_I3D_CACHE: Dict[Tuple[str, str], Optional[Callable[[torch.Tensor], torch.Tensor]]] = {}

def _build_fvd_extractor(
    weights_dir: str | Path | None,
    device: str | torch.device = "cpu",
):
    """
    Build a Kinetics-400 I3D feature extractor for FVD, loaded LOCALLY (offline).
    Args:
        weights_dir : local dir (or file) holding the I3D checkpoint (or None).
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
    if weights_dir is None:
        return None

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
    fvd_dir: str | Path | None = None,
    fid_dir: str | Path | None = None,
) -> Dict[str, float]:
    """
    Distribution-level metrics.

    FID uses `fid_dir` (local InceptionV3 weights); FVD uses `fvd_dir` (local I3D
    weights). Each backbone is resolved + cached inside fid()/fvd(); either dir
    may be None — FID then falls back to the torchmetrics default, and FVD is
    dropped from the result when its backbone is unavailable.
    """
    out: Dict[str, Optional[float]] = {
        "fid": fid(pred, gt, value_range, fid_dir=fid_dir),
        "fvd": fvd(pred, gt, value_range, fvd_dir=fvd_dir),
    }
    return {k: v for k, v in out.items() if v is not None}
