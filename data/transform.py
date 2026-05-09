"""
Video / mask / ref-image transforms.

Keypoints:
- VideoTfm:  numpy [T,H,W,3] uint8 -> torch [T,3,H,W] float in [-1, 1]
             aspect-ratio preserving resize + black pad (handles portrait sources)
- MaskPerturb: numpy [T,H,W]  uint8  -> torch [T,1,H,W] float in {0,1}
               same perturb params shared across all T frames
- RefTfm:    numpy [H,W,3]  uint8  -> torch [3,H,W] float in [-1, 1]
- ref background augmentation lives in RefSampler, here only geometry + normalization
"""

from __future__ import annotations
import math
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _norm(t: torch.Tensor) -> torch.Tensor:
    """uint8 [0, 255] -> float [-1, 1].
       transform video from uint8 [0, 255] to float [-1, 1]
       '_' means no extra space for new variable, saving space for video processing
    """
    return t.float().div_(127.5).sub_(1.0)


def _fit_pad(x: torch.Tensor, th: int, tw: int, mode: str) -> torch.Tensor:
    """Aspect-ratio preserving resize to fit inside (th, tw), then center-pad with 0.

    Args:
        x:  [N, C, H, W] float tensor
        th, tw: target height / width
        mode: "bilinear" for video/ref, "nearest" for mask
    Returns:
        [N, C, th, tw] float tensor, 0-padded on the short side
    """
    _, _, h, w = x.shape
    if h == th and w == tw:
        return x
    # choose the larger scale that still fits inside the target box
    # resulting in padding instead of cropping  
    s = min(th / h, tw / w)  # larger side has smaller scaling factor
    nh, nw = int(round(h * s)), int(round(w * s))
    # avoid 0-sized edges from extreme ratios:
    nh = max(nh, 1)
    nw = max(nw, 1)
    if mode == "bilinear": # for video/ref  
        x = F.interpolate(x, (nh, nw), mode="bilinear", align_corners=False, antialias=True)
    else:                  # for mask
        x = F.interpolate(x, (nh, nw), mode="nearest")
    # center pad to (th, tw)
    pad_h = th - nh
    pad_w = tw - nw
    top = pad_h // 2
    bot = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    # F.pad order is (left, right, top, bottom)
    return F.pad(x, (left, right, top, bot), mode="constant", value=0.0)


class VideoTfm:
    """[T,H,W,3] uint8 -> [T,3,H,W] float in [-1,1].
       transform video from uint8 [0, 255] to float [-1, 1] & resize (aspect ratio preserved).
    """

    def __init__(self, h: int, w: int):
        """
        h: height -> target height
        w: width -> target width 
        h = 480, w = 832 for WAN2.2-TI2V:5B 480p resolution
        """
        self.h, self.w = h, w

    def __call__(self, arr: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()  # [T, H, W, 3] uint8 -> [T,3,H,W] uint8
        # F.interpolate requires contiguous tensor:
        if t.shape[-2] != self.h or t.shape[-1] != self.w:
            t = _fit_pad(t.float(), self.h, self.w, mode="bilinear")
        else:
            t = t.float()
        return _norm(t)


class MaskPerturb:
    """
    Perturbation on mask sequence boundary.

    Input: numpy [T,H,W] uint8 ({0,1} or {0,255}). -> mask
    Output: torch [T,1,H,W] float in {0,1}. -> perturbed mask

    Key: dilate / erode / elastic three perturbations are applied to all T frames,
         ensuring that the perturbation amplitude of the black area is continuous when the person walks far away / turns,
         without [flickering] between frames.
    """
    #NOTE: Mask Perturbation is for simulating user-drawn UI mask imprecision
    def __init__(
        # share the same parameters for all T frames
        self,
        h: int, w: int,
        p_dilate: float = 0.5, # probability of dilating the mask
        p_erode:  float = 0.5, # probability of eroding the mask
        dilate_range: Tuple[int, int] = (3, 7),  # max ~7px expansion
        erode_range:  Tuple[int, int] = (3, 5),  # max ~5px shrink
        p_elastic: float = 0.3, # probability of elastic deformation
        elastic_alpha: float = 8.0,   # amplitude -> small for mild distortion
        elastic_sigma: float = 12.0,  # smoothness -> large for smooth distortion
    ):
        self.h, self.w = h, w
        self.p_dilate, self.p_erode = p_dilate, p_erode
        self.dilate_range, self.erode_range = dilate_range, erode_range
        self.p_elastic = p_elastic
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma

    def __call__(self, arr: np.ndarray) -> torch.Tensor:
        m = torch.from_numpy(arr).float()
        if m.max() > 1.5:
            m = m.div_(255.0) # if uint8, normalize to [0, 1]
        m = m.unsqueeze(1)  # [T,H,W] -> [T,1,H,W]

        # FIXME: 本身mask就已经是在resize后的基础上进行扰动的 这里再resize一次是否合理? 似乎不需要了
        # 不过如果尺寸相同的话 也不会触发resize 所以先暂时保留
        if m.shape[-2] != self.h or m.shape[-1] != self.w:
            m = _fit_pad(m, self.h, self.w, mode="nearest")

        if random.random() < self.p_dilate:
            m = self._dilate(m, self._odd(random.randint(*self.dilate_range)))
        if random.random() < self.p_erode:
            m = self._erode(m, self._odd(random.randint(*self.erode_range)))
        if random.random() < self.p_elastic:
            m = self._warp(m, self._elastic_grid(self.h, self.w))

        return (m > 0.5).float()

    # augmentation functions: dilate, erode, elastic deformation
    @staticmethod
    def _odd(k: int) -> int:
        """ensure the kernel size is odd"""
        return k if k % 2 == 1 else k + 1

    @staticmethod
    def _dilate(m: torch.Tensor, k: int) -> torch.Tensor:
        """dilate the mask
           m: [T,1,H,W]
        """
        return F.max_pool2d(m, k, stride=1, padding=k // 2)

    @staticmethod
    def _erode(m: torch.Tensor, k: int) -> torch.Tensor:
        """erode the mask
           m: [T,1,H,W]
        """
        return 1.0 - F.max_pool2d(1.0 - m, k, stride=1, padding=k // 2)

    @staticmethod
    def _warp(m: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """warp the masks using the same grid for all T frames
           m: [T,1,H,W]; grid: [1,H,W,2] -> expand T frames using the same grid
        """
        T = m.shape[0]
        grid = grid.expand(T, -1, -1, -1) # [1,H,W,2] -> [T,H,W,2], so that all T frames use the same grid
        return F.grid_sample(m, grid, mode="nearest",
                             padding_mode="zeros", align_corners=False) #[T,1,H,W]
        # grid_sample: backward sampling, "nearest" mode: use the nearest integer value

    def _elastic_grid(self, h: int, w: int) -> torch.Tensor:
        """generate the elastic grid
           core process: generate noise -> apply gaussian filter -> obtain elastic deformation -> apply back to the mask
        """
        # elastic_alpha: control the amplitude of the elastic deformation
        # elastic_sigma: control the smoothness of the elastic deformation
        dx = self._gauss(torch.randn(h, w) * self.elastic_alpha, self.elastic_sigma)
        dy = self._gauss(torch.randn(h, w) * self.elastic_alpha, self.elastic_sigma)
        # pytorch grid_sample requires grid in [-1, 1]
        # ys, xs: mapping from [0, h] to [-1, 1] and [0, w] to [-1, 1]
        ys = torch.linspace(-1, 1, h).view(-1, 1).expand(h, w)
        xs = torch.linspace(-1, 1, w).view(1, -1).expand(h, w)
        grid = torch.stack([xs + 2 * dx / w, ys + 2 * dy / h], dim=-1) # [h,w,2]
        return grid.unsqueeze(0) # [h,w,2] -> [1,H,W,2]

    @staticmethod
    def _gauss(t: torch.Tensor, sigma: float) -> torch.Tensor:
        """apply the gaussian filter"""
        # a. generate the gaussian filter
        ks = int(2 * round(3 * sigma) + 1) # 3 * sigma of the gaussian filter
        # radius: 3 * sigma, kernel size: 2 * radius + 1, +1: ensure the kernel size is odd
        x = torch.arange(ks).float() - ks // 2
        g = torch.exp(-x * x / (2 * sigma * sigma)) # 1-dimensional gaussian filter
        g = g / g.sum() # normalize the gaussian filter
        kx = g.view(1, 1, 1, ks) # horizontal gaussian filter
        ky = g.view(1, 1, ks, 1) # vertical gaussian filter
        # b. apply the gaussian filter
        # t is a noise map generated by torch.randn(h, w)
        t4 = t.view(1, 1, *t.shape) # [H,W] -> [1,1,H,W], since [F.conv2d requires 4D tensor]
        t4 = F.conv2d(t4, kx, padding=(0, ks // 2))
        t4 = F.conv2d(t4, ky, padding=(ks // 2, 0))
        return t4.view(t.shape) # [1,1,H,W] -> [H,W]


class RefTfm:
    """[H,W,3] uint8 -> [3,H,W] float in [-1,1]. 
       default size: 480x480 square.
       input is expected to be RGB (alpha already consumed upstream by RefSampler).
    """

    def __init__(self, size: int = 480):
        self.size = size

    def __call__(self, arr: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float()  # [H,W,3] -> [3,H,W]
        if t.shape[-1] != self.size or t.shape[-2] != self.size:
            t = _fit_pad(t.unsqueeze(0), self.size, self.size, mode="bilinear").squeeze(0)
        return _norm(t)


class TfmBundle:
    """bundle video / mask / ref three transforms, inject into BaseVideoDataset."""

    def __init__(self, video: VideoTfm, mask: MaskPerturb, ref: RefTfm):
        self.video = video
        self.mask = mask
        self.ref = ref

    @classmethod
    def from_cfg(cls, cfg, shared) -> "TfmBundle":
        """
        Args:
            cfg:    per-dataset cfg (data/{name}/config.yaml DotDict).
                    Reads height / width / ref_size.
            shared: top-level cfg (data/config.yaml DotDict).
                    Reads mask_perturb block (cross-dataset shared).
        """
        h, w = cfg.height, cfg.width
        mp = shared["mask_perturb"]
        return cls(
            video=VideoTfm(h, w),
            mask=MaskPerturb(
                h, w,
                p_dilate=mp["p_dilate"], p_erode=mp["p_erode"],
                dilate_range=tuple(mp["dilate_range"]),
                erode_range=tuple(mp["erode_range"]),
                p_elastic=mp["p_elastic"],
                elastic_alpha=mp["elastic_alpha"],
                elastic_sigma=mp["elastic_sigma"],
            ),
            ref=RefTfm(size=cfg.ref_size),
        )
