"""
Video / mask / ref-image transforms.

Keypoints:
- VideoTfm:  numpy [T,H,W,3] uint8 -> torch [T,3,H,W] float in [-1, 1]
- MaskPerturb: numpy [T,H,W]  uint8  -> torch [T,1,H,W] float in {0,1}
- RefTfm:    numpy [H,W,3]  uint8  -> torch [3,H,W] float in [-1, 1]
- ref background enhancement is placed in RefSampler, here only does geometry + normalization
"""

from __future__ import annotations
import math
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
# TODO: 写一个constants.py在work_root 即 Facet文件夹里面
# 里面放一些针对视频的长宽值以及帧数 fps 等 例如默认视频需要被normalized到h=480 w=832 
# 然后对于reference image 默认normalized到h=480 w=480 square 符合OmniControl的风格
# 另外 transform需要的超参数 就放置在 data文件夹中 因为好像处理mask和video的transform是共享的

def _norm(t: torch.Tensor) -> torch.Tensor:
    """uint8 [0, 255] -> float [-1, 1].
       transform video from uint8 [0, 255] to float [-1, 1]
       '_' means no extra space for new variable, saving space for video processing
    """
    return t.float().div_(127.5).sub_(1.0)


class VideoTfm:
    """[T,H,W,3] uint8 -> [T,3,H,W] float in [-1,1].
       transform video from uint8 [0, 255] to float [-1, 1] & resize
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
            t = F.interpolate(t.float(), size=(self.h, self.w),
                              mode="bilinear", align_corners=False, antialias=True)
        # FIXME: 替换掉F.interpolate 因为视频resize的过程中要尽量保持长宽比 然后再pad到target size 
        # 否则内容产生严重形变
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
    def __init__(
        # share the same parameters for all T frames
        self,
        h: int, w: int,
        p_dilate: float = 0.5, # probability of dilating the mask
        p_erode:  float = 0.5, # probability of eroding the mask
        dilate_range: Tuple[int, int] = (3, 11), # range of dilate kernel size
        erode_range:  Tuple[int, int] = (3, 7), # range of erode kernel size
        p_elastic: float = 0.3, # probability of elastic deformation
        elastic_alpha: float = 30.0, # alpha parameter for elastic deformation
        elastic_sigma: float = 6.0, # sigma parameter for elastic deformation
        # FIXME: 超参数的选择是否合理未知 
        # 需要设置相对较小的值 略微对mask进行扰动 所以需要cursor预设一个较小的值
        # 所以alpha需要小 sigma偏大 产生轻微的形变就可以了 要不然mask形变太离谱了
        # 试想一个场景 用户在使用我的模型想要更换一个视频中人物的衣物 它在UI界面通过涂抹选择了区域
        # 这个区域不会很离谱 但肯定不会完全与衣物的轮廓接口 所以这里的mask perturbation就是在模拟这种情况
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
            m = F.interpolate(m, size=(self.h, self.w), mode="nearest")

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
       default size: 480x480 square
    """

    def __init__(self, size: int = 480):
        self.size = size

    def __call__(self, arr: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float()  # [H,W,3] -> [3,H,W]
        if t.shape[-1] != self.size or t.shape[-2] != self.size:
            t = F.interpolate(t.unsqueeze(0), size=(self.size, self.size),
                              mode="bilinear", align_corners=False).squeeze(0)
        return _norm(t)


class TfmBundle:
    """bundle video / mask / ref three transforms, inject into BaseVideoDataset."""

    def __init__(self, video: VideoTfm, mask: MaskPerturb, ref: RefTfm):
        self.video = video
        self.mask = mask
        self.ref = ref

    @classmethod
    def from_cfg(cls, cfg) -> "TfmBundle":
        h, w = cfg.height, cfg.width
        mp = cfg.mask_perturb
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
