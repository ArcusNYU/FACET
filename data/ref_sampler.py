"""
Ref image sampling with on-the-fly background augmentation in dataloader __getitem__ function.

ref_pool: prepare.py already stored N 480x480 jpg, each is tight bbox + pad + square pad.
pick():   randomly sample 1 image from ref_pool, then fill the background with solid color / distractors according to the probability.
          purpose: make the model robust to different background in inference time.
"""
# FIXME: prepare.py 在处理reference image的时候 当通过mask的bbox进行crop时 缩放+pad的程度不用到20%那么多 10%左右

from __future__ import annotations
import random
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


class RefSampler:
    def __init__(
        self,
        p_keep: float = 0.4,
        p_bg: float = 0.3,
        p_distract: float = 0.3,
        solid_colors: Sequence = ((0, 0, 0), (128, 128, 128), (255, 255, 255)), # black, gray, white
        bg_thresh: int = 16, # threshold to determine if the pixel is background
        # TODO: solid_colors 是否可以添加其他种类的颜色
    ):
    """
    p < p_keep: pass
    p_keep < p < p_keep + p_solid: fill solid color
    p_keep + p_solid < p < p_keep + p_solid + p_distract: fill distractors
    """
        s = p_keep + p_solid + p_distract
        if abs(s - 1.0) > 1e-3:
            # non-strict probability sum limit currently
            pass
        self.p_keep = p_keep
        self.p_solid = p_solid
        self.p_distract = p_distract
        self.solid_colors = list(solid_colors)
        self.bg_thresh = bg_thresh

    def pick(self, ref_pool, mask_seq=None) -> np.ndarray:
        """
        Args:
            ref_pool:  List[Path|str], jpg/png path list
            mask_seq:  keep interface
        Returns:
            numpy [H,W,3] uint8
        """
        if not ref_pool:
            raise ValueError("Empty ref_pool")
        path = random.choice(ref_pool)
        img = np.asarray(Image.open(path).convert("RGB"))
        # pick a random candidate from ref_pool and use random augmentation
        return self._aug(img)

    def _aug(self, img: np.ndarray) -> np.ndarray:
        """random augmentation: keep / background / distractors"""
        r = random.random()
        if r < self.p_keep:
            return img
        bg = (img.sum(axis=-1) < self.bg_thresh) # obtain background boolean mask
        # .sum(aixs=-1): [H, W, 3] -> [H, W] add up the three channels
        # bg_thresh >5: Prevent black color from not strictly being (0,0,0) due to JPEG compression, etc.
        if not bg.any():
            return img
        if r < self.p_keep + self.p_solid:
            return self._fill_solid(img, bg)
        return self._fill_distract(img, bg)

    # def _obtain_bg(self, img: np.ndarray) -> np.ndarray:
    #     """obtain the background mask"""
    #     return img.sum(axis=-1) < self.bg_thresh

    def _fill_solid(self, img: np.ndarray, bg: np.ndarray) -> np.ndarray:
        """
        fill the background with solid color
        """
        c = random.choice(self.solid_colors)
        out = img.copy()
        out[bg] = np.array(c, dtype=np.uint8)
        return out

    def _fill_distract(self, img: np.ndarray, bg: np.ndarray) -> np.ndarray:
        """
        fill the background with distractors - low frequency texture
        """
        h, w = img.shape[:2]
        rng = np.random.default_rng()
        small = rng.integers(0, 256, size=(max(h // 8, 1), max(w // 8, 1), 3),
                             dtype=np.uint8) # generate a smaller texture first 
        tex = np.asarray(Image.fromarray(small).resize((w, h), Image.BILINEAR))
        out = img.copy()
        out[bg] = tex[bg]
        return out

    @classmethod
    def from_cfg(cls, cfg) -> "RefSampler":
        a = cfg.ref_aug
        return cls(
            p_keep=a["p_keep"],
            p_solid=a["p_solid"],
            p_distract=a["p_distract"],
        )
