"""
Ref image sampling with on-the-fly background augmentation in dataloader __getitem__ function.

ref_pool: prepare.py already stored N 480x480 RGBA png, each is tight bbox + 10% pad + square pad.
          foreground bbox alpha=255, padding region alpha=0.
pick():   randomly sample 1 image from ref_pool, then fill the padding region
          with solid color / distractors according to the probability.
          purpose: make the model robust to different background in inference time.

Why RGBA (not jpg):
    the foreground garment may happen to be pure black/white/gray,
    so alpha-channel based bg detection is strictly safer than color-threshold guessing.
"""
# NOTE: prepare.py reference image 的 bbox crop pad ratio 目前定为 10%.

from __future__ import annotations
import random
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


class RefSampler:
    """
    Probability branches:
        p < p_keep                             -> pass through (keep padding as-is)
        p_keep <= p < p_keep + p_solid         -> fill padding with a random solid color
        p_keep + p_solid <= p < 1              -> fill padding with low-frequency distract texture

    Output: numpy [H,W,3] uint8 (alpha has been consumed and fused into RGB).
    """

    def __init__(
        self,
        p_keep: float = 0.4,
        p_solid: float = 0.3,
        p_distract: float = 0.3,
        solid_colors: Sequence = (
            (0, 0, 0), (64, 64, 64), (128, 128, 128),
            (192, 192, 192), (255, 255, 255),
        ),  #FIXME: 这些都是什么
    ):
        s = p_keep + p_solid + p_distract
        if abs(s - 1.0) > 1e-3:
            # non-strict probability sum limit currently
            pass
        self.p_keep = p_keep
        self.p_solid = p_solid
        self.p_distract = p_distract
        self.solid_colors = list(solid_colors)

    def pick(self, ref_pool, mask_seq=None) -> np.ndarray:
        """
        Args:
            ref_pool:  List[Path|str], RGBA png path list
            mask_seq:  keep interface, currently unused
        Returns:
            numpy [H,W,3] uint8
        """
        if not ref_pool:
            raise ValueError("Empty ref_pool")
        path = random.choice(ref_pool)
        rgba = np.asarray(Image.open(path).convert("RGBA"))   # [H,W,4]
        rgb, alpha = rgba[..., :3], rgba[..., 3]
        bg = (alpha == 0)                                     # [H,W] background boolean mask
        return self._aug(rgb, bg)

    def _aug(self, rgb: np.ndarray, bg: np.ndarray) -> np.ndarray:
        """random augmentation on padding region: keep / solid / distractors."""
        # FIXME: 如果padding的对象衣服恰好为灰色或者黑色 那么依然存在形状边界出现错误的问题
        r = random.random()
        if r < self.p_keep:
            # [Important!] fill padding with 0 (black) if we pass through, so the downstream
            # tensor still has a deterministic value rather than whatever the
            # PNG encoder left under the transparent pixels
            if bg.any():
                out = rgb.copy()
                out[bg] = 0
                return out
            return rgb
        if not bg.any():
            return rgb
        if r < self.p_keep + self.p_solid:
            return self._fill_solid(rgb, bg)
        return self._fill_distract(rgb, bg)

    def _fill_solid(self, rgb: np.ndarray, bg: np.ndarray) -> np.ndarray:
        """fill the padding region with a random solid color."""
        c = random.choice(self.solid_colors)
        out = rgb.copy()
        out[bg] = np.array(c, dtype=np.uint8)
        return out

    def _fill_distract(self, rgb: np.ndarray, bg: np.ndarray) -> np.ndarray:
        """fill the padding region with low-frequency noise texture."""
        h, w = rgb.shape[:2]
        rng = np.random.default_rng()
        small = rng.integers(0, 256, size=(max(h // 8, 1), max(w // 8, 1), 3),
                             dtype=np.uint8)   # generate a smaller texture first
        tex = np.asarray(Image.fromarray(small).resize((w, h), Image.BILINEAR))
        out = rgb.copy()
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
