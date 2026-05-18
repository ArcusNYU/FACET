"""
Ref image sampling with on-the-fly background augmentation in dataloader __getitem__ function.

ref_pool: prepare.py already stored N 480x480 RGBA png, each is tight bbox + 20% pad + square pad.
          foreground bbox alpha=255, padding region alpha=0.
pick():   randomly sample 1 image from ref_pool, then fill the padding region with
          either a contrast-aware solid grayscale or a low-frequency distract texture.
          purpose: make the model robust to different backgrounds at inference time
                   while AVOIDING two failure modes:
                   (a) keeping the original surrounding background -> shortcut bias
                       (here ref is sampled from the same clip's later frames, so the
                        surrounding context overlaps with tgt video context)
                   (b) filling a fixed-palette solid that happens to match the garment
                       (e.g. black pad on a black hoodie -> indistinct boundary).

Why RGBA (not jpg):
    the foreground garment may happen to be pure black/white/gray,
    so alpha-channel based bg detection is strictly safer than color-threshold guessing.
"""
# NOTE: prepare.py reference image bbox crop pad ratio currently 20%.

from __future__ import annotations
import random
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


class RefSampler:
    """
    Pick one reference image from the pool and implement random augmentation on the padding region.
    Two probability branches:
        p < p_solid                  -> contrast-aware random grayscale in padding region
        p_solid <= p < 1             -> low-frequency distract texture in padding region

    Output: numpy [H,W,3] uint8 (alpha consumed and fused into RGB).
    """

    def __init__(
        self,
        p_solid: float = 1.0,
        p_distract: float = 0.0,
        solid_min_dist: int = 60,      # min |candidate_gray - fg_gray| on [0,255] luminance 
        solid_max_tries: int = 20,     # rejection-sampling cap before falling back to extreme
    ):
        s = p_solid + p_distract
        if abs(s - 1.0) > 1e-3:
            # non-strict probability sum limit currently
            pass
        self.p_solid = p_solid
        self.p_distract = p_distract
        self.solid_min_dist = int(solid_min_dist)
        self.solid_max_tries = int(solid_max_tries)

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
        """random augmentation on padding region: solid (contrast-aware) or distract texture."""
        # If the image has no padding (bbox already filled the canvas), return as-is.
        if not bg.any():
            return rgb
        if random.random() < self.p_solid:
            return self._fill_solid(rgb, bg)
        return self._fill_distract(rgb, bg)

    @staticmethod
    def _fg_gray(rgb: np.ndarray, bg: np.ndarray) -> float:
        """Median luminance (BT.601 Y) over the foreground (alpha != 0) region."""
        fg = ~bg  # obtain the foreground region by inverting the background mask
        if not fg.any():
            return 128.0
        fg_rgb = np.median(rgb[fg].astype(np.float32), axis=0) # median is robust to outliers
        return float(0.299 * fg_rgb[0] + 0.587 * fg_rgb[1] + 0.114 * fg_rgb[2])
        # BT.601 Y standard formula: Y = 0.299 * R + 0.587 * G + 0.114 * B

    def _pick_contrast_gray(self, rgb: np.ndarray, bg: np.ndarray) -> int:
        """Rejection-sample a grayscale value in [0, 255] that is at least
        solid_min_dist away from the foreground median luminance.
        Falls back to the farthest extreme if sampling exhausts.
        """
        fg_gray = self._fg_gray(rgb, bg)
        for _ in range(self.solid_max_tries):
            v = random.randint(0, 255)
            if abs(v - fg_gray) >= self.solid_min_dist:
                return v
        return 0 if fg_gray > 127 else 255

    def _fill_solid(self, rgb: np.ndarray, bg: np.ndarray) -> np.ndarray:
        """fill the padding region with a contrast-aware solid grayscale."""
        v = self._pick_contrast_gray(rgb, bg)
        out = rgb.copy()
        out[bg] = np.array([v, v, v], dtype=np.uint8)
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
    def from_cfg(cls, shared) -> "RefSampler":
        """ref_aug is dataset-agnostic, read from data/config.yaml shared block."""
        a = shared["ref_aug"]
        return cls(
            p_solid=a["p_solid"],
            p_distract=a["p_distract"],
        )
