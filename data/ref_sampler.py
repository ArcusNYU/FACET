"""
Ref image sampling with on-the-fly background augmentation in dataloader __getitem__ function.

ref_pool: prepare.py already stored N 480x480 RGBA png, each is tight bbox + 10% pad + square pad.
          foreground bbox alpha=255, padding region alpha=0.
pick():   randomly sample 1 image from ref_pool, then fill the padding region with
          solid color (p=0.5) / low-frequeny distract texture (p=0.5).
          purpose: make the model robust to different backgrounds at inference time
                   while AVOIDING two failure modes:
                   (a) keeping the original surrounding background -> shortcut bias
                       (model would learn that the ref's surrounding context appears in tgt video,
                       since here ref is sampled from the target video itself)
                   (b) blindly filling a solid color that happens to match the garment color
                       (e.g. black pad on a black hoodie -> indistinct boundary at bbox edge)

Why RGBA (not jpg):
    the foreground garment may happen to be pure black/white/gray,
    so alpha-channel based bg detection is strictly safer than color-threshold guessing.
"""
# NOTE: prepare.py reference image bbox crop pad ratio currently 10%.

from __future__ import annotations
import random
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
from PIL import Image


# 5-step grayscale palette: black, dark gray, mid gray, light gray, white.
# FIXME: 只要不是相近的都可以随机选择 不一定选取距离最远的 只把距离最近的排除掉就可以了 
# TODO: ...
_DEFAULT_SOLID = (
    (0, 0, 0),
    (64, 64, 64),
    (128, 128, 128),
    (192, 192, 192),
    (255, 255, 255),
)


class RefSampler:
    """
    Two probability branches:
        p < p_solid                  -> contrast-aware solid color in padding region
        p_solid <= p < 1             -> low-frequency distract texture in padding region

    Output: numpy [H,W,3] uint8 (alpha consumed and fused into RGB).
    """

    def __init__(
        self,
        p_solid: float = 0.5,
        p_distract: float = 0.5,
        solid_colors: Sequence = _DEFAULT_SOLID,
    ):
        s = p_solid + p_distract
        if abs(s - 1.0) > 1e-3:
            # non-strict probability sum limit currently
            pass
        self.p_solid = p_solid
        self.p_distract = p_distract
        self.solid_colors = np.array(solid_colors, dtype=np.float32)  # [K,3] for vectorized distance

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

    def _fg_median(self, rgb: np.ndarray, bg: np.ndarray) -> np.ndarray:
        """Median RGB over the foreground (alpha != 0) region; shape [3], float32."""
        fg = ~bg
        if not fg.any():
            return np.array([128.0, 128.0, 128.0], dtype=np.float32)
        return np.median(rgb[fg].astype(np.float32), axis=0)

    @staticmethod
    def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
        """Approximate sRGB -> CIE Lab (D65). Input [...,3] in [0,255], output same shape in Lab.
           Implemented numpy-only to avoid a colormath/skimage dependency.
        """
        x = rgb.astype(np.float32) / 255.0
        # sRGB gamma decompression
        x = np.where(x > 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)
        # sRGB -> XYZ (D65)
        m = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ], dtype=np.float32)
        xyz = x @ m.T
        # XYZ -> Lab (reference white D65)
        ref = np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
        f = xyz / ref
        f = np.where(f > 0.008856, np.cbrt(f), 7.787 * f + 16.0 / 116.0)
        L = 116.0 * f[..., 1] - 16.0
        a = 500.0 * (f[..., 0] - f[..., 1])
        b = 200.0 * (f[..., 1] - f[..., 2])
        return np.stack([L, a, b], axis=-1)

    def _pick_contrast_solid(self, rgb: np.ndarray, bg: np.ndarray) -> np.ndarray:
        """Pick the palette entry whose Lab distance to fg median color is largest."""
        fg_rgb = self._fg_median(rgb, bg)             # [3]
        # Lab distance is perceptually uniform; max distance == best contrast.
        fg_lab = self._rgb_to_lab(fg_rgb[None, :])[0]                   # [3]
        cand_lab = self._rgb_to_lab(self.solid_colors[None, :, :])[0]   # [K,3]
        d = np.linalg.norm(cand_lab - fg_lab[None, :], axis=-1)         # [K]
        return self.solid_colors[int(d.argmax())].astype(np.uint8)

    def _fill_solid(self, rgb: np.ndarray, bg: np.ndarray) -> np.ndarray:
        c = self._pick_contrast_solid(rgb, bg)
        out = rgb.copy()
        out[bg] = c
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
            p_solid=a["p_solid"],
            p_distract=a["p_distract"],
        )
