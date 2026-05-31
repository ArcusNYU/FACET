"""
Ref image sampling with on-the-fly background augmentation in dataloader __getitem__ function.

ref_pool: main.py already stored N 480x480 RGBA png, each is tight bbox + 20% pad + square pad.
          foreground bbox alpha=255, padding region alpha=0.
pick():   randomly sample 1 image from the pool, then fill the alpha=0 padding
          region with one of:
            (1) a fixed gray=127 solid       (p_solid branch)
            (2) a contrast-aware random gray (p_random branch)
            (3) a low-frequency RGB noise    (p_distract branch -- DISABLED)

Why bg augmentation at all:
   ref crops come from the same clip's later frames, so the *natural*
   surrounding pixels overlap with target-video context. Keeping them would
   leak that context into the model as a shortcut. We fill the padding with a
   neutral / contrast-aware color instead.
"""
# NOTE: prepare.py reference image bbox crop pad ratio currently 20%.

from __future__ import annotations
import random

import numpy as np
from PIL import Image


# 127 is mid-gray in uint8 -- the same neutral colour used to fill the masked
# region of src_video, so the network treats "padded-out background" identically
# whether it appears in the ref crop or in the masked source frames.
GRAY_127 = 127


class RefSampler:
    """
    Pick one reference image from the pool and fill its alpha=0 padding.

    Three probability branches (mutually exclusive, must sum to <= 1.0):
        [0,                       p_solid)               -> constant gray=127
        [p_solid,        p_solid + p_random)             -> contrast-aware random gray
        [p_solid+p_random, p_solid+p_random+p_distract)  -> low-freq RGB noise (DISABLED)
        anything beyond                                   -> no-op (raw RGB returned)

    Output: numpy [H,W,3] uint8 (alpha consumed and fused into RGB).
    """

    def __init__(
        self,
        p_solid:   float = 1.0,        # constant gray=127 padding
        p_random:  float = 0.0,        # contrast-aware random gray padding
        p_distract: float = 0.0,       # low-freq RGB noise padding (disabled by default)
        solid_min_dist: int = 60,      # min |gray - fg_gray| on [0,255] luminance
        solid_max_tries: int = 20,     # rejection-sampling cap before extreme fallback
    ):
        s = p_solid + p_random + p_distract
        if not (0.0 <= s <= 1.0 + 1e-3):
            raise ValueError(
                f"require p_solid + p_random + p_distract <= 1.0, got {s:.3f}"
            )
        self.p_solid = float(p_solid)
        self.p_random = float(p_random)
        self.p_distract = float(p_distract)
        self.solid_min_dist = int(solid_min_dist)
        self.solid_max_tries = int(solid_max_tries)

    def pick(self, ref_pool, mask_seq=None) -> np.ndarray:
        """
        Args:
            ref_pool: List[Path|str], RGBA png path list
            mask_seq: kept for interface compat, currently unused
        Returns:
            numpy [H,W,3] uint8
        """
        if not ref_pool:
            raise ValueError("Empty ref_pool")
        path = random.choice(ref_pool)
        rgba = np.asarray(Image.open(path).convert("RGBA"))   # [H,W,4]
        rgb, alpha = rgba[..., :3], rgba[..., 3]
        bg = (alpha == 0)                                     # background bool mask
        return self._aug(rgb, bg)

    # ---- branching --------------------------------------------------------
    def _aug(self, rgb: np.ndarray, bg: np.ndarray) -> np.ndarray:
        """random-branch padding fill. raw RGB if no padding."""
        if not bg.any():
            return rgb
        r = random.random()
        if r < self.p_solid:
            return self._fill_const_gray(rgb, bg, GRAY_127)
        if r < self.p_solid + self.p_random:
            return self._fill_contrast_gray(rgb, bg)
        # if r < self.p_solid + self.p_random + self.p_distract:
        #     return self._fill_distract(rgb, bg)
        return rgb

    # ---- branch implementations -------------------------------------------
    @staticmethod
    def _fill_const_gray(rgb: np.ndarray, bg: np.ndarray, value: int) -> np.ndarray:
        """Constant grayscale `value` on the bg region. value=127 matches the
        masked-region fill in src_video so the model sees one neutral colour."""
        out = rgb.copy()
        out[bg] = np.array([value, value, value], dtype=np.uint8)
        return out

    def _fill_contrast_gray(self, rgb: np.ndarray, bg: np.ndarray) -> np.ndarray:
        """Rejection-sample a grayscale that contrasts the foreground median."""
        v = self._pick_contrast_gray(rgb, bg)
        # out = rgb.copy()
        # out[bg] = np.array([v, v, v], dtype=np.uint8)
        return self._fill_const_gray(rgb, bg, v)

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

    # ---- helpers ----------------------------------------------------------
    @staticmethod
    def _fg_gray(rgb: np.ndarray, bg: np.ndarray) -> float:
        """Median luminance (BT.601 Y) of the foreground (alpha != 0) region."""
        fg = ~bg
        if not fg.any():
            return 128.0
        fg_rgb = np.median(rgb[fg].astype(np.float32), axis=0)
        return float(0.299 * fg_rgb[0] + 0.587 * fg_rgb[1] + 0.114 * fg_rgb[2])

    def _pick_contrast_gray(self, rgb: np.ndarray, bg: np.ndarray) -> int:
        """Sample a grayscale at least solid_min_dist away from fg luminance.
        Falls back to the farthest extreme if rejection sampling exhausts."""
        fg_gray = self._fg_gray(rgb, bg)
        for _ in range(self.solid_max_tries):
            v = random.randint(0, 255)
            if abs(v - fg_gray) >= self.solid_min_dist:
                return v
        return 0 if fg_gray > 127 else 255

    # ---- factory ----------------------------------------------------------
    @classmethod
    def from_cfg(cls, shared) -> "RefSampler":
        """ref_aug is dataset-agnostic, read from data/config.yaml shared block."""
        a = shared["ref_aug"]
        return cls(
            p_solid=a.get("p_solid", 1.0),
            p_random=a.get("p_random", 0.0),
            p_distract=a.get("p_distract", 0.0),
        )
