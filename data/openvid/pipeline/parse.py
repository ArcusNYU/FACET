"""
Dataset Pipeline Stage 2 Step 1 (per-clip): SCHP human parsing for a video clip.
Reference: https://github.com/GoGoDuck912/Self-Correction-Human-Parsing

API:
    parser = SchpParser(weight_path, device, batch_size=B)
    parsing = parser.parse_video(video_rgb)              # [T,H,W] uint8 class ids in [0..19]
    mask    = SchpParser.select(parsing, keep_ids)       # [T,H,W] uint8 in {0,1}
    mask    = SchpParser.smooth(mask, k=3)               # temporal majority filter

Notes:
- Input video frames are RGB (decord default), uint8, [T,H,W,3].
- Internally we use cv2.warpAffine equivalently to schp.py (LIPDataValSet style):
  pad to square aspect and resize to 473x473 -> infer -> inverse-warp to (H,W).
  instead of using resizing directly.
- ROI Representation:
  OpenPose / HRNet / SCHP uses ROI: center + scale (per 200 pixels)
  https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/software-and-datasets/mpii-human-pose-dataset
- Inference uses bf16 autocast; on A100 80GB B=81 (full clip) fits comfortably for
  ResNet101 + 473x473, and B is configurable via cfg.prepare.schp_batch.
- LIP class table:
    0 Background  1 Hat   2 Hair  3 Glove  4 Sunglasses  5 Upper-clothes
    6 Dress       7 Coat  8 Socks 9 Pants 10 Jumpsuits 11 Scarf
   12 Skirt      13 Face 14 Left-arm 15 Right-arm 16 Left-leg 17 Right-leg
   18 Left-shoe  19 Right-shoe
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


_INPUT_SIZE = (473, 473)   # fixed for SCHP model
_NUM_CLASSES = 20          # fixed for SCHP model
_ARCH = "resnet101"
# _FLIP_PAIRS = [(14, 15), (16, 17), (18, 19)]


# ---- affine helpers (verbatim from schp.py LIPDataValSet style) ----
# reference: https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/blob/master/lib/utils/transforms.py

def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Construct an auxiliary third point so that (a, b, c) form a right triangle with the right angle at b.
    [Important!] So that the affine can be restricted to Similarity Transformation, without having diagonal deformation.
    Similarity Transformation: transformation that is only composed of rotation, translation, and scaling.
    cv2.getAffineTransform requires 3 non-collinear point pairs to uniquely determine a 2x3 affine matrix.
    """
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def _get_direct(src_point, rot_rad: float) -> np.ndarray:
    """
    Rotate a 2D vector around the origin by `rot_rad` radians.
    Args:
        src_point: A 2D point/vector (length-2 iterable) in image coordinates. 
        rot_rad: Rotation angle in radians.
    """
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    return np.array([
        src_point[0] * cs - src_point[1] * sn,
        src_point[0] * sn + src_point[1] * cs,
    ], dtype=np.float32)


def _get_affine(center, scale, rot, output_size, inv: bool = False) -> np.ndarray:
    """
    Build a 2x3 affine matrix that maps an ROI in the original image to a fixed-size square network input (or its inverse).
    Pick three corresponding point pairs in (src_image, dst_image):
        point 0: ROI center           <-> output image center
        point 1: ROI center + "up"    <-> output center + "up"
        point 2: a perpendicular kick (see `_get_3rd_point`)
    Args:
        center: (cx, cy) of the ROI in source-image pixel coords.
        scale:  Either a scalar or a length-2 array (sx, sy) in the HRNet/MPII "200-pixel unit". 
        NOTE: only scale[0] (the width) is actually used to size the triangle;
        this assumes scale[0]/scale[1] already matches the output aspect ratio (which `_box2cs` enforces).
        rot: Rotation augmentation in DEGREES. 0 at inference time.
        output_size: (H_out, W_out) of the target square network input, e.g. (473, 473).
        inv: If True, return the inverse matrix that maps dst -> src.
    """
    if not isinstance(scale, (np.ndarray, list, tuple)):
        scale = np.array([scale, scale], dtype=np.float32)
    scale_tmp = np.asarray(scale, dtype=np.float32) * 200.0  # actual scale pixels
    src_w = scale_tmp[0]  # width of the ROI in pixels
    dst_h, dst_w = output_size  # height and width of the output image (473, 473)

    rot_rad = np.pi * rot / 180.0  # convert rotation angle to radians
    src_direct = _get_direct([0.0, src_w * -0.5], rot_rad)  # direction vector of the ROI
    dst_direct = np.array([0.0, (dst_w - 1) * -0.5], dtype=np.float32)  # direction vector of the output image
    # -1: index rounding correction, for example the center of (473, 473) is (236, 236) instead of (236.5, 236.5)

    src = np.zeros((3, 2), dtype=np.float32) 
    dst = np.zeros((3, 2), dtype=np.float32)  
    src[0] = center               # affine point 1: ROI center
    src[1] = center + src_direct  # affine point 2: ROI center upward
    dst[0] = [(dst_w - 1) * 0.5, (dst_h - 1) * 0.5]  # affine target point 1: ROI center
    dst[1] = dst[0] + dst_direct             # affine target point 2: ROI center upward
    src[2] = _get_3rd_point(src[0], src[1])  # affine source point 3: perpendicular point
    dst[2] = _get_3rd_point(dst[0], dst[1])  # affine target point 3: perpendicular point

    if inv:
        return cv2.getAffineTransform(dst, src)
    return cv2.getAffineTransform(src, dst)


def _box2cs(w: int, h: int, aspect_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert bounding box (w, h)to ROI representation (center, scale)
    e.g.: (832, 480) -> center=(416, 240), scale=(4.16, 4.16)
    """
    center = np.array([w * 0.5, h * 0.5], dtype=np.float32)
    # would rather square pad than crop:
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
    return center, scale


# ---- public class ----
class SchpParser:
    """Stateful SCHP wrapper."""

    def __init__(
        self,
        weight_path: str | Path,
        device: torch.device | str = "cuda",
        batch_size: int = 81,
        amp_dtype: str = "float16",
    ):
        # SCHP/networks lives at the repo root; importing here avoids work_root sys.path issues
        # when this module is imported by something that doesn't need SCHP.
        from SCHP.networks import init_model

        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        _dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16":  torch.float16,
            "float32":  torch.float32,
        }
        requested = _dtype_map[amp_dtype]
        # BFloat16 requires Ampere (sm_80+); fall back to float16 if not supported.
        if requested == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            print("[SCHP] BFloat16 not supported on this GPU, falling back to float16")
            requested = torch.float16
        self.amp_dtype = requested

        model = init_model(_ARCH, num_classes=_NUM_CLASSES, pretrained=None)
        ckpt = torch.load(str(weight_path), map_location="cpu", weights_only=False)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        new_state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
        model.load_state_dict(new_state, strict=True)
        model.to(self.device).eval()
        self.model = model
        # SCHP exposes mean/std/input_space on the model object
        self.mean = np.asarray(model.mean, dtype=np.float32)
        self.std  = np.asarray(model.std,  dtype=np.float32)
        self.input_space = getattr(model, "input_space", "BGR") #BGR

    # ---- main API ----
    @torch.no_grad()
    def parse_video(self, video_rgb: np.ndarray) -> np.ndarray:
        """
        Args:
            video_rgb: [T,H,W,3] uint8 (RGB; decord default).
        Returns:
            [T,H,W] uint8, per-pixel LIP class id in [0..19].
        """
        if video_rgb.dtype != np.uint8 or video_rgb.ndim != 4 or video_rgb.shape[-1] != 3:
            raise ValueError(f"video_rgb must be [T,H,W,3] uint8, got {video_rgb.shape} {video_rgb.dtype}")
        T, H, W, _ = video_rgb.shape

        center, scale = _box2cs(W, H, _INPUT_SIZE[1] / _INPUT_SIZE[0])
        trans = _get_affine(center, scale, 0, _INPUT_SIZE)
        inv_trans = _get_affine(center, scale, 0, _INPUT_SIZE, inv=True)

        # warp all frames to 473x473:
        warped = np.empty((T, _INPUT_SIZE[0], _INPUT_SIZE[1], 3), dtype=np.uint8)
        for i in range(T):
            warped[i] = cv2.warpAffine(
                video_rgb[i], trans, (_INPUT_SIZE[1], _INPUT_SIZE[0]),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
        # SCHP weights requires input_space=="BGR": 
        if self.input_space == "BGR":
            warped = warped[..., ::-1]

        # Normalize -> NCHW float
        x = warped.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))  # [T,3,473,473]

        # Inference, sub-batched
        argmax_chunks: List[np.ndarray] = []
        for s in range(0, T, self.batch_size):
            xb = x[s:s + self.batch_size].to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype != torch.float32):
                logits = self.model(xb)
                # SCHP returns nested lists; fusion branch is parsing[0][-1]
                fusion = logits[0][-1]                                  # [B, 20, h, w]
                fusion = F.interpolate(fusion, size=_INPUT_SIZE,
                                       mode="bilinear", align_corners=True)
            argmax_chunks.append(fusion.float().argmax(dim=1).to(torch.uint8).cpu().numpy())  # [B,473,473]

        parsing_473 = np.concatenate(argmax_chunks, axis=0)            # [T,473,473] uint8

        # Inverse warp back to (H, W) using flags=NEAREST so class ids stay integer
        out = np.empty((T, H, W), dtype=np.uint8)
        for i in range(T):
            out[i] = cv2.warpAffine(
                parsing_473[i], inv_trans, (W, H),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
        return out

    # ---- mask helpers ----
    @staticmethod
    def select(parsing: np.ndarray, keep_ids: Iterable[int]) -> np.ndarray:
        """Merge selected LIP class ids into a single binary mask. Returns uint8 in {0,1}."""
        keep = np.asarray(list(keep_ids), dtype=np.int64)
        return np.isin(parsing, keep).astype(np.uint8)

    @staticmethod
    def smooth(mask: np.ndarray, k: int = 3) -> np.ndarray:
        """Temporal majority-vote filter along T axis with a window of k.
        Reduces per-frame SCHP flicker on mask boundary."""
        if k <= 1:
            return mask
        T = mask.shape[0]
        if T == 0:
            return mask
        r = k // 2 # half window size
        # e.g. k=3, r=1, so for frame k, consider frames [k-1, k, k+1]
        # pad replicate along T
        idx = np.clip(np.arange(T)[:, None] + np.arange(-r, r + 1)[None, :], 0, T - 1)  # [T, k]
        # np.clip: clip the values to the range [0, T-1] using replicate padding
        # e.g.: [-1, 0, 1] -> clip to -> [0, 0, 1]
        window = mask[idx]                                                                  # [T,k,H,W]
        # majority vote: sum > k/2 -> True
        s = window.sum(axis=1)
        return (s > (k // 2)).astype(np.uint8) # True/False
