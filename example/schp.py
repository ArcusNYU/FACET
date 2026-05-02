#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCHP MVP local inference.

参考: https://github.com/GoGoDuck912/Self-Correction-Human-Parsing/blob/master/evaluate.py
预处理与原仓库 LIPDataValSet 对齐: center/scale + affine warp 到 473x473,
推理后再用逆 affine 把结果映射回原图.

LIP 20 类:
    0  Background       1  Hat            2  Hair           3  Glove
    4  Sunglasses       5  Upper-clothes  6  Dress          7  Coat
    8  Socks            9  Pants         10  Jumpsuits     11  Scarf
   12  Skirt           13  Face          14  Left-arm      15  Right-arm
   16  Left-leg        17  Right-leg     18  Left-shoe     19  Right-shoe

输出:
    {stem}_mask.png      二值 mask (0/255), 仅包含 --classes 指定的类别
    {stem}_overlay.png   原图 + mask 高亮叠加 (肉眼验证用)
"""

import os
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from SCHP.networks import init_model


LIP_CLASSES = [
    "Background", "Hat", "Hair", "Glove", "Sunglasses",
    "Upper-clothes", "Dress", "Coat", "Socks", "Pants",
    "Jumpsuits", "Scarf", "Skirt", "Face", "Left-arm",
    "Right-arm", "Left-leg", "Right-leg", "Left-shoe", "Right-shoe",
]
NUM_CLASSES = 20
INPUT_SIZE = (473, 473)  # h, w
ARCH = "resnet101"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# LIP 翻转后需要左右互换的类别索引
FLIP_PAIRS = [(14, 15), (16, 17), (18, 19)]


def parse_args():
    p = argparse.ArgumentParser("SCHP MVP local inference")
    p.add_argument("--input", required=True, help="image path or directory")
    p.add_argument("--output-dir", default="./schp_results")
    p.add_argument("--model-restore", default="./weights/SCHP/schp.pth")
    p.add_argument("--gpu", default="0", help="gpu id, or 'None' for cpu")
    p.add_argument(
        "--classes", default="5,7",
        help="comma-separated class ids to keep, default 5,7 = Upper-clothes + Coat",
    )
    p.add_argument("--flip", action="store_true", help="enable horizontal flip TTA")
    p.add_argument("--recursive", action="store_true")
    return p.parse_args()


# ---------- 预处理: center/scale + affine warp (对齐 LIPDataValSet) ----------
def _get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def _get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    return np.array([
        src_point[0] * cs - src_point[1] * sn,
        src_point[0] * sn + src_point[1] * cs,
    ], dtype=np.float32)


def get_affine_transform(center, scale, rot, output_size, inv=False):
    """与原仓库 utils/transforms.py::get_affine_transform 等价的最小实现."""
    if not isinstance(scale, (np.ndarray, list, tuple)):
        scale = np.array([scale, scale], dtype=np.float32)
    scale_tmp = np.asarray(scale, dtype=np.float32) * 200.0
    src_w = scale_tmp[0]
    dst_h, dst_w = output_size

    rot_rad = np.pi * rot / 180.0
    src_dir = _get_dir([0.0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0.0, (dst_w - 1) * -0.5], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0] = center
    src[1] = center + src_dir
    dst[0] = [(dst_w - 1) * 0.5, (dst_h - 1) * 0.5]
    dst[1] = dst[0] + dst_dir
    src[2] = _get_3rd_point(src[0], src[1])
    dst[2] = _get_3rd_point(dst[0], dst[1])

    if inv:
        return cv2.getAffineTransform(dst, src)
    return cv2.getAffineTransform(src, dst)


def box2cs(w, h, aspect_ratio):
    """整张图作为 box, 转成 LIPDataValSet 风格的 center/scale."""
    center = np.array([w * 0.5, h * 0.5], dtype=np.float32)
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
    return center, scale


def preprocess(img_bgr, mean, std, input_space):
    """BGR -> affine warp 473x473 -> ToTensor -> (BGR2RGB if needed) -> Normalize."""
    h, w = img_bgr.shape[:2]
    aspect_ratio = INPUT_SIZE[1] * 1.0 / INPUT_SIZE[0]
    center, scale = box2cs(w, h, aspect_ratio)

    trans = get_affine_transform(center, scale, 0, INPUT_SIZE)
    warped = cv2.warpAffine(
        img_bgr, trans, (INPUT_SIZE[1], INPUT_SIZE[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    x = warped.astype(np.float32) / 255.0
    if input_space == "RGB":
        x = x[:, :, ::-1].copy()  # 等价 BGR2RGB_transform
    x = (x - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    x = torch.from_numpy(x.transpose(2, 0, 1)).float()

    meta = {"center": center, "scale": scale, "h": h, "w": w}
    return x, meta


# ---------- 推理 (multi_scale_testing 的 scale=1 简化版) ----------
def infer_logits(model, img_tensor, flip=False):
    """返回 [num_classes, H, W] (H=W=473)."""
    x = img_tensor.unsqueeze(0)  # [1, C, H, W]
    if flip:
        x = torch.cat([x, torch.flip(x, dims=[-1])], dim=0)  # [2, C, H, W]

    parsing = model(x)
    parsing = parsing[0][-1]  # 取 fusion 分支: [N, 20, h, w]

    out = parsing[0]  # [20, h, w]
    if flip:
        flipped = parsing[1].clone()
        for a, b in FLIP_PAIRS:
            flipped[[a, b]] = flipped[[b, a]]
        flipped = torch.flip(flipped, dims=[-1])
        out = (out + flipped) * 0.5

    out = F.interpolate(
        out.unsqueeze(0), size=INPUT_SIZE,
        mode="bilinear", align_corners=True,
    )[0]
    return out


def restore_to_original(parsing_473, center, scale, h, w):
    """逆 affine, 把 473x473 的预测映射回原图 hxw."""
    trans = get_affine_transform(center, scale, 0, INPUT_SIZE, inv=True)
    return cv2.warpAffine(
        parsing_473.astype(np.uint8), trans, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


# ---------- 可视化 ----------
def overlay_mask(img_bgr, mask01, color_bgr=(0, 255, 0), alpha=0.5):
    """只在 mask 区域上半透明染色, 非 mask 区域保持原图亮度."""
    out = img_bgr.copy()
    region = mask01 > 0
    if region.any():
        c = np.array(color_bgr, dtype=np.float32)
        layer = out[region].astype(np.float32)
        out[region] = ((1 - alpha) * layer + alpha * c).astype(np.uint8)
    return out


# ---------- 模型 ----------
def load_model(weight_path, device):
    model = init_model(ARCH, num_classes=NUM_CLASSES, pretrained=None)
    ckpt = torch.load(weight_path, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    new_state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(new_state, strict=True)
    model.to(device).eval()
    return model


# ---------- 文件 ----------
def collect_inputs(path, recursive):
    p = Path(path)
    if p.is_file():
        return [p]
    it = p.rglob("*") if recursive else p.iterdir()
    return sorted([f for f in it if f.suffix.lower() in IMG_EXTS])


# ---------- 主流程 ----------
def main():
    args = parse_args()

    if args.gpu != "None":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    keep_ids = sorted({int(x) for x in args.classes.split(",") if x.strip()})
    keep_names = [LIP_CLASSES[i] if 0 <= i < NUM_CLASSES else str(i) for i in keep_ids]

    paths = collect_inputs(args.input, args.recursive)
    if not paths:
        raise RuntimeError(f"No images found: {args.input}")
    print(f"[schp] found {len(paths)} image(s); device={device}; flip={args.flip}")
    print(f"[schp] keep classes={keep_ids} ({keep_names})")

    model = load_model(args.model_restore, device)
    mean, std, input_space = model.mean, model.std, model.input_space
    print(f"[schp] image_mean={mean}, image_std={std}, input_space={input_space}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for img_path in paths:
            img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                print(f"[skip] failed to read: {img_path}")
                continue

            x, meta = preprocess(img_bgr, mean, std, input_space)
            x = x.to(device)

            logits = infer_logits(model, x, flip=args.flip)            # [20, 473, 473]
            parsing_473 = torch.argmax(logits, dim=0).cpu().numpy()    # [473, 473]
            parsing_full = restore_to_original(
                parsing_473, meta["center"], meta["scale"], meta["h"], meta["w"],
            )

            keep_mask = np.isin(parsing_full, np.asarray(keep_ids, dtype=np.int64)).astype(np.uint8)
            kept = int(keep_mask.sum())
            total = int(keep_mask.size)
            ratio = kept / max(total, 1)

            stem = img_path.stem
            mask_path = out_dir / f"{stem}_mask.png"
            overlay_path = out_dir / f"{stem}_overlay.png"

            cv2.imwrite(str(mask_path), keep_mask * 255)
            cv2.imwrite(str(overlay_path), overlay_mask(img_bgr, keep_mask))

            uniq = np.unique(parsing_full).tolist()
            print(
                f"[{stem}] unique={uniq}, kept={kept}/{total} ({ratio:.4%}) "
                f"-> {mask_path.name}, {overlay_path.name}"
            )


if __name__ == "__main__":
    main()
