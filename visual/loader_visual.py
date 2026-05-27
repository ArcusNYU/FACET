"""
Dataloader smoke test + visual dump.

Iterates the train loader for --n samples (default 20) and:
  - Prints clip_id / category / tensor shapes to terminal.
  - Dumps ref_img.png / masked_video.mp4 to ./visual_exampler/{cid}/.
  - Checks tgt_latent / t5_emb shape (shape only -- they are encoded).

build_loaders / collate_batch  -> loader.py
tensor helpers / write_mp4     -> utils.py

Run:
    cd /Facet
    python visual/loader_visual.py [--cfg data/config.yaml] [--n 20] [--out ./visual_exampler]
"""

from __future__ import annotations
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
from typing import Optional

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0") # for A100
import torch

from loader import build_loaders
from utils import image_to_uint8, video_to_uint8, write_mp4, write_png


def report_shape(name: str, t: Optional[torch.Tensor]) -> str:
    """Shape-only check for encoded tensors (latent / t5_emb)."""
    if t is None:
        return f"{name}: None  (cache miss -- run data/openvid/pipeline/cache.py)"
    if not torch.is_tensor(t):
        return f"{name}: not a tensor ({type(t).__name__})"
    return f"{name}: shape={tuple(t.shape)}  dtype={t.dtype}"


def main():
    p = argparse.ArgumentParser("dataloader smoke test + visual dump")
    p.add_argument("--cfg", default="data/config.yaml")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--out", default="./visual_exampler")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, train_sampler, val_sampler = build_loaders(
        cfg_path=args.cfg,
        batch_size=1,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    train_sampler.set_epoch(0)

    print(f"[loader] train concat={len(train_loader.dataset)}  quota/epoch={len(train_sampler)}")
    print(f"[loader] val   concat={len(val_loader.dataset)}    quota/epoch={len(val_sampler)}")
    print(f"[loader] dumping {args.n} samples -> {out_root.resolve()}\n")

    seen = 0
    for batch in train_loader:
        if seen >= args.n:
            break

        cid     = batch["clip_id"][0]
        cat     = batch["category"][0]
        masked  = batch["masked_video"][0]   # [T,3,H,W] in [-1,1]
        ref_img = batch["ref_img"][0]        # [3,H,W]   in [-1,1]
        tgt_lat = batch["tgt_latent"][0]     # Tensor or None
        t5_emb  = batch["t5_emb"][0]        # Tensor or None
        # source   = batch["source"][0]
        # path     = batch["path"][0]
        # caption  = batch["caption"][0]
        # mask     = batch["mask"][0]         # [T,1,H,W] in {0,1}
        # FIXME: loader更改后这里的内容获取也要添加mask的可视化&形状检查


        print(f"[{seen+1:02d}/{args.n}] clip_id      = {cid}")
        print(f"        category     = {cat}")
        print(f"        masked_video : {tuple(masked.shape)}"
              f"  min={masked.min().item():+.3f}  max={masked.max().item():+.3f}")
        print(f"        ref_img      : {tuple(ref_img.shape)}"
              f"  min={ref_img.min().item():+.3f}  max={ref_img.max().item():+.3f}")
        print(f"        {report_shape('tgt_latent', tgt_lat)}")
        print(f"        {report_shape('t5_emb    ', t5_emb)}")

        cdir = out_root / cid
        cdir.mkdir(parents=True, exist_ok=True)

        write_png(image_to_uint8(ref_img), cdir / "ref_img.png")
        write_mp4(video_to_uint8(masked),  cdir / "masked_video.mp4",
                  fps=args.fps, allow_fallback=True)

        summary = {
            "clip_id":  cid,
            "category": cat,
            # "source":   source,
            # "path":     path,
            # "caption":  caption,
            "shapes": {
                # "mask":         list(mask.shape),
                "masked_video": list(masked.shape),
                "ref_img":      list(ref_img.shape),
                "tgt_latent":   list(tgt_lat.shape) if torch.is_tensor(tgt_lat) else None,
                "t5_emb":       list(t5_emb.shape)  if torch.is_tensor(t5_emb)  else None,
            },
        }
        with open(cdir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"        -> {cdir}\n")
        seen += 1

    print(f"[loader] done. {seen} samples written to {out_root.resolve()}")


if __name__ == "__main__":
    main()
