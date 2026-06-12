"""
debug.py — smoke test for the heavy-metrics path used at train.py:334-335.

Exercises, on RANDOM videos (no step_dir / valid.heavy_eval needed yet):
    metrics.heavy_metrics(pred, gt, fvd_dir=..., fid_dir=...)   # FID (+ FVD)
    -> fid() builds the local InceptionV3 from fid_dir, fvd() builds the local
       TorchScript I3D from fvd_dir (both cached); either dir may be None.

The I3D is loaded LOCALLY via torch.jit.load from --fvd_dir (default weights/I3D),
never from the HuggingFace hub. Drop flateon/FVD-I3D-torchscript's
`i3d_torchscript.pt` into weights/I3D/ first.

Each stage is isolated + wrapped in try/except so you can see exactly which call
(extractor build / direct features / FID / FVD / heavy_metrics) errors, if any.

Run:
    python debug.py
    python debug.py --device cpu --n 16 --frames 16 --hw 64
"""

from __future__ import annotations

# Mirror train.py's offline lock: nothing should ever hit the network.
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import argparse
import logging
import sys
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

import metrics


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FVD/FID heavy-metrics smoke test.")
    p.add_argument("--fvd_dir", default=str(_ROOT / "weights" / "I3D"),
                   help="local dir holding i3d_torchscript.pt (offline).")
    p.add_argument("--inception_dir", default=str(_ROOT / "weights" / "INC"),
                   help="local dir holding the InceptionV3-FID .pth (offline).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n", type=int, default=2, help="number of clips (N).")
    p.add_argument("--frames", type=int, default=16, help="frames per clip (T).")
    p.add_argument("--hw", type=int, default=64, help="frame height = width.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level="INFO", format="%(levelname)s %(name)s | %(message)s")
    device = torch.device(args.device)
    print(f"[debug] device={device}  fvd_dir={args.fvd_dir}  inception_dir={args.inception_dir}")

    # Random pred/gt videos: [N, T, 3, H, W] float in [-1, 1] (the project's range).
    gen = torch.Generator().manual_seed(0)
    shape = (args.n, args.frames, 3, args.hw, args.hw)
    pred = (torch.rand(shape, generator=gen) * 2.0 - 1.0).to(device)
    gt = (torch.rand(shape, generator=gen) * 2.0 - 1.0).to(device)
    print(f"[debug] pred/gt shape={tuple(pred.shape)} "
          f"range=[{pred.min():.2f}, {pred.max():.2f}]")

    # ---- 1. build the local TorchScript I3D extractor -----------------------
    #        (private helper; fvd() now builds + caches this internally)
    print("\n[1] metrics._build_fvd_extractor(...)")
    fvd_ext = None
    try:
        fvd_ext = metrics._build_fvd_extractor(args.fvd_dir, device=device)
    except Exception:
        traceback.print_exc()
    print(f"    -> {'OK (callable)' if fvd_ext is not None else 'None (FVD will be skipped)'}")

    # ---- 2. direct feature extraction (shape sanity) ------------------------
    if fvd_ext is not None:
        print("\n[2] fvd_ext(gt) feature shape")
        try:
            feats = fvd_ext(gt)
            print(f"    -> features={tuple(feats.shape)}  (expect [{args.n}, 400])")
        except Exception:
            traceback.print_exc()

    # ---- 3. FID alone (local InceptionV3 weights) ---------------------------
    print("\n[3] metrics.fid(pred, gt, fid_dir=...)")
    try:
        print(f"    -> fid={metrics.fid(pred, gt, fid_dir=args.inception_dir):.4f}")
    except Exception:
        traceback.print_exc()

    # ---- 4. FVD alone (I3D + Fréchet, built from fvd_dir) -------------------
    print("\n[4] metrics.fvd(pred, gt, fvd_dir=...)")
    try:
        print(f"    -> fvd={metrics.fvd(pred, gt, fvd_dir=args.fvd_dir)}")
    except Exception:
        traceback.print_exc()

    # ---- 5. heavy_metrics (the actual train.py call) ------------------------
    print("\n[5] metrics.heavy_metrics(pred, gt, fvd_dir=..., fid_dir=...)")
    try:
        heavy = metrics.heavy_metrics(
            pred, gt,
            fvd_dir=args.fvd_dir,
            fid_dir=args.inception_dir,
        )
        print(f"    -> {heavy}")
    except Exception:
        traceback.print_exc()

    print("\n[debug] done.")


if __name__ == "__main__":
    main()
