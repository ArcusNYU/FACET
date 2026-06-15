#!/usr/bin/env bash
set -euo pipefail

# pip install yt-dlp
# winget install DenoLand.Deno      # deno --version
# winget install Gyan.FFmpeg        # ffmpeg -version
# (linux server) sudo apt-get install -y aria2 ffmpeg

CFG=./data/celebv/config.yaml

# ============================================================
# Stage 1: acquire (download + trim + bbox-crop) -> raw_root/clip/{cid}.mp4
# ============================================================
python data/celebv/pipeline/acquire.py \
    --limit 500 --pool 100 --workers 3
    # --proxy ... --cookies ...

# ============================================================
# Stage 2: per-clip preprocessing -> dataset_root/clip/{cid}/...
# ============================================================
# Single worker:
python -m data.celebv.pipeline.main --config "$CFG"
#
# Co-locate two workers on one GPU (split the clip set by md5(cid)%2):
# python -m data.celebv.pipeline.main --config "$CFG" --shard 0/2 &
# python -m data.celebv.pipeline.main --config "$CFG" --shard 1/2 &
# wait
#
# Re-attempt previously failed clips (e.g. after fixing a transient issue):
# python -m data.celebv.pipeline.main --config "$CFG" --retry-failed
#
# Stage 2 smoke test:
# python -m data.celebv.pipeline.main --config "$CFG" --limit 50

# ============================================================
# Stage 3: T5 caption + Wan VAE video latent cache -> dataset_root/latents/{cid}.pt
# ============================================================
# Single GPU:
python -m data.celebv.pipeline.cache --config "$CFG"
#
# Single A100 80GB, two co-resident workers (T5+VAE ~35GB each):
# python -m data.celebv.pipeline.cache --config "$CFG" --shard 0/2 &
# python -m data.celebv.pipeline.cache --config "$CFG" --shard 1/2 &
# wait
#
# Multi GPU (one shard per card):
# CUDA_VISIBLE_DEVICES=0 python -m data.celebv.pipeline.cache --config "$CFG" --shard 0/2 &
# CUDA_VISIBLE_DEVICES=1 python -m data.celebv.pipeline.cache --config "$CFG" --shard 1/2 &
# wait

# ============================================================
# Stage 4: train/val split -> data/celebv/splits/{train,val}.jsonl
# ============================================================
# python -m data.celebv.split --config "$CFG" --val-ratio 0.1
