#!/usr/bin/env bash
set -euo pipefail

# pip install yt-dlp
# winget install DenoLand.Deno      # deno --version
# winget install Gyan.FFmpeg        # ffmpeg -version
# (linux server) sudo apt-get install -y aria2 ffmpeg

CFG=./data/celebv/config.yaml

# ============================================================
# Stage 1: filter (hair-color-balanced subset) -> candidate.json
# ============================================================
# Reads the full celebvhq_info.json ONCE and writes a balanced 10k-clip manifest.
# Run once; re-run with --force to regenerate (changes which clips get downloaded).
python -m data.celebv.pipeline.filter \
    # --info data/celebv/pipeline/celebvhq_info.json \
    # --out data/celebv/pipeline/candidate.json \
    # --total 10000 --seed 42

# ============================================================
# Stage 2: acquire (download + trim + bbox-crop) -> raw_root/clip/{cid}.mp4
# ============================================================
python -m data.celebv.pipeline.acquire \
    --limit 500 --pool 100 --workers 3 \
    # --info data/celebv/pipeline/candidate.json \
    # --proxy ... --cookies ...

# ============================================================
# Stage 3: per-clip preprocessing -> dataset_root/clip/{cid}/...
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
# Stage 3 smoke test (verbose ref-selection + caption preview, writes no outputs):
# python -m data.celebv.pipeline.main_test --config "$CFG" --limit 50

# ============================================================
# Stage 4: T5 caption + Wan VAE video latent cache -> dataset_root/latents/{cid}.pt
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
# Stage 5: train/val split (leakage-safe group-by-ytb) -> data/celebv/splits/{train,val}.jsonl
# ============================================================
python -m data.celebv.split --config "$CFG" --val-ratio 0.1
