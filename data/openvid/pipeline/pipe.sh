#!/usr/bin/env bash
set -euo pipefail

cd /nvmedata/workspace2/users/Arcus/Facet

CFG=./data/openvid/config.yaml

# Stage 1: caption single-person filter (writes <stem>.single.csv under cfg.out_root).
python -m data.openvid.pipeline.filters \
    --config "$CFG" \
    --provider cuda

# Stage 2: per-clip preprocessing -> dataset_root/clips/{part}/{ab}/{cd}/{cid}/...
# main.py processes one part at a time. SCHP+IQA+VLM is ~25GB / proc, so a single
# 80GB A100 fits 2-3 workers. index.jsonl is append-protected by fcntl.flock so
# concurrent appends are safe even on NFS/Lustre.
#
# Layout A: Single GPU - two workers on different parts
python -m data.openvid.pipeline.main --config "$CFG" --part 1 &
python -m data.openvid.pipeline.main --config "$CFG" --part 2 &
wait
#
# Layout B: Single GPU - co-locate two workers on the same part
# python -m data.openvid.pipeline.main --config "$CFG" --part 1 --shard 0/2 &
# python -m data.openvid.pipeline.main --config "$CFG" --part 1 --shard 1/2 &
# wait

# Stage 2 smoke test:
# python -m data.openvid.pipeline.main_test --config "$CFG" --part 1 --limit 50

# Stage 3: T5 caption + Wan VAE video latent cache
#         -> dataset_root/latents/{part}/{ab}/{cd}/{cid}.pt
# Single GPU:
python -m data.openvid.pipeline.cache --config "$CFG"
# T5+VAE ~35GB each
# Single A100 80GB with two co-resident workers:
# CUDA_VISIBLE_DEVICES=0 python -m data.openvid.pipeline.cache --shard 0/2 &
# CUDA_VISIBLE_DEVICES=0 python -m data.openvid.pipeline.cache --shard 1/2 &
# wait
# 
# Multi GPU (one shard per card):
# CUDA_VISIBLE_DEVICES=0 python -m data.openvid.pipeline.cache --shard 0/2 &
# CUDA_VISIBLE_DEVICES=1 python -m data.openvid.pipeline.cache --shard 1/2 &
# wait

# Stage 3 smoke test:
# python -m data.openvid.pipeline.cache --config "$CFG" --limit 50
