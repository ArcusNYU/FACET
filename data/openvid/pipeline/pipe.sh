#!/usr/bin/env bash
set -euo pipefail

cd /nvmedata/workspace2/users/Arcus/Facet

CFG=./data/openvid/config.yaml

# Stage 1: caption single-person filter (writes <stem>.single.csv under cfg.out_root).
python -m data.openvid.pipeline.filters \
    --config "$CFG" \
    --provider cuda

# Stage 2: per-clip preprocessing -> dataset_root/clips/{part}/{ab}/{cd}/{cid}/...
# main.py only processes one part at a time. On a single 80GB A100 card, SCHP+IQA+VLM consumes only ~25GB,
# allowing two parts to be run in parallel to fully utilize the card 
# (index.jsonl is append-only, atomic line writes, shared safely).
python -m data.openvid.pipeline.main --config "$CFG" --part 1 &
python -m data.openvid.pipeline.main --config "$CFG" --part 2 &
wait

# Stage 2 smoke test (comment the full run above when iterating):
# python -m data.openvid.pipeline.main --config "$CFG" --part 1 --limit 50

# Stage 3: T5 caption + Wan VAE video latent cache
#         -> dataset_root/latents/{part}/{ab}/{cd}/{cid}.pt
python -m data.openvid.pipeline.cache \
    --config "$CFG"

# Stage 3 smoke test:
# python -m data.openvid.pipeline.cache --config "$CFG" --limit 50
