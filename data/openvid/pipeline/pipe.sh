#!/usr/bin/env bash
set -euo pipefail

cd /nvmedata/workspace2/users/Arcus/Facet

CFG=./data/openvid/config.yaml

# Stage 1: caption single-person filter (writes <stem>.single.csv under cfg.out_root).
python -m data.openvid.pipeline.filters \
    --config "$CFG" \
    --provider cuda

# Stage 2: per-clip preprocessing -> dataset_root/clips/{part}/{ab}/{cd}/{cid}/...
python -m data.openvid.pipeline.main \
    --config "$CFG"

# Stage 2 smoke test (comment the full run above when iterating):
# python -m data.openvid.pipeline.main --config "$CFG" --limit 50

# Stage 3: T5 caption + Wan VAE video latent cache
#         -> dataset_root/latents/{part}/{ab}/{cd}/{cid}.pt
python -m data.openvid.pipeline.cache \
    --config "$CFG"

# Stage 3 smoke test:
# python -m data.openvid.pipeline.cache --config "$CFG" --limit 50
