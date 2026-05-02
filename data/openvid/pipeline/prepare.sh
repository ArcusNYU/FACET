#!/usr/bin/env bash
set -euo pipefail

cd /nvmedata/workspace2/users/Arcus/Facet

CFG=data/openvid/config.yaml

# Stage 1: caption single-person filter (writes <stem>.single.csv next to each input csv).
python -m data.openvid.pipeline.filters \
    --config "$CFG" \
    --provider cuda

# Stage 2: per-clip preprocessing -> dataset_root/clips/{part}/{ab}/{cd}/{cid}/...
python -m data.openvid.pipeline.prepare \
    --config "$CFG"
