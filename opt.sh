#!/usr/bin/env bash
# TODO: AI复查完善此脚本

# == env preparation ========================================================
git clone https://github.com/ArcusNYU/FACET.git && cd FACET
pip install -r requirements.txt
# git clone https://github.com/facebookresearch/DiffSynth.git / pip install diffsynth
pip install -U "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B \
    --local-dir ./weights/WAN2.2 \
    --local-dir-use-symlinks False
# TODO: FACET LoRA权重的huggingface下载

# --- environment (edit to match your machine) -------------------------------
# conda activate torch
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1


# == launch gradio demo =====================================================



# == launch model training ===================================================
# GPU selection + python/accelerate choice all come from train.yaml (accel.*).
python launch_train.py --train_yaml train.yaml

# --- handy variants ---------------------------------------------------------
# Inspect the assembled command without running it:
#   python launch_train.py --dry-run
# Force multi-gpu DDP:
#   python launch_train.py --launcher accelerate
# Quick single-process run:
#   python launch_train.py --launcher python




# == data pipeline===========================================================
git clone https://github.com/Wan-Video/Wan2.2.git





