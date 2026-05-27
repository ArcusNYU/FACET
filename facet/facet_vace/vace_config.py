from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


# ============================================================
# Config
# ============================================================


@dataclass
class FACETBaseConfig:
    """WAN2.1-VACE base checkpoint location."""
    load_from: str = "local"                # "local" | "huggingface"
    id: str = "Wan-AI/Wan2.1-VACE-1.3B"
    dir: str = "./weights/WAN2.1"           # local root of WAN weights
    dit: str = "diffusion_pytorch_model*.safetensors"
    t5: str = "models_t5_umt5-xxl-enc-bf16.pth"
    vae: str = "Wan2.1_VAE.pth"
    tokenizer: str = "google/umt5-xxl"


@dataclass
class FACETWanConfig:
    model_type: str = "vace"                          # "ti2v" | "vace"
    patch_size: Tuple[int, int, int] = (1, 2, 2)
    vae_temporal_stride: int = 4                      # F_lat = (F-1)//4 + 1
    vae_spatial_stride: int = 8                       # WAN2.1: 8 | WAN2.2: 16
    token_temporal_stride: int = 4                    # patch_size[0] * vae_temporal_stride
    token_spatial_stride: int = 16                    # patch_size[1] * vae_spatial_stride


@dataclass
class FACETTargetConfig:
    height: int = 480
    width: int = 832
    num_frames: int = 81
    hw_multiple: int = 32


@dataclass
class FACETReferenceConfig:
    image_size: int = 480
    f_offset: int = 21                      # place ref token f index at 21 (right of latent grid f=[0..20])
    detach_latent: bool = True
    timestep: float = 0.0                   # clean signal for ref branch
    injection_mode: str = "branch_attention"
    kv_cache_reference: bool = True         # inference-time only; training does not cache


@dataclass
class FACETLoRAConfig:
    target_modules: Tuple[str, ...] = ("q", "k", "v", "o", "ffn.0", "ffn.2")
    on_base_blocks: bool = True             # inject LoRA on dit.blocks.*
    on_vace_blocks: bool = True             # inject LoRA on dit.vace_blocks.*  (incl. before_proj/after_proj)
    on_cross_attn:  bool = False            # if False, q/k/v/o in cross_attn are skipped
    rank: int = 32
    alpha: int = 32
    dropout: float = 0.0
    init: str = "kaiming_zero"              # only kaiming_zero supported in v1


@dataclass
class FACETTextConfig:
    max_text_len: int = 512


@dataclass
class FACETInferenceConfig:
    num_inference_steps: int = 50
    cfg_scale: float = 5.0
    reference_guidance_scale: float = 1.0
    seed: Optional[int] = None
    output_type: str = "video"              # "frames" | "video"


@dataclass
class FACETTrainingConfig:
    prediction_type: str = "velocity"       # "noise" | "velocity"
    timestep_sampling: str = "logit_normal" # "uniform" | "logit_normal"
    loss_type: str = "mse"
    ref_dropout_prob: float = 0.05
    text_dropout_prob: float = 0.10
