from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


# ============================================================
# Config (loaded by FACETConfig.from_yaml)
# ============================================================


@dataclass
class FACETBaseConfig:
    """WAN2.2-TI2V-5B base checkpoint location."""
    load_from: str = "local"                # "local" | "huggingface"
    id: str = "Wan-AI/Wan2.2-TI2V-5B"
    dir: str = "./weights/WAN2.2"           # local root of WAN weights (resolved against project root)
    dit: str = "diffusion_pytorch_model*.safetensors"
    t5: str = "models_t5_umt5-xxl-enc-bf16.pth"
    vae: str = "Wan2.2_VAE.pth"
    tokenizer: str = "google/umt5-xxl"


@dataclass
class FACETWanConfig:
    model_type: str = "ti2v"                
    patch_size: Tuple[int, int, int] = (1, 2, 2)
    vae_temporal_stride: int = 4            # F_lat = (F-1)//4 + 1
    vae_spatial_stride: int = 16            # WAN2.2 VAE = 16, WAN2.1 VAE = 8
    token_temporal_stride: int = 4          # patch_size[0] * vae_temporal_stride
    token_spatial_stride: int = 32          # patch_size[1] * vae_spatial_stride


@dataclass
class FACETTargetConfig:
    height: int = 480
    width: int = 832
    num_frames: int = 81
    hw_multiple: int = 32


@dataclass
class FACETSourceConfig:
    """
    Configures the OminiControl-style 'src branch'  (masked source video).

    Geometry:
      f_offset: along the time axis where src tokens are placed in the shared
                RoPE coordinate space. Per `frame.txt`, src is spatially-aligned
                with the target, so src and base SHARE positions (offset = 0).

      timestep: the diffusion timestep injected into src's per-branch AdaLN modulation. 
                Per OminiControl & frame.txt, src is a CLEAN signal (no noise), 
                so timestep=0 lets the model treat it as a fixed feature.

    Attention (OminiControl-style mask-aware bias, applied to Q_base):
      attention_mode: "asymmetric" - Q_base sees K_all; Q_src / Q_ref see only
                their own K. This is the FACET default.
                "full_iteration" reserved for ablation.

      gamma:    scale on log(eps + (1-m))  /  log(eps + m) biases. Higher gamma
                means harder suppression of invalid tokens / harder push toward
                ref when mask=1. Default 1.0. Kept as a single global scalar
                (per-layer learning is for future ablation).

      safe_epsilon: numerical floor in log to avoid -inf under bf16. 1e-3 is
                stable; lower epsilons may cause overflow.

      mask_dilation:  pixel-space dilation (kernel side length) applied to
                src_mask before pooling to token grid. 0 = no dilation.
                Increase if edge texture is leaking past the mask boundary.

      mask_pool: "avg" (default) preserves soft boundary; "nearest"
                produces a hard binary coverage.

    Inference-only:
      kv_cache: whether to cache (K_src, V_src) / (K_ref, V_ref) across
                denoising steps. Used in inference instead of training.
    """
    # geometry
    f_offset: int = 0                       # 0 per frame.txt
    timestep: float = 0.0
    injection_mode: str = "branch_attention"

    # OminiControl-style mask-aware attention
    attention_mode: str = "asymmetric"      # "asymmetric" | "full_iteration"
    gamma: float = 1.0
    safe_epsilon: float = 1e-3
    mask_dilation: int = 0                  # 0 = no dilation
    mask_pool: str = "avg"                  # "avg" | "nearest"

    # runtime
    detach_latent: bool = True
    kv_cache: bool = True                   # inference-time only


@dataclass
class FACETReferenceConfig:
    image_size: int = 480
    f_offset: int = 21                      # place ref token f index at 21 (right of latent grid f=[0..20])
    detach_latent: bool = True
    timestep: float = 0.0                   # clean signal for ref branch
    injection_mode: str = "branch_attention"
    kv_cache: bool = True                   # inference-time only; training does not cache


@dataclass
class FACETLoRAConfig:
    target_modules: Tuple[str, ...] = ("q", "k", "v", "o", "ffn.0", "ffn.2")
    on_base_blocks: bool = True             # inject LoRA on dit.blocks.*
    on_cross_attn: bool = False             # if False, q/k/v/o in cross_attn are skipped
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
