from dataclasses import dataclass
from typing import Tuple


# ============================================================
# 1. Config
# ============================================================

@dataclass
class FACETLoRAConfig:
    enabled: bool = True
    base_model: str = "dit"
    target_modules: Tuple[str, ...] = ("q", "k", "v", "o", "ffn.0", "ffn.2")
    rank: int = 32
    alpha: int = 32
    dropout: float = 0.0

    # TODO: adapter ???
    # LoRA routing for multi-branch reference injection.
    # "default" means use the same LoRA adapter.
    # None means no LoRA on that branch.
    # target_adapter: Optional[str] = "default"
    # reference_adapter: Optional[str] = "default"
    # text_adapter: Optional[str] = None


@dataclass
class FACETReferenceConfig:
    # enabled: bool = True
    image_size: int = 480
    # resize_mode: str = "center_crop_resize"
    # encode_with_wan_vae: bool = True
    # latent_frames: int = 1
    detach_latent: bool = True   #TODO: ???

    # OminiControl-style setting
    timestep: float = 0.0
    injection_mode: str = "branch_attention"  # branch_attention | concat_tokens
    # update_reference_branch: bool = True
    # independent_reference: bool = False
    # kv_cache_reference: bool = True   #FIXME: 暂时不需要进行缓存 因为不涉及ref条件的复用


@dataclass
class FACETTargetConfig:
    height: int = 480
    width: int = 832
    num_frames: int = 81
    hw_multiple: int = 32


@dataclass
class FACETWanConfig:
    # model_type: str = "ti2v"
    patch_size: Tuple[int, int, int] = (1, 2, 2)
    vae_temporal_stride: int = 4
    vae_spatial_stride: int = 16
    # require_num_frames_4n_plus_1: bool = True


@dataclass
class FACETInferenceConfig:
    num_inference_steps: int = 50
    cfg_scale: float = 5.0
    reference_guidance_scale: float = 1.0
    output_type: str = "pil"  #TODO: ???