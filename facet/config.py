from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple
import yaml


# ============================================================
# Config
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
    mask_bias: bool = True                   # NOTE: ablation switch: True = apply the mask-aware
                                             # routing bias on Q_base->K_src/K_ref; False =
                                             # plain 3-branch attention (model learns routing).
    gamma: float = 1.0
    safe_epsilon: float = 1e-3
    mask_dilation: int = 0                  # 0 = no dilation
    mask_pool: str = "avg"                  # "avg" | "nearest"
    q_chunk: int = 2048                     # base-branch attn query-chunk size; caps O(L^2) VRAM. <=0 disables

    # runtime
    detach_latent: bool = True
    kv_cache: bool = True                   # inference-time only


@dataclass
class FACETReferenceConfig:
    """
    Configures the OminiControl-style 'ref branch' (reference image).

    RoPE placement (per frame.txt - design changed in WAN2.2 era):
      - h_offset / w_offset:
            ref tokens are placed to the RIGHT of base in the w-axis to avoid
            spatial overlap. For target 480x832 with token_spatial_stride=32:
              base.w range: 0..25  ->  ref.w range: 26..40   (w_offset = 26)
            h does NOT need an offset (per OminiControl: "only need to avoid
            spatial overlap"; with different w ranges, (h, w) plane is already
            disjoint).
            w_offset=None means "auto = target.width // token_spatial_stride".

      - f-axis is SPECIAL:
            placing ref at any fixed f>0 would (a) be misread as a 'future
            frame' and (b) put different base frames at unequal temporal
            distances to ref. So we DISABLE the f-RoPE on ref entirely
            (treat ref as effectively f=0). When Q_base attends to K_ref, we
            also disable Q_base's f-RoPE on that path so delta_f = 0
            (achieved by both branches using f_freqs[0:1] which is the
            identity rotation exp(i*0) = 1+0j).

    Other:
      timestep: clean signal (typically 0) injected into ref's per-branch
                AdaLN modulation, so the model treats it as a fixed feature.
      kv_cache: inference-time only; training does not cache.
    """
    image_size: int = 480
    h_offset: int = 0                       # no h-axis shift; matches OminiControl design
    w_offset: int = 26                      # None -> auto = target.width // token_spatial_stride (26 for 480x832@stride32)
    detach_latent: bool = True
    timestep: float = 0.0                   # clean signal for ref branch
    injection_mode: str = "branch_attention"
    kv_cache: bool = True                   # inference-time only; training does not cache


@dataclass
class FACETLoRAConfig:
    target_modules: Tuple[str, ...] = ("q", "k", "v", "o", "ffn.0", "ffn.2")
    on_base_blocks: bool = True             # inject LoRA on dit.blocks.*
    on_cross_attn: bool = False             # if False, q/k/v/o in cross_attn are skipped
    rank: int = 64                          # recommended start; ablation arms: 32 / 64 / 128
    alpha: int = 64                         # keep alpha == rank (scale = 1.0)
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


# ============================================================
# Top-level model config
# ============================================================


@dataclass
class FACETConfig:
    """
    Aggregates every sub-config under `model:` in facet/config.yaml.

    Construct via `FACETConfig.from_yaml(path)`.
    """

    name: str = "FACET-WAN2.2"
    dtype: str = "bf16"
    device: str = "cuda"
    gradient_checkpointing: bool = True

    base: FACETBaseConfig = field(default_factory=FACETBaseConfig)
    wan: FACETWanConfig = field(default_factory=FACETWanConfig)
    target: FACETTargetConfig = field(default_factory=FACETTargetConfig)
    source: FACETSourceConfig = field(default_factory=FACETSourceConfig)
    reference: FACETReferenceConfig = field(default_factory=FACETReferenceConfig)
    lora: FACETLoRAConfig = field(default_factory=FACETLoRAConfig)
    text: FACETTextConfig = field(default_factory=FACETTextConfig)
    inference: FACETInferenceConfig = field(default_factory=FACETInferenceConfig)

    _SUB_CONFIGS = {
        "base": ("base", FACETBaseConfig),
        "wan": ("wan", FACETWanConfig),
        "target": ("target", FACETTargetConfig),
        "source": ("source", FACETSourceConfig),
        "reference": ("reference", FACETReferenceConfig),
        "lora": ("lora", FACETLoRAConfig),
        "text": ("text", FACETTextConfig),
        "inference": ("inference", FACETInferenceConfig),
    }
    _FLAT_FIELDS = {"name", "dtype", "device", "gradient_checkpointing"}

    @staticmethod
    def from_yaml(path: str) -> "FACETConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        model_block = raw.get("model", {}) or {}
        cfg = FACETConfig()

        for k, v in model_block.items():
            if k in FACETConfig._FLAT_FIELDS:
                setattr(cfg, k, v)
                continue

            if k in FACETConfig._SUB_CONFIGS:
                attr_name, _ = FACETConfig._SUB_CONFIGS[k]
                sub = getattr(cfg, attr_name)
                for sk, sv in (v or {}).items():
                    if not hasattr(sub, sk):
                        print(f"[FACETConfig] Unknown key model.{k}.{sk} in yaml, ignored.")
                        continue
                    if sk == "patch_size" and isinstance(sv, list):
                        sv = tuple(sv)
                    if sk == "target_modules" and isinstance(sv, list):
                        sv = tuple(sv)
                    setattr(sub, sk, sv)
                continue

            print(f"[FACETConfig] Unknown top-level key model.{k} in yaml, ignored.")

        return cfg


