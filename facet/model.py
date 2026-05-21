"""
FACET / WAN2.1-VACE-1.3B + OminiControl LoRA fine-tuning model.

This file contains everything except the per-step `forward()` and the inference
`generate()` loop (those land in Step 2). What is implemented here:

    A. FACETConfig        : top-level config, mirrors facet/config.yaml -> dataclasses
    B. LoRA injection     : suffix-based matcher + LoRALinear replacement
    C. Utilities          : tensor <-> list conversion, video size validator
    D. FACETWanModel      : ctor + base component loader + freeze + LoRA + (no_grad) encoders

Reference layout (from facet/model_frame.py):

  base (target) branch (x):   [B, 16, F_lat=21, 60, 104]  -- WAN2.1 VAE z_dim=16
  vace branch        (c):   [B, 96, F_lat=21, 60, 104]  -- (inactive,reactive) latent + 64ch mask
  reference branch   (r):   [B, 16,         1, 30, 30]  -- single image VAE latent
  text branch        (t):   [B, 512, 4096] T5 -> 1536 dim

Hidden dims for WAN2.1-VACE-1.3B: dim=1536, num_heads=12, ffn_dim=8960, num_layers=30,
vace_layers = [0, 2, ..., 28] (15 layers).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import yaml
from PIL import Image

from .config import (
    FACETBaseConfig,
    FACETInferenceConfig,
    FACETLoRAConfig,
    FACETReferenceConfig,
    FACETTargetConfig,
    FACETTextConfig,
    FACETTrainingConfig,
    FACETWanConfig,
)
from .lora import LoRALinear
from .utils import (
    _get_parent_module,
    _resolve_dtype,
    _resolve_local_path,
    _resolve_local_paths,
)


logger = logging.getLogger(__name__)


# ============================================================
# A. Top-level model config
# ============================================================


@dataclass
class FACETConfig:
    """
    Aggregates every sub-config under `model:` in facet/config.yaml.

    Construct via `FACETConfig.from_yaml(path)`.
    """

    name: str = "FACET-WAN2.1-VACE"
    dtype: str = "bf16"
    device: str = "cuda"
    gradient_checkpointing: bool = True

    base: FACETBaseConfig = field(default_factory=FACETBaseConfig)
    wan: FACETWanConfig = field(default_factory=FACETWanConfig)
    target: FACETTargetConfig = field(default_factory=FACETTargetConfig)
    reference: FACETReferenceConfig = field(default_factory=FACETReferenceConfig)
    lora: FACETLoRAConfig = field(default_factory=FACETLoRAConfig)
    text: FACETTextConfig = field(default_factory=FACETTextConfig)
    inference: FACETInferenceConfig = field(default_factory=FACETInferenceConfig)
    training: FACETTrainingConfig = field(default_factory=FACETTrainingConfig)

    # Map yaml-block-name -> (attr_name, dataclass)
    _SUB_CONFIGS = {
        "base": ("base", FACETBaseConfig),
        "wan": ("wan", FACETWanConfig),
        "target": ("target", FACETTargetConfig),
        "reference": ("reference", FACETReferenceConfig),
        "lora": ("lora", FACETLoRAConfig),
        "text": ("text", FACETTextConfig),
        "inference": ("inference", FACETInferenceConfig),
        "training": ("training", FACETTrainingConfig),
    }

    # Flat scalar fields on the top-level FACETConfig
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
                attr_name, dc_cls = FACETConfig._SUB_CONFIGS[k]
                sub = getattr(cfg, attr_name)
                for sk, sv in (v or {}).items():
                    if not hasattr(sub, sk):
                        logger.warning(
                            "Unknown key model.%s.%s in yaml, ignored.", k, sk
                        )
                        continue
                    if sk == "patch_size" and isinstance(sv, list):
                        sv = tuple(sv)
                    if sk == "target_modules" and isinstance(sv, list):
                        sv = tuple(sv)
                    setattr(sub, sk, sv)
                continue

            logger.warning("Unknown top-level key model.%s in yaml, ignored.", k)

        return cfg


# ============================================================
# B. LoRA targeting and injection
# ============================================================


def _module_name_matches_suffix(name: str, target_modules: Sequence[str]) -> bool:
    """
    True if `name` ends with any of `target_modules`.

    Handles both single-segment ("q") and multi-segment ("ffn.0") suffixes.

    Examples:
        "blocks.0.self_attn.q"  ends with "q"      -> True
        "blocks.0.ffn.0"        ends with "ffn.0"  -> True
        "blocks.0.cross_attn.q" ends with "q"      -> True
        "vace_blocks.3.before_proj" ends with "before_proj" -> True
    """
    for tm in target_modules:
        if name == tm or name.endswith("." + tm):
            return True
    return False


def lora_targets(
    name: str,
    target_modules: Sequence[str],
    in_base_block: bool,
    in_vace_block: bool,
    lora_cfg: FACETLoRAConfig,
) -> bool:
    """
    Decide whether the nn.Linear at module path `name` should be wrapped.

    Rules:
      - The module must be inside dit.blocks.* (`in_base_block`) or
        dit.vace_blocks.* (`in_vace_block`).
      - `lora_cfg.base_blocks` / `lora_cfg.vace_blocks` toggles whole regions.
      - Suffix must be in `target_modules`, OR for vace blocks we additionally
        always include {before_proj, after_proj} since these are VACE-only
        projections without a counterpart in base WAN blocks.
    """
    if in_base_block and not lora_cfg.base_blocks:
        return False
    if in_vace_block and not lora_cfg.vace_blocks:
        return False
    if not (in_base_block or in_vace_block):
        return False

    if _module_name_matches_suffix(name, target_modules):
        return True

    if in_vace_block and (
        name.endswith(".before_proj") or name.endswith(".after_proj")
    ):
        return True

    return False


def inject_lora(
    root: nn.Module,
    lora_cfg: FACETLoRAConfig,
) -> List[str]:
    """
    Walk `root`, replace every matching nn.Linear with LoRALinear in-place.

    Returns the list of replaced module paths (for logging / debugging).

    NOTE: `root` must be the dit (transformer) module. We deliberately do NOT
    iterate over vae / text_encoder.
    """
    replaced: List[str] = []

    # list(...) freezes the iterator since we mutate `root` during the loop.
    for name, module in list(root.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        in_base_block = name.startswith("blocks.")
        in_vace_block = name.startswith("vace_blocks.")

        if not lora_targets(
            name=name,
            target_modules=lora_cfg.target_modules,
            in_base_block=in_base_block,
            in_vace_block=in_vace_block,
            lora_cfg=lora_cfg,
        ):
            continue

        parent, child_name = _get_parent_module(root, name)
        setattr(
            parent,
            child_name,
            LoRALinear(
                base=module,
                rank=lora_cfg.rank,
                alpha=lora_cfg.alpha,
                dropout=lora_cfg.dropout,
            ),
        )
        replaced.append(name)

    return replaced


# ============================================================
# C. Utilities
# ============================================================


def latent_frames_from_num_frames(num_frames: int, temporal_stride: int = 4) -> int:
    """F_lat = (F - 1) // temporal_stride + 1, with WAN's 4n+1 constraint."""
    assert (num_frames - 1) % temporal_stride == 0, (
        f"num_frames should be {temporal_stride}n+1 for WAN-style video VAE, "
        f"got {num_frames}"
    )
    return (num_frames - 1) // temporal_stride + 1


def ensure_latent_list(
    x: Union[torch.Tensor, List[torch.Tensor]],
) -> List[torch.Tensor]:
    """
    Wan official forward expects List[Tensor], not a stacked tensor.

    Accept:
      [B, C, F, H, W] -> List of B tensors [C, F, H, W]
      [C, F, H, W]    -> List of 1 tensor
      List[Tensor]    -> unchanged
    """
    if isinstance(x, list):
        return x
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected Tensor or List[Tensor], got {type(x)}")
    if x.ndim == 5:
        return [x[i] for i in range(x.shape[0])]
    if x.ndim == 4:
        return [x]
    raise ValueError(
        f"Expected latent tensor with ndim 4 or 5, got shape {tuple(x.shape)}"
    )


def ensure_context_list(
    x: Union[torch.Tensor, List[torch.Tensor]],
) -> List[torch.Tensor]:
    """
    T5/WAN cross-attn context list normalizer.

    Accept:
      [B, L, D]    -> List of B tensors [L, D]
      [L, D]       -> List of 1 tensor
      List[Tensor] -> unchanged
    """
    if isinstance(x, list):
        return x
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected Tensor or List[Tensor], got {type(x)}")
    if x.ndim == 3:
        return [x[i] for i in range(x.shape[0])]
    if x.ndim == 2:
        return [x]
    raise ValueError(
        f"Expected context tensor with ndim 2 or 3, got shape {tuple(x.shape)}"
    )


def validate_video_size(
    height: int,
    width: int,
    num_frames: int,
    hw_multiple: int = 32,
    temporal_stride: int = 4,
) -> None:
    if height % hw_multiple != 0 or width % hw_multiple != 0:
        raise ValueError(
            f"height and width must be divisible by {hw_multiple}. "
            f"Got height={height}, width={width}."
        )
    if (num_frames - 1) % temporal_stride != 0:
        raise ValueError(
            f"num_frames must be {temporal_stride}n+1 for WAN-style video VAE. "
            f"Got {num_frames}."
        )


# ============================================================
# D. FACET model wrapper
# ============================================================


class FACETWanModel(nn.Module):
    """
    FACET model wrapper.

    Training:
        forward() does one denoising prediction step. (implemented in Step 2)

    Inference:
        generate() does full denoising loop. (implemented in Step 2)

    This class uses composition over inheritance: it owns a DiffSynth
    WanVideoPipeline and delegates dit / vae / text_encoder to it.
    """

    def __init__(self, cfg: FACETConfig):
        super().__init__()
        self.cfg = cfg
        self.dtype = _resolve_dtype(cfg.dtype)
        self.device = cfg.device

        # Components assigned by _load_base_components().
        self.pipe = None          # DiffSynth WanVideoPipeline
        self.dit = None           # WanModel (with VACE blocks)
        self.vace = None          # VaceWanModel  (DiffSynth keeps VACE blocks on pipe.vace)
        self.vae = None           # WanVideoVAE
        self.text_encoder = None  # WanTextEncoder (UMT5)
        self.tokenizer = None     # HuggingfaceTokenizer
        self.scheduler = None     # FlowMatchScheduler

        self._load_base_components()
        self._freeze_base()
        self._lora_replaced: List[str] = []
        self._init_lora()

        # NOTE: device placement is intentionally left to the trainer:
        # train.py (accelerate) will move the whole module to acc.device.
        # Calling self.to(self.device) here would conflict with accelerate's
        # DDP wrap and is unnecessary because DiffSynth's pipeline has already
        # placed each sub-model on `device` during from_pretrained.

    @classmethod
    def from_config(cls, path: str) -> "FACETWanModel":
        cfg = FACETConfig.from_yaml(path)
        return cls(cfg)

    # --------------------------------------------------------
    # Base-component loading
    # --------------------------------------------------------

    def _load_base_components(self) -> None:
        """
        Load Wan components via DiffSynth's WanVideoPipeline.

        We strictly load from local files when cfg.base.load_from == "local".
        The yaml block:

            base:
              load_from: local
              dir: ./weights/WAN2.1
              dit: diffusion_pytorch_model*.safetensors
              t5:  models_t5_umt5-xxl-enc-bf16.pth
              vae: Wan2.1_VAE.pth

        is resolved to absolute local paths via glob.
        """
        # Late import: DiffSynth pulls in flash_attn etc, so we only import
        # when actually constructing a model.
        from diffsynth.pipelines.wan_video import (
            WanVideoPipeline,
            ModelConfig as DSModelConfig,
        )
        from diffsynth.diffusion import FlowMatchScheduler

        bcfg = self.cfg.base

        if bcfg.load_from == "local":
            dit_paths = _resolve_local_paths(bcfg.dir, bcfg.dit)   # may be sharded
            t5_path = _resolve_local_path(bcfg.dir, bcfg.t5)
            vae_path = _resolve_local_path(bcfg.dir, bcfg.vae)

            model_configs = [
                DSModelConfig(path=dit_paths if len(dit_paths) > 1 else dit_paths[0]),
                DSModelConfig(path=t5_path),
                DSModelConfig(path=vae_path),
            ]
            # tokenizer: either a yaml override, or fall back to local subdir.
            if bcfg.tokenizer:
                tok_path = bcfg.tokenizer
            else:
                # DiffSynth's default is "google/umt5-xxl" tokenizer dir.
                # Look for it under base.dir/google/umt5-xxl
                tok_dir = os.path.join(bcfg.dir, "google", "umt5-xxl")
                if not os.path.isdir(tok_dir):
                    raise FileNotFoundError(
                        f"Tokenizer dir not found: {tok_dir}. "
                        "Set model.base.tokenizer in yaml to override."
                    )
                tok_path = tok_dir
            tokenizer_config = DSModelConfig(path=tok_path)
        elif bcfg.load_from == "huggingface":
            # Reserved for ablation. Today we only allow local because the
            # weights are pre-staged on disk.
            raise NotImplementedError(
                "huggingface loading is intentionally disabled in FACET v1. "
                "Pre-download weights and use load_from: local."
            )
        else:
            raise ValueError(
                f"Unknown base.load_from={bcfg.load_from!r} "
                "(expected 'local' or 'huggingface')."
            )

        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=self.dtype,
            device=self.device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
        )

        self.dit = self.pipe.dit
        self.vace = self.pipe.vace
        self.vae = self.pipe.vae
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer
        # DiffSynth's WanVideoPipeline pre-instantiates a FlowMatchScheduler;
        # we hold a direct reference for clarity.
        self.scheduler = getattr(self.pipe, "scheduler", None) or FlowMatchScheduler("Wan")

        if self.dit is None:
            raise RuntimeError(
                "WanVideoPipeline did not load the dit. Check that "
                f"{bcfg.dir}/{bcfg.dit} contains the diffusion model shards."
            )
        if self.vace is None:
            raise RuntimeError(
                "WanVideoPipeline did not load the VACE branch. "
                "Make sure you are using the VACE checkpoint "
                "(Wan2.1-VACE-1.3B), not the plain T2V."
            )

    # --------------------------------------------------------
    # Freeze / LoRA
    # --------------------------------------------------------

    def _freeze_base(self) -> None:
        """
        Freeze everything before LoRA injection.

        LoRA-bearing modules will re-enable grad on their own lora_down/lora_up
        parameters during forward; we additionally re-set requires_grad after
        _init_lora() to be explicit.
        """
        for p in self.parameters():
            p.requires_grad_(False)

        for sub in (self.dit, self.vace, self.vae, self.text_encoder):
            if sub is None:
                continue
            if hasattr(sub, "eval"):
                sub.eval()

    def _init_lora(self) -> None:
        """
        Inject LoRA into dit.blocks.* (base) and dit.vace_blocks.* (vace).

        We rely on the convention that VACE blocks are stored as
            self.pipe.vace.vace_blocks
        in DiffSynth's VaceWanModel. To allow `inject_lora` to find both
        regions with simple prefix tests, we walk vace under its own root.
        """
        replaced: List[str] = []

        if self.dit is not None and self.cfg.lora.base_blocks:
            replaced += [
                "dit." + n for n in inject_lora(self.dit, self.cfg.lora)
            ]

        if self.vace is not None and self.cfg.lora.vace_blocks:
            replaced += [
                "vace." + n for n in inject_lora(self.vace, self.cfg.lora)
            ]

        if len(replaced) == 0:
            raise RuntimeError(
                "No LoRA modules were injected. "
                f"Check lora.target_modules={list(self.cfg.lora.target_modules)} "
                "against your Wan / VACE module names."
            )

        # Re-enable grad ONLY on lora_down / lora_up parameters.
        for name, p in self.named_parameters():
            p.requires_grad_("lora_down" in name or "lora_up" in name)

        self._lora_replaced = replaced

        logger.info("[FACET] Injected LoRA into %d modules.", len(replaced))
        for name in replaced[:20]:
            logger.info("  - %s", name)
        if len(replaced) > 20:
            logger.info("  ... and %d more", len(replaced) - 20)

    # --------------------------------------------------------
    # Save / load LoRA
    # --------------------------------------------------------

    def save_lora(self, path: str) -> None:
        """
        Save only LoRA-related weights (lora_down / lora_up).

        File format: safetensors.
        """
        from safetensors.torch import save_file

        state = {
            k: v.detach().cpu()
            for k, v in self.state_dict().items()
            if ("lora_down" in k) or ("lora_up" in k)
        }
        if len(state) == 0:
            raise RuntimeError("No LoRA params found in state_dict; nothing to save.")

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        save_file(state, path)
        logger.info("[FACET] Saved %d LoRA tensors -> %s", len(state), path)

    def load_lora(self, path: str, strict: bool = False) -> None:
        """
        Load LoRA weights. _init_lora() must have run first so that the matching
        LoRALinear modules already exist in the model.
        """
        from safetensors.torch import load_file

        state = load_file(path)
        missing, unexpected = self.load_state_dict(state, strict=strict)
        if len(unexpected) > 0:
            logger.warning("[FACET] Unexpected LoRA keys: %s", unexpected[:10])
        if len(missing) > 0:
            # `missing` will contain ALL base + vae + text_encoder keys when
            # the safetensors file only stores LoRA params; that is expected.
            logger.info(
                "[FACET] %d missing keys when loading LoRA (mostly frozen base, expected).",
                len(missing),
            )

    # --------------------------------------------------------
    # Encoding helpers
    # --------------------------------------------------------

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    ) -> List[torch.Tensor]:
        """
        Returns a WAN-compatible context list of variable-length T5 embeddings.

        Priority:
          - if `prompt_embeds` is given, normalize and return as a list directly
            (this is the common training path: we precompute and cache T5 hidden
             states in the data pipeline).
          - else encode `prompt` via WAN text encoder + tokenizer.

        Output shape: List length B, each [L_i, 4096], L_i <= cfg.text.max_text_len.
        """
        if prompt_embeds is not None:
            return ensure_context_list(prompt_embeds)

        if prompt is None:
            raise ValueError("Either prompt or prompt_embeds must be provided.")

        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = list(prompt)

        # DiffSynth's WanTextEncoder is callable: (texts, device) -> list of tensors
        # with the padding already stripped per sample.
        if self.text_encoder is None or self.tokenizer is None:
            raise RuntimeError("text_encoder/tokenizer not loaded.")

        # WanVideoPipeline keeps the text_encoder on its own device. We use
        # the same call signature DiffSynth uses internally.
        device = next(self.text_encoder.parameters()).device

        ids, mask = self.tokenizer(
            prompts,
            return_mask=True,
            add_special_tokens=True,
        )
        ids = ids.to(device)
        mask = mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)  # [B, L_max, 4096]
        # strip padding per sample to match WAN forward's expected list format.
        return [u[:int(l)] for u, l in zip(context, seq_lens)]

    @torch.no_grad()
    def encode_reference_image(
        self,
        reference_images: Union[Image.Image, List[Image.Image], torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Encode reference image(s) through WAN VAE to single-frame latents.

        Accepted inputs:
          - PIL.Image                       -> batch 1
          - List[PIL.Image]                 -> batch len(list)
          - torch.Tensor [3, H, W]          -> batch 1
          - torch.Tensor [B, 3, H, W]       -> batch B
          - torch.Tensor [B, 3, 1, H, W]    -> already video-shaped, batch B
          - torch.Tensor [B, 16, 1, h, w]   -> already latent (pre-cached); returned as-is

        Returned list element shape: [16, 1, H/8, W/8].

        We DO NOT resize here when the input is already a tensor at
        cfg.reference.image_size; we trust the data pipeline to have done it
        (see data/transform.py:RefTfm). For raw PIL we delegate to DiffSynth's
        pipe.preprocess_video on a single-frame list, which handles
        center-crop / resize / normalize to [-1, 1].

        NOTE: We do not consume RGBA alpha here. The dataset has already done
        background augmentation before storing the tensor on disk. If you pass
        a PIL.Image with mode RGBA, the alpha channel is dropped here.
        """
        if self.vae is None:
            raise RuntimeError("VAE not loaded.")

        ref_size = self.cfg.reference.image_size

        # Case 1: list / single PIL.
        if isinstance(reference_images, Image.Image):
            pil_list = [reference_images]
            return self._encode_ref_pil(pil_list, ref_size)

        if isinstance(reference_images, list) and all(
            isinstance(x, Image.Image) for x in reference_images
        ):
            return self._encode_ref_pil(reference_images, ref_size)

        if not isinstance(reference_images, torch.Tensor):
            raise TypeError(
                f"Unsupported reference_images type: {type(reference_images)}"
            )

        x = reference_images

        # Case 2: pre-cached latent (z_dim=16, F=1). Return as-is in list form.
        if x.ndim == 5 and x.shape[1] == 16 and x.shape[2] == 1:
            return ensure_latent_list(x)
        if x.ndim == 4 and x.shape[0] == 16 and x.shape[1] == 1:
            return ensure_latent_list(x)

        # Case 3: raw pixel tensor. Normalize shape to [B, 3, 1, H, W].
        if x.ndim == 3:                           # [3, H, W]
            x = x.unsqueeze(0).unsqueeze(2)
        elif x.ndim == 4:                         # [B, 3, H, W]
            x = x.unsqueeze(2)
        elif x.ndim == 5:                         # [B, 3, 1, H, W]
            pass
        else:
            raise ValueError(
                f"Unsupported reference_images tensor shape: {tuple(x.shape)}"
            )

        x = x.to(device=self.device, dtype=self.dtype)

        # DiffSynth VAE.encode expects a list of [3, F, H, W] tensors -OR-
        # a stacked [B, 3, F, H, W] tensor. We use the stacked form.
        ref_latents = self.vae.encode(
            x, device=self.device,
        ).to(dtype=self.dtype, device=self.device)

        # split [B, 16, 1, h, w] -> list of [16, 1, h, w]
        latents = ensure_latent_list(ref_latents)

        if self.cfg.reference.detach_latent:
            latents = [l.detach() for l in latents]
        return latents

    def _encode_ref_pil(
        self,
        pil_list: List[Image.Image],
        ref_size: int,
    ) -> List[torch.Tensor]:
        """
        Path used when caller passed PIL images. Routes through pipe.preprocess_video
        on a single-frame "video" so that we reuse DiffSynth's normalization
        (resize + (x-0.5)/0.5 + permute) instead of re-implementing it here.
        """
        # Each PIL becomes a single-frame "video" of length 1.
        resized = [im.convert("RGB").resize((ref_size, ref_size)) for im in pil_list]
        # pipe.preprocess_video accepts a list of PIL frames; we wrap each PIL
        # individually as its own one-frame clip.
        out_latents: List[torch.Tensor] = []
        for im in resized:
            v = self.pipe.preprocess_video([im])  # [1, 3, 1, H, W]
            z = self.vae.encode(
                v.to(device=self.device, dtype=self.dtype),
                device=self.device,
            ).to(dtype=self.dtype, device=self.device)
            # z: [1, 16, 1, h, w] -> [16, 1, h, w]
            z = z[0]
            if self.cfg.reference.detach_latent:
                z = z.detach()
            out_latents.append(z)
        return out_latents

    @torch.no_grad()
    def decode_latents(
        self,
        latents: Union[List[torch.Tensor], torch.Tensor],
        output_type: str = "video",
    ) -> torch.Tensor:
        """
        Decode target video latents [B, 16, F_lat, H/8, W/8] -> pixel-space tensor.

        Returns a stacked tensor [B, 3, F, H, W] in [-1, 1].
        Postprocess (uint8, PIL frame list, mp4 encode...) is the pipeline's job.
        """
        latents = ensure_latent_list(latents)

        # DiffSynth's WanVideoVAE.decode wants a stacked tensor.
        stacked = torch.stack(latents, dim=0)  # [B, 16, F_lat, h, w]
        video = self.vae.decode(stacked.to(self.device), device=self.device)
        # video: [B, 3, F, H, W] in [-1, 1] (DiffSynth keeps this convention)
        return video

    # --------------------------------------------------------
    # Latent initialization
    # --------------------------------------------------------

    def prepare_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        num_frames: int,
        generator: Optional[torch.Generator] = None,
    ) -> List[torch.Tensor]:
        """
        Sample initial Gaussian latents for the target (base) branch.

        Shape per sample: [z_dim=16, F_lat, H/8, W/8].
        """
        validate_video_size(
            height=height,
            width=width,
            num_frames=num_frames,
            hw_multiple=self.cfg.target.hw_multiple,
            temporal_stride=self.cfg.wan.vae_temporal_stride,
        )

        f_lat = latent_frames_from_num_frames(
            num_frames,
            temporal_stride=self.cfg.wan.vae_temporal_stride,
        )
        h_lat = height // self.cfg.wan.vae_spatial_stride
        w_lat = width // self.cfg.wan.vae_spatial_stride

        # Read from dit when possible; default to 16 for WAN2.1 VAE.
        c = getattr(self.dit, "in_dim", 16)

        latents = torch.randn(
            batch_size,
            c,
            f_lat,
            h_lat,
            w_lat,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        return ensure_latent_list(latents)

    # --------------------------------------------------------
    # RoPE helpers
    # --------------------------------------------------------

    def build_freqs(
        self,
        f: int,
        h: int,
        w: int,
        f_offset: int = 0,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Build the 3D RoPE freqs tensor for one branch.

        Args:
            f, h, w   : grid size in patchified-token space.
            f_offset  : start index along the time axis. Use 0 for base/vace
                        branches and `cfg.reference.f_offset` for the reference
                        branch so the two grids do not overlap.
            device    : freqs target device. Defaults to self.dit.freqs[0].device.

        Returns:
            freqs of shape [f*h*w, 1, head_dim/2] ready to multiply with q/k.
        """
        if self.dit is None:
            raise RuntimeError("dit not loaded.")
        # DiffSynth WanModel stores precomputed 1D freqs as a tuple of three
        # complex tensors (f_freqs, h_freqs, w_freqs) on self.dit.freqs.
        f_freqs, h_freqs, w_freqs = self.dit.freqs
        if device is None:
            device = f_freqs.device

        max_f = f_freqs.shape[0]
        if f_offset + f > max_f:
            raise ValueError(
                f"f_offset({f_offset}) + f({f}) = {f_offset + f} exceeds "
                f"precomputed freqs length {max_f}. "
                "Increase precompute end if you need a larger temporal range."
            )

        freqs = torch.cat(
            [
                f_freqs[f_offset : f_offset + f]
                    .view(f, 1, 1, -1).expand(f, h, w, -1),
                h_freqs[:h].view(1, h, 1, -1).expand(f, h, w, -1),
                w_freqs[:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, 1, -1).to(device)
        return freqs

    # --------------------------------------------------------
    # Time modulation helper
    # --------------------------------------------------------

    def compute_time_modulation(
        self,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert [B]-shaped diffusion timestep into the (B, 6, dim) modulation
        tensor that every WAN attention block consumes.

        Used by the custom block forward in Step 2 so the ref branch can be
        modulated with t=0 while the base branch is modulated with the current
        diffusion timestep.
        """
        # Late-bound import to avoid pulling DiffSynth at module load time.
        from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d

        if self.dit is None:
            raise RuntimeError("dit not loaded.")

        dit = self.dit
        device = next(dit.parameters()).device
        timestep = timestep.to(device=device)

        # Cast to the dtype the rest of the dit expects.
        t_emb = sinusoidal_embedding_1d(dit.freq_dim, timestep).to(self.dtype)
        t = dit.time_embedding(t_emb)
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))  # [B, 6, dim]
        return t_mod


# ============================================================
# E. Public pipeline (kept light here; full inference lands in Step 2)
# ============================================================


class FACETPipeline:
    """
    Thin user-facing wrapper. Real inference loop lives in
    FACETWanModel.generate(); we add input validation + RNG plumbing here.
    """

    def __init__(self, model: FACETWanModel):
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        # generate() will be implemented in Step 2; for now this routes through
        # so importing the module is safe before the loop is finished.
        if not hasattr(self.model, "generate"):
            raise NotImplementedError(
                "FACETWanModel.generate() is not implemented yet (Step 2)."
            )
        return self.model.generate(*args, **kwargs)
