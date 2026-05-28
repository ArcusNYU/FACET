"""
FACET / WAN2.2-TI2V-5B + OminiControl LoRA fine-tuning model.

Reference layout (WAN2.2-TI2V-5B, 480x832x81):

  base (target) branch(x):    [B, 16, F_lat=21, 30, 52]  -- WAN2.2 VAE z_dim=16, vae_stride=16
                              -- patchify (k=s=(1,2,2)) -> token grid [B, 21, 15, 26], L_base = 8190

  src branch          (s):    [B, 16, F_lat=21, 30, 52]  -- masked source video latent
                              -- shares dit.patch_embedding -> same token grid as base, L_src = 8190
                              -- RoPE f_offset = 0 (spatially aligned with base)

  reference branch    (r):    [B, 16,         1, 30, 30]  -- single image VAE latent (480x480)
                              -- shares dit.patch_embedding -> [B, 1, 15, 15], L_ref = 225
                              -- RoPE f_offset = 21 (placed right of base in shared coord space)

  text branch         (t):    [B, 512, 4096]  T5 hidden states  ->  text_embedding  ->  [B, 512, 3072]

Hidden dims for WAN2.2-TI2V-5B:
    dim=3072, num_heads=24, head_dim=128, ffn_dim=14336, num_layers=30, freq_dim=256.
    head_dim/2 = 64 complex pairs split as 22(f) + 21(h) + 21(w) for 3D RoPE.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image

from .config import (
    FACETBaseConfig,  FACETInferenceConfig,
    FACETLoRAConfig,  FACETReferenceConfig,
    FACETSourceConfig,FACETTargetConfig,
    FACETTextConfig,  FACETTrainingConfig,
    FACETWanConfig,
)
from .lora import inject_lora

from utils import (
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
    training: FACETTrainingConfig = field(default_factory=FACETTrainingConfig)

    # Map yaml-block-name -> (attr_name, dataclass)
    _SUB_CONFIGS = {
        "base": ("base", FACETBaseConfig),
        "wan": ("wan", FACETWanConfig),
        "target": ("target", FACETTargetConfig),
        "source": ("source", FACETSourceConfig),
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
                attr_name, _ = FACETConfig._SUB_CONFIGS[k]
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
# B. OminiControl-style block forward (Step 2)
# ============================================================
# NOTE: facet_block_forward is implemented in Step 2.
# Three branches (base, src, ref) with asymmetric attention + mask-aware bias.
# A stub here would only obscure the intent; we leave it for the next pass.


# ============================================================
# C. FACET Wan Model wrapper
# ============================================================


class FACETWanModel(nn.Module):
    """
    FACET model wrapper.

    Training:
        forward() does one denoising prediction step.

    Inference:
        generate() drives the full denoising loop.

    Composition over inheritance: this class owns a DiffSynth WanVideoPipeline
    and delegates dit / vae / text_encoder to it.
    """

    def __init__(self, cfg: FACETConfig):
        super().__init__()
        self.cfg = cfg
        self.dtype = _resolve_dtype(cfg.dtype)
        self.device = cfg.device

        # Components will be assigned by _load_base_components().
        self.pipe = None          # DiffSynth WanVideoPipeline
        self.dit = None           # WanModel (TI2V-5B)
        self.vae = None           # WanVideoVAE (Wan2.2 VAE, stride 16)
        self.text_encoder = None  # WanTextEncoder (UMT5)
        self.tokenizer = None     # HuggingfaceTokenizer
        self.scheduler = None     # FlowMatchScheduler ("Wan" template)

        self._load_base_components()
        self._freeze_base()

        self._lora_replaced: List[str] = []
        self._init_lora()

        # Device placement is intentionally left to the trainer.
        # accelerate handles wrapping; DiffSynth's from_pretrained has already
        # placed each sub-model on `device`.

    @classmethod
    def from_config(cls, path: str) -> "FACETWanModel":
        cfg = FACETConfig.from_yaml(path)
        return cls(cfg)

    # --------------------------------------------------------
    # Base-component loading
    # --------------------------------------------------------

    def _load_base_components(self) -> None:
        """
        Load WAN2.2-TI2V-5B components via DiffSynth's WanVideoPipeline.

        Strictly loads from local paths when cfg.base.load_from == "local".
        Relative paths in base.dir / base.dit / ... are resolved against the
        FACET project root (see utils._resolve_against_project_root), so train.py
        can be launched from anywhere without breaking paths.

        Note (vs FACET-WAN2.1-VACE):
            * We do NOT touch / require pipe.vace anymore.
            * pipe.dit is a TI2V WanModel; its native i2v injection paths
              (input_image / fuse_vae_embedding_in_latents / seperated_timestep)
              are bypassed because FACET feeds the image-conditioning signal via
              an OminiControl branch instead.
        """
        # Late import keeps DiffSynth's heavy deps out of module load time.
        from diffsynth.pipelines.wan_video import (
            WanVideoPipeline,
            ModelConfig as DSModelConfig,
        )
        from diffsynth.diffusion import FlowMatchScheduler

        bcfg = self.cfg.base

        if bcfg.load_from == "local":
            dit_paths = _resolve_local_paths(bcfg.dir, bcfg.dit)   # may be sharded (5B has 3)
            t5_path = _resolve_local_path(bcfg.dir, bcfg.t5)
            vae_path = _resolve_local_path(bcfg.dir, bcfg.vae)
            tokenizer_path = _resolve_local_path(bcfg.dir, bcfg.tokenizer)

            model_configs = [
                DSModelConfig(path=dit_paths if len(dit_paths) > 1 else dit_paths[0]),
                DSModelConfig(path=t5_path),
                DSModelConfig(path=vae_path),
            ]
            tokenizer_config = DSModelConfig(path=tokenizer_path)
        elif bcfg.load_from == "huggingface":
            raise NotImplementedError(
                "huggingface loading is intentionally disabled in FACET. "
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
        self.vae = self.pipe.vae
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer
        # DiffSynth's WanVideoPipeline pre-instantiates a FlowMatchScheduler.
        self.scheduler = getattr(self.pipe, "scheduler", None) or FlowMatchScheduler("Wan")

        if self.dit is None:
            raise RuntimeError(
                "WanVideoPipeline did not load the dit. Check that "
                f"{bcfg.dir}/{bcfg.dit} contains the diffusion model shards."
            )

        # Sanity-check we actually loaded a TI2V-5B (or compatible) backbone.
        # TI2V-5B uses dim=3072; warn if it diverges so the user notices a wrong checkpoint.
        if getattr(self.dit, "dim", None) != 3072:
            logger.warning(
                "[FACET] dit.dim=%s; expected 3072 for Wan2.2-TI2V-5B. "
                "If you are using a different base (e.g. 14B), update frame.txt accordingly.",
                getattr(self.dit, "dim", None),
            )

    # --------------------------------------------------------
    # Freeze / LoRA
    # --------------------------------------------------------

    def _freeze_base(self) -> None:
        """
        Freeze pipeline base components before LoRA injection.
        将基础参数冻结 并将模块设置为eval模式

        LoRA-bearing modules will re-enable grad on their own lora_down/lora_up
        parameters during _init_lora().
        """
        for p in self.parameters():
            p.requires_grad_(False)

        for sub in (self.dit, self.vae, self.text_encoder):
            if sub is None:
                continue
            if hasattr(sub, "eval"):
                sub.eval()

    def _init_lora(self) -> None:
        """
        Inject LoRA into dit.blocks.* (base WAN transformer blocks).

        WAN2.2-TI2V-5B has NO vace_blocks, so the vace-branch path in
        `inject_lora` is naturally skipped. The yaml-level on_vace_blocks
        toggle defaults to False for forward-compat.
        """
        replaced: List[str] = []

        if self.dit is not None and self.cfg.lora.on_base_blocks:
            replaced += [
                "dit." + n for n in inject_lora(self.dit, self.cfg.lora)
            ]

        if len(replaced) == 0:
            raise RuntimeError(
                "No LoRA modules were injected. "
                f"Check lora.target_modules={list(self.cfg.lora.target_modules)} "
                "against Wan module names."
            )

        # Enable grad ONLY on lora_down / lora_up parameters.
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
        """Save only LoRA-related weights (lora_down / lora_up) as safetensors."""
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
        Load LoRA weights. Must be called AFTER _init_lora() so the matching
        LoRALinear modules already exist in the model.
        """
        from safetensors.torch import load_file

        state = load_file(path)
        missing, unexpected = self.load_state_dict(state, strict=strict)
        if len(unexpected) > 0:
            logger.warning("[FACET] Unexpected LoRA keys: %s", unexpected[:10])
        if len(missing) > 0:
            # `missing` will contain the frozen base + vae + text_encoder keys
            # when the safetensors file only stores LoRA params: expected.
            logger.info(
                "[FACET] %d missing keys when loading LoRA (frozen base, expected).",
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
            (common training path: precomputed T5 hidden states from the data
             pipeline).
          - else encode `prompt` via WAN text encoder + tokenizer.

        Output: List length B, each [L_i, 4096], L_i <= cfg.text.max_text_len.
        """
        if prompt_embeds is not None:
            if prompt is not None:
                logger.info(
                    "[FACET] Both prompt and prompt_embeds given; using prompt_embeds."
                )
            return ensure_context_list(prompt_embeds)

        if prompt is None:
            raise ValueError("Either prompt or prompt_embeds must be provided.")

        if self.text_encoder is None or self.tokenizer is None:
            raise RuntimeError("text_encoder/tokenizer not loaded.")

        prompts = [prompt] if isinstance(prompt, str) else list(prompt)

        device = next(self.text_encoder.parameters()).device

        ids, mask = self.tokenizer(
            prompts,
            return_mask=True,
            add_special_tokens=True,
        )
        ids = ids.to(device)
        mask = mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()   # real sequence length
        context = self.text_encoder(ids, mask)    # [B, L_max, 4096]
        # strip padding per sample to match WAN forward's expected list format.
        return [u[: int(l)] for u, l in zip(context, seq_lens)]

    @torch.no_grad()
    def encode_reference_image(
        self,
        reference_images: Union[Image.Image, List[Image.Image], torch.Tensor],
    ) -> torch.Tensor:
        """
        Encode reference image(s) through WAN VAE to single-frame latents.

        Accepted:
          PIL.Image / List[PIL.Image]
          Tensor [3, H, W]            -> batch 1
          Tensor [B, 3, H, W]         -> batch B
          Tensor [B, 3, 1, H, W]      -> already video-shaped, batch B
          Tensor [16, 1, h, w]        -> already latent (pre-cached), batch 1
          Tensor [B, 16, 1, h, w]     -> already latent (pre-cached), batch B

        Returns: stacked tensor [B, 16, 1, H/16, W/16].

        Resize / normalize are skipped when the input is already a Tensor; we
        assume data/transform.py:RefTfm has done it. For raw PIL, we delegate
        to DiffSynth's pipe.preprocess_video on a single-frame list.

        NOTE: RGBA alphas are dropped here. The dataset bakes the alpha into
        the reference background before storing the tensor on disk.

        NOTE: VAE runs in fp32; inputs are cast just before encode and outputs
        cast back to self.dtype.
        """
        if self.vae is None:
            raise RuntimeError("VAE not loaded.")

        ref_size = self.cfg.reference.image_size

        if isinstance(reference_images, Image.Image):
            return self._encode_ref_pil([reference_images], ref_size)

        if isinstance(reference_images, list) and all(
            isinstance(x, Image.Image) for x in reference_images
        ):
            return self._encode_ref_pil(reference_images, ref_size)

        if not isinstance(reference_images, torch.Tensor):
            raise TypeError(
                f"Unsupported reference_images type: {type(reference_images)}"
            )

        x = reference_images

        # Pre-cached latent (z_dim=16, F=1). Normalize to [B, 16, 1, h, w].
        if x.ndim == 4 and x.shape[0] == 16 and x.shape[1] == 1:
            x = x.unsqueeze(0)
        if x.ndim == 5 and x.shape[1] == 16 and x.shape[2] == 1:
            out = x.to(device=self.device, dtype=self.dtype)
            return out.detach() if self.cfg.reference.detach_latent else out

        # Raw pixel tensor. Normalize shape to [B, 3, 1, H, W].
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

        x = x.to(device=self.device, dtype=torch.float32)

        ref_latents = self.vae.encode(x, device=self.device).to(
            dtype=self.dtype, device=self.device
        )  # [B, 16, 1, H/16, W/16]

        if self.cfg.reference.detach_latent:
            ref_latents = ref_latents.detach()
        return ref_latents

    def _encode_ref_pil(
        self,
        pil_list: List[Image.Image],
        ref_size: int,
    ) -> torch.Tensor:
        """
        PIL fast path: route through pipe.preprocess_video (handles resize +
        (x-0.5)/0.5 + permute) for one-frame "videos".

        Returns: [B, 16, 1, H/16, W/16].
        """
        resized = [im.convert("RGB").resize((ref_size, ref_size)) for im in pil_list]
        # Each PIL -> a single-frame "video" [1, 3, 1, H, W].
        clips = [self.pipe.preprocess_video([im]) for im in resized]
        v = torch.cat(clips, dim=0).to(device=self.device, dtype=torch.float32)
        z = self.vae.encode(v, device=self.device).to(
            dtype=self.dtype, device=self.device
        )  # [B, 16, 1, H/16, W/16]
        if self.cfg.reference.detach_latent:
            z = z.detach()
        return z

    @torch.no_grad()
    def encode_src_video(
        self,
        src_video: Union[torch.Tensor, List[torch.Tensor]],
    ) -> torch.Tensor:
        """
        VAE-encode the masked source video into its latent representation.

        Replaces FACET-WAN2.1-VACE's encode_vace_context: there is no
        inactive/reactive split anymore. The dataloader is expected to deliver
        a single 'masked source video' (= raw video with the editing region
        replaced by gray / noise / black per data/transform.py). This function
        just lifts that pixel tensor into the WAN VAE's latent space so it can
        join the dit as an OminiControl-style src branch.

        Accepted:
          Tensor [3, F, H, W]            -> batch 1
          Tensor [B, 3, F, H, W]         -> batch B
          Tensor [16, F_lat, h, w]       -> already-latent, batch 1
          Tensor [B, 16, F_lat, h, w]    -> already-latent, batch B
          List[Tensor]                   -> stacked along batch dim

        Returns: stacked tensor [B, 16, F_lat, H/16, W/16].

        NOTE: VAE runs in fp32; inputs are cast just before encode and outputs
        cast back to self.dtype.
        """
        if self.vae is None:
            raise RuntimeError("VAE not loaded.")

        if isinstance(src_video, list):
            src_video = torch.stack(src_video, dim=0)

        if not isinstance(src_video, torch.Tensor):
            raise TypeError(f"Unsupported src_video type: {type(src_video)}")

        if src_video.ndim == 4:
            src_video = src_video.unsqueeze(0)
        if src_video.ndim != 5:
            raise ValueError(
                f"Expected src_video ndim 4 or 5, got shape {tuple(src_video.shape)}"
            )

        c = src_video.shape[1]
        z_dim = getattr(self.dit, "in_dim", 16)

        if c == z_dim:  # already-latent fast path
            out = src_video.to(device=self.device, dtype=self.dtype)
            return out.detach() if self.cfg.source.detach_latent else out

        if c != 3:
            raise ValueError(
                f"src_video should have 3 channels (pixel) or {z_dim} (latent); "
                f"got channel={c}, shape={tuple(src_video.shape)}"
            )

        src_video = src_video.to(self.device, dtype=torch.float32)
        src_lat = self.vae.encode(src_video, device=self.device).to(
            dtype=self.dtype, device=self.device
        )  # [B, 16, F_lat, H/16, W/16]

        if self.cfg.source.detach_latent:
            src_lat = src_lat.detach()
        return src_lat

    def compute_mask_coverage(
        self,
        src_mask: torch.Tensor,
        f_lat: Optional[int] = None,
        h_tok: Optional[int] = None,
        w_tok: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Pool the pixel-space src_mask into a token-grid-aligned coverage map.

        Used by the OminiControl-style mask-aware bias in facet_block_forward
        (Step 2). Roughly:

            Q_base attends to K_src with extra bias  gamma * log(eps + (1 - m))
            Q_base attends to K_ref with extra bias  gamma * log(eps + m)

        Args:
            src_mask: [B, 1, F, H, W] in [0, 1].
            f_lat, h_tok, w_tok: token grid. Defaults derived from cfg:
                f_lat = (F - 1) // vae_temporal_stride + 1
                h_tok = H // token_spatial_stride
                w_tok = W // token_spatial_stride

        Returns:
            coverage: [B, L_base] in [0, 1], where
                L_base = f_lat * h_tok * w_tok

        Flatten ordering matches dit.patch_embedding's output flatten
        (`flatten(2).transpose(1, 2)`), i.e. F -> H -> W with W as the fastest
        varying axis. This is critical: any mismatch silently shifts the bias
        relative to the corresponding K_src / K_ref token, breaking the
        editing-region prior.
        """
        if src_mask.ndim == 4:
            src_mask = src_mask.unsqueeze(1)  # [B, 1, F, H, W] when called with [B, F, H, W]
        if src_mask.ndim != 5:
            raise ValueError(
                f"src_mask should be [B, 1, F, H, W] or [B, F, H, W], "
                f"got shape {tuple(src_mask.shape)}"
            )

        _, _, F_, H_, W_ = src_mask.shape
        if f_lat is None:
            f_lat = latent_frames_from_num_frames(
                F_, temporal_stride=self.cfg.wan.vae_temporal_stride,
            )
        if h_tok is None:
            h_tok = H_ // self.cfg.wan.token_spatial_stride
        if w_tok is None:
            w_tok = W_ // self.cfg.wan.token_spatial_stride

        src_mask = src_mask.to(device=self.device, dtype=torch.float32)

        if self.cfg.source.mask_dilation > 0:
            src_mask = self._dilate_mask(
                src_mask, kernel_size=self.cfg.source.mask_dilation,
            )

        if self.cfg.source.mask_pool == "nearest":
            pooled = F.interpolate(
                src_mask, size=(f_lat, h_tok, w_tok), mode="nearest-exact",
            )
        elif self.cfg.source.mask_pool == "adaptive_avg":
            pooled = F.adaptive_avg_pool3d(
                src_mask, output_size=(f_lat, h_tok, w_tok),
            )
        else:
            raise ValueError(
                f"Unknown source.mask_pool={self.cfg.source.mask_pool!r}. "
                "Expected 'adaptive_avg' or 'nearest'."
            )

        # pooled: [B, 1, f_lat, h_tok, w_tok]
        # Flatten in F -> H -> W order to match dit.patch_embedding's output
        # flatten ordering (see DiffSynth wan_video_dit: x.flatten(2).transpose(1,2)).
        coverage = pooled.flatten(2).squeeze(1).to(self.dtype)  # [B, L_base]
        return coverage

    @staticmethod
    def _dilate_mask(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """
        Pixel-space dilation via 3D max-pooling with stride=1 and same-padding.
        Operates on [B, 1, F, H, W] tensors. Returns same shape.
        """
        k = int(kernel_size)
        if k <= 1:
            return mask
        pad = k // 2
        return F.max_pool3d(mask, kernel_size=k, stride=1, padding=pad)

    @torch.no_grad()
    def decode_latents(
        self,
        latents: Union[List[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Decode target video latents -> pixel-space tensor.

        Input:  [B, 16, F_lat, H/16, W/16] (or list form)
        Output: [B, 3, F, H, W] in [-1, 1]

        Format conversion (uint8 / PIL frames / mp4) is handled by FACETPipeline.

        NOTE: VAE is fp32-only. Latents are cast to fp32 just before decode.
        """
        latents = ensure_latent_list(latents)
        stacked = torch.stack(latents, dim=0).to(
            device=self.device, dtype=torch.float32,
        )
        video = self.vae.decode(stacked, device=self.device)
        return video

    def prepare_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        num_frames: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Sample initial Gaussian latents for the base (target) branch.

        Returns: [B, z_dim=16, F_lat, H/16, W/16].
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

        c = getattr(self.dit, "in_dim", 16)

        return torch.randn(
            batch_size, c, f_lat, h_lat, w_lat,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )

    # --------------------------------------------------------
    # RoPE / time helpers
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
        Build the 3D RoPE freqs tensor for one branch in WAN's shared rotary space.

        DiffSynth WanModel stores precomputed 1D freqs as a tuple of three
        complex tensors (f_freqs, h_freqs, w_freqs) on self.dit.freqs. With
        head_dim=128, the canonical WAN split allocates 22 complex pairs to f,
        21 to h, 21 to w (sums to head_dim/2 = 64).

        f_offset shifts the time-axis index. For FACET-WAN2.2:
            - base branch: f_offset = 0,       f range = [0..F_lat-1]
            - src  branch: f_offset = 0,       same as base (spatially-aligned;
                                               see frame.txt). cfg.source.f_offset
                                               controls this, default 0.
            - ref  branch: f_offset = 21,      single frame placed right of base
                                               (cfg.reference.f_offset default).

        All three branches consume the SAME `dit.freqs` table -> they live in a
        single shared rotary coordinate space.

        Returns:
            freqs of shape [f*h*w, 1, head_dim/2] ready to multiply with q/k.
        """
        if self.dit is None:
            raise RuntimeError("DiT not loaded.")
        f_freqs, h_freqs, w_freqs = self.dit.freqs
        if device is None:
            device = f_freqs.device

        max_f = f_freqs.shape[0]
        if f_offset + f > max_f:
            raise ValueError(
                f"f_offset({f_offset}) + f({f}) = {f_offset + f} exceeds "
                f"precomputed freqs length {max_f}."
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

    def compute_time_features(
        self,
        timestep: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute (t, t_mod) for a batch of diffusion timesteps.

        Returns:
            t:     [B, dim]     used by dit.head(x, t) (final adaln).
            t_mod: [B, 6, dim]  used by each block's modulation
                                (shift_msa, scale_msa, gate_msa,
                                 shift_mlp, scale_mlp, gate_mlp).

        Called THREE times per forward in FACET-WAN2.2:
            - timestep = real diffusion t      -> for base branch
            - timestep = cfg.source.timestep   -> for src  branch (typically 0)
            - timestep = cfg.reference.timestep-> for ref  branch (typically 0)

        Since cfg.source.timestep == cfg.reference.timestep == 0 by default,
        Step 2 will compute t_mod_clean once and share it across src/ref.
        """
        from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d

        if self.dit is None:
            raise RuntimeError("DiT not loaded.")

        dit = self.dit
        device = next(dit.parameters()).device
        timestep = timestep.to(device=device)

        t_emb = sinusoidal_embedding_1d(dit.freq_dim, timestep).to(self.dtype)
        t = dit.time_embedding(t_emb)
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
        return t, t_mod

    def _prepare_text_context(
        self,
        prompt_embeds: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Pad variable-length T5 embeddings to text_len, apply dit.text_embedding,
        return [B, text_len, dim].
        """
        text_len = self.cfg.text.max_text_len
        padded = torch.stack([
            torch.cat([u, u.new_zeros(text_len - u.size(0), u.size(1))])
            for u in prompt_embeds
        ])
        return self.dit.text_embedding(padded.to(self.device, dtype=self.dtype))

    # --------------------------------------------------------
    # Forward / Generate  (Step 2)
    # --------------------------------------------------------

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: List[torch.Tensor],
        reference_latents: torch.Tensor,
        src_video_latents: torch.Tensor,
        src_mask: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        One denoising step. Implemented in Step 2.

        Planned signature:
            noisy_latents:      [B, 16, F_lat, h, w]
            timesteps:          [B], fp32
            prompt_embeds:      List[B] of [L_i, 4096]
            reference_latents:  [B, 16, 1, h_ref, w_ref]   (REQUIRED)
            src_video_latents:  [B, 16, F_lat, h, w]       (REQUIRED, masked src)
            src_mask:           [B, 1, F, H, W] in [0, 1]  (REQUIRED, raw mask;
                                will be pooled by compute_mask_coverage internally)
            return_dict:        if True, return {"pred": ...}; else return tensor

        Asymmetric branch attention (cfg.source.attention_mode=="asymmetric"):
            Q_base attends to [K_base | K_src | K_ref], with mask-aware log-bias.
            Q_src  attends to  K_src only.
            Q_ref  attends to  K_ref only.

        Mask-aware bias (Q_base -> {K_src, K_ref}):
            on K_src: + gamma * log(eps + (1 - m))
            on K_ref: + gamma * log(eps +      m )
            where m = compute_mask_coverage(src_mask) is [B, L_base]
            broadcast appropriately.

        Output: predicted velocity (cfg.training.prediction_type="velocity"),
                shape [B, 16, F_lat, h, w].
        """
        raise NotImplementedError(
            "FACETWanModel.forward is implemented in Step 2."
        )

    @torch.no_grad()
    def generate(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        reference_image: Optional[Union[Image.Image, List[Image.Image], torch.Tensor]] = None,
        reference_latents: Optional[torch.Tensor] = None,
        src_video: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        src_video_latents: Optional[torch.Tensor] = None,
        negative_prompt: Optional[Union[str, List[str]]] = "",
        negative_prompt_embeds: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        cfg_scale: float = 1.0,
        latents: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        sigma_shift: float = 5.0,
        denoising_strength: float = 1.0,
    ) -> torch.Tensor:
        """
        Full inference loop. Implemented in Step 2.

        Required (one from each pair):
            - prompt OR prompt_embeds
            - reference_image OR reference_latents
            - (src_video AND src_mask) OR src_video_latents (+ src_mask for the bias)

        Returns: decoded video tensor [B, 3, F, H, W] in [-1, 1].
        """
        raise NotImplementedError(
            "FACETWanModel.generate is implemented in Step 2."
        )

    # --------------------------------------------------------
    # Scheduler helpers
    # --------------------------------------------------------

    def _prepare_inference_timesteps(
        self,
        num_inference_steps: int,
        sigma_shift: float = 5.0,
        denoising_strength: float = 1.0,
    ) -> torch.Tensor:
        """
        Build the FlowMatch (Wan template) inference schedule.

        sigma_shift:
            FlowMatch sigma rescaling: sigmas <- shift*s / (1 + (shift-1)*s).
            shift > 1 squeezes more steps into the high-noise region (early
            timesteps), i.e. coarser early / finer late. Wan default = 5.

        denoising_strength:
            Scales the starting sigma. 1.0 = start from pure noise (sigma~1).
            <1.0 starts partway through the schedule (img2img-style local
            denoising). FACET always uses 1.0.

        Returns CPU tensor of timesteps in [0, 1000].
        """
        self.scheduler.set_timesteps(
            num_inference_steps=num_inference_steps,
            denoising_strength=denoising_strength,
            shift=sigma_shift,
            training=False,
        )
        return self.scheduler.timesteps

    def _scheduler_step(
        self,
        pred: torch.Tensor,
        timestep: torch.Tensor,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """
        One full-batch scheduler step.
        DiffSynth's FlowMatchScheduler.step does element-wise math:
            prev = sample + model_output * (sigma_next - sigma)
        which natively supports stacked-batch tensors.
        """
        return self.scheduler.step(pred, timestep, latents)


# ============================================================
# D. Public pipeline
# ============================================================


class FACETPipeline:
    """
    User-facing inference wrapper. Adds:
      * input checks
      * deterministic seed plumbing
      * output_type post-processing (raw tensor / PIL frames)

    Step 1 keeps the call surface in place but defers actual generation
    to model.generate() (which raises NotImplementedError until Step 2).
    """

    def __init__(self, model: FACETWanModel):
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        reference_image: Optional[Union[Image.Image, List[Image.Image], torch.Tensor]] = None,
        reference_latents: Optional[torch.Tensor] = None,
        src_video: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        src_video_latents: Optional[torch.Tensor] = None,
        negative_prompt: Optional[Union[str, List[str]]] = "",
        negative_prompt_embeds: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        cfg_scale: float = 1.0,
        seed: Optional[int] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = None,
        sigma_shift: float = 5.0,
        denoising_strength: float = 1.0,
    ):
        # All required-input validation lives in `generate`.
        # Pipeline only adds: RNG plumbing + output post-processing.
        if seed is not None:
            generator = torch.Generator(device=self.model.device)
            generator.manual_seed(int(seed))
        else:
            generator = None

        video = self.model.generate(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            reference_image=reference_image,
            reference_latents=reference_latents,
            src_video=src_video,
            src_mask=src_mask,
            src_video_latents=src_video_latents,
            negative_prompt=negative_prompt,
            negative_prompt_embeds=negative_prompt_embeds,
            height=height, width=width, num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            cfg_scale=cfg_scale,
            latents=latents,
            generator=generator,
            sigma_shift=sigma_shift,
            denoising_strength=denoising_strength,
        )  # video: [B, 3, F, H, W] in [-1, 1]

        output_type = output_type or self.model.cfg.inference.output_type
        if output_type == "video":
            return video
        if output_type == "frames":
            return self._to_pil_frames(video)
        raise ValueError(
            f"Unknown output_type={output_type!r}. Expected 'video' | 'frames'."
        )

    @staticmethod
    def _to_pil_frames(video: torch.Tensor) -> List[List[Image.Image]]:
        """
        Convert a video tensor [B, 3, F, H, W] in [-1, 1] to a nested list of
        PIL frames: outer batch, inner per-frame.
        """
        v = ((video.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
        # [B, 3, F, H, W] -> [B, F, H, W, 3]
        v = v.permute(0, 2, 3, 4, 1).cpu().numpy()
        out: List[List[Image.Image]] = []
        for clip in v:
            out.append([Image.fromarray(frame) for frame in clip])
        return out


# ============================================================
# E. Utilities
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
