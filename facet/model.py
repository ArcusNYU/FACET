"""
FACET / WAN2.1-VACE-1.3B + OminiControl LoRA fine-tuning model.

Reference layout:

  base (target) branch (x):   [B, 16, F_lat=21, 60, 104]  -- WAN2.1 VAE z_dim=16
  vace branch          (c):   [B, 96, F_lat=21, 60, 104]  -- (inactive,reactive) latent + 64ch mask
  reference branch     (r):   [B, 16,         1, 30, 30]  -- single image VAE latent
  text branch          (t):   [B, 512, 4096] T5 -> 1536 dim

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
    FACETBaseConfig,     FACETInferenceConfig,
    FACETLoRAConfig,     FACETReferenceConfig,
    FACETTargetConfig,   FACETTextConfig,
    FACETTrainingConfig, FACETWanConfig,
)
from .lora import LoRALinear

from utils import (
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
                attr_name, _ = FACETConfig._SUB_CONFIGS[k] # attr_name, dc_cls
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


def _suffix_match(name: str, target_modules: Sequence[str]) -> Optional[str]:
    """Return the matched suffix or None. Suffixes can be 'q' or 'ffn.0'."""
    for tm in target_modules:
        if name == tm or name.endswith("." + tm):
            return tm
    return None


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
      - Must be under dit.blocks.* or dit.vace_blocks.*.
      - `lora_cfg.on_base_blocks` / `lora_cfg.on_vace_blocks` toggles whole regions.
      - For q/k/v/o, cross_attn is excluded unless on_cross_attn=True (default
        False to keep the current task focused on visual conditioning).
      - For vace blocks, before_proj / after_proj are automatically included.
    """
    if in_base_block and not lora_cfg.on_base_blocks:
        return False
    if in_vace_block and not lora_cfg.on_vace_blocks:
        return False
    if not (in_base_block or in_vace_block):
        return False

    matched = _suffix_match(name, target_modules)

    if matched is None:
        if in_vace_block and (
            name.endswith(".before_proj") or name.endswith(".after_proj")
        ):
            return True
        return False

    # q/k/v/o: skip cross_attn unless explicitly allowed.
    if matched in ("q", "k", "v", "o"):
        if ".cross_attn." in name and not lora_cfg.on_cross_attn:
            return False
        # Belt-and-suspenders: ensure self_attn membership unless cross_attn opt-in.
        if ".self_attn." not in name and ".cross_attn." not in name:
            return False

    return True


def inject_lora(
    root: nn.Module,
    lora_cfg: FACETLoRAConfig,
) -> List[str]:
    """
    Replace target nn.Linear modules with LoRALinear in-place.

    Returns the list of replaced module paths (for logging / debugging).

    NOTE: `root` must be the DiT (transformer) module.
    Deliberately do NOT iterate over vae / text_encoder.
    """
    replaced: List[str] = []

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
# TODO: 之后把utitlies部分移到末尾 提升model.py阅读性

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
    WAN expects List[Tensor] in several interface.

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
# D. Block-level branch attention forward
# ============================================================
# TODO: check or moved to block.py / module.py

def facet_block_forward(
    block: nn.Module,
    x_base: torch.Tensor,
    x_ref: torch.Tensor,
    t_mod_base: torch.Tensor,
    t_mod_ref: torch.Tensor,
    context: torch.Tensor,
    freqs_base: torch.Tensor,
    freqs_ref: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    OminiControl-style branch attention on top of one DiffSynth DiTBlock.

    Compared to `DiTBlock.forward(x, context, t_mod, freqs)`, this function:

      * Modulates two branches independently:
          - x_base   uses t_mod_base   (current diffusion timestep)
          - x_ref    uses t_mod_ref    (reference timestep, typically 0)
      * Self-attention uses **shared** Q/K/V/O projections (incl. LoRA) for both
        branches but builds K_all / V_all as follows:
          - base Q attends to [K_base | K_ref]  (group_mask[0] = [1, 1])
          - ref  Q attends to  K_ref            (group_mask[1] = [0, 1])
      * Cross-attention (text) is computed ONLY for the base branch.
      * FFN runs independently on each branch with its own modulation.

    Shapes:
        x_base:     [B, L_base, dim]
        x_ref:      [B, L_ref,  dim]
        t_mod_base: [B, 6, dim]
        t_mod_ref:  [B, 6, dim]
        context:    [B, L_text, dim]   (already passed through dit.text_embedding)
        freqs_base: [L_base, 1, head_dim/2]
        freqs_ref:  [L_ref,  1, head_dim/2]

    Returns:
        (x_base_out, x_ref_out) with the same shapes as the inputs.
    """
    # NOTE: late import keeps the import surface narrow.
    from diffsynth.models.wan_video_dit import modulate, rope_apply

    # ---- 1. Six-way modulation per branch ----
    def expand_chunks(t_mod: torch.Tensor):
        # t_mod: [B, 6, dim]; block.modulation: [1, 6, dim]
        m = (block.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
        return m  # tuple of 6 tensors [B, 1, dim]

    sm_b, sc_b, gm_b, sl_b, scl_b, gl_b = expand_chunks(t_mod_base)
    sm_r, sc_r, gm_r, sl_r, scl_r, gl_r = expand_chunks(t_mod_ref)

    # ---- 2. Self-attention (branch attention) ----
    sa = block.self_attn
    n_heads = sa.num_heads

    input_b = modulate(block.norm1(x_base), sm_b, sc_b)
    input_r = modulate(block.norm1(x_ref), sm_r, sc_r)

    # Shared (LoRA-augmented) q/k/v projections.
    # 计算base branch & ref banch的QKV
    q_b = sa.norm_q(sa.q(input_b))
    k_b = sa.norm_k(sa.k(input_b))
    v_b = sa.v(input_b)

    q_r = sa.norm_q(sa.q(input_r))
    k_r = sa.norm_k(sa.k(input_r))
    v_r = sa.v(input_r)

    # 3D RoPE on Q and K only, per branch with its own freqs.
    # 根据空间位置对base branch & ref branch计算RoPE
    q_b = rope_apply(q_b, freqs_base, n_heads)
    k_b = rope_apply(k_b, freqs_base, n_heads)
    q_r = rope_apply(q_r, freqs_ref, n_heads)
    k_r = rope_apply(k_r, freqs_ref, n_heads)

    # OminiControl-style branch attention:
    # Base Query attends to K_all & V_all.
    # Ref Query attends to K_ref & V_ref.
    k_all = torch.cat([k_b, k_r], dim=1)
    v_all = torch.cat([v_b, v_r], dim=1)

    y_b = sa.attn(q_b, k_all, v_all)
    y_r = sa.attn(q_r, k_r, v_r)

    y_b = sa.o(y_b)
    y_r = sa.o(y_r)

    x_base = block.gate(x_base, gm_b, y_b)
    x_ref = block.gate(x_ref, gm_r, y_r)

    # ---- 3. Cross-attention (text) - base only ----
    x_base = x_base + block.cross_attn(block.norm3(x_base), context)

    # ---- 4. FFN per branch ----
    input_b = modulate(block.norm2(x_base), sl_b, scl_b)
    input_r = modulate(block.norm2(x_ref), sl_r, scl_r)

    x_base = block.gate(x_base, gl_b, block.ffn(input_b))
    x_ref = block.gate(x_ref, gl_r, block.ffn(input_r))

    return x_base, x_ref


# ============================================================
# D. FACET model wrapper
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

        # Components assigned by _load_base_components().
        self.pipe = None          # DiffSynth WanVideoPipeline
        self.dit = None           # WanModel (with VACE blocks)
        self.vace = None          # VaceWanModel
        self.vae = None           # WanVideoVAE
        self.text_encoder = None  # WanTextEncoder (UMT5)
        self.tokenizer = None     # HuggingfaceTokenizer
        self.scheduler = None     # FlowMatchScheduler

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
        Load WAN components via DiffSynth's WanVideoPipeline.

        Strictly loads from local paths when cfg.base.load_from == "local".

        Relative paths in base.dir / base.dit / ... are resolved against the
        FACET project root (see utils._resolve_against_project_root), so that
        train.py can be launched from anywhere without breaking paths.
        """
        # Late import keeps DiffSynth's heavy deps out of module load time.
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
        self.vace = self.pipe.vace
        self.vae = self.pipe.vae
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer
        self.scheduler = getattr(self.pipe, "scheduler", None) or FlowMatchScheduler("Wan")

        if self.dit is None:
            raise RuntimeError(
                "WanVideoPipeline did not load the dit. Check that "
                f"{bcfg.dir}/{bcfg.dit} contains the diffusion model shards."
            )
        if self.vace is None:
            raise RuntimeError(
                "WanVideoPipeline did not load the VACE branch. "
                "Make sure using the VACE checkpoint (Wan2.1-VACE-1.3B/14B)."
            )

    # --------------------------------------------------------
    # Freeze / LoRA
    # --------------------------------------------------------

    def _freeze_base(self) -> None:
        """
        Freeze pipeline base components before LoRA injection.
        将基础参数冻结 并将模块设置为eval模式
        LoRA-bearing modules will re-enable grad on their own lora_down/lora_up parameters during forward; 
        Additionally re-set 'requires_grad' after _init_lora() to be explicit.

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

        Note: in DiffSynth's VaceWanModel, only `block_id == 0` carries a
        before_proj attribute (entry projection from control features into
        the hidden feature space).
        """
        replaced: List[str] = []

        if self.dit is not None and self.cfg.lora.on_base_blocks:
            replaced += [
                "dit." + n for n in inject_lora(self.dit, self.cfg.lora)
            ]

        if self.vace is not None and self.cfg.lora.on_vace_blocks:
            replaced += [
                "vace." + n for n in inject_lora(self.vace, self.cfg.lora)
            ]

        if len(replaced) == 0:
            raise RuntimeError(
                "No LoRA modules were injected. "
                f"Check lora.target_modules={list(self.cfg.lora.target_modules)} "
                "against Wan / VACE module names."
            )

        # Enable grad ONLY on lora_down / lora_up parameters.
        for name, p in self.named_parameters():
            p.requires_grad_("lora_down" in name or "lora_up" in name)

        self._lora_replaced = replaced

        logger.info("[FACET] Injected LoRA into %d modules.", len(replaced))
        # TODO: # 
        for name in replaced[:20]:
            logger.info("  - %s", name)
        if len(replaced) > 20:
            logger.info("  ... and %d more", len(replaced) - 20)

    # --------------------------------------------------------
    # Save / load LoRA
    # --------------------------------------------------------

    def save_lora(self, path: str) -> None:
        """Save only LoRA-related weights (safetensors)."""
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
        Load LoRA weights. 
        # NOTE: using after _init_lora().
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
            (training path: precomputed T5 hidden states from the data pipeline).
          - else encode `prompt` via WAN text encoder + tokenizer.

        Output: List length B, each [L_i, 4096], L_i <= cfg.text.max_text_len.
        """
        if prompt_embeds is not None:
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
        seq_lens = mask.gt(0).sum(dim=1).long() # real sequence length
        context = self.text_encoder(ids, mask)  # [B, L_max, 4096]
        # strip padding per sample to match WAN forward's expected list format.
        return [u[:int(l)] for u, l in zip(context, seq_lens)] # strip padding using ":int(l)"

    @torch.no_grad()
    def encode_reference_image(
        self,
        reference_images: Union[Image.Image, List[Image.Image], torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Encode reference image(s) through WAN VAE to single-frame latents.

        Accepted:
          PIL.Image / List[PIL.Image]
          Tensor [3, H, W]            -> batch 1
          Tensor [B, 3, H, W]         -> batch B
          Tensor [B, 3, 1, H, W]      -> already video-shaped, batch B
          Tensor [(B,) 16, 1, h, w]   -> already latent (pre-cached); returned as-is

        Returned element shape: [16, 1, H/8, W/8].

        'Resize' & 'Normalize' are not performed here since data pipeline has already done it. (data/transform.py:RefTfm). 
        For raw PIL, delegate to DiffSynth's pipe.preprocess_video on a single-frame list, which handles
        center-crop / resize / normalize to [-1, 1].

        NOTE: RGBA alphas are not consumed here.
        The dataset has already done background augmentation before storing the tensor on disk. 
        If a PIL.Image with mode RGBA is passed, the alpha channel is dropped here.
        NOTE: VAE is fp32-only.
        """

        # NOTE: 如果传入的pil/tensor是多个 那么该函数会默认是batch维度 并不具备是否是单张reference image的判断
        # 在后续的模型forward的过程中 会检测 batch size 是否与 reference image总数恰好对齐
        # 在此项目的对外展示demo中 如果用户走gradio demo 那么 reference image也会先经过SCHP得到mask 再转化为tensor传入
        # 所以此函数默认了 如果传入的类型是tensor 那么就已经是走transform进行了 resize & normalize
        # 如果用户绕开整个pre-process 直接传入PIL Image 那么就默认使用粗糙的_encode_ref_pil进行处理

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

        # Pre-cached latent (z_dim=16, F=1). Return as-is in list form.
        if x.ndim == 5 and x.shape[1] == 16 and x.shape[2] == 1:
            return ensure_latent_list(x)
        if x.ndim == 4 and x.shape[0] == 16 and x.shape[1] == 1:
            return ensure_latent_list(x)

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

        # NOTE: VAE runs in fp32.
        x = x.to(device=self.device, dtype=torch.float32)

        ref_latents = self.vae.encode(x, device=self.device).to(
            dtype=self.dtype, device=self.device
        )

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
        PIL fast path: route through pipe.preprocess_video (which handles
        resize + (x-0.5)/0.5 + permute) for one-frame "videos".
        """
        # Each PIL becomes a single-frame "video" of length 1.
        resized = [im.convert("RGB").resize((ref_size, ref_size)) for im in pil_list]
        # pipe.preprocess_video accepts a list of PIL frames; we wrap each PIL
        # individually as its own one-frame clip.
        out_latents: List[torch.Tensor] = []
        for im in resized:
            v = self.pipe.preprocess_video([im])  # [1, 3, 1, H, W]
            v = v.to(device=self.device, dtype=torch.float32)
            z = self.vae.encode(v, device=self.device).to(
                dtype=self.dtype, device=self.device
            )
            z = z[0]
            if self.cfg.reference.detach_latent:
                z = z.detach()
            out_latents.append(z)
        return out_latents

    @torch.no_grad()
    def encode_vace_context(
        self,
        src_video: Union[torch.Tensor, List[torch.Tensor]],
        src_mask: Union[torch.Tensor, List[torch.Tensor]],
    ) -> List[torch.Tensor]:
        """
        Produce the 96-channel VACE context tensor (z0 || m0) per sample.

        Layout (mirrors WAN2.1 VACE / DiffSynth):

            inactive = src_video * (1 - mask)            # background path
            reactive = src_video *      mask             # editing path

            z0 = concat(VAE(inactive), VAE(reactive), dim=channel)   # [32, F_lat, h, w]
            m0 = pixel_unshuffle_8x8(mask)                           # [64, F,    h, w]
                 -> nearest-exact temporal downsample to F_lat       # [64, F_lat, h, w]

            vace_context = concat(z0, m0, dim=channel)               # [96, F_lat, h, w]

        Input shapes:
            src_video: [B, 3, F, H, W] in [-1, 1]  OR List[Tensor]
            src_mask:  [B, 1, F, H, W] in [0, 1]   OR List[Tensor]

        Output: List length B, each [96, F_lat, H/8, W/8].
        """
        if self.vae is None:
            raise RuntimeError("VAE not loaded.")

        from einops import rearrange

        def _to_stacked(t):
            if isinstance(t, list):
                return torch.stack(t, dim=0)
            if t.ndim == 4:
                return t.unsqueeze(0)
            if t.ndim == 5:
                return t
            raise ValueError(f"Unsupported tensor ndim for VACE encode: {t.ndim}")

        src_video = _to_stacked(src_video).to(self.device, dtype=torch.float32)
        src_mask = _to_stacked(src_mask).to(self.device, dtype=torch.float32)

        # Soft mask is fine; the data pipeline produces near-binary masks already.
        inactive = src_video * (1.0 - src_mask)
        reactive = src_video * src_mask

        inactive_lat = self.vae.encode(inactive, device=self.device).to(self.dtype)
        reactive_lat = self.vae.encode(reactive, device=self.device).to(self.dtype)
        z0 = torch.cat([inactive_lat, reactive_lat], dim=1)  # [B, 32, F_lat, h, w]

        B, _, F, H, W = src_mask.shape
        # 8x8 spatial pixel-unshuffle, then nearest-exact temporal downsample.
        m_unshuf = rearrange(
            src_mask[:, 0], "b t (h p) (w q) -> b (p q) t h w", p=8, q=8
        )
        # m_unshuf: [B, 64, F, H/8, W/8]
        F_lat = (F + 3) // 4
        m0 = torch.nn.functional.interpolate(
            m_unshuf, size=(F_lat, H // 8, W // 8), mode="nearest-exact",
        ).to(self.dtype)

        vace_context = torch.cat([z0, m0], dim=1)  # [B, 96, F_lat, h, w]
        return ensure_latent_list(vace_context)

    @torch.no_grad()
    def decode_latents(
        self,
        latents: Union[List[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Decode target video latents [B, 16, F_lat, H/8, W/8] -> pixel-space tensor.

        Returns a stacked tensor [B, 3, F, H, W] in [-1, 1].
        Format conversion (uint8 / PIL frames / mp4) is handled by FACETPipeline.

        NOTE: VAE is fp32-only. Latents are cast to fp32 just before decode.
        """
        latents = ensure_latent_list(latents)
        stacked = torch.stack(latents, dim=0).to(
            device=self.device, dtype=torch.float32,
        )
        video = self.vae.decode(stacked, device=self.device)
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

        c = getattr(self.dit, "in_dim", 16)

        latents = torch.randn(
            batch_size, c, f_lat, h_lat, w_lat,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        return ensure_latent_list(latents)

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
        Build the 3D RoPE freqs tensor for one branch.
        Args:
            f, h, w   : grid size in patchified-token space.
            f_offset  : start index along the time axis. Use 0 for base/vace
                        branches and `cfg.reference.f_offset` for the reference
                        branch so the two grids do not overlap.
            device    : freqs target device. Defaults to self.dit.freqs[0].device.
        Uses dit.freqs which is precomputed by DiffSynth as a tuple of three
        complex tensors (f_freqs, h_freqs, w_freqs). With head_dim=128, the
        canonical WAN split allocates 22 complex pairs to f, 21 to h, 21 to w
        (sums to head_dim/2 = 64).
        Returns:
            freqs of shape [f*h*w, 1, head_dim/2] ready to multiply with q/k.

        f_offset shifts the time axis index, used to place the reference
        branch at f=21 so it does NOT overlap the base branch's f=[0..20].
        Both branches still consume the SAME `dit.freqs` table, ensuring they
        live in a single shared rotary coordinate space.
        """
        if self.dit is None:
            raise RuntimeError("DiT not loaded.")
        # DiffSynth WanModel stores precomputed 1D freqs as a tuple of three
        # complex tensors (f_freqs, h_freqs, w_freqs) on self.dit.freqs.
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
            t:     [B, dim]     used by `dit.head(x, t)` (final adaln).
            t_mod: [B, 6, dim]  used by each block's modulation.
        """
        # Late-bound import to avoid pulling DiffSynth at module load time.
        from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d

        if self.dit is None:
            raise RuntimeError("DiT not loaded.")

        dit = self.dit
        device = next(dit.parameters()).device
        timestep = timestep.to(device=device)

        # Cast to the dtype the rest of the dit expects.
        t_emb = sinusoidal_embedding_1d(dit.freq_dim, timestep).to(self.dtype) #freq_dim = 1024
        t = dit.time_embedding(t_emb)
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))  # [B, 6, dim]
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
    # Training forward
    # --------------------------------------------------------

    def forward(
        self,
        noisy_latents: Union[List[torch.Tensor], torch.Tensor],
        timesteps: torch.Tensor,
        prompt_embeds: Union[List[torch.Tensor], torch.Tensor],
        reference_latents: Union[List[torch.Tensor], torch.Tensor],
        vace_context: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        vace_scale: float = 1.0,
        return_dict: bool = True,
    ) -> Union[Dict[str, Any], List[torch.Tensor]]:
        """
        One denoising step (training & generation share this).

        Args:
            noisy_latents:     List of [16, F_lat, h, w]   (or stacked tensor)
            timesteps:         [B]
            prompt_embeds:     List of [L_i, 4096]         (or stacked tensor)
            reference_latents: List of [16, 1, h_ref, w_ref]  (REQUIRED for FACET)
            vace_context:      List of [96, F_lat, h, w]      (REQUIRED for VACE)
            vace_scale:        hint mixing scale (DiffSynth's vace_scale)
            return_dict:       if True, return {"pred": ...}; else return list

        Returns:
            pred: List of [16, F_lat, h, w] predictions for the BASE branch.
            Interpretation (noise vs velocity) follows cfg.training.prediction_type;
            for FlowMatch + 'velocity' the model predicts noise - sample.
        """
        # ---- 0. Normalize inputs ----
        noisy_latents = ensure_latent_list(noisy_latents)
        prompt_embeds = ensure_context_list(prompt_embeds)
        reference_latents = ensure_latent_list(reference_latents)
        if vace_context is not None:
            vace_context = ensure_latent_list(vace_context)

        B = len(noisy_latents)
        if len(reference_latents) != B:
            raise ValueError(
                f"reference batch size {len(reference_latents)} != noisy_latents {B}."
            )
        if len(prompt_embeds) != B:
            raise ValueError(
                f"prompt batch size {len(prompt_embeds)} != noisy_latents {B}."
            )
        if vace_context is not None and len(vace_context) != B:
            raise ValueError(
                f"vace_context batch size {len(vace_context)} != noisy_latents {B}."
            )
        if timesteps.shape != (B,):
            raise ValueError(
                f"timesteps shape {tuple(timesteps.shape)} does not match batch {B}."
            )

        # Cast latents to model dtype (training script may pass fp32).
        noisy_latents = [l.to(device=self.device, dtype=self.dtype) for l in noisy_latents]
        reference_latents = [l.to(device=self.device, dtype=self.dtype) for l in reference_latents]
        if vace_context is not None:
            vace_context = [c.to(device=self.device, dtype=self.dtype) for c in vace_context]

        # ---- 1. Patchify base branch ----
        # patch_embedding: Conv3d (kernel=stride=(1,2,2)) -> [1, dim, F_lat, h/2, w/2]
        x_list = [self.dit.patch_embedding(u.unsqueeze(0)) for u in noisy_latents]
        # Assume same grid across batch (true for fixed H,W,F training).
        f_base, h_base, w_base = x_list[0].shape[2:]
        x_base = torch.cat(
            [u.flatten(2).transpose(1, 2) for u in x_list], dim=0,
        )  # [B, L_base, dim]

        # ---- 2. Patchify reference branch (uses same dit.patch_embedding) ----
        r_list = [self.dit.patch_embedding(u.unsqueeze(0)) for u in reference_latents]
        f_ref, h_ref, w_ref = r_list[0].shape[2:]
        x_ref = torch.cat(
            [u.flatten(2).transpose(1, 2) for u in r_list], dim=0,
        )  # [B, L_ref, dim]

        # ---- 3. Time features ----
        t_base, t_mod_base = self.compute_time_features(timesteps)
        # Ref branch always sees a fixed clean-image timestep.
        t_ref_val = torch.full_like(
            timesteps, fill_value=float(self.cfg.reference.timestep)
        )
        _, t_mod_ref = self.compute_time_features(t_ref_val)

        # ---- 4. Text context (dit.text_embedding) ----
        context = self._prepare_text_context(prompt_embeds)

        # ---- 5. RoPE freqs per branch (same precomputed table) ----
        freqs_base = self.build_freqs(
            f_base, h_base, w_base,
            f_offset=0, device=x_base.device,
        )
        freqs_ref = self.build_freqs(
            f_ref, h_ref, w_ref,
            f_offset=self.cfg.reference.f_offset, device=x_base.device,
        )

        # ---- 6. VACE hints (vace sees only base; ref does NOT enter VACE) ----
        # NOTE: the VACE branch is gated independently of the ref branch.
        # Per design: vace_context is a List of [96, F_lat, h, w] tensors;
        # x_base / context / t_mod_base / freqs_base are stacked tensors.
        gc_enabled = bool(self.cfg.gradient_checkpointing) and self.training
        vace_hints = None
        if vace_context is not None:
            vace_hints = self.vace(
                x_base, vace_context, context, t_mod_base, freqs_base,
                use_gradient_checkpointing=gc_enabled,
                use_gradient_checkpointing_offload=False,
            )

        # ---- 7. Iterate base DiT blocks (custom branch attention) ----
        for block_id, block in enumerate(self.dit.blocks):
            if gc_enabled:
                x_base, x_ref = torch.utils.checkpoint.checkpoint(
                    facet_block_forward,
                    block, x_base, x_ref,
                    t_mod_base, t_mod_ref, context,
                    freqs_base, freqs_ref,
                    use_reentrant=False,
                )
            else:
                x_base, x_ref = facet_block_forward(
                    block, x_base, x_ref,
                    t_mod_base, t_mod_ref, context,
                    freqs_base, freqs_ref,
                )

            # VACE hint injection (after the block, before next).
            if vace_hints is not None and block_id in self.vace.vace_layers_mapping:
                hint = vace_hints[self.vace.vace_layers_mapping[block_id]]
                x_base = x_base + hint * vace_scale

        # ---- 8. Head (base branch only) + unpatchify ----
        x_base = self.dit.head(x_base, t_base)  # [B, L_base, out_dim * prod(patch_size)]
        out = self.dit.unpatchify(x_base, (f_base, h_base, w_base))
        # out: [B, 16, F_lat, H, W]

        pred = ensure_latent_list(out)

        if return_dict:
            return {"pred": pred}
        return pred

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        reference_image: Optional[Union[Image.Image, List[Image.Image], torch.Tensor]] = None,
        src_video: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        src_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        prompt_embeds: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = "",
        negative_prompt_embeds: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        vace_scale: float = 1.0,
        latents: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        vace_context: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        generator: Optional[torch.Generator] = None,
        sigma_shift: float = 5.0,
        denoising_strength: float = 1.0,
    ) -> torch.Tensor:
        """
        Full inference loop. Returns the decoded video tensor [B, 3, F, H, W] in [-1, 1].

        FACET requires reference_image + (src_video, src_mask) OR a pre-computed
        vace_context tensor list.

        FACETPipeline wraps this with input validation / RNG plumbing /
        output_type post-processing.

        NOTE: KV-cache for the reference branch (cfg.reference.kv_cache_reference)
        is NOT implemented yet; ref tokens are re-computed every step. The
        forward interface will not change when caching is added later.
        """
        # ---- 0. Resolve defaults ----
        height = height or self.cfg.target.height
        width = width or self.cfg.target.width
        num_frames = num_frames or self.cfg.target.num_frames
        num_inference_steps = num_inference_steps or self.cfg.inference.num_inference_steps
        cfg_scale = cfg_scale if cfg_scale is not None else self.cfg.inference.cfg_scale

        validate_video_size(
            height=height, width=width, num_frames=num_frames,
            hw_multiple=self.cfg.target.hw_multiple,
            temporal_stride=self.cfg.wan.vae_temporal_stride,
        )

        # ---- 1. Prompt embeddings (cond + uncond if doing CFG) ----
        cond_context = self.encode_prompt(prompt=prompt, prompt_embeds=prompt_embeds)
        do_cfg = cfg_scale is not None and cfg_scale > 1.0
        if do_cfg:
            uncond_context = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
            )
        else:
            uncond_context = None

        batch_size = len(cond_context)

        def _broadcast_to_batch(lst, name):
            if len(lst) == batch_size:
                return lst
            if len(lst) == 1:
                return lst * batch_size
            raise ValueError(
                f"{name} batch size {len(lst)} cannot be broadcast to {batch_size}."
            )

        # ---- 2. Reference latents (required) ----
        if reference_image is None:
            raise ValueError("reference_image is required for FACET inference.")
        ref_latents = self.encode_reference_image(reference_image)
        ref_latents = _broadcast_to_batch(ref_latents, "reference")

        # ---- 3. VACE context (build from src_video + src_mask if not given) ----
        if vace_context is not None:
            vace_ctx_list = ensure_latent_list(vace_context)
            vace_ctx_list = _broadcast_to_batch(vace_ctx_list, "vace_context")
        else:
            if src_video is None or src_mask is None:
                raise ValueError(
                    "FACET inference requires either vace_context, or "
                    "(src_video AND src_mask) to build it."
                )
            vace_ctx_list = self.encode_vace_context(src_video, src_mask)
            vace_ctx_list = _broadcast_to_batch(vace_ctx_list, "vace_context")

        # ---- 4. Initial noisy latents ----
        if latents is None:
            cur_latents = self.prepare_latents(
                batch_size=batch_size,
                height=height, width=width, num_frames=num_frames,
                generator=generator,
            )
        else:
            cur_latents = ensure_latent_list(latents)

        # ---- 5. Scheduler timesteps ----
        timesteps = self._prepare_inference_timesteps(
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            denoising_strength=denoising_strength,
        )

        # ---- 6. Denoising loop ----
        for t in timesteps:
            t_batch = torch.full(
                (batch_size,), float(t),
                device=self.device, dtype=torch.float32,
            )

            pred_cond = self.forward(
                noisy_latents=cur_latents, timesteps=t_batch,
                prompt_embeds=cond_context, reference_latents=ref_latents,
                vace_context=vace_ctx_list, vace_scale=vace_scale,
                return_dict=True,
            )["pred"]

            if do_cfg:
                pred_uncond = self.forward(
                    noisy_latents=cur_latents, timesteps=t_batch,
                    prompt_embeds=uncond_context, reference_latents=ref_latents,
                    vace_context=vace_ctx_list, vace_scale=vace_scale,
                    return_dict=True,
                )["pred"]
                pred = [
                    u + cfg_scale * (c - u)
                    for c, u in zip(pred_cond, pred_uncond)
                ]
            else:
                pred = pred_cond

            cur_latents = self._scheduler_step(
                pred=pred, timestep=t, latents=cur_latents,
            )

        # ---- 7. Decode final latents ----
        return self.decode_latents(cur_latents)

    def _prepare_inference_timesteps(
        self,
        num_inference_steps: int,
        sigma_shift: float = 5.0,
        denoising_strength: float = 1.0,
    ) -> torch.Tensor:
        """
        Build the FlowMatch (Wan template) inference schedule.

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
        pred: List[torch.Tensor],
        timestep: torch.Tensor,
        latents: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """One scheduler step per batch element (DiffSynth's step is per-sample)."""
        return [self.scheduler.step(p, timestep, l) for p, l in zip(pred, latents)]


# ============================================================
# E. Public pipeline
# ============================================================


class FACETPipeline:
    """
    User-facing inference wrapper. Adds:
      * input checks
      * deterministic seed plumbing
      * output_type post-processing (raw tensor / PIL frames)
    """

    def __init__(self, model: FACETWanModel):
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        reference_image: Optional[Union[Image.Image, List[Image.Image], torch.Tensor]] = None,
        src_video: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        src_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        prompt_embeds: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = "",
        negative_prompt_embeds: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        vace_scale: float = 1.0,
        seed: Optional[int] = None,
        latents: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        vace_context: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        output_type: Optional[str] = None,
        sigma_shift: float = 5.0,
        denoising_strength: float = 1.0,
    ):
        if prompt is None and prompt_embeds is None:
            raise ValueError("Either prompt or prompt_embeds must be provided.")

        if reference_image is None:
            raise ValueError("reference_image is required for FACET inference.")

        if vace_context is None and (src_video is None or src_mask is None):
            raise ValueError(
                "Either vace_context, or (src_video AND src_mask) must be provided."
            )

        # Seed -> generator
        if seed is not None:
            generator = torch.Generator(device=self.model.device)
            generator.manual_seed(int(seed))
        else:
            generator = None

        video = self.model.generate(
            prompt=prompt,
            reference_image=reference_image,
            src_video=src_video,
            src_mask=src_mask,
            prompt_embeds=prompt_embeds,
            negative_prompt=negative_prompt,
            negative_prompt_embeds=negative_prompt_embeds,
            height=height, width=width, num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            cfg_scale=cfg_scale,
            vace_scale=vace_scale,
            latents=latents,
            vace_context=vace_context,
            generator=generator,
            sigma_shift=sigma_shift,
            denoising_strength=denoising_strength,
        )
        # video: [B, 3, F, H, W] in [-1, 1]

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
        # [-1, 1] -> [0, 255]
        v = ((video.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
        # [B, 3, F, H, W] -> [B, F, H, W, 3]
        v = v.permute(0, 2, 3, 4, 1).cpu().numpy()
        out: List[List[Image.Image]] = []
        for clip in v:
            out.append([Image.fromarray(frame) for frame in clip])
        return out
