"""
FACET / WAN2.2-TI2V-5B + OminiControl LoRA fine-tuning model.

Reference layout (WAN2.2-TI2V-5B, 480x832x81):

  base (target) branch (x):  [B, 48, F_lat=21, 30, 52]  -- WAN2.2 VAE z_dim=48, vae_stride=16
                             -- patchify (k=s=(1,2,2)) -> token grid [B, 21, 15, 26], L_base = 8190

  src branch           (s):  [B, 48, F_lat=21, 30, 52]  -- masked source video latent
                             -- shares dit.patch_embedding -> same token grid as base, L_src = 8190
                             -- RoPE: shared with base (f=0..20, h=0..14, w=0..25)

  reference branch     (r):  [B, 48, F_lat=1, 30, 30]   -- single image VAE latent (480x480)
                             -- shares dit.patch_embedding -> [B, 1, 15, 15], L_ref = 225
                             -- RoPE: f-axis DISABLED (delta_f=0 with base);
                                      h=0..14  (no h-axis offset),
                                      w=26..40 (placed right of base, w_offset=26 for 480x832)

  text branch          (t):  [B, 512, 4096]  T5 hidden states  ->  text_embedding  ->  [B, 512, 3072]

Hidden dims (WAN2.2-TI2V-5B):
    dim=3072, num_heads=24, head_dim=128, ffn_dim=14336, num_layers=30, freq_dim=256.
    head_dim/2 = 64 complex pairs split as 22(f) + 21(h) + 21(w) for 3D RoPE.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .config import FACETConfig
from .lora import inject_lora

from utils import (
    _resolve_dtype,
    _resolve_local_path,
    _resolve_local_paths,
)


logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _suppress_stdout(enabled: bool = True):
    """Silence stdout while DiffSynth loads its model components.
    """
    if not enabled or os.environ.get("FACET_DIFFSYNTH_VERBOSE") == "1":
        yield
        return
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        yield


# ============================================================
# A. OminiControl-style block forward
# ============================================================
# Three branches (base, src, ref) sharing the SAME q/k/v/o/ffn weights and
# their LoRA adapters, but with independent AdaLN modulation and RoPE phase.
#
# Attention rules (asymmetric mode):
#
#   |          | K_base | K_src       | K_ref       |
#   |----------|--------|-------------|-------------|
#   | Q_base   |   1    |  1 + bias_s |  1 + bias_r |
#   | Q_src    |   0    |  1          |  0          |
#   | Q_ref    |   0    |  0          |  1          |
#
# Mask-aware biases (routing mechanism for Q_base):
#   m_q       = mask_coverage[:, None, :, None]                # [B, 1, L_base, 1]
#   bias_src  = gamma * log( (1 - m_q).clamp_min(eps) )        # suppresses src where m_q=1
#   bias_ref  = gamma * log(      m_q.clamp_min(eps) )         # suppresses ref where m_q=0
#
# RoPE rules:
#   Q_base @ K_base, Q_base @ K_src     ->  full (f, h, w) RoPE   (src shares base coords)
#   Q_base @ K_ref                      ->  delta_f = 0
#                                           ("disable f-RoPE" = use f_freqs[0:1] = identity)
#   Q_src @ K_src                       ->  full (f, h, w) RoPE
#   Q_ref @ K_ref                       ->  (h, w) RoPE only  (ref has 1 frame so it equals to fulL RoPE)


def facet_block_forward(
    block: nn.Module,
    x_base: torch.Tensor,
    x_src: torch.Tensor,
    x_ref: torch.Tensor,
    t_mod_base: torch.Tensor,
    t_mod_cond: torch.Tensor,
    context: torch.Tensor,
    freqs_base: torch.Tensor,
    freqs_src: torch.Tensor,
    freqs_ref: torch.Tensor,
    mask_coverage: torch.Tensor,
    gamma: float = 1.0,
    safe_epsilon: float = 1e-3,
    bias_floor: float = -1e4,
    attention_mode: str = "asymmetric",
    q_chunk: int = 1024,
    mask_bias: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward Logic for FACET-WAN DiT block: OminiControl-style 3-branch attention.

    Args:
        block             : DiffSynth WanModel DiTBlock (in dit.blocks)
        x_base            : [B, L_base, dim]
        x_src             : [B, L_src,  dim]    (L_src == L_base)
        x_ref             : [B, L_ref,  dim]
        t_mod_base        : [B, 6, dim]   per-batch modulation for base (current t)
        t_mod_cond        : [B, 6, dim]   per-batch modulation for src AND ref (t=0)
        context           : [B, L_text, dim]   already passed through dit.text_embedding
        freqs_base        : [L_base, 1, head_dim/2]  full (f, h, w) RoPE  (f=[0..F-1])
        freqs_src         : [L_src,  1, head_dim/2]  full (f, h, w) RoPE  (shares base coords)
        freqs_ref         : [L_ref,  1, head_dim/2]  RoPE with w_offset; ref has 1 frame so its
                                                     f-component is naturally f_freqs[0:1] = identity.
        mask_coverage     : [B, L_base]  in [0, 1], token-grid-aligned soft mask
        gamma             : scalar bias scale
        safe_epsilon      : numerical floor inside log
        bias_floor        : clamp lower bound after log (-1e4 keeps bf16 stable)
        attention_mode    : "asymmetric" (default; only supported in the current version)

    Inside the block we additionally derive `freqs_base_f0` from `freqs_base` by
    overwriting its f-axis sub-channels with the identity rotation (1+0j). It is
    consumed ONLY in the Q_base -> K_ref pathway so the f-phase contribution
    cancels out (delta_f = 0 between base and ref).

    Returns:
        (x_base_out, x_src_out, x_ref_out) - same shapes as inputs.
    """
    if attention_mode != "asymmetric":
        raise NotImplementedError(
            f"attention_mode={attention_mode!r} is not implemented yet. "
        )

    # Late imports avoid pulling DiffSynth at module load time.
    from diffsynth.models.wan_video_dit import modulate, rope_apply

    # ---- 1. Six-way modulation per branch ----
    # block.modulation: [1, 6, dim]; t_mod_*: [B, 6, dim] -> chunk(6, dim=1) -> 6 x [B, 1, dim]
    def expand_chunks(t_mod: torch.Tensor):
        return (block.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)

    sm_b, sc_b, gm_b, sl_b, scl_b, gl_b = expand_chunks(t_mod_base)
    sm_c, sc_c, gm_c, sl_c, scl_c, gl_c = expand_chunks(t_mod_cond)   # used for both src and ref

    sa = block.self_attn
    n_heads = sa.num_heads # 24
    head_dim = sa.head_dim # 128
    # dim = sa.dim           # 3072

    # ---- 2. Pre-attention modulation ----
    input_b = modulate(block.norm1(x_base), sm_b, sc_b)
    input_s = modulate(block.norm1(x_src),  sm_c, sc_c)
    input_r = modulate(block.norm1(x_ref),  sm_c, sc_c)

    # ---- 3. Q/K/V projections (shared weights & LoRA across branches) ----
    q_b = sa.norm_q(sa.q(input_b))
    k_b = sa.norm_k(sa.k(input_b))
    v_b = sa.v(input_b)

    q_s = sa.norm_q(sa.q(input_s))
    k_s = sa.norm_k(sa.k(input_s))
    v_s = sa.v(input_s)

    q_r = sa.norm_q(sa.q(input_r))
    k_r = sa.norm_k(sa.k(input_r))
    v_r = sa.v(input_r)

    # ---- 4. Build freqs_base_f0 from freqs_base ----
    #   position index 0  ->  exp(i * 0 * theta_k)  =  1 + 0j  (identity rotation)
    # In WAN's precompute (DiffSynth wan_video_dit.precompute_freqs_cis):
    #     freqs_cis[pos, k] = torch.polar(1.0, pos * theta_k)
    # so freqs_cis[0, k] = (cos 0) + i (sin 0) = 1 + 0j for every channel k.
    #
    # Layout of freqs_* along the last dim:
    #     [ f-channels (length c_f) | h-channels (c_h) | w-channels (c_w) ]
    # For WAN's split, c_f = (head_dim - 2*(head_dim // 3)) // 2.
    # head_dim=128  ->  c_f = 22, c_h = c_w = 21, total = 64 = head_dim/2.
    #
    # Result: Q_base sees every K_ref token without temporal misalignment.
    c_f = (head_dim - 2 * (head_dim // 3)) // 2
    freqs_base_f0 = freqs_base.clone()
    # For complex tensors, torch.ones_like returns 1+0j with the same complex dtype.
    freqs_base_f0[..., :c_f] = torch.ones_like(freqs_base_f0[..., :c_f])

    # ---- 5. Apply RoPE ----
    q_b      = rope_apply(q_b, freqs_base,    n_heads)
    q_b_f0   = rope_apply(q_b, freqs_base_f0, n_heads)
    k_b      = rope_apply(k_b, freqs_base,    n_heads)

    q_s = rope_apply(q_s, freqs_src, n_heads)
    k_s = rope_apply(k_s, freqs_src, n_heads)

    q_r = rope_apply(q_r, freqs_ref, n_heads)
    k_r = rope_apply(k_r, freqs_ref, n_heads)

    # ---- 5. Base Branch: Q_base global attention with mask-aware bias ----
    # Customized attention logic for FACET base branch:
    # Compute scores per K-block then concat + global softmax, because
    # (a) per-K-block biases need to be added BEFORE softmax;
    # (b) Q_base uses different RoPE for K_ref vs K_base/K_src.
    B = x_base.shape[0]
    L_base = x_base.shape[1]

    def to_heads(t: torch.Tensor) -> torch.Tensor:
        # [B, S, n*d] -> [B, n, S, d]
        return t.view(B, -1, n_heads, head_dim).transpose(1, 2)

    q_b_h = to_heads(q_b)
    q_b_f0_h = to_heads(q_b_f0)
    k_b_h = to_heads(k_b)
    v_b_h = to_heads(v_b)
    k_s_h = to_heads(k_s)
    v_s_h = to_heads(v_s)
    k_r_h = to_heads(k_r)
    v_r_h = to_heads(v_r)

    scale = 1.0 / math.sqrt(head_dim)

    # Mask-aware bias: per-Q-base routing.
    
    # NOTE: ABLATION: mask_bias=False -> bias_src/bias_ref stay None, so Q_base attends
    # K_src/K_ref with NO routing prior (plain learned 3-branch attention). This
    # is the ablation arm for the OminiControl mask bias (see facet/config.yaml).

    bias_src = None
    bias_ref = None
    if mask_bias:
    # m_q = m[:, None, :, None]  -> [B, 1, L_base, 1]; broadcasts over heads and K dim.
        m_q = mask_coverage.to(v_b_h.dtype)[:, None, :, None]

        m_q_reverse = (1.0 - m_q).clamp_min(safe_epsilon)
        m_q = m_q.clamp_min(safe_epsilon)

        bias_src = (gamma * torch.log(m_q_reverse)).clamp_min(bias_floor)   # [B, 1, L_base, 1]
        bias_ref = (gamma * torch.log(m_q)        ).clamp_min(bias_floor)   # [B, 1, L_base, 1]

    v_all = torch.cat([v_b_h, v_s_h, v_r_h], dim=2)   # [B, n_heads, L_all, head_dim]

    # NOTE: Memory: a full score [B, n_heads, L_base, 2*L_base + L_ref] plus its fp32 softmax
    # copy is O(L_base^2) and dominates VRAM (e.g. L_base=8190 -> ~12 GiB for the fp32
    # score alone). Chunk the QUERY axis so peak memory scales with q_chunk instead of
    # L_base; the math is identical (each query row still does a global softmax over the
    # full [K_base | K_src | K_ref] axis). q_chunk <= 0 disables chunking.
    step = L_base if q_chunk is None or q_chunk <= 0 else int(q_chunk)
    y_parts: List[torch.Tensor] = []
    for i in range(0, L_base, step):
        j = min(i + step, L_base)
        qb_c = q_b_h[:, :, i:j]                                          # [B, n, c, d]
        qf0_c = q_b_f0_h[:, :, i:j]

        # Scores for this query chunk: [B, n_heads, c, L_*]
        s_base = (qb_c  @ k_b_h.transpose(-2, -1)) * scale
        s_src  = (qb_c  @ k_s_h.transpose(-2, -1)) * scale
        s_ref  = (qf0_c @ k_r_h.transpose(-2, -1)) * scale            # delta_f = 0
        if mask_bias:
            s_src = s_src + bias_src[:, :, i:j]
            s_ref = s_ref + bias_ref[:, :, i:j]

        # Global softmax over [K_base | K_src | K_ref] in fp32 for stability.
        s = torch.cat([s_base, s_src, s_ref], dim=-1)
        a = torch.softmax(s.float(), dim=-1).to(v_b_h.dtype)
        y_parts.append(a @ v_all)                                        # [B, n_heads, c, head_dim]

    y_b = torch.cat(y_parts, dim=2)                                      # [B, n_heads, L_base, head_dim]

    # Back to [B, L_base, n_heads*head_dim]
    y_b = y_b.transpose(1, 2).reshape(B, L_base, n_heads * head_dim)
    y_b = sa.o(y_b)

    # ---- 7. Src Branch: Q_src self-attention (standard, flash-able) ----
    y_s = sa.attn(q_s, k_s, v_s)
    y_s = sa.o(y_s)

    # ---- 8. Ref Branch: Q_ref self-attention (standard, flash-able) ----
    y_r = sa.attn(q_r, k_r, v_r)
    y_r = sa.o(y_r)

    # ---- 9. Gate (residual + attn output) ----
    x_base = block.gate(x_base, gm_b, y_b)
    x_src  = block.gate(x_src,  gm_c, y_s)
    x_ref  = block.gate(x_ref,  gm_c, y_r)

    # ---- 10. Cross-attention (text) - Base Branch only ----
    x_base = x_base + block.cross_attn(block.norm3(x_base), context)

    # ---- 11. FFN per branch ----
    input_b = modulate(block.norm2(x_base), sl_b, scl_b)
    input_s = modulate(block.norm2(x_src),  sl_c, scl_c)
    input_r = modulate(block.norm2(x_ref),  sl_c, scl_c)

    x_base = block.gate(x_base, gl_b, block.ffn(input_b))
    x_src  = block.gate(x_src,  gl_c, block.ffn(input_s))
    x_ref  = block.gate(x_ref,  gl_c, block.ffn(input_r))

    return x_base, x_src, x_ref


# ============================================================
# B. FACET Wan Model wrapper
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
        self.dit = None           # WanModel (TI2V-5B)
        self.vae = None           # WanVideoVAE (Wan2.2 VAE, stride 16)
        self.text_encoder = None  # WanTextEncoder (UMT5)
        self.tokenizer = None     # HuggingfaceTokenizer
        self.scheduler = None     # FlowMatchScheduler ("Wan" template)

        self._load_base_components()
        self._freeze_base()

        self._lora_replaced: List[str] = []
        self._init_lora()

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
        """
        from diffsynth.pipelines.wan_video import (
            WanVideoPipeline,
            ModelConfig as DSModelConfig,
        )
        from diffsynth.diffusion import FlowMatchScheduler

        bcfg = self.cfg.base

        if bcfg.load_from == "local":
            dit_paths = _resolve_local_paths(bcfg.dir, bcfg.dit)
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

        # DiffSynth prints a large block of load messages to stdout here; mute.
        with _suppress_stdout():
            self.pipe = WanVideoPipeline.from_pretrained(
                torch_dtype=self.dtype,
                device=self.device,
                model_configs=model_configs,
                tokenizer_config=tokenizer_config,
            )
        logger.info("[FACET] WanVideoPipeline loaded (dit/t5/vae) from %s", bcfg.dir)

        self.dit = self.pipe.dit
        self.vae = self.pipe.vae
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer

        # (the VAE is frozen, so DDP/AMP never recast it back).
        if self.vae is not None:
            self.vae.to(torch.float32)
        # DiffSynth's WanVideoPipeline pre-instantiates a FlowMatchScheduler.
        self.scheduler = getattr(self.pipe, "scheduler", None) or FlowMatchScheduler("Wan")

        if self.dit is None:
            raise RuntimeError(
                "WanVideoPipeline did not load the dit. Check that "
                f"{bcfg.dir}/{bcfg.dit} contains the diffusion model shards."
            )

        # Sanity-check on backbone loading for TI2V-5B.
        # TI2V-5B uses dim=3072; warn if it diverges so the user notices a wrong checkpoint.
        if getattr(self.dit, "dim", None) != 3072:
            logger.warning(
                "[FACET] dit.dim=%s; expected 3072 for Wan2.2-TI2V-5B. "
                "If using a different base (e.g. 14B), update configs accordingly.",
                getattr(self.dit, "dim", None),
            )

    # --------------------------------------------------------
    # Freeze / LoRA
    # --------------------------------------------------------

    def _freeze_base(self) -> None:
        """
        Freeze pipeline base components and set to eval mode.

        Called BEFORE _init_lora.
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

        self._lora_replaced = replaced

        logger.info("[FACET] Injected LoRA into %d modules.", len(replaced))
        # TODO: 打印lora的模块名 可以删除 仅用于debug
        # for name in replaced[:20]:
        #    logger.info("  - %s", name)
        # if len(replaced) > 20:
        #    logger.info("  ... and %d more", len(replaced) - 20)

    def set_lora(self, trainable: bool = True) -> None:
        """
        Optional helper for the trainer: re-set requires_grad on lora_down/lora_up.

        Not called from __init__ on purpose; the default PyTorch behavior after
        injection already matches `trainable=True`. Call this explicitly if you
        ever need to freeze LoRA (e.g. for a sanity-check inference of the
        loaded weights, where requires_grad=False saves a tiny bit of bookkeeping).
        """
        for name, p in self.named_parameters():
            if ("lora_down" in name) or ("lora_up" in name):
                p.requires_grad_(bool(trainable))

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
            raise RuntimeError("No LoRA params found in state_dict.")

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        save_file(state, path)
        logger.info("[FACET] Saved %d LoRA tensors -> %s", len(state), path)

    def load_lora(self, path: str, strict: bool = False) -> None:
        """
        Load LoRA weights. Call AFTER _init_lora() so the matching LoRALinear
        modules already exist in the model.
        """
        from safetensors.torch import load_file

        state = load_file(path)
        missing, unexpected = self.load_state_dict(state, strict=strict)
        if len(unexpected) > 0:
            logger.warning("[FACET] Unexpected LoRA keys: %s", unexpected[:10])
        if len(missing) > 0:
            # Expected when safetensors only stores LoRA params: 
            # frozen base + vae + text_encoder keys show up here.
            logger.info(
                "[FACET] %d missing keys when loading LoRA (frozen base, expected).",
                len(missing),
            )


    # --------------------------------------------------------
    # Training forward
    # --------------------------------------------------------

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: List[torch.Tensor],
        ref_latents: torch.Tensor,
        src_latents: torch.Tensor,
        src_mask: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        One denoising step (shared by training & generation).

        Args:
            noisy_latents:  [B, 48, F_lat, h, w]                       (REQUIRED)
            timesteps:      [B]  fp32                                  (REQUIRED)
            prompt_embeds:  List length B, each [L_i, 4096]            (REQUIRED)
            ref_latents:    [B, 48, 1, h_ref, w_ref]                   (REQUIRED)
            src_latents:    [B, 48, F_lat, h, w]                       (REQUIRED)
            src_mask:       [B, 1, F, H, W] in [0, 1]                  (REQUIRED)
            return_dict:    if True returns {"pred": ...}, else returns tensor

        Returns:
            pred: [B, 48, F_lat, h, w]  velocity-target prediction.

        NOTE: All tensor inputs (noisy_latents, ref_latents, src_latents,
        src_mask, prompt_embeds elements) MUST already live on `self.device`
        with dtype = `self.dtype` (= accelerate's mixed-precision dtype,
        typically bf16 on A100). The caller (train.py / generate) is responsible.
        """
        B = noisy_latents.shape[0]

        # ---- 1. Patchify base branch ----
        # patch_embedding: Conv3d (k=s=(1,2,2)): [B, 48, F_lat, H/16, W/16]
        #   -> [B, dim, F_lat, H/32, W/32]
        x_base = self.dit.patch_embedding(noisy_latents)
        f_base, h_base, w_base = x_base.shape[2:]
        x_base = x_base.flatten(2).transpose(1, 2)
        # x_base: [B, L_base, dim]    L_base = f_base * h_base * w_base

        # ---- 2. Patchify src branch (shares dit.patch_embedding) ----
        x_src = self.dit.patch_embedding(src_latents)
        f_src, h_src, w_src = x_src.shape[2:]
        x_src = x_src.flatten(2).transpose(1, 2)
        # x_src: [B, L_src, dim]; spatially aligned with base: L_src == L_base.

        # ---- 3. Patchify ref branch (shares dit.patch_embedding) ----
        x_ref = self.dit.patch_embedding(ref_latents)
        f_ref, h_ref, w_ref = x_ref.shape[2:]
        x_ref = x_ref.flatten(2).transpose(1, 2)
        # x_ref: [B, L_ref, dim];  L_ref = 1 * h_ref * w_ref

        # ---- 4. Time features ----
        # base uses the actual diffusion timestep; src and ref share t=0.
        t_base, t_mod_base = self.compute_time_features(timesteps)
        # t_base: [B, dim] -> head(t_base); t_mod_base: [B, 6, dim] -> dit.blocks(t_mod_base)
        t_cond = torch.full_like(
            timesteps, fill_value=float(self.cfg.source.timestep),
        )
        _, t_mod_cond = self.compute_time_features(t_cond)

        # ---- 5. Text context (pad + dit.text_embedding -> [B, 512, dim]) ----
        context = self._prepare_text_context(prompt_embeds)

        # ---- 6. RoPE freqs per branch ----
        # All three share the SAME dit.freqs table (a single shared rotary
        # coordinate space). Their differences are just which f/h/w slice they
        # use. The "delta_f = 0" special case for Q_base -> K_ref is handled
        # INSIDE facet_block_forward by deriving freqs_base_f0 from freqs_base.
        freqs_base = self.build_freqs(
            f_base, h_base, w_base,
            f_offset=0, h_offset=0, w_offset=0,
            device=self.device,
        )
        freqs_src = self.build_freqs(
            f_src, h_src, w_src,
            f_offset=int(self.cfg.source.f_offset),
            h_offset=0, w_offset=0,
            device=self.device,
        )
        freqs_ref = self.build_freqs(
            f_ref, h_ref, w_ref,
            f_offset=0,
            h_offset=int(self.cfg.reference.h_offset),
            w_offset=int(self.cfg.reference.w_offset),
            device=self.device,
        )

        # ---- 7. Mask coverage (Q_base routing prior) ----
        # [B, L_base], same flatten order as patch_embedding output.
        mask_coverage = self.compute_mask_coverage(
            src_mask,
            f_lat=f_base, h_tok=h_base, w_tok=w_base,
        )

        gamma = float(self.cfg.source.gamma)
        safe_epsilon = float(self.cfg.source.safe_epsilon)
        attention_mode = self.cfg.source.attention_mode
        # Ablation switch: when False, skip the mask-aware routing bias entirely.
        mask_bias = bool(getattr(self.cfg.source, "mask_bias", True))
        # Query-chunk size for the base-branch attention (caps O(L^2) VRAM). Tunable
        # via cfg.source.q_chunk; <=0 disables chunking (full-length attention).
        q_chunk = int(getattr(self.cfg.source, "q_chunk", 2048)) # for A100-80GB

        # ---- 8. Iterate DiT blocks with custom branch attention ----
        gc_enabled = bool(self.cfg.gradient_checkpointing) and self.training

        # NOTE: q_chunk is always enabled in case of OOM.(at least needed on A100-80GB)
        for block in self.dit.blocks:
            if gc_enabled:
                x_base, x_src, x_ref = torch.utils.checkpoint.checkpoint(
                    facet_block_forward,
                    block, x_base, x_src, x_ref,
                    t_mod_base, t_mod_cond, context,
                    freqs_base, freqs_src, freqs_ref,
                    mask_coverage,
                    gamma=gamma,
                    safe_epsilon=safe_epsilon,
                    attention_mode=attention_mode,
                    q_chunk=q_chunk,
                    mask_bias=mask_bias,
                    use_reentrant=False,
                )
            else:
                x_base, x_src, x_ref = facet_block_forward(
                    block, x_base, x_src, x_ref,
                    t_mod_base, t_mod_cond, context,
                    freqs_base, freqs_src, freqs_ref,
                    mask_coverage,
                    gamma=gamma,
                    safe_epsilon=safe_epsilon,
                    attention_mode=attention_mode,
                    q_chunk=q_chunk,
                    mask_bias=mask_bias,
                )

        # ---- 9. Head (base branch only) + unpatchify ----
        # OminiControl convention: src and ref are 'register' branches; only
        # base produces the noise/velocity prediction.
        x_base = self.dit.head(x_base, t_base)
        out = self.dit.unpatchify(x_base, (f_base, h_base, w_base))

        if return_dict:
            return {"pred": out}
        return out

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        reference_images: Optional[Union[Image.Image, List[Image.Image], torch.Tensor]] = None,
        ref_latents: Optional[torch.Tensor] = None,
        src_video: Optional[torch.Tensor] = None,
        src_latents: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
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
        Full inference loop. Returns decoded video tensor [B, 3, F, H, W] in [-1, 1].

        Required (one from each pair):
            - prompt OR prompt_embeds
            - reference_images OR ref_latents
            - (src_video AND src_mask)   OR  (src_latents AND src_mask)

        Batch-size resolution priority:
            1. src_video.shape[0] / src_latents.shape[0] if provided
            2. else inferred from prompt / ref shapes.

        CFG (classifier-free guidance) - TODO scaffolding:
            Currently implements text-only CFG. cfg_scale > 1.0 triggers two
            forward passes: cond (full prompt) and uncond (negative prompt or "").
            src + ref + mask are ALWAYS provided to both branches (the task
            inherently requires them; the user always uploads ref + masked video).
            TODO: reference_guidance_scale / image-CFG variants. 

        TODO: KV-cache for src/ref branches (cfg.source.kv_cache / cfg.reference.kv_cache) 
        """
        # ---- 0. Resolve scalar defaults ----
        height = height or self.cfg.target.height
        width = width or self.cfg.target.width
        num_frames = num_frames or self.cfg.target.num_frames
        num_inference_steps = num_inference_steps or self.cfg.inference.num_inference_steps

        validate_video_size(
            height=height, width=width, num_frames=num_frames,
            hw_multiple=self.cfg.target.hw_multiple,
            temporal_stride=self.cfg.wan.vae_temporal_stride,
        )

        # ---- 1. src branch + mask ----
        if src_latents is None and src_video is None:
            raise ValueError("Either src_video or src_latents must be provided.")
        if src_mask is None:
            raise ValueError(
                "src_mask is required for FACET inference (mask-aware attention bias)."
            )

        # Latent grid the model will operate on.
        f_lat = latent_frames_from_num_frames(
            num_frames, temporal_stride=self.cfg.wan.vae_temporal_stride,
        )
        h_lat = height // self.cfg.wan.vae_spatial_stride
        w_lat = width // self.cfg.wan.vae_spatial_stride
        z_dim = getattr(self.dit, "in_dim", 48)

        if src_latents is not None:
            if src_video is not None:
                logger.info(
                    "[FACET] Both src_video and src_latents given; using src_latents."
                )
            if src_latents.ndim == 4:
                src_latents = src_latents.unsqueeze(0)
            if src_latents.ndim != 5:
                raise ValueError(
                    f"src_latents should be [B, {z_dim}, F_lat, h, w] or [{z_dim}, F_lat, h, w], "
                    f"got shape {tuple(src_latents.shape)}"
                )
            if tuple(src_latents.shape[1:]) != (z_dim, f_lat, h_lat, w_lat):
                raise ValueError(
                    f"src_latents shape mismatch: got {tuple(src_latents.shape)}, "
                    f"expected (B, {z_dim}, {f_lat}, {h_lat}, {w_lat}) "
                    f"for target ({height}x{width}x{num_frames})."
                )
            src_latents = src_latents.to(device=self.device, dtype=self.dtype)
        else:
            if src_video.ndim == 4:
                src_video = src_video.unsqueeze(0)
            if src_video.ndim != 5:
                raise ValueError(
                    f"src_video should be [B, 3, F, H, W] or [3, F, H, W], "
                    f"got shape {tuple(src_video.shape)}"
                )
            if tuple(src_video.shape[1:]) != (3, num_frames, height, width):
                raise ValueError(
                    f"src_video shape mismatch: got {tuple(src_video.shape)}, "
                    f"expected (B, 3, {num_frames}, {height}, {width})."
                )
            src_latents = self.encode_src_video(src_video)

        # src_mask: must be at PIXEL resolution (same H/W/F as src_video).
        if src_mask.ndim == 4:
            src_mask = src_mask.unsqueeze(0)
        if src_mask.ndim != 5:
            raise ValueError(
                f"src_mask should be [B, 1, F, H, W] or [1, F, H, W], "
                f"got shape {tuple(src_mask.shape)}"
            )
        if src_mask.shape[1] != 1:
            raise ValueError(
                f"src_mask must have exactly 1 channel; got {src_mask.shape[1]}."
            )
        if tuple(src_mask.shape[2:]) != (num_frames, height, width):
            raise ValueError(
                f"src_mask spatial/temporal shape mismatch: got "
                f"{tuple(src_mask.shape[2:])}, expected ({num_frames}, {height}, {width}). "
                "src_mask must live in PIXEL space; compute_mask_coverage pools it down."
            )

        batch_size = src_latents.shape[0]

        def _broadcast_list(lst, name):
            if len(lst) == batch_size:
                return lst
            if len(lst) == 1:
                return lst * batch_size
            raise ValueError(
                f"{name} batch size {len(lst)} cannot be broadcast to {batch_size}."
            )

        def _broadcast_tensor(t, name):
            # NOTE: check / broadcast
            if t.shape[0] == batch_size:
                return t
            if t.shape[0] == 1:
                rep = [batch_size] + [1] * (t.ndim - 1)
                return t.repeat(*rep)
            raise ValueError(
                f"{name} batch size {t.shape[0]} cannot be broadcast to {batch_size}."
            )

        src_mask = _broadcast_tensor(src_mask, "src_mask")

        # ---- 2. Prompt embeddings (cond + optional uncond for text CFG) ----
        cond_context = self.encode_prompt(prompt=prompt, prompt_embeds=prompt_embeds)
        cond_context = _broadcast_list(cond_context, "prompt")

        do_cfg = cfg_scale > 1.0
        if do_cfg:
            if negative_prompt is None and negative_prompt_embeds is None:
                negative_prompt = ""
            uncond_context = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
            )
            uncond_context = _broadcast_list(uncond_context, "negative_prompt")
        else:
            uncond_context = None

        # ---- 3. Reference latents ----
        if ref_latents is None and reference_images is None:
            raise ValueError(
                "Either reference_images or ref_latents must be provided."
            )

        ref_img_size = self.cfg.reference.image_size
        ref_lat_hw = ref_img_size // self.cfg.wan.vae_spatial_stride  # e.g. 30 for 480

        if ref_latents is not None:
            if reference_images is not None:
                logger.info(
                    "[FACET] Both reference_images and ref_latents given; "
                    "using ref_latents."
                )
            if ref_latents.ndim == 4:
                ref_latents = ref_latents.unsqueeze(0)
            if ref_latents.ndim != 5:
                raise ValueError(
                    f"ref_latents should be [B, {z_dim}, 1, h_ref, w_ref] or "
                    f"[{z_dim}, 1, h_ref, w_ref], got shape {tuple(ref_latents.shape)}"
                )
            if tuple(ref_latents.shape[1:3]) != (z_dim, 1):
                raise ValueError(
                    f"ref_latents shape mismatch: got {tuple(ref_latents.shape)}, "
                    f"expected (B, {z_dim}, 1, h_ref, w_ref). "
                    "ref_latents must be a single-frame latent."
                )
            ref_latents = ref_latents.to(device=self.device, dtype=self.dtype)
        else:
            ref_latents = self.encode_reference_image(reference_images)
        ref_latents = _broadcast_tensor(ref_latents, "ref_latents")

        # ---- 4. Initial noisy latents ----
        if latents is None:
            cur_latents = self.prepare_latents(
                batch_size=batch_size,
                height=height, width=width, num_frames=num_frames,
                generator=generator,
            )
        else:
            cur_latents = latents.unsqueeze(0) if latents.ndim == 4 else latents
            if cur_latents.ndim != 5:
                raise ValueError(
                    f"latents should be [B, {z_dim}, F_lat, h, w] or "
                    f"[{z_dim}, F_lat, h, w], got shape {tuple(cur_latents.shape)}"
                )
            if tuple(cur_latents.shape[1:]) != (z_dim, f_lat, h_lat, w_lat):
                raise ValueError(
                    f"latents shape mismatch: got {tuple(cur_latents.shape)}, "
                    f"expected (B, {z_dim}, {f_lat}, {h_lat}, {w_lat})."
                )
            cur_latents = _broadcast_tensor(cur_latents, "latents")
            cur_latents = cur_latents.to(device=self.device, dtype=self.dtype)

        # ---- 5. Scheduler timesteps ----
        timesteps = self._prepare_inference_timesteps(
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            denoising_strength=denoising_strength,
        )

        # ---- 6. Denoising loop ----
        for t in timesteps:
            # fp32 timestep avoids bf16 quantization (e.g. 999 -> 998) inside
            # sinusoidal_embedding_1d / time_embedding.
            t_batch = torch.full(
                (batch_size,), float(t),
                device=self.device, dtype=torch.float32,
            )

            pred_cond = self.forward(
                noisy_latents=cur_latents,
                timesteps=t_batch,
                prompt_embeds=cond_context,
                ref_latents=ref_latents,
                src_latents=src_latents,
                src_mask=src_mask,
                return_dict=True,
            )["pred"]

            if do_cfg:
                pred_uncond = self.forward(
                    noisy_latents=cur_latents,
                    timesteps=t_batch,
                    prompt_embeds=uncond_context,
                    ref_latents=ref_latents,
                    src_latents=src_latents,
                    src_mask=src_mask,
                    return_dict=True,
                )["pred"]
                pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
            else:
                pred = pred_cond

            cur_latents = self._scheduler_step(
                pred=pred, timestep=t, latents=cur_latents,
            )

        # ---- 7. Decode final latents ----
        # Returned as [B, 3, F, H, W] in [-1, 1]; FACETPipeline.__call__ handles
        # the final mapping back to uint8 / PIL frames when output_type=="frames".
        return self.decode_latents(cur_latents)


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
        return [u[: int(l)] for u, l in zip(context, seq_lens)] # strip padding using ":int(l)"

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
          Tensor [48, 1, h, w]        -> already latent (pre-cached), batch 1
          Tensor [B, 48, 1, h, w]     -> already latent (pre-cached), batch B

        Returns: stacked tensor [B, 48, 1, H/16, W/16].

        Resize / normalize are skipped when input is already a Tensor; the data
        pipeline (data/transform.py:RefTfm) is assumed to have done it. For raw
        PIL, it is delegated to DiffSynth's pipe.preprocess_video.

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

        # Pre-cached latent [B, 48, 1, h, w] or [48, 1, h, w].
        if x.ndim == 4 and x.shape[0] == 48 and x.shape[1] == 1:
            x = x.unsqueeze(0)
        if x.ndim == 5 and x.shape[1] == 48 and x.shape[2] == 1:
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

        # NOTE: VAE runs in fp32.
        x = x.to(device=self.device, dtype=torch.float32)
        ref_latents = self.vae.encode(x, device=self.device).to(
            dtype=self.dtype, device=self.device
        )  # [B, 48, 1, H/16, W/16]

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

        Returns: [B, 48, 1, H/16, W/16].
        """
        resized = [im.convert("RGB").resize((ref_size, ref_size)) for im in pil_list]
        # Stack into one [B, 3, 1, H, W] batch and VAE-encode in one shot.
        clips = [self.pipe.preprocess_video([im]) for im in resized] # each [1, 3, 1, H, W]
        v = torch.cat(clips, dim=0).to(device=self.device, dtype=torch.float32)
        z = self.vae.encode(v, device=self.device).to(
            dtype=self.dtype, device=self.device
        )  # [B, 48, 1, H/8, W/8]
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

        Accepted:
          Tensor [3, F, H, W]            -> batch 1
          Tensor [B, 3, F, H, W]         -> batch B
          Tensor [48, F_lat, h, w]       -> already-latent, batch 1
          Tensor [B, 48, F_lat, h, w]    -> already-latent, batch B
          List[Tensor]                   -> stacked along batch dim

        Returns: stacked tensor [B, 48, F_lat, H/16, W/16].

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
        z_dim = getattr(self.dit, "in_dim", 48)

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
        )  # [B, 48, F_lat, H/16, W/16]

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
        Pool a pixel-space src_mask into a token-grid-aligned coverage map.

        Args:
            src_mask: [B, 1, F, H, W] in [0, 1]. ([B, F, H, W] also accepted.)
            f_lat, h_tok, w_tok: token grid. Defaults derived from cfg:
                f_lat = (F - 1) // vae_temporal_stride + 1
                h_tok = H // token_spatial_stride
                w_tok = W // token_spatial_stride

        Returns:
            coverage: [B, L_base] in [0, 1] where L_base = f_lat * h_tok * w_tok.

        Why fp32 inside:
            F.adaptive_avg_pool3d is dtype-aware but bf16 has only ~8 mantissa
            bits. With 8x8 spatial pooling on a {0,1} mask, soft coverage values
            in the range ~[0, 0.01] get quantized to large relative error
            (smallest bf16 step ~0.0078 vs fp32 ~6e-8). We do the pool in fp32
            and only cast back to self.dtype at the very end.

        Flatten order matches dit.patch_embedding's output flatten
        (`flatten(2).transpose(1, 2)`): F -> H -> W with W as the fastest
        varying axis. Any mismatch silently shifts the bias relative to its
        K_src / K_ref token, breaking the editing-region prior.
        """
        if src_mask.ndim == 4:
            src_mask = src_mask.unsqueeze(1)
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
        elif self.cfg.source.mask_pool == "avg":
            pooled = F.adaptive_avg_pool3d(
                src_mask, output_size=(f_lat, h_tok, w_tok),
            )
        else:
            raise ValueError(
                f"Unknown source.mask_pool={self.cfg.source.mask_pool!r}. "
                "Expected 'avg' or 'nearest'."
            )

        coverage = pooled.flatten(2).squeeze(1).to(self.dtype)
        return coverage

    @staticmethod
    def _dilate_mask(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """Pixel-space dilation via 3D max-pool with stride=1 and same-padding."""
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

        Input:  [B, 48, F_lat, H/16, W/16] (or list form)
        Output: [B, 3, F, H, W] in [-1, 1]

        Format conversion (uint8 / PIL frames / mp4) is handled by FACETPipeline.
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

        Returns: [B, z_dim=48, F_lat, H/16, W/16].
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

        c = getattr(self.dit, "in_dim", 48)

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
        h_offset: int = 0,
        w_offset: int = 0,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Build a 3D RoPE freqs tensor for one branch in WAN's shared rotary space.

        DiffSynth WanModel stores precomputed 1D complex freqs as a tuple
        (f_freqs, h_freqs, w_freqs) on self.dit.freqs. With head_dim=128, the
        canonical WAN split allocates 22 complex pairs to f, 21 to h, 21 to w
        (sums to head_dim/2 = 64).

        Args:
            f, h, w:      grid sizes in patchified-token space.
            f_offset:     starting index along the f-axis. 0 for base/src.
            h_offset:     starting index along the h-axis. 0 for all branches.
            w_offset:     starting index along the w-axis. 0 for base/src;
                          cfg.reference.w_offset for ref (default 26 for 480x832).
            device:       target device; defaults to dit.freqs[0].device.

        Returns:
            freqs of shape [f*h*w, 1, head_dim/2], complex tensor ready for rope_apply.
        """
        if self.dit is None:
            raise RuntimeError("DiT not loaded.")
        f_freqs, h_freqs, w_freqs = self.dit.freqs
        if device is None:
            device = f_freqs.device

        # --- f-axis ---
        max_f = f_freqs.shape[0]
        if f_offset + f > max_f:
            raise ValueError(
                f"f_offset({f_offset}) + f({f}) = {f_offset + f} exceeds "
                f"precomputed freqs length {max_f}."
            )
        f_part = f_freqs[f_offset : f_offset + f].view(f, 1, 1, -1).expand(f, h, w, -1)

        # --- h-axis ---
        max_h = h_freqs.shape[0]
        if h_offset + h > max_h:
            raise ValueError(
                f"h_offset({h_offset}) + h({h}) = {h_offset + h} exceeds "
                f"precomputed freqs length {max_h}."
            )
        h_part = h_freqs[h_offset : h_offset + h].view(1, h, 1, -1).expand(f, h, w, -1)

        # --- w-axis ---
        max_w = w_freqs.shape[0]
        if w_offset + w > max_w:
            raise ValueError(
                f"w_offset({w_offset}) + w({w}) = {w_offset + w} exceeds "
                f"precomputed freqs length {max_w}."
            )
        w_part = w_freqs[w_offset : w_offset + w].view(1, 1, w, -1).expand(f, h, w, -1)

        freqs = torch.cat([f_part, h_part, w_part], dim=-1).reshape(f * h * w, 1, -1).to(device)
        return freqs

    def compute_time_features(
        self,
        timestep: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute (t, t_mod) for a batch of diffusion timesteps.

        Returns:
            t:     [B, dim]     used by dit.head(x, t).
            t_mod: [B, 6, dim]  used by each block's modulation.

        Called twice per forward:
            (1) timestep = real diffusion t  ->  for base branch
            (2) timestep = 0                 ->  for src AND ref branches
                                                (cfg.source.timestep ==
                                                cfg.reference.timestep == 0)
        """
        from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d

        if self.dit is None:
            raise RuntimeError("DiT not loaded.")

        dit = self.dit
        timestep = timestep.to(device=self.device)

        # Cast to the dtype the rest of the dit expects.
        t_emb = sinusoidal_embedding_1d(dit.freq_dim, timestep).to(self.dtype)  # freq_dim = 256
        t = dit.time_embedding(t_emb)
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))  # [B, 6, dim=3072]
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
            shift > 1 squeezes more steps into the high-noise region. Wan default = 5.

        denoising_strength:
            Scales the starting sigma. 1.0 = start from pure noise. FACET uses 1.0.

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
        DiffSynth's FlowMatchScheduler.step computes
            prev = sample + model_output * (sigma_next - sigma)
        which natively supports stacked-batch tensors.
        """
        return self.scheduler.step(pred, timestep, latents)


# ============================================================
# C. Public pipeline (for users)
# ============================================================


class FACETPipeline:
    """
    User-facing inference wrapper. Adds:
      * deterministic seed plumbing
      * output_type post-processing (raw tensor / PIL frames)

    Required-input validation lives in model.generate; pipeline only forwards
    arguments.
    """

    def __init__(self, model: FACETWanModel):
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        reference_images: Optional[Union[Image.Image, List[Image.Image], torch.Tensor]] = None,
        ref_latents: Optional[torch.Tensor] = None,
        src_video: Optional[torch.Tensor] = None,
        src_latents: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
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
        if seed is not None:
            generator = torch.Generator(device=self.model.device)
            generator.manual_seed(int(seed))
        else:
            generator = None

        video = self.model.generate(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            reference_images=reference_images,
            ref_latents=ref_latents,
            src_video=src_video,
            src_latents=src_latents,
            src_mask=src_mask,
            negative_prompt=negative_prompt,
            negative_prompt_embeds=negative_prompt_embeds,
            height=height, width=width, num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            cfg_scale=cfg_scale,
            latents=latents,
            generator=generator,
            sigma_shift=sigma_shift,
            denoising_strength=denoising_strength,
        )  # [B, 3, F, H, W] in [-1, 1]

        output_type = output_type or self.model.cfg.inference.output_type
        if output_type == "video":
            # Return the raw decoded tensor in [-1, 1]. This is the canonical
            # research output (downstream metrics like LPIPS / SSIM / VBench
            # operate on this range or trivially shift to [0, 1]). Conversion
            # to uint8 / [0, 255] is the caller's responsibility (see
            # _to_pil_frames for the standard mapping).
            return video
        if output_type == "frames":
            return self._to_pil_frames(video)
        raise ValueError(
            f"Unknown output_type={output_type!r}. Expected 'video' | 'frames'."
        )

    @staticmethod
    def _to_pil_frames(video: torch.Tensor) -> List[List[Image.Image]]:
        """
        Convert [B, 3, F, H, W] in [-1, 1] -> nested list (outer batch, inner per-frame PIL).
        """
        v = ((video.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8) # [-1, 1] -> [0, 255]
        # [B, 3, F, H, W] -> [B, F, H, W, 3]
        v = v.permute(0, 2, 3, 4, 1).cpu().numpy()
        out: List[List[Image.Image]] = []
        for clip in v:
            out.append([Image.fromarray(frame) for frame in clip])
        return out


# ============================================================
# D. Utilities
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
