Class FACET 结构 
 ├── pipe / components
 │    ├── dit              # Wan diffusion transformer, frozen base + LoRA
 │    ├── vae              # Wan2.2_VAE.pth, frozen
 │    ├── text_encoder     # UMT5/T5, frozen
 │    └── scheduler        # 
 │
 ├── lora manager
 ├── reference encoder helpers  # for encoding reference branch 可以取名叫 ref_branch
 ├── forward()             # training one-step denoising
 └── generate() / pipeline.__call__()  # inference denoising loop

class FACET(nn.Module):
    def __init__(...):
        self.dit = load_wan_dit(...)
        self.vae = load_wan_vae(...)
        self.text_encoder = load_t5(...)
        self.scheduler = load_scheduler(...)
        
        # 初始化结束后 设置WAN基座上lora构型的初始化 可能调用lora manager中的函数?
        # 只确保pipeline构型存在 不负责初始化参数或者加载训练好的参数
        # 可能涉及某些内容的替换
        
# 官方 WanModel 的 forward() 签名比较固定。官方代码里的 WanModel.forward() 期望 x 是 List[Tensor]，context 也是 List[Tensor]，并在内部做 patch embedding、RoPE、block forward、unpatchify
# 用 wrapper 可以保持 base Wan checkpoint 原样，只替换或扩展 forward path
# 保存时只保存 LoRA 和 FACET config

# lora注入的位置:
--lora_target_weights: "q, k, v, o, ffn.0, ffn.2"
注意力机制中用于生成q k v o的线性nn层部分(包括ffn.0 ffn.2)本身-base部分冻结
lora权重同样是用于生层q k v o的线性nn层部分 作为base部分的残差
最开始对lora up使用零初始化 lora down使用kaiming uniform 因为激活函数为GeLU 不影响WAN模型本身的生成能力
y = base(x) + scale * B(A(dropout(x))) scale:残差放缩系数

def lora manager()
# 可能涉及lora权重的初始化
def inject_lora()
# 查找模型中的nn模块或者"q, k, v, o, ffn.0, ffn.2"部分 替换为实例化的LoRALinear

def save_lora_weights() 用于保存lora权重 & 模型超参数config 非内部函数

def load_lora_weights() 用于加载lora权重 & 模型超参数config 非内部函数

def reference encoder helpers 
def ref_branch():
流程大致如下: (参照于OmniControl架构)
ref_image [1, 480, 480]c=3
  ↓ same VAE
ref_latent [1, 30, 30]c=4
  ↓ same DiT patch embedding
ref_tokens [1, 15, 15]c=4
  ↓ attention with target noisy video tokens
target output tokens

# 用于训练和推理的模块函数:
def from_config()
def encode_ref()
def encode_prompt()  # 属于diffusion必需
def prepare_latents() 
def decode_latents()

def forward(
    self,
    noisy_latents: list[torch.Tensor], # B, [C, F, H, W]
    timesteps: torch.Tensor, # [B,]
    prompt_embeds: list[torch.Tensor], #B, [L, D_text]
    ref_tokens: list[torch.Tensor] | None = None, #B, [C, 1, H_ref, W_ref]
    # input_image_tokens: list[torch.Tensor] | None = None,  不需要
    # attention_mask: torch.Tensor | None = None, 暂时不需要
    category_ids: torch.Tensor | None = None,
    return_dict: bool = True, # List[Tensor], each [C, F, H, W]
) -> dict | list[torch.Tensor]: 
# 训练时使用的前向传递 该函数自用所以可保证输入合规性 不需要做input check 也不需要高兼容性 流程为: 
输入target video latent(已经提前由WAN2.2VAE编码好的) List[Tensor], each [C, Fv, Hv, Wv]
进行patchify(& RoPE) 从而得到 latent token
输入reference image latanet(同样已经由WAN2.2VAE编码好) List[Tensor], each [C, 1, Hr, Wr]
进行patchify(positional embedding的具体策略待定) 得到ref token
输入caption embedding(已经提前由T5 text encoder编码好) shape: List[Tensor], each [L, text_dim]
进行text embedding 得到 context token???
return pred_velocity / pred_noise List[Tensor], each [C, F, H, W] 

@torch.no_grad()
def generate(): 实际上的FACETpipeline call核心
# 需要做input check 而且需要有兼容性 例如wan期待传入的list 但是传入tensor stack 
# 所以需要tensor转list[tensor] 还另外包括shape mismatch的尝试处理 能解决就解决(例如squeeze/unsqueeze)
# 不能解决就只能报错了
# 而且call函数需要兼容直接给原始视频或者直接给已经编码好的target video 
# 兼容caption 或者已经编码好的T5 caption embedding
generate流程:
1)输入检查
2)prompt 编码
3)reference image resize + VAE encode
4)初始化 Gaussian latent
5)scheduler timesteps
6)CFG
7)denoising loop
8)VAE decode
9)输出 PIL/video tensor


def call()
return self.model.generate(...)

def check_inputs()
# 被call函数所使用

# 工具函数 下面后期可以放到 /Facet/utils.py中
normalize_video_tensor()
normalize_image_tensor()


# 模型中冻结与可训练的参数:
frozen:
  - Wan DiT base weights
  - Wan VAE
  - T5 / UMT5 text encoder
  - scheduler

trainable:
  - LoRA on dit.q / dit.k / dit.v / dit.o / dit.ffn.0 / dit.ffn.2



# model.py

from __future__ import annotations

import math
import yaml
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

from config import FACETWanConfig, FACETTargetConfig, FACETReferenceConfig, FACETLoRAConfig, FACETInferenceConfig


@dataclass
class FACETModelConfig:
    name: str = "FACET-Wan2.2-TI2V-5B"
    base_model_id: str = "Wan-AI/Wan2.2-TI2V-5B" #FIXME: 改为本地路径加载 而非huggingface下载
    dtype: str = "bf16"
    device: str = "cuda"
    # freeze_base: bool = True
    gradient_checkpointing: bool = True

    wan: FACETWanConfig = field(default_factory=FACETWanConfig)
    target: FACETTargetConfig = field(default_factory=FACETTargetConfig)
    reference: FACETReferenceConfig = field(default_factory=FACETReferenceConfig)
    lora: FACETLoRAConfig = field(default_factory=FACETLoRAConfig)
    inference: FACETInferenceConfig = field(default_factory=FACETInferenceConfig)

    @staticmethod
    def from_yaml(path: str) -> "FACETModelConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        # Keep this parser simple in v1.
        # You can replace it with OmegaConf/Hydra later.
        cfg = FACETModelConfig()

        for k, v in raw.get("model", {}).items():
            if hasattr(cfg, k) and not isinstance(getattr(cfg, k), (FACETWanConfig, FACETTargetConfig, FACETReferenceConfig, FACETLoRAConfig, FACETInferenceConfig)):
                setattr(cfg, k, v)

        if "wan" in raw.get("model", {}):
            for k, v in raw["model"]["wan"].items():
                setattr(cfg.wan, k, tuple(v) if k == "patch_size" else v)

        if "target" in raw.get("model", {}):
            for k, v in raw["model"]["target"].items():
                setattr(cfg.target, k, v)

        if "reference" in raw.get("model", {}):
            for k, v in raw["model"]["reference"].items():
                setattr(cfg.reference, k, v)

        if "lora" in raw.get("model", {}):
            lora_raw = raw["model"]["lora"]
            for k, v in lora_raw.items():
                if k == "target_modules":
                    v = tuple(v)
                if k == "branch_routing":
                    cfg.lora.target_adapter = v.get("target", "default")
                    cfg.lora.reference_adapter = v.get("reference", "default")
                    cfg.lora.text_adapter = v.get("text", None)
                elif hasattr(cfg.lora, k):
                    setattr(cfg.lora, k, v)

        if "inference" in raw.get("model", {}):
            for k, v in raw["model"]["inference"].items():
                setattr(cfg.inference, k, v)

        return cfg


def resolve_dtype(dtype: str) -> torch.dtype:
    if dtype in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype in ("fp16", "float16"):
        return torch.float16
    if dtype in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def _get_parent_module(root: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    """
    Given 'blocks.0.self_attn.q', return:
      parent = root.blocks[0].self_attn
      child_name = 'q'
    """
    parts = module_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def lora_targets(module_name: str, target_modules: Sequence[str]) -> bool:
    """
    Match by suffix.

    For Wan:
      blocks.0.self_attn.q     -> q
      blocks.0.self_attn.k     -> k
      blocks.0.self_attn.v     -> v
      blocks.0.self_attn.o     -> o
      blocks.0.ffn.0           -> ffn.0
      blocks.0.ffn.2           -> ffn.2
    """
    return any(module_name.endswith(t) for t in target_modules)


def inject_lora(
    root: nn.Module,
    target_modules: Sequence[str],
    rank: int,
    alpha: int,
    dropout: float,
) -> List[str]:
    """
    Replace matching nn.Linear modules with LoRALinear.

    Returns:
        List of replaced module names.
    """
    replaced = []

    # Important: list() to avoid modifying module tree during iteration.
    for name, module in list(root.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not lora_targets(name, target_modules):
            continue

        parent, child_name = _get_parent_module(root, name)
        setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
        replaced.append(name)

    return replaced


# ============================================================
# 3. Utility
# ============================================================

def latent_frames_from_num_frames(num_frames: int, temporal_stride: int = 4) -> int:
    assert (num_frames - 1) % temporal_stride == 0, (
        f"num_frames should be 4n+1 for Wan-style video VAE, got {num_frames}"
    )
    return (num_frames - 1) // temporal_stride + 1


def ensure_latent_list(x: Union[torch.Tensor, List[torch.Tensor]]) -> List[torch.Tensor]:
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

    raise ValueError(f"Expected latent tensor with ndim 4 or 5, got shape {tuple(x.shape)}")


def ensure_context_list(x: Union[torch.Tensor, List[torch.Tensor]]) -> List[torch.Tensor]:
    """
    Accept:
      [B, L, D] -> List of B tensors [L, D]
      [L, D]    -> List of 1 tensor
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

    raise ValueError(f"Expected context tensor with ndim 2 or 3, got shape {tuple(x.shape)}")


def validate_video_size(height: int, width: int, num_frames: int, hw_multiple: int = 32) -> None:
    if height % hw_multiple != 0 or width % hw_multiple != 0:
        raise ValueError(
            f"height and width should be divisible by {hw_multiple}. "
            f"Got height={height}, width={width}."
        )

    if (num_frames - 1) % 4 != 0:
        raise ValueError(
            f"num_frames should be 4n+1 for Wan-style video VAE. Got {num_frames}."
        )


# ============================================================
# 4. Main FACET model
# ============================================================

class FACETWanModel(nn.Module):
    """
    FACET model wrapper around Wan2.2-TI2V-5B.

    Training:
        forward() does one denoising prediction step.

    Inference:
        generate() does full denoising loop.

    This class intentionally uses composition instead of subclassing WanModel.
    """

    def __init__(self, cfg: FACETModelConfig):
        super().__init__()
        self.cfg = cfg
        self.dtype = resolve_dtype(cfg.dtype)
        self.device_name = cfg.device

        # These will be assigned by _load_base_components().
        self.pipe = None
        self.dit = None
        self.vae = None
        self.text_encoder = None
        self.tokenizer = None
        self.scheduler = None

        self._load_base_components()

        # if cfg.freeze_base:
        self._freeze_base()

        # if cfg.lora.enabled:
        self._init_lora()

        self.to(cfg.device)

    @classmethod
    def from_config(cls, path: str) -> "FACETWanModel":
        cfg = FACETModelConfig.from_yaml(path)
        return cls(cfg)

    def _load_base_components(self) -> None:
        """
        Load Wan2.2-TI2V-5B components.

        You have two implementation choices:

        Option A:
            Use DiffSynth-Studio WanVideoPipeline.

        Option B:
            Use official Wan2.2 repo classes directly.

        In FACET v1 I recommend using DiffSynth loading for convenience,
        but keep this function isolated so you can swap backend later.
        """
        
        #TODO: 使用optionA
        #FIXME: 使用本地路径加载 而非huggingface下载
        # Pseudocode:
        #
        # from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
        #
        # self.pipe = WanVideoPipeline.from_pretrained(
        #     torch_dtype=self.dtype,
        #     device=self.cfg.device,
        #     model_configs=[
        #         ModelConfig(
        #             model_id=self.cfg.base_model_id,
        #             origin_file_pattern="diffusion_pytorch_model*.safetensors",
        #         ),
        #         ModelConfig(
        #             model_id=self.cfg.base_model_id,
        #             origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
        #         ),
        #         ModelConfig(
        #             model_id=self.cfg.base_model_id,
        #             origin_file_pattern="Wan2.2_VAE.pth",
        #         ),
        #     ],
        # )
        #
        # self.dit = self.pipe.dit
        # self.vae = self.pipe.vae
        # self.text_encoder = self.pipe.text_encoder
        # self.tokenizer = self.pipe.tokenizer
        # self.scheduler = self.pipe.scheduler

        raise NotImplementedError("Implement Wan2.2 component loading here.")

    def _freeze_base(self) -> None:
        """
        Freeze everything first.
        LoRA params will be re-enabled after injection.
        """
        for p in self.parameters():
            p.requires_grad_(False)

        if self.dit is not None:
            self.dit.eval()
        if self.vae is not None:
            self.vae.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()

    def _init_lora(self) -> None:
        assert self.cfg.lora.base_model == "dit", (
            "FACET v1 only supports LoRA on dit. "
            "This corresponds to DiffSynth --lora_base_model dit."
        )

        replaced = inject_lora(
            root=self.dit,
            target_modules=self.cfg.lora.target_modules,
            rank=self.cfg.lora.rank,
            alpha=self.cfg.lora.alpha,
            dropout=self.cfg.lora.dropout,
        )

        if len(replaced) == 0:
            raise RuntimeError(
                "No LoRA modules were injected. "
                f"Check target_modules={self.cfg.lora.target_modules} "
                "against your Wan module names."
            )

        # mark_only_lora_trainable(self)
        for name, p in self.named_parameters():
            p.requires_grad_("lora_" in name)

        print(f"[FACET] Injected LoRA into {len(replaced)} modules:")
        for name in replaced[:20]:
            print("  -", name)
        if len(replaced) > 20:
            print(f"  ... and {len(replaced) - 20} more")

    # --------------------------------------------------------
    # Encoding helpers
    # --------------------------------------------------------

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        max_length: int = 512,
    ) -> List[torch.Tensor]:
        """
        Return Wan-compatible context list.

        If prompt_embeds is provided, use it directly.
        Otherwise encode prompt using Wan/T5 text encoder.
        """
        if prompt_embeds is not None:
            return ensure_context_list(prompt_embeds)

        if prompt is None:
            raise ValueError("Either prompt or prompt_embeds must be provided.")

        # Pseudocode:
        #
        # embeds = self.pipe.encode_prompt(prompt, max_length=max_length)
        # return ensure_context_list(embeds)

        raise NotImplementedError("Implement prompt encoding according to your Wan backend.")

    @torch.no_grad()
    def encode_reference_image(
        self,
        reference_images: Union[Image.Image, List[Image.Image], torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Convert reference image(s) to Wan VAE latents.

        Expected output:
            List length B
            each tensor shape [C, 1, H_ref_lat, W_ref_lat]

        For FACET:
            reference image is resized to 480x480 by default.
        """
        # Pseudocode:
        #
        # 1. normalize input to list of PIL or tensor batch
        # 2. center crop + resize to cfg.reference.image_size
        # 3. convert to tensor [B, 3, 1, H, W] or [B, 1, 3, H, W]
        # 4. VAE encode
        # 5. return List[Tensor], each [C, 1, H_lat, W_lat]
        #
        # with torch.no_grad():
        #     ref_latents = self.vae.encode(ref_video_like)
        #
        # if cfg.reference.detach_latent:
        #     ref_latents = ref_latents.detach()

        raise NotImplementedError("Implement reference image VAE encoding.")

    @torch.no_grad()
    def decode_latents(
        self,
        latents: Union[List[torch.Tensor], torch.Tensor],
        output_type: str = "pil",
    ):
        """
        Decode target video latents to RGB video.
        """
        latents = ensure_latent_list(latents)

        # Pseudocode:
        #
        # video = self.vae.decode(latents)
        # postprocess to PIL frames or torch tensor
        #
        raise NotImplementedError("Implement Wan VAE decoding.")

    def prepare_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        num_frames: int,
        generator: Optional[torch.Generator] = None,
    ) -> List[torch.Tensor]:
        """
        Initialize Gaussian latent for inference.
        """
        validate_video_size(
            height=height,
            width=width,
            num_frames=num_frames,
            hw_multiple=self.cfg.target.hw_multiple,
        )

        f_lat = latent_frames_from_num_frames(
            num_frames,
            temporal_stride=self.cfg.wan.vae_temporal_stride,
        )
        h_lat = height // self.cfg.wan.vae_spatial_stride
        w_lat = width // self.cfg.wan.vae_spatial_stride

        # Wan latent channels are typically 16 for these models.
        # Better: read from self.dit.in_dim or self.vae config.
        c = getattr(self.dit, "in_dim", 16)

        latents = torch.randn(
            batch_size,
            c,
            f_lat,
            h_lat,
            w_lat,
            generator=generator,
            device=self.cfg.device,
            dtype=self.dtype,
        )
        return ensure_latent_list(latents)

    # --------------------------------------------------------
    # Training forward
    # --------------------------------------------------------

    def forward(
        self,
        noisy_latents: Union[List[torch.Tensor], torch.Tensor],
        timesteps: torch.Tensor,
        prompt_embeds: Union[List[torch.Tensor], torch.Tensor],
        reference_latents: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        input_image_latents: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        category_ids: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[Dict[str, Any], List[torch.Tensor]]:
        """
        One-step denoising prediction for training.

        Args:
            noisy_latents:
                List[Tensor] or Tensor.
                Each target latent shape [C, F, H, W].

            timesteps:
                Tensor [B].

            prompt_embeds:
                List[Tensor] or Tensor.
                Each text embedding shape [L, D].

            reference_latents:
                Optional reference latent list.
                Each shape [C, 1, H_ref, W_ref].

            input_image_latents:
                Optional TI2V/I2V condition latents if you want to preserve
                original Wan TI2V behavior.

            category_ids:
                Optional clothing part category ids.

        Returns:
            pred:
                List[Tensor], same latent shape as noisy_latents.
        """
        x = ensure_latent_list(noisy_latents)
        context = ensure_context_list(prompt_embeds)
        y = ensure_latent_list(input_image_latents) if input_image_latents is not None else None

        # if reference_latents is not None:
        # FIXME: 必须要求有 reference_latents
        ref = ensure_latent_list(reference_latents)
        pred = self._forward_with_reference(
            x=x,
            t=timesteps,
            context=context,
            ref=ref,
            y=y,
            category_ids=category_ids,
        )

        # FIXME: forward函数一次性写出来 不需要反复根据选择option进行分支套壳 
        # 选择_forward_with_reference_branch_attention
        # else:
        #     pred = self._forward_base_dit(
        #         x=x,
        #         t=timesteps,
        #         context=context,
        #         y=y,
        #     )

        if return_dict:
            return {"pred": pred}
        return pred

    # def _forward_base_dit(
    #     self,
    #     x: List[torch.Tensor],
    #     t: torch.Tensor,
    #     context: List[torch.Tensor],
    #     y: Optional[List[torch.Tensor]] = None,
    # ) -> List[torch.Tensor]:
    #     """
    #     Original Wan forward path.
    #     """
    #     seq_len = self._infer_seq_len_for_wan(x, y=y)
    #     return self.dit(x=x, t=t, context=context, seq_len=seq_len, y=y)

    def _forward_with_reference(
        self,
        x: List[torch.Tensor],
        t: torch.Tensor,
        context: List[torch.Tensor],
        ref: List[torch.Tensor],
        y: Optional[List[torch.Tensor]] = None,
        category_ids: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        OminiControl-style reference token branch.

        Important:
            - target branch timestep = t
            - reference branch timestep = 0
            - reference branch participates in attention
            - only target branch output is decoded / returned
        """

        if self.cfg.reference.injection_mode == "concat_tokens":
            return self._forward_with_reference_concat_tokens(
                x=x,
                t=t,
                context=context,
                ref=ref,
                y=y,
            )

        #TODO: 使用branch_attention
        if self.cfg.reference.injection_mode == "branch_attention":
            return self._forward_with_reference_branch_attention(
                x=x,
                t=t,
                context=context,
                ref=ref,
                y=y,
                category_ids=category_ids,
            )

        raise ValueError(f"Unknown reference injection mode: {self.cfg.reference.injection_mode}")

    def _forward_with_reference_concat_tokens(
        self,
        x: List[torch.Tensor],
        t: torch.Tensor,
        context: List[torch.Tensor],
        ref: List[torch.Tensor],
        y: Optional[List[torch.Tensor]] = None,
    ) -> List[torch.Tensor]:
        """
        Simpler baseline.

        Not recommended as final CVPR method because reference tokens do not
        naturally fit the target 3D RoPE grid. But useful for debugging.
        """
        raise NotImplementedError(
            "Implement only as a baseline. "
            "Recommended method is branch_attention."
        )

    def _forward_with_reference_branch_attention(
        self,
        x: List[torch.Tensor],
        t: torch.Tensor,
        context: List[torch.Tensor],
        ref: List[torch.Tensor],
        y: Optional[List[torch.Tensor]] = None,
        category_ids: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Recommended FACET v1 path.

        This function should be implemented by copying / imitating the official WanModel.forward()
        logic and modifying the transformer block attention to support multiple
        latent branches.

        High-level steps:
            1. Patchify target latents.
            2. Patchify reference latents with the same Wan patch_embedding.
            3. Build time embedding for target branch using t.
            4. Build time embedding for reference branch using zeros.
            5. Build text embeddings.
            6. Run Wan blocks with branch-aware self-attention.
            7. Apply Wan head only on target branch.
            8. Unpatchify only target branch.
        """
        # TODO: OmniControl-style reference token branch

        # Pseudocode:
        #
        # target_state = self._patchify_branch(x, y=y)
        # ref_state = self._patchify_branch(ref, y=None)
        #
        # context_state = self._embed_text_context(context)
        #
        # target_t = t
        # ref_t = torch.zeros_like(t) + self.cfg.reference.timestep
        #
        # target_e, target_e0 = self._build_time_embedding(target_t, target_state.seq_len)
        # ref_e, ref_e0 = self._build_time_embedding(ref_t, ref_state.seq_len)
        #
        # hidden_target = target_state.hidden
        # hidden_ref = ref_state.hidden
        #
        # for block in self.dit.blocks:
        #     hidden_target, hidden_ref = self._facet_block_forward(
        #         block=block,
        #         target_hidden=hidden_target,
        #         ref_hidden=hidden_ref,
        #         target_e0=target_e0,
        #         ref_e0=ref_e0,
        #         target_seq_lens=target_state.seq_lens,
        #         ref_seq_lens=ref_state.seq_lens,
        #         target_grid_sizes=target_state.grid_sizes,
        #         ref_grid_sizes=ref_state.grid_sizes,
        #         context=context_state.hidden,
        #         context_lens=context_state.lens,
        #     )
        #
        # target_out = self._apply_wan_head(
        #     hidden_target,
        #     target_e,
        # )
        #
        # pred = self.dit.unpatchify(target_out, target_state.grid_sizes)
        # return [u.float() for u in pred]

        raise NotImplementedError("Implement branch-aware Wan forward here.")

    # def _infer_seq_len_for_wan(
    #     self,
    #     x: List[torch.Tensor],
    #     y: Optional[List[torch.Tensor]] = None,
    # ) -> int:
    #     """
    #     Infer max target token length for original Wan forward.

    #     Wan patch size is usually (1, 2, 2).
    #     Input latent shape: [C, F, H, W]
    #     Token grid after patch: [F/1, H/2, W/2].
    #     """
    #     p_t, p_h, p_w = self.cfg.wan.patch_size
    #     max_len = 0

    #     for u in x:
    #         _, f, h, w = u.shape
    #         length = (f // p_t) * (h // p_h) * (w // p_w)
    #         max_len = max(max_len, length)

    #     return max_len

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        reference_image: Optional[Union[Image.Image, List[Image.Image], torch.Tensor]] = None,
        prompt_embeds: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = "",
        negative_prompt_embeds: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        reference_guidance_scale: Optional[float] = None,
        latents: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = None,
    ):
        """
        Full inference loop.

        Public users should normally call FACETPipeline.__call__(),
        which calls this function internally.
        """
        height = height or self.cfg.target.height
        width = width or self.cfg.target.width
        num_frames = num_frames or self.cfg.target.num_frames
        num_inference_steps = num_inference_steps or self.cfg.inference.num_inference_steps
        cfg_scale = cfg_scale if cfg_scale is not None else self.cfg.inference.cfg_scale
        reference_guidance_scale = (
            reference_guidance_scale
            if reference_guidance_scale is not None
            else self.cfg.inference.reference_guidance_scale
        )
        output_type = output_type or self.cfg.inference.output_type

        validate_video_size(
            height=height,
            width=width,
            num_frames=num_frames,
            hw_multiple=self.cfg.target.hw_multiple,
        )

        # 1. Prepare prompt embeddings.
        cond_context = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
        )

        do_cfg = cfg_scale is not None and cfg_scale > 1.0
        if do_cfg:
            uncond_context = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
            )
        else:
            uncond_context = None

        batch_size = len(cond_context)

        # 2. Prepare reference latents.
        ref_latents = None
        if reference_image is not None: # and self.cfg.reference.enabled:
            ref_latents = self.encode_reference_image(reference_image)
            if len(ref_latents) != batch_size:
                if len(ref_latents) == 1:
                    ref_latents = ref_latents * batch_size
                else:
                    raise ValueError(
                        f"reference batch size {len(ref_latents)} does not match "
                        f"prompt batch size {batch_size}."
                    )

        # 3. Prepare initial noisy latents.
        if latents is None:
            cur_latents = self.prepare_latents(
                batch_size=batch_size,
                height=height,
                width=width,
                num_frames=num_frames,
                generator=generator,
            )
        else:
            cur_latents = ensure_latent_list(latents)

        # 4. Prepare scheduler timesteps.
        # Pseudocode:
        #
        # timesteps = self.scheduler.set_timesteps(num_inference_steps, device=self.cfg.device)
        #
        timesteps = self._prepare_inference_timesteps(num_inference_steps)

        # 5. Denoising loop.
        for step_idx, t in enumerate(timesteps):
            t_batch = torch.full(
                (batch_size,),
                float(t),
                device=self.cfg.device,
                dtype=torch.float32,
            )

            pred_cond = self.forward(
                noisy_latents=cur_latents,
                timesteps=t_batch,
                prompt_embeds=cond_context,
                reference_latents=ref_latents,
                return_dict=True,
            )["pred"]

            if do_cfg:
                pred_uncond = self.forward(
                    noisy_latents=cur_latents,
                    timesteps=t_batch,
                    prompt_embeds=uncond_context,
                    reference_latents=ref_latents,
                    return_dict=True,
                )["pred"]

                pred = [
                    u + cfg_scale * (c - u)
                    for c, u in zip(pred_cond, pred_uncond)
                ]
            else:
                pred = pred_cond

            # Optional reference guidance:
            # cond with reference vs cond with empty/no reference.
            # This is useful if model tends to ignore reference.
            if reference_guidance_scale is not None and reference_guidance_scale != 1.0 and ref_latents is not None:
                pred_no_ref = self.forward(
                    noisy_latents=cur_latents,
                    timesteps=t_batch,
                    prompt_embeds=cond_context,
                    reference_latents=None,
                    return_dict=True,
                )["pred"]

                pred = [
                    nr + reference_guidance_scale * (r - nr)
                    for r, nr in zip(pred, pred_no_ref)
                ]

            # 6. Scheduler step.
            cur_latents = self._scheduler_step(
                pred=pred,
                timestep=t,
                latents=cur_latents,
            )

        # 7. Decode final target latents only.
        return self.decode_latents(cur_latents, output_type=output_type)

    def _prepare_inference_timesteps(self, num_inference_steps: int):
        """
        Backend-specific scheduler setup.
        """
        raise NotImplementedError("Implement according to DiffSynth/Wan scheduler.")

    def _scheduler_step(
        self,
        pred: List[torch.Tensor],
        timestep: Any,
        latents: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Backend-specific scheduler step.
        """
        raise NotImplementedError("Implement according to DiffSynth/Wan scheduler.")

    # --------------------------------------------------------
    # Save/load
    # --------------------------------------------------------

    def save_lora(self, path: str) -> None:
        """
        Save only LoRA weights plus config.
        """
        from safetensors.torch import save_file
        state = {
        k: v.detach().cpu()
        for k, v in self.state_dict().items()
        if "lora_down" in k or "lora_up" in k
        }
        save_file(state, path)

    def load_lora(self, path: str, strict: bool = False) -> None:
        """
        Load LoRA weights. Make sure LoRA modules are injected first.
        """
        from safetensors.torch import load_file
        state = load_file(path)
        missing, unexpected = self.load_state_dict(state, strict=strict)
        if len(unexpected) > 0:
            print("[FACET] Unexpected LoRA keys:", unexpected)
        if len(missing) > 0:
            print("[FACET] Missing LoRA keys:", missing)


# ============================================================
# 5. Public pipeline
# ============================================================

class FACETPipeline:
    """
    User-facing inference API.

    Usage:
        model = FACETWanModel.from_config("model_cfg.yaml")
        pipe = FACETPipeline(model)
        video = pipe(prompt=..., reference_image=...)
    """

    def __init__(self, model: FACETWanModel):
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        reference_image: Optional[Union[Image.Image, List[Image.Image], torch.Tensor]] = None,
        prompt_embeds: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = "",
        negative_prompt_embeds: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        reference_guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        latents: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        output_type: str = "pil",
    ):
        """
        Public call function with stricter input checking.
        """
        if prompt is None and prompt_embeds is None:
            raise ValueError("Either prompt or prompt_embeds must be provided.")

        if reference_image is None: # and self.model.cfg.reference.enabled:
            raise ValueError("reference_image is required.")

        if seed is not None:
            generator = torch.Generator(device=self.model.cfg.device)
            generator.manual_seed(seed)
        else:
            generator = None

        return self.model.generate(
            prompt=prompt,
            reference_image=reference_image,
            prompt_embeds=prompt_embeds,
            negative_prompt=negative_prompt,
            negative_prompt_embeds=negative_prompt_embeds,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            cfg_scale=cfg_scale,
            reference_guidance_scale=reference_guidance_scale,
            latents=latents,
            generator=generator,
            output_type=output_type,
        )