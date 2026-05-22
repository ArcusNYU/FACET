Model.py需要包含以下内容:

import_ 部分导入config.py 以及 FACET/utils.py 中的函数 _resolve_dtype() & _get_parent_module() 涉及同时完善 utils.py

A - Model Config -> Class FACETConfig: 从同目录 facet 中导入配置 config.py 中各项配置并映射

B - LoRA Config -> LoRA targets 决定模块上的哪些具体参数添加LoRA权重 & inject LoRA 决定对于模型中的哪些模块进行LoRA注入

C - Utility ->  模型forward或generate时需要用到的工具函数 ensure_latent_list() & ensure_context_list() & validate_video_size()
                因为WAN model期望的输入是List[Tensor] 而实际传入的可能是Tensor stack 所以需要进行转换
                另外还有关于视频尺寸的验证 因为WAN model期望的视频尺寸是32的倍数

D - Wan Model -> Class FACETWanModel
Class FACETWanModel 结构 
 ├── __init__
 ├── from_config 
 ├── load_base_components
 │    ├── dit              # Wan diffusion transformer, frozen base + LoRA
 │    ├── vae              # Wan2.1_VAE.pth, frozen
 │    ├── text_encoder     # UMT5/T5, frozen
 │    └── scheduler        # FlowMatchScheduler
 # 权重管理
 ├── _freeze_base # 不管是训练还是推理 管道base都是冻结的
 ├── _init_lora   # 初始化LoRA并注入到pipeline中
 ├── save_lora    # 保存LoRA权重
 ├── load_lora    # 加载LoRA权重
 # 编码辅助
 ├── encode_prompt
 ├── encode_reference_image
 ├── decode_latents
 ├── prepare_latents
 ├── apply_3d_rope
 # 训练与推理
 ├── forward()             # training one-step denoising
 ├── generate() / pipeline.__call__()  # inference denoising loop
 ├── utility functions:
 |   ├── _prepare_inference_timesteps
 │   ├── _scheduler_step

 E - FACETPipeline:
 ├── __init__
 ├── __call__


====================================
背景信息:
在WAN2.1-VACE-1.3B的基础上结合OmniControl的架构 实现一个FACET模型 用于单人视频的衣物饰品等编辑
模型需要的输入大致为 [从概念阐述上讲] :
1. src video [源视频也叫做 masked video]   2. src mask [源视频的掩码mask序列]
3. ref image [单张参考图像]                4. prompt / caption [标题或者文本提示]
训练时 dataloader会提供1个未被掩码的原视频 target video 作为训练目标

各个输入的形状变化流程:
a. latent: 
由perpare_latents()函数得到 在对标加和src video的情况下 形状应当为 [z_dim=16, 21, 60, 104] (z_dim, F, H, W)
即标准的 WAN VAE latent: [z_dim, F_lat, H/8, W/8] 
在后续被称为 base branch (x) 

b. src video # 参照于 WAN2.1/wan/vace.py 中的 vace_encode_frames() 函数
src video: [3, 81, 480, 832] 在pipeline中首先使用WAN2.1_VAE进行编码 
WAN2.1_VAE的下采样率是8 而WAN2.2_VAE的下采样率是16
所以 -> VAE: 下采样 (4, 8, 8) -> src latent: [z_dim=16, 21, 60, 104]
由于对于inactive & reactive的拼接 所以 src latent: [z_dim=32, 21, 60, 104] 作为z0

c. src mask  # 参照于 WAN2.1/wan/vace.py 中的 vace_encode_masks() 函数
src mask: [1, 81, 480, 832] 
src mask 不能被VAE处理 所以参考VAE处理src video处理后的形状 即(4, 8, 8)的降采样率
计算src mask在pixel-unshuffle之后的形状 应该为[dim, 21, 60, 104]
随后使用8x8的pixel-unshffle src mask形状由[1, 81, 480, 832]变为 [1, 81, 60, 104, 8]
使用permute(2, 4, 0, 1, 3)以及reshape src mask形状变为 [64, 81, 60, 104] 
再对时间维度使用最邻近插值 使得src mask形状变为 [64, 21, 60, 104] 作为m0

d. vace_context
m0 与 z0 拼接为 vace_context 形状为 [z_dim=96, 21, 60, 104] 
在后续被称为 VACE branch (c) condition

e. reference image
由于单张的ref image需要以OminiControl的形式参与pipeline 所以首先使用同样的WAN2.1_VAE进行编码
形状由[1, 3, 480, 480] -> [z_dim=16, 1, 60, 60] 
在后续被称为 ref branch (r)

f. prompt/caption
使用T5 text encoder进行编码 List[str] [B,] -> tokenizer -> [B, text_len=512]
-> 使用umT5-XXL encoder -> context [B, 512, text_dim=4096]
-> 按seq_lens切掉padding 返回list [[L_i, 4096], ...] Li是prompt的真实token数量
-> 重新堆回 [B, 512, 4096] -> text embedding 
在text embedding中 经过两层linear 分别为(4096->1536)(1536->1536)
最后输出context 为 [B, 512, 1536] 不进行RoPE/AdaN modulation 进入 text cross-attention 模块中进行进一步操作
作为 text branch (t) 

g. base branch (x)
在进入DiT之后 首先使用patchify 得到 patch embedding 采用self.patch_embedding: conv3d k=(1,2,2) s=(1,2,2) 降采样率为(1, 2, 2)
形状变化: [B=1, 16, 21, 60, 104] -> [B=1, dim, 21, 30, 52]
随后采用flatten -> [B=1, L=21*30*52=32760, dim=1536] 成为base branch token 

h. vace branch(c) 
采用类似的patchify + flatten 流程 使用self.vace_patch_embedding: conv3d k=(1,2,2) s=(1,2,2) 降采样率为(1, 2, 2)
[B=1, 96, 21, 60, 104] -> [B=1, dim, 21, 30, 52] -> [B=1, L=21*30*52=32760, dim] 成为vace branch token
此时 base branch 与 vace branch对齐 vace branch 以 residual hints的形式注入base branch中
注入方式有具体的规则和方式 参考 WAN2.1/wan/modules/vace_model.py 默认是偶数层

i. timestep embedding
每个样本需要对应一个时刻 假设时间步t [B,] 在经过sin_embedding之后得到 [B, freq_dim=256]
随后经过time_embedding: 包括两个linear层(freq_dim->time_dim=1536)(time_dim->time_dim) 得到e [B, time_dim=1536]
再经过time_projection 得到 e0 [B, 6, 1536] 分别对应 shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
shift_msa / scale_msa: attention前LN的偏移和缩放 LN(x)(1 + scale_msa) + shift_msa
gate_msa: attention的门控 用于控制attention的强度
shift_mlp / scale_mlp: FFN前LN的偏移和缩放 LN(x)(1 + scale_mlp) + shift_mlp
gate_mlp: FFN的门控 用于控制FFN的强度
随后e0被反复使用 不再重新计算


====================================
LoRA配置
--lora_target_weights: "q, k, v, o, ffn.0, ffn.2"
注意力机制中用于生成q k v o的线性nn层部分(包括ffn.0 ffn.2)本身-base部分冻结
lora权重同样是用于生层q k v o的线性nn层部分 作为base部分的残差
最开始对lora up使用零初始化 lora down使用kaiming uniform 因为激活函数为GeLU 不影响WAN模型本身的生成能力
y = base(x) + scale * B(A(dropout(x))) scale:残差放缩系数

--lora_target_modules 
LoRA需要注入的模块
在WAN2.1-VACE-1.3B中 存在30个base block vace-block每隔1层注入1次 所以应当有15个VACE注入点
对于VaceWanBlock: 考虑self-attn q, k, v, o, ffn.0, ffn.2, vace_block.*.after_proj, vace_block.0.before_proj
对于WanBlock: 考虑所有wanblock的 self-attn q, k, v, o, ffn.0, ffn.2; 
两者的cross-attn 也可以在后期考虑加入 但是目前的任务不以文字控制训练为主
目前A100 80GB 显存 + 1.3B的模型 显存比较充裕

--不适宜注入LoRA的模块/层:
patch embedding; text embedding; timestep embedding; timestep projection;
freqs / RoPE; head.head; norm layers; modulation;


====================================
RoPE设置
rope的freqs在pipeline初始化的时候提前被构造完成  # 参照于 WAN2.1/wan/modules/model.py 
rope只在spatial self-attention中使用 并且在注意力头head内部只对q & k 进行计算 具体位置是每个self-attn里的q/k进行linear之后使用
WAN中的做法是在head_dim=128的情况下 需要给128/2=64个复数对 需要把64分给3个轴 
做法即是 64//3=21 余1 所以x轴21个 剩y轴21个 剩下1个给z轴 z轴22个  # 参照于 WAN2.1/wan/modules/model.py Lines 478 ~ Lines 485
所以对于一个token grid坐标 它的 q / k 会被三组旋转共同编码

base branch 与 vace branch 调用的是同一个 self.freqs张量表 但是使用各自的grid_sizes
两者可通过同一个self attention机制计算 因为如上述信息流中计算的 两个branch在patch embedding输出的token grid一致 
在 forward_vace 里可以把 kwargs (含 grid_sizes、freqs、seq_lens) 直接转给 vace_blocks # 参照于 WAN2.1/wan/modules/vace_model.py
对于ref branch的设置需要参考OminiControl中的实现:
本身OminiControl是用于2d图像生成 根据总的下采样倍率(VAE+patchify)算出了每个token的单位长度是 16px 
所以添加了position delta 一个负值将 reference image放置在了生成图的空间左侧
原论文提到: [spatially-aligned task 让 condition 和 target 共享位置non-aligned task; 
比如 subject-driven generation则给 condition tokens 加固定 offset 避免和 target token 空间重叠 并且这种 shifting 会带来更快收敛和更好效果]

但是在WAN rope中 position embedding的计算 依赖于使用坐标索引freqs 例如-1意味着可能是索引倒数第1个 
解决方法是在计算rope的时候 临时将reference image的f设置为0  latent token全部向右移动1个token 这样所有坐标都是正的了 
或者把referece image的f值临时设置为21(注意这是一个索引值 论数量的话这应该是第22个) 即放置在latent token的右侧
目前更倾向于后者 因为这样不会改变原始latent token的空间位置 符合我在WAN2.1-VACE这种已经经过训练分布适应的模型上进行fine-tuning
并且放置在右侧也不违背OminiControl的工程目的 - ref branch & base branch 在grid空间上没有重叠

同时原始apply rope需要传入grid sizes 目前的 reference branch 在 vae & patchify后形状是 [z_dim=16, 1, 30, 30]
无法连同 latent token 构成一个连续的grid size空间 所以 apply rope函数需要重写 即利用3d position id 来计算
或者说不需要position id 把apply rope改成允许支持offset也可以 然后在 facet/config.yaml中的position项中配置 offset 

RoPE的计算流程 # 参照于 WAN2.1/wan/modules/model.py Lines 31 ~ Lines 70:
如上所说 latent token会被flatten成[1, 32760, 1536] 所以期待RoPE也是相同的形状
首先构造 freqs 位置复数坐标表格 长度为 [1024, 21]或者[1024, 22] 实际上目前的latent只会使用f=0~20, h=0~29, w=0~51
例如 freqs[0][:f].shape = [21, 22]; f_freqs.shape=[21 30, 52, 22] ; h_freqs = w_freqs = [21, 30, 52, 21]
随后对f h w的freqs进行拼接 从而得到3d表格 shape = [21, 30, 52, 64] -> reshape [32760, 1, 64]
此时就能与latent token进行相乘 其中的1会在相乘的时候进行广播


====================================
timestep embedding 设置
在WAN注意力block中 timestep embedding 会作为modulation的输入 同时会作为 AdaLN的输入
        timestep t
            │
     sinusoidal_embedding
            │
        MLP (全局共享)
            │
       e: [B, 6, 1536]  
            │
            ▼
  ┌──────────────────────────┐
  │  + self.modulation       │ ← [1, 6, 1536] 静态 每个 block 可自我学习的参数
  │     [1, 6, 1536]         │
  └──────────────────────────┘
            │  (fp32, 防数值误差)
            ▼
       chunk(6, dim=1)
            │
  ┌─────────┼─────────┬─────────┬─────────┬─────────┐
  ▼         ▼         ▼         ▼         ▼         ▼
shift_msa scale_msa gate_msa shift_mlp scale_mlp gate_mlp
  │         │          │        │         │          │
  └─用于 Attention 分支─┘        └─────用于 FFN 分支───┘

   x ── LN ──×(1+scale_msa) ──+shift_msa ── Attn ── *gate_msa ──+──▶
                                                              │
   ──── LN ──×(1+scale_mlp) ──+shift_mlp ── FFN ── *gate_mlp ──+──▶
# 参照于 WAN2.1/wan/modules/model.py Lines296 ~ Lines316:
在空间自注意力机制中 e使用 shift_msa & scale_msa & gate_msa 来调制空间自注意力机制layernorm
在文本交叉注意力机制中 e使用 shift_mlp & scale_mlp & gate_mlp 来调制交叉注意力机制后的FFN 文本交叉注意力本身完全不被AdaLN调制

base branch & vace branch 走的都是WanAttentionBlock 所以使用同一个e0
对于base branch 和 vace branch 根据当前时间步使用timestep embedding e(t)

对于ref branch 暂时先遵循OminiControl的实现 使用timestep embedding e(0) 即在训练和推理时设置时间步为0 
使得模型间接知道该内容是完全干净的 不需要进行去噪的内容 只进行特征提取  同时 condition k/v 在推理时可以跨step cache 
所以这里涉及对WAN的block中的修改 使不同branch在AdaLN / modulation里使用不同的timestep embedding


====================================
模型中冻结与可训练的参数:
frozen:
  - Wan DiT base weights
  - Wan VAE
  - T5 / UMT5 text encoder
  - scheduler

trainable:
  - LoRA 




# =============================================================================
# model.py: 基于WAN2.1-VACE-1.3B 的FACET模型实现

from __future__ import annotations

import math
import yaml
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

from config import FACETWanConfig, FACETTargetConfig, FACETReferenceConfig, FACETLoRAConfig, FACETInferenceConfig

logger = logging.getLogger(__name__)


# ============================================================
# 1. Model Config
# ============================================================

@dataclass
class FACETConfig:
    name: str = "FACET-WAN2.1-VACE" 
    # TODO: 本地路径权重加载相关的config 坚决不允许huggingface在线下载 权重已在本地文件夹中
    dtype: str = "bf16"
    device: str = "cuda"
    gradient_checkpointing: bool = True

    wan: FACETWanConfig = field(default_factory=FACETWanConfig)
    target: FACETTargetConfig = field(default_factory=FACETTargetConfig)
    reference: FACETReferenceConfig = field(default_factory=FACETReferenceConfig)
    lora: FACETLoRAConfig = field(default_factory=FACETLoRAConfig)
    inference: FACETInferenceConfig = field(default_factory=FACETInferenceConfig)

    @staticmethod
    def from_yaml(path: str) -> "FACETConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        # Keep this parser simple in v1.
        # You can replace it with OmegaConf/Hydra later.
        cfg = FACETConfig()

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


# ============================================================
# 2. LoRA Config
# ============================================================
# FIXME: 在LoRA注入的时候 需要根据config.yaml中的lora部分来进行修改
def lora_targets(module_name: str, target_modules: Sequence[str]) -> bool:
    # """
    # Match by suffix.

    # For Wan:
    #   blocks.0.self_attn.q     -> q
    #   blocks.0.self_attn.k     -> k
    #   blocks.0.self_attn.v     -> v
    #   blocks.0.self_attn.o     -> o
    #   blocks.0.ffn.0           -> ffn.0
    #   blocks.0.ffn.2           -> ffn.2
    # """
    # return any(module_name.endswith(t) for t in target_modules)


# FIXME: 在LoRA注入的时候 需要根据config.yaml中的lora部分来进行修改
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
    # for name, module in list(root.named_modules()):
    #     if not isinstance(module, nn.Linear):
    #         continue
    #     if not lora_targets(name, target_modules):
    #         continue

    #     parent, child_name = _get_parent_module(root, name)
    #     setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
    #     replaced.append(name)

    # return replaced

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
# 3. Utility
# ============================================================

def latent_frames_from_num_frames(num_frames: int, temporal_stride: int = 4) -> int:
    assert (num_frames - 1) % temporal_stride == 0, (
        f"num_frames should be 4n+1 for Wan-style video VAE, got {num_frames}"
    )
    return (num_frames - 1) // temporal_stride + 1


# NOTE: WAN2.1是否如同WAN2.2一样 也期待list[tensor]输入?
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
    FACET model wrapper.

    Training:
        forward() does one denoising prediction step.

    Inference:
        generate() does full denoising loop.

    This class intentionally uses composition instead of subclassing WanModel.
    """

    def __init__(self, cfg: FACETConfig):
        super().__init__()
        self.cfg = cfg
        self.dtype = resolve_dtype(cfg.dtype)
        self.device= cfg.device

        # The following components will be assigned by _load_base_components().
        self.pipe = None
        self.dit = None
        self.vae = None
        self.text_encoder = None
        self.tokenizer = None
        self.scheduler = None

        self._load_base_components()

        self._freeze_base()

        self._init_lora()

        self.to(self.device)  #FIXME: ???  在train.py 中把pipe放置到acc.device上 此处不做设置?

    @classmethod
    def from_config(cls, path: str) -> "FACETWanModel":
        cfg = FACETConfig.from_yaml(path)
        return cls(cfg)

    def _load_base_components(self) -> None:
        """
        Load Wan components.

        You have two implementation choices:

        Option A:
            Use DiffSynth-Studio WanVideoPipeline.

        Option B:
            Use official Wan repo classes directly.
        """
        
        #NOTE: 暂时使用optionA
        #FIXME: 使用本地路径加载 先glob本地文件夹 而非huggingface下载
        # 参照于 DiffSynth/examples/wanvideo/model_inference/Wan2.1-VACE-1.3B.py中的权重载入

        raise NotImplementedError("Implement Wan component loading here.")

    def _freeze_base(self) -> None:
        """
        Freeze originial pipeline components.
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
        # assert self.cfg.lora.base_model == "dit", (
        #     "FACET v1 only supports LoRA on dit. "
        #     "This corresponds to DiffSynth --lora_base_model dit."
        # )

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
        
        #TODO: 兼容prompt/prompt_embeds两种输入方式
        # 如果 prompt_embeds 不为空，优先用 prompt_embeds。
        # 如果 prompt 和 prompt_embeds 都为空，报错。

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
        # TODO: reference_image的格式支持:
        # PIL.Image
        # List[PIL.Image]
        # torch.Tensor [3,H,W]
        # torch.Tensor [B,3,H,W]
        # 目前仅支持1张reference_image输入
        # 需要兼容已经是reference latent / processed reference image tensor / raw PIL 
        # 对于raw PIL 需要的预处理函数可以在 data.transform找到对应的参考代码
        # 预处理伪代码Pseudocode:
        # 1. normalize input to list of PIL or tensor batch
        # 2. center crop + resize to cfg.reference.image_size
        # 3. convert to tensor [B, 3, 1, H, W] or [B, 1, 3, H, W]
        # 4. VAE encode
        # 5. return List[Tensor], each [C, 1, H_lat, W_lat]
        
        # NOTE: 需要注意的是 如果传入的reference image 已经是 cfg.reference.image_size 则不需要再进行尺寸预处理
        # 把tensor转成合适的shape 给VAE就可以了
        # 目前来自data loader的 reference image已经是经过预处理了的

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
        output_type: str = "video",
    ):
        """
        Decode target video latents to RGB video.
        """
        latents = ensure_latent_list(latents)

        # Pseudocode:
        #
        # video = self.vae.decode(latents)
        # postprocess to PIL frames or torch tensor
        # 即在decode之后还需要转成video格式再让pipeline进行return
        # 或者说这部分只负责decode 具体的postprocess以及组装为视频tensor交给pipeline来进行后处理

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
        # NOTE: 好像的确是除以vae_stride 而不是 token_stride
        # 因为patchify步骤出现在latent初始化之后
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
        c = getattr(self.dit, "in_dim", 16) # WAN2.1_VAE z_dim=16

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
    # Rotary Position Embedding
    # --------------------------------------------------------
    # FIXME: 如上所述 这部分需要根据我的架构进行改动 
    @torch.amp.autocast("cuda", enabled=False)
    def apply_3d_rope(
        x: torch.Tensor,
        pos_ids: torch.Tensor,
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        """
        x:
            [B, L, num_heads, head_dim]

        pos_ids:
            [B, L, 3], containing f,h,w position ids.

        freqs:
            Wan freqs, same as self.dit.freqs.
        """
        # B, L, N, D = x.shape
        # c = D // 2

        # freqs_f, freqs_h, freqs_w = freqs.split(
        #     [c - 2 * (c // 3), c // 3, c // 3],
        #     dim=1,
        # )

        # out = []
        # for b in range(B):
        #     ids = pos_ids[b].long()
        #     f_ids = ids[:, 0].clamp(min=0, max=freqs_f.shape[0] - 1)
        #     h_ids = ids[:, 1].clamp(min=0, max=freqs_h.shape[0] - 1)
        #     w_ids = ids[:, 2].clamp(min=0, max=freqs_w.shape[0] - 1)

        #     freqs_i = torch.cat(
        #         [
        #             freqs_f[f_ids],
        #             freqs_h[h_ids],
        #             freqs_w[w_ids],
        #         ],
        #         dim=-1,
        #     )  # [L, D/2]

        #     x_i = torch.view_as_complex(
        #         x[b].to(torch.float64).reshape(L, N, -1, 2)
        #     )  # [L, N, D/2]

        #     x_i = torch.view_as_real(
        #         x_i * freqs_i[:, None, :]
        #     ).flatten(2)

        #     out.append(x_i)

        # return torch.stack(out, dim=0).float()


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
        # TODO: 传入masks 掩膜序列作为vace_context
        # masks会作为vace_context的一部分 用于在VaceWanBlock中进行attention计算
    ) -> Union[Dict[str, Any], List[torch.Tensor]]:
        
        """
        Training forward.

        noisy_latents:
            List length B.
            Each [C, F, H, W].

        timesteps:
            [B], values in [0, 1] if flow matching.

        prompt_embeds:
            List length B.
            Each [L, text_dim], T5 hidden states.

        reference_latents:
            List length B.
            Each [C, 1, H_ref, W_ref].

        edit_masks:
            Optional latent-space or token-space masks.

        Returns:
            pred velocity for target video branch only.
        """
        assert reference_latents is not None, "FACET requires reference_latents."

        # TODO: 在VAE得到 ref_latent之后 记得进行detach操作
        # FIXME: forward函数一次性写出来 不需要反复根据选择option进行分支套壳 

        关于实现OminiControl-style WAN-VACE forward函数:
        
        1. BaseWanBlock:
        在空间自注意力机制Spatial Self-Attention中:
            reference 计算 k_ref / q_ref / v_ref 
            latent token 及实验 k / q / v
            接合concat k_ref / v_ref 得到 k_all / v_all
            将latent branch v attention 到 k_all / v_all 并将 q_ref attend 仅attend 至自身 k_ref & v_ref
            (可能会涉及 group mask的使用) 
            于是 ref token 形成一种类似于 layer-wise condition memory
            在forward过程中 ref token本身也要发生变化 以适应WAN不同stage latent注意力计算的需要 例如不同的尺度细粒度纹理等...

        在文本交叉注意力机制中
            ref_branch暂时不参与 (需要做成一个forward函数中的optional选项)
            让latent token生成的query attend 到 text token的key和value

        target branch:
            uses Wan patch_embedding
            uses Wan q/k/v/o
            uses Wan ffn
            uses LoRA on those modules
        reference branch:
            uses same Wan patch_embedding
            uses same Wan q/k/v/o
            uses same Wan ffn
            uses same LoRA on those modules
        所以Lora注入和OminiControl架构下的branch attention是可以相互独立进行的 因为reference branch复用了原始&LoRA的参数
        
        2. VACEWanBlock:
        src video & src mask 作为vace_branch 进入 VACEWanBlock 
        vace_context以hints的形式作为残差注入target branch - 参照于WAN-VACE 

        3. head:
            只作用于target branch

        由于现在的FACET模型既基于VACE架构 又使用了OminiControl 所以forward需要手搓
        并且利用比较底层的 VaceWanBlock进行forward
        会产生类似于如下的代码:
        # for block in self.dit.blocks:
        #     xx, yy = self.(xxx)
    



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
        # 需要做input check 而且需要有兼容性 例如wan期待传入的list 但是传入tensor stack 
        # 所以需要tensor转list[tensor] 还另外包括shape mismatch的尝试处理 能解决就解决(例如squeeze/unsqueeze)
        # 不能解决就只能报错了
        # 而且call函数需要兼容直接给原始视频或者直接给已经编码好的target video 
        # 兼容caption 或者已经编码好的T5 caption embedding
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

            # FIXME: 这一段删除 不可能不要reference image
            # Optional reference guidance:
            # cond with reference vs cond with empty/no reference.
            # This is useful if model tends to ignore reference.
            # if reference_guidance_scale is not None and reference_guidance_scale != 1.0 and ref_latents is not None:
            #     pred_no_ref = self.forward(
            #         noisy_latents=cur_latents,
            #         timesteps=t_batch,
            #         prompt_embeds=cond_context,
            #         reference_latents=None,
            #         return_dict=True,
            #     )["pred"]

            #     pred = [
            #         nr + reference_guidance_scale * (r - nr)
            #         for r, nr in zip(pred, pred_no_ref)
            #     ]

            # 6. Scheduler step.
            cur_latents = self._scheduler_step(
                pred=pred,
                timestep=t,
                latents=cur_latents,
            )

        # 7. Decode final target latents only.
        # FIXME: 如果是要求传回video 这里还需要post-process
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