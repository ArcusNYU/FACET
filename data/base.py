"""
BaseVideoDataset: ref-video dataset unified contract (Template Method).

subclasses only need to implement `_build_index` and `_load`. 
The rest of the process (mask boundary perturbation / ref sampling /masked_video synthesis / normalization) 
is completed in the base class, ensuring consistency across multiple datasets.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from data.transform import TfmBundle
from data.ref_sampler import RefSampler


class BaseVideoDataset(Dataset):
    def __init__(self, cfg, transforms: TfmBundle, ref_sampler: RefSampler):
        self.cfg = cfg
        self.tfm = transforms                      # mask perturb + resize + normalize
        self.ref_sampler = ref_sampler             # reference image sampling strategy
        self.items: List[Any] = self._build_index()# List[clip_id]

    def __len__(self) -> int:
        return len(self.items)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        meta = self._load(idx)
        video = self.tfm.video(meta["video"])           # [T,3,H,W] in [-1,1], clean target
        #FIXME: 在使用prepare.py对数据盘中数据集进行转换的时候 可能已经进行了resize 但是不一定转成[-1,1]
        #所以这里依然可以保留 因为进入self.tfm.video的时候会自动跳过resize
        mask  = self.tfm.mask(meta["mask"])             # [T,1,H,W] in {0,1} unified perturbated mask sequence
        masked_video = video * (1.0 - mask)             # [T,3,H,W] on-time mask

        ref_img = self.ref_sampler.pick(meta["ref_pool"], meta["mask"])
        ref_img = self.tfm.ref(ref_img)                 # [3,H,W] in [-1,1]

        out: Dict[str, Any] = {
            "video": video,
            "masked_video": masked_video,
            "mask": mask,
            "ref_img": ref_img,
            "caption": meta["caption"],
            "category": meta["category"],
            "clip_id": meta["clip_id"],
            "source": meta.get("source", ""), # source: original dataset name
            "path": meta["path"],
        }
        # TODO: phase2 使用cache-latents.py准备好latent cache(tgt latent)后填充, 否则保持 None
        out["tgt_latent"] = meta.get("tgt_latent")
        out["t5_emb"]     = meta.get("t5_emb")
        return out

    # ---- subclass hooks ---------------------------------------------------------
    # class xxx_dataset inherit the hook methods

    def _build_index(self) -> List[Any]:
        """index list of the dataset clips
           e.g. [{"id": "f605...", "part": "part_001"}, ...]
        """
        raise NotImplementedError

    def _load(self, idx: int) -> Dict[str, Any]:
        """
        子类返回原始资产字典:
            {
              "clip_id": str, "source": str, "path": str,
              "video":   np.ndarray [T,H,W,3] uint8,
              "mask":    np.ndarray [T,H,W]   uint8,
              "ref_pool":List[Path],            # 候选 ref jpg 路径
              "caption": str,
              "category":str,
              # 可选: "tgt_latent": Tensor, "t5_emb": Tensor
            }
        """
        raise NotImplementedError

    # ---- 缓存可选 ---------------------------------------------------------
    def _load_cache(self, cid: str) -> Dict[str, Any]:
        """latent_cache 默认空; 子类可覆盖, 命中则返回 {tgt_latent, t5_emb}."""
        return {}
