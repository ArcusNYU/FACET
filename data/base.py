"""
BaseVideoDataset: ref-video dataset unified contract (Template Method).

subclasses only need to implement `_build_index` and `_load`. 
The rest of the process (mask boundary perturbation / ref sampling /masked_video synthesis / normalization) 
is completed in the base class, ensuring consistency across multiple datasets.
"""

from __future__ import annotations
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from data.transform import TfmBundle
from data.ref_sampler import RefSampler


class BaseVideoDataset(Dataset):
    def __init__(
        self,
        cfg,
        transforms: TfmBundle,
        ref_sampler: RefSampler,
        split: str = "train",
    ):
        self.cfg = cfg
        self.split = split
        self.tfm = transforms                      # mask perturb + resize + normalize
        self.ref_sampler = ref_sampler             # reference image sampling strategy
        self.items: List[Any] = self._build_index()# List[clip_id]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        meta = self._load(idx)
        video = self.tfm.video(meta["video"])           # [T,3,H,W] in [-1,1]
        mask  = self.tfm.mask(meta["mask"])             # [T,1,H,W] in {0,1}, perturbed
        masked_video = video * (1.0 - mask)             # [T,3,H,W]

        ref_img = self.ref_sampler.pick(meta["ref_pool"], meta["mask"])
        ref_img = self.tfm.ref(ref_img)                 # [3,H,W] in [-1,1]

        return {
            "clip_id":      meta["clip_id"],
            "category":     meta["category"],
            "masked_video": masked_video,
            "ref_img":      ref_img,
            "tgt_latent":   meta.get("tgt_latent"),     # Tensor or None 
            "t5_emb":       meta.get("t5_emb"),         # Tensor or None 
            # "video": video,
            # "mask": mask,
            # "caption": meta["caption"],
            # "source": meta["source"], # source: original dataset name
            # "path": meta["path"],
        }

    # ---- subclass hooks ---------------------------------------------------------
    # class xxx_dataset inherit the hook methods

    def _build_index(self) -> List[Any]:
        """Return list of index entries, e.g. [{"id": "f605...", "part": "part_001"}, ...]"""
        raise NotImplementedError

    def _load(self, idx: int) -> Dict[str, Any]:
        """Return raw asset dict (before transforms):
            clip_id  : str
            category : str
            video    : np.ndarray [T,H,W,3] uint8
            mask     : np.ndarray [T,H,W]   uint8
            ref_pool : List[Path]            RGBA png candidates
            tgt_latent : Tensor | absent     from latent cache
            t5_emb     : Tensor | absent     from latent cache
        """
        raise NotImplementedError

    # ---- optional cache hook --------------------------------------------------

    def _load_cache(self, part: str, cid: str) -> Dict[str, Any]:
        """Empty by default; subclass overrides to return {tgt_latent, t5_emb}."""
        return {}
