"""
CelebVDataset: https://github.com/CelebV-HQ/CelebV-HQ

Dataset Layout (dataset_root):
    clip/{clip_id}/
        {clip_id}.mp4    raw normalized video (832x480 target, aspect-preserving + pad, >=81f, not blacked out)
        masks.npz        uint8 [T,H,W] binary mask
        meta.json        category / ref_candidates / (empty) caption
        ref_imgs/
            *.png        RGBA: bbox fg alpha=255, padding region alpha=0; 480x480
    latents/{clip_id}.pt {tgt_latent: [48,T',H',W'], t5_emb: [L,4096]}

splits/{train,val}.jsonl schema:
    one JSON object per line, same shape as a row in index.jsonl, e.g.
        {"clip_id": "M2Ohb0FAaJU_1", "ytb_id": "M2Ohb0FAaJU", "n_refs": 3,
         "path": "clip/M2Ohb0FAaJU_1"}
    Only `clip_id` is read at training time; extra fields are ignored.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from decord import VideoReader, cpu

from data.base import BaseVideoDataset
from data.datasets import register


@register("celebv")
class CelebV(BaseVideoDataset):
    SOURCE = "celebv"

    # self.items = self._build_index() automatically called in BaseVideoDataset.__init__
    def _build_index(self) -> List[Dict[str, str]]:
        """Read split_dir / {split}.jsonl. Each line = {"clip_id": "...", ...}."""
        split_path = Path(self.cfg.split_dir) / f"{self.split}.jsonl"
        items: List[Dict[str, str]] = []
        with open(split_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items

    def _load(self, idx: int) -> Dict[str, Any]:  # idx -> from global sampler index
        item = self.items[idx]
        cid = item["clip_id"]
        d = self._clip_dir(cid)

        with open(d / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        video = self._read_video(d / f"{cid}.mp4", self.cfg.num_frames)  # [T,H,W,3] uint8
        mask = np.load(d / "masks.npz")["mask"]                          # [T,H,W] uint8
        T = video.shape[0]
        if mask.shape[0] != T:
            mask = mask[:T]

        ref_pool = sorted((d / "ref_imgs").glob("*.png"))

        out: Dict[str, Any] = {
            "clip_id": cid,
            "video": video,
            "mask": mask,
            "ref_pool": ref_pool,
            "category": meta.get("category", "hair"),
        }
        out.update(self._load_cache(cid))
        return out

    def _clip_dir(self, cid: str) -> Path:
        return Path(self.cfg.data_root) / "clip" / cid

    def _load_cache(self, cid: str) -> Dict[str, Any]:
        """Load cache produced by data/celebv/pipeline/cache.py.
           Layout: {latent_cache_dir}/{cid}.pt = {tgt_latent, t5_emb} (bf16 by default).
        """
        if not getattr(self.cfg, "latent_cache", False):
            return {}
        p = Path(self.cfg.latent_cache_dir) / f"{cid}.pt"
        if not p.exists():
            return {}
        # weights_only=True: safe load since cache only contains tensors.
        d = torch.load(p, map_location="cpu", weights_only=True)
        return {"tgt_latent": d.get("tgt_latent"), "t5_emb": d.get("t5_emb")}

    @staticmethod
    def _read_video(path: Path, num_frames: int) -> np.ndarray:
        vr = VideoReader(str(path), ctx=cpu(0))
        n = len(vr)
        if n < num_frames:  # repeat last frame as a safeguard (main.py keeps >= NF frames)
            idx = list(range(n)) + [n - 1] * (num_frames - n)
        else:
            idx = list(range(num_frames))
        return vr.get_batch(idx).asnumpy()  # [T,H,W,3] uint8
