"""
OpenVidDataset: https://huggingface.co/datasets/Owen777/HQ-OpenHumanVid/tree/main.

Dataset Layout (dataset_root):
    clips/{part}/{ab}/{cd}/{clip_id}/
        {clip_id}.mp4    raw normalized video (832x480 target, aspect-preserving + pad, 24fps, >=81f, not blacked out)
        masks.npz        uint8 [T,H,W] binary mask
        meta.json        caption / category / ref_candidates / scores
        ref_imgs/
            *.png        RGBA: bbox fg alpha=255, padding region alpha=0; 480x480

splits/train.json, splits/val.json schema:
    {
      "clips": [
        {"id": "f605...", "part": "part_001"}, the first 4 characters of clip_id are 2 hash index
        ...
      ]
    }
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from decord import VideoReader, cpu

from data.base import BaseVideoDataset


class OpenVid(BaseVideoDataset):
    SOURCE = "openvid"

    # self.items = self._build_index() automatically called in BaseVideoDataset.__init__
    def _build_index(self) -> List[Dict[str, str]]:
        """index list of the dataset clips
           e.g. [{"id": "f605...", "part": "part_001"}, ...]
        split_dir holds {train.json, val.json}; 
        """
        split_path = Path(self.cfg.split_dir) / f"{self.split}.json"
        with open(split_path, "r", encoding="utf-8") as f:
            split = json.load(f)
        return split["clips"]

    def _load(self, idx: int) -> Dict[str, Any]: # idx -> from global sampler index
        item = self.items[idx]  
        cid, part = item["id"], item["part"]
        d = self._clip_dir(part, cid)

        with open(d / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        video = self._read_video(d / f"{cid}.mp4", self.cfg.num_frames)  # [T,H,W,3] uint8
        # masks.npz is a zip archive produced by np.savez_compressed(path, mask=arr);
        # np.load returns an NpzFile mapping, so ["mask"] pulls the array back out.
        # zlib-compressed .npz beats per-frame png sequence on both inode count and IO.
        mask  = np.load(d / "masks.npz")["mask"]                         # [T,H,W] uint8
        T = video.shape[0]
        if mask.shape[0] != T:
            # mask & video shape mismatch: truncate mask to video length
            mask = mask[:T]

        ref_pool = sorted((d / "ref_imgs").glob("*.png"))

        out: Dict[str, Any] = {
            "clip_id": cid,
            "source": self.SOURCE,
            "path": f"clips/{part}/{cid[:2]}/{cid[2:4]}/{cid}",
            "video": video,
            "mask": mask,
            "ref_pool": ref_pool,
            "caption": meta["caption"],
            "category": meta.get("category", "upper_clothes"),
        }
        out.update(self._load_cache(cid))
        return out

    def _clip_dir(self, part: str, cid: str) -> Path:
        return Path(self.cfg.data_root) / "clips" / part / cid[:2] / cid[2:4] / cid

    def _load_cache(self, cid: str) -> Dict[str, Any]:
        if not getattr(self.cfg, "latent_cache", False):
            return {}
        p = Path(self.cfg.latent_cache_dir) / cid[:2] / cid[2:4] / f"{cid}.pt"
        if not p.exists():
            return {}
        # weights_only=True: safe load since cache only contains tensors.
        d = torch.load(p, map_location="cpu", weights_only=True)
        return {"tgt_latent": d.get("tgt_latent"), "t5_emb": d.get("t5_emb")}

    @staticmethod
    def _read_video(path: Path, num_frames: int) -> np.ndarray:
        vr = VideoReader(str(path), ctx=cpu(0))
        n = len(vr)
        # idx: list of frame indices to be read from the video
        # NOTE: 实际在前期数据集准备的时候 只会保留大于等于81帧的视频 不过这里做一个保险
        if n < num_frames: # repeat the last frame to make up the difference
            idx = list(range(n)) + [n - 1] * (num_frames - n)
        else: # first 'num_frames' frames
            idx = list(range(num_frames))
        return vr.get_batch(idx).asnumpy()  # [T,H,W,3] uint8
