"""
Training dataloader factory.
FACET Training pipeline Step 5.

Exported:
    build_loaders   -> (train_loader, val_loader, train_sampler, val_sampler)
    collate_batch   -> DataLoader collate_fn

collate design:
    train.py / FACETWanModel.forward receive batched tensors directly (no manual
    torch.stack downstream). Two field groups stay as Python lists:
      - t5_emb            : [L_i, 4096], L_i differs per sample (cannot stack)
      - clip_id / category: plain str metadata
    A tensor field whose value can be missing (e.g. tgt_latent on a cache miss)
    is kept as a list when ANY element is None, otherwise stacked.

    Resulting batch layout (B = batch size):
      clip_id    : List[str]
      category   : List[str]
      tgt_video  : [B, T, 3, H, W]   in [-1, 1]
      src_video  : [B, T, 3, H, W]   in [-1, 1]
      src_mask   : [B, T, 1, H, W]   in {0, 1}
      ref_img    : [B, 3, H, W]      in [-1, 1]
      tgt_latent : [B, z, T', H', W']  (cached) | List[None] (cache miss)
      t5_emb     : List[[L_i, 4096]]   (cached) | List[None] (cache miss)
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from data.datasets import build_datasets
from data.sampler import MultiSampler


def build_loaders(
    cfg_path: str = "data/config.yaml",
    batch_size: int = 1,
    num_workers: int = 0,
    seed: int = 0,
    rank: int = 0,          # DDP rank id  (0-indexed)
    num_replicas: int = 1,  # DDP world size
    drop_last_train: bool = True,
) -> Tuple[DataLoader, DataLoader, MultiSampler, MultiSampler]:
    """Compose (ConcatDataset + MultiSampler + DataLoader) for train & val.

    Call `train_sampler.set_epoch(epoch)` at the start of every epoch so the
    per-epoch index draw is refreshed deterministically.
    """
    train_concat, val_concat, train_q, val_q = build_datasets(cfg_path)

    train_sampler = MultiSampler(
        train_concat, train_q,
        num_replicas=num_replicas, rank=rank, seed=seed,
        drop_last=drop_last_train,
    )
    val_sampler = MultiSampler(
        val_concat, val_q,
        num_replicas=num_replicas, rank=rank, seed=seed,
        drop_last=False,
    )

    train_loader = DataLoader(
        train_concat,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_batch,
        drop_last=drop_last_train,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_concat,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=collate_batch,
        drop_last=False,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader, train_sampler, val_sampler


# Fields that stay Python lists:
#   - t5_emb  : variable-length [L_i, 4096] per sample (FACETWanModel.forward
#               consumes prompt_embeds as a List[Tensor], padding inside)
#   - clip_id / category : non-tensor str metadata
_LIST_FIELDS = frozenset({"clip_id", "category", "t5_emb"})


def collate_batch(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack per-sample tensors into batched tensors; keep list fields as lists.
    """
    if not samples:
        return {}
    out: Dict[str, Any] = {}
    for k in samples[0].keys():
        vals = [s.get(k) for s in samples]
        if k in _LIST_FIELDS:
            out[k] = vals
        elif all(torch.is_tensor(v) for v in vals):
            out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals
    return out
