"""
Training dataloader factory.

Exported:
    build_loaders   -> (train_loader, val_loader, train_sampler, val_sampler)
    collate_batch   -> DataLoader collate_fn

Usage in train.py / eval.py:
    from loader import build_loaders
    train_loader, val_loader, train_sampler, _ = build_loaders(...)
    for epoch in range(n_epochs):
        train_sampler.set_epoch(epoch)
        for batch in train_loader:
            x       = batch["masked_video"]   # List[Tensor [T,3,H,W]] in [-1,1]
            masks   = batch["mask"]           # List[Tensor [T,1,H,W]] in {0,1}
            ref_imgs = batch["ref_img"]       # List[Tensor [3,H,W]]   in [-1,1]
            context = batch["t5_emb"]         # List[Tensor [L,4096]]  or List[None]
            tgt     = batch["tgt_latent"]     # List[Tensor [48,T',H',W']] or List[None]

collate design:
    WanModel.forward takes  x: List[Tensor [C_in, F, H, W]] natively -- each sample
    has NO extra batch dim.  All fields are kept as plain Python lists so training
    code can pass them straight to vae.encode(x) / model(x, context=context, ...).
    Nothing is stacked; no shape assumptions are imposed here.
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


def collate_batch(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Keep every field as a Python list -- no stacking.

    WanModel.forward(x, context, ...) already expects List[Tensor] where each
    tensor has no batch dimension ([C, F, H, W] / [L, D] / ...).
    """
    if not samples:
        return {}
    out: Dict[str, Any] = {}
    for k in samples[0].keys():
        out[k] = [s.get(k) for s in samples]
    return out
