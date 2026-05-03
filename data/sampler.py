"""
Per-dataset epoch-budgeted sampler for `torch.utils.data.ConcatDataset`.

Semantics
---------
Each epoch, for every sub-dataset `i` with quota `q_i`:
    - if q_i <= size_i  : pick q_i indices WITHOUT replacement
    - if q_i  > size_i  : full pass over the dataset + (q_i - size_i) random
                          indices WITH replacement (cap over-subscription)
We then offset each local index by the ConcatDataset cumulative offset and
shuffle the union so cross-dataset batches are mixed.

DDP
---
If `num_replicas > 1`, the final global index list is sliced per rank after
shuffling, so each rank sees a disjoint subset of the same epoch draw.

Reproducibility
---------------
Call `set_epoch(epoch)` from the training loop. Indices drawn in epoch K are a
deterministic function of (seed, K), matching the standard DistributedSampler
idiom.
"""

from __future__ import annotations
from typing import Iterator, List

import torch
from torch.utils.data import ConcatDataset, Sampler


class MultiSampler(Sampler[int]):
    def __init__(
        self,
        concat: ConcatDataset,
        quotas: List[int],
        num_replicas: int = 1, # for DDP multi-rank
        rank: int = 0,
        seed: int = 0,
        drop_last: bool = True,
    ):
        if len(quotas) != len(concat.datasets):
            raise ValueError(
                f"len(quotas)={len(quotas)} != len(concat.datasets)={len(concat.datasets)}"
            )
        if num_replicas < 1 or not (0 <= rank < num_replicas):
            raise ValueError(f"bad DDP config: num_replicas={num_replicas}, rank={rank}")
 
        self.concat = concat
        self.quotas = [int(q) for q in quotas]
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        # cumulative_sizes = [n0, n0+n1, ...]
        self.cum = list(concat.cumulative_sizes)
        # offset of each sub-dataset in the global index space, for indicing on cumulative sizes:
        self.offsets = [0] + self.cum[:-1]

    # ---- public ------------------------------------------------------------
    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self._per_rank_total()

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        chunks: List[torch.Tensor] = []
        for i, (off, q) in enumerate(zip(self.offsets, self.quotas)):
            size = len(self.concat.datasets[i])
            if q == 0 or size == 0:
                continue
            if q <= size:
                local = torch.randperm(size, generator=g)[:q]
            else: # Use `randint` to randomly select `q-size` items with replacement to make up for the shortfall.
                base  = torch.randperm(size, generator=g)                  # full pass
                extra = torch.randint(0, size, (q - size,), generator=g)   # top-up w/ replacement
                local = torch.cat([base, extra])
            chunks.append(local + off)

        if not chunks:
            return iter([])

        stacked = torch.cat(chunks) # stack for all sub-datasets
        perm = torch.randperm(len(stacked), generator=g)
        stacked = stacked[perm] # len(stacked) = sum(quotas)

        if self.num_replicas > 1:
            total = self._global_total()
            stacked = stacked[:total][self.rank::self.num_replicas]

        return iter(stacked.tolist())

    # ---- internals ---------------------------------------------------------
    def _global_total(self) -> int: # total number should be divisible by num_replicas for exact division on DDP
        t = sum(self.quotas)
        if self.drop_last and self.num_replicas > 1:
            t = (t // self.num_replicas) * self.num_replicas
        return t

    def _per_rank_total(self) -> int:
        t = self._global_total()
        if self.num_replicas <= 1:
            return t
        # drop_last guarantees exact division; without it rank may see ceil(t/n) or floor
        per = t // self.num_replicas
        if not self.drop_last and self.rank < (t - per * self.num_replicas):
            per += 1
        return per
