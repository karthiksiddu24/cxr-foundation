"""Batch loaders that yield uint8 image batches already on the GPU.

Two strategies, same interface:
- `GpuResidentLoader`: whole uint8 array lives on the GPU (224 px NIH = 5.6 GB). Batching is a
  GPU gather, so the 8-core CPU never touches pixels during training.
- `MemmapLoader`: for arrays too large for VRAM (512 px = 29 GB). Worker processes slice whole
  batches from the memmap (one sorted fancy-index read per batch) into pinned memory.

Both yield `Batch(x_u8 [B,S,S] cuda uint8, y [B,K] cuda float32, rows [B] int64 cpu)`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


@dataclass
class Batch:
    x: torch.Tensor  # uint8 [B, S, S] on device
    y: torch.Tensor  # float32 [B, K] on device
    rows: torch.Tensor  # int64 [B] (cpu): row in the full index


class GpuResidentLoader:
    def __init__(
        self,
        images: torch.Tensor,  # uint8 [N, S, S] already on device (full array, shared)
        labels: np.ndarray,  # [N, K] for the full array
        rows: np.ndarray,  # subset of rows to iterate
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        seed: int = 0,
    ) -> None:
        self.images = images
        self.device = images.device
        self.labels = torch.as_tensor(labels, dtype=torch.float32, device=self.device)
        self.rows = torch.as_tensor(np.asarray(rows, dtype=np.int64))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.gen = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        n = len(self.rows)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def __iter__(self) -> Iterator[Batch]:
        order = (
            self.rows[torch.randperm(len(self.rows), generator=self.gen)]
            if self.shuffle
            else self.rows
        )
        for i in range(len(self)):
            r = order[i * self.batch_size : (i + 1) * self.batch_size]
            rd = r.to(self.device, non_blocking=True)
            yield Batch(self.images.index_select(0, rd), self.labels.index_select(0, rd), r)

    @staticmethod
    def load_to_device(path: Path, n: int, size: int, device: torch.device) -> torch.Tensor:
        """Read a uint8 memmap into device memory in chunks (avoids a 2x host RAM peak)."""
        mm = np.memmap(path, dtype=np.uint8, mode="r", shape=(n, size, size))
        out = torch.empty((n, size, size), dtype=torch.uint8, device=device)
        step = 8192
        for i in range(0, n, step):
            out[i : i + step] = torch.from_numpy(np.array(mm[i : i + step])).to(device)
        return out


class _BatchMemmap(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Dataset whose __getitem__ receives a whole batch of rows (via a batch sampler)."""

    def __init__(self, path: Path, n: int, size: int, labels: np.ndarray) -> None:
        self.path, self.n, self.size = path, n, size
        self.labels = labels.astype(np.float32)
        self._mm: np.memmap | None = None

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, rows: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._mm is None:  # opened lazily in each worker
            self._mm = np.memmap(
                self.path, dtype=np.uint8, mode="r", shape=(self.n, self.size, self.size)
            )
        r = np.sort(np.asarray(rows, dtype=np.int64))
        x = torch.from_numpy(np.ascontiguousarray(self._mm[r]))
        return x, torch.from_numpy(self.labels[r]), torch.from_numpy(r)


class _RowBatchSampler(Sampler[list[int]]):
    def __init__(
        self, rows: np.ndarray, batch_size: int, shuffle: bool, drop_last: bool, seed: int
    ):
        self.rows, self.bs, self.shuffle, self.drop_last = rows, batch_size, shuffle, drop_last
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        n = len(self.rows)
        return n // self.bs if self.drop_last else -(-n // self.bs)

    def __iter__(self) -> Iterator[list[int]]:
        order = self.rng.permutation(self.rows) if self.shuffle else self.rows
        for i in range(len(self)):
            yield order[i * self.bs : (i + 1) * self.bs].tolist()


class MemmapLoader:
    def __init__(
        self,
        path: Path,
        n: int,
        size: int,
        labels: np.ndarray,
        rows: np.ndarray,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        device: torch.device,
        seed: int = 0,
        workers: int = 6,
    ) -> None:
        self.device = device
        self.sampler = _RowBatchSampler(np.asarray(rows), batch_size, shuffle, drop_last, seed)
        self.dl = DataLoader(
            _BatchMemmap(path, n, size, labels),
            sampler=self.sampler,
            batch_size=None,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
            prefetch_factor=4 if workers > 0 else None,
        )

    def __len__(self) -> int:
        return len(self.sampler)

    def __iter__(self) -> Iterator[Batch]:
        for x, y, r in self.dl:
            yield Batch(
                x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True), r
            )
