"""Filesystem layout. Everything lives under CXR_DATA_ROOT (default /data/cxr), outside the repo."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataPaths:
    root: Path

    @classmethod
    def from_env(cls) -> DataPaths:
        return cls(Path(os.environ.get("CXR_DATA_ROOT", "/data/cxr")))

    # raw inputs
    @property
    def nih_raw(self) -> Path:
        return Path(os.environ.get("CXR_NIH_RAW", self.root / "raw" / "nih"))

    @property
    def chexpert_raw(self) -> Path:
        return Path(os.environ.get("CXR_CHEXPERT_RAW", self.root / "raw" / "chexpert"))

    @property
    def chexlocalize_raw(self) -> Path:
        return Path(os.environ.get("CXR_CHEXLOCALIZE_RAW", self.root / "raw" / "chexlocalize"))

    @property
    def vindr_raw(self) -> Path:
        return Path(os.environ.get("CXR_VINDR_RAW", self.root / "raw" / "vindr"))

    # processed
    def processed(self, dataset: str) -> Path:
        return self.root / "processed" / dataset

    def images(self, dataset: str, size: int) -> Path:
        return self.processed(dataset) / f"img_{size}.u8"

    def index(self, dataset: str) -> Path:
        return self.processed(dataset) / "index.parquet"

    # outputs
    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def eval_artifacts(self) -> Path:
        return self.root / "artifacts" / "eval"

    @property
    def bundles(self) -> Path:
        return self.root / "artifacts" / "bundles"
