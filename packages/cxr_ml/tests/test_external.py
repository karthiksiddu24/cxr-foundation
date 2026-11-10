"""CheXpert/VinDr prep on synthetic files laid out like the real releases (no real data)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import polars as pl

from cxr_common.labels import CHEXPERT_MAP
from cxr_ml.data.external import chexpert_index, prep_chexpert

LABELS = [m.target for m in CHEXPERT_MAP]


def fake_chexpert(root: Path) -> None:
    rows = []
    for p in range(4):
        for view in ("frontal", "lateral"):
            rel = f"valid/patient{64540 + p}/study1/view1_{view}.jpg"
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(root / rel), np.full((320, 390), 100 + p, np.uint8))
            row = {"Path": f"CheXpert-v1.0-small/{rel}", "Frontal/Lateral": view.capitalize()}
            row.update({k: str(float((p + i) % 2)) for i, k in enumerate(LABELS)})
            rows.append(row)
    rows[0][LABELS[0]] = "-1.0"  # an uncertain label must become null
    pl.DataFrame(rows).write_csv(root / "valid.csv")


def test_chexpert_index_frontal_only_prefix_stripped(tmp_path: Path) -> None:
    fake_chexpert(tmp_path)
    idx = chexpert_index(tmp_path, "valid")
    assert idx.height == 4
    assert idx["path"][0].startswith("valid/patient")
    assert idx["patient_id"].to_list() == [64540, 64541, 64542, 64543]
    assert idx[LABELS[0]][0] is None


def test_prep_chexpert_writes_letterboxed_memmaps(tmp_path: Path) -> None:
    raw, out = tmp_path / "raw", tmp_path / "out"
    fake_chexpert(raw)
    prep_chexpert(raw, out, [64])
    mm = np.memmap(
        out / "chexpert_valid" / "img_64.u8", dtype=np.uint8, mode="r", shape=(4, 64, 64)
    )
    # 320x390 landscape -> letterboxed with black bars top/bottom, content in the middle
    assert mm[0, 0].max() == 0
    assert mm[0, 32, 32] == 100
    assert (out / "chexpert_valid" / "index.parquet").exists()
