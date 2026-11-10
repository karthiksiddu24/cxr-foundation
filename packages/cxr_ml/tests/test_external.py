"""CheXpert/VinDr prep on synthetic files laid out like the real releases (no real data)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import polars as pl
import pytest

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


# ---- CheXlocalize: COCO RLE masks and labels derived from them --------------------------------

from cxr_ml.data.rle import decode_counts, rle_decode, rle_encode, rle_is_empty  # noqa: E402


def test_rle_roundtrip_and_column_major_order() -> None:
    rng = np.random.default_rng(0)
    cases = [
        np.zeros((5, 7), bool),
        np.ones((5, 7), bool),
        rng.random((37, 53)) > 0.5,
        np.pad(np.ones((40, 30), bool), ((300, 10), (5, 900))),  # long runs: multi-char values
    ]
    for m in cases:
        r = rle_encode(m)
        assert r["size"] == list(m.shape)
        np.testing.assert_array_equal(rle_decode(r), m)
        assert rle_is_empty(r) == (not m.any())
    # COCO RLE runs down columns: a 2x3 mask with only (row 1, col 0) set -> runs [1, 1, 4]
    m = np.zeros((2, 3), bool)
    m[1, 0] = True
    assert decode_counts(str(rle_encode(m)["counts"])) == [1, 1, 4]


def fake_chexlocalize(chex: Path, loc: Path) -> None:
    """3 frontal + 1 lateral image; patient 0 has cardiomegaly, patient 1 nothing segmented."""
    loc.mkdir(parents=True)
    seg = {}
    for p in range(3):
        for view in ("frontal", "lateral"):
            if view == "lateral" and p:
                continue
            rel = f"valid/patient{64540 + p}/study1/view1_{view}.png"
            (chex / rel).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(chex / rel), np.full((40, 60), 120, np.uint8))  # H=40, W=60
    empty = rle_encode(np.zeros((40, 60), bool))
    heart = np.zeros((40, 60), bool)
    heart[20:30, 25:45] = True
    for p, ann in ((0, {"Cardiomegaly": rle_encode(heart)}), (2, {"Edema": rle_encode(heart)})):
        seg[f"patient{64540 + p}_study1_view1_frontal"] = {
            n: ann.get(n, empty)
            for n in ("Cardiomegaly", "Edema", "Atelectasis", "Pleural Effusion", "Lung Lesion")
        }
    (loc / "gt_segmentations_val.json").write_text(json.dumps(seg))


def test_chexlocalize_labels_and_masks(tmp_path: Path) -> None:
    chex, loc, out = tmp_path / "chexpert", tmp_path / "chexlocalize", tmp_path / "out"
    fake_chexlocalize(chex, loc)
    with pytest.raises(FileNotFoundError):
        chexpert_index(chex, "valid")  # no CSV and no CheXlocalize root -> clear error
    idx = chexpert_index(chex, "valid", loc)
    assert idx.height == 3  # frontal only
    assert idx["label_source"].unique().to_list() == ["chexlocalize_masks"]
    assert idx["Cardiomegaly"].to_list() == [1.0, 0.0, 0.0]
    assert idx["Edema"].to_list() == [0.0, 0.0, 1.0]
    assert idx["Consolidation"].to_list() == [0.0, 0.0, 0.0]  # not segmented anywhere -> negative
    assert idx["Pneumonia"].null_count() == 3  # not part of CheXlocalize -> unlabeled

    prep_chexpert(chex, out, [30], loc)
    d = out / "chexpert_valid"
    meta = pl.read_parquet(d / "masks.parquet")
    assert meta.select("row", "finding").rows() == [(0, "Cardiomegaly"), (2, "Edema")]
    masks = np.memmap(d / "masks_30.u8", dtype=np.uint8, mode="r", shape=(2, 30, 30))
    # 40x60 -> scale 0.5, padded 5 px top: rows 20:30 -> 15:20, cols 25:45 -> 12/13:22/23
    ys, xs = np.nonzero(masks[0])
    assert (ys.min(), ys.max()) == (15, 19)
    assert 12 <= xs.min() <= 13 and 21 <= xs.max() <= 22
    assert pl.read_parquet(d / "index.parquet")["orig_w"].to_list() == [60, 60, 60]


def test_chexlocalize_refuses_missing_segmented_images(tmp_path: Path) -> None:
    chex, loc = tmp_path / "chexpert", tmp_path / "chexlocalize"
    fake_chexlocalize(chex, loc)
    (chex / "valid/patient64540/study1/view1_frontal.png").unlink()
    with pytest.raises(FileNotFoundError, match="segmented images missing"):
        chexpert_index(chex, "valid", loc)
