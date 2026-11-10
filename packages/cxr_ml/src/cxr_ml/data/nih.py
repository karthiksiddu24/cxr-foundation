"""NIH ChestX-ray14 index construction and patient-level splitting.

Split policy:
- The official `test_list.txt` is the test set, untouched.
- `train_val_list.txt` is split into train/val **by patient** (seeded).
- Label-fraction subsets (e.g. 1%, 10%) are sampled by patient from train and are nested.

The official lists are themselves patient-disjoint; `check_no_patient_leakage` verifies this
rather than assuming it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl

from cxr_common.labels import NIH_LABELS, parse_nih_finding_string

SPLITS = ("train", "val", "test")


def read_list(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def build_index(raw: Path) -> pl.DataFrame:
    """Parse Data_Entry_2017_v2020.csv into a typed index sorted by image_id.

    Row order of this frame defines the row order of the image memmaps.
    """
    df = pl.read_csv(raw / "Data_Entry_2017_v2020.csv", infer_schema_length=0)
    df = df.rename(
        {
            "Image Index": "image_id",
            "Finding Labels": "findings",
            "Follow-up #": "follow_up",
            "Patient ID": "patient_id",
            "Patient Age": "age",
            "Patient Gender": "sex",
            "View Position": "view",
            "OriginalImage[Width": "orig_w",
            "Height]": "orig_h",
            "OriginalImagePixelSpacing[x": "spacing_x",
            "y]": "spacing_y",
        }
    )
    df = df.with_columns(
        pl.col("follow_up").cast(pl.Int32),
        pl.col("patient_id").cast(pl.Int32),
        pl.col("age").cast(pl.Int32),
        pl.col("orig_w").cast(pl.Int32),
        pl.col("orig_h").cast(pl.Int32),
        pl.col("spacing_x").cast(pl.Float64),
        pl.col("spacing_y").cast(pl.Float64),
    )
    labels = np.array([parse_nih_finding_string(s) for s in df["findings"]], dtype=np.uint8)
    df = df.with_columns(
        [pl.Series(name, labels[:, i]) for i, name in enumerate(NIH_LABELS)]
    )
    test = set(read_list(raw / "test_list.txt"))
    trainval = set(read_list(raw / "train_val_list.txt"))
    if test & trainval:
        raise ValueError("official test and train_val lists overlap")
    ids = df["image_id"].to_list()
    unknown = set(ids) - test - trainval
    if unknown:
        raise ValueError(f"{len(unknown)} images in neither official list")
    df = df.with_columns(
        pl.Series(
            "official_split", ["test" if i in test else "train_val" for i in ids], dtype=pl.Utf8
        )
    )
    return df.sort("image_id")


def assign_splits(df: pl.DataFrame, val_frac: float = 0.1, seed: int = 0) -> pl.DataFrame:
    """Add a `split` column: train/val by patient from train_val; test from the official list."""
    tv_patients = np.array(
        sorted(df.filter(pl.col("official_split") == "train_val")["patient_id"].unique().to_list())
    )
    rng = np.random.default_rng(seed)
    perm = rng.permutation(tv_patients)
    n_val = round(len(perm) * val_frac)
    val_patients = set(perm[:n_val].tolist())
    split = [
        "test" if o == "test" else ("val" if p in val_patients else "train")
        for o, p in zip(df["official_split"], df["patient_id"], strict=True)
    ]
    return df.with_columns(pl.Series("split", split, dtype=pl.Utf8))


def label_fraction_subsets(
    df: pl.DataFrame, fractions: Sequence[float], seed: int = 0
) -> dict[float, set[str]]:
    """Nested patient-level subsets of the train split.

    Patients are shuffled once; each fraction takes the shortest prefix of that order whose image
    count reaches `fraction * n_train_images`. Prefixes of one ordering are nested by construction.
    """
    train = df.filter(pl.col("split") == "train")
    per_patient = train.group_by("patient_id").agg(pl.col("image_id")).sort("patient_id")
    rng = np.random.default_rng(seed)
    order = rng.permutation(per_patient.height)
    images_in_order = [per_patient["image_id"][int(i)].to_list() for i in order]
    counts = np.cumsum([len(x) for x in images_in_order])
    total = int(counts[-1])
    out: dict[float, set[str]] = {}
    for f in sorted(fractions):
        if not 0 < f <= 1:
            raise ValueError(f"fraction must be in (0, 1], got {f}")
        k = int(np.searchsorted(counts, f * total, side="left")) + 1
        out[f] = {img for group in images_in_order[:k] for img in group}
    return out


def check_no_patient_leakage(df: pl.DataFrame, split_col: str = "split") -> dict[str, int]:
    """Raise if any patient appears in more than one split. Returns patients per split."""
    per = df.group_by("patient_id").agg(pl.col(split_col).n_unique().alias("n"))
    leaked = per.filter(pl.col("n") > 1)
    if leaked.height:
        ex = leaked["patient_id"].head(5).to_list()
        raise AssertionError(f"{leaked.height} patients span multiple splits, e.g. {ex}")
    counts = df.group_by(split_col).agg(pl.col("patient_id").n_unique())
    return dict(zip(counts[split_col].to_list(), counts["patient_id"].to_list(), strict=True))


def frame_digest(df: pl.DataFrame, cols: Sequence[str]) -> str:
    """Stable short hash of selected columns (recorded with every checkpoint)."""
    h = hashlib.sha256()
    for row in df.select(cols).iter_rows():
        h.update("|".join(map(str, row)).encode())
        h.update(b"\n")
    return h.hexdigest()[:16]
