"""Patient-level split guarantees.

Synthetic tests run everywhere (CI). The `data`-marked test checks the real processed index and is
skipped when it is not present.
"""

from __future__ import annotations

import polars as pl
import pytest

from cxr_common.labels import NIH_LABELS
from cxr_ml.data.nih import assign_splits, check_no_patient_leakage, label_fraction_subsets
from cxr_ml.paths import DataPaths


def synthetic_index(n_patients: int = 300, seed: int = 0) -> pl.DataFrame:
    import numpy as np

    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        official = "test" if p % 5 == 0 else "train_val"
        for k in range(int(rng.integers(1, 8))):
            rows.append(
                {"image_id": f"{p:08d}_{k:03d}.png", "patient_id": p, "official_split": official}
            )
    df = pl.DataFrame(rows)
    return df.with_columns([pl.lit(0, dtype=pl.UInt8).alias(n) for n in NIH_LABELS])


def test_splits_are_patient_disjoint() -> None:
    df = assign_splits(synthetic_index(), val_frac=0.2, seed=1)
    counts = check_no_patient_leakage(df)
    assert set(counts) == {"train", "val", "test"}


def test_official_test_untouched() -> None:
    df = assign_splits(synthetic_index(), seed=3)
    assert (df.filter(pl.col("official_split") == "test")["split"] == "test").all()
    assert (df.filter(pl.col("split") == "test")["official_split"] == "test").all()


def test_leak_detector_fires() -> None:
    df = assign_splits(synthetic_index(), seed=0)
    first_train = df.filter(pl.col("split") == "train")["image_id"][0]
    bad = df.with_columns(
        pl.when(pl.col("image_id") == first_train)
        .then(pl.lit("val"))
        .otherwise(pl.col("split"))
        .alias("split")
    )
    # Only leaks if that patient has >1 image; pick a patient that does.
    multi = df.group_by("patient_id").len().filter(pl.col("len") > 1)["patient_id"]
    pid = int(
        df.filter(pl.col("patient_id").is_in(multi.to_list()) & (pl.col("split") == "train"))[
            "patient_id"
        ][0]
    )
    img = df.filter(pl.col("patient_id") == pid)["image_id"][0]
    bad = df.with_columns(
        pl.when(pl.col("image_id") == img)
        .then(pl.lit("test"))
        .otherwise(pl.col("split"))
        .alias("split")
    )
    with pytest.raises(AssertionError, match="span multiple splits"):
        check_no_patient_leakage(bad)


def test_fraction_subsets_nested_patient_level_and_sized() -> None:
    df = assign_splits(synthetic_index(1000), seed=0)
    subs = label_fraction_subsets(df, [0.01, 0.1, 0.5], seed=0)
    assert subs[0.01] <= subs[0.1] <= subs[0.5]
    train = df.filter(pl.col("split") == "train")
    n = train.height
    for f, s in subs.items():
        assert f * n <= len(s) < f * n + 8  # at most one patient's worth over
        chosen = train.filter(pl.col("image_id").is_in(list(s)))
        # whole patients only
        pts = chosen["patient_id"].unique().to_list()
        assert train.filter(pl.col("patient_id").is_in(pts)).height == len(s)


def test_split_is_deterministic() -> None:
    a = assign_splits(synthetic_index(), seed=7)["split"].to_list()
    b = assign_splits(synthetic_index(), seed=7)["split"].to_list()
    assert a == b


@pytest.mark.data
def test_real_nih_index_has_no_leakage() -> None:
    p = DataPaths.from_env().index("nih")
    if not p.exists():
        pytest.skip("processed NIH index not present")
    df = pl.read_parquet(p)
    check_no_patient_leakage(df)
    assert df.filter(pl.col("split") == "test").height == 25596
    assert df.height == 112120
    # fraction subsets are inside train and nested
    f1, f10 = df.filter(pl.col("in_frac_0p01")), df.filter(pl.col("in_frac_0p1"))
    assert (f1["split"] == "train").all() and (f10["split"] == "train").all()
    assert set(f1["image_id"]) <= set(f10["image_id"])
