#!/usr/bin/env python3
"""Build the NIH index, patient-level splits, label-fraction subsets, and uint8 memmaps.

Usage:
    uv run scripts/prep_nih.py [--sizes 224 512] [--val-frac 0.1] [--seed 0] [--index-only]

Outputs under $CXR_DATA_ROOT/processed/nih/:
    index.parquet        one row per image, row order == memmap row order
    img_<S>.u8           uint8 [N, S, S] letterboxed images
    splits.json          split seed, fractions, digests, counts
"""

from __future__ import annotations

import argparse
import json

import polars as pl

from cxr_ml.data.nih import (
    assign_splits,
    build_index,
    check_no_patient_leakage,
    frame_digest,
    label_fraction_subsets,
)
from cxr_ml.data.prep import write_memmaps
from cxr_ml.paths import DataPaths

FRACTIONS = (0.01, 0.1)


def frac_col(f: float) -> str:
    return f"in_frac_{f:g}".replace(".", "p")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[224, 512])
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--index-only", action="store_true")
    args = ap.parse_args()

    paths = DataPaths.from_env()
    raw, out = paths.nih_raw, paths.processed("nih")
    out.mkdir(parents=True, exist_ok=True)

    print("building index")
    df = build_index(raw)
    df = assign_splits(df, val_frac=args.val_frac, seed=args.seed)
    patients = check_no_patient_leakage(df)
    subsets = label_fraction_subsets(df, FRACTIONS, seed=args.seed)
    df = df.with_columns(
        [pl.col("image_id").is_in(list(s)).alias(frac_col(f)) for f, s in subsets.items()]
    )
    df = df.with_row_index("row")
    df.write_parquet(out / "index.parquet")

    counts = {s: df.filter(pl.col("split") == s).height for s in ("train", "val", "test")}
    meta = {
        "source": (raw / "SOURCE.txt").read_text().strip().splitlines(),
        "split_seed": args.seed,
        "val_frac": args.val_frac,
        "images_per_split": counts,
        "patients_per_split": patients,
        "fractions": {
            f"{f:g}": {
                "images": len(s),
                "patients": df.filter(pl.col(frac_col(f)))["patient_id"].n_unique(),
            }
            for f, s in subsets.items()
        },
        "split_digest": frame_digest(df, ["image_id", "split"]),
        "index_digest": frame_digest(df, ["image_id", "findings", "patient_id"]),
    }
    (out / "splits.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))

    if args.index_only:
        return
    img_paths = [raw / "images" / i for i in df["image_id"]]
    missing = [p for p in img_paths if not p.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} images missing, e.g. {missing[:3]}")
    print(f"writing memmaps {args.sizes} for {len(img_paths)} images")
    write_memmaps(img_paths, out, args.sizes, workers=args.workers)
    (out / ".done").write_text("ok\n")


if __name__ == "__main__":
    main()
