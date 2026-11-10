#!/usr/bin/env python3
"""Pick a few NIH *test-split* images as one-click demo samples for the web app.

Only NIH ChestX-ray14 images are used (the NIH Clinical Center permits redistribution with
citation). CheXpert/VinDr images are never shipped. Selection is deterministic.

Writes web/public/samples/<id>.png (original 1024px PNG, re-encoded) and samples.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import polars as pl

from cxr_common.labels import DISPLAY_NAMES, NIH_LABELS
from cxr_ml.paths import DataPaths

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "web" / "public" / "samples"
WANT = [("Cardiomegaly", "Enlarged heart"), ("Effusion", "Pleural effusion"),
        ("Pneumothorax", "Pneumothorax"), (None, "No finding in report")]


def main() -> None:
    paths = DataPaths.from_env()
    df = pl.read_parquet(paths.index("nih")).filter((pl.col("split") == "test") & (pl.col("view") == "PA"))
    n_pos = df.select(list(NIH_LABELS)).sum_horizontal()
    df = df.with_columns(n_pos.alias("n_pos"))
    OUT.mkdir(parents=True, exist_ok=True)
    items, used_patients = [], set()
    for label, title in WANT:
        cand = df.filter((pl.col("n_pos") == 0) if label is None else ((pl.col(label) == 1) & (pl.col("n_pos") == 1)))
        cand = cand.filter(~pl.col("patient_id").is_in(list(used_patients))).sort("image_id")
        row = cand.row(len(cand) // 2, named=True)  # deterministic, not cherry-picked by score
        used_patients.add(row["patient_id"])
        img = cv2.imread(str(paths.nih_raw / "images" / row["image_id"]), cv2.IMREAD_GRAYSCALE)
        sid = row["image_id"].removesuffix(".png")
        cv2.imwrite(str(OUT / f"{sid}.png"), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        items.append({
            "id": sid,
            "file": f"/samples/{sid}.png",
            "title": title,
            "dataset_labels": [DISPLAY_NAMES[n] for n in NIH_LABELS if row[n] == 1] or ["No finding"],
            "view": row["view"],
            "age": row["age"],
            "sex": row["sex"],
        })
    meta = {
        "source": "NIH ChestX-ray14 official test split (not used for training or model selection).",
        "citation": "Wang X. et al. ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks. CVPR 2017.",
        "label_note": "Dataset labels were text-mined from radiology reports and can be wrong.",
        "samples": items,
    }
    (OUT / "samples.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(items, indent=1))


if __name__ == "__main__":
    main()
