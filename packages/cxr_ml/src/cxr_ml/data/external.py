"""CheXpert and VinDr-CXR preparation (files must be obtained by the user under their licenses).

Expected layout (override roots with CXR_CHEXPERT_RAW / CXR_VINDR_RAW):

CheXpert ($CXR_CHEXPERT_RAW), radiologist-labeled sets only:
    valid.csv                + valid/patient*/study*/view*_frontal.jpg
    test_labels.csv (or test.csv) + test/patient*/study*/view*_frontal.jpg
  The 'Path' column may carry a 'CheXpert-v1.0/' or 'CheXpert-v1.0-small/' prefix; it is stripped.
VinDr-CXR ($CXR_VINDR_RAW), test split:
    annotations/annotations_test.csv   (image_id, class_name, x_min, y_min, x_max, y_max)
    test/<image_id>.dicom
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import polars as pl
import pydicom

from cxr_common.dicom import dicom_to_uint8
from cxr_common.labels import CHEXPERT_MAP
from cxr_common.preprocess import letterbox, to_gray_uint8

CHEXPERT_LABELS = [m.target for m in CHEXPERT_MAP]
_PREFIX = re.compile(r"^CheXpert-v1\.0(-small)?/")


def _first_existing(root: Path, names: list[str]) -> Path | None:
    for n in names:
        if (root / n).exists():
            return root / n
    return None


def chexpert_index(root: Path, split: str) -> pl.DataFrame:
    csv = _first_existing(
        root,
        ["valid.csv"] if split == "valid" else ["test_labels.csv", "test.csv", "groundtruth.csv"],
    )
    if csv is None:
        raise FileNotFoundError(f"CheXpert {split} labels not found under {root}")
    df = pl.read_csv(csv, infer_schema_length=0)
    path_col = next(c for c in df.columns if c.lower() in ("path", "image", "study"))
    df = df.with_columns(pl.col(path_col).str.replace(_PREFIX.pattern, "").alias("path"))
    if not df["path"][0].startswith(split):  # some test CSVs omit the split folder
        df = df.with_columns((pl.lit(f"{split}/") + pl.col("path")).alias("path"))
    frontal = (
        pl.col("Frontal/Lateral") == "Frontal"
        if "Frontal/Lateral" in df.columns
        else pl.col("path").str.contains("frontal")
    )
    df = df.filter(frontal)
    missing = [c for c in CHEXPERT_LABELS if c not in df.columns]
    if missing:
        raise ValueError(f"CheXpert {split} csv lacks columns {missing}")
    out = df.select(
        pl.col("path"),
        pl.col("path").str.extract(r"patient(\d+)", 1).cast(pl.Int64).alias("patient_id"),
        # Expert sets are binary; any uncertain (-1) value becomes null and is excluded per label.
        *[
            pl.col(c).cast(pl.Float64, strict=False).replace(-1.0, None).alias(c)
            for c in CHEXPERT_LABELS
        ],
    )
    return out.with_row_index("row")


def write_images(
    paths: list[Path], out_dir: Path, sizes: list[int], dicom: bool = False
) -> list[dict[str, float]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(paths)
    mms = {
        s: np.memmap(out_dir / f"img_{s}.u8", dtype=np.uint8, mode="w+", shape=(n, s, s))
        for s in sizes
    }
    geo: list[dict[str, float]] = []
    for i, p in enumerate(paths):
        if dicom:
            img, _ = dicom_to_uint8(pydicom.dcmread(str(p)))
        else:
            raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise RuntimeError(f"cannot read {p}")
            img = to_gray_uint8(raw)
        ref = max(sizes)
        _, lb = letterbox(img, ref)
        geo.append(
            {
                "orig_h": lb.orig_h,
                "orig_w": lb.orig_w,
                "scale": lb.scale,
                "pad_x": lb.pad_x,
                "pad_y": lb.pad_y,
                "letterbox_size": ref,
            }
        )
        for s, mm in mms.items():
            mm[i] = letterbox(img, s)[0]
    for mm in mms.values():
        mm.flush()
    return geo


def prep_chexpert(root: Path, out_root: Path, sizes: list[int]) -> None:
    for split in ("valid", "test"):
        try:
            idx = chexpert_index(root, split)
        except FileNotFoundError as e:
            print(f"skip: {e}")
            continue
        out = out_root / f"chexpert_{split}"
        write_images([root / p for p in idx["path"]], out, sizes)
        idx.write_parquet(out / "index.parquet")
        print(
            f"CheXpert {split}: {idx.height} frontal images, {idx['patient_id'].n_unique()} patients"
        )


def prep_vindr(root: Path, out_root: Path, sizes: list[int]) -> None:
    ann_path = _first_existing(root, ["annotations/annotations_test.csv", "annotations_test.csv"])
    if ann_path is None:
        print(f"skip: VinDr annotations_test.csv not found under {root}")
        return
    ann = pl.read_csv(ann_path).filter(pl.col("class_name") != "No finding")
    ids = sorted(ann["image_id"].unique().to_list())
    files = [_first_existing(root, [f"test/{i}.dicom", f"test/{i}.dcm"]) for i in ids]
    missing = [i for i, f in zip(ids, files, strict=True) if f is None]
    if missing:
        raise FileNotFoundError(f"{len(missing)} VinDr test DICOMs missing, e.g. {missing[:3]}")
    out = out_root / "vindr"
    geo = write_images([f for f in files if f is not None], out, sizes, dicom=True)
    idx = pl.DataFrame(
        {"image_id": ids, **{k: [g[k] for g in geo] for k in geo[0]}}
    ).with_row_index("row")
    idx.write_parquet(out / "index.parquet")
    g = idx.select("image_id", "scale", "pad_x", "pad_y")
    boxes = ann.join(g, on="image_id").with_columns(
        (pl.col("x_min") * pl.col("scale") + pl.col("pad_x")).alias("mx0"),
        (pl.col("y_min") * pl.col("scale") + pl.col("pad_y")).alias("my0"),
        (pl.col("x_max") * pl.col("scale") + pl.col("pad_x")).alias("mx1"),
        (pl.col("y_max") * pl.col("scale") + pl.col("pad_y")).alias("my1"),
    )
    boxes.write_parquet(out / "boxes.parquet")
    print(f"VinDr test: {len(ids)} images with boxes, {boxes.height} boxes")
