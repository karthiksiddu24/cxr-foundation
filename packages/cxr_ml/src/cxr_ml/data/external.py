"""CheXpert and VinDr-CXR preparation (files must be obtained by the user under their licenses).

Expected layout (override roots with CXR_CHEXPERT_RAW / CXR_VINDR_RAW):

CheXpert ($CXR_CHEXPERT_RAW), radiologist-labeled sets only:
    valid.csv                + valid/patient*/study*/view*_frontal.jpg
    test_labels.csv (or test.csv) + test/patient*/study*/view*_frontal.jpg
  The 'Path' column may carry a 'CheXpert-v1.0/' or 'CheXpert-v1.0-small/' prefix; it is stripped.
  Alternative when no label CSV is present (the CheXpert Plus release on Redivis ships the
  validation images but not the radiologist labels):
    $CXR_CHEXPERT_RAW/valid/patient*/study*/view*_frontal.png          (CheXpert Plus "PNG_valid")
    $CXR_CHEXLOCALIZE_RAW/gt_segmentations_val.json                    (CheXlocalize)
  CheXlocalize radiologists segmented every finding that is positive in the CheXpert ground truth,
  so a finding is labeled positive iff its ground-truth mask is non-empty, and images absent from
  the file are negative for all CheXlocalize findings. Pneumonia is not part of CheXlocalize and
  is left unlabeled. Masks are also exported for heatmap localization.
VinDr-CXR ($CXR_VINDR_RAW), test split:
    annotations/annotations_test.csv   (image_id, class_name, x_min, y_min, x_max, y_max)
    test/<image_id>.dicom
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np
import polars as pl
import pydicom

from cxr_common.dicom import dicom_to_uint8
from cxr_common.labels import CHEXPERT_MAP
from cxr_common.preprocess import letterbox, to_gray_uint8
from cxr_ml.data.rle import rle_decode, rle_is_empty

CHEXPERT_LABELS = [m.target for m in CHEXPERT_MAP]
_PREFIX = re.compile(r"^CheXpert-v1\.0(-small)?/")
_CHEXLOCALIZE_SPLIT = {"valid": "val", "test": "test"}
# CheXlocalize finding names coincide with CheXpert's; Pneumonia was not segmented.
CHEXLOCALIZE_FINDINGS = [n for n in CHEXPERT_LABELS if n != "Pneumonia"]
_IMG_SUFFIXES = (".png", ".jpg", ".jpeg")


def _first_existing(root: Path, names: list[str]) -> Path | None:
    for n in names:
        if (root / n).exists():
            return root / n
    return None


def chexpert_index(root: Path, split: str, chexlocalize_root: Path | None = None) -> pl.DataFrame:
    csv = _first_existing(
        root,
        ["valid.csv"] if split == "valid" else ["test_labels.csv", "test.csv", "groundtruth.csv"],
    )
    if csv is None:
        if chexlocalize_root is not None:
            seg = chexlocalize_root / f"gt_segmentations_{_CHEXLOCALIZE_SPLIT[split]}.json"
            if seg.exists() and (root / split).is_dir():
                return chexlocalize_index(root, split, seg)
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
    return out.with_columns(pl.lit("chexpert_csv").alias("label_source")).with_row_index("row")


def _image_key(rel: str) -> str:
    """'valid/patient64541/study1/view1_frontal.png' -> 'patient64541_study1_view1_frontal'."""
    p = Path(rel)
    return f"{p.parts[-3]}_{p.parts[-2]}_{p.stem}"


def load_chexlocalize(seg_path: Path) -> dict[str, dict[str, dict[str, object]]]:
    data: dict[str, dict[str, dict[str, object]]] = json.loads(seg_path.read_text())
    return data


def chexlocalize_index(root: Path, split: str, seg_path: Path) -> pl.DataFrame:
    """Frontal images under root/split labeled from CheXlocalize ground-truth masks."""
    rels = sorted(
        str(p.relative_to(root))
        for p in (root / split).glob("patient*/study*/view*_frontal.*")
        if p.suffix.lower() in _IMG_SUFFIXES
    )
    if not rels:
        raise FileNotFoundError(f"no frontal images under {root / split}")
    seg = load_chexlocalize(seg_path)
    keys = [_image_key(r) for r in rels]
    missing = sorted(k for k in seg if "frontal" in k and k not in set(keys))
    if missing:
        raise FileNotFoundError(f"{len(missing)} segmented images missing, e.g. {missing[:3]}")
    cols: dict[str, list[float | None]] = {}
    for name in CHEXPERT_LABELS:
        if name not in CHEXLOCALIZE_FINDINGS:
            cols[name] = [None] * len(keys)
            continue
        cols[name] = [
            float(k in seg and name in seg[k] and not rle_is_empty(seg[k][name])) for k in keys
        ]
    df = pl.DataFrame(
        {
            "path": rels,
            "patient_id": [int(re.search(r"patient(\d+)", r).group(1)) for r in rels],
            **{k: pl.Series(v, dtype=pl.Float64) for k, v in cols.items()},
            "label_source": ["chexlocalize_masks"] * len(rels),
        }
    )
    return df.with_row_index("row")


def write_masks(
    idx: pl.DataFrame, geo: list[dict[str, float]], seg_path: Path, out: Path, sizes: list[int]
) -> int:
    """Letterbox each non-empty ground-truth mask with the image's own geometry."""
    seg = load_chexlocalize(seg_path)
    items: list[tuple[int, str, np.ndarray]] = []
    for row, rel in zip(idx["row"].to_list(), idx["path"].to_list(), strict=True):
        ann = seg.get(_image_key(rel), {})
        g = geo[row]
        for name in CHEXLOCALIZE_FINDINGS:
            if name in ann and not rle_is_empty(ann[name]):
                m = rle_decode(ann[name])
                if m.shape != (int(g["orig_h"]), int(g["orig_w"])):
                    raise ValueError(f"{rel} {name}: mask {m.shape} vs image {g}")
                items.append((row, name, m))
    for s in sizes:
        mm = np.memmap(out / f"masks_{s}.u8", dtype=np.uint8, mode="w+", shape=(len(items), s, s))
        for i, (_, _, m) in enumerate(items):
            mm[i] = letterbox(m.astype(np.uint8) * 255, s)[0] >= 128
        mm.flush()
    pl.DataFrame(
        {
            "mask": list(range(len(items))),
            "row": [r for r, _, _ in items],
            "finding": [n for _, n, _ in items],
        }
    ).write_parquet(out / "masks.parquet")
    return len(items)


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


def prep_chexpert(
    root: Path, out_root: Path, sizes: list[int], chexlocalize_root: Path | None = None
) -> None:
    for split in ("valid", "test"):
        try:
            idx = chexpert_index(root, split, chexlocalize_root)
        except FileNotFoundError as e:
            print(f"skip: {e}")
            continue
        out = out_root / f"chexpert_{split}"
        geo = write_images([root / p for p in idx["path"]], out, sizes)
        idx = idx.with_columns(
            *[
                pl.Series(k, [g[k] for g in geo])
                for k in ("orig_h", "orig_w", "scale", "pad_x", "pad_y")
            ]
        )
        idx.write_parquet(out / "index.parquet")
        msg = f"CheXpert {split}: {idx.height} frontal images, {idx['patient_id'].n_unique()} patients"
        if chexlocalize_root is not None:
            seg = chexlocalize_root / f"gt_segmentations_{_CHEXLOCALIZE_SPLIT[split]}.json"
            if seg.exists():
                msg += f", {write_masks(idx, geo, seg, out, sizes)} ground-truth masks"
        print(f"{msg} (labels: {idx['label_source'][0]})")


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
