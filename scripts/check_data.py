#!/usr/bin/env python3
"""Report which datasets are present and exactly where missing files are expected."""

from __future__ import annotations

from cxr_ml.paths import DataPaths

p = DataPaths.from_env()
checks = [
    (
        "NIH ChestX-ray14 (auto-downloaded)",
        p.nih_raw,
        [
            "Data_Entry_2017_v2020.csv",
            "train_val_list.txt",
            "test_list.txt",
            "BBox_List_2017.csv",
            "images/",
        ],
    ),
    ("NIH processed", p.processed("nih"), ["index.parquet", "img_224.u8", "img_512.u8", ".done"]),
    (
        "NEATX chest-drain labels (auto-downloaded)",
        p.root / "raw" / "neatx",
        ["NIH-CX14_TubeAnnotations_NonExperts_aggregated.csv"],
    ),
    (
        "CheXpert expert sets (manual, Stanford AIMI license)",
        p.chexpert_raw,
        ["valid.csv", "valid/", "test_labels.csv", "test/"],
    ),
    (
        "VinDr-CXR test (manual, PhysioNet credentialed)",
        p.vindr_raw,
        ["annotations/annotations_test.csv", "test/"],
    ),
]
for title, root, items in checks:
    present = [(i, (root / i).exists()) for i in items]
    ok = all(x for _, x in present)
    print(f"[{'ok' if ok else '--'}] {title}\n     {root}")
    for i, x in present:
        if not x:
            print(f"       missing: {i}")
