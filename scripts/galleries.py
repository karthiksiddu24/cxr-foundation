#!/usr/bin/env python3
"""Qualitative failure galleries with heatmaps (NIH test images only; redistributable).

    uv run scripts/galleries.py <run>

Writes docs/figures/galleries/<run>__<finding>_errors.png (highest-scored negatives = false
positives, lowest-scored positives = false negatives) and <run>__pneumothorax_drains.png.
Selection is by score rank only, never hand-picked.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from cxr_common.labels import DISPLAY_NAMES, NIH_LABELS
from cxr_ml.eval.evaluate import load_preds
from cxr_ml.eval.metrics import sigmoid
from cxr_ml.explain.localization import load_run_model
from cxr_ml.paths import DataPaths
from cxr_ml.viz import style

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures" / "galleries"
FINDINGS = ["Cardiomegaly", "Effusion", "Pneumothorax", "Mass"]
N = 4


def cams_for(model, imgs: np.ndarray, k: int) -> np.ndarray:  # type: ignore[no-untyped-def]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        x = torch.from_numpy(imgs).to(dev).unsqueeze(1).float() / 255
        _, cams, _ = model.to(dev).forward_with_cam(x)
        up = F.interpolate(
            cams[:, k : k + 1].float(), size=imgs.shape[-2:], mode="bilinear", align_corners=False
        )
    u = up[:, 0].cpu().numpy()
    lo = u.reshape(len(u), -1).min(1)[:, None, None]
    hi = u.reshape(len(u), -1).max(1)[:, None, None]
    return (u - lo) / np.maximum(hi - lo, 1e-6)


def panel(ax, img: np.ndarray, cam: np.ndarray, title: str) -> None:  # type: ignore[no-untyped-def]
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.imshow(np.ma.masked_less(cam, 0.25), cmap="inferno", alpha=0.45, vmin=0, vmax=1)
    ax.set_title(title, fontsize=7, loc="left")
    ax.set_axis_off()


def grid(
    rows: list[tuple[str, list[tuple[np.ndarray, np.ndarray, str]]]], path: Path, title: str
) -> None:
    style.apply()
    fig, axes = plt.subplots(len(rows), N, figsize=(2.2 * N, 2.4 * len(rows)), squeeze=False)
    for r, (label, items) in enumerate(rows):
        for c in range(N):
            ax = axes[r, c]
            if c < len(items):
                panel(ax, *items[c])
            else:
                ax.set_axis_off()
        axes[r, 0].text(
            -0.08,
            0.5,
            label,
            transform=axes[r, 0].transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=8,
        )
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    run = sys.argv[1]
    p = DataPaths.from_env()
    model, cfg = load_run_model(run)
    size = cfg["data"]["size"]
    idx = pl.read_parquet(p.index("nih"))
    mm = np.memmap(p.images("nih", size), dtype=np.uint8, mode="r", shape=(idx.height, size, size))
    y, logits, _, df = load_preds(p.runs / run / "preds_test.parquet")
    rows_np = df["row"].to_numpy()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in FINDINGS:
        k = NIH_LABELS.index(name)
        prob = sigmoid(logits[:, k])
        fp = np.argsort(-np.where(y[:, k] == 0, prob, -1))[:N]
        fn = np.argsort(np.where(y[:, k] == 1, prob, 2))[:N]
        sel = []
        for group in (fp, fn):
            imgs = np.stack([np.array(mm[rows_np[i]]) for i in group])
            cams = cams_for(model, imgs, k)
            sel.append(
                [
                    (imgs[j], cams[j], f"{df['image_id'][int(i)]}  p={prob[i]:.2f}")
                    for j, i in enumerate(group)
                ]
            )
        grid(
            [
                ("False positives\n(highest-scored negatives)", sel[0]),
                ("False negatives\n(lowest-scored positives)", sel[1]),
            ],
            OUT / f"{run}__{name.lower()}_errors.png",
            f"{DISPLAY_NAMES[name]} — {run} (heatmap: {DISPLAY_NAMES[name]})",
        )

    drains = pl.read_csv(
        p.root / "raw" / "neatx" / "NIH-CX14_TubeAnnotations_NonExperts_aggregated.csv"
    ).select(pl.col("Image Index").alias("image_id"), pl.col("Drain").alias("drain"))
    d = (
        df.with_row_index("i")
        .join(drains, on="image_id", how="inner")
        .filter(pl.col("Pneumothorax") == 1)
    )
    k = NIH_LABELS.index("Pneumothorax")
    prob = sigmoid(logits[:, k])
    with_d = d.filter(pl.col("drain") == 1)["i"].to_numpy()
    without = d.filter(pl.col("drain") == 0)["i"].to_numpy()
    groups = [with_d[np.argsort(-prob[with_d])][:N], without[np.argsort(prob[without])][:N]]
    rows = []
    for label, group in zip(
        ["With drain:\nhighest scores", "Without drain:\nlowest scores"], groups, strict=True
    ):
        imgs = np.stack([np.array(mm[rows_np[i]]) for i in group])
        cams = cams_for(model, imgs, k)
        rows.append(
            (
                label,
                [
                    (imgs[j], cams[j], f"{df['image_id'][int(i)]}  p={prob[i]:.2f}")
                    for j, i in enumerate(group)
                ],
            )
        )
    grid(rows, OUT / f"{run}__pneumothorax_drains.png", f"Pneumothorax and chest drains — {run}")
    print(f"galleries written to {OUT}")


if __name__ == "__main__":
    main()
