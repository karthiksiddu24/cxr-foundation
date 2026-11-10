"""Shortcut audit: does pneumothorax detection rely on chest drains?

    uv run python -m cxr_ml.eval.shortcut <run> [<run> ...]

Chest drains are a *treatment* for pneumothorax. A model that keys on the drain will score
treated (drain-present) pneumothoraces far higher than untreated (drain-absent) ones, which are
the clinically important cases. Using NEATX chest-drain annotations of NIH pneumothorax images
(Jiménez-Sánchez et al., CC BY-NC-SA, zenodo.org/records/14944064), we compare

    AUROC(drain+ positives vs. all negatives)   and   AUROC(drain- positives vs. all negatives)

on the same model and the same negatives, with paired patient-bootstrap CIs on the difference.
Caveat: drain status of *negatives* is unknown, and drain annotations are non-expert.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from cxr_ml.eval.evaluate import BOOT_SEED, load_preds
from cxr_ml.eval.metrics import patient_bootstrap_weights, weighted_auroc
from cxr_ml.paths import DataPaths

NEATX_FILE = "NIH-CX14_TubeAnnotations_NonExperts_aggregated.csv"
K_PTX = "Pneumothorax"


def run(run: str, n_boot: int = 2000) -> Path:
    paths = DataPaths.from_env()
    drains = pl.read_csv(paths.root / "raw" / "neatx" / NEATX_FILE).select(
        pl.col("Image Index").alias("image_id"), pl.col("Drain").alias("drain")
    )
    _, _, _, df = load_preds(paths.runs / run / "preds_test.parquet")
    df = df.join(drains, on="image_id", how="left")
    y = df[K_PTX].to_numpy().astype(np.int64)
    s = df[f"logit_{K_PTX}"].to_numpy().astype(np.float64)
    drain = df["drain"].to_numpy()
    pids = df["patient_id"].to_numpy()
    neg = y == 0
    pos_d = (y == 1) & (drain == 1)
    pos_nd = (y == 1) & (drain == 0)
    w = patient_bootstrap_weights(pids, n_boot, BOOT_SEED)
    ones = np.ones((1, len(y)))

    def auroc(mask: np.ndarray, weights: np.ndarray) -> np.ndarray:
        sub = mask | neg
        return weighted_auroc(s[sub], y[sub], weights[:, sub])

    point_d, point_nd = auroc(pos_d, ones)[0], auroc(pos_nd, ones)[0]
    boot_d, boot_nd = auroc(pos_d, w), auroc(pos_nd, w)
    diff = boot_d - boot_nd
    ok = np.isfinite(diff)
    res = {
        "run": run,
        "finding": K_PTX,
        "source": "NEATX non-expert chest-drain annotations (zenodo 14944064), CC BY-NC-SA",
        "n_negatives": int(neg.sum()),
        "n_pos_with_drain": int(pos_d.sum()),
        "n_pos_without_drain": int(pos_nd.sum()),
        "n_pos_unannotated": int(((y == 1) & ~np.isin(drain, [0, 1])).sum()),
        "auroc_with_drain": {
            "point": float(point_d),
            "ci_low": float(np.nanpercentile(boot_d, 2.5)),
            "ci_high": float(np.nanpercentile(boot_d, 97.5)),
        },
        "auroc_without_drain": {
            "point": float(point_nd),
            "ci_low": float(np.nanpercentile(boot_nd, 2.5)),
            "ci_high": float(np.nanpercentile(boot_nd, 97.5)),
        },
        "difference": {
            "point": float(point_d - point_nd),
            "ci_low": float(np.percentile(diff[ok], 2.5)),
            "ci_high": float(np.percentile(diff[ok], 97.5)),
        },
        "median_prob_rank_note": "AUROC computed against the same set of all test negatives",
    }
    out = paths.eval_artifacts / run / "shortcut_pneumothorax_drain.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    d, nd, df_ = res["auroc_with_drain"], res["auroc_without_drain"], res["difference"]
    print(
        f"{run}: PTX AUROC with drain {d['point']:.3f} [{d['ci_low']:.3f},{d['ci_high']:.3f}] "
        f"without {nd['point']:.3f} [{nd['ci_low']:.3f},{nd['ci_high']:.3f}] "
        f"diff {df_['point']:.3f} [{df_['ci_low']:.3f},{df_['ci_high']:.3f}]"
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    for r in a.runs:
        run(r)


if __name__ == "__main__":
    main()
