"""Full evaluation of a finished classification run from its saved predictions.

    uv run python -m cxr_ml.eval.evaluate <run_name> [--n-boot 2000]

Reads  $CXR_DATA_ROOT/runs/<run>/preds_{val,test}.parquet
Writes $CXR_DATA_ROOT/artifacts/eval/<run>/
    metrics.json        per-class + mean AUROC/AUPRC with patient-bootstrap 95% CIs (test),
                        calibration (ECE before/after per-class temperature scaling),
                        val-chosen thresholds and their test sensitivity/specificity,
                        AUROC stratified by view position (PA/AP)
    boot_auroc.npy      bootstrap samples [R, K] (for paired comparisons across runs)
    reliability.png     reliability diagrams
Every number in RESULTS.md and the model card is read from these files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from cxr_common.labels import DISPLAY_NAMES, NIH_LABELS
from cxr_ml.eval.metrics import (
    bootstrap_metric,
    ece_equal_mass,
    fit_temperature,
    patient_bootstrap_weights,
    reliability_curve,
    sens_spec,
    sigmoid,
    weighted_auroc,
    youden_threshold,
)
from cxr_ml.paths import DataPaths

BOOT_SEED = 12345  # shared by all runs => identical resamples => paired comparisons are valid


def load_preds(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, pl.DataFrame]:
    df = pl.read_parquet(path).sort("row")
    y = df.select(list(NIH_LABELS)).to_numpy().astype(np.int64)
    logits = df.select([f"logit_{n}" for n in NIH_LABELS]).to_numpy().astype(np.float64)
    return y, logits, df["patient_id"].to_numpy(), df


def summarize(res: Any, labels: tuple[str, ...]) -> dict[str, Any]:
    lo, hi = res.ci()
    mean_samples = np.nanmean(res.samples, axis=1)
    return {
        "per_class": {
            n: {"point": _f(res.point[k]), "ci_low": _f(lo[k]), "ci_high": _f(hi[k])}
            for k, n in enumerate(labels)
        },
        "mean": {
            "point": _f(np.nanmean(res.point)),
            "ci_low": _f(np.percentile(mean_samples, 2.5)),
            "ci_high": _f(np.percentile(mean_samples, 97.5)),
        },
    }


def _f(x: float) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), 5)


def evaluate(run: str, n_boot: int = 2000) -> Path:
    paths = DataPaths.from_env()
    rdir = paths.runs / run
    out = paths.eval_artifacts / run
    out.mkdir(parents=True, exist_ok=True)
    yv, lv, _, _ = load_preds(rdir / "preds_val.parquet")
    yt, lt, pt, dft = load_preds(rdir / "preds_test.parquet")
    k = len(NIH_LABELS)

    weights = patient_bootstrap_weights(pt, n_boot, BOOT_SEED)
    auroc = bootstrap_metric("auroc", yt, lt, pt, weights=weights)
    auprc = bootstrap_metric("auprc", yt, lt, pt, weights=weights)
    np.save(out / "boot_auroc.npy", auroc.samples.astype(np.float32))

    # calibration: per-class temperature fit on val, applied to test
    temps = [fit_temperature(lv[:, j], yv[:, j]) for j in range(k)]
    p_raw = sigmoid(lt)
    p_cal = sigmoid(lt / np.array(temps))
    pv_cal = sigmoid(lv / np.array(temps))
    calib = {}
    thresholds = {}
    for j, n in enumerate(NIH_LABELS):
        calib[n] = {
            "temperature": round(temps[j], 4),
            "ece_raw": _f(ece_equal_mass(p_raw[:, j], yt[:, j])),
            "ece_calibrated": _f(ece_equal_mass(p_cal[:, j], yt[:, j])),
        }
        thr, vsens, vspec = youden_threshold(pv_cal[:, j], yv[:, j])
        tsens, tspec = sens_spec(p_cal[:, j], yt[:, j], thr)
        thresholds[n] = {
            "threshold": round(thr, 5),
            "method": "youden_j_on_val",
            "val_sensitivity": _f(vsens),
            "val_specificity": _f(vspec),
            "test_sensitivity": _f(tsens),
            "test_specificity": _f(tspec),
        }

    # stratified by view position
    by_view: dict[str, Any] = {}
    for view in ("PA", "AP"):
        m = (dft["view"] == view).to_numpy()
        ones = np.ones((1, int(m.sum())))
        per = [weighted_auroc(lt[m, j], yt[m, j], ones)[0] for j in range(k)]
        by_view[view] = {
            "n_images": int(m.sum()),
            "mean_auroc": _f(np.nanmean(per)),
            "per_class": {n: _f(per[j]) for j, n in enumerate(NIH_LABELS)},
        }

    val_ones = np.ones((1, len(yv)))
    val_auroc = [weighted_auroc(lv[:, j], yv[:, j], val_ones)[0] for j in range(k)]
    metrics = {
        "run": run,
        "n_test_images": len(yt),
        "n_test_patients": len(np.unique(pt)),
        "n_bootstrap": n_boot,
        "bootstrap": "patient-clustered, percentile 95% CI",
        "bootstrap_seed": BOOT_SEED,
        "val_mean_auroc": _f(np.nanmean(val_auroc)),
        "test_auroc": summarize(auroc, NIH_LABELS),
        "test_auprc": summarize(auprc, NIH_LABELS),
        "test_prevalence": {n: _f(float(yt[:, j].mean())) for j, n in enumerate(NIH_LABELS)},
        "calibration": calib,
        "mean_ece_raw": _f(float(np.mean([c["ece_raw"] for c in calib.values()]))),
        "mean_ece_calibrated": _f(float(np.mean([c["ece_calibrated"] for c in calib.values()]))),
        "thresholds": thresholds,
        "by_view": by_view,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    _reliability_figure(p_raw, p_cal, yt, out / "reliability.png")
    print(
        f"{run}: test mean AUROC {metrics['test_auroc']['mean']['point']:.4f} "
        f"[{metrics['test_auroc']['mean']['ci_low']:.4f}, {metrics['test_auroc']['mean']['ci_high']:.4f}] "
        f"ECE {metrics['mean_ece_raw']:.4f} -> {metrics['mean_ece_calibrated']:.4f}"
    )
    return out


def _reliability_figure(p_raw: np.ndarray, p_cal: np.ndarray, y: np.ndarray, path: Path) -> None:
    import matplotlib.pyplot as plt

    from cxr_ml.viz import style

    style.apply()
    fig, axes = plt.subplots(2, 7, figsize=(16, 5), sharex=True, sharey=True)
    for j, (ax, n) in enumerate(zip(axes.flat, NIH_LABELS, strict=True)):
        ax.plot([0, 1], [0, 1], color=style.MUTED, lw=1, ls="--")
        for p, c, lab in ((p_raw, style.SERIES[1], "raw"), (p_cal, style.SERIES[0], "calibrated")):
            mp, fo, _ = reliability_curve(p[:, j], y[:, j])
            ax.plot(mp, fo, color=c, marker="o", ms=3, lw=1.5, label=lab)
        ax.set_title(DISPLAY_NAMES[n], fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0, 0].legend(loc="upper left")
    fig.supxlabel("Mean predicted probability (equal-mass bins)", fontsize=9)
    fig.supylabel("Observed frequency", fontsize=9)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    for r in a.runs:
        evaluate(r, a.n_boot)


if __name__ == "__main__":
    main()
