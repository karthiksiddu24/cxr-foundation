"""External validation of an NIH-trained run on CheXpert's radiologist-labeled sets.

    uv run python -m cxr_ml.eval.external <run> [--split valid|test]

Each CheXpert label is scored by the max of its mapped NIH logits (cxr_common.labels.CHEXPERT_MAP).
Reports AUROC with patient-bootstrap CIs next to the same run's NIH-test AUROC for that finding,
so the domain-shift drop is explicit. Approximate mappings (Lung Lesion = Mass|Nodule) are
flagged and excluded from the shared-finding mean. Findings that are unlabeled in the prepared
split (Pneumonia when labels come from CheXlocalize masks) or have fewer than MIN_POS positives
are reported with their counts but no AUROC.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

from cxr_common.labels import CHEXPERT_MAP, NIH_LABELS
from cxr_ml.eval.evaluate import BOOT_SEED
from cxr_ml.eval.metrics import patient_bootstrap_weights, weighted_auroc
from cxr_ml.explain.localization import load_run_model
from cxr_ml.paths import DataPaths

MIN_POS = 5


@torch.no_grad()
def predict_memmap(model: torch.nn.Module, path: Path, n: int, size: int) -> np.ndarray:
    mm = np.memmap(path, dtype=np.uint8, mode="r", shape=(n, size, size))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = model.to(dev).eval()
    out = []
    for i in range(0, n, 128):
        x = torch.from_numpy(np.array(mm[i : i + 128])).to(dev).unsqueeze(1).float() / 255
        out.append(m(x).float().cpu().numpy())
    return np.concatenate(out)


def evaluate_chexpert(run: str, split: str, n_boot: int = 2000) -> Path:
    paths = DataPaths.from_env()
    d = paths.processed(f"chexpert_{split}")
    idx = pl.read_parquet(d / "index.parquet")
    model, cfg = load_run_model(run)
    size = cfg["data"]["size"]
    logits = predict_memmap(model, d / f"img_{size}.u8", idx.height, size)
    pids = idx["patient_id"].to_numpy()
    weights = patient_bootstrap_weights(pids, n_boot, BOOT_SEED)
    nih_metrics_path = paths.eval_artifacts / run / "metrics.json"
    nih = (
        json.loads(nih_metrics_path.read_text())["test_auroc"]["per_class"]
        if nih_metrics_path.exists()
        else {}
    )
    per: dict[str, Any] = {}
    exact_points, exact_samples = [], []
    for m in CHEXPERT_MAP:
        y_all = idx[m.target].to_numpy().astype(np.float64)
        keep = ~np.isnan(y_all)
        y = y_all[keep].astype(np.int64)
        if int(y.sum()) < MIN_POS:
            per[m.target] = {
                "n_pos": int(y.sum()),
                "n": len(y),
                "approximate": m.approximate,
                "nih_outputs": list(m.nih),
                "auroc": None,
                "skipped": "unlabeled in this split"
                if len(y) == 0
                else f"fewer than {MIN_POS} positives",
            }
            continue
        s = logits[keep][:, [NIH_LABELS.index(n) for n in m.nih]].max(axis=1).astype(np.float64)
        pt = weighted_auroc(s, y, np.ones((1, len(y))))[0]
        bs = weighted_auroc(s, y, weights[:, keep])
        entry = {
            "n_pos": int(y.sum()),
            "n": len(y),
            "approximate": m.approximate,
            "nih_outputs": list(m.nih),
            "auroc": {
                "point": float(pt),
                "ci_low": float(np.nanpercentile(bs, 2.5)),
                "ci_high": float(np.nanpercentile(bs, 97.5)),
            },
        }
        if len(m.nih) == 1 and m.nih[0] in nih:
            entry["nih_test_auroc"] = nih[m.nih[0]]
            entry["drop"] = (nih[m.nih[0]]["point"] or 0) - float(pt)
        per[m.target] = entry
        if not m.approximate and np.isfinite(pt):
            exact_points.append(pt)
            exact_samples.append(bs)
    mean_s = np.nanmean(np.stack(exact_samples, 1), axis=1)
    res = {
        "run": run,
        "dataset": "CheXpert",
        "split": split,
        "label_source": idx["label_source"][0] if "label_source" in idx.columns else "chexpert_csv",
        "n_images": idx.height,
        "n_patients": len(np.unique(pids)),
        "bootstrap": f"patient-clustered, {n_boot}",
        "mean_auroc_exact_mappings": {
            "point": float(np.mean(exact_points)),
            "ci_low": float(np.percentile(mean_s, 2.5)),
            "ci_high": float(np.percentile(mean_s, 97.5)),
        },
        "per_finding": per,
    }
    out = paths.eval_artifacts / run / f"external_chexpert_{split}.json"
    out.write_text(json.dumps(res, indent=2))
    print(
        f"{run} CheXpert-{split}: mean AUROC (exact mappings) {res['mean_auroc_exact_mappings']['point']:.3f}"
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--split", default="valid", choices=["valid", "test"])
    a = ap.parse_args()
    for r in a.runs:
        evaluate_chexpert(r, a.split)


if __name__ == "__main__":
    main()
