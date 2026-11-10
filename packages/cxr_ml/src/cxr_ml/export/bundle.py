"""Export a trained run to a self-contained serving bundle (ONNX + manifest + OOD stats).

    uv run python -m cxr_ml.export.bundle <run_name> [--version 1.0.0] [--out DIR]
    uv run python -m cxr_ml.export.bundle --placeholder      # untrained dev bundle

Steps: load best.pt -> wrap (logits, cams, features) -> ONNX export -> numerical parity check
against PyTorch -> fit Mahalanobis OOD stats on train features, threshold on val -> manifest with
calibration/thresholds/metrics copied from artifacts/eval/<run>/metrics.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from torch import nn

from cxr_common.bundle import (
    ExternalResult,
    Finding,
    Interval,
    Manifest,
    ModelCard,
    OODInfo,
    Provenance,
    ThresholdInfo,
)
from cxr_common.labels import DESCRIPTIONS, DISPLAY_NAMES, NIH_LABELS
from cxr_ml.data.augment import u8_to_float
from cxr_ml.models.classifier import CXRClassifier, ModelConfig, build_classifier
from cxr_ml.paths import DataPaths
from cxr_ml.train.common import file_sha256, git_info

OPSET = 18
PARITY_ATOL = 1e-3  # fp32 PyTorch vs ONNX Runtime CPU, logits


class ServingModule(nn.Module):
    def __init__(self, model: CXRClassifier) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model.forward_with_cam(image)


def export_onnx(model: CXRClassifier, size: int, path: Path) -> None:
    model = model.eval().float().cpu()
    wrapped = ServingModule(model)
    x = torch.rand(2, 1, size, size)
    torch.onnx.export(
        wrapped,
        (x,),
        str(path),
        input_names=["image"],
        output_names=["logits", "cams", "features"],
        dynamic_axes={
            "image": {0: "batch"},
            "logits": {0: "batch"},
            "cams": {0: "batch"},
            "features": {0: "batch"},
        },
        opset_version=OPSET,
        dynamo=False,
    )


def check_parity(model: CXRClassifier, onnx_path: Path, x: np.ndarray) -> dict[str, float]:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = [t.numpy() for t in ServingModule(model.eval().float().cpu())(torch.from_numpy(x))]
    got = sess.run(None, {"image": x})
    diffs = {
        n: float(np.abs(r - g).max())
        for n, r, g in zip(("logits", "cams", "features"), ref, got, strict=True)
    }
    if diffs["logits"] > PARITY_ATOL:
        raise AssertionError(f"ONNX parity failed: {diffs}")
    return diffs


@torch.no_grad()
def collect_features(model: CXRClassifier, rows: np.ndarray, size: int) -> np.ndarray:
    paths = DataPaths.from_env()
    n = pl.read_parquet(paths.index("nih"), columns=["row"]).height
    mm = np.memmap(paths.images("nih", size), dtype=np.uint8, mode="r", shape=(n, size, size))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = model.to(dev).eval()
    feats = []
    for i in range(0, len(rows), 256):
        x = u8_to_float(torch.from_numpy(np.array(mm[np.sort(rows[i : i + 256])])).to(dev))
        feats.append(m.forward_features(x).float().cpu().numpy())
    m.cpu()
    return np.concatenate(feats).astype(np.float64)


def fit_ood(
    model: CXRClassifier, size: int, n_fit: int = 20000, q: float = 0.995
) -> tuple[dict[str, np.ndarray], OODInfo]:
    from sklearn.covariance import LedoitWolf

    df = pl.read_parquet(DataPaths.from_env().index("nih"))
    rng = np.random.default_rng(0)
    tr = df.filter(pl.col("split") == "train")["row"].to_numpy().astype(np.int64)
    va = df.filter(pl.col("split") == "val")["row"].to_numpy().astype(np.int64)
    ftr = collect_features(model, rng.choice(tr, min(n_fit, len(tr)), replace=False), size)
    fva = collect_features(model, va, size)
    lw = LedoitWolf().fit(ftr)
    mean, prec = lw.location_, lw.precision_
    d = fva - mean
    scores = np.einsum("ij,jk,ik->i", d, prec, d)
    thr = float(np.quantile(scores, q))
    info = OODInfo(
        feature_dim=ftr.shape[1], threshold=thr, threshold_quantile=q, fit_images=len(ftr)
    )
    return {"mean": mean.astype(np.float32), "precision": prec.astype(np.float32)}, info


def _iv(d: dict[str, Any] | None) -> Interval | None:
    return Interval(**d) if d else None


CARD_TEXT = {
    "intended_use": (
        "Research and education: demonstrating self-supervised pretraining, evaluation and "
        "explainability for multi-label chest X-ray classification. Outputs are model scores, "
        "not diagnoses."
    ),
    "not_intended_for": [
        "Clinical diagnosis, triage, screening or any patient-care decision.",
        "Pediatric images, lateral views, non-chest images, or non-radiograph modalities.",
        "Populations or equipment that differ from the NIH Clinical Center 1992-2015 archive "
        "without separate validation.",
    ],
    "training_data": (
        "NIH ChestX-ray14 (112,120 frontal radiographs of 30,805 patients, NIH Clinical Center). "
        "Labels for 14 findings were text-mined from radiology reports and are noisy. Trained on "
        "the official train_val list with a patient-level validation split."
    ),
    "evaluation_data": (
        "NIH official test list (25,596 images, 2,797 patients), patient-level bootstrap CIs. "
        "External: CheXpert radiologist-labeled sets where available (see external results)."
    ),
    "limitations": [
        "Training labels come from NLP on reports, not image review; measured performance is "
        "agreement with those noisy labels.",
        "Single-institution training data; performance drops under domain shift (see external "
        "validation).",
        "Heatmaps show where evidence for a score is located in the model's features; they are "
        "coarse and are not segmentations.",
        "The plausibility check flags images that look unlike the training data, but it cannot "
        "catch every inappropriate input.",
        "Probabilities are calibrated on the NIH validation split; calibration may not transfer "
        "to other sites.",
    ],
}


def build_manifest(
    run: str,
    cfg: dict[str, Any],
    prov: dict[str, Any],
    onnx_path: Path,
    ckpt: Path | None,
    grid: tuple[int, int],
    ood: OODInfo | None,
    metrics: dict[str, Any] | None,
    version: str,
    placeholder: bool,
) -> Manifest:
    findings = []
    for n in NIH_LABELS:
        cal = metrics["calibration"][n] if metrics else {"temperature": 1.0}
        thr = (
            metrics["thresholds"][n]
            if metrics
            else {"threshold": 0.5, "method": "placeholder_fixed"}
        )
        findings.append(
            Finding(
                name=n,
                display_name=DISPLAY_NAMES[n],
                description=DESCRIPTIONS[n],
                temperature=cal["temperature"],
                threshold=ThresholdInfo(**thr),
                test_auroc=_iv(metrics["test_auroc"]["per_class"][n]) if metrics else None,
                test_auprc=_iv(metrics["test_auprc"]["per_class"][n]) if metrics else None,
                test_prevalence=metrics["test_prevalence"][n] if metrics else None,
            )
        )
    card_kwargs: dict[str, Any] = dict(CARD_TEXT)
    if placeholder:
        card_kwargs["limitations"] = [
            "UNTRAINED PLACEHOLDER: classification head is random. Scores are meaningless.",
            *CARD_TEXT["limitations"],
        ]
    external: list[ExternalResult] = []
    ext_dir = DataPaths.from_env().eval_artifacts / run
    ext_path = next(
        (
            p
            for p in (ext_dir / f"external_chexpert_{s}.json" for s in ("test", "valid"))
            if p.exists()
        ),
        None,
    )
    if metrics and ext_path is not None:
        e = json.loads(ext_path.read_text())
        skipped = [k for k, v in e["per_finding"].items() if v["auroc"] is None]
        notes = "Mean over exactly-mapped findings; Lung Lesion maps approximately to Mass|Nodule."
        if e.get("label_source") == "chexlocalize_masks":
            notes += " Labels derived from CheXlocalize radiologist segmentations."
        if skipped:
            notes += f" Not scored (unlabeled or too few positives): {', '.join(skipped)}."
        external.append(
            ExternalResult(
                dataset="CheXpert (radiologist-labeled)",
                split=e["split"],
                n_images=e["n_images"],
                mean_auroc=Interval(**e["mean_auroc_exact_mappings"]),
                per_finding={
                    k: Interval(**v["auroc"])
                    for k, v in e["per_finding"].items()
                    if v["auroc"] is not None
                },
                notes=notes,
            )
        )
    card = ModelCard(
        **card_kwargs,
        external=external,
        mean_test_auroc=_iv(metrics["test_auroc"]["mean"]) if metrics else None,
        mean_ece_raw=metrics["mean_ece_raw"] if metrics else None,
        mean_ece_calibrated=metrics["mean_ece_calibrated"] if metrics else None,
        eval_artifact=f"artifacts/eval/{run}/metrics.json" if metrics else None,
    )
    git = prov.get("git", {}) or git_info()
    return Manifest(
        model_id=run,
        version=version,
        placeholder=placeholder,
        arch=cfg["arch"],
        init=cfg["init"],
        input_size=cfg["img_size"],
        onnx_sha256=file_sha256(onnx_path),
        opset=OPSET,
        heatmap_grid=grid,
        findings=findings,
        ood=ood,
        card=card,
        provenance=Provenance(
            run=run,
            git_commit=git.get("commit"),
            git_dirty=git.get("dirty"),
            config_hash=prov.get("config_hash"),
            split_digest=(prov.get("data") or {}).get("split_digest"),
            checkpoint_sha256=file_sha256(ckpt) if ckpt else None,
            created_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        ),
    )


def export_run(run: str, out: Path, version: str) -> Path:
    paths = DataPaths.from_env()
    ckpt = paths.runs / run / "best.pt"
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    mcfg = blob["config"]["model"]
    cfg = ModelConfig(**{**mcfg, "init": "random", "ssl_checkpoint": None})
    model = build_classifier(cfg)
    model.load_state_dict(blob["model"])
    model.eval()
    metrics_path = paths.eval_artifacts / run / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit(f"run `python -m cxr_ml.eval.evaluate {run}` first")
    metrics = json.loads(metrics_path.read_text())
    return _write(
        model,
        cfg,
        run,
        out,
        version,
        blob,
        ckpt,
        metrics,
        placeholder=False,
        init_label=mcfg["init"],
    )


def export_placeholder(out: Path) -> Path:
    cfg = ModelConfig(arch="densenet121", init="imagenet")
    torch.manual_seed(0)
    model = build_classifier(cfg).eval()
    return _write(
        model,
        cfg,
        "placeholder-densenet121",
        out,
        "0.0.0-placeholder",
        {},
        None,
        None,
        placeholder=True,
        init_label="imagenet",
        fit_ood_stats=False,
    )


def _write(
    model: CXRClassifier,
    cfg: ModelConfig,
    run: str,
    out: Path,
    version: str,
    blob: dict[str, Any],
    ckpt: Path | None,
    metrics: dict[str, Any] | None,
    placeholder: bool,
    init_label: str,
    fit_ood_stats: bool = True,
) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    onnx_path = out / "model.onnx"
    export_onnx(model, cfg.img_size, onnx_path)
    x = np.random.default_rng(0).random((3, 1, cfg.img_size, cfg.img_size), dtype=np.float32)
    diffs = check_parity(model, onnx_path, x)
    print(f"ONNX parity max abs diff: {diffs}")
    with torch.no_grad():
        cams = model.forward_with_cam(torch.from_numpy(x[:1]))[1]
    grid = (int(cams.shape[-2]), int(cams.shape[-1]))
    ood = None
    if fit_ood_stats:
        stats, ood = fit_ood(model, cfg.img_size)
        np.savez(out / "ood.npz", mean=stats["mean"], precision=stats["precision"])
    cfgd = {"arch": cfg.arch, "init": init_label, "img_size": cfg.img_size}
    man = build_manifest(run, cfgd, blob, onnx_path, ckpt, grid, ood, metrics, version, placeholder)
    man.save(out)
    (out / "parity.json").write_text(json.dumps(diffs, indent=2))
    print(f"bundle written to {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?")
    ap.add_argument("--placeholder", action="store_true")
    ap.add_argument("--version", default="1.0.0")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    paths = DataPaths.from_env()
    if a.placeholder:
        export_placeholder(a.out or paths.bundles / "placeholder")
    elif a.run:
        export_run(a.run, a.out or paths.bundles / a.run, a.version)
    else:
        ap.error("give a run name or --placeholder")


if __name__ == "__main__":
    main()
