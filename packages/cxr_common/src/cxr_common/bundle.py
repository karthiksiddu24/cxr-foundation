"""Model bundle: the versioned contract between the ML pipeline and the inference server.

A bundle directory contains:
    manifest.json   this schema (validated on load)
    model.onnx      inputs: image float32 [B,1,S,S] in [0,1]
                    outputs: logits [B,K], cams [B,K,h,w] (logit units), features [B,C]
    ood.npz         mean [C], precision [C,C] for the Mahalanobis plausibility check (optional)

Everything numeric in `card.metrics` is copied by the exporter from evaluation artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Interval(_Strict):
    point: float | None
    ci_low: float | None = None
    ci_high: float | None = None


class ThresholdInfo(_Strict):
    threshold: float = Field(ge=0.0, le=1.0)
    method: str
    val_sensitivity: float | None = None
    val_specificity: float | None = None
    test_sensitivity: float | None = None
    test_specificity: float | None = None


class Finding(_Strict):
    name: str
    display_name: str
    description: str
    temperature: float = Field(gt=0.0)
    threshold: ThresholdInfo
    test_auroc: Interval | None = None
    test_auprc: Interval | None = None
    test_prevalence: float | None = None


class OODInfo(_Strict):
    file: str = "ood.npz"
    method: Literal["mahalanobis"] = "mahalanobis"
    feature_dim: int
    threshold: float  # squared Mahalanobis distance above which input is flagged as atypical
    threshold_quantile: float  # quantile of in-distribution val scores used to set it
    fit_images: int


class ExternalResult(_Strict):
    dataset: str
    split: str
    n_images: int
    mean_auroc: Interval
    per_finding: dict[str, Interval]
    notes: str = ""


class ModelCard(_Strict):
    intended_use: str
    not_intended_for: list[str]
    training_data: str
    evaluation_data: str
    limitations: list[str]
    mean_test_auroc: Interval | None = None
    mean_ece_raw: float | None = None
    mean_ece_calibrated: float | None = None
    external: list[ExternalResult] = Field(default_factory=list)
    eval_artifact: str | None = None  # path of the metrics.json these numbers came from


class Provenance(_Strict):
    run: str
    git_commit: str | None
    git_dirty: bool | None
    config_hash: str | None
    split_digest: str | None
    checkpoint_sha256: str | None
    created_at: str


class Manifest(_Strict):
    schema_version: int = SCHEMA_VERSION
    model_id: str
    version: str
    placeholder: bool = False  # True => untrained/dev model; UI must say so loudly
    arch: str
    init: str
    input_size: int
    onnx_file: str = "model.onnx"
    onnx_sha256: str
    opset: int
    heatmap: Literal["class_activation_map"] = "class_activation_map"
    heatmap_grid: tuple[int, int]
    findings: list[Finding]
    ood: OODInfo | None = None
    card: ModelCard
    provenance: Provenance

    @classmethod
    def load(cls, bundle_dir: Path) -> Manifest:
        m = cls.model_validate_json((bundle_dir / "manifest.json").read_text())
        if m.schema_version != SCHEMA_VERSION:
            raise ValueError(f"bundle schema {m.schema_version} != supported {SCHEMA_VERSION}")
        return m

    def save(self, bundle_dir: Path) -> None:
        bundle_dir.mkdir(parents=True, exist_ok=True)
        (bundle_dir / "manifest.json").write_text(
            json.dumps(self.model_dump(mode="json"), indent=2) + "\n"
        )
