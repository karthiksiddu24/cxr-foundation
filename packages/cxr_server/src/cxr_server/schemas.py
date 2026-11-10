"""Public API response models (these define the OpenAPI schema the frontend types are generated from)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from cxr_common.bundle import Interval, ModelCard


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QualityWarning(_M):
    code: str = Field(examples=["color_image"])
    severity: Literal["info", "caution", "high"]
    message: str


class Quality(_M):
    plausible: bool = Field(
        description="False when any high-severity warning fired. Scores must then be treated as "
        "unreliable; the UI hides them behind an explicit 'show anyway'."
    )
    warnings: list[QualityWarning]
    ood_score: float | None = Field(
        None, description="Squared Mahalanobis distance of the image embedding to training data"
    )
    ood_threshold: float | None = None


class Heatmap(_M):
    grid: list[list[float]] = Field(
        description="Class activation map normalized to [0,1] per finding, rows top-to-bottom, in "
        "model-input space (see HeatmapGeometry)."
    )
    logit_min: float
    logit_max: float


class FindingResult(_M):
    name: str
    display_name: str
    description: str
    probability: float = Field(ge=0, le=1, description="Calibrated model score (not a diagnosis)")
    threshold: float = Field(ge=0, le=1)
    flagged: bool = Field(description="probability >= threshold chosen on validation data")
    test_auroc: Interval | None = None
    heatmap: Heatmap


class HeatmapGeometry(_M):
    """Maps the square model-input space onto the original image.

    original_x = (model_x - pad_x) / scale, likewise for y. The heatmap grid spans the full
    square [0, input_size) in both axes.
    """

    input_size: int
    grid_h: int
    grid_w: int
    scale: float
    pad_x: int
    pad_y: int


class ImageInfo(_M):
    format: Literal["png", "jpeg", "dicom"]
    width: int
    height: int
    display_png_base64: str = Field(
        description="Grayscale PNG used for display (after DICOM windowing)"
    )
    display_width: int
    display_height: int


class ModelRef(_M):
    model_id: str
    version: str
    placeholder: bool


class Timing(_M):
    decode_ms: float
    inference_ms: float
    total_ms: float


class PredictResponse(_M):
    request_id: str
    model: ModelRef
    image: ImageInfo
    geometry: HeatmapGeometry
    quality: Quality
    findings: list[FindingResult] = Field(description="Sorted by probability, highest first")
    timing: Timing
    disclaimer: str


class FindingInfo(_M):
    name: str
    display_name: str
    description: str
    threshold: float
    threshold_method: str
    temperature: float
    test_auroc: Interval | None
    test_auprc: Interval | None
    test_prevalence: float | None
    test_sensitivity: float | None
    test_specificity: float | None


class ModelInfo(_M):
    model_id: str
    version: str
    placeholder: bool
    arch: str
    init: str
    input_size: int
    heatmap_grid: tuple[int, int]
    findings: list[FindingInfo]
    card: ModelCard
    git_commit: str | None
    created_at: str
    execution_provider: str
    disclaimer: str


class Health(_M):
    status: Literal["ok"]


class Ready(_M):
    ready: bool
    model_id: str | None = None
    execution_provider: str | None = None


class Problem(_M):
    """RFC 7807 problem details."""

    type: str = "about:blank"
    title: str
    status: int
    detail: str
    code: str
    request_id: str | None = None
