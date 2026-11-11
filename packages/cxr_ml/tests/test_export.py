"""ONNX export + PyTorch/ONNX Runtime parity for both model families served by the app."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cxr_ml.export.bundle import PARITY_ATOL, check_parity, export_onnx
from cxr_ml.models.classifier import ModelConfig, build_classifier


@pytest.mark.parametrize(
    ("arch", "size", "grid"),
    [("resnet18", 64, 2), ("vit_tiny_patch16_224", 64, 4)],
)
def test_export_parity(tmp_path: Path, arch: str, size: int, grid: int) -> None:
    kw = {"img_size": size} if arch.startswith("vit") else {}
    model = build_classifier(ModelConfig(arch=arch, init="random", **kw))
    path = tmp_path / "model.onnx"
    export_onnx(model, size, path)
    x = np.random.default_rng(0).random(
        (3, 1, size, size), dtype=np.float32
    )  # batch != export batch
    diffs = check_parity(model, path, x)
    assert diffs["logits"] <= PARITY_ATOL

    import onnxruntime as ort

    _, cams, _ = ort.InferenceSession(str(path)).run(None, {"image": x})
    assert cams.shape == (3, 14, grid, grid)
