"""ONNX Runtime inference engine. Loaded once at startup; thread-safe `run` calls."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from cxr_common.bundle import Manifest
from cxr_common.preprocess import Letterbox, U8Image, letterbox, to_model_input


@dataclass
class RawPrediction:
    probabilities: npt.NDArray[np.float64]  # [K] calibrated
    cams: npt.NDArray[np.float32]  # [K, h, w] logit units
    ood_score: float | None
    letterbox: Letterbox
    inference_ms: float


def _providers(pref: str) -> list[str]:
    avail = ort.get_available_providers()
    if pref == "cpu":
        return ["CPUExecutionProvider"]
    if pref == "cuda":
        if "CUDAExecutionProvider" not in avail:
            raise RuntimeError("CUDA execution provider requested but not available")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class Engine:
    def __init__(self, bundle_dir: Path, providers: str = "auto") -> None:
        self.bundle_dir = bundle_dir
        self.manifest = Manifest.load(bundle_dir)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(bundle_dir / self.manifest.onnx_file), so, providers=_providers(providers)
        )
        self.provider = self.session.get_providers()[0]
        self.temps = np.array([f.temperature for f in self.manifest.findings], dtype=np.float64)
        self.ood_mean: npt.NDArray[np.float64] | None = None
        self.ood_prec: npt.NDArray[np.float64] | None = None
        if self.manifest.ood is not None:
            z = np.load(bundle_dir / self.manifest.ood.file)
            self.ood_mean = z["mean"].astype(np.float64)
            self.ood_prec = z["precision"].astype(np.float64)
        self.ready = False

    @property
    def input_size(self) -> int:
        return self.manifest.input_size

    def warmup(self, n: int = 2) -> None:
        x = np.zeros((1, 1, self.input_size, self.input_size), dtype=np.float32)
        for _ in range(n):
            self.session.run(None, {"image": x})
        self.ready = True

    def run(self, img: U8Image) -> RawPrediction:
        sq, lb = letterbox(img, self.input_size)
        x = to_model_input(sq)
        t = time.perf_counter()
        logits, cams, feats = self.session.run(None, {"image": x})
        ms = (time.perf_counter() - t) * 1000
        z = logits[0].astype(np.float64) / self.temps
        probs = 1.0 / (1.0 + np.exp(-z))
        ood = None
        if self.ood_mean is not None and self.ood_prec is not None:
            d = feats[0].astype(np.float64) - self.ood_mean
            ood = float(d @ self.ood_prec @ d)
        return RawPrediction(probs, cams[0], ood, lb, ms)
