"""Helpers and fixtures: a tiny hand-built ONNX bundle (no torch, no real data) and synthetic images."""

from __future__ import annotations

import io
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import cv2
import httpx
import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from cxr_common.bundle import Finding, Manifest, ModelCard, OODInfo, Provenance, ThresholdInfo
from cxr_common.labels import DESCRIPTIONS, DISPLAY_NAMES, NIH_LABELS
from cxr_server.app import create_app
from cxr_server.engine import Engine
from cxr_server.settings import Settings

S = 64  # model input size
G = 4  # heatmap grid
K = len(NIH_LABELS)


def tiny_onnx(path: Path) -> None:
    """image[B,1,S,S] -> avgpool to GxG -> 1x1 conv to K maps (cams); logits = mean(cams)."""
    rng = np.random.default_rng(0)
    w = numpy_helper.from_array(rng.normal(0, 3, (K, 1, 1, 1)).astype(np.float32), "W")
    b = numpy_helper.from_array(rng.normal(0, 1, (K,)).astype(np.float32), "B")
    axes = numpy_helper.from_array(np.array([2, 3], dtype=np.int64), "axes")
    k = S // G
    nodes = [
        helper.make_node("AveragePool", ["image"], ["pooled"], kernel_shape=[k, k], strides=[k, k]),
        helper.make_node("Conv", ["pooled", "W", "B"], ["cams"]),
        helper.make_node("ReduceMean", ["cams", "axes"], ["logits"], keepdims=0),
        helper.make_node("Flatten", ["pooled"], ["features"], axis=1),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["batch", 1, S, S])],
        [
            helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", K]),
            helper.make_tensor_value_info("cams", TensorProto.FLOAT, ["batch", K, G, G]),
            helper.make_tensor_value_info("features", TensorProto.FLOAT, ["batch", G * G]),
        ],
        initializer=[w, b, axes],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


@pytest.fixture(scope="session")
def bundle_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("bundle")
    tiny_onnx(d / "model.onnx")
    # OOD stats: typical radiograph-like inputs have pooled features ~0.3-0.7.
    np.savez(
        d / "ood.npz",
        mean=np.full(G * G, 0.5, np.float32),
        precision=np.eye(G * G, dtype=np.float32) * 25.0,
    )
    Manifest(
        model_id="tiny-test",
        version="0.0.1",
        placeholder=False,
        arch="tiny",
        init="random",
        input_size=S,
        onnx_sha256="0" * 64,
        opset=18,
        heatmap_grid=(G, G),
        findings=[
            Finding(
                name=n,
                display_name=DISPLAY_NAMES[n],
                description=DESCRIPTIONS[n],
                temperature=1.0,
                threshold=ThresholdInfo(threshold=0.5, method="test"),
            )
            for n in NIH_LABELS
        ],
        ood=OODInfo(feature_dim=G * G, threshold=40.0, threshold_quantile=0.99, fit_images=1),
        card=ModelCard(
            intended_use="test",
            not_intended_for=["x"],
            training_data="none",
            evaluation_data="none",
            limitations=["test model"],
        ),
        provenance=Provenance(
            run="tiny",
            git_commit=None,
            git_dirty=None,
            config_hash=None,
            split_digest=None,
            checkpoint_sha256=None,
            created_at="2026-01-01",
        ),
    ).save(d)
    return d


def make_settings(bundle_dir: Path, **kw: object) -> Settings:
    base: dict[str, object] = dict(
        bundle_dir=bundle_dir,
        ort_providers="cpu",
        log_json=True,
        rate_limit="1000/minute",
        min_image_side=64,
    )
    base.update(kw)
    return Settings(**base)  # type: ignore[arg-type]


ClientFactory = Callable[..., httpx.AsyncClient]


@pytest.fixture
async def client_factory(bundle_dir: Path) -> AsyncIterator[ClientFactory]:
    """Builds clients against fresh apps. httpx's ASGITransport does not run the lifespan, so the
    warmed engine is attached to app.state directly (the lifespan itself is tested separately)."""
    clients: list[httpx.AsyncClient] = []

    def make(**kw: object) -> httpx.AsyncClient:
        engine = Engine(bundle_dir, "cpu")
        engine.warmup()
        app = create_app(make_settings(bundle_dir, **kw), engine)
        app.state.engine = engine
        c = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
        clients.append(c)
        return c

    yield make
    for c in clients:
        await c.aclose()


@pytest.fixture
def client(client_factory: ClientFactory) -> httpx.AsyncClient:
    return client_factory()


def chest_like(h: int = 512, w: int = 512) -> np.ndarray:
    """Synthetic grayscale 'radiograph': bright mediastinum, darker lungs, smooth gradients."""
    yy, xx = np.mgrid[0:h, 0:w] / max(h, w)
    img = (
        0.55
        + 0.25 * np.exp(-((xx - 0.5) ** 2) / 0.01)
        - 0.2 * np.exp(-(((xx - 0.3) ** 2) + (yy - 0.45) ** 2) / 0.03)
        - 0.2 * np.exp(-(((xx - 0.7) ** 2) + (yy - 0.45) ** 2) / 0.03)
    )
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def png_bytes(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def jpeg_bytes(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def dicom_bytes(
    img16: np.ndarray,
    photometric: str = "MONOCHROME2",
    view: str = "PA",
    patient_name: str = "DOE^JANE",
    body_part: str = "CHEST",
) -> bytes:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset("x.dcm", {}, file_meta=meta, preamble=b"\0" * 128)
    ds.PatientName = patient_name
    ds.PatientID = "PHI-998877"
    ds.Modality = "DX"
    ds.ViewPosition = view
    ds.BodyPartExamined = body_part
    ds.Rows, ds.Columns = img16.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
    ds.PixelData = img16.astype(np.uint16).tobytes()
    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    return buf.getvalue()
