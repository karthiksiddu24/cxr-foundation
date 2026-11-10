from __future__ import annotations

import base64
import logging
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest
from server_helpers import (
    ClientFactory,
    G,
    K,
    chest_like,
    dicom_bytes,
    jpeg_bytes,
    make_settings,
    png_bytes,
)

from cxr_server.app import create_app
from cxr_server.engine import Engine


async def post(client: httpx.AsyncClient, data: bytes, name: str = "x.png") -> httpx.Response:
    return await client.post(
        "/v1/predict", files={"file": (name, data, "application/octet-stream")}
    )


# ------------------------------------------------------------------ ops endpoints


async def test_health_and_ready(client: httpx.AsyncClient) -> None:
    assert (await client.get("/health")).json() == {"status": "ok"}
    r = await client.get("/ready")
    assert r.status_code == 200
    assert r.json()["ready"] is True
    assert r.json()["model_id"] == "tiny-test"


async def test_ready_is_503_before_warmup(bundle_dir: Path) -> None:
    eng = Engine(bundle_dir, "cpu")
    app = create_app(make_settings(bundle_dir), eng)
    app.state.engine = eng  # not warmed
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        assert (await c.get("/ready")).status_code == 503
        r = await post(c, png_bytes(chest_like()))
        assert r.status_code == 503


async def test_lifespan_loads_and_warms(bundle_dir: Path) -> None:
    app = create_app(make_settings(bundle_dir))
    async with app.router.lifespan_context(app):
        assert app.state.engine.ready


async def test_model_info(client: httpx.AsyncClient) -> None:
    r = await client.get("/v1/model-info")
    assert r.status_code == 200
    body = r.json()
    assert len(body["findings"]) == K
    assert "Not for clinical use" in body["disclaimer"]
    assert body["card"]["limitations"]


async def test_openapi_has_predict_schema(client: httpx.AsyncClient) -> None:
    spec = (await client.get("/openapi.json")).json()
    assert "/v1/predict" in spec["paths"]
    assert "PredictResponse" in spec["components"]["schemas"]


# ------------------------------------------------------------------ predict: happy paths


async def test_predict_png(client: httpx.AsyncClient) -> None:
    r = await post(client, png_bytes(chest_like()))
    assert r.status_code == 200, r.text
    body = r.json()
    probs = [f["probability"] for f in body["findings"]]
    assert len(probs) == K
    assert probs == sorted(probs, reverse=True)
    f0 = body["findings"][0]
    assert f0["flagged"] == (f0["probability"] >= f0["threshold"])
    grid = np.array(f0["heatmap"]["grid"])
    assert grid.shape == (G, G)
    assert grid.min() >= 0 and grid.max() <= 1
    assert body["geometry"]["grid_h"] == G
    assert body["image"]["format"] == "png"
    png = base64.b64decode(body["image"]["display_png_base64"])
    assert png[:4] == b"\x89PNG"
    assert r.headers["x-request-id"] == body["request_id"]
    assert "Not a medical device" in body["disclaimer"]


async def test_predict_jpeg_and_portrait_geometry(client: httpx.AsyncClient) -> None:
    r = await post(client, jpeg_bytes(chest_like(600, 400)), "a.jpg")
    assert r.status_code == 200
    g = r.json()["geometry"]
    assert g["pad_x"] > 0
    assert g["pad_y"] == 0
    assert r.json()["image"]["width"] == 400


async def test_type_is_sniffed_not_taken_from_extension(client: httpx.AsyncClient) -> None:
    r = await post(client, png_bytes(chest_like()), "scan.dcm")
    assert r.status_code == 200
    assert r.json()["image"]["format"] == "png"
    r = await post(client, b"hello, this is text pretending to be an image", "scan.png")
    assert r.status_code == 415
    assert r.json()["code"] == "unsupported_type"
    assert r.headers["content-type"].startswith("application/problem+json")


async def test_predict_dicom_monochrome1_is_inverted(client: httpx.AsyncClient) -> None:
    base = (chest_like().astype(np.uint16) * 16).astype(np.uint16)
    r2 = await post(client, dicom_bytes(base, "MONOCHROME2"), "a.dcm")
    r1 = await post(client, dicom_bytes(4095 - base, "MONOCHROME1"), "b.dcm")
    assert r1.status_code == r2.status_code == 200
    d1 = cv2.imdecode(
        np.frombuffer(base64.b64decode(r1.json()["image"]["display_png_base64"]), np.uint8), 0
    )
    d2 = cv2.imdecode(
        np.frombuffer(base64.b64decode(r2.json()["image"]["display_png_base64"]), np.uint8), 0
    )
    assert np.abs(d1.astype(int) - d2.astype(int)).mean() < 3
    assert r1.json()["image"]["format"] == "dicom"


# ------------------------------------------------------------------ validation


async def test_rejects_tiny_and_huge_images(client: httpx.AsyncClient) -> None:
    r = await post(client, png_bytes(np.zeros((32, 32), np.uint8)))
    assert r.status_code == 422
    assert r.json()["code"] == "image_too_small"
    r = await post(client, png_bytes(np.zeros((9000, 300), np.uint8)))
    assert r.status_code == 422
    assert r.json()["code"] == "image_too_large"


async def test_rejects_oversized_upload(client_factory: ClientFactory) -> None:
    c = client_factory(max_upload_mb=0.01)
    r = await post(c, png_bytes(np.random.default_rng(0).integers(0, 255, (300, 300), np.uint8)))
    assert r.status_code == 413
    assert r.json()["code"] == "file_too_large"


async def test_rejects_empty_and_truncated(client: httpx.AsyncClient) -> None:
    assert (await post(client, b"")).status_code in (415, 422)
    trunc = png_bytes(chest_like())[:200]
    r = await post(client, trunc)
    assert r.status_code == 422
    assert r.json()["code"] == "invalid_image"


async def test_missing_file_field_is_problem_json(client: httpx.AsyncClient) -> None:
    r = await client.post("/v1/predict")
    assert r.status_code == 422
    assert r.json()["code"] == "validation_error"


# ------------------------------------------------------------------ plausibility guard


async def test_color_photo_is_flagged_implausible(client: httpx.AsyncClient) -> None:
    img = np.zeros((300, 300, 3), np.uint8)
    img[..., 2] = 200  # strongly red
    img[100:200, 100:200, 1] = 180
    r = await post(client, png_bytes(img))
    assert r.status_code == 200
    q = r.json()["quality"]
    assert q["plausible"] is False
    assert "color_image" in [w["code"] for w in q["warnings"]]


async def test_lateral_dicom_is_flagged(client: httpx.AsyncClient) -> None:
    r = await post(client, dicom_bytes(chest_like().astype(np.uint16) * 16, view="LL"))
    codes = [w["code"] for w in r.json()["quality"]["warnings"]]
    assert "lateral_view" in codes
    assert r.json()["quality"]["plausible"] is False


async def test_ood_flags_atypical_input(client: httpx.AsyncClient) -> None:
    r = await post(client, png_bytes(np.full((300, 300), 255, np.uint8)))
    q = r.json()["quality"]
    assert q["ood_score"] > q["ood_threshold"]
    assert "atypical_image" in [w["code"] for w in q["warnings"]]


async def test_typical_input_is_plausible(client: httpx.AsyncClient) -> None:
    q = (await post(client, png_bytes(chest_like()))).json()["quality"]
    assert q["plausible"] is True, q


# ------------------------------------------------------------------ ops: rate limit, ids, privacy


async def test_rate_limit(client_factory: ClientFactory) -> None:
    c = client_factory(rate_limit="2/minute")
    data = png_bytes(chest_like())
    codes = [(await post(c, data)).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


async def test_request_id_is_propagated_or_generated(client: httpx.AsyncClient) -> None:
    r = await client.get("/health", headers={"X-Request-ID": "abc12345-trace"})
    assert r.headers["x-request-id"] == "abc12345-trace"
    r = await client.get("/health", headers={"X-Request-ID": "bad id with spaces"})
    assert r.headers["x-request-id"] != "bad id with spaces"


async def test_cors_preflight(client: httpx.AsyncClient) -> None:
    r = await client.options(
        "/v1/predict",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
    )
    assert r.headers["access-control-allow-origin"] == "http://localhost:3000"
    r = await client.options(
        "/v1/predict",
        headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in r.headers


async def test_no_phi_or_filename_in_logs(
    client: httpx.AsyncClient, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    data = dicom_bytes(chest_like().astype(np.uint16) * 16, patient_name="SMITH^SECRET")
    r = await post(client, data, "SMITH_SECRET_mrn998877.dcm")
    assert r.status_code == 200
    out = capsys.readouterr()
    logs = out.out + out.err + caplog.text
    assert "prediction" in logs  # the structured log line was emitted
    for secret in ("SMITH", "SECRET", "PHI-998877", "998877"):
        assert secret not in logs
        assert secret not in r.text


async def test_upload_never_touches_disk(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tempfile

    def boom(*a: object, **k: object) -> None:
        raise AssertionError("upload was spooled to disk")

    monkeypatch.setattr(tempfile.SpooledTemporaryFile, "rollover", boom)
    big = np.random.default_rng(0).integers(0, 255, (1500, 1500), np.uint8)  # ~2.2 MB PNG
    data = png_bytes(big)
    assert len(data) > 1024 * 1024
    r = await post(client, data)
    assert r.status_code == 200
