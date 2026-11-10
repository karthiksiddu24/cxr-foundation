from __future__ import annotations

import cv2
import numpy as np
import pytest

from cxr_common.preprocess import decode_raster, letterbox, to_gray_uint8, to_model_input


def test_letterbox_square_is_plain_resize() -> None:
    img = np.random.default_rng(0).integers(0, 256, (1024, 1024), dtype=np.uint8)
    out, lb = letterbox(img, 224)
    assert out.shape == (224, 224)
    assert (lb.pad_x, lb.pad_y) == (0, 0)
    assert lb.content_box == (0, 0, 224, 224)


def test_letterbox_portrait_pads_left_right_and_maps_coords() -> None:
    img = np.full((400, 200), 200, dtype=np.uint8)
    out, lb = letterbox(img, 100)
    assert out.shape == (100, 100)
    assert lb.scale == pytest.approx(0.25)
    assert (lb.pad_x, lb.pad_y) == (25, 0)
    assert out[:, :25].max() == 0 and out[:, 75:].max() == 0
    assert out[:, 25:75].min() == 200
    x, y = lb.orig_to_model(100, 200)
    assert lb.model_to_orig(x, y) == pytest.approx((100, 200))


def test_letterbox_upscales_small_images() -> None:
    out, lb = letterbox(np.zeros((50, 100), dtype=np.uint8), 200)
    assert out.shape == (200, 200)
    assert lb.scale == 2.0


def test_to_gray_handles_rgba_and_16bit() -> None:
    bgra = np.zeros((4, 4, 4), dtype=np.uint8)
    bgra[..., :3] = 100
    assert (to_gray_uint8(bgra) == 100).all()
    g16 = np.array([[0, 1000], [2000, 4000]], dtype=np.uint16)
    g = to_gray_uint8(g16)
    assert g.dtype == np.uint8 and g.min() == 0 and g.max() == 255
    assert to_gray_uint8(np.full((3, 3), 7, dtype=np.uint16)).max() == 0  # constant -> zeros


def test_decode_raster_roundtrip_png_and_jpeg() -> None:
    img = np.random.default_rng(1).integers(0, 256, (64, 48), dtype=np.uint8)
    ok, png = cv2.imencode(".png", img)
    assert ok
    assert np.array_equal(decode_raster(png.tobytes()), img)
    ok, jpg = cv2.imencode(".jpg", img)
    assert decode_raster(jpg.tobytes()).shape == (64, 48)


def test_decode_raster_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="decode"):
        decode_raster(b"not an image at all")


def test_to_model_input_shape_and_range() -> None:
    x = to_model_input(np.array([[0, 255]], dtype=np.uint8))
    assert x.shape == (1, 1, 1, 2) and x.dtype == np.float32
    assert x.min() == 0.0 and x.max() == 1.0
