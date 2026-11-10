"""Image preprocessing shared by training-time data prep and the inference server.

Both paths MUST go through these functions so that the model sees identically processed pixels in
training and serving. The model itself (inside the ONNX graph) applies intensity normalization,
so the contract at the model boundary is simply: float32 [B, 1, S, S] in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

U8Image = npt.NDArray[np.uint8]


@dataclass(frozen=True)
class Letterbox:
    """Geometry of a letterbox resize, used to map heatmaps/boxes between spaces."""

    orig_h: int
    orig_w: int
    size: int
    scale: float  # resized = orig * scale
    pad_y: int  # top padding in resized space
    pad_x: int  # left padding in resized space

    def orig_to_model(self, x: float, y: float) -> tuple[float, float]:
        return x * self.scale + self.pad_x, y * self.scale + self.pad_y

    def model_to_orig(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.pad_x) / self.scale, (y - self.pad_y) / self.scale

    @property
    def content_box(self) -> tuple[int, int, int, int]:
        """(x0, y0, x1, y1) of the image content inside the square, model space."""
        h = round(self.orig_h * self.scale)
        w = round(self.orig_w * self.scale)
        return self.pad_x, self.pad_y, self.pad_x + w, self.pad_y + h


def to_gray_uint8(arr: npt.NDArray[np.generic]) -> U8Image:
    """Convert a decoded image of any common layout/bit depth to 2-D uint8 grayscale.

    - uint8 gray: unchanged.
    - color/alpha (H, W, 3|4) or 2-channel gray+alpha: luminance of the color channels.
    - higher bit depth (uint16, float): min-max scaled to 0..255. (DICOM goes through dicom.py,
      which applies proper windowing before reaching here.)
    """
    a = np.asarray(arr)
    if a.ndim == 3:
        c = a.shape[2]
        if c == 1 or c == 2:
            a = a[:, :, 0]
        elif c in (3, 4):
            rgb = a[:, :, :3].astype(np.float32)
            # OpenCV decodes BGR; luminance weights applied in BGR order.
            a = rgb[:, :, 0] * 0.114 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.299
            if arr.dtype == np.uint8:
                gray: U8Image = np.clip(np.rint(a), 0, 255).astype(np.uint8)
                return gray
        else:
            raise ValueError(f"unsupported channel count {c}")
    if a.ndim != 2:
        raise ValueError(f"expected 2-D or 3-D image, got shape {a.shape}")
    if a.dtype == np.uint8:
        out: U8Image = np.ascontiguousarray(a, dtype=np.uint8)
        return out
    f = a.astype(np.float64)
    lo, hi = float(f.min()), float(f.max())
    if hi <= lo:
        return np.zeros(a.shape, dtype=np.uint8)
    return np.clip(np.rint((f - lo) / (hi - lo) * 255.0), 0, 255).astype(np.uint8)


def decode_raster(data: bytes) -> U8Image:
    """Decode PNG/JPEG bytes to uint8 grayscale. Raises ValueError if undecodable."""
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("could not decode image")
    return to_gray_uint8(img)


def letterbox(img: U8Image, size: int) -> tuple[U8Image, Letterbox]:
    """Resize so the longer side equals `size`, then center-pad with black to a square.

    Padding rather than cropping guarantees no anatomy is cut off. Area interpolation is used for
    downscaling (anti-aliased), linear for upscaling.
    """
    if img.ndim != 2 or img.dtype != np.uint8:
        raise ValueError("letterbox expects 2-D uint8")
    h, w = img.shape
    scale = size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp) if (nh, nw) != (h, w) else img
    out = np.zeros((size, size), dtype=np.uint8)
    py, px = (size - nh) // 2, (size - nw) // 2
    out[py : py + nh, px : px + nw] = resized
    return out, Letterbox(h, w, size, scale, py, px)


def to_model_input(batch_u8: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    """[B, S, S] or [S, S] uint8 -> float32 [B, 1, S, S] in [0, 1]."""
    a = np.asarray(batch_u8)
    if a.ndim == 2:
        a = a[None]
    if a.ndim != 3:
        raise ValueError(f"expected [B,S,S] or [S,S], got {a.shape}")
    return (a.astype(np.float32) / 255.0)[:, None]
