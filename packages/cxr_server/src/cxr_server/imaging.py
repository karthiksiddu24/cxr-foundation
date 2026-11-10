"""Upload sniffing, decoding and validation. Everything happens in memory; nothing is persisted.

Types are identified by magic bytes, never by filename or client-declared content type.
Image dimensions are read from headers *before* full decoding to block decompression bombs.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from enum import StrEnum

import cv2
import numpy as np
import pydicom
from PIL import Image

from cxr_common.dicom import DicomInfo, dicom_to_uint8, looks_like_dicom
from cxr_common.preprocess import U8Image, to_gray_uint8

Image.MAX_IMAGE_PIXELS = None  # we enforce our own limit from header dimensions


class ImageFormat(StrEnum):
    png = "png"
    jpeg = "jpeg"
    dicom = "dicom"


class Severity(StrEnum):
    info = "info"
    caution = "caution"
    high = "high"  # output should not be trusted


@dataclass(frozen=True)
class QualityIssue:
    code: str
    severity: Severity
    message: str


class InputError(ValueError):
    """Client error with a stable machine-readable code."""

    def __init__(self, code: str, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


@dataclass
class DecodedImage:
    pixels: U8Image
    format: ImageFormat
    warnings: list[QualityIssue] = field(default_factory=list)
    dicom: DicomInfo | None = None


def sniff(data: bytes) -> ImageFormat:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ImageFormat.png
    if data[:3] == b"\xff\xd8\xff":
        return ImageFormat.jpeg
    if looks_like_dicom(data):
        return ImageFormat.dicom
    raise InputError(
        "unsupported_type", "Unsupported file type. Upload a PNG, JPEG or DICOM file.", 415
    )


def _check_dims(w: int, h: int, min_side: int, max_side: int) -> None:
    if max(w, h) > max_side:
        raise InputError("image_too_large", f"Image is {w}x{h}; the maximum side is {max_side}px.")
    if min(w, h) < min_side:
        raise InputError(
            "image_too_small",
            f"Image is {w}x{h}; at least {min_side}px on the shorter side is required.",
        )


def _color_warning(img: np.ndarray) -> QualityIssue | None:
    if img.ndim == 3 and img.shape[2] >= 3:
        c = img[:, :, :3].astype(np.int16)
        spread = float(np.mean(np.abs(c - c.mean(axis=2, keepdims=True))))
        if spread > 6.0:
            return QualityIssue(
                "color_image",
                Severity.high,
                "This is a color image. Chest X-rays are grayscale, so this is probably not a "
                "radiograph.",
            )
    return None


def decode(data: bytes, min_side: int, max_side: int) -> DecodedImage:
    fmt = sniff(data)
    warnings: list[QualityIssue] = []
    if fmt is ImageFormat.dicom:
        try:
            ds = pydicom.dcmread(io.BytesIO(data))
            rows, cols = int(ds.get("Rows", 0)), int(ds.get("Columns", 0))
        except Exception as exc:
            raise InputError("invalid_dicom", "The DICOM file could not be read.") from exc
        _check_dims(cols, rows, min_side, max_side)
        try:
            pixels, info = dicom_to_uint8(ds)
        except Exception as exc:
            raise InputError(
                "invalid_dicom",
                "The DICOM pixel data could not be decoded (unsupported compression or "
                "corrupt file).",
            ) from exc
        warnings.extend(_dicom_warnings(info))
        return DecodedImage(pixels, fmt, warnings, info)

    try:
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
    except Exception as exc:
        raise InputError("invalid_image", "The image file is corrupt or truncated.") from exc
    _check_dims(w, h, min_side, max_side)
    raw = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise InputError("invalid_image", "The image file is corrupt or truncated.")
    cw = _color_warning(raw)
    if cw:
        warnings.append(cw)
    return DecodedImage(to_gray_uint8(raw), fmt, warnings)


_FRONTAL = {"PA", "AP", "PA ERECT", "AP SUPINE", "AP ERECT"}
_LATERAL = {"LL", "RL", "LAT", "LATERAL", "LLD", "RLD"}


def _dicom_warnings(info: DicomInfo) -> list[QualityIssue]:
    out: list[QualityIssue] = []
    if info.modality and info.modality.upper() not in {"CR", "DX", "DR", "RG", "OT"}:
        out.append(
            QualityIssue(
                "unexpected_modality",
                Severity.high,
                f"DICOM modality is {info.modality}; this model only handles plain radiographs.",
            )
        )
    vp = (info.view_position or "").upper()
    if vp in _LATERAL:
        out.append(
            QualityIssue(
                "lateral_view",
                Severity.high,
                "This appears to be a lateral view. The model was trained on frontal views only.",
            )
        )
    elif vp and vp not in _FRONTAL:
        out.append(
            QualityIssue("unknown_view", Severity.caution, f"Unrecognized view position {vp}.")
        )
    bp = (info.body_part or "").upper()
    if bp and "CHEST" not in bp and "THORAX" not in bp and "LUNG" not in bp:
        out.append(
            QualityIssue(
                "unexpected_body_part",
                Severity.high,
                f"DICOM body part is {info.body_part}; this model only handles chest images.",
            )
        )
    return out


def image_heuristics(img: U8Image) -> list[QualityIssue]:
    out: list[QualityIssue] = []
    h, w = img.shape
    ar = w / h
    if ar < 0.6 or ar > 1.6:
        out.append(
            QualityIssue(
                "unusual_aspect_ratio",
                Severity.caution,
                f"Aspect ratio {ar:.2f} is unusual for a chest radiograph.",
            )
        )
    if float(img.std()) < 12.0:
        out.append(
            QualityIssue(
                "low_contrast",
                Severity.caution,
                "The image has very low contrast; results may be unreliable.",
            )
        )
    return out


def display_png(img: U8Image, max_side: int) -> tuple[bytes, int, int]:
    h, w = img.shape
    s = min(1.0, max_side / max(h, w))
    out = (
        cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        if s < 1.0
        else img
    )
    ok, buf = cv2.imencode(".png", out, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes(), int(out.shape[1]), int(out.shape[0])
