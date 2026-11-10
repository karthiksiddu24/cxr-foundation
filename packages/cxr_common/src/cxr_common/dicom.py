"""DICOM -> display-ready uint8 grayscale, with correct LUT handling and no PHI exposure.

Only pixel data and a small allow-list of rendering/acquisition tags are read. Patient identifiers
are never accessed, returned, or logged. Note: identifiers burned into the pixels cannot be removed
here; that is a documented limitation.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pydicom
from pydicom.pixels.processing import apply_modality_lut, apply_voi_lut

from cxr_common.preprocess import U8Image, to_gray_uint8

DICM_MAGIC_OFFSET = 128
DICM_MAGIC = b"DICM"

# Tags we are allowed to read. Everything else (including all Patient/Study identifiers) is ignored.
SAFE_TAGS: tuple[str, ...] = (
    "Modality",
    "ViewPosition",
    "BodyPartExamined",
    "PhotometricInterpretation",
    "BitsStored",
    "Rows",
    "Columns",
)


@dataclass(frozen=True)
class DicomInfo:
    """Non-identifying acquisition info, used only for plausibility warnings."""

    modality: str | None
    view_position: str | None
    body_part: str | None
    photometric: str | None
    bits_stored: int | None
    window_source: str  # "voi_lut" | "window_tags" | "percentile" | "color"


def looks_like_dicom(data: bytes) -> bool:
    return len(data) >= DICM_MAGIC_OFFSET + 4 and (
        data[DICM_MAGIC_OFFSET : DICM_MAGIC_OFFSET + 4] == DICM_MAGIC
    )


def _str_tag(ds: pydicom.Dataset, name: str) -> str | None:
    v = ds.get(name)
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _percentile_window(arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    lo, hi = (float(v) for v in np.percentile(arr, [0.5, 99.5]))
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max()) + 1e-6
    out: npt.NDArray[np.float64] = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return out


def _first(v: object) -> float:
    """Window tags may be multi-valued; use the first (the default presentation)."""
    if isinstance(v, (list, tuple)) or type(v).__name__ == "MultiValue":
        return float(v[0])  # type: ignore[index]
    return float(v)  # type: ignore[arg-type]


def apply_linear_window(
    x: npt.NDArray[np.float64], center: float, width: float
) -> npt.NDArray[np.float64]:
    """DICOM PS3.3 C.11.2.1.2 'LINEAR' window, mapped to [0, 1]."""
    width = max(width, 1.0)
    y = (x - (center - 0.5)) / (width - 1.0) + 0.5 if width > 1 else (x >= center) * 1.0
    return np.clip(np.asarray(y, dtype=np.float64), 0.0, 1.0)


def dicom_to_uint8(data: bytes | pydicom.Dataset) -> tuple[U8Image, DicomInfo]:
    """Decode a DICOM file (bytes or dataset) to uint8 grayscale where bright = dense (bone).

    Order of operations follows the DICOM grayscale pipeline:
    stored values -> Modality LUT (rescale) -> VOI LUT / window -> polarity (MONOCHROME1 invert).
    """
    ds = data if isinstance(data, pydicom.Dataset) else pydicom.dcmread(io.BytesIO(data))
    if "PixelData" not in ds:
        raise ValueError("DICOM has no pixel data")
    arr = ds.pixel_array
    photometric = _str_tag(ds, "PhotometricInterpretation")
    n_frames = int(ds.get("NumberOfFrames", 1) or 1)
    if n_frames > 1:
        arr = arr[0]
    bits = ds.get("BitsStored")

    def info(source: str) -> DicomInfo:
        return DicomInfo(
            modality=_str_tag(ds, "Modality"),
            view_position=_str_tag(ds, "ViewPosition"),
            body_part=_str_tag(ds, "BodyPartExamined"),
            photometric=photometric,
            bits_stored=int(bits) if bits is not None else None,
            window_source=source,
        )

    if photometric not in ("MONOCHROME1", "MONOCHROME2"):
        # Color (e.g. secondary capture RGB/YBR): pydicom returns RGB; convert to gray.
        rgb = np.asarray(arr)
        if rgb.ndim == 3 and rgb.shape[-1] == 3:
            bgr = rgb[..., ::-1]
            return to_gray_uint8(bgr), info("color")
        return to_gray_uint8(rgb), info("color")

    x = np.asarray(apply_modality_lut(arr, ds), dtype=np.float64)
    if "VOILUTSequence" in ds:
        lut_item = ds.VOILUTSequence[0]
        out_bits = int(lut_item.LUTDescriptor[2])
        y = np.asarray(apply_voi_lut(x, ds, index=0, prefer_lut=True), dtype=np.float64)
        y = np.clip(y / float(2**out_bits - 1), 0.0, 1.0)
        source = "voi_lut"
    elif "WindowCenter" in ds and "WindowWidth" in ds:
        y = apply_linear_window(x, _first(ds.WindowCenter), _first(ds.WindowWidth))
        source = "window_tags"
    else:
        y = _percentile_window(x)
        source = "percentile"

    if photometric == "MONOCHROME1":
        y = 1.0 - y
    out = np.clip(np.rint(y * 255.0), 0, 255).astype(np.uint8)
    return out, info(source)
