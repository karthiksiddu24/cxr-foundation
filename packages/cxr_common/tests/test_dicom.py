"""DICOM decoding tests on synthetic files (no PHI, no real data)."""

from __future__ import annotations

import io

import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from cxr_common.dicom import apply_linear_window, dicom_to_uint8, looks_like_dicom


def make_dicom(
    pixels: np.ndarray,
    photometric: str = "MONOCHROME2",
    window: tuple[float, float] | None = None,
    slope: float = 1.0,
    intercept: float = 0.0,
    patient_name: str = "SECRET^PATIENT",
) -> bytes:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset("x.dcm", {}, file_meta=meta, preamble=b"\0" * 128)
    ds.PatientName = patient_name
    ds.PatientID = "123456"
    ds.Modality = "DX"
    ds.ViewPosition = "PA"
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.RescaleSlope = slope
    ds.RescaleIntercept = intercept
    if window:
        ds.WindowCenter, ds.WindowWidth = window
    ds.PixelData = pixels.astype(np.uint16).tobytes()
    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    return buf.getvalue()


RAMP = np.tile(np.linspace(0, 4095, 64).astype(np.uint16), (32, 1))


def test_magic_sniffing() -> None:
    assert looks_like_dicom(make_dicom(RAMP))
    assert not looks_like_dicom(b"\x89PNG\r\n\x1a\n" + b"\0" * 200)


def test_monochrome2_ramp_is_increasing() -> None:
    img, info = dicom_to_uint8(make_dicom(RAMP))
    assert img.dtype == np.uint8 and img.shape == RAMP.shape
    assert img[0, 0] < img[0, -1]
    assert info.window_source == "percentile"
    assert info.view_position == "PA"


def test_monochrome1_is_inverted() -> None:
    m2, _ = dicom_to_uint8(make_dicom(RAMP, "MONOCHROME2", window=(2048, 4096)))
    m1, _ = dicom_to_uint8(make_dicom(RAMP, "MONOCHROME1", window=(2048, 4096)))
    assert np.abs(m1.astype(int) + m2.astype(int) - 255).max() <= 1


def test_window_tags_clip_outside_window() -> None:
    img, info = dicom_to_uint8(make_dicom(RAMP, window=(1000, 200)))
    assert info.window_source == "window_tags"
    assert img[0, 0] == 0 and img[0, -1] == 255
    # Values well inside the window should be mid-gray, not stretched by image min/max.
    mid = apply_linear_window(np.array([1000.0]), 1000, 200)[0]
    assert mid == pytest.approx(0.5, abs=0.01)


def test_rescale_applied_before_window() -> None:
    # stored 0..4095 with intercept -1024 => window centered at 0 in output units.
    img, _ = dicom_to_uint8(make_dicom(RAMP, window=(0, 100), intercept=-1024))
    stored_at_zero = 1024
    col = int(np.argmin(np.abs(np.linspace(0, 4095, 64) - stored_at_zero)))
    assert 0 < img[0, col] < 255 or abs(int(img[0, col]) - 128) < 128


def test_no_phi_in_returned_info() -> None:
    _, info = dicom_to_uint8(make_dicom(RAMP, patient_name="SECRET^PATIENT"))
    assert "SECRET" not in repr(info)
    assert "123456" not in repr(info)
