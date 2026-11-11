# API

FastAPI service serving one model bundle with ONNX Runtime. The OpenAPI schema is in
[`openapi.json`](openapi.json). Interactive docs are at `http://localhost:8000/docs` when the
API runs.

> Research demonstration only. Not a medical device. Not for clinical use.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `GET` | `/ready` | Readiness: a bundle is loaded and the model has run once |
| `GET` | `/v1/model-info` | Model card, findings, thresholds and heatmap grid from the bundle manifest |
| `POST` | `/v1/predict` | Analyze one image (`multipart/form-data`, field `file`) |

Through the web app the same endpoints are under `/api/*` (same-origin proxy).

## `POST /v1/predict`

```bash
curl -s -F "file=@chest.png" http://localhost:8000/v1/predict
```

**Accepted inputs:** PNG, JPEG or DICOM.
- The type is detected from file content, not the extension.
- Size limit: 25 MB (`CXR_MAX_UPLOAD_MB`).
- Image dimensions are checked from the header before decoding.
- DICOM modality and VOI LUTs are applied, and MONOCHROME1 images are inverted.
- Patient fields are never read.

**Response** (abridged, from a real request against the served model):

```json
{
  "request_id": "9454cde3ed474a2a8a5d410d8299b28e",
  "model": { "model_id": "ft_vitb_ssl_f100_s0", "version": "1.0.0", "placeholder": false },
  "image": { "format": "png", "width": 1024, "height": 1024,
             "display_png_base64": "iVBORw0KGgo...", "display_width": 1024, "display_height": 1024 },
  "geometry": { "input_size": 224, "grid_h": 14, "grid_w": 14, "scale": 0.21875, "pad_x": 0, "pad_y": 0 },
  "quality": { "plausible": true, "warnings": [], "ood_score": 835.14, "ood_threshold": 1797.9 },
  "findings": [
    {
      "name": "Cardiomegaly",
      "display_name": "Cardiomegaly",
      "description": "The heart appears larger than expected relative to the chest.",
      "probability": 0.7598,
      "threshold": 0.02591,
      "flagged": true,
      "test_auroc": { "point": 0.89682, "ci_low": 0.87967, "ci_high": 0.91085 },
      "heatmap": { "grid": [[0.415, 0.454, 0.493, 0.46, "..."]], "logit_min": -0.5114, "logit_max": 2.448 }
    }
  ],
  "timing": { "decode_ms": 8.5, "inference_ms": 46.4, "total_ms": 154.6 },
  "disclaimer": "Research demonstration only. Not a medical device. Not for clinical use."
}
```

### Findings
- **Order:** sorted by `probability`, highest first. All 14 findings are always returned.
- **`probability`:** the calibrated score, after per-finding temperature scaling fitted on
  validation data.
- **`threshold` and `flagged`:** `threshold` maximizes Youden's J on validation data, and
  `flagged` means `probability >= threshold`. These are model scores, not diagnoses.

### Heatmaps
- **Shape and scale:** `heatmap.grid` is `grid_h × grid_w`, min-max scaled to [0, 1] *within
  each finding*. It shows relative location, not absolute strength. `logit_min` and `logit_max`
  give the raw range.
- **Placing it on the image:** the grid covers the square model input. Map it onto the original
  image with `original = (model - pad) / scale`.

### Quality
- `plausible: false` means the image looks unlike a frontal chest X-ray: either a heuristic
  warning has severity `high` (color image, lateral view, unexpected modality or body part,
  extreme aspect ratio), or the Mahalanobis feature distance exceeds the 99.5th percentile seen
  on validation data.
- The web app hides scores in that case unless the user chooses to reveal them.

## Errors

Every error is RFC 7807 `application/problem+json`:

```json
{ "type": "about:blank", "title": "Invalid input", "status": 415, "code": "unsupported_type",
  "detail": "Unsupported file type. Upload a PNG, JPEG or DICOM file.", "request_id": "..." }
```

| Status | `code` examples | Meaning |
|---|---|---|
| 413 | `file_too_large` | Upload exceeds the size limit |
| 415 | `unsupported_type` | Not PNG, JPEG or DICOM by content |
| 422 | `invalid_image`, `invalid_dicom`, `image_too_small`, `image_too_large` | Cannot be decoded or dimensions out of range |
| 422 | `validation_error` | Malformed request (e.g. missing `file` field) |
| 500 | `internal_error` | Unexpected server error (logged with the request id) |
| 429 | `rate_limited` | More than `CXR_RATE_LIMIT` requests (default 30/minute per client) |
| 503 | `not_ready` | No bundle loaded |

## Configuration

Environment variables use the `CXR_` prefix. See [`../.env.example`](../.env.example).

| Variable | Default |
|---|---|
| `CXR_BUNDLE_DIR` | `/models/current` |
| `CXR_ORT_PROVIDERS` | `cpu` (`cuda` with the GPU image) |
| `CXR_MAX_UPLOAD_MB` | `25` |
| `CXR_RATE_LIMIT` | `30/minute` |
| `CXR_CORS_ORIGINS` | `["http://localhost:3000"]` |
| `CXR_LOG_JSON` | `true` (structured logs; image bytes and patient fields are never logged) |
