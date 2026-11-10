"""FastAPI application.

    uvicorn cxr_server.app:create_app --factory --host 0.0.0.0 --port 8000

Privacy: uploads are read into memory, processed, and dropped. Nothing is written to disk; file
names, DICOM tags and pixel data are never logged.
"""

from __future__ import annotations

import base64
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
import structlog
from fastapi import FastAPI, File, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.formparsers import MultiPartParser

from cxr_server import __version__
from cxr_server.engine import Engine
from cxr_server.imaging import (
    InputError,
    QualityIssue,
    Severity,
    decode,
    display_png,
    image_heuristics,
)
from cxr_server.schemas import (
    FindingInfo,
    FindingResult,
    Health,
    Heatmap,
    HeatmapGeometry,
    ImageInfo,
    ModelInfo,
    ModelRef,
    PredictResponse,
    Problem,
    Quality,
    QualityWarning,
    Ready,
    Timing,
)
from cxr_server.settings import Settings, get_settings

DISCLAIMER = "Research demonstration only. Not a medical device. Not for clinical use."
_REQ_ID_RE = re.compile(r"^[A-Za-z0-9._-]{8,64}$")

log = structlog.get_logger("cxr_server")


def configure_logging(level: str, json_logs: bool) -> None:
    logging.basicConfig(format="%(message)s", level=level.upper())
    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level.upper())),
        cache_logger_on_first_use=True,
    )


def problem(status: int, code: str, title: str, detail: str, request: Request) -> JSONResponse:
    body = Problem(
        title=title,
        status=status,
        detail=detail,
        code=code,
        request_id=getattr(request.state, "request_id", None),
    )
    return JSONResponse(
        body.model_dump(), status_code=status, media_type="application/problem+json"
    )


def create_app(settings: Settings | None = None, engine: Engine | None = None) -> FastAPI:
    s = settings or get_settings()
    configure_logging(s.log_level, s.log_json)
    # Starlette spools multipart files larger than 1 MB to a temp file on disk. Raise the spool
    # threshold above our upload limit so uploads never touch the filesystem.
    MultiPartParser.spool_max_size = s.max_upload_bytes + 1024 * 1024
    limiter = Limiter(key_func=get_remote_address, default_limits=[])

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        eng = engine or Engine(s.bundle_dir, s.ort_providers)
        app.state.engine = eng
        log.info(
            "model_loaded",
            model_id=eng.manifest.model_id,
            provider=eng.provider,
            placeholder=eng.manifest.placeholder,
        )
        if s.warmup and not eng.ready:
            await run_in_threadpool(eng.warmup)
            log.info("warmup_complete")
        else:
            eng.ready = True
        yield

    app = FastAPI(
        title="CXR-Foundation API",
        version=__version__,
        description=f"Chest X-ray finding scores with heatmaps. **{DISCLAIMER}**",
        lifespan=lifespan,
    )
    app.state.limiter = limiter
    app.add_middleware(
        CORSMiddleware,
        allow_origins=s.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        incoming = request.headers.get("x-request-id", "")
        rid = incoming if _REQ_ID_RE.match(incoming) else uuid.uuid4().hex
        request.state.request_id = rid
        structlog.contextvars.bind_contextvars(request_id=rid)
        t = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("unhandled_error", path=request.url.path)
            response = problem(
                500, "internal_error", "Internal error", "An unexpected error occurred.", request
            )
        response.headers["X-Request-ID"] = rid
        log.info(
            "request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            ms=round((time.perf_counter() - t) * 1000, 1),
        )
        structlog.contextvars.clear_contextvars()
        return response

    @app.exception_handler(InputError)
    async def input_error(request: Request, exc: InputError) -> JSONResponse:
        return problem(exc.status, exc.code, "Invalid input", str(exc), request)

    @app.exception_handler(RateLimitExceeded)
    async def rate_limited(request: Request, exc: RateLimitExceeded) -> JSONResponse:
        return problem(
            429,
            "rate_limited",
            "Too many requests",
            f"Rate limit exceeded ({exc.detail}). Please wait and try again.",
            request,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return problem(
            422,
            "validation_error",
            "Invalid request",
            "; ".join(e.get("msg", "") for e in exc.errors()),
            request,
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return problem(exc.status_code, "http_error", str(exc.detail), str(exc.detail), request)

    def engine_of(request: Request) -> Engine:
        eng: Engine = request.app.state.engine
        return eng

    @app.get("/health", response_model=Health, tags=["ops"], summary="Liveness probe")
    async def health() -> Health:
        return Health(status="ok")

    @app.get(
        "/ready",
        response_model=Ready,
        tags=["ops"],
        summary="Readiness probe",
        responses={503: {"model": Ready}},
    )
    async def ready(request: Request, response: Response) -> Ready:
        eng: Engine | None = getattr(request.app.state, "engine", None)
        if eng is None or not eng.ready:
            response.status_code = 503
            return Ready(ready=False)
        return Ready(ready=True, model_id=eng.manifest.model_id, execution_provider=eng.provider)

    @app.get(
        "/v1/model-info",
        response_model=ModelInfo,
        tags=["model"],
        summary="Model card, metrics and intended use",
    )
    async def model_info(request: Request) -> ModelInfo:
        m = engine_of(request).manifest
        return ModelInfo(
            model_id=m.model_id,
            version=m.version,
            placeholder=m.placeholder,
            arch=m.arch,
            init=m.init,
            input_size=m.input_size,
            heatmap_grid=m.heatmap_grid,
            findings=[
                FindingInfo(
                    name=f.name,
                    display_name=f.display_name,
                    description=f.description,
                    threshold=f.threshold.threshold,
                    threshold_method=f.threshold.method,
                    temperature=f.temperature,
                    test_auroc=f.test_auroc,
                    test_auprc=f.test_auprc,
                    test_prevalence=f.test_prevalence,
                    test_sensitivity=f.threshold.test_sensitivity,
                    test_specificity=f.threshold.test_specificity,
                )
                for f in m.findings
            ],
            card=m.card,
            git_commit=m.provenance.git_commit,
            created_at=m.provenance.created_at,
            execution_provider=engine_of(request).provider,
            disclaimer=DISCLAIMER,
        )

    @app.post(
        "/v1/predict",
        response_model=PredictResponse,
        tags=["model"],
        summary="Score a chest X-ray (PNG, JPEG or DICOM)",
        responses={
            413: {"model": Problem},
            415: {"model": Problem},
            422: {"model": Problem},
            429: {"model": Problem},
            503: {"model": Problem},
        },
    )
    @limiter.limit(s.rate_limit)
    async def predict(
        request: Request,
        file: Annotated[
            UploadFile, File(description="PNG, JPEG or DICOM, max size per server config")
        ],
    ) -> PredictResponse | JSONResponse:
        t0 = time.perf_counter()
        eng = engine_of(request)
        if not eng.ready:
            return problem(
                503, "not_ready", "Model loading", "The model is still loading.", request
            )
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > s.max_upload_bytes + 64 * 1024:
            raise InputError("file_too_large", f"File exceeds {s.max_upload_mb:g} MB.", 413)
        data = await file.read(s.max_upload_bytes + 1)
        await file.close()
        if len(data) > s.max_upload_bytes:
            raise InputError("file_too_large", f"File exceeds {s.max_upload_mb:g} MB.", 413)
        if not data:
            raise InputError("empty_file", "The uploaded file is empty.")

        decoded = await run_in_threadpool(decode, data, s.min_image_side, s.max_image_side)
        del data
        t_dec = time.perf_counter()
        raw = await run_in_threadpool(eng.run, decoded.pixels)
        warnings: list[QualityIssue] = [*decoded.warnings, *image_heuristics(decoded.pixels)]
        ood_thr = eng.manifest.ood.threshold if eng.manifest.ood else None
        if raw.ood_score is not None and ood_thr is not None and raw.ood_score > ood_thr:
            warnings.append(
                QualityIssue(
                    "atypical_image",
                    Severity.high,
                    "This image looks unlike the chest X-rays the model was trained on. Scores are "
                    "unreliable.",
                )
            )
        if eng.manifest.placeholder:
            warnings.append(
                QualityIssue(
                    "placeholder_model",
                    Severity.high,
                    "The server is running an untrained placeholder model. Scores are meaningless.",
                )
            )
        png, dw, dh = await run_in_threadpool(display_png, decoded.pixels, s.display_max_side)

        findings: list[FindingResult] = []
        for k, f in enumerate(eng.manifest.findings):
            cam = raw.cams[k].astype(np.float64)
            lo, hi = float(cam.min()), float(cam.max())
            norm = (cam - lo) / (hi - lo) if hi > lo else np.zeros_like(cam)
            p = float(raw.probabilities[k])
            findings.append(
                FindingResult(
                    name=f.name,
                    display_name=f.display_name,
                    description=f.description,
                    probability=round(p, 5),
                    threshold=f.threshold.threshold,
                    flagged=p >= f.threshold.threshold,
                    test_auroc=f.test_auroc,
                    heatmap=Heatmap(
                        grid=np.round(norm, 3).tolist(),
                        logit_min=round(lo, 4),
                        logit_max=round(hi, 4),
                    ),
                )
            )
        findings.sort(key=lambda r: r.probability, reverse=True)
        lb = raw.letterbox
        h, w = decoded.pixels.shape
        gh, gw = raw.cams.shape[-2:]
        total = (time.perf_counter() - t0) * 1000
        log.info(
            "prediction",
            format=decoded.format.value,
            width=w,
            height=h,
            inference_ms=round(raw.inference_ms, 1),
            plausible=not any(x.severity is Severity.high for x in warnings),
            warnings=[x.code for x in warnings],
        )
        return PredictResponse(
            request_id=request.state.request_id,
            model=ModelRef(
                model_id=eng.manifest.model_id,
                version=eng.manifest.version,
                placeholder=eng.manifest.placeholder,
            ),
            image=ImageInfo(
                format=decoded.format.value,
                width=w,
                height=h,
                display_png_base64=base64.b64encode(png).decode("ascii"),
                display_width=dw,
                display_height=dh,
            ),
            geometry=HeatmapGeometry(
                input_size=lb.size,
                grid_h=int(gh),
                grid_w=int(gw),
                scale=lb.scale,
                pad_x=lb.pad_x,
                pad_y=lb.pad_y,
            ),
            quality=Quality(
                plausible=not any(x.severity is Severity.high for x in warnings),
                warnings=[
                    QualityWarning(code=x.code, severity=x.severity.value, message=x.message)
                    for x in warnings
                ],
                ood_score=None if raw.ood_score is None else round(raw.ood_score, 2),
                ood_threshold=None if ood_thr is None else round(ood_thr, 2),
            ),
            findings=findings,
            timing=Timing(
                decode_ms=round((t_dec - t0) * 1000, 1),
                inference_ms=round(raw.inference_ms, 1),
                total_ms=round(total, 1),
            ),
            disclaimer=DISCLAIMER,
        )

    return app
