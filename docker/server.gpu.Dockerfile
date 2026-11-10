# GPU inference server: onnxruntime-gpu on a CUDA runtime base. Falls back to CPU if no GPU.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS base
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=/opt/python
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/* && uv python install 3.11
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
COPY packages/cxr_common packages/cxr_common
COPY packages/cxr_server packages/cxr_server
COPY packages/cxr_ml/pyproject.toml packages/cxr_ml/pyproject.toml
RUN mkdir -p packages/cxr_ml/src/cxr_ml && touch packages/cxr_ml/src/cxr_ml/__init__.py \
 && uv sync --frozen --no-dev --package cxr-server --extra gpu --no-install-package onnxruntime --no-editable \
 && useradd --uid 10001 --create-home app
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1 CXR_BUNDLE_DIR=/models/current CXR_ORT_PROVIDERS=auto
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/ready',timeout=4).status==200 else 1)"
CMD ["uvicorn", "cxr_server.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
