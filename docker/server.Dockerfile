# CPU inference server. Installs only cxr-server + cxr-common (no torch): small image, fast boot.
FROM python:3.11.14-slim-bookworm AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
COPY packages/cxr_common packages/cxr_common
COPY packages/cxr_server packages/cxr_server
# cxr_ml is a workspace member; uv needs its pyproject to resolve the lock, not its code.
COPY packages/cxr_ml/pyproject.toml packages/cxr_ml/pyproject.toml
RUN mkdir -p packages/cxr_ml/src/cxr_ml && touch packages/cxr_ml/src/cxr_ml/__init__.py \
 && uv sync --frozen --no-dev --package cxr-server --no-editable

FROM python:3.11.14-slim-bookworm
RUN useradd --uid 10001 --create-home app
COPY --from=build /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1 CXR_BUNDLE_DIR=/models/current
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/ready',timeout=4).status==200 else 1)"
CMD ["uvicorn", "cxr_server.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
