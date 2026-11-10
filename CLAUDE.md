# CLAUDE.md — CXR-Foundation

Self-supervised (MAE) chest X-ray model + evaluation suite + FastAPI/Next.js demo app.
PLAN.md holds the original plan; this file is the source of truth for how things work *now*.

## Hard rules
- **Never fabricate numbers.** Every metric in README/RESULTS.md/model card is generated from
  files under `$CXR_DATA_ROOT/artifacts/eval/` by scripts. If something wasn't run, say so.
- **Test split is sacred.** Hyperparameters are chosen on `val` only (`eval_test=false` for sweeps;
  `scripts/select_lr.py` reads only val). MAE pretraining refuses `include_splits` containing test.
- **Patient level everywhere:** splits, label-fraction subsets, bootstrap CIs.
- Do not download credentialed datasets (CheXpert, VinDr, MIMIC); scripts expect them under
  `$CXR_DATA_ROOT/raw/<name>`.
- CheXpert valid images: `raw/chexpert/valid/` (CheXpert Plus PNG_valid). Labels: no CSV in
  that release, so `cxr_ml.data.external` derives them from CheXlocalize ground-truth masks in
  `raw/chexlocalize/gt_segmentations_val.json` (positive iff mask non-empty; Pneumonia
  unlabeled). Positives on 202 frontal images: Atel 75, Cardio 66, Effusion 64, Edema 42,
  Consol 32, PTX 7, Lung Lesion 1 (findings with < 5 positives are not scored). The same masks
  drive `localization --dataset chexlocalize`. The test set is not available in that release.
- Uploaded images never touch disk or logs (see `cxr_server/app.py`; enforced by tests).
- Commit only on request; end commit messages with the Co-Authored-By line used in history.

## Layout
```
packages/cxr_common   pure NumPy: preprocessing (letterbox), DICOM->uint8, labels, bundle schema
packages/cxr_ml       torch: data, models (classifier w/ in-graph CAMs, MAE), train, eval, export
packages/cxr_server   FastAPI + ONNX Runtime (must never import torch)
web/                  Next.js 16 App Router, TS strict, Tailwind v4, shadcn-style ui on radix-ui
configs/              YAML configs (OmegaConf structured dataclasses) + queues/ (job lists)
scripts/              download, prep, benchmarks, reports, launchers
docker/, docker-compose.yml (+ docker-compose.gpu.yml), .github/workflows/ci.yml
```
Data/checkpoints/artifacts live in `$CXR_DATA_ROOT` (default `/data/cxr`), never in git:
`raw/ processed/nih/{index.parquet,img_224.u8,img_512.u8,splits.json} runs/<name>/
artifacts/{eval,bundles}/ mlflow/ logs/`.

## Commands
```bash
# env
uv sync                                   # Python 3.11 workspace (.venv)
export PATH=$HOME/.local/node/bin:$PATH   # Node 24 is installed user-locally on this box

# data
uv run scripts/download_nih.py --out /data/cxr/raw/nih    # resumable, sha256-verified
uv run scripts/prep_nih.py                # index + patient splits + memmaps (~5 min)
uv run scripts/data_report.py             # docs/data_report.md
uv run scripts/benchmark_loader.py        # docs/benchmarks/loader.json

# training (all resumable; finished runs are skipped via DONE marker)
uv run python -m cxr_ml.train.classify --config configs/sup/base.yaml model.arch=densenet121 name=X
uv run python -m cxr_ml.train.mae --config configs/mae/vit_base.yaml [--benchmark]
scripts/run_queue.sh configs/queues/<file>.txt /data/cxr/logs/<dir>   # sequential job queue
uv run python -m cxr_ml.eval.evaluate <run> [<run> ...]               # metrics.json + figures
uv run scripts/prep_external.py && uv run python -m cxr_ml.eval.external <run>   # CheXpert valid
uv run python -m cxr_ml.explain.localization <run> --dataset chexlocalize        # vs. masks
uv run python -m cxr_ml.export.bundle <run> --version 1.0.0           # ONNX bundle (+parity check)
uv run python -m cxr_ml.export.bundle --placeholder                   # untrained dev bundle

# tests / quality
uv run pytest -q                          # all Python tests (data-marked ones skip w/o data)
uv run ruff check . && uv run ruff format --check . && uv run mypy
cd web && npm run lint && npm run typecheck && npm test && npm run build
cd web && E2E_BASE_URL=http://localhost:3000 npx playwright test     # needs API + web running

# app
uv run uvicorn cxr_server.app:create_app --factory --port 8000       # API (CXR_BUNDLE_DIR)
cd web && npm run dev                                                 # http://localhost:3000
docker compose up --build                 # full stack; bundle from CXR_BUNDLE_HOST_DIR
uv run scripts/export_openapi.py && (cd web && npm run gen:api)       # after API schema changes
```

## Architecture decisions (and why)
- **Stored at 224 and 512 px, letterboxed** (long side = S, zero pad). Eval and serving consume
  exactly the stored geometry, so train/serve preprocessing is identical (`cxr_common.preprocess`).
- **224 px set lives on the GPU** (5.6 GB uint8); augmentation is batched on GPU
  (`data/augment.py`, one affine `grid_sample`). The box has only 8 CPU cores; PNG decode tops out
  at ~615 img/s vs ~27k img/s for GPU gather+augment (measured, docs/benchmarks/loader.json).
- **No horizontal flips** by default (anatomical asymmetry, L/R markers).
- **Model input contract:** float32 `[B,1,S,S]` in [0,1]; intensity normalization is inside the
  model (GRAY_MEAN/STD), so it is inside the ONNX graph too.
- **In-graph CAMs for CNNs and ViTs:** both end in avg-pool + linear (ViT uses `fc_norm=False`,
  i.e. norm *before* pooling), so `mean(cams) == logits` exactly. One forward pass gives scores,
  14 heatmaps and the pooled embedding (for Mahalanobis OOD). Tested, and cross-checked against
  MONAI's GradCAM in `tests/test_cam_vs_monai_gradcam.py` (MONAI's default normalizer is
  inverted: max -> 0). MONAI is used only for this verification; the training pipeline uses a
  custom GPU-batched path for throughput.
- **Bundle = contract** (`cxr_common.bundle.Manifest`): onnx + calibration temps + val-chosen
  thresholds + OOD stats + model card copied from eval metrics.json.
- **Bootstrap:** patient-clustered, 2000 resamples, fixed seed shared by all runs (paired diffs
  valid); weighted AUROC/AP implementation matches sklearn exactly (tests).
- **Config:** OmegaConf structured dataclasses + dotlist overrides instead of full Hydra
  (simpler, no cwd changes); sweeps are plain queue files.
- **Tracking:** MLflow file store at `$CXR_DATA_ROOT/mlflow/runs` (MLflow 3 needs
  `MLFLOW_ALLOW_FILE_STORE=true`, set automatically; mlflow-skinny can't use sqlite).
  `tracking=wandb` works offline. Every run also writes `metrics.jsonl` + `provenance.json`
  (git commit/dirty, config hash, split digest, env).
- **Web:** results are held only in React state (never storage); browser calls same-origin
  `/api/*`, proxied by Next rewrites (destination baked at build time → Docker build arg).
- **Versions:** Python deps pinned in pyproject + uv.lock; npm deps pinned exactly; TypeScript
  5.9 (openapi-typescript peer), ESLint 9 (eslint-config-next plugin compat).

## Gotchas
- `pkill -f "<pattern>"` inside a Bash tool call can match its own shell; use
  `fuser -k PORT/tcp` or a `[b]racket` pattern in a separate call.
- Starlette spools multipart >1 MB to disk unless `MultiPartParser.spool_max_size` is raised
  (done in `create_app`; `test_upload_never_touches_disk` guards it).
- `uv run` re-syncs the full workspace (pulls torch); use `uv run --no-sync` in light envs.
- The NIH HF mirror zips contain macOS `._*.png` files; the downloader skips them.
- CheXlocalize masks are COCO RLE (column-major); `cxr_ml.data.rle` is a tested port, no pycocotools.
  Its `gradcam_maps_val/*.pkl` files are pickles from the authors' model: not needed, never load them.
- Concurrent GPU jobs (MAE + supervised) share throughput; MAE ETA in logs reflects that.
