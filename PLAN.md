# CXR-Foundation — Project Plan

Status: **DRAFT, waiting for approval.** No code has been written. Nothing in this document is a
result; all timings below are *estimates to be replaced by measured benchmarks*.

---

## 0. What I found on the machine (2026-11-10)

| Item | Finding | Consequence |
|---|---|---|
| Repo | Did not exist; created `/home/ubuntu/cxr-foundation/` containing only this file | Greenfield |
| GPU | NVIDIA L40S, 46 GB, driver 610.57 | bf16 + `torch.compile` fine |
| GPU contention | **ComfyUI (PID 2398967) holds ~33 GB VRAM and ~23 GB RAM** | Must be stopped during training (see Q1) |
| CPU / RAM | **8 cores**, 31 GB RAM (≈5 GB free right now) | CPU JPEG decode + augmentation will bottleneck → decode once, augment **on GPU** |
| Disk | `/` 68 GB free; `/data` 183 GB free (QEMU virtual disk, reported as rotational) | Code on `/`, data+checkpoints on `/data/cxr`. Disk budget is tight (§3.6) |
| Python | System 3.12; `uv` installed | `uv python install 3.11`, pin in `.python-version` |
| Node | **Not installed** | Install Node 22 LTS user-locally (no sudo) for the web app |
| Docker | 29.8 with `nvidia` runtime + CDI | GPU containers work |
| git / gh | git present, **no global identity**; `gh` not installed | Need name/email for commits (Q7); no pushing until you say so |

---

## 1. Goals & non-goals

**Goals**
1. A rigorous, reproducible study answering: *does MAE self-supervised pretraining on chest X-rays
   alone improve label efficiency over random and ImageNet initialization?*
2. Honest external validation (CheXpert expert labels) and localization checks (VinDr boxes).
3. A production-quality, accessible web app serving the best model, with clear non-diagnostic framing.

**Non-goals**
- Beating state of the art. NIH-only MAE pretraining (~112k images) is small by SSL standards;
  published work suggests MAE-on-CXR helps mostly in low-label regimes, and ImageNet init may still
  win at 100% labels. **We report whatever we find.** A negative result, clearly analyzed, is still
  a strong portfolio piece.
- Clinical use, regulatory claims, PHI handling beyond "never store, never log".

---

## 2. Architecture overview

```
                 ┌──────────────── offline (GPU box) ────────────────┐
 raw NIH PNGs ──►│ scripts/prep → uint8 memmaps + parquet index       │
 CheXpert/VinDr ►│ ml: train (sup / MAE / finetune) → checkpoints     │
                 │ ml: evaluate → eval artifacts (parquet/JSON)       │
                 │ ml: export → model bundle (ONNX + card + thresholds)│
                 │ scripts/report → RESULTS.md + docs/figures         │
                 └──────────────────────┬────────────────────────────┘
                                        │ model bundle (versioned dir)
                 ┌──────────────────────▼────────────────────────────┐
 browser ◄──────►│ web (Next.js)  ◄── typed client ──► server (FastAPI│
                 │                                     + ONNX Runtime)│
                 └────────────────────────────────────────────────────┘
```

Key design decisions:

1. **One shared preprocessing module (`cxr_common`)**, pure NumPy, used by both training-time
   export and the server. A parity test feeds the same file through both paths. This is the main
   defense against train/serve skew.
2. **Server never imports torch.** It depends on `onnxruntime(-gpu)`, NumPy, Pillow, pydicom only →
   small images, fast boot, CPU fallback.
3. **Heatmaps computed inside the ONNX graph.** For the CNN path, DenseNet/ResNet end in global
   average pooling + linear, so class activation maps (CAM = classifier weights · last feature map)
   are exact, need no gradients, and give all 14 maps in one forward pass. (For this head, Grad-CAM
   on the final conv layer reduces to CAM up to a positive scale, so we can validate the in-graph
   CAM against a PyTorch Grad-CAM in a test.) For ViTs we export attention-rollout outputs, with the
   caveat that rollout is not class-specific (see Q5).
4. **Model bundle is the contract** between ML and serving:
   `model.onnx`, `preprocess.json`, `calibration.json` (per-class temperature), `thresholds.json`,
   `model_card.json` (generated from eval artifacts — no hand-typed numbers), `MANIFEST.json`
   (git hash, config hash, checkpoint hash, data split hash).
5. **Every reported number** lives in an eval artifact under `/data/cxr/artifacts/eval/<run_id>/`
   and is rendered into RESULTS.md / model card by a script. CI checks RESULTS.md is up to date
   with the artifacts it references.

---

## 3. Part 1 — ML pipeline

### 3.1 Phase 0 — Scaffolding (small)
- `uv` workspace, Python 3.11, pinned lockfile. Packages: `cxr_common`, `cxr_ml`, `cxr_server`.
- ruff, mypy (strict on `cxr_common` and `cxr_server`, gradual on `cxr_ml`), pytest, pre-commit.
- CLAUDE.md with conventions and commands. `.env.example`. Skeleton CI (lint + type + unit tests on
  CPU with synthetic data — CI has no GPU and no real datasets).
- `make`-style task runner (`justfile` or `Makefile`) for the common commands.

### 3.2 Phase 1 — Data foundation

**NIH ChestX-ray14 download.** Open download from the official NIH links (12 tarballs, ~42 GB),
script is resumable and verifies files. Also downloads `Data_Entry_2017.csv`,
`train_val_list.txt`, `test_list.txt`, `BBox_List_2017.csv`.

**Storage format: uint8 NumPy memmaps + parquet index.**
- `nih_256.u8` (~7.3 GB) and `nih_512.u8` (~29 GB), shape `[N, H, W]`, plus `index.parquet`
  (row → image id, patient id, labels, view, age, sex, split).
- Resize: preserve aspect ratio (NIH is 1024², so it is a plain resize), area interpolation,
  done once with multiprocessing.
- Why memmap over LMDB/WebDataset: random access with zero decode cost, trivially shared across
  DataLoader workers, the 256-px set fits in page cache once ComfyUI is stopped. With only 8 CPU
  cores, the loader only slices bytes; **augmentation runs on the GPU** in batched torch ops.
- Benchmark script reports images/sec for: raw PNG decode, memmap read, memmap + GPU aug, and
  end-to-end training step. Reported in Phase 1 summary (measured, not estimated).

**Splits.**
- Respect the official `test_list.txt` as the held-out test set.
- Split `train_val_list.txt` by **patient** into train / val (default 90/10, seeded, stratified on
  label prevalence at patient level as far as possible).
- Unit test: fails if any patient ID appears in more than one split; also asserts the official test
  list is untouched. Run on the real index when present and on a synthetic fixture in CI.
- Label-fraction subsets (1%, 10%, 100%) are sampled **by patient** from train, nested
  (1% ⊂ 10% ⊂ 100%), saved as files, with a test for nesting and no leakage. Note: 1% is roughly
  ~800 images, so very rare classes (e.g. Hernia) may have only a handful of positives —
  we will report per-class positive counts per subset and flag classes with too few positives
  to evaluate meaningfully.

**Label harmonization (`cxr_common.labels`).**
Shared finding set and mappings, documented in `docs/labels.md`:

| Shared finding | NIH | CheXpert | VinDr (boxes) |
|---|---|---|---|
| Atelectasis | Atelectasis | Atelectasis | Atelectasis |
| Cardiomegaly | Cardiomegaly | Cardiomegaly | Cardiomegaly |
| Consolidation | Consolidation | Consolidation | Consolidation |
| Edema | Edema | Edema | — |
| Effusion | Effusion | Pleural Effusion | Pleural effusion |
| Pneumonia | Pneumonia | Pneumonia | — |
| Pneumothorax | Pneumothorax | Pneumothorax | Pneumothorax |
| Lung lesion (approx.) | Mass ∪ Nodule | Lung Lesion | Nodule/Mass |
| NIH-only | Infiltration, Emphysema, Fibrosis, Pleural_Thickening, Hernia, Mass, Nodule | — | Infiltration, Pleural thickening, Pulmonary fibrosis (localization only) |

- The model is trained on all 14 NIH labels; cross-dataset evaluation uses only the overlapping
  findings, with approximate mappings (lesion) reported separately and marked as approximate.
- **CheXpert uncertainty:** the CheXpert *valid* and *test* sets are radiologist-labeled and binary,
  so no uncertainty policy is needed for evaluation. CheXpert *training* labels (with `-1`
  uncertain) are not used for supervised training in this project. If they are ever used, the
  policy will be explicit and configurable (U-Ignore / U-Zeros / U-Ones per finding), documented in
  `docs/labels.md`.

**Stats report** (`scripts/data_report.py` → `docs/data_report.md` + figures, regenerated from the
index): per-split class prevalence, co-occurrence, images-per-patient distribution, view position
(PA/AP), age/sex, patient overlap check result. Also documents known NIH label noise (NLP-mined
labels).

### 3.3 Phase 2 — Supervised baselines
- Models: DenseNet-121, ResNet-50 (timm/torchvision), each with random init and ImageNet init → 4 runs.
- Input: grayscale replicated to 3 channels for ImageNet compatibility (or first conv summed to
  1 channel — tested equivalent). Resolution default **224 from the 256 store** for comparability
  with Phase 4 (see Q4 for a 512 run of the final CNN).
- Loss: BCE-with-logits with per-class `pos_weight` (configurable; alternative: asymmetric loss as
  ablation only if time permits).
- AdamW, cosine schedule with linear warmup, bf16 autocast, channels_last, `torch.compile`,
  EMA of weights (optional flag), early stopping on val mean AUROC.
- Augmentation (GPU): random resized crop (mild scale range), small rotation (±10°),
  brightness/contrast/gamma jitter in intensity space, **no color jitter, no horizontal flip by
  default** (see §3.4 note).
- **Metrics module** (`cxr_ml.eval`):
  - per-class AUROC and AUPRC, mean AUROC (over 14, and over the shared subset),
  - **patient-clustered bootstrap** 95% CIs (resample patients, not images, because NIH test
    patients have many images), 2,000 resamples, seeded,
  - **paired** bootstrap CIs for differences between two models on the same test set,
  - calibration: ECE (equal-mass bins), reliability diagrams, before/after temperature scaling,
  - all outputs saved as parquet/JSON + PNG figures.
- Config: **Hydra with typed structured configs** (dataclasses), so the label-fraction × init grid
  is a single multirun command. Every checkpoint stores resolved config, git hash, dirty-flag,
  data index hash, split hash, seed, library versions.
- Tracking: thin `Tracker` interface with W&B and MLflow backends + offline mode (see Q6).
- Reproducibility: global seeding, deterministic dataloader seeding, `cudnn.benchmark` noted as the
  one source of nondeterminism (documented; bitwise determinism costs too much speed).

### 3.4 Phase 3 — MAE pretraining from scratch
- Own MAE implementation (encoder = timm ViT, our masking + lightweight decoder), ViT-Small/16 and
  ViT-Base/16 at 224, mask ratio 0.75, normalized-pixel loss, AdamW, cosine schedule, bf16.
- Data: all NIH images (train_val **and** test? — **no**: we exclude the NIH test list from
  pretraining by default to keep the test set fully unseen; see Q3), labels discarded.
- Augmentation: random resized crop (scale 0.4–1.0), mild intensity jitter. **Horizontal flip:
  off by default.** Rationale for docs: CXRs have strong left–right anatomical asymmetry (heart,
  aortic arch, gastric bubble, liver); flipping creates anatomically implausible images
  (pseudo-dextrocardia), mirrors L/R lead markers and text, and could teach invariance to
  laterality that downstream findings (e.g. cardiomegaly, side of effusion) depend on. It is a
  config flag for an optional ablation.
- Resumable training: checkpoint every N minutes + every epoch (model, optimizer, scaler, scheduler,
  RNG states, dataloader position), atomic writes, keep-last-k, auto-resume from latest.
  Tested by killing and resuming a tiny run and comparing loss curves.
- Reconstruction grids (original / masked / reconstruction) logged every few epochs.
- Optional DINO: only after MAE results are in and if time remains.

**Wall-clock estimate (to be replaced by a measured benchmark before launch).** Rough FLOP-based
reasoning: MAE ViT-B/16 encodes only 25% of tokens, so a training step costs roughly 20–25 GFLOPs
per image. At a realistic 30–50% of L40S bf16 peak, that suggests roughly **1,500–3,000 img/s**,
i.e. ~1 min per epoch over ~86k pretraining images, so **800 epochs ≈ 12–25 h for ViT-B** and
roughly a third to half of that for ViT-S. This is shorter than "days" because the dataset is
small; the uncertainty is wide and depends on whether data loading keeps up. Before launching I
will run a 10-minute benchmark and give you a measured estimate.

### 3.5 Phase 4 — Fine-tuning & label-efficiency study
- Backbone for the headline comparison: **ViT-B/16** (so the three inits share an architecture):
  random, ImageNet (timm supervised weights), our MAE. ViT-S as secondary if time allows.
  DenseNet-121 ImageNet shown as a reference line.
- Label fractions 1%, 10%, 100% (patient-level, nested). **Seeds: 3 for 1% and 10%**, 1 for 100%
  (3 if budget allows), since small-data runs are high variance. Grid ≈ 3 inits × 7 runs = 21 runs.
- Same recipe across inits except layer-wise LR decay for pretrained ViTs (standard for MAE
  fine-tuning; applied to ImageNet ViT too for fairness), tuned only on val.
- Headline figure: mean AUROC (test) vs labeled fraction (log x-axis), one line per init, with
  bootstrap 95% CIs and per-seed points. Plus a paired-difference table (MAE − ImageNet, MAE −
  random) with CIs.
- Rough cost: fine-tuning ViT-B at 224 on 100% (~78k images), ~20–30 epochs ≈ 1–2 h; 1%/10% runs
  are minutes. Total grid likely under a day of GPU time (estimate).

### 3.6 Phase 5 — External validation, explainability, audit
- **CheXpert:** evaluate on the expert-labeled valid and test sets for the shared findings only.
  Report NIH-test vs CheXpert AUROC side-by-side with CIs, and analyze the drop (label definition
  mismatch, AP/PA mix, patient population, image processing differences, prevalence shift).
  The CheXpert expert sets are small, so CIs will be wide — stated explicitly.
- **Heatmaps:** CAM/Grad-CAM for CNNs, attention rollout (and optionally Grad-CAM on ViT tokens)
  for ViTs.
- **Localization:** VinDr-CXR **test set** (radiologist consensus boxes) for overlapping findings,
  plus the in-domain NIH `BBox_List_2017.csv` (~1k boxes). Metrics: pointing game (hit rate, with
  a randomized-map baseline and a "center of lung field" baseline), IoU / hit at thresholded
  heatmap, reported per finding with CIs.
- **Shortcut audit (pneumothorax ↔ chest drains):** preferred source is the public NIH
  pneumothorax chest-drain annotations released alongside shortcut-learning research (I will
  verify availability and license in Phase 5; if unavailable, I will build a labeling tool and you
  or I label ~200–300 test positives). Report AUROC on drain-present vs drain-absent positives —
  a large gap indicates shortcut reliance. Saliency galleries for both groups.
- **Failure galleries:** top false positives / false negatives per finding with heatmaps.
- **RESULTS.md** generated by `scripts/report.py` from eval artifacts; each table cell links to the
  artifact file it came from.

**Disk budget on `/data` (183 GB free):** NIH tarballs ~42 GB (deleted after verified extraction)
+ extracted PNGs ~45 GB + memmaps ~37 GB + VinDr test DICOMs (est. 30–40 GB, converted then
optionally deleted) + CheXpert valid/test (small) + checkpoints/artifacts ~20–30 GB.
Fits, but only if tarballs are removed after extraction (Q8).

### 3.7 Export
- Export selected model to ONNX (opset pinned) with outputs `logits` and `cams`
  (and TorchScript as backup).
- Parity test: max abs diff of logits/probs between PyTorch and ONNX Runtime on a fixed batch
  (tolerance set for fp32; bf16 is not exported).
- Calibration: per-class temperature scaling fit on val. Thresholds per class chosen on val
  (default: maximize Youden's J; alternative: target sensitivity, configurable) and saved in the
  bundle together with the val sensitivity/specificity they imply.
- The served model is the one with best **val** mean AUROC among models that support class-specific
  heatmaps (Q5).

---

## 4. Part 2 — Web application

### 4.1 Backend (`server/`, FastAPI)
- Python 3.11, fully typed (mypy strict), Pydantic v2 models, OpenAPI 3.1 at `/docs`.
- ONNX Runtime with `CUDAExecutionProvider` → `CPUExecutionProvider` fallback; model loaded once in
  lifespan, warmup inference on boot; `/ready` false until warmup completes.
- **Inputs:** PNG, JPEG, DICOM.
  - Type sniffing by magic bytes (`\x89PNG`, `\xFF\xD8\xFF`, DICOM `DICM` at offset 128), not
    extension. Size limit (default 25 MB), dimension limits (min 256 px, max 8192 px), decompression
    bomb guard.
  - DICOM: read in memory with pydicom; apply Modality LUT (rescale slope/intercept), VOI LUT /
    window center-width when present (else robust percentile window), **MONOCHROME1 inversion**,
    handle BitsStored correctly. Only pixel data and the handful of rendering tags are accessed;
    **no tag is logged, returned, or persisted**. Compressed transfer syntaxes via
    `pylibjpeg` plugins. Known limitation: PHI burned into pixels cannot be removed.
  - Nothing is written to disk; request bodies are processed in memory and dropped.
- **Plausibility guard** ("is this a frontal CXR?"): heuristics (grayscale check, aspect ratio,
  intensity histogram) + Mahalanobis distance of penultimate features to the NIH training
  distribution, threshold set on held-out NIH vs a negative set (CheXpert lateral views + generic
  non-CXR images). Out-of-distribution inputs get `quality.warnings[]` and a UI that withholds
  confident presentation. Documented as a basic guard, not a guarantee.
- **Endpoints:** `POST /v1/predict` (calibrated probabilities, thresholds, flags, per-finding
  heatmaps as compact base64 grayscale PNGs, display image, warnings, timing),
  `GET /health` (liveness), `GET /ready` (model loaded + warmed), `GET /v1/model-info`
  (from `model_card.json`).
- structlog JSON logs, request-ID middleware (`X-Request-ID` echoed), rate limiting (slowapi,
  in-memory), CORS allowlist from env, RFC 7807 problem+json errors.
- Tests: pytest + httpx `AsyncClient`, using a tiny ONNX fixture model generated in tests and
  synthetic PNG/DICOM fixtures (MONOCHROME1/2, 12/16-bit, compressed), so CI needs no real data.

### 4.2 Frontend (`web/`, Next.js App Router)
- TypeScript strict, Tailwind, shadcn/ui, Lucide, `next-themes` (dark default), Inter/IBM Plex
  Mono-style pairing (final choice in a design pass), `openapi-typescript` to generate types from the
  server's OpenAPI schema (CI fails if generated types are stale).
- **Design direction:** neutral near-black surfaces, one restrained accent (cool cyan-teal) used for
  focus and interactive state only; amber for "flagged", never red/green-only signaling;
  8-pt spacing scale; tabular numerals for probabilities; motion ≤ 200 ms and respects
  `prefers-reduced-motion`.
- **Screens:** Upload (drag-and-drop, one-click samples, privacy note) → Viewer + findings →
  Model card → Methodology.
- **Viewer:** canvas/WebGL renderer with zoom (wheel/pinch), pan, window/level (drag + sliders,
  shader-based so it's real W/L on the image, not CSS filters), reset, fullscreen, heatmap overlay
  with colormap + opacity slider, keyboard shortcuts (`+/-`, `0` reset, `W` W/L mode, `H` toggle
  heatmap, `F` fullscreen, `↑/↓` move between findings, `?` help).
- **Findings panel:** sorted by probability, bars with threshold tick, "Above threshold" badge
  (wording deliberately not "positive"), plain-language tooltips, calibrated-probability wording,
  explicit warnings when the plausibility guard fires.
- **Disclaimer** persistent in header/footer: "Research demonstration only. Not a medical device.
  Not for clinical use."
- **Sample images:** a few images from the **NIH** test set only (NIH data is openly
  redistributable with citation). **CheXpert and VinDr images are never shipped** (their licenses
  forbid redistribution).
- Polish: skeletons, inference progress, empty/error states with retry, toasts (sonner), no CLS,
  responsive to tablet, usable on mobile (viewer simplifies to pinch-zoom + findings sheet),
  WCAG 2.1 AA.
- Tests: Vitest + Testing Library (components), Playwright e2e (upload → results, keyboard path,
  axe accessibility scan), Lighthouse CI budget (≥ 90 perf/a11y/best-practices) on the built app.

### 4.3 Infrastructure
- Dockerfiles: `server` CPU (python-slim + onnxruntime), `server` GPU (CUDA runtime +
  onnxruntime-gpu), `web` (multi-stage, Next.js standalone output). Non-root users, healthchecks.
- `docker compose up` launches web + server (CPU by default; `--profile gpu` for GPU). Model bundle
  mounted read-only from a configurable path.
- GitHub Actions: ruff, mypy, pytest (CPU), eslint, tsc, vitest, Playwright against compose stack,
  docker build of all images. Pre-commit mirrors the lint/type steps.
- `.env.example` documents every variable.

**Unblocking the web app early:** before real models exist, a *placeholder bundle* (ImageNet
DenseNet with a randomly initialized 14-class head, clearly labeled "UNTRAINED PLACEHOLDER" in
`/model-info` and the UI) lets backend/frontend work proceed while MAE trains in the background.

---

## 5. Repo structure (refined)

```
cxr-foundation/
  pyproject.toml            # uv workspace root, tool config (ruff, mypy, pytest)
  uv.lock  .python-version  justfile  .pre-commit-config.yaml  .env.example
  packages/
    cxr_common/             # pure-NumPy preprocessing, label maps, bundle schema (shared)
    cxr_ml/                 # data, models, train, ssl, eval, explain, export (torch)
    cxr_server/             # FastAPI app (no torch)
  configs/                  # Hydra configs: data/, model/, train/, experiment/
  scripts/                  # download_nih, prep_*, benchmark_loader, data_report, report, launch_*
  web/                      # Next.js app
  tests/                    # cross-package tests: train/serve preprocessing parity, ONNX parity
  docker/                   # Dockerfiles; docker-compose.yml at root
  docs/                     # labels.md, data_report.md, model_card, figures/, architecture.svg
  README.md  RESULTS.md  CLAUDE.md  PLAN.md
```
Package-level unit tests live in each package's `tests/`. Data, checkpoints and eval artifacts live
outside the repo under `CXR_DATA_ROOT` (default `/data/cxr`) and are gitignored.

**Where you will place licensed data** (configurable via env/YAML):
```
/data/cxr/raw/chexpert/          # CheXpert: valid/ + valid.csv, test/ + test_labels.csv
/data/cxr/raw/vindr/             # VinDr-CXR: test/*.dicom + annotations_test.csv (+ image_labels_test.csv)
/data/cxr/raw/nih/               # downloaded automatically
```
Exact expected filenames will be checked by a `scripts/check_data.py` that prints what is missing.

---

## 6. Milestones (each ends with tests, summary, limitations, and a stop for review)

| # | Milestone | Main deliverables | GPU time (est.) |
|---|---|---|---|
| 0 | Scaffolding | workspace, CLAUDE.md, CI skeleton, pre-commit | — |
| 1 | Data foundation | NIH download+prep, memmaps, splits + leakage test, labels, stats report, loader benchmark | — |
| 2 | Supervised baselines | 4 CNN runs, metrics/bootstrap/calibration module, tracking | ~3–6 h |
| 3 | MAE pretraining | MAE impl, resumable training, measured time estimate → launch ViT-S, ViT-B | ~1–2 days |
| 2b | *(while MAE trains)* Backend + placeholder bundle | FastAPI, DICOM, guard, tests | — |
| 4 | Fine-tuning study | 21-run grid, headline figure | < 1 day |
| 2c | *(parallel)* Frontend | all screens, tests, Lighthouse | — |
| 5 | External val + XAI + audit | CheXpert eval, localization, shortcut audit, RESULTS.md | a few h |
| 6 | Export + integration | ONNX bundle, parity tests, real model in app | — |
| 7 | Infra + README | Docker, compose, CI green, README, demo GIF | — |

---

## 7. Open questions (need your answer before / during Phase 0–1)

1. **ComfyUI** is holding ~33 GB VRAM and ~23 GB RAM. OK for me to stop it during training runs
   (and you restart it later)? Training with only ~13 GB of VRAM and ~5 GB of RAM is not realistic.
2. **Repo location**: `/home/ubuntu/cxr-foundation` (code) + `/data/cxr` (data, checkpoints)?
3. **MAE pretraining data**: (a) NIH train+val only (my recommendation — cleanest story), or
   (b) also add CheXpert *training* images unlabeled. (b) adds data but means CheXpert is no longer
   a fully unseen domain for the encoder, weakening the external-validation story. I'd do (a) as
   the main result and (b) only as an optional extra run.
4. **Resolution**: 224 for all comparison runs (recommended; fair and fast), plus one 512-px
   DenseNet run as the candidate served model? Higher resolution usually helps small findings
   (nodules, pneumothorax).
5. **Served model / heatmaps**: if the best model is a ViT, attention rollout is not finding-specific.
   Preference: serve the best CNN (true per-finding CAMs) even if a ViT scores slightly higher,
   and show ViT attention maps only in the research results? (My recommendation: yes.)
6. **Experiment tracking**: W&B (needs your account/API key, set in your shell — not in chat) or
   MLflow (local, no account)? I'll build both behind one interface; which should be the default?
7. **Git**: what name/email should commits use? Will this be pushed to GitHub (repo name,
   public/private)? I won't push or create remotes without your go-ahead.
8. **Disk**: OK to delete NIH tarballs after verified extraction, and VinDr DICOMs after converting
   the test set? Do you want only the VinDr **test** set (enough for localization, ~3k images)?
9. **Node**: OK to install Node 22 LTS user-locally (no sudo) for the frontend?
10. **Shortcut audit labels**: if public chest-drain annotations aren't usable, are you willing to
    label ~200–300 images in a small tool I'd build, or should I skip to a qualitative audit?

---

## 8. Risks

- **MAE may not beat ImageNet init** on 112k images. Mitigation: report honestly; label-efficiency
  regime is where gains are most likely.
- **NIH labels are noisy** (NLP-mined); test AUROC is bounded by label quality. Documented.
- **CheXpert / VinDr evaluation sets are small** → wide CIs; we avoid over-claiming.
- **Data loading on 8 CPUs** → mitigated by memmap + GPU augmentation; measured in Phase 1.
- **Disk headroom** on `/data` is ~180 GB → cleanup steps built into the prep scripts.
