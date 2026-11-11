# Reproducing the results

Everything here was run on one machine: **1× NVIDIA L40S (46 GB), 8 CPU cores, 31 GB RAM**,
about 300 GB of disk. All steps are resumable: a finished run writes a `DONE` marker and is
skipped when re-run.

## 0. Setup

```bash
uv sync                                  # Python 3.11 workspace, pinned by uv.lock
cp .env.example .env                     # paths; defaults put data under /data/cxr
```

## 1. Data

| Dataset | How | Where |
|---|---|---|
| NIH ChestX-ray14 | `uv run scripts/download_nih.py --out /data/cxr/raw/nih` (Hugging Face mirror pinned to a revision, SHA-256 verified; ~45 GB) | `raw/nih/` |
| NEATX chest-drain labels | downloaded with the NIH data (CC BY-NC-SA, not redistributed) | `raw/neatx/` |
| CheXpert validation images | Manual: Stanford AIMI on Redivis, dataset **CheXpert Plus**, table `PNG_valid` (requires accepting the Stanford research agreement) | `raw/chexpert/valid/patient*/study*/view*_frontal.png` |
| CheXlocalize segmentations | Manual: Stanford AIMI on Redivis, dataset **CheXlocalize**, the nine root JSON files (skip `gradcam_maps_val/`) | `raw/chexlocalize/gt_segmentations_val.json` etc. |
| VinDr-CXR (optional) | Manual: PhysioNet credentialed access | `raw/vindr/` (not used in the reported results) |

`uv run scripts/check_data.py` lists exactly what is present and what is missing.

The CheXpert Plus release has no radiologist label CSV, so labels are derived from the CheXlocalize
ground-truth masks: a finding is positive if and only if radiologists segmented it. This
reproduces the published validation-set positive counts.

```bash
uv run scripts/prep_nih.py --subset-seeds 0 1 2   # index, patient splits, memmaps (~5 min)
uv run scripts/prep_external.py                    # CheXpert valid + masks
uv run scripts/data_report.py                      # docs/data_report.md
```

## 2. Training

```bash
scripts/run_queue.sh configs/queues/mae.txt /data/cxr/logs/mae   # MAE ViT-B/16, 800 epochs
scripts/launch_baselines.sh                                       # LR sweep -> 5 CNN baselines
scripts/launch_finetune.sh random
scripts/launch_finetune.sh imagenet
scripts/launch_finetune.sh ssl                                     # needs the MAE encoder
```

Each `launch_finetune.sh` runs three things:
1. A learning-rate sweep at 10% labels, scored on validation only.
2. The grid: 1% and 10% labels × 3 seeds (each seed is an independent patient subset), plus 100%
   labels.
3. Evaluation of every run.

Measured wall-clock times from the logs are below. "Shared" means another job was on the GPU at
the same time.

| Job | Time |
|---|---|
| MAE ViT-B, 800 epochs | 20.7 h (shared for ~10 h; about 13 h alone at ~1,410 img/s) |
| DenseNet-121, 224 px, ImageNet init (100% labels) | 34 min (shared) |
| DenseNet-121, 512 px, ImageNet init | 3.1 h (shared) |
| ViT-B fine-tune, 100% labels | 12.6 min (alone) / 33–68 min (shared) |
| ViT-B fine-tune, 10% labels | 5.1 min (alone) / 10–25 min (shared) |
| ViT-B fine-tune, 1% labels | ~2 min (alone) / 3–8 min (shared) |
| Evaluating 7 runs (bootstrap CIs) | 1.7 min |

## 3. Analysis, report, bundle

```bash
scripts/analysis.sh
```

This runs the following, skipping anything already done:
1. CheXpert external validation for every 100%-label run.
2. Heatmap localization: CAM for all runs, plus attention rollout for ViTs, against NIH boxes and
   CheXlocalize masks.
3. The chest-drain shortcut audit.
4. Failure galleries.
5. Validation-only choice of the served model, with ONNX bundle export.
6. `RESULTS.md`, the README results block, `web/src/content/results.json` and
   `docs/MODEL_CARD.md`.

No number in these files is typed by hand.

## 4. App and demo

```bash
CXR_BUNDLE_HOST_DIR=/data/cxr/artifacts/bundles/current docker compose up --build -d
cd web && E2E_BASE_URL=http://localhost:3000 npx playwright test   # 18 end-to-end + a11y tests
node scripts/record-demo.mjs                                       # docs/demo.mp4 + docs/demo.gif
```

## Determinism notes

- Seeds are fixed for data subsets, model initialization, augmentation and bootstrap resampling.
- cuDNN kernels and bf16 autocast are not bit-deterministic, so re-runs agree to within normal
  run-to-run noise, not exactly. The 1% and 10% results average three independent patient
  subsets for this reason.
- Every checkpoint records the git commit, config hash and split digest (`provenance.json`).
