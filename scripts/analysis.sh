#!/usr/bin/env bash
# Post-training analysis on the evaluated runs, then the report.
#   scripts/analysis.sh
# Runs (all read saved checkpoints/predictions; the test split is only *scored*, never used to choose):
#   - CheXpert external validation for every 100%-label run
#   - heatmap localization (CAM; plus attention rollout for ViTs) on NIH boxes and CheXlocalize masks
#   - chest-drain shortcut audit
#   - error galleries for the best baseline and the served model
#   - select_served.py (validation-only choice) -> bundle export
#   - report.py -> RESULTS.md, README results block, web/src/content/results.json
set -euo pipefail
cd "$(dirname "$0")/.."
EV=${CXR_DATA_ROOT:-/data/cxr}/artifacts/eval
PY="uv run --no-sync python"

mapfile -t RUNS < <(ls "$EV" | grep -E '^(sup_|ft_vitb_.*_f100_)' | sort)
echo "100%-label runs: ${RUNS[*]}"

need() { [[ ! -e "$EV/$1/$2" ]]; }

for r in "${RUNS[@]}"; do
  need "$r" external_chexpert_valid.json && $PY -m cxr_ml.eval.external "$r"
  need "$r" shortcut_pneumothorax_drain.json && $PY -m cxr_ml.eval.shortcut "$r"
  for d in nih chexlocalize; do
    need "$r" "localization_${d}_cam.json" && $PY -m cxr_ml.explain.localization "$r" --dataset "$d"
    if [[ $r == ft_vitb_* ]] && need "$r" "localization_${d}_rollout.json"; then
      $PY -m cxr_ml.explain.localization "$r" --dataset "$d" --method rollout
    fi
  done
done

$PY scripts/select_served.py
SERVED=$(basename "$(readlink -f "${CXR_DATA_ROOT:-/data/cxr}/artifacts/bundles/current")")
BEST_BASE=$(for r in "${RUNS[@]}"; do [[ $r == sup_* ]] && echo "$r"; done | while read -r r; do
  python3 -c "import json,sys; print(json.load(open('$EV/$r/metrics.json'))['val_mean_auroc'], '$r')"
done | sort -rn | head -1 | cut -d' ' -f2)
for r in $(printf '%s\n' "$BEST_BASE" "$SERVED" | sort -u); do
  ls docs/figures/galleries/"$r"__* > /dev/null 2>&1 || uv run --no-sync scripts/galleries.py "$r"
done
uv run --no-sync scripts/report.py
uv run --no-sync scripts/model_card.py
echo "served: $SERVED   best baseline: $BEST_BASE"
