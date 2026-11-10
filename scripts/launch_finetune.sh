#!/usr/bin/env bash
# Phase 4 for one initialization:  scripts/launch_finetune.sh <random|imagenet|ssl> [wait-for-file]
#  1. LR sweep at 10% labels (seed 0, validation only, test never evaluated)
#  2. select LR by val mean AUROC
#  3. grid: 1% x 3 seeds, 10% x 3 seeds, 100% x 1 seed (each seed = independent patient subset)
#  4. evaluate every grid run (patient-bootstrap CIs)
set -euo pipefail
cd "$(dirname "$0")/.."
INIT=$1
WAIT_FOR=${2:-}
if [[ -n "$WAIT_FOR" ]]; then while [[ ! -e "$WAIT_FOR" ]]; do sleep 120; done; fi
B="uv run python -m cxr_ml.train.classify --config configs/ft/vit_base.yaml"
EXTRA="model.init=$INIT"
case $INIT in
  random) ;;
  imagenet) EXTRA="$EXTRA optim.layer_decay=0.75" ;;
  ssl) EXTRA="$EXTRA optim.layer_decay=0.75 model.ssl_checkpoint=${SSL_CKPT:-/data/cxr/runs/mae_vit_base_ep800/encoder.pt}" ;;
  *) echo "unknown init $INIT"; exit 2 ;;
esac
Q=configs/queues/ft_${INIT}.txt
: > "$Q"
for LR in 1e-4 3e-4 1e-3; do
  echo "$B --config configs/ft/frac_010.yaml $EXTRA optim.lr=$LR eval_test=false name=ftsweep_vitb_${INIT}_lr$LR" >> "$Q"
done
scripts/run_queue.sh "$Q" /data/cxr/logs/ft
LR=$(uv run scripts/select_lr.py "ftsweep_vitb_${INIT}_lr")
echo "$INIT selected lr=$LR" >> configs/queues/selected_lr.txt
G=configs/queues/ft_${INIT}_grid.txt
: > "$G"
RUNS=()
for S in 0 1 2; do
  for F in 001 010; do
    N=ft_vitb_${INIT}_f${F}_s$S
    echo "$B --config configs/ft/frac_$F.yaml $EXTRA optim.lr=$LR seed=$S data.subset_seed=$S name=$N" >> "$G"
    RUNS+=("$N")
  done
done
N=ft_vitb_${INIT}_f100_s0
echo "$B --config configs/ft/frac_100.yaml $EXTRA optim.lr=$LR seed=0 name=$N" >> "$G"
RUNS+=("$N")
echo "uv run python -m cxr_ml.eval.evaluate ${RUNS[*]}  # name=eval_ft_${INIT}" >> "$G"
scripts/run_queue.sh "$G" /data/cxr/logs/ft
