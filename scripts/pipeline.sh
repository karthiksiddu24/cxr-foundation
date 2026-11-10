#!/usr/bin/env bash
# Orchestrates the remaining GPU work as two streams (each stream is sequential).
#   A: CNN baselines (launched separately) -> evaluate them -> ViT random & ImageNet grids
#   B: wait for MAE ViT-B -> ViT MAE-init grid -> optional MAE ViT-S
# Every step is restartable: finished runs are skipped via their DONE markers.
set -uo pipefail
cd "$(dirname "$0")/.."
stream_a() {
  while pgrep -f "[l]aunch_baselines.sh" > /dev/null; do sleep 120; done
  uv run python -m cxr_ml.eval.evaluate sup_densenet121_imagenet_224 sup_resnet50_imagenet_224 \
    sup_densenet121_random_224 sup_resnet50_random_224 sup_densenet121_imagenet_512 \
    > /data/cxr/logs/eval_baselines.log 2>&1
  scripts/launch_finetune.sh random
  scripts/launch_finetune.sh imagenet
}
stream_b() {
  scripts/launch_finetune.sh ssl /data/cxr/runs/mae_vit_base_ep800/DONE
  echo "uv run python -m cxr_ml.train.mae --config configs/mae/vit_small.yaml name=mae_vit_small_ep800" \
    > configs/queues/mae_small.txt
  scripts/run_queue.sh configs/queues/mae_small.txt /data/cxr/logs/mae
}
case ${1:-all} in
  a) stream_a ;;
  b) stream_b ;;
  all) stream_a > /data/cxr/logs/stream_a.log 2>&1 & stream_b > /data/cxr/logs/stream_b.log 2>&1 & wait ;;
esac
