#!/usr/bin/env bash
# Phase 2: after the LR sweep finishes, train the 4 CNN baselines at 100% labels with the
# val-selected LRs, then a 512px DenseNet-121 (ImageNet init) as the candidate served model.
set -euo pipefail
cd "$(dirname "$0")/.."
while pgrep -f "[r]un_queue.sh configs/queues/sup.txt" > /dev/null; do sleep 60; done
LR_IM=$(uv run scripts/select_lr.py lrsweep_densenet121_imagenet_lr)
LR_RA=$(uv run scripts/select_lr.py lrsweep_densenet121_random_lr)
echo "selected lr imagenet=$LR_IM random=$LR_RA" | tee configs/queues/selected_lr.txt
B="uv run python -m cxr_ml.train.classify --config configs/sup/base.yaml"
cat > configs/queues/sup100.txt <<Q
$B model.arch=densenet121 model.init=imagenet optim.lr=$LR_IM name=sup_densenet121_imagenet_224
$B model.arch=resnet50 model.init=imagenet optim.lr=$LR_IM name=sup_resnet50_imagenet_224
$B model.arch=densenet121 model.init=random optim.lr=$LR_RA schedule.epochs=40 name=sup_densenet121_random_224
$B model.arch=resnet50 model.init=random optim.lr=$LR_RA schedule.epochs=40 name=sup_resnet50_random_224
$B model.arch=densenet121 model.init=imagenet model.img_size=512 data.size=512 data.batch_size=48 data.eval_batch_size=96 optim.lr=$LR_IM schedule.epochs=20 name=sup_densenet121_imagenet_512
Q
exec scripts/run_queue.sh configs/queues/sup100.txt /data/cxr/logs/sup
