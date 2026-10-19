#!/usr/bin/env bash

set -e

source .venv/bin/activate

python train.py \
  --dataset data/train_clean \
  --source-tokenizer shared_tokenizer.json \
  --target-tokenizer shared_tokenizer.json \
  --batch-size 80 \
  --gradient-accumulation 2 \
  --d-model 640 \
  --n-heads 10 \
  --d-ff 2560 \
  --encoder-layers 8 \
  --decoder-layers 8 \
  --max-source-length 192 \
  --max-target-length 256 \
  --epochs 1 \
  --learning-rate 1e-4 \
  --warmup-ratio 0.05 \
  --validation-size 10000 \
  --validation-batches 100 \
  --validation-generation-samples 1000 \
  --validate-every 1000 \
  --logged-generation-samples 4 \
  --early-stopping-patience 8 \
  --num-workers 2 \
  --compile \
  --comet-project hw2-code-generation \
  --comet-name transformer-100m-shared-tokenizer \
  --log-every 10 \
  --checkpoint-every 1000 \
  --archive-every 10000 \
  --output checkpoints/transformer-100m-shared-tokenizer \
  "$@"
