#!/usr/bin/env bash

set -e

source .venv/bin/activate

python train.py \
  --dataset data/train_clean \
  --source-tokenizer source_tokenizer.json \
  --target-tokenizer target_tokenizer.json \
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
  --learning-rate 5e-5 \
  --warmup-ratio 0.05 \
  --validation-size 10000 \
  --validation-batches 100 \
  --validate-every 1000 \
  --logged-generation-samples 4 \
  --early-stopping-patience 3 \
  --num-workers 2 \
  --comet-project hw2-code-generation \
  --comet-name transformer-100m-lr5e5 \
  --log-every 10 \
  --checkpoint-every 1000 \
  --archive-every 10000 \
  --output checkpoints/transformer-100m-lr5e5 \
  "$@"
