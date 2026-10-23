#!/usr/bin/env bash

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

set -uo pipefail
source /venv/main/bin/activate
cd /workspace/hw-2

output_dir=checkpoints/wide832-8x8-b240-lr175e4-5333

exec python -u train.py \
  --dataset-repo Lunfus/hw2-code-generation-train-clean \
  --source-tokenizer shared_tokenizer.json \
  --target-tokenizer shared_tokenizer.json \
  --batch-size 40 \
  --gradient-accumulation 6 \
  --d-model 832 \
  --n-heads 13 \
  --d-ff 3328 \
  --encoder-layers 8 \
  --decoder-layers 8 \
  --max-source-length 192 \
  --max-target-length 256 \
  --epochs 1 \
  --max-steps 10666 \
  --scheduler-steps 49283 \
  --learning-rate 1.75e-4 \
  --warmup-steps 493 \
  --validation-size 10000 \
  --validation-batches 100 \
  --validation-generation-samples 1000 \
  --validation-generation-beams 1 \
  --validate-every 667 \
  --logged-generation-samples 4 \
  --early-stopping-patience 100 \
  --num-workers 2 \
  --compile \
  --resume "${output_dir}/model.pt" \
  --extend-resume \
  --comet-project hw2-code-generation \
  --comet-name wide832-8x8-b240-lr175e4-continued-10666 \
  --log-every 10 \
  --checkpoint-every 1000 \
  --archive-every 8000 \
  --no-upload-inference-checkpoints \
  --output "${output_dir}"
