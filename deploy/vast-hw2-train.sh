#!/bin/bash

set -e

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"

source /venv/main/bin/activate
cd /workspace/hw-2

output_dir=checkpoints/transformer-100m-shared-tokenizer

if [[ -f "${output_dir}/model.pt" ]]; then
  echo "Training is already complete: ${output_dir}/model.pt"
  exit 0
fi

resume_args=()
if [[ -f "${output_dir}/latest.pt" ]]; then
  resume_args=(--resume "${output_dir}/latest.pt")
fi

exec python -u train.py \
  --dataset-repo Lunfus/hw2-code-generation-train-clean \
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
  --output "${output_dir}" \
  "${resume_args[@]}"
