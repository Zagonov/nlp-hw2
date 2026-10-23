#!/usr/bin/env bash
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"
set -euo pipefail
source /venv/main/bin/activate
cd /workspace/hw-2
exec python -u train.py \
  --dataset-repo Lunfus/hw2-code-generation-train-clean \
  --source-tokenizer shared_tokenizer.json --target-tokenizer shared_tokenizer.json \
  --finetune-from checkpoints/source/inference-best.pt --finetune-steps 3000 \
  --output checkpoints/finetune-best-opweight2-3k \
  --d-model 640 --n-heads 10 --d-ff 2560 --encoder-layers 8 --decoder-layers 8 \
  --max-source-length 192 --max-target-length 256 \
  --batch-size 40 --gradient-accumulation 6 \
  --learning-rate 2e-5 --warmup-ratio 0.01 --scheduler-steps 3000 \
  --arithmetic-operator-weight 2 --data-seed 42 --seed 42 \
  --validation-size 10000 --validation-batches 100 \
  --validation-generation-samples 1000 --validation-generation-beams 1 \
  --validate-every 500 --logged-generation-samples 4 \
  --early-stopping-patience 100 --checkpoint-every 1000 --archive-every 10000 \
  --num-workers 2 --compile --log-every 10 \
  --new-comet-run --comet-project hw2-code-generation \
  --comet-name finetune-640-best40000-opweight2-lr2e5-3k \
  --no-upload-inference-checkpoints
