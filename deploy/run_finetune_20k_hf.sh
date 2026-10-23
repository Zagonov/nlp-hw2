#!/usr/bin/env bash

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

set -uo pipefail

source /venv/main/bin/activate

project_dir=/workspace/hw-2
source_checkpoint="${project_dir}/checkpoints/source/model.pt"
output_dir="${project_dir}/checkpoints/finetune-20k-lr2e5"
repo_id=Lunfus/nlp-hw2-codegen-finetune-20k
mkdir -p "${output_dir}"

upload_results() {
  train_exit_code="$1"
  trap - EXIT INT TERM
  echo "Starting final Hugging Face upload (training exit code: ${train_exit_code})"
  python -u "${project_dir}/deploy/upload_finetune_hf.py" \
    --output "${output_dir}" \
    --repo-id "${repo_id}" \
    --project-dir "${project_dir}" \
    --train-exit-code "${train_exit_code}"
  upload_exit_code=$?
  if [[ ${upload_exit_code} -ne 0 ]]; then
    echo "Hugging Face upload failed with exit code ${upload_exit_code}"
    exit "${upload_exit_code}"
  fi
  exit "${train_exit_code}"
}

on_signal() {
  echo "Fine-tuning wrapper received a termination signal"
  upload_results 143
}

trap 'upload_results $?' EXIT
trap on_signal INT TERM

cd "${project_dir}"

if [[ -f "${output_dir}/model.pt" ]]; then
  echo "Training output already exists; retrying/finalizing the Hugging Face upload"
  upload_results 0
fi

common_args=(
  --dataset-repo Lunfus/hw2-code-generation-train-clean
  --source-tokenizer shared_tokenizer.json
  --target-tokenizer shared_tokenizer.json
  --batch-size 40
  --gradient-accumulation 6
  --d-model 640
  --n-heads 10
  --d-ff 2560
  --encoder-layers 8
  --decoder-layers 8
  --max-source-length 192
  --max-target-length 256
  --learning-rate 2e-5
  --warmup-ratio 0.01
  --scheduler-steps 20000
  --validation-size 10000
  --validation-batches 100
  --validation-generation-samples 1000
  --validation-generation-beams 1
  --validate-every 1000
  --logged-generation-samples 4
  --early-stopping-patience 100
  --num-workers 2
  --compile
  --comet-project hw2-code-generation
  --comet-name finetune-20k-lr2e5-batch240
  --log-every 10
  --checkpoint-every 1000
  --archive-every 20000
  --no-upload-inference-checkpoints
  --output "${output_dir}"
)

if [[ -f "${output_dir}/latest.pt" ]]; then
  echo "Resuming interrupted fine-tune from ${output_dir}/latest.pt"
  python -u train.py \
    "${common_args[@]}" \
    --resume "${output_dir}/latest.pt" \
    --epochs 2 \
    --max-steps 68000
else
  echo "Starting a new 20k-step fine-tune from ${source_checkpoint}"
  python -u train.py \
    "${common_args[@]}" \
    --finetune-from "${source_checkpoint}" \
    --finetune-steps 20000
fi

train_exit_code=$?
upload_results "${train_exit_code}"
