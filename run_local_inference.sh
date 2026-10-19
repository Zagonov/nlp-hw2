#!/usr/bin/env bash

set -e

cd "$(dirname "$0")"
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1

python local_inference.py "$@"
