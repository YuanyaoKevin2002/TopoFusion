#!/usr/bin/env bash
set -euo pipefail

export PROJECT_ROOT="/path/to/topofusion"
export DATA_ROOT="/path/to/deidentified/data"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

python -m topofusion_v3.multitask.train_multitask \
  --config "${PROJECT_ROOT}/configs/topofusion_mt.yaml" \
  --manifest "${DATA_ROOT}/manifests/train_manifest.csv" \
  --output-dir "${PROJECT_ROOT}/runs/example_train"
