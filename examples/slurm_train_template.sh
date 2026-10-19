#!/usr/bin/env bash
#SBATCH --job-name=topofusion_train
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
export PROJECT_ROOT="/path/to/topofusion"
export DATA_ROOT="/path/to/deidentified/data"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

python -m topofusion_v3.multitask.train_multitask \
  --config "${PROJECT_ROOT}/configs/topofusion_mt.yaml" \
  --manifest "${DATA_ROOT}/manifests/train_manifest.csv" \
  --output-dir "${PROJECT_ROOT}/runs/example_train"
