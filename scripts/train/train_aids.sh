#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export CUDA_VISIBLE_DEVICES="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

exec "${PYTHON_BIN:-python}" -B -u "$PROJECT_DIR/src/train.py" \
  --dataset AIDS700nef \
  --dataset-root "$PROJECT_DIR/datasets/AIDS700nef" \
  --device cuda:0 \
  --anchor-count 16 \
  --anchor-dim 32 \
  --seed 1 \
  --epochs 30000 \
  --validation-interval 10 \
  --batch-size 128 \
  --learning-rate 0.001 \
  --weight-decay 0.0 \
  --residual-alpha 0.1 \
  --attention-heads 4 \
  --ffn-ratio 2 \
  --layer-scale-init 0.1 \
  --dropout 0.0 \
  --attention-dropout 0.0 \
  --output-dir "$PROJECT_DIR/runs/AIDS700nef" \
  "$@"
