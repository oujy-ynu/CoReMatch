#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export CUDA_VISIBLE_DEVICES="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

exec "${PYTHON_BIN:-python}" -B -u "$PROJECT_DIR/src/train.py" \
  --dataset IMDBMulti \
  --evaluate-only \
  --dataset-root "$PROJECT_DIR/datasets/IMDBMulti" \
  --device cuda:0 \
  --anchor-count 6 \
  --anchor-dim 16 \
  --seed 2 \
  --batch-size 128 \
  --residual-alpha 0.1 \
  --attention-heads 4 \
  --ffn-ratio 2 \
  --layer-scale-init 0.1 \
  --dropout 0.0 \
  --attention-dropout 0.0 \
  --checkpoint "$PROJECT_DIR/checkpoints/IMDBMulti/best_mse.pt" \
  "$@"
