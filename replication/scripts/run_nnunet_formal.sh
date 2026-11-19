#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 FOLD SEED [DEVICE]" >&2
  exit 2
fi

FOLD="$1"
SEED="$2"
DEVICE="${3:-cuda}"

if [[ ! "$FOLD" =~ ^[0-2]$ ]]; then
  echo "FOLD must be 0, 1, or 2." >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "scripts/nnunet_env.sh"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-4}"

ARGS=(
  --fold "$FOLD"
  --seed "$SEED"
  --device "$DEVICE"
  --data-processes "$nnUNet_n_proc_DA"
  --save-probabilities
)

if [[ "${RESUME:-0}" == "1" ]]; then
  ARGS+=(--resume)
fi

exec ".venv/bin/python" -u "scripts/run_nnunet_seeded.py" "${ARGS[@]}"
