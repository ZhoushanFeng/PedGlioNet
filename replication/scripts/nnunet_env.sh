#!/usr/bin/env bash

if [[ -n "${ZSH_VERSION:-}" ]]; then
  SCRIPT_PATH="${(%):-%N}"
else
  SCRIPT_PATH="${BASH_SOURCE[0]}"
fi

ROOT="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"

export nnUNet_raw="$ROOT/data/nnunet/raw"
export nnUNet_preprocessed="$ROOT/data/nnunet/preprocessed"
export nnUNet_results="$ROOT/data/nnunet/results"

unset SCRIPT_PATH
