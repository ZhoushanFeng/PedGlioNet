#!/usr/bin/env bash
# P1 joint ET recipe v2: the proven two-stage schedule, replacing the
# collapsed single-stage v1 runs (segmentation loss stuck at Tversky~0).
#   Stage A: positives repeated 2x, no empty-target penalty, damped
#            presence loss -> escapes the all-background optimum.
#   Stage B: initialise from stage A, low LR, empty-target penalty on,
#            full presence loss -> restores ET-negative specificity and
#            calibrates the presence head.
# Development-only: uses fit+calibration cases of each outer fold; outer
# held-out sources remain untouched.
# Three lanes run in parallel (one seed per lane, one GPU per lane).
set -uo pipefail

LANE_GPUS=(${LANE_GPUS:-1 2 3})
SEEDS=(2026 2027 2028)
FOLDS=(0 1 2)
VRAM_MIN_MB="${VRAM_MIN_MB:-3000}"
VRAM_WAIT_SECONDS="${VRAM_WAIT_SECONDS:-600}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"

STAGE_A_ARGS=(--positive-repeat 2 --empty-target-weight 0
  --presence-loss-weight 0.3 --epochs 25 --learning-rate 1e-3)
STAGE_B_ARGS=(--positive-repeat 1 --empty-target-weight 2.0
  --presence-loss-weight 1.0 --epochs 15 --learning-rate 2e-4)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p results/p1 results/checkpoints logs
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

gpu_free_mb() {
  local free
  free="$(nvidia-smi -i "$1" --query-gpu=memory.free \
    --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
  [[ "$free" =~ ^[0-9]+$ ]] && echo "$free" || echo 0
}

wait_for_vram() {
  local gpu="$1"
  while (( $(gpu_free_mb "$gpu") < VRAM_MIN_MB )); do
    echo "[lane gpu$gpu] waiting for ${VRAM_MIN_MB}MB free VRAM"
    sleep "$VRAM_WAIT_SECONDS"
  done
}

run_stage() {
  local gpu="$1" fold="$2" seed="$3" stage="$4" log="$5"
  shift 5
  CUDA_VISIBLE_DEVICES="$gpu" nice -n 5 \
    .venv/bin/python -u scripts/joint_et_refiner.py \
    --outer-fold "$fold" \
    --seed "$seed" \
    --device cuda \
    --threads 4 \
    --inter-step-sleep-ms 0 \
    "$@" \
    >> "$log" 2>&1
}

lane() {
  local gpu="$1" seed="$2"
  local fold tag final stage_a_json ckpt_a ckpt_b log attempt ok
  for fold in "${FOLDS[@]}"; do
    tag="v2_fold${fold}_seed${seed}"
    final="results/p1/joint_et_${tag}.json"
    if [[ -f "$final" ]] && grep -q '"status": "completed' "$final"; then
      echo "[lane gpu$gpu] skip $tag (already completed)"
      continue
    fi
    stage_a_json="results/p1/joint_et_${tag}_stageA.json"
    ckpt_a="results/checkpoints/joint_et_${tag}/stageA"
    ckpt_b="results/checkpoints/joint_et_${tag}/stageB"
    log="logs/p1_${tag}.log"
    ok=0
    for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
      wait_for_vram "$gpu"
      echo "[lane gpu$gpu] $tag stage A (attempt $attempt)"
      if run_stage "$gpu" "$fold" "$seed" A "$log" \
        "${STAGE_A_ARGS[@]}" \
        --output "$stage_a_json" \
        --status-output "results/p1/joint_et_${tag}_stageA_status.json" \
        --checkpoint-dir "$ckpt_a"; then
        ok=1
        break
      fi
      echo "[lane gpu$gpu] $tag stage A FAILED (see $log)" >&2
      sleep "$VRAM_WAIT_SECONDS"
    done
    if (( !ok )); then
      echo "[lane gpu$gpu] $tag giving up on stage A" >&2
      continue
    fi
    ok=0
    for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
      wait_for_vram "$gpu"
      echo "[lane gpu$gpu] $tag stage B (attempt $attempt)"
      if run_stage "$gpu" "$fold" "$seed" B "$log" \
        "${STAGE_B_ARGS[@]}" \
        --initial-model "$ckpt_a/joint_et_best.pt" \
        --output "$final" \
        --status-output "results/p1/joint_et_${tag}_status.json" \
        --checkpoint-dir "$ckpt_b"; then
        ok=1
        break
      fi
      echo "[lane gpu$gpu] $tag stage B FAILED (see $log)" >&2
      sleep "$VRAM_WAIT_SECONDS"
    done
    if (( ok )); then
      echo "[lane gpu$gpu] $tag done"
    else
      echo "[lane gpu$gpu] $tag giving up on stage B" >&2
    fi
  done
}

PIDS=()
for index in "${!SEEDS[@]}"; do
  gpu="${LANE_GPUS[$((index % ${#LANE_GPUS[@]}))]}"
  lane "$gpu" "${SEEDS[$index]}" &
  PIDS+=("$!")
done

STATUS=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || STATUS=1
done

COMPLETED="$(grep -l '"status": "completed' \
  results/p1/joint_et_v2_fold*_seed*[0-9].json 2>/dev/null | wc -l | tr -d ' ')"
echo "P1_V2_COMPLETED=${COMPLETED}/9"
if [[ "$COMPLETED" == "9" ]]; then
  echo "ALL_P1_V2_DONE"
else
  exit 1
fi
