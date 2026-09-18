#!/bin/bash
set -euo pipefail

# Waymo rds_hq trajectory eval runner (mADE/mFDE/ADE/FDE @1s/3s/5s/full).
#
# Usage:
#   bash scripts/run_waymo_eval.sh              # full val split, 4 GPUs x bs 4
#   bash scripts/run_waymo_eval.sh 20           # first 20 clips (small batch)
#   bash scripts/run_waymo_eval.sh 20 --num-traj-samples 3
#   NPROC=8 BATCH_SIZE=8 bash scripts/run_waymo_eval.sh   # custom parallelism
#
# Optional positional arg: number of clips to evaluate (maps to --max-clips).
# Everything after it is passed through to waymo_eval/eval_traj.py
# (including --batch-size, which overrides BATCH_SIZE).
# NPROC (default 4) = torchrun processes/GPUs; BATCH_SIZE (default 4) =
# samples per GPU forward. Output: waymo_eval/eval_outputs/<split>_<tag>/
# (per_frame.csv, summary.json, preds.npz) with a run.log. preds.npz is the
# source of truth for offline metric recompute (compute_metrics.py).
# Visualize afterwards with scripts/run_waymo_viz.sh.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

VENV_DIR="${PROJECT_ROOT}/../alpamayo/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: venv not found at $VENV_DIR" >&2
    exit 1
fi
PY="$VENV_DIR/bin/python"

# Multi-process (torchrun) uses all visible GPUs unless CUDA_VISIBLE_DEVICES
# is already set; single-process defaults to GPU 0.
NPROC="${NPROC:-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
if [ "$NPROC" -le 1 ] && [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=0
fi

# --- parse: [N_CLIPS] [extra args...] ---
N_CLIPS=""
EXTRA=("$@")
if [ $# -ge 1 ] && [[ "$1" =~ ^[0-9]+$ ]]; then
    N_CLIPS=$1
    EXTRA=("${@:2}")
fi

# --- resolve split / out-dir from the passthrough args (defaults: val) ---
SPLIT="val"
TAG_VAL=""
OUT_VAL=""
for ((i = 0; i < ${#EXTRA[@]}; i++)); do
    case "${EXTRA[$i]}" in
        --split)   SPLIT="${EXTRA[$((i + 1))]:-val}" ;;
        --tag)     TAG_VAL="${EXTRA[$((i + 1))]:-}" ;;
        --out-dir) OUT_VAL="${EXTRA[$((i + 1))]:-}" ;;
    esac
done

ARGS=()
if [ -n "$N_CLIPS" ]; then
    ARGS+=(--max-clips "$N_CLIPS")
fi
if [ -z "$TAG_VAL" ] && [ -z "$OUT_VAL" ]; then
    SCOPE="full"
    [ -n "$N_CLIPS" ] && SCOPE="c${N_CLIPS}"
    TAG_VAL="${SCOPE}_$(date +%m%d_%H%M)"
    ARGS+=(--tag "$TAG_VAL")
fi

if [ -n "$OUT_VAL" ]; then
    FINAL_OUT="$OUT_VAL"
elif [ -n "$TAG_VAL" ]; then
    FINAL_OUT="waymo_eval/eval_outputs/${SPLIT}_${TAG_VAL}"
else
    FINAL_OUT="waymo_eval/eval_outputs/${SPLIT}"
fi
mkdir -p "$FINAL_OUT"
LOG="$FINAL_OUT/run.log"

echo "[eval] split=${SPLIT} clips=$([ -n "$N_CLIPS" ] && echo "$N_CLIPS" || echo all) nproc=${NPROC} batch_size=${BATCH_SIZE}"
echo "[eval] out_dir=${FINAL_OUT}"
if [ ${#EXTRA[@]} -gt 0 ]; then
    echo "[eval] extra args: ${EXTRA[*]}"
fi

if [ "$NPROC" -gt 1 ]; then
    echo "[eval] launching $NPROC processes via torchrun"
    "$PY" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
        waymo_eval/eval_traj.py --split "$SPLIT" --batch-size "$BATCH_SIZE" \
        "${ARGS[@]}" "${EXTRA[@]}" 2>&1 | tee "$LOG"
else
    "$PY" waymo_eval/eval_traj.py --split "$SPLIT" --batch-size "$BATCH_SIZE" \
        "${ARGS[@]}" "${EXTRA[@]}" 2>&1 | tee "$LOG"
fi

echo
echo "[eval] done. summary: ${FINAL_OUT}/summary.json"
echo "[eval] visualize with: bash scripts/run_waymo_viz.sh ${FINAL_OUT}"
