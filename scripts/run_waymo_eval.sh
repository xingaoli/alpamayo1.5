#!/bin/bash
set -euo pipefail

# Waymo rds_hq trajectory eval runner (mADE/mFDE/ADE/FDE @1s/3s/5s/full).
#
# Usage:
#   bash scripts/run_waymo_eval.sh              # full val split (~2424 samples, ~1.2h)
#   bash scripts/run_waymo_eval.sh 20           # first 20 clips (small batch)
#   bash scripts/run_waymo_eval.sh 20 --num-traj-samples 3
#   bash scripts/run_waymo_eval.sh -- --split train --num-traj-samples 3
#
# Optional positional arg: number of clips to evaluate (maps to --max-clips).
# Everything after it is passed through to waymo_eval/eval_traj.py.
# Output: waymo_eval/eval_outputs/<split>_<tag>/ (per_frame.csv, summary.json,
# preds.npz) with a run.log. Visualize afterwards with scripts/run_waymo_viz.sh.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

VENV_DIR="${PROJECT_ROOT}/../alpamayo/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: venv not found at $VENV_DIR" >&2
    exit 1
fi
PY="$VENV_DIR/bin/python"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

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

echo "[eval] split=${SPLIT} clips=$([ -n "$N_CLIPS" ] && echo "$N_CLIPS" || echo all)"
echo "[eval] out_dir=${FINAL_OUT}"
if [ ${#EXTRA[@]} -gt 0 ]; then
    echo "[eval] extra args: ${EXTRA[*]}"
fi

"$PY" waymo_eval/eval_traj.py --split "$SPLIT" "${ARGS[@]}" "${EXTRA[@]}" 2>&1 | tee "$LOG"

echo
echo "[eval] done. summary: ${FINAL_OUT}/summary.json"
echo "[eval] visualize with: bash scripts/run_waymo_viz.sh ${FINAL_OUT}"
