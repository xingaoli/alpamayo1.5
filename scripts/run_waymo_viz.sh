#!/bin/bash
set -euo pipefail

# Offline visualization of stored eval results (preds.npz from
# run_waymo_eval.sh / waymo_eval/eval_traj.py). No model, no GPU.
#
# Usage:
#   bash scripts/run_waymo_viz.sh                          # newest eval output,
#                                                          #   worst-mADE@3s clip
#   bash scripts/run_waymo_viz.sh waymo_eval/eval_outputs/val_c20_0115_1234
#   bash scripts/run_waymo_viz.sh <eval_dir> --clip-id <seg> [--segment-idx 5]
#   bash scripts/run_waymo_viz.sh <eval_dir> --clip-id <seg> --video
#
# Arg 1 (optional): eval output dir (must contain preds.npz); defaults to the
# most recent one under waymo_eval/eval_outputs/. Remaining args are passed to
# waymo_eval/viz_traj.py (offline --from-eval mode).
# With no clip selector given, the clip with the worst mean mADE@3s in
# per_frame.csv is picked automatically. Figures are grouped per eval run:
# waymo_eval/viz_outputs/<eval-run-name>/ (or a custom --out-dir).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

VENV_DIR="${PROJECT_ROOT}/../alpamayo/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: venv not found at $VENV_DIR" >&2
    exit 1
fi
PY="$VENV_DIR/bin/python"

EXTRA=("$@")
EVAL_DIR=""
if [ $# -ge 1 ] && [ -d "$1" ]; then
    EVAL_DIR=$1
    EXTRA=("${@:2}")
else
    EVAL_DIR="$(find waymo_eval/eval_outputs -name preds.npz -printf '%T@ %h\n' 2>/dev/null \
        | sort -rn | head -1 | cut -d' ' -f2)"
    if [ -z "$EVAL_DIR" ]; then
        echo "ERROR: no eval output with preds.npz under waymo_eval/eval_outputs/" >&2
        echo "       run scripts/run_waymo_eval.sh first, or pass the dir as arg 1" >&2
        exit 1
    fi
fi
EVAL_DIR="$(cd "$EVAL_DIR" && pwd)"

# clip selector already given? then pass everything through untouched
HAS_SELECTOR=0
for a in "${EXTRA[@]:-}"; do
    case "$a" in --clip-id|--clip-index|--t0-us) HAS_SELECTOR=1 ;; esac
done

CLIP_ARGS=()
if [ "$HAS_SELECTOR" -eq 0 ]; then
    WORST="$("$PY" - "$EVAL_DIR/per_frame.csv" <<'EOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
by_clip = {}
for r in rows:
    by_clip.setdefault(r["clip_id"], []).append(float(r["mADE@3s"]))
worst = max(by_clip, key=lambda c: sum(by_clip[c]) / len(by_clip[c]))
print(worst)
EOF
)"
    echo "[viz] no clip selected -> worst mean mADE@3s clip: ${WORST}"
    CLIP_ARGS=(--clip-id "$WORST")
fi

echo "[viz] eval_dir=${EVAL_DIR}"
if [ ${#EXTRA[@]} -gt 0 ]; then
    echo "[viz] extra args: ${EXTRA[*]}"
fi

"$PY" waymo_eval/viz_traj.py --from-eval "$EVAL_DIR" "${CLIP_ARGS[@]}" "${EXTRA[@]}"
