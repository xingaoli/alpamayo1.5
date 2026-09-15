#!/bin/bash
set -euo pipefail

# Batch VQA runner: run vqa_inference.py on every wfm_gen_data clip whose
# labels/<clip_id>.json id starts with a given prefix, with one prompt.
#
# Usage:
#   bash scripts/run_vqa_prefix.sh <clip_prefix> [extra args for vqa_inference.py]
#
# Examples:
#   bash scripts/run_vqa_prefix.sh fire-seed1
#   bash scripts/run_vqa_prefix.sh fire-seed1 --prompt "Describe the scene."
#   bash scripts/run_vqa_prefix.sh fire-seed1 --prompt "Any hazards ahead?" --t0-seconds 3.0
#   bash scripts/run_vqa_prefix.sh fire-seed1 --out-dir /tmp/vqa_out
#
# Note: "fire-seed1" also matches fire-seed10000_*, etc. (plain prefix match).
# --prompt is an alias of vqa_inference.py's --question. Outputs default to
#   wfm_gen_test/vqa_test/ar1_5_wfm_outputs/<prefix>_<YYYYmmdd_HHMMSS>/
# with a run.log capturing the full stdout/stderr.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="${PROJECT_ROOT}/src"

# Activate the sibling project's venv.
VENV_DIR="${PROJECT_ROOT}/../alpamayo/.venv"
if [ -d "$VENV_DIR" ]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
else
    echo "ERROR: venv not found at $VENV_DIR" >&2
    exit 1
fi

if [ $# -lt 1 ]; then
    echo "Usage: $0 <clip_prefix> [extra args for vqa_inference.py]" >&2
    echo "  e.g. $0 fire-seed1 --prompt \"Describe the scene.\"" >&2
    exit 1
fi

PREFIX=$1
shift

OUT_DIR="${PROJECT_ROOT}/wfm_gen_test/vqa_test/ar1_5_wfm_outputs/${PREFIX}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/run.log"

echo "[vqa] prefix=${PREFIX}"
echo "[vqa] out_dir=${OUT_DIR}"
if [ $# -gt 0 ]; then
    echo "[vqa] extra args: $*"
fi
python "${PROJECT_ROOT}/wfm_gen_test/vqa_test/vqa_inference.py" \
    --clip-prefix "$PREFIX" \
    --out-dir "$OUT_DIR" \
    "$@" 2>&1 | tee "$LOG"
