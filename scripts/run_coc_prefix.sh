#!/bin/bash
set -euo pipefail

# Batch CoC test runner: run ar1_5_wfm_test.py on every wfm_gen_data clip
# whose labels/<clip_id>.json id starts with a given prefix.
#
# Usage:
#   bash scripts/run_coc_prefix.sh <clip_prefix> [extra args for ar1_5_wfm_test.py]
#
# Examples:
#   bash scripts/run_coc_prefix.sh fire-seed1
#   bash scripts/run_coc_prefix.sh fire-seed1 --frame-start 20 --frame-end 60
#   bash scripts/run_coc_prefix.sh fire-seed1 --num-traj-samples 4
#   bash scripts/run_coc_prefix.sh fire-seed1 --out-dir /tmp/coc_out
#
# Note: "fire-seed1" also matches fire-seed10000_*, etc. (plain prefix match).
# Outputs default to
#   wfm_gen_test/coc_test/ar1_5_wfm_outputs/<prefix>_<YYYYmmdd_HHMMSS>/
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
    echo "Usage: $0 <clip_prefix> [extra args for ar1_5_wfm_test.py]" >&2
    echo "  e.g. $0 fire-seed1 --frame-start 20 --frame-end 60 --num-traj-samples 4" >&2
    exit 1
fi

PREFIX=$1
shift

OUT_DIR="${PROJECT_ROOT}/wfm_gen_test/coc_test/ar1_5_wfm_outputs/${PREFIX}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/run.log"

echo "[coc] prefix=${PREFIX}"
echo "[coc] out_dir=${OUT_DIR}"
if [ $# -gt 0 ]; then
    echo "[coc] extra args: $*"
fi
python "${PROJECT_ROOT}/wfm_gen_test/coc_test/ar1_5_wfm_test.py" \
    --clip-prefix "$PREFIX" \
    --out-dir "$OUT_DIR" \
    "$@" 2>&1 | tee "$LOG"
