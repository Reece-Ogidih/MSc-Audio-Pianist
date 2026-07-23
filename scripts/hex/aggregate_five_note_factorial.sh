#!/usr/bin/env bash
set -euo pipefail
: "${HEX_SCRATCH:?Set HEX_SCRATCH}"
ROOT="${HEX_SCRATCH}/five_note_factorial_1m"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${ROOT}/aggregate/${STAMP}"
mkdir -p "${OUT}"
PYTHONPATH="${PYTHONPATH:-/app/src:/app/third_party/robopianist}" python scripts/aggregate_five_note_factorial.py --root "${ROOT}" --output-dir "${OUT}"
PYTHONPATH="${PYTHONPATH:-/app/src:/app/third_party/robopianist}" python scripts/plot_five_note_factorial_learning_curves.py --aggregate-dir "${OUT}"
PYTHONPATH="${PYTHONPATH:-/app/src:/app/third_party/robopianist}" python scripts/select_five_note_factorial_candidates.py --aggregate-dir "${OUT}"
PYTHONPATH="${PYTHONPATH:-/app/src:/app/third_party/robopianist}" python scripts/analyse_five_note_continuation.py --aggregate-dir "${OUT}"
echo "aggregate_dir=${OUT}"
