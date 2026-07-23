#!/usr/bin/env bash
set -euo pipefail
: "${HEX_RUN_DIR:?Set HEX_RUN_DIR}"
: "${CONFIG_PATH:?Set CONFIG_PATH}"
PYTHONPATH="${PYTHONPATH:-/app/src:/app/third_party/robopianist}" python scripts/evaluate_five_note_factorial_checkpoint_sweep.py --run-dir "${HEX_RUN_DIR}" --config "${CONFIG_PATH}" --output-dir "${HEX_RUN_DIR}/evaluation"
