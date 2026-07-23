#!/usr/bin/env bash
set -euo pipefail
: "${HEX_RUN_DIR:?Set HEX_RUN_DIR}"
find "${HEX_RUN_DIR}" -name 'checkpoint*.zip' -o -name '*.zip' | sort
echo "Use scripts/evaluate_general_one_hand_policy.py with --model-kind sb3_sac for selected checkpoints."
