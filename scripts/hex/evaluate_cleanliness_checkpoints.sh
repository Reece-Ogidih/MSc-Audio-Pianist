#!/usr/bin/env bash
set -euo pipefail
: "${HEX_RUN_DIR:?Set HEX_RUN_DIR}"
find "${HEX_RUN_DIR}" -name 'checkpoint_*_steps.pt' -type f | sort
echo "Use scripts/evaluate_general_one_hand_policy.py with --model-kind droq and --reward-profile transition_cleanup_sensitive_v1."
