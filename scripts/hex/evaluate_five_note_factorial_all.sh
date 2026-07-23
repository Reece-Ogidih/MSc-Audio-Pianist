#!/usr/bin/env bash
set -euo pipefail
: "${HEX_SCRATCH:?Set HEX_SCRATCH}"
ROOT="${HEX_SCRATCH}/five_note_factorial_1m"
for condition in droq_original droq_sensitive_v1 sac_original sac_sensitive_v1; do
  latest="$(find "${ROOT}/${condition}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1)"
  test -n "${latest}"
  config="${latest}/resolved_training_config.json"
  test -f "${config}"
  PYTHONPATH="${PYTHONPATH:-/app/src:/app/third_party/robopianist}" python scripts/evaluate_five_note_factorial_checkpoint_sweep.py --run-dir "${latest}" --config "${config}" --output-dir "${latest}/evaluation"
done
