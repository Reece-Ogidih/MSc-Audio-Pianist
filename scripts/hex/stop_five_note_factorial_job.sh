#!/usr/bin/env bash
set -euo pipefail
: "${HEX_SCRATCH:?Set HEX_SCRATCH}"
: "${CONDITION_ID:?Set CONDITION_ID}"
: "${HEX_RUN_NAME:?Set HEX_RUN_NAME to the timestamped run directory name}"
RUN_DIR="${HEX_SCRATCH}/five_note_factorial_1m/${CONDITION_ID}/${HEX_RUN_NAME}"
test -d "${RUN_DIR}"
HARE_NAME="$(cat "${RUN_DIR}/hare_name.txt")"
echo "target_hare_name=${HARE_NAME}"
if [[ "${CONFIRM_STOP:-0}" != "1" ]]; then
  echo "Set CONFIRM_STOP=1 to stop this run."
  exit 2
fi
hare stop "${HARE_NAME}"
date -u +%Y-%m-%dT%H:%M:%SZ > "${RUN_DIR}/stopped_at.txt"
