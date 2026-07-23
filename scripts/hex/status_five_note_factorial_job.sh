#!/usr/bin/env bash
set -euo pipefail
: "${HEX_SCRATCH:?Set HEX_SCRATCH}"
: "${CONDITION_ID:?Set CONDITION_ID}"
ROOT="${HEX_SCRATCH}/five_note_factorial_1m/${CONDITION_ID}"
test -d "${ROOT}"
find "${ROOT}" -maxdepth 2 -type f \( -name 'started_at.txt' -o -name 'finished_at.txt' -o -name 'exit_code.txt' -o -name 'hare_name.txt' -o -name 'hare_launcher_pid.txt' \) -print -exec sed -n '1,5p' {} \;
find "${ROOT}" -path '*/logs/train.log' -type f -print -exec tail -40 {} \;
hare ps -a || true
