#!/usr/bin/env bash
set -euo pipefail
export CONDITION_ID=sac_original
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_five_note_factorial_job.sh"
