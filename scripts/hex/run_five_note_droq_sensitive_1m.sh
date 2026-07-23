#!/usr/bin/env bash
set -euo pipefail
export CONDITION_ID=droq_sensitive_v1
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_five_note_factorial_job.sh"
