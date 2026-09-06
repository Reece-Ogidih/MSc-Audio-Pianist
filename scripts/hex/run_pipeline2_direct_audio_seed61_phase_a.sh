#!/usr/bin/env bash
set -euo pipefail

# Prepared final-experiment job: third independent Pipeline 2 seed.
# Does not launch by itself; run inside the Hex container or via hare.

export SEED=61
export TIMESTEPS="${TIMESTEPS:-1000000}"
export RUN_NAME="${RUN_NAME:-pipeline2_direct_audio_droq_v1_seed61_1m}"
export DEVICE="${DEVICE:-cuda}"

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_pipeline2_direct_audio_droq_phase_a.sh"
