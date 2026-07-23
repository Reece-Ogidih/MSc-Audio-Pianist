#!/usr/bin/env bash
set -euo pipefail
: "${HEX_GPU_INDEX:?Set HEX_GPU_INDEX}"
: "${HEX_SCRATCH:?Set HEX_SCRATCH}"
: "${HEX_IMAGE_TAG:?Set HEX_IMAGE_TAG}"
: "${CONDITION_ID:?Set CONDITION_ID}"
: "${RESUME_CHECKPOINT:?Set RESUME_CHECKPOINT to a full checkpoint path under HEX_SCRATCH}"
: "${ADDITIONAL_TIMESTEPS:?Set ADDITIONAL_TIMESTEPS}"

echo "Resume is supported for DroQ full checkpoints directly."
echo "For SAC, use the matching .zip plus *_replay_buffer.pkl pair saved by the full checkpoint callback."
echo "condition_id=${CONDITION_ID}"
echo "resume_checkpoint=${RESUME_CHECKPOINT}"
echo "additional_timesteps=${ADDITIONAL_TIMESTEPS}"
echo "Use the condition config in configs/five_note_factorial_1m and launch under a new HEX_RUN_NAME."
