#!/usr/bin/env bash
set -euo pipefail

# Evaluate one completed Pipeline 2 Phase-A run.
# Intended usage:
#   CUDA_VISIBLE_DEVICES=0 RUN_NAME=pipeline2_direct_audio_droq_v1_seed13_1m_retry1 \
#     scripts/hex/evaluate_pipeline2_direct_audio_phase_a.sh

PROJECT_ROOT="${PROJECT_ROOT:-/app}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/workspace}"
RUN_NAME="${RUN_NAME:?Set RUN_NAME to a completed Pipeline 2 run directory name.}"
RUN_DIR="${RUN_DIR:-${SCRATCH_ROOT}/experiments/pipeline2_direct_audio/${RUN_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/evaluation}"
DEVICE="${DEVICE:-cuda}"
DEFAULT_SOUNDFONT="${PROJECT_ROOT}/third_party/robopianist/robopianist/soundfonts/TimGM6mb.sf2"
SOUNDFONT_PATH="${SOUNDFONT:-${DEFAULT_SOUNDFONT}}"

cd "${PROJECT_ROOT}"
git config --global --add safe.directory "${PROJECT_ROOT}"

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "Run directory does not exist: ${RUN_DIR}" >&2
  exit 2
fi
if [[ ! -f "${SOUNDFONT_PATH}" || ! -s "${SOUNDFONT_PATH}" ]]; then
  echo "Soundfont is missing or empty: ${SOUNDFONT_PATH}" >&2
  exit 3
fi

mkdir -p "${OUTPUT_DIR}/logs"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/third_party/robopianist:${PROJECT_ROOT}/scripts"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export SOUNDFONT="${SOUNDFONT_PATH}"

python scripts/evaluate_pipeline2_direct_audio.py \
  --run-dir "${RUN_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --evaluation-audio-root "${OUTPUT_DIR}/canonical_eval_audio" \
  --device "${DEVICE}" \
  2>&1 | tee "${OUTPUT_DIR}/logs/evaluate.log"
