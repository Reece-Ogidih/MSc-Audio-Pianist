#!/usr/bin/env bash
set -euo pipefail

# Prepared final-experiment job: true full-state resume from seed13 1M to 1.5M.

PROJECT_ROOT="${PROJECT_ROOT:-/app}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/experiments/pipeline2_direct_audio}"
SOURCE_RUN_NAME="${SOURCE_RUN_NAME:-pipeline2_direct_audio_droq_v1_seed13_1m_retry1}"
RUN_NAME="${RUN_NAME:-pipeline2_direct_audio_droq_v1_seed13_resume_1m_to_1p5m}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-${OUTPUT_ROOT}/${SOURCE_RUN_NAME}/checkpoints/full_checkpoint_1000000_steps.pt}"
DEVICE="${DEVICE:-cuda}"
DEFAULT_SOUNDFONT="${PROJECT_ROOT}/third_party/robopianist/robopianist/soundfonts/TimGM6mb.sf2"
SOUNDFONT_PATH="${SOUNDFONT:-${DEFAULT_SOUNDFONT}}"
LOG_DIR="${RUN_DIR}/logs"

if [[ -e "${RUN_DIR}" ]] && [[ -n "$(find "${RUN_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite non-empty run directory: ${RUN_DIR}" >&2
  exit 2
fi
if [[ ! -f "${RESUME_CHECKPOINT}" || ! -s "${RESUME_CHECKPOINT}" ]]; then
  echo "Full resume checkpoint is missing or empty: ${RESUME_CHECKPOINT}" >&2
  exit 3
fi
if [[ ! -f "${SOUNDFONT_PATH}" || ! -s "${SOUNDFONT_PATH}" ]]; then
  echo "Soundfont is missing or empty: ${SOUNDFONT_PATH}" >&2
  exit 4
fi

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"
git config --global --add safe.directory "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/third_party/robopianist:${PROJECT_ROOT}/scripts"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export SOUNDFONT="${SOUNDFONT_PATH}"

python scripts/train_direct_audio_droq.py \
  --resume-checkpoint "${RESUME_CHECKPOINT}" \
  --additional-timesteps 500000 \
  --seed 13 \
  --stage-name "${RUN_NAME}" \
  --output-dir "${RUN_DIR}" \
  --generated-root "${RUN_DIR}/audio_bank" \
  --sequence-pitches "72;73;74;75;76;72,73;73,72;73,74;74,73;74,75;75,74;75,76;76,75" \
  --sequence-sampling-weights "0.07,0.07,0.07,0.07,0.07,0.08125,0.08125,0.08125,0.08125,0.08125,0.08125,0.08125,0.08125" \
  --variants-per-sequence 4 \
  --learning-starts 1000 \
  --batch-size 64 \
  --utd-ratio 2 \
  --buffer-size 2000000 \
  --lightweight-checkpoint-steps "1250000,1500000" \
  --full-checkpoint-steps "1500000" \
  --device "${DEVICE}" \
  2>&1 | tee "${LOG_DIR}/train.log"
