#!/usr/bin/env bash
set -euo pipefail

# Prepared job only: Pipeline 2 pair-complete + long-horizon curriculum refinement.

PROJECT_ROOT="${PROJECT_ROOT:-/app}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/experiments/pipeline2_direct_audio}"
RUN_NAME="${RUN_NAME:-pipeline2_seed13_paircomplete_longhorizon_v1}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${OUTPUT_ROOT}/pipeline2_direct_audio_droq_v1_seed13_1m_retry1/checkpoints/full_checkpoint_1000000_steps.pt}"
CURRICULUM_MANIFEST="${CURRICULUM_MANIFEST:-${PROJECT_ROOT}/configs/long_horizon_curriculum_v1.json}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-13}"
TIMESTEPS="${TIMESTEPS:-500000}"
DEFAULT_SOUNDFONT="${PROJECT_ROOT}/third_party/robopianist/robopianist/soundfonts/TimGM6mb.sf2"
SOUNDFONT_PATH="${SOUNDFONT:-${DEFAULT_SOUNDFONT}}"
LOG_DIR="${RUN_DIR}/logs"
LOG_PATH="${LOG_DIR}/train.log"
GIT_CONFIG_GLOBAL="${GIT_CONFIG_GLOBAL:-${RUN_DIR}/gitconfig}"

if [[ -e "${RUN_DIR}" ]] && [[ -n "$(find "${RUN_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite non-empty run directory: ${RUN_DIR}" >&2
  exit 2
fi
if [[ ! -f "${SOURCE_CHECKPOINT}" || ! -s "${SOURCE_CHECKPOINT}" ]]; then
  echo "Source full checkpoint missing or empty: ${SOURCE_CHECKPOINT}" >&2
  exit 3
fi
if [[ ! -f "${CURRICULUM_MANIFEST}" || ! -s "${CURRICULUM_MANIFEST}" ]]; then
  echo "Curriculum manifest missing or empty: ${CURRICULUM_MANIFEST}" >&2
  exit 4
fi
if [[ ! -f "${SOUNDFONT_PATH}" || ! -s "${SOUNDFONT_PATH}" ]]; then
  echo "Soundfont missing or empty: ${SOUNDFONT_PATH}" >&2
  exit 5
fi

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"
export GIT_CONFIG_GLOBAL
git config --global --add safe.directory "${PROJECT_ROOT}"

eval "$(python scripts/curriculum_manifest_args.py --manifest "${CURRICULUM_MANIFEST}" --format shell)"
CURRICULUM_SHA256="$(sha256sum "${CURRICULUM_MANIFEST}" | awk '{print $1}')"
SOURCE_SHA256="$(sha256sum "${SOURCE_CHECKPOINT}" | awk '{print $1}')"
SOUNDFONT_SHA256="$(sha256sum "${SOUNDFONT_PATH}" | awk '{print $1}')"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/third_party/robopianist:${PROJECT_ROOT}/scripts"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export SOUNDFONT="${SOUNDFONT_PATH}"

python - <<PY
import json, subprocess
from pathlib import Path
payload = {
    "run_name": "${RUN_NAME}",
    "run_dir": "${RUN_DIR}",
    "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], check=True, stdout=subprocess.PIPE, text=True).stdout.strip(),
    "curriculum_manifest": "${CURRICULUM_MANIFEST}",
    "curriculum_sha256": "${CURRICULUM_SHA256}",
    "source_checkpoint": "${SOURCE_CHECKPOINT}",
    "source_checkpoint_sha256": "${SOURCE_SHA256}",
    "soundfont": "${SOUNDFONT_PATH}",
    "soundfont_sha256": "${SOUNDFONT_SHA256}",
    "adaptation_semantics": "curriculum_finetune_network_optimizer_alpha_warm_start_fresh_replay",
}
Path("${RUN_DIR}/launch_metadata_start.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
PY

python scripts/train_direct_audio_droq.py \
  --warm-start-checkpoint "${SOURCE_CHECKPOINT}" \
  --timesteps "${TIMESTEPS}" \
  --seed "${SEED}" \
  --stage-name "${RUN_NAME}" \
  --output-dir "${RUN_DIR}" \
  --generated-root "${RUN_DIR}/audio_bank" \
  --sequence-pitches "${SEQUENCE_PITCHES}" \
  --sequence-sampling-weights "${SEQUENCE_SAMPLING_WEIGHTS}" \
  --variants-per-sequence 4 \
  --horizon-steps "${HORIZON_STEPS}" \
  --learning-starts 10000 \
  --batch-size 64 \
  --utd-ratio 2 \
  --buffer-size 2000000 \
  --lightweight-checkpoint-steps "100000,250000,500000" \
  --full-checkpoint-steps "500000" \
  --device "${DEVICE}" \
  2>&1 | tee "${LOG_PATH}"
