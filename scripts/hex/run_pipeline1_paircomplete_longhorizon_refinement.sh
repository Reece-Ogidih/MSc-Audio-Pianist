#!/usr/bin/env bash
set -euo pipefail

# Prepared job only: Pipeline 1 symbolic pair-complete + long-horizon refinement.

PROJECT_ROOT="${PROJECT_ROOT:-/app}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/experiments/general_one_hand/droq}"
RUN_NAME="${RUN_NAME:-pipeline1_symbolic_paircomplete_longhorizon_v1}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
SOURCE_POLICY="${SOURCE_POLICY:-${PROJECT_ROOT}/artifacts/frozen_models/five_note_symbolic_controller_v1/checkpoint_800000_steps.pt}"
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
if [[ ! -f "${SOURCE_POLICY}" || ! -s "${SOURCE_POLICY}" ]]; then
  echo "Source symbolic policy missing or empty: ${SOURCE_POLICY}" >&2
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
SOURCE_SHA256="$(sha256sum "${SOURCE_POLICY}" | awk '{print $1}')"

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
    "source_policy": "${SOURCE_POLICY}",
    "source_policy_sha256": "${SOURCE_SHA256}",
    "adaptation_semantics": "actor_warm_start_fresh_critics_optimizers_replay",
}
Path("${RUN_DIR}/launch_metadata_start.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
PY

python scripts/train_droq_general_one_hand_policy.py \
  --warm-start-policy-path "${SOURCE_POLICY}" \
  --timesteps "${TIMESTEPS}" \
  --seed "${SEED}" \
  --stage-name "${RUN_NAME}" \
  --output-dir "${RUN_DIR}" \
  --curriculum sequence_cleanup \
  --midi-pitches "72,73,74,75,76" \
  --sequence-pitches "${SEQUENCE_PITCHES}" \
  --sequence-sampling-weights "${SEQUENCE_SAMPLING_WEIGHTS}" \
  --sequence-timing-profile aligned \
  --lookahead 1 \
  --horizon-steps "${HORIZON_STEPS}" \
  --action-mode direct \
  --action-repeat 1 \
  --reward-profile transition_cleanup_sensitive_v1 \
  --learning-starts 10000 \
  --batch-size 256 \
  --utd-ratio 4 \
  --lightweight-checkpoint-steps "100000,250000,500000" \
  --full-checkpoint-steps "500000" \
  --device "${DEVICE}" \
  2>&1 | tee "${LOG_PATH}"
