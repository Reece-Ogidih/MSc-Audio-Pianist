#!/usr/bin/env bash
set -euo pipefail

# Prepared evaluation only: refined Pipeline 1/2 pair-complete and long-horizon battery.

PROJECT_ROOT="${PROJECT_ROOT:-/app}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/experiments/paircomplete_refined_evaluation}"
RUN_NAME="${RUN_NAME:-paircomplete_refined_model_evaluation_v1}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
PIPELINE1_CONTROLLER="${PIPELINE1_CONTROLLER:-/workspace/experiments/general_one_hand/droq/pipeline1_symbolic_paircomplete_longhorizon_v1/lightweight_checkpoints/pipeline1_symbolic_paircomplete_longhorizon_v1_droq_sequence_cleanup_lookahead1_directx1_transition_cleanup_sensitive_v1_seed13_500000/checkpoint_500000_steps.pt}"
PIPELINE2_CHECKPOINT="${PIPELINE2_CHECKPOINT:-/workspace/experiments/pipeline2_direct_audio/pipeline2_seed13_paircomplete_longhorizon_v1/lightweight_checkpoints/checkpoint_500000_steps.pt}"
PIPELINE2_LABEL="${PIPELINE2_LABEL:-pipeline2_paircomplete_longhorizon_v1}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-20260808}"
DEFAULT_SOUNDFONT="${PROJECT_ROOT}/third_party/robopianist/robopianist/soundfonts/TimGM6mb.sf2"
SOUNDFONT_PATH="${SOUNDFONT:-${DEFAULT_SOUNDFONT}}"
GIT_CONFIG_GLOBAL="${GIT_CONFIG_GLOBAL:-${RUN_DIR}/gitconfig}"

if [[ -e "${RUN_DIR}" ]] && [[ -n "$(find "${RUN_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite non-empty evaluation directory: ${RUN_DIR}" >&2
  exit 2
fi
if [[ ! -f "${PIPELINE1_CONTROLLER}" || ! -s "${PIPELINE1_CONTROLLER}" ]]; then
  echo "Pipeline 1 refined controller missing or empty: ${PIPELINE1_CONTROLLER}" >&2
  exit 3
fi
if [[ ! -f "${PIPELINE2_CHECKPOINT}" || ! -s "${PIPELINE2_CHECKPOINT}" ]]; then
  echo "Pipeline 2 refined checkpoint missing or empty: ${PIPELINE2_CHECKPOINT}" >&2
  exit 4
fi
if [[ ! -f "${SOUNDFONT_PATH}" || ! -s "${SOUNDFONT_PATH}" ]]; then
  echo "Soundfont missing or empty: ${SOUNDFONT_PATH}" >&2
  exit 5
fi

mkdir -p "${RUN_DIR}"
cd "${PROJECT_ROOT}"
export GIT_CONFIG_GLOBAL
git config --global --add safe.directory "${PROJECT_ROOT}"

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
    "pipeline1_controller": "${PIPELINE1_CONTROLLER}",
    "pipeline2_checkpoint": "${PIPELINE2_CHECKPOINT}",
    "pipeline2_label": "${PIPELINE2_LABEL}",
    "soundfont": "${SOUNDFONT_PATH}",
    "benchmarks": [
        "configs/complete_pairwise_v1.json",
        "configs/long_horizon_compositional_v1.json",
        "configs/long_horizon_validation_v1.json",
        "configs/long_horizon_clean_test_v1.json",
        "configs/long_horizon_extrapolation_v1.json",
    ],
}
Path("${RUN_DIR}/evaluation_launch_metadata.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
PY

run_eval() {
  local manifest="$1"
  local name="$2"
  local allow_short="$3"
  local extra_args=()
  if [[ "${allow_short}" == "true" ]]; then
    extra_args+=(--allow-short-primitives)
  fi
  python scripts/evaluate_long_horizon_compositional.py \
    --manifest "${PROJECT_ROOT}/${manifest}" \
    --output-dir "${RUN_DIR}/${name}" \
    --pipeline1-controller "${PIPELINE1_CONTROLLER}" \
    --pipeline2-checkpoint-path "${PIPELINE2_LABEL}=${PIPELINE2_CHECKPOINT}" \
    --pipeline2-model "${PIPELINE2_LABEL}" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    "${extra_args[@]}" \
    2>&1 | tee "${RUN_DIR}/${name}.log"
}

run_eval "configs/complete_pairwise_v1.json" "complete_pairwise" "true"
run_eval "configs/long_horizon_compositional_v1.json" "frozen_long_horizon_v1" "false"
run_eval "configs/long_horizon_validation_v1.json" "validation_compositions" "false"
run_eval "configs/long_horizon_clean_test_v1.json" "clean_composition_test" "false"
run_eval "configs/long_horizon_extrapolation_v1.json" "extrapolation_30_40" "false"
