#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/app}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/runs/general_one_hand/droq}"
RUN_NAME="${RUN_NAME:-pipeline1_symbolic_paironly_complete_v1}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
SOURCE_POLICY="${SOURCE_POLICY:-${PROJECT_ROOT}/artifacts/frozen_models/five_note_symbolic_controller_v1/checkpoint_800000_steps.pt}"
CURRICULUM_MANIFEST="${CURRICULUM_MANIFEST:-${PROJECT_ROOT}/configs/pair_only_complete_v1.json}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-13}"
TIMESTEPS=500000
SEMANTICS="actor_weights_only_fresh_critics_optimizers_replay_buffer_rng_from_seed"

if [[ -e "${RUN_DIR}" ]] && [[ -n "$(find "${RUN_DIR}" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite non-empty run directory: ${RUN_DIR}" >&2; exit 2
fi
for path in "${SOURCE_POLICY}" "${CURRICULUM_MANIFEST}"; do
  [[ -s "${path}" ]] || { echo "Required input missing or empty: ${path}" >&2; exit 3; }
done
mkdir -p "${RUN_DIR}/logs"
cd "${PROJECT_ROOT}"
git config --global --add safe.directory "${PROJECT_ROOT}"
eval "$(python scripts/curriculum_manifest_args.py --manifest "${CURRICULUM_MANIFEST}" --format shell)"
CURRICULUM_SHA256="$(sha256sum "${CURRICULUM_MANIFEST}" | awk '{print $1}')"
SOURCE_SHA256="$(sha256sum "${SOURCE_POLICY}" | awk '{print $1}')"
export PYTHONUNBUFFERED=1 PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/third_party/robopianist:${PROJECT_ROOT}/scripts"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/ala-numba-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/ala-xdg-cache}"
mkdir -p "${NUMBA_CACHE_DIR}" "${XDG_CACHE_HOME}"

echo "source_checkpoint=${SOURCE_POLICY}"
echo "source_checkpoint_sha256=${SOURCE_SHA256}"
echo "curriculum_manifest=${CURRICULUM_MANIFEST}"
echo "curriculum_sha256=${CURRICULUM_SHA256}"
echo "warm_start_semantics=${SEMANTICS}"
python - <<PY
import json, subprocess
from pathlib import Path
Path("${RUN_DIR}/launch_metadata_start.json").write_text(json.dumps({
  "run_name": "${RUN_NAME}", "run_dir": "${RUN_DIR}",
  "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
  "source_model_identity": "original canonical Pipeline 1 frozen controller at 800k",
  "source_checkpoint": "${SOURCE_POLICY}", "source_checkpoint_sha256": "${SOURCE_SHA256}",
  "curriculum_manifest": "${CURRICULUM_MANIFEST}", "curriculum_sha256": "${CURRICULUM_SHA256}",
  "warm_start_semantics": "${SEMANTICS}", "seed": ${SEED}, "adaptation_steps": ${TIMESTEPS},
  "reward_profile": "transition_cleanup_sensitive_v1", "checkpoint_steps": [100000, 250000, 500000]
}, indent=2, sort_keys=True), encoding="utf-8")
PY

started="$(date +%s)"
python scripts/train_droq_general_one_hand_policy.py \
  --warm-start-policy-path "${SOURCE_POLICY}" --timesteps "${TIMESTEPS}" --seed "${SEED}" \
  --stage-name "${RUN_NAME}" --output-dir "${RUN_DIR}" --curriculum sequence_cleanup \
  --midi-pitches 72,73,74,75,76 --sequence-pitches "${SEQUENCE_PITCHES}" \
  --sequence-sampling-weights "${SEQUENCE_SAMPLING_WEIGHTS}" --sequence-timing-profile aligned \
  --lookahead 1 --horizon-steps "${HORIZON_STEPS}" --action-mode direct --action-repeat 1 \
  --reward-profile transition_cleanup_sensitive_v1 --learning-starts 10000 --batch-size 256 --utd-ratio 4 \
  --lightweight-checkpoint-steps 100000,250000,500000 --full-checkpoint-steps 500000 --device "${DEVICE}" \
  2>&1 | tee "${RUN_DIR}/logs/train.log"
runtime="$(( $(date +%s) - started ))"
python - <<PY
import json
from pathlib import Path
Path("${RUN_DIR}/completion_summary.json").write_text(json.dumps({
  "status": "complete", "run_name": "${RUN_NAME}", "runtime_seconds": ${runtime},
  "adaptation_steps": ${TIMESTEPS}, "warm_start_semantics": "${SEMANTICS}",
  "source_checkpoint_sha256": "${SOURCE_SHA256}", "curriculum_sha256": "${CURRICULUM_SHA256}"
}, indent=2, sort_keys=True), encoding="utf-8")
PY
echo "PAIR_ONLY_PIPELINE1_COMPLETE=true"
