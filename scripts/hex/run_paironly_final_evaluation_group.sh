#!/usr/bin/env bash
set -euo pipefail

# Container-side worker for the final synthetic comparison. The host
# orchestrator assigns GROUP so each container owns a disjoint output subtree.

PROJECT_ROOT="${PROJECT_ROOT:-/app}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/runs/paironly_final_evaluation}"
RUN_NAME="${RUN_NAME:-paironly_final_synthetic_evaluation_v1}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
GROUP="${GROUP:?GROUP is required: p1_all, p2_all, or p2_audio_interventions}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-20260808}"
DEFAULT_SOUNDFONT="${PROJECT_ROOT}/third_party/robopianist/robopianist/soundfonts/TimGM6mb.sf2"
SOUNDFONT_PATH="${SOUNDFONT:-${DEFAULT_SOUNDFONT}}"
GIT_CONFIG_GLOBAL="${GIT_CONFIG_GLOBAL:-/tmp/ala-pianist-gitconfig}"

P1_BASE="${P1_BASE:-${PROJECT_ROOT}/artifacts/frozen_models/five_note_symbolic_controller_v1/checkpoint_800000_steps.pt}"
P1_PAIRONLY="${P1_PAIRONLY:-/workspace/runs/general_one_hand/droq/pipeline1_symbolic_paironly_complete_v1/lightweight_checkpoints/pipeline1_symbolic_paironly_complete_v1_droq_sequence_cleanup_lookahead1_directx1_transition_cleanup_sensitive_v1_seed13_500000/checkpoint_500000_steps.pt}"
P1_PAIRHORIZON="${P1_PAIRHORIZON:-/workspace/runs/general_one_hand/droq/pipeline1_symbolic_paircomplete_longhorizon_v1/lightweight_checkpoints/pipeline1_symbolic_paircomplete_longhorizon_v1_droq_sequence_cleanup_lookahead1_directx1_transition_cleanup_sensitive_v1_seed13_500000/checkpoint_500000_steps.pt}"
P2_BASE="${P2_BASE:-/workspace/experiments/pipeline2_direct_audio/pipeline2_direct_audio_droq_v1_seed13_1m_retry1/lightweight_checkpoints/checkpoint_1000000_steps.pt}"
P2_PAIRONLY="${P2_PAIRONLY:-/workspace/experiments/pipeline2_direct_audio/pipeline2_seed13_paironly_complete_v1/lightweight_checkpoints/checkpoint_500000_steps.pt}"
P2_PAIRHORIZON="${P2_PAIRHORIZON:-/workspace/experiments/pipeline2_direct_audio/pipeline2_seed13_paircomplete_longhorizon_v1/lightweight_checkpoints/checkpoint_500000_steps.pt}"

fail() { echo "FINAL_EVAL_WORKER_FAILURE: $*" >&2; exit 2; }
require_file() {
  local path="$1"
  [[ -s "${path}" ]] || fail "Missing or empty required file: ${path}"
}
require_group_empty() {
  local path="$1"
  if [[ -e "${path}" ]] && [[ -n "$(find "${path}" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
    fail "Refusing to overwrite non-empty group output directory: ${path}"
  fi
}

cd "${PROJECT_ROOT}"
export GIT_CONFIG_GLOBAL
git config --global --add safe.directory "${PROJECT_ROOT}" >/dev/null 2>&1 || true

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/third_party/robopianist:${PROJECT_ROOT}/scripts"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/ala-numba-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/ala-xdg-cache}"
export SOUNDFONT="${SOUNDFONT_PATH}"
mkdir -p "${NUMBA_CACHE_DIR}" "${XDG_CACHE_HOME}" "${RUN_DIR}"
[[ -w "${NUMBA_CACHE_DIR}" ]] || fail "NUMBA_CACHE_DIR is not writable: ${NUMBA_CACHE_DIR}"
[[ -w "${XDG_CACHE_HOME}" ]] || fail "XDG_CACHE_HOME is not writable: ${XDG_CACHE_HOME}"
require_file "${SOUNDFONT_PATH}"

for path in "${P1_BASE}" "${P1_PAIRONLY}" "${P1_PAIRHORIZON}" "${P2_BASE}" "${P2_PAIRONLY}" "${P2_PAIRHORIZON}"; do
  require_file "${path}"
done

RETENTION_ARGS=(
  --sequence-name anchor_000_72 --sequence-name anchor_001_73 --sequence-name anchor_002_74
  --sequence-name anchor_003_75 --sequence-name anchor_004_76
  --sequence-name originally_seen_adjacent_006_72_73 --sequence-name originally_seen_adjacent_010_73_72
  --sequence-name originally_seen_adjacent_012_73_74 --sequence-name originally_seen_adjacent_016_74_73
  --sequence-name originally_seen_adjacent_018_74_75 --sequence-name originally_seen_adjacent_022_75_74
  --sequence-name originally_seen_adjacent_024_75_76 --sequence-name originally_seen_adjacent_028_76_75
)

run_p1_eval() {
  local model_id="$1" controller="$2" manifest="$3" battery="$4" allow_short="$5"
  local out="${RUN_DIR}/pipeline1/${model_id}/${battery}"
  local extra=()
  [[ "${allow_short}" == "true" ]] && extra+=(--allow-short-primitives)
  require_group_empty "${out}"
  mkdir -p "$(dirname "${out}")" "${RUN_DIR}/logs"
  python scripts/evaluate_long_horizon_compositional.py \
    --manifest "${PROJECT_ROOT}/${manifest}" \
    --output-dir "${out}" \
    --pipeline1-controller "${controller}" \
    --pipeline1-condition both \
    --skip-pipeline2 \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    "${extra[@]}" \
    2>&1 | tee "${RUN_DIR}/logs/pipeline1_${model_id}_${battery}.log"
}

run_p1_retention() {
  local model_id="$1" controller="$2"
  local out="${RUN_DIR}/pipeline1/${model_id}/exact_13_retention"
  require_group_empty "${out}"
  mkdir -p "$(dirname "${out}")" "${RUN_DIR}/logs"
  python scripts/evaluate_long_horizon_compositional.py \
    --manifest "${PROJECT_ROOT}/configs/complete_pairwise_v1.json" \
    --output-dir "${out}" \
    --allow-short-primitives \
    "${RETENTION_ARGS[@]}" \
    --pipeline1-controller "${controller}" \
    --pipeline1-condition both \
    --skip-pipeline2 \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    2>&1 | tee "${RUN_DIR}/logs/pipeline1_${model_id}_exact_13_retention.log"
}

run_p2_eval() {
  local model_id="$1" checkpoint="$2" manifest="$3" battery="$4" allow_short="$5"
  local out="${RUN_DIR}/pipeline2/${model_id}/${battery}"
  local extra=()
  [[ "${allow_short}" == "true" ]] && extra+=(--allow-short-primitives)
  require_group_empty "${out}"
  mkdir -p "$(dirname "${out}")" "${RUN_DIR}/logs"
  python scripts/evaluate_long_horizon_compositional.py \
    --manifest "${PROJECT_ROOT}/${manifest}" \
    --output-dir "${out}" \
    --skip-pipeline1 \
    --pipeline2-checkpoint-path "${model_id}=${checkpoint}" \
    --pipeline2-model "${model_id}" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    "${extra[@]}" \
    2>&1 | tee "${RUN_DIR}/logs/pipeline2_${model_id}_${battery}.log"
}

run_p2_retention() {
  local model_id="$1" checkpoint="$2"
  local out="${RUN_DIR}/pipeline2/${model_id}/exact_13_retention"
  require_group_empty "${out}"
  mkdir -p "$(dirname "${out}")" "${RUN_DIR}/logs"
  python scripts/evaluate_long_horizon_compositional.py \
    --manifest "${PROJECT_ROOT}/configs/complete_pairwise_v1.json" \
    --output-dir "${out}" \
    --allow-short-primitives \
    "${RETENTION_ARGS[@]}" \
    --skip-pipeline1 \
    --pipeline2-checkpoint-path "${model_id}=${checkpoint}" \
    --pipeline2-model "${model_id}" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    2>&1 | tee "${RUN_DIR}/logs/pipeline2_${model_id}_exact_13_retention.log"
}

run_p2_intervention_eval() {
  local model_id="$1" checkpoint="$2" manifest="$3" battery="$4" allow_short="$5"
  shift 5
  local out="${RUN_DIR}/pipeline2_audio_interventions/${model_id}/${battery}"
  local extra=()
  [[ "${allow_short}" == "true" ]] && extra+=(--allow-short-primitives)
  require_group_empty "${out}"
  mkdir -p "$(dirname "${out}")" "${RUN_DIR}/logs"
  python scripts/evaluate_long_horizon_compositional.py \
    --manifest "${PROJECT_ROOT}/${manifest}" \
    --output-dir "${out}" \
    --skip-pipeline1 \
    --pipeline2-checkpoint-path "${model_id}=${checkpoint}" \
    --pipeline2-model "${model_id}" \
    --include-audio-interventions \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    "${extra[@]}" "$@" \
    2>&1 | tee "${RUN_DIR}/logs/pipeline2_${model_id}_${battery}_audio_interventions.log"
}

run_all_batteries_p1() {
  local model_id="$1" controller="$2"
  run_p1_retention "${model_id}" "${controller}"
  run_p1_eval "${model_id}" "${controller}" configs/complete_pairwise_v1.json complete_pairwise true
  run_p1_eval "${model_id}" "${controller}" configs/long_horizon_compositional_v1.json frozen_long_horizon_compositional false
  run_p1_eval "${model_id}" "${controller}" configs/long_horizon_clean_test_v1.json clean_unseen_compositions false
  run_p1_eval "${model_id}" "${controller}" configs/long_horizon_extrapolation_v1.json extrapolation_30_40 false
}

run_all_batteries_p2() {
  local model_id="$1" checkpoint="$2"
  run_p2_retention "${model_id}" "${checkpoint}"
  run_p2_eval "${model_id}" "${checkpoint}" configs/complete_pairwise_v1.json complete_pairwise true
  run_p2_eval "${model_id}" "${checkpoint}" configs/long_horizon_compositional_v1.json frozen_long_horizon_compositional false
  run_p2_eval "${model_id}" "${checkpoint}" configs/long_horizon_clean_test_v1.json clean_unseen_compositions false
  run_p2_eval "${model_id}" "${checkpoint}" configs/long_horizon_extrapolation_v1.json extrapolation_30_40 false
}

case "${GROUP}" in
  p1_all)
    run_all_batteries_p1 p1_base "${P1_BASE}"
    run_all_batteries_p1 p1_paironly "${P1_PAIRONLY}"
    run_all_batteries_p1 p1_pairhorizon "${P1_PAIRHORIZON}"
    ;;
  p2_all)
    run_all_batteries_p2 p2_base "${P2_BASE}"
    run_all_batteries_p2 p2_paironly "${P2_PAIRONLY}"
    run_all_batteries_p2 p2_pairhorizon "${P2_PAIRHORIZON}"
    ;;
  p2_audio_interventions)
    SHORT_INTERVENTIONS=(
      --sequence-name anchor_000_72 --sequence-name anchor_002_74 --sequence-name anchor_004_76
      --sequence-name originally_seen_adjacent_006_72_73 --sequence-name originally_seen_adjacent_018_74_75
      --sequence-name newly_added_nonadjacent_014_73_76 --sequence-name newly_added_nonadjacent_021_75_73
      --audio-intervention-sequence 72 --audio-intervention-sequence 74 --audio-intervention-sequence 76
      --audio-intervention-sequence 72,73 --audio-intervention-sequence 74,75
      --audio-intervention-sequence 73,76 --audio-intervention-sequence 75,73
    )
    CLEAN_INTERVENTIONS=(
      --sequence-name clean_test_000_len3 --sequence-name clean_test_001_len3
      --sequence-name clean_test_002_len5 --sequence-name clean_test_003_len5
      --sequence-name clean_test_004_len10 --sequence-name clean_test_005_len10
      --audio-intervention-sequence 74,76,75 --audio-intervention-sequence 73,72,74
      --audio-intervention-sequence 74,76,75,73,72 --audio-intervention-sequence 72,75,73,76,74
      --audio-intervention-sequence 74,76,75,73,72,74,75,76,73,72
      --audio-intervention-sequence 72,74,76,73,75,72,76,74,75,73
    )
    for spec in "p2_base:${P2_BASE}" "p2_paironly:${P2_PAIRONLY}" "p2_pairhorizon:${P2_PAIRHORIZON}"; do
      model_id="${spec%%:*}"
      checkpoint="${spec#*:}"
      run_p2_intervention_eval "${model_id}" "${checkpoint}" configs/complete_pairwise_v1.json representative_short true "${SHORT_INTERVENTIONS[@]}"
      run_p2_intervention_eval "${model_id}" "${checkpoint}" configs/long_horizon_clean_test_v1.json representative_clean false "${CLEAN_INTERVENTIONS[@]}"
    done
    ;;
  *)
    fail "Unknown GROUP=${GROUP}. Expected p1_all, p2_all, or p2_audio_interventions."
    ;;
esac

echo "FINAL_EVALUATION_GROUP_COMPLETE=${GROUP}"
