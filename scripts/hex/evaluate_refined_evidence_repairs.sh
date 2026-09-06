#!/usr/bin/env bash
set -euo pipefail

# Evaluation-only repair for missing Basic Pitch evidence, behavioural audio
# interventions, and the identical original-13 retention comparison.
PROJECT_ROOT="${PROJECT_ROOT:-/app}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/experiments/paircomplete_refined_evaluation_repairs}"
RUN_NAME="${RUN_NAME:-refined_evidence_repairs_v1}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-20260808}"
P1_ORIGINAL="${P1_ORIGINAL:-${PROJECT_ROOT}/artifacts/frozen_models/five_note_symbolic_controller_v1/checkpoint_800000_steps.pt}"
P1_REFINED="${P1_REFINED:-/workspace/runs/general_one_hand/droq/pipeline1_symbolic_paircomplete_longhorizon_v1/lightweight_checkpoints/pipeline1_symbolic_paircomplete_longhorizon_v1_droq_sequence_cleanup_lookahead1_directx1_transition_cleanup_sensitive_v1_seed13_500000/checkpoint_500000_steps.pt}"
P2_ORIGINAL="${P2_ORIGINAL:-${PROJECT_ROOT}/artifacts/pipeline2_phase_a_hex/pipeline2_direct_audio_droq_v1_seed13_1m_retry1/lightweight_checkpoints/checkpoint_1000000_steps.pt}"
P2_REFINED="${P2_REFINED:-/workspace/experiments/pipeline2_direct_audio/pipeline2_seed13_paircomplete_longhorizon_v1/lightweight_checkpoints/checkpoint_500000_steps.pt}"

if [[ -e "${RUN_DIR}" ]] && [[ -n "$(find "${RUN_DIR}" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite non-empty repair directory: ${RUN_DIR}" >&2
  exit 2
fi
for path in "${P1_ORIGINAL}" "${P1_REFINED}" "${P2_ORIGINAL}" "${P2_REFINED}"; do
  [[ -s "${path}" ]] || { echo "Required checkpoint missing or empty: ${path}" >&2; exit 3; }
done

mkdir -p "${RUN_DIR}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/third_party/robopianist:${PROJECT_ROOT}/scripts"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/ala-numba-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/ala-xdg-cache}"
mkdir -p "${NUMBA_CACHE_DIR}" "${XDG_CACHE_HOME}"
[[ -w "${NUMBA_CACHE_DIR}" ]] || { echo "NUMBA_CACHE_DIR is not writable: ${NUMBA_CACHE_DIR}" >&2; exit 4; }
python -c 'from basic_pitch import inference; from pathlib import Path; p=Path(inference.ICASSP_2022_MODEL_PATH).with_suffix(".onnx"); assert p.is_file(), p; print(f"basic_pitch_onnx={p}")'

run_p1_basic_pitch() {
  local manifest="$1" name="$2" allow_short="$3" extra=()
  [[ "${allow_short}" == true ]] && extra+=(--allow-short-primitives)
  python scripts/evaluate_long_horizon_compositional.py \
    --manifest "${PROJECT_ROOT}/${manifest}" --output-dir "${RUN_DIR}/basic_pitch/${name}" \
    --pipeline1-controller "${P1_REFINED}" --pipeline1-condition transcribed --skip-pipeline2 \
    --device "${DEVICE}" --seed "${SEED}" "${extra[@]}"
}

run_p1_basic_pitch configs/complete_pairwise_v1.json complete_pairwise true
run_p1_basic_pitch configs/long_horizon_compositional_v1.json frozen_long_horizon_v1 false
run_p1_basic_pitch configs/long_horizon_clean_test_v1.json clean_composition_test false
run_p1_basic_pitch configs/long_horizon_extrapolation_v1.json extrapolation_30_40 false

# Three anchors, four directed pairs spanning adjacent/nonadjacent behaviour,
# and matched pairs of clean compositions at lengths 3, 5, and 10.
python scripts/evaluate_long_horizon_compositional.py \
  --manifest "${PROJECT_ROOT}/configs/complete_pairwise_v1.json" \
  --output-dir "${RUN_DIR}/audio_interventions/short" --skip-pipeline1 \
  --pipeline2-checkpoint-path "refined_p2=${P2_REFINED}" --pipeline2-model refined_p2 \
  --include-audio-interventions --allow-short-primitives --device "${DEVICE}" --seed "${SEED}" \
  --sequence-name anchor_000_72 --sequence-name anchor_002_74 --sequence-name anchor_004_76 \
  --sequence-name originally_seen_adjacent_006_72_73 --sequence-name originally_seen_adjacent_010_73_72 \
  --sequence-name newly_added_nonadjacent_014_73_76 --sequence-name newly_added_nonadjacent_021_75_73 \
  --audio-intervention-sequence 72 --audio-intervention-sequence 74 --audio-intervention-sequence 76 \
  --audio-intervention-sequence 72,73 --audio-intervention-sequence 73,72 \
  --audio-intervention-sequence 73,76 --audio-intervention-sequence 75,73
python scripts/evaluate_long_horizon_compositional.py \
  --manifest "${PROJECT_ROOT}/configs/long_horizon_clean_test_v1.json" \
  --output-dir "${RUN_DIR}/audio_interventions/clean" --skip-pipeline1 \
  --pipeline2-checkpoint-path "refined_p2=${P2_REFINED}" --pipeline2-model refined_p2 \
  --include-audio-interventions --device "${DEVICE}" --seed "${SEED}" \
  --sequence-name clean_test_000_len3 --sequence-name clean_test_001_len3 \
  --sequence-name clean_test_002_len5 --sequence-name clean_test_003_len5 \
  --sequence-name clean_test_004_len10 --sequence-name clean_test_005_len10 \
  --audio-intervention-sequence 74,76,75 --audio-intervention-sequence 73,72,74 \
  --audio-intervention-sequence 74,76,75,73,72 --audio-intervention-sequence 72,75,73,76,74 \
  --audio-intervention-sequence 74,76,75,73,72,74,75,76,73,72 \
  --audio-intervention-sequence 72,74,76,73,75,72,76,74,75,73

RETENTION_ARGS=(
  --sequence-name anchor_000_72 --sequence-name anchor_001_73 --sequence-name anchor_002_74
  --sequence-name anchor_003_75 --sequence-name anchor_004_76
  --sequence-name originally_seen_adjacent_006_72_73 --sequence-name originally_seen_adjacent_010_73_72
  --sequence-name originally_seen_adjacent_012_73_74 --sequence-name originally_seen_adjacent_016_74_73
  --sequence-name originally_seen_adjacent_018_74_75 --sequence-name originally_seen_adjacent_022_75_74
  --sequence-name originally_seen_adjacent_024_75_76 --sequence-name originally_seen_adjacent_028_76_75
)
for spec in "p1_original:${P1_ORIGINAL}" "p1_refined:${P1_REFINED}"; do
  label="${spec%%:*}"; path="${spec#*:}"
  python scripts/evaluate_long_horizon_compositional.py \
    --manifest "${PROJECT_ROOT}/configs/complete_pairwise_v1.json" \
    --output-dir "${RUN_DIR}/retention/${label}" --allow-short-primitives "${RETENTION_ARGS[@]}" \
    --pipeline1-controller "${path}" --pipeline1-condition oracle --skip-pipeline2 \
    --device "${DEVICE}" --seed "${SEED}"
done
python scripts/evaluate_long_horizon_compositional.py \
  --manifest "${PROJECT_ROOT}/configs/complete_pairwise_v1.json" \
  --output-dir "${RUN_DIR}/retention/pipeline2" --allow-short-primitives "${RETENTION_ARGS[@]}" --skip-pipeline1 \
  --pipeline2-checkpoint-path "p2_original=${P2_ORIGINAL}" --pipeline2-checkpoint-path "p2_refined=${P2_REFINED}" \
  --pipeline2-model p2_original --pipeline2-model p2_refined --device "${DEVICE}" --seed "${SEED}"

echo "EVALUATION_REPAIRS_COMPLETE=true"
