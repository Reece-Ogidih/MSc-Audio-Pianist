#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/reece_dev/msc-audio-pianist}"
RAW_DIR="${RAW_DIR:-${PROJECT_ROOT}/data/real_piano/raw}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/artifacts/real_audio_distribution_shift/prepared_v1}"
EVALUATION_DIR="${EVALUATION_DIR:-${PROJECT_ROOT}/artifacts/real_audio_distribution_shift/evaluation_v1}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/third_party/robopianist:${PYTHONPATH:-}"

python scripts/real_audio/prepare_real_audio_distribution_shift.py \
  --raw-dir "${RAW_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --realizations "${REALIZATIONS:-3}" \
  --takes-per-pitch "${TAKES_PER_PITCH:-3}" \
  --seed "${REAL_AUDIO_SEED:-20260823}"

cat <<EOF
REAL_AUDIO_PREPARED=true
prepared_output=${OUTPUT_DIR}

Next evaluation step:
  PYTHONPATH=${PROJECT_ROOT}/src:${PROJECT_ROOT}/third_party/robopianist \\
  python scripts/real_audio/evaluate_real_audio_distribution_shift.py \\
    --prepared-manifest ${OUTPUT_DIR}/benchmark_realization_manifest.csv \\
    --output-dir ${EVALUATION_DIR} \\
    --benchmark exact_13_retention \\
    --benchmark complete_pairwise \\
    --benchmark clean_unseen_compositions \\
    --include-audio-interventions

Suggested analysis command after evaluation outputs exist:
  PYTHONPATH=${PROJECT_ROOT}/src:${PROJECT_ROOT}/third_party/robopianist \\
  python scripts/real_audio/analyze_real_audio_distribution_shift.py \\
    --evaluation-root ${PROJECT_ROOT}/artifacts/real_audio_distribution_shift/evaluation_v1 \\
    --output-dir ${PROJECT_ROOT}/artifacts/real_audio_distribution_shift/analysis_v1
EOF
