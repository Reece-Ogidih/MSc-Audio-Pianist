#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/homes/rgkgo20/msc-audio-pianist}"
SCRATCH="${SCRATCH:-/mnt/fast1/rgkgo20/msc-audio-pianist}"
STATE="${SCRATCH}/orchestration/paironly_final_evaluation/latest.env"
[[ -f "${STATE}" ]] && source "${STATE}"
RUN_DIR="${RUN_DIR:-${SCRATCH}/runs/paironly_final_evaluation/${RUN_NAME:-paironly_final_synthetic_evaluation_v1}}"

echo "=== containers ==="
hare ps -a 2>/dev/null | grep -E "CONTAINER|${P1_CONTAINER:-paironly-final-p1_all}|${P2_CONTAINER:-paironly-final-p2_all}|${AUDIO_CONTAINER:-paironly-final-p2_audio_interventions}" || true
echo "=== GPUs ==="
nvidia-smi || true
echo "=== launch manifest ==="
[[ -f "${RUN_DIR}/evaluation_launch_manifest.json" ]] && cat "${RUN_DIR}/evaluation_launch_manifest.json" || true
echo "=== logs ==="
find "${RUN_DIR}/logs" -maxdepth 1 -type f -printf '%p %s bytes\n' 2>/dev/null | sort || true
for log in $(find "${RUN_DIR}/logs" -maxdepth 1 -type f -name '*.log' 2>/dev/null | sort); do
  echo "--- ${log} ---"
  tail -40 "${log}" || true
done
echo "=== output summaries ==="
find "${RUN_DIR}" -path '*/long_horizon_summary.json' -o -path '*/long_horizon_model_summary.csv' 2>/dev/null | sort || true
echo "=== failure scan ==="
if [[ -d "${RUN_DIR}" ]]; then
  grep -R "Basic Pitch is not available" "${RUN_DIR}" 2>/dev/null || echo "basic_pitch_unavailable_placeholder_absent=true"
  grep -R "Traceback (most recent call last)" "${RUN_DIR}/logs" 2>/dev/null || echo "traceback_absent_from_logs=true"
fi
echo "=== scratch ==="
df -h "${SCRATCH}" || true
