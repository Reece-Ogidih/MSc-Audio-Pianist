#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/homes/rgkgo20/msc-audio-pianist}"
SCRATCH="${SCRATCH:-/mnt/fast1/rgkgo20/msc-audio-pianist}"
STATE="${SCRATCH}/orchestration/paironly_and_evidence_repairs/latest.env"
[[ -f "${STATE}" ]] && source "${STATE}"
P1_OUT="${SCRATCH}/runs/general_one_hand/droq/pipeline1_symbolic_paironly_complete_v1"
P2_OUT="${SCRATCH}/experiments/pipeline2_direct_audio/pipeline2_seed13_paironly_complete_v1"
EVAL_OUT="${SCRATCH}/experiments/paircomplete_refined_evaluation_repairs/refined_evidence_repairs_v1"

echo "=== containers ==="
hare ps -a 2>/dev/null | grep -E "CONTAINER|${P1_CONTAINER:-pipeline1-paironly}|${P2_CONTAINER:-pipeline2-paironly}|${EVAL_CONTAINER:-evidence-repairs}" || true
echo "=== GPUs ==="
nvidia-smi
echo "=== Pipeline 2 pair-only ==="
[[ -f "${P2_OUT}/launch_metadata_start.json" ]] && cat "${P2_OUT}/launch_metadata_start.json" || true
[[ -f "${P2_OUT}/logs/train.log" ]] && tail -60 "${P2_OUT}/logs/train.log" || true
find "${P2_OUT}/lightweight_checkpoints" -type f -name '*.pt' -printf '%p %s bytes\n' 2>/dev/null | sort || true
echo "=== Pipeline 1 pair-only ==="
[[ -f "${P1_OUT}/launch_metadata_start.json" ]] && cat "${P1_OUT}/launch_metadata_start.json" || true
[[ -f "${P1_OUT}/logs/train.log" ]] && tail -60 "${P1_OUT}/logs/train.log" || true
find "${P1_OUT}/lightweight_checkpoints" -type f -name '*.pt' -printf '%p %s bytes\n' 2>/dev/null | sort || true
echo "=== Evidence repair ==="
find "${EVAL_OUT}" -maxdepth 3 -type f -printf '%p %s bytes\n' 2>/dev/null | sort | tail -80 || true
if [[ -d "${EVAL_OUT}" ]]; then
  grep -R "Basic Pitch is not available" "${EVAL_OUT}" 2>/dev/null || echo "basic_pitch_unavailable_placeholder_absent=true"
fi
echo "=== scratch ==="
df -h /mnt/fast1
