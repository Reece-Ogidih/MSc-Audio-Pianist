#!/usr/bin/env bash
set -euo pipefail

# One-command Hex launcher for the final synthetic Base vs Pair-only vs
# Pair+Horizon evaluation. It performs all validation before starting any
# experiment container.

REPO="${REPO:-/homes/rgkgo20/msc-audio-pianist}"
SCRATCH="${SCRATCH:-/mnt/fast1/rgkgo20/msc-audio-pianist}"
IMAGE="${IMAGE:-rgkgo20/msc-audio-pianist:d88e29b}"
EXPECTED_BRANCH="${EXPECTED_BRANCH:-pipeline2-direct-audio}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:-}"
MIN_FREE_GB="${MIN_FREE_GB:-80}"
MAX_IDLE_MEMORY_MIB="${MAX_IDLE_MEMORY_MIB:-100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH}/runs/paironly_final_evaluation}"
RUN_NAME="${RUN_NAME:-paironly_final_synthetic_evaluation_v1}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
STATE_DIR="${SCRATCH}/orchestration/paironly_final_evaluation"

P1_BASE="${P1_BASE:-${REPO}/artifacts/frozen_models/five_note_symbolic_controller_v1/checkpoint_800000_steps.pt}"
P1_PAIRONLY="${P1_PAIRONLY:-${SCRATCH}/runs/general_one_hand/droq/pipeline1_symbolic_paironly_complete_v1/lightweight_checkpoints/pipeline1_symbolic_paironly_complete_v1_droq_sequence_cleanup_lookahead1_directx1_transition_cleanup_sensitive_v1_seed13_500000/checkpoint_500000_steps.pt}"
P1_PAIRHORIZON="${P1_PAIRHORIZON:-${SCRATCH}/runs/general_one_hand/droq/pipeline1_symbolic_paircomplete_longhorizon_v1/lightweight_checkpoints/pipeline1_symbolic_paircomplete_longhorizon_v1_droq_sequence_cleanup_lookahead1_directx1_transition_cleanup_sensitive_v1_seed13_500000/checkpoint_500000_steps.pt}"
P2_BASE="${P2_BASE:-${SCRATCH}/experiments/pipeline2_direct_audio/pipeline2_direct_audio_droq_v1_seed13_1m_retry1/lightweight_checkpoints/checkpoint_1000000_steps.pt}"
P2_BASE_FULL="${P2_BASE_FULL:-${SCRATCH}/experiments/pipeline2_direct_audio/pipeline2_direct_audio_droq_v1_seed13_1m_retry1/checkpoints/full_checkpoint_1000000_steps.pt}"
P2_PAIRONLY="${P2_PAIRONLY:-${SCRATCH}/experiments/pipeline2_direct_audio/pipeline2_seed13_paironly_complete_v1/lightweight_checkpoints/checkpoint_500000_steps.pt}"
P2_PAIRHORIZON="${P2_PAIRHORIZON:-${SCRATCH}/experiments/pipeline2_direct_audio/pipeline2_seed13_paircomplete_longhorizon_v1/lightweight_checkpoints/checkpoint_500000_steps.pt}"
SOUNDFONT="${SOUNDFONT:-${REPO}/third_party/robopianist/robopianist/soundfonts/TimGM6mb.sf2}"

fail() { echo "PRELAUNCH_FAILURE: $*" >&2; exit 2; }
require_hash() {
  local label="$1" path="$2" expected="$3" actual
  [[ -s "${path}" ]] || fail "${label} missing or empty: ${path}"
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || fail "${label} SHA-256 mismatch: expected ${expected}, got ${actual} at ${path}"
  echo "verified_${label}_path=${path}"
  echo "verified_${label}_sha256=${actual}"
}
report_hash() {
  local label="$1" path="$2" actual
  [[ -s "${path}" ]] || fail "${label} missing or empty: ${path}"
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  echo "verified_${label}_path=${path}"
  echo "verified_${label}_sha256=${actual}"
}
require_empty_destination() {
  local path="$1"
  if [[ -e "${path}" ]] && [[ -n "$(find "${path}" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
    fail "Output destination is already non-empty: ${path}"
  fi
}

[[ -d "${REPO}/.git" ]] || fail "Repository not found: ${REPO}"
[[ -d "${SCRATCH}" ]] || fail "Scratch root not found: ${SCRATCH}"
cd "${REPO}"
git config --global --add safe.directory "${REPO}" >/dev/null 2>&1 || true
branch="$(git branch --show-current)"
commit="$(git rev-parse HEAD)"
short="$(git rev-parse --short HEAD)"
[[ "${branch}" == "${EXPECTED_BRANCH}" ]] || fail "Expected branch ${EXPECTED_BRANCH}, got ${branch}"
if [[ -n "${EXPECTED_COMMIT}" && "${commit}" != "${EXPECTED_COMMIT}" ]]; then
  fail "Expected commit ${EXPECTED_COMMIT}, got ${commit}"
fi
git diff --quiet || fail "Repository has tracked working-tree modifications"
git diff --cached --quiet || fail "Repository has staged modifications"
minimum_safe_commit="4c222b13840c76e334b51db8741af7d0d76e71eb"
git merge-base --is-ancestor "${minimum_safe_commit}" "${commit}" || \
  fail "HEAD ${commit} does not contain required launch fixes from ${minimum_safe_commit}"
echo "repository_commit=${commit}"
echo "repository_branch=${branch}"

require_hash p1_base "${P1_BASE}" 927c1050c08769c49568013ead0c69d69d4bd19ff23eb632e89bd89fb735ac4c
require_hash p1_paironly "${P1_PAIRONLY}" d32b1c64c549df03c80c464d4108908fcca6eab53695eebafb475c7ccbef1670
require_hash p1_pairhorizon "${P1_PAIRHORIZON}" 6d7bb442e6a1726a2c268123cb6ea6831929a6dd7b782bca4ad1b7d226a3f966
report_hash p2_base_lightweight "${P2_BASE}"
require_hash p2_base_full "${P2_BASE_FULL}" 5ac0420d6a0ef0f7e4644bfdc6520dbb36eec761b9d33f28a4a5c1e9b19ab602
require_hash p2_paironly "${P2_PAIRONLY}" 11f65ffc14976db055806599ca53e78a19a75cd74f024374a1effb6f31a98df1
require_hash p2_pairhorizon "${P2_PAIRHORIZON}" 3fb2ae99d4926b635edcd9a1fb9406b8f515b66a4f4c3a60d102694a2c2abd43
[[ -s "${SOUNDFONT}" ]] || fail "Soundfont missing or empty: ${SOUNDFONT}"
echo "soundfont_path=${SOUNDFONT}"
echo "soundfont_sha256=$(sha256sum "${SOUNDFONT}" | awk '{print $1}')"

for manifest in \
  configs/complete_pairwise_v1.json \
  configs/long_horizon_compositional_v1.json \
  configs/long_horizon_clean_test_v1.json \
  configs/long_horizon_extrapolation_v1.json
do
  [[ -s "${REPO}/${manifest}" ]] || fail "Benchmark manifest missing: ${REPO}/${manifest}"
  echo "manifest_${manifest//\//_}_sha256=$(sha256sum "${REPO}/${manifest}" | awk '{print $1}')"
done

python - "${REPO}/configs/complete_pairwise_v1.json" <<'PY'
import json
import sys
from itertools import product
payload = json.load(open(sys.argv[1], encoding="utf-8"))
seqs = [tuple(item["pitches"]) for item in payload["sequences"]]
anchors = {(pitch,) for pitch in range(72, 77)}
pairs = set(product(range(72, 77), repeat=2))
assert len(seqs) == 30, len(seqs)
assert set(seqs[:5]) == anchors, seqs[:5]
assert set(seqs[5:]) == pairs and len(seqs[5:]) == 25
assert max(map(len, seqs)) <= 2
print("complete_pairwise_manifest_valid=true")
PY

df -h "${SCRATCH}"
free_kb="$(df -Pk "${SCRATCH}" | awk 'NR==2 {print $4}')"
(( free_kb >= MIN_FREE_GB * 1024 * 1024 )) || fail "Less than ${MIN_FREE_GB} GiB free on scratch"
require_empty_destination "${RUN_DIR}"

echo "gpu_inventory:"
nvidia-smi
declare -A occupied=()
while IFS=',' read -r uuid pid; do
  uuid="${uuid// /}"
  pid="${pid// /}"
  [[ -n "${uuid}" && -n "${pid}" ]] && occupied["${uuid}"]=1
done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null || true)
free_gpus=()
while IFS=',' read -r index uuid memory; do
  index="${index// /}"
  uuid="${uuid// /}"
  memory="${memory// /}"
  if [[ -z "${occupied[${uuid}]:-}" ]] && (( memory <= MAX_IDLE_MEMORY_MIB )); then
    free_gpus+=("${index}")
  fi
done < <(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits)
(( ${#free_gpus[@]} >= 3 )) || fail "Need three idle GPUs; eligible GPUs: ${free_gpus[*]:-none}"
GPU_P1="${free_gpus[0]}"
GPU_P2="${free_gpus[1]}"
GPU_AUDIO="${free_gpus[2]}"
echo "gpu_assignment_p1=${GPU_P1}"
echo "gpu_assignment_p2=${GPU_P2}"
echo "gpu_assignment_audio_interventions=${GPU_AUDIO}"

echo "Running real Basic Pitch/CUDA/RoboPianist smoke before final evaluation launch"
NUMBA_CACHE_DIR=/tmp/ala-numba-cache XDG_CACHE_HOME=/tmp/ala-xdg-cache \
  bash scripts/hex/smoke_test.sh --gpu "${GPU_P1}" --scratch "${SCRATCH}" --image "${IMAGE}"

mkdir -p "${RUN_DIR}" "${STATE_DIR}"
python - "${RUN_DIR}/evaluation_launch_manifest.json" <<PY
import json
import subprocess
from pathlib import Path
payload = {
    "run_name": "${RUN_NAME}",
    "run_dir": "${RUN_DIR}",
    "repository_commit": "${commit}",
    "repository_branch": "${branch}",
    "image": "${IMAGE}",
    "model_identities": {
        "p1_base": "${P1_BASE}",
        "p1_paironly": "${P1_PAIRONLY}",
        "p1_pairhorizon": "${P1_PAIRHORIZON}",
        "p2_base": "${P2_BASE}",
        "p2_base_full_resume_source": "${P2_BASE_FULL}",
        "p2_paironly": "${P2_PAIRONLY}",
        "p2_pairhorizon": "${P2_PAIRHORIZON}",
    },
    "benchmarks": [
        "configs/complete_pairwise_v1.json",
        "configs/long_horizon_compositional_v1.json",
        "configs/long_horizon_clean_test_v1.json",
        "configs/long_horizon_extrapolation_v1.json",
        "exact_13_retention_subset_from_complete_pairwise_v1",
        "representative_pipeline2_audio_interventions",
    ],
    "pipeline1_modes": ["oracle", "basic_pitch"],
    "pipeline2_audio_modes": ["correct", "zero", "mismatched"],
    "basic_pitch_backend": "real Basic Pitch smoke required before launch",
    "soundfont": "${SOUNDFONT}",
    "gpu_assignment": {"p1_all": "${GPU_P1}", "p2_all": "${GPU_P2}", "p2_audio_interventions": "${GPU_AUDIO}"},
}
Path("${RUN_DIR}/evaluation_launch_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
PY

common_env=(
  -e MUJOCO_GL=egl
  -e PYTHONUNBUFFERED=1
  -e GIT_CONFIG_GLOBAL=/tmp/.gitconfig
  -e NUMBA_CACHE_DIR=/tmp/ala-numba-cache
  -e XDG_CACHE_HOME=/tmp/ala-xdg-cache
  -e SOUNDFONT=/app/third_party/robopianist/robopianist/soundfonts/TimGM6mb.sf2
  -e OUTPUT_ROOT=/workspace/runs/paironly_final_evaluation
  -e RUN_NAME="${RUN_NAME}"
)
launch_group() {
  local group="$1" gpu="$2" name="paironly-final-${group}-${short}"
  hare run -d --name "${name}" --gpus "device=${gpu}" --user "$(id -u):$(id -g)" \
    "${common_env[@]}" -e GROUP="${group}" \
    -v "${REPO}:/app" -v "${SCRATCH}:/workspace" --workdir /app \
    "${IMAGE}" bash /app/scripts/hex/run_paironly_final_evaluation_group.sh
  echo "launched_${group}_container=${name}"
}

launch_group p1_all "${GPU_P1}"
launch_group p2_all "${GPU_P2}"
launch_group p2_audio_interventions "${GPU_AUDIO}"

cat >"${STATE_DIR}/latest.env" <<EOF
REPO=${REPO}
SCRATCH=${SCRATCH}
COMMIT=${commit}
IMAGE=${IMAGE}
RUN_NAME=${RUN_NAME}
RUN_DIR=${RUN_DIR}
GPU_P1=${GPU_P1}
GPU_P2=${GPU_P2}
GPU_AUDIO=${GPU_AUDIO}
P1_CONTAINER=paironly-final-p1_all-${short}
P2_CONTAINER=paironly-final-p2_all-${short}
AUDIO_CONTAINER=paironly-final-p2_audio_interventions-${short}
EOF

echo "Waiting 90 seconds before startup verification"
sleep 90
bash scripts/hex/status_paironly_final_evaluation.sh
echo "Later status command: cd ${REPO} && bash scripts/hex/status_paironly_final_evaluation.sh"
echo "PAIRONLY_FINAL_EVALUATION_LAUNCHED=true"
