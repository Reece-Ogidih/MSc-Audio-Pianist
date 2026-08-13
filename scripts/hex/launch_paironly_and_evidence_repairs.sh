#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/homes/rgkgo20/msc-audio-pianist}"
SCRATCH="${SCRATCH:-/mnt/fast1/rgkgo20/msc-audio-pianist}"
IMAGE="${IMAGE:-rgkgo20/msc-audio-pianist:d88e29b}"
EXPECTED_BRANCH="pipeline2-direct-audio"
MIN_FREE_GB="${MIN_FREE_GB:-35}"
MAX_IDLE_MEMORY_MIB="${MAX_IDLE_MEMORY_MIB:-100}"
P1_BASE="${REPO}/artifacts/frozen_models/five_note_symbolic_controller_v1/checkpoint_800000_steps.pt"
P2_BASE="${SCRATCH}/experiments/pipeline2_direct_audio/pipeline2_direct_audio_droq_v1_seed13_1m_retry1/checkpoints/full_checkpoint_1000000_steps.pt"
P1_REFINED="${SCRATCH}/runs/general_one_hand/droq/pipeline1_symbolic_paircomplete_longhorizon_v1/lightweight_checkpoints/pipeline1_symbolic_paircomplete_longhorizon_v1_droq_sequence_cleanup_lookahead1_directx1_transition_cleanup_sensitive_v1_seed13_500000/checkpoint_500000_steps.pt"
P2_REFINED="${SCRATCH}/experiments/pipeline2_direct_audio/pipeline2_seed13_paircomplete_longhorizon_v1/lightweight_checkpoints/checkpoint_500000_steps.pt"
P1_OUT="${SCRATCH}/runs/general_one_hand/droq/pipeline1_symbolic_paironly_complete_v1"
P2_OUT="${SCRATCH}/experiments/pipeline2_direct_audio/pipeline2_seed13_paironly_complete_v1"
EVAL_OUT="${SCRATCH}/experiments/paircomplete_refined_evaluation_repairs/refined_evidence_repairs_v1"
STATE_DIR="${SCRATCH}/orchestration/paironly_and_evidence_repairs"

fail() { echo "PRELAUNCH_FAILURE: $*" >&2; exit 2; }
require_hash() {
  local path="$1" expected="$2" actual
  [[ -s "${path}" ]] || fail "Required checkpoint missing or empty: ${path}"
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || fail "SHA-256 mismatch for ${path}: ${actual}"
  echo "verified_checkpoint=${path} sha256=${actual}"
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
branch="$(git branch --show-current)"
[[ "${branch}" == "${EXPECTED_BRANCH}" ]] || fail "Expected branch ${EXPECTED_BRANCH}, got ${branch}"
git diff --quiet || fail "Repository has tracked working-tree modifications"
git diff --cached --quiet || fail "Repository has staged modifications"
commit="$(git rev-parse HEAD)"
short="$(git rev-parse --short HEAD)"
minimum_safe_commit="3dfe049ec4eba05549941aec6c5ed792707fef50"
git merge-base --is-ancestor "${minimum_safe_commit}" "${commit}" || \
  fail "HEAD ${commit} does not contain required launch fixes from ${minimum_safe_commit}"
echo "repository_commit=${commit}"
echo "repository_branch=${branch}"

MANIFEST="${REPO}/configs/pair_only_complete_v1.json"
[[ -s "${MANIFEST}" ]] || fail "Pair-only manifest missing: ${MANIFEST}"
MANIFEST_SHA="$(sha256sum "${MANIFEST}" | awk '{print $1}')"
python - "${MANIFEST}" <<'PY'
import json, sys
from itertools import product
payload=json.load(open(sys.argv[1], encoding="utf-8"))
seqs=[tuple(item["pitches"]) for item in payload["sequences"]]
anchors={(p,) for p in range(72,77)}
pairs=set(product(range(72,77), repeat=2))
assert len(seqs)==30 and set(seqs[:5])==anchors, "manifest must contain five anchors"
assert set(seqs[5:])==pairs and len(seqs[5:])==25, "manifest must contain all 25 ordered pairs"
assert max(map(len,seqs))<=2, "pair-only manifest contains a sequence longer than two"
print("pair_only_manifest_valid=true")
PY
echo "pair_only_manifest_sha256=${MANIFEST_SHA}"

require_hash "${P1_BASE}" 927c1050c08769c49568013ead0c69d69d4bd19ff23eb632e89bd89fb735ac4c
require_hash "${P2_BASE}" 5ac0420d6a0ef0f7e4644bfdc6520dbb36eec761b9d33f28a4a5c1e9b19ab602
require_hash "${P1_REFINED}" 6d7bb442e6a1726a2c268123cb6ea6831929a6dd7b782bca4ad1b7d226a3f966
require_hash "${P2_REFINED}" 3fb2ae99d4926b635edcd9a1fb9406b8f515b66a4f4c3a60d102694a2c2abd43

df -h /mnt/fast1
free_kb="$(df -Pk "${SCRATCH}" | awk 'NR==2 {print $4}')"
(( free_kb >= MIN_FREE_GB * 1024 * 1024 )) || fail "Less than ${MIN_FREE_GB} GiB free on scratch"
require_empty_destination "${P1_OUT}"
require_empty_destination "${P2_OUT}"
require_empty_destination "${EVAL_OUT}"

echo "gpu_inventory:"
nvidia-smi
declare -A occupied=()
while IFS=',' read -r uuid pid; do
  uuid="${uuid// /}"; pid="${pid// /}"
  [[ -n "${uuid}" && -n "${pid}" ]] && occupied["${uuid}"]=1
done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null || true)
free_gpus=()
while IFS=',' read -r index uuid memory; do
  index="${index// /}"; uuid="${uuid// /}"; memory="${memory// /}"
  if [[ -z "${occupied[${uuid}]:-}" ]] && (( memory <= MAX_IDLE_MEMORY_MIB )); then
    free_gpus+=("${index}")
  fi
done < <(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits)
(( ${#free_gpus[@]} >= 3 )) || fail "Need three idle GPUs; eligible GPUs: ${free_gpus[*]:-none}"
GPU_P2="${free_gpus[0]}"; GPU_P1="${free_gpus[1]}"; GPU_EVAL="${free_gpus[2]}"
echo "gpu_assignment_p2=${GPU_P2}"
echo "gpu_assignment_p1=${GPU_P1}"
echo "gpu_assignment_evaluation=${GPU_EVAL}"

echo "Running real Basic Pitch/CUDA/RoboPianist smoke before launch"
NUMBA_CACHE_DIR=/tmp/ala-numba-cache XDG_CACHE_HOME=/tmp/ala-xdg-cache \
  bash scripts/hex/smoke_test.sh --gpu "${GPU_P2}" --scratch "${SCRATCH}" --image "${IMAGE}"

P2_CONTAINER="pipeline2-paironly-${short}"
P1_CONTAINER="pipeline1-paironly-${short}"
EVAL_CONTAINER="evidence-repairs-${short}"
common_env=(
  -e MUJOCO_GL=egl -e PYTHONUNBUFFERED=1 -e GIT_CONFIG_GLOBAL=/tmp/.gitconfig
  -e NUMBA_CACHE_DIR=/tmp/ala-numba-cache -e XDG_CACHE_HOME=/tmp/ala-xdg-cache
  -e SOUNDFONT=/app/third_party/robopianist/robopianist/soundfonts/TimGM6mb.sf2
)
launch() {
  local name="$1" gpu="$2" command="$3"
  hare run -d --name "${name}" --gpus "device=${gpu}" --user "$(id -u):$(id -g)" \
    "${common_env[@]}" -v "${REPO}:/app" -v "${SCRATCH}:/workspace" --workdir /app \
    "${IMAGE}" bash -lc "${command}"
}

launch "${P2_CONTAINER}" "${GPU_P2}" "scripts/hex/run_pipeline2_paironly_refinement.sh"
launch "${P1_CONTAINER}" "${GPU_P1}" "scripts/hex/run_pipeline1_paironly_refinement.sh"
launch "${EVAL_CONTAINER}" "${GPU_EVAL}" "scripts/hex/evaluate_refined_evidence_repairs.sh"

mkdir -p "${STATE_DIR}"
cat >"${STATE_DIR}/latest.env" <<EOF
REPO=${REPO}
SCRATCH=${SCRATCH}
COMMIT=${commit}
IMAGE=${IMAGE}
GPU_P2=${GPU_P2}
GPU_P1=${GPU_P1}
GPU_EVAL=${GPU_EVAL}
P2_CONTAINER=${P2_CONTAINER}
P1_CONTAINER=${P1_CONTAINER}
EVAL_CONTAINER=${EVAL_CONTAINER}
EOF

echo "Waiting 105 seconds before startup verification"
sleep 105
bash scripts/hex/status_paironly_and_evidence_repairs.sh

for log in "${P2_OUT}/logs/train.log" "${P1_OUT}/logs/train.log"; do
  [[ -f "${log}" ]] || { echo "STARTUP_WARNING: log not yet present: ${log}"; continue; }
  grep -q "Traceback (most recent call last)" "${log}" && echo "STARTUP_FAILURE_DETECTED=${log}" || true
done
if [[ -d "${EVAL_OUT}" ]] && grep -Rqs "Basic Pitch is not available" "${EVAL_OUT}"; then
  echo "STARTUP_FAILURE_DETECTED=Basic Pitch unavailable placeholder appeared"
fi
echo "Later status command: cd ${REPO} && bash scripts/hex/status_paironly_and_evidence_repairs.sh"
echo "PAIR_ONLY_AND_EVIDENCE_LAUNCH_COMPLETE=true"
