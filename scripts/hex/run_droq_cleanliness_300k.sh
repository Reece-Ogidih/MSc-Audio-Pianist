#!/usr/bin/env bash
set -euo pipefail

: "${HEX_GPU_INDEX:?Set HEX_GPU_INDEX}"
: "${HEX_SCRATCH:?Set HEX_SCRATCH}"
: "${HEX_IMAGE_TAG:?Set HEX_IMAGE_TAG}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
RUN_NAME="${HEX_RUN_NAME:-droq_cleanliness_sensitive_v1_seed13_300k_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${HEX_SCRATCH}/runs/${RUN_NAME}"
test ! -e "${RUN_DIR}"
mkdir -p "${RUN_DIR}/logs"
git rev-parse HEAD | tee "${RUN_DIR}/git_commit.txt"
date -u +%Y-%m-%dT%H:%M:%SZ | tee "${RUN_DIR}/started_at.txt"

TRAIN_CMD="set -euo pipefail; export PYTHONUNBUFFERED=1; export PYTHONPATH=/app/src:/app/third_party/robopianist; export MUJOCO_GL=\${MUJOCO_GL:-egl}; python - <<'PY'
import platform, torch
print('python', platform.python_version(), flush=True)
print('torch', torch.__version__, 'cuda', torch.version.cuda, flush=True)
print('cuda_available', torch.cuda.is_available(), flush=True)
if not torch.cuda.is_available(): raise SystemExit('CUDA unavailable')
print('cuda_device', torch.cuda.get_device_name(0), flush=True)
PY
python -u scripts/train_droq_general_one_hand_policy.py --timesteps 300000 --seed 13 --lookahead 1 --curriculum sequence_cleanup --sequence-pitches '73;74;75;73,75;75,73;74,75' --sequence-sampling-weights '0.22,0.22,0.22,0.14,0.10,0.10' --stage-name droq_cleanliness_sensitive_v1_seed13_300k --action-mode direct --action-repeat 1 --reward-profile transition_cleanup_sensitive_v1 --checkpoint-freq 50000 --sequence-timing-profile aligned --utd-ratio 4 --batch-size 256 --device cuda --output-dir /workspace/runs/${RUN_NAME}/output 2>&1 | tee /workspace/runs/${RUN_NAME}/logs/train.log"
printf '%s\n' "${TRAIN_CMD}" > "${RUN_DIR}/launch_command.txt"
set +e
hare run --rm --gpus "device=${HEX_GPU_INDEX}" --name "${RUN_NAME}" --user "$(id -u):$(id -g)" -v "${REPO_ROOT}:/app" -v "${HEX_SCRATCH}:/workspace" --workdir /app "${HEX_IMAGE_TAG}" bash -lc "${TRAIN_CMD}"
exit_code=$?
set -e
echo "${exit_code}" > "${RUN_DIR}/exit_code.txt"
date -u +%Y-%m-%dT%H:%M:%SZ | tee "${RUN_DIR}/finished_at.txt"
exit "${exit_code}"
