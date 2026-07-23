#!/usr/bin/env bash
set -euo pipefail

: "${HEX_GPU_INDEX:?Set HEX_GPU_INDEX}"
: "${HEX_SCRATCH:?Set HEX_SCRATCH}"
: "${HEX_IMAGE_TAG:?Set HEX_IMAGE_TAG}"
: "${HEX_DROQ_300K_CHECKPOINT:?Set HEX_DROQ_300K_CHECKPOINT}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
RUN_NAME="${HEX_RUN_NAME:-droq_five_note_expansion_from_300k_seed13_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${HEX_SCRATCH}/runs/${RUN_NAME}"
test ! -e "${RUN_DIR}"
test -f "${HEX_DROQ_300K_CHECKPOINT}"
mkdir -p "${RUN_DIR}/logs"
git rev-parse HEAD | tee "${RUN_DIR}/git_commit.txt"
date -u +%Y-%m-%dT%H:%M:%SZ | tee "${RUN_DIR}/started_at.txt"

TRAIN_CMD=$(cat <<TRAIN
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH=/app/src:/app/third_party/robopianist
export MUJOCO_GL=\${MUJOCO_GL:-egl}
python - <<'PY'
import platform, torch
print('python', platform.python_version(), flush=True)
print('torch', torch.__version__, 'cuda', torch.version.cuda, flush=True)
print('cuda_available', torch.cuda.is_available(), flush=True)
if not torch.cuda.is_available(): raise SystemExit('CUDA unavailable')
print('cuda_device', torch.cuda.get_device_name(0), flush=True)
PY
python -u scripts/train_droq_general_one_hand_policy.py --resume-checkpoint /workspace/${HEX_DROQ_300K_CHECKPOINT#${HEX_SCRATCH}/} --additional-timesteps 200000 --seed 13 --lookahead 1 --midi-min 72 --midi-max 76 --curriculum sequence_cleanup --sequence-pitches '73;74;75;73,75;75,73;74,75;72;76' --sequence-sampling-weights '0.18,0.18,0.18,0.12,0.10,0.10,0.07,0.07' --stage-name ${RUN_NAME}_phase1 --action-mode direct --action-repeat 1 --reward-profile transition_cleanup --checkpoint-freq 100000 --sequence-timing-profile aligned --utd-ratio 4 --batch-size 256 --device cuda --output-dir /workspace/runs/${RUN_NAME}/output 2>&1 | tee /workspace/runs/${RUN_NAME}/logs/phase1.log
PHASE1_CKPT="/workspace/runs/${RUN_NAME}/output/checkpoints/${RUN_NAME}_phase1_droq_sequence_cleanup_lookahead1_directx1_transition_cleanup_seed13_500000/checkpoint_500000_steps.pt"
test -f "\${PHASE1_CKPT}"
python -u scripts/train_droq_general_one_hand_policy.py --resume-checkpoint "\${PHASE1_CKPT}" --additional-timesteps 300000 --seed 13 --lookahead 1 --midi-min 72 --midi-max 76 --curriculum sequence_cleanup --sequence-pitches '73;74;75;72;76;73,75;75,73;74,75;72,73;73,72;75,76;76,75' --sequence-sampling-weights '0.12,0.12,0.12,0.08,0.08,0.10,0.08,0.08,0.07,0.07,0.04,0.04' --stage-name ${RUN_NAME}_phase2 --action-mode direct --action-repeat 1 --reward-profile transition_cleanup --checkpoint-freq 100000 --sequence-timing-profile aligned --utd-ratio 4 --batch-size 256 --device cuda --output-dir /workspace/runs/${RUN_NAME}/output 2>&1 | tee /workspace/runs/${RUN_NAME}/logs/phase2.log
PHASE2_CKPT="/workspace/runs/${RUN_NAME}/output/checkpoints/${RUN_NAME}_phase2_droq_sequence_cleanup_lookahead1_directx1_transition_cleanup_seed13_800000/checkpoint_800000_steps.pt"
test -f "\${PHASE2_CKPT}"
python -u scripts/train_droq_general_one_hand_policy.py --resume-checkpoint "\${PHASE2_CKPT}" --additional-timesteps 200000 --seed 13 --lookahead 1 --midi-min 72 --midi-max 76 --curriculum sequence_cleanup --sequence-pitches '73;74;75;72;76;73,75;75,73;74,75;72,73;73,72;75,76;76,75;72,73,74;74,75,76;72,73,74,75,76;76,75,74,73,72' --sequence-sampling-weights '0.10,0.10,0.10,0.07,0.07,0.08,0.07,0.07,0.06,0.06,0.04,0.04,0.04,0.04,0.03,0.03' --stage-name ${RUN_NAME}_phase3 --action-mode direct --action-repeat 1 --reward-profile transition_cleanup --checkpoint-freq 100000 --sequence-timing-profile aligned --utd-ratio 4 --batch-size 256 --device cuda --output-dir /workspace/runs/${RUN_NAME}/output 2>&1 | tee /workspace/runs/${RUN_NAME}/logs/phase3.log
TRAIN
)
printf '%s\n' "${TRAIN_CMD}" > "${RUN_DIR}/launch_command.txt"
set +e
hare run --rm --gpus "device=${HEX_GPU_INDEX}" --name "${RUN_NAME}" --user "$(id -u):$(id -g)" -v "${REPO_ROOT}:/app" -v "${HEX_SCRATCH}:/workspace" --workdir /app "${HEX_IMAGE_TAG}" bash -lc "${TRAIN_CMD}"
exit_code=$?
set -e
echo "${exit_code}" > "${RUN_DIR}/exit_code.txt"
date -u +%Y-%m-%dT%H:%M:%SZ | tee "${RUN_DIR}/finished_at.txt"
exit "${exit_code}"
