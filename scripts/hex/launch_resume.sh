#!/usr/bin/env bash
set -euo pipefail

GPU_INDEX=""
SCRATCH=""
IMAGE_TAG=""
CHECKPOINT=""
ADDITIONAL_TIMESTEPS=""
RUN_NAME=""
CHECKPOINT_EVERY="100000"
MODE="detached"
CONFIRM="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU_INDEX="${2:?}"; shift 2 ;;
    --scratch) SCRATCH="${2:?}"; shift 2 ;;
    --image) IMAGE_TAG="${2:?}"; shift 2 ;;
    --checkpoint) CHECKPOINT="${2:?}"; shift 2 ;;
    --additional-timesteps) ADDITIONAL_TIMESTEPS="${2:?}"; shift 2 ;;
    --run-name) RUN_NAME="${2:?}"; shift 2 ;;
    --checkpoint-every) CHECKPOINT_EVERY="${2:?}"; shift 2 ;;
    --mode) MODE="${2:?}"; shift 2 ;;
    --confirm-launch) CONFIRM="true"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "${GPU_INDEX}" ]] || { echo "--gpu is required" >&2; exit 2; }
[[ -n "${SCRATCH}" ]] || { echo "--scratch is required" >&2; exit 2; }
[[ -n "${IMAGE_TAG}" ]] || { echo "--image is required" >&2; exit 2; }
[[ -n "${CHECKPOINT}" ]] || { echo "--checkpoint is required" >&2; exit 2; }
[[ -n "${ADDITIONAL_TIMESTEPS}" ]] || { echo "--additional-timesteps is required" >&2; exit 2; }
[[ -n "${RUN_NAME}" ]] || { echo "--run-name is required" >&2; exit 2; }
[[ "${MODE}" =~ ^(tmux|detached|foreground)$ ]] || { echo "--mode must be tmux, detached, or foreground" >&2; exit 2; }
[[ -d "${SCRATCH}" ]] || { echo "Scratch path does not exist: ${SCRATCH}" >&2; exit 2; }
[[ -f "${CHECKPOINT}" ]] || { echo "Checkpoint does not exist: ${CHECKPOINT}" >&2; exit 2; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DIR="${SCRATCH}/runs/${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"
if [[ -e "${RUN_DIR}/.launched" ]]; then
  echo "Run appears to have been launched already: ${RUN_DIR}" >&2
  exit 3
fi

case "${CHECKPOINT}" in
  "${SCRATCH}"/*) CONTAINER_CHECKPOINT="/workspace/${CHECKPOINT#"${SCRATCH}/"}" ;;
  /workspace/*|/app/*) CONTAINER_CHECKPOINT="${CHECKPOINT}" ;;
  *) echo "Checkpoint should be under scratch, /workspace, or /app for container visibility" >&2; exit 2 ;;
esac

CONTAINER_NAME="${RUN_NAME}"
COMMON_ARGS=(
  run --rm --gpus "device=${GPU_INDEX}"
  --name "${CONTAINER_NAME}"
  --user "$(id -u):$(id -g)"
  -v "${REPO_ROOT}:/app"
  -v "${SCRATCH}:/workspace"
  --workdir /app
  "${IMAGE_TAG}"
)
TRAIN_CMD="set -euo pipefail; export PYTHONUNBUFFERED=1; export PYTHONPATH=/app/src:/app/third_party/robopianist; export MUJOCO_GL=\${MUJOCO_GL:-egl}; python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('CUDA is not available; refusing to start training')
print('cuda_device_count', torch.cuda.device_count(), flush=True)
PY
python -u scripts/train_droq_general_one_hand_policy.py --resume-checkpoint '${CONTAINER_CHECKPOINT}' --additional-timesteps '${ADDITIONAL_TIMESTEPS}' --lookahead 1 --curriculum sequence_cleanup --sequence-pitches '73;74;75;73,75;75,73;74,75' --sequence-sampling-weights '0.22,0.22,0.22,0.14,0.10,0.10' --stage-name '${RUN_NAME}' --action-mode direct --action-repeat 1 --reward-profile transition_cleanup --checkpoint-freq '${CHECKPOINT_EVERY}' --sequence-timing-profile aligned --utd-ratio 4 --batch-size 256 --device cuda --output-dir /workspace/runs/${RUN_NAME}/droq 2>&1 | tee /workspace/runs/${RUN_NAME}/logs/train.log"

echo "run_name=${RUN_NAME}"
echo "run_dir=${RUN_DIR}"
echo "container_checkpoint=${CONTAINER_CHECKPOINT}"
printf 'resolved_command='
printf '%q ' hare "${COMMON_ARGS[@]}" bash -lc "${TRAIN_CMD}"
printf '\n'

if [[ "${CONFIRM}" != "true" ]]; then
  echo "Dry-run only. Re-run with --confirm-launch to start."
  exit 0
fi
touch "${RUN_DIR}/.launched"

case "${MODE}" in
  foreground)
    hare "${COMMON_ARGS[@]}" bash -lc "${TRAIN_CMD}"
    ;;
  detached)
    hare "${COMMON_ARGS[@]:0:1}" -d "${COMMON_ARGS[@]:1}" bash -lc "${TRAIN_CMD}"
    ;;
  tmux)
    tmux new-session -d -s "${RUN_NAME}" "$(printf '%q ' hare "${COMMON_ARGS[@]}" bash -lc "${TRAIN_CMD}")"
    ;;
esac
