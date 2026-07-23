#!/usr/bin/env bash
set -euo pipefail

: "${HEX_GPU_INDEX:?Set HEX_GPU_INDEX}"
: "${HEX_SCRATCH:?Set HEX_SCRATCH}"
: "${HEX_IMAGE_TAG:?Set HEX_IMAGE_TAG}"
: "${CONDITION_ID:?Set CONDITION_ID}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_DIR="${REPO_ROOT}/configs/five_note_factorial_1m"
MANIFEST="${CONFIG_DIR}/factorial_manifest_seed13.json"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/third_party/robopianist"
python scripts/validate_five_note_factorial_configs.py --config-dir "${CONFIG_DIR}" >/dev/null

CONFIG_NAME="$(python - <<'PY'
import json, os
from pathlib import Path
manifest=json.loads(Path("configs/five_note_factorial_1m/factorial_manifest_seed13.json").read_text())
cid=os.environ["CONDITION_ID"]
for condition in manifest["conditions"]:
    if condition["condition_id"] == cid:
        print(condition["training_config"])
        break
else:
    raise SystemExit(f"Unknown CONDITION_ID={cid}")
PY
)"
CONFIG_PATH="${CONFIG_DIR}/${CONFIG_NAME}"

readarray -t CONFIG_VALUES < <(python - "${CONFIG_PATH}" <<'PY'
import json, sys
from pathlib import Path
cfg=json.loads(Path(sys.argv[1]).read_text())
seq=";".join(",".join(str(p) for p in s) for s in cfg["sequence_pitches"])
weights=",".join(f"{float(w):g}" for w in cfg["sequence_sampling_weights"])
print(cfg["algorithm"])
print(cfg["reward_profile"])
print(cfg["run_name"])
print(cfg["timesteps"])
print(seq)
print(weights)
print(",".join(str(step) for step in cfg["full_resumable_checkpoint_steps"]))
PY
)
ALGORITHM="${CONFIG_VALUES[0]}"
REWARD_PROFILE="${CONFIG_VALUES[1]}"
BASE_RUN_NAME="${CONFIG_VALUES[2]}"
TIMESTEPS="${CONFIG_VALUES[3]}"
SEQUENCE_PITCHES="${CONFIG_VALUES[4]}"
SEQUENCE_WEIGHTS="${CONFIG_VALUES[5]}"
FULL_STEPS="${CONFIG_VALUES[6]}"

EXPECTED_COMMIT="$(python - <<'PY'
import json
from pathlib import Path
print(json.loads(Path("configs/five_note_factorial_1m/factorial_manifest_seed13.json").read_text())["required_code_commit"])
PY
)"
CURRENT_COMMIT="$(git rev-parse HEAD)"
if [[ "${EXPECTED_COMMIT}" != "TO_BE_FILLED_BY_COMMIT_AMEND" && "${CURRENT_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
  echo "Commit mismatch: current=${CURRENT_COMMIT} expected=${EXPECTED_COMMIT}" >&2
  exit 1
fi

RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="${HEX_RUN_NAME:-${BASE_RUN_NAME}_${RUN_STAMP}}"
RUN_ROOT="${HEX_SCRATCH}/five_note_factorial_1m/${CONDITION_ID}"
RUN_DIR="${RUN_ROOT}/${RUN_NAME}"
if [[ -e "${RUN_DIR}" ]]; then
  echo "Refusing to overwrite existing run dir: ${RUN_DIR}" >&2
  exit 1
fi
mkdir -p "${RUN_DIR}/logs"

{
  echo "hostname=$(hostname)"
  echo "utc_date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_commit=${CURRENT_COMMIT}"
  echo "git_status=$(git status --short --branch | tr '\n' ';')"
  echo "condition_id=${CONDITION_ID}"
  echo "algorithm=${ALGORITHM}"
  echo "reward_profile=${REWARD_PROFILE}"
  echo "hex_gpu_index=${HEX_GPU_INDEX}"
  echo "hex_image_tag=${HEX_IMAGE_TAG}"
  echo "hex_scratch=${HEX_SCRATCH}"
} | tee "${RUN_DIR}/resolved_metadata.env"
cp "${CONFIG_PATH}" "${RUN_DIR}/resolved_training_config.json"
cp "${CONFIG_DIR}/five_note_curriculum_v1.json" "${RUN_DIR}/five_note_curriculum_v1.json"
cp "${MANIFEST}" "${RUN_DIR}/factorial_manifest_seed13.json"
date -u +%Y-%m-%dT%H:%M:%SZ | tee "${RUN_DIR}/started_at.txt"

TRAIN_SCRIPT="scripts/train_droq_general_one_hand_policy.py"
DEVICE_ARGS="--device cuda --utd-ratio 4 --batch-size 256"
if [[ "${ALGORITHM}" == "sac" ]]; then
  TRAIN_SCRIPT="scripts/train_general_one_hand_policy.py"
  DEVICE_ARGS="--device cuda"
fi

TRAIN_CMD=$(cat <<TRAIN
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH=/app/src:/app/third_party/robopianist
export MUJOCO_GL=\${MUJOCO_GL:-egl}
cd /app
python - <<'PY'
import platform, torch, mujoco
print('python', platform.python_version(), flush=True)
print('torch', torch.__version__, 'cuda', torch.version.cuda, flush=True)
print('mujoco', mujoco.__version__, flush=True)
print('cuda_available', torch.cuda.is_available(), flush=True)
if not torch.cuda.is_available(): raise SystemExit('CUDA unavailable')
print('cuda_device_count', torch.cuda.device_count(), flush=True)
print('cuda_device', torch.cuda.get_device_name(0), flush=True)
PY
df -h /workspace | tee /workspace/five_note_factorial_1m/${CONDITION_ID}/${RUN_NAME}/logs/disk_before.txt
(while true; do date -u +%Y-%m-%dT%H:%M:%SZ; nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu --format=csv,noheader,nounits || true; ps -o pid,%cpu,%mem,rss,cmd -C python || true; du -sh /workspace/five_note_factorial_1m/${CONDITION_ID}/${RUN_NAME} || true; sleep 60; done) > /workspace/five_note_factorial_1m/${CONDITION_ID}/${RUN_NAME}/resource_usage.csv 2>/workspace/five_note_factorial_1m/${CONDITION_ID}/${RUN_NAME}/logs/resource_usage.warn &
RESOURCE_PID=\$!
set +e
python -u ${TRAIN_SCRIPT} --timesteps ${TIMESTEPS} --seed 13 --lookahead 1 --midi-min 72 --midi-max 76 --curriculum sequence_cleanup --sequence-pitches '${SEQUENCE_PITCHES}' --sequence-sampling-weights '${SEQUENCE_WEIGHTS}' --stage-name ${BASE_RUN_NAME} --action-mode direct --action-repeat 1 --reward-profile ${REWARD_PROFILE} --checkpoint-freq 0 --lightweight-checkpoint-freq 100000 --full-checkpoint-steps ${FULL_STEPS} --rolling-full-checkpoint --sequence-timing-profile aligned \${DEVICE_ARGS} --output-dir /workspace/five_note_factorial_1m/${CONDITION_ID}/${RUN_NAME}/output 2>&1 | tee /workspace/five_note_factorial_1m/${CONDITION_ID}/${RUN_NAME}/logs/train.log
EXIT_CODE=\${PIPESTATUS[0]}
set -e
kill "\${RESOURCE_PID}" 2>/dev/null || true
echo "\${EXIT_CODE}" > /workspace/five_note_factorial_1m/${CONDITION_ID}/${RUN_NAME}/exit_code.txt
date -u +%Y-%m-%dT%H:%M:%SZ > /workspace/five_note_factorial_1m/${CONDITION_ID}/${RUN_NAME}/finished_at.txt
if [[ "\${EXIT_CODE}" -eq 0 && "\${AUTO_EVALUATE:-0}" == "1" ]]; then
  python -u scripts/evaluate_five_note_factorial_checkpoint_sweep.py --run-dir /workspace/five_note_factorial_1m/${CONDITION_ID}/${RUN_NAME} --config /workspace/five_note_factorial_1m/${CONDITION_ID}/${RUN_NAME}/resolved_training_config.json --output-dir /workspace/five_note_factorial_1m/${CONDITION_ID}/${RUN_NAME}/evaluation 2>&1 | tee /workspace/five_note_factorial_1m/${CONDITION_ID}/${RUN_NAME}/logs/evaluate.log
fi
exit "\${EXIT_CODE}"
TRAIN
)
printf '%s\n' "${TRAIN_CMD}" > "${RUN_DIR}/resolved_command.sh"

HARE_NAME="${RUN_NAME//[^A-Za-z0-9_.-]/_}"
echo "${HARE_NAME}" > "${RUN_DIR}/hare_name.txt"
nohup hare run --rm --gpus "device=${HEX_GPU_INDEX}" --name "${HARE_NAME}" --user "$(id -u):$(id -g)" -v "${REPO_ROOT}:/app" -v "${HEX_SCRATCH}:/workspace" --workdir /app -e AUTO_EVALUATE="${AUTO_EVALUATE:-0}" -e DEVICE_ARGS="${DEVICE_ARGS}" "${HEX_IMAGE_TAG}" bash -lc "${TRAIN_CMD}" > "${RUN_DIR}/logs/hare_stdout.log" 2>&1 &
echo $! | tee "${RUN_DIR}/hare_launcher_pid.txt"
echo "run_dir=${RUN_DIR}"
