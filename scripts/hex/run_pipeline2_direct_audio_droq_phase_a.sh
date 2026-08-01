#!/usr/bin/env bash
set -euo pipefail

# Pipeline 2 direct-audio DroQ Phase-A launcher.
# Intended usage:
#   SEED=13 DEVICE=cuda scripts/hex/run_pipeline2_direct_audio_droq_phase_a.sh
#   SEED=37 DEVICE=cuda scripts/hex/run_pipeline2_direct_audio_droq_phase_a.sh

PROJECT_ROOT="${PROJECT_ROOT:-/app}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/experiments/pipeline2_direct_audio}"
SEED="${SEED:-13}"
TIMESTEPS="${TIMESTEPS:-1000000}"
DEVICE="${DEVICE:-cuda}"
RUN_NAME="${RUN_NAME:-pipeline2_direct_audio_droq_v1_seed${SEED}_1m}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"
LOG_PATH="${LOG_DIR}/train.log"
METADATA_START="${RUN_DIR}/launch_metadata_start.json"
METADATA_END="${RUN_DIR}/launch_metadata_end.json"

if [[ -e "${RUN_DIR}" ]] && [[ -n "$(find "${RUN_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite non-empty run directory: ${RUN_DIR}" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/third_party/robopianist:${PROJECT_ROOT}/scripts"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

python - <<PY
import json, os, socket, subprocess
from datetime import datetime, timezone
from pathlib import Path
import torch

payload = {
    "event": "start",
    "utc": datetime.now(timezone.utc).isoformat(),
    "hostname": socket.gethostname(),
    "seed": int("${SEED}"),
    "run_name": "${RUN_NAME}",
    "run_dir": "${RUN_DIR}",
    "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], check=True, stdout=subprocess.PIPE, text=True).stdout.strip(),
    "device": "${DEVICE}",
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "audio": {"sample_rate": 16000, "past_context_seconds": 0.10, "future_context_seconds": 0.40, "window_samples": 8000},
    "architecture": "raw waveform -> Conv1D -> GRU -> fusion MLP -> 22D DroQ actor",
    "replay_schema": "indexed_audio_references",
    "reward_profile": "transition_cleanup_sensitive_v1",
    "sequence_distribution": {
        "sequences": [[72], [73], [74], [75], [76], [72, 73], [73, 72], [73, 74], [74, 73], [74, 75], [75, 74], [75, 76], [76, 75]],
        "weights": [0.07, 0.07, 0.07, 0.07, 0.07, 0.08125, 0.08125, 0.08125, 0.08125, 0.08125, 0.08125, 0.08125, 0.08125],
    },
}
Path("${METADATA_START}").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
PY

set +e
python scripts/train_direct_audio_droq.py \
  --timesteps "${TIMESTEPS}" \
  --seed "${SEED}" \
  --stage-name "${RUN_NAME}" \
  --output-dir "${RUN_DIR}" \
  --generated-root "${RUN_DIR}/audio_bank" \
  --sequence-pitches "72;73;74;75;76;72,73;73,72;73,74;74,73;74,75;75,74;75,76;76,75" \
  --sequence-sampling-weights "0.07,0.07,0.07,0.07,0.07,0.08125,0.08125,0.08125,0.08125,0.08125,0.08125,0.08125,0.08125" \
  --variants-per-sequence 4 \
  --learning-starts 1000 \
  --batch-size 64 \
  --utd-ratio 2 \
  --buffer-size 2000000 \
  --lightweight-checkpoint-steps "10000,25000,50000,100000,250000,500000,750000,1000000" \
  --full-checkpoint-steps "1000000" \
  --device "${DEVICE}" \
  2>&1 | tee "${LOG_PATH}"
EXIT_CODE="${PIPESTATUS[0]}"
set -e

python - <<PY
import json
from datetime import datetime, timezone
from pathlib import Path

payload = {
    "event": "end",
    "utc": datetime.now(timezone.utc).isoformat(),
    "seed": int("${SEED}"),
    "run_name": "${RUN_NAME}",
    "run_dir": "${RUN_DIR}",
    "exit_code": int("${EXIT_CODE}"),
    "log_path": "${LOG_PATH}",
}
Path("${METADATA_END}").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
PY

exit "${EXIT_CODE}"
