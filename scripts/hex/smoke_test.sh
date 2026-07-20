#!/usr/bin/env bash
set -euo pipefail

GPU_INDEX=""
SCRATCH=""
IMAGE_TAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU_INDEX="${2:?}"; shift 2 ;;
    --scratch) SCRATCH="${2:?}"; shift 2 ;;
    --image) IMAGE_TAG="${2:?}"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "${GPU_INDEX}" ]] || { echo "--gpu is required" >&2; exit 2; }
[[ -n "${SCRATCH}" ]] || { echo "--scratch is required" >&2; exit 2; }
[[ -n "${IMAGE_TAG}" ]] || { echo "--image is required" >&2; exit 2; }
[[ -d "${SCRATCH}" ]] || { echo "Scratch path does not exist: ${SCRATCH}" >&2; exit 2; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SMOKE_MARKER="hex_smoke_$(date +%s).txt"

echo "image=${IMAGE_TAG}"
echo "gpu=${GPU_INDEX}"
echo "scratch=${SCRATCH}"
echo "repo=${REPO_ROOT}"
echo "This script is intended to run on Hex through Hare."

hare run --rm --gpus "device=${GPU_INDEX}" \
  --user "$(id -u):$(id -g)" \
  -e "SMOKE_MARKER=${SMOKE_MARKER}" \
  -v "${REPO_ROOT}:/app" \
  -v "${SCRATCH}:/workspace" \
  --workdir /app \
  "${IMAGE_TAG}" \
  bash -lc "set -euo pipefail
export PYTHONPATH=/app/src:/app/third_party/robopianist
export MUJOCO_GL=\${MUJOCO_GL:-egl}
python - <<'PY'
from pathlib import Path
import os
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit('CUDA is not available inside the Hare container')
x = torch.ones((8, 8), device='cuda')
print('cuda_sum', float((x @ x).sum().detach().cpu()))
import mujoco, dm_control, robopianist
from ala_pianist.rl import GeneralOneHandGoalEnv
sf2 = Path(robopianist.SF2_PATH)
print('soundfont', sf2, sf2.exists())
if not sf2.exists():
    raise SystemExit(f'Soundfont not found: {sf2}')
env = GeneralOneHandGoalEnv(
    generated_midi_dir='/workspace/smoke_midi',
    curriculum='single_notes',
    midi_pitches=(73,),
    lookahead=1,
    horizon_steps=3,
    action_mode='direct',
    action_repeat=1,
)
obs, info = env.reset(seed=1)
for _ in range(3):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        break
marker = Path('/workspace') / os.environ['SMOKE_MARKER']
marker.write_text('hex smoke ok\n', encoding='utf-8')
print('wrote', marker)
PY
test -f /workspace/${SMOKE_MARKER}
"

test -f "${SCRATCH}/${SMOKE_MARKER}"
echo "smoke_marker=${SCRATCH}/${SMOKE_MARKER}"
