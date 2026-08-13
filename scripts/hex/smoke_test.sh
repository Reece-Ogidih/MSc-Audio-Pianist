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
export NUMBA_CACHE_DIR=\${NUMBA_CACHE_DIR:-/tmp/ala-numba-cache}
export XDG_CACHE_HOME=\${XDG_CACHE_HOME:-/tmp/ala-xdg-cache}
mkdir -p "\${NUMBA_CACHE_DIR}" "\${XDG_CACHE_HOME}"
test -w "\${NUMBA_CACHE_DIR}"
python - <<'PY'
from pathlib import Path
import os
import torch
import onnxruntime
import basic_pitch
from basic_pitch import inference as basic_pitch_inference
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('basic_pitch', getattr(basic_pitch, '__version__', '0.4.0'), basic_pitch.__file__)
print('onnxruntime', onnxruntime.__version__)
print('numba_cache_dir', os.environ['NUMBA_CACHE_DIR'])
if not os.access(os.environ['NUMBA_CACHE_DIR'], os.W_OK):
    raise SystemExit(f"NUMBA_CACHE_DIR is not writable: {os.environ['NUMBA_CACHE_DIR']}")
basic_pitch_model = Path(basic_pitch_inference.ICASSP_2022_MODEL_PATH).with_suffix('.onnx')
print('basic_pitch_model', basic_pitch_model)
if not basic_pitch_model.is_file():
    raise SystemExit(f'Basic Pitch ONNX model missing: {basic_pitch_model}')
if not torch.cuda.is_available():
    raise SystemExit('CUDA is not available inside the Hare container')
x = torch.ones((8, 8), device='cuda')
print('cuda_sum', float((x @ x).sum().detach().cpu()))
import mujoco, dm_control, robopianist
from ala_pianist.rl import GeneralOneHandGoalEnv
from scripts import evaluate_long_horizon_compositional
from ala_pianist.audio import BasicPitchTranscriber
from ala_pianist.music.midi_utils import NoteEvent, write_monophonic_midi
from ala_pianist.pipelines.indirect import render_midi_with_fluidsynth
sf2 = Path(robopianist.SF2_PATH)
print('soundfont', sf2, sf2.exists())
if not sf2.exists():
    raise SystemExit(f'Soundfont not found: {sf2}')
bp_dir = Path('/workspace/basic_pitch_smoke')
bp_midi = write_monophonic_midi([NoteEvent(74, 0.0, 0.28, 90)], bp_dir / 'note.mid')
bp_wav = render_midi_with_fluidsynth(bp_midi, bp_dir / 'note.wav', soundfont_path=sf2)
bp_result = BasicPitchTranscriber().transcribe(bp_wav)
print('basic_pitch_note_count', len(bp_result.notes))
if not bp_result.notes:
    raise SystemExit('Basic Pitch smoke produced no notes')
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
