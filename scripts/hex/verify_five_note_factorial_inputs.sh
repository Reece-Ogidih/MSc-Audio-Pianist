#!/usr/bin/env bash
set -euo pipefail

: "${HEX_SCRATCH:?Set HEX_SCRATCH}"
: "${HEX_IMAGE_TAG:?Set HEX_IMAGE_TAG}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/third_party/robopianist"

echo "hostname=$(hostname)"
date -u +%Y-%m-%dT%H:%M:%SZ
echo "git_commit=$(git rev-parse HEAD)"
git status --short --branch
test -d "${HEX_SCRATCH}"
test -w "${HEX_SCRATCH}"
python scripts/validate_five_note_factorial_configs.py --config-dir configs/five_note_factorial_1m
python - <<'PY'
import json
from pathlib import Path
cfg=json.loads(Path("configs/five_note_factorial_1m/five_note_curriculum_v1.json").read_text())
assert cfg["midi_min"] == 72 and cfg["midi_max"] == 76
assert abs(sum(cfg["sequence_sampling_weights"]) - 1.0) < 1e-12
assert {s[0] for s in cfg["sequence_pitches"] if len(s) == 1} == {72,73,74,75,76}
print("five_note_factorial_inputs=ok")
PY
df -h "${HEX_SCRATCH}"
nvidia-smi || true
echo "image=${HEX_IMAGE_TAG}"
