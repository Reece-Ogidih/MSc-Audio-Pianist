#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/third_party/robopianist"
python scripts/validate_five_note_factorial_configs.py --config-dir configs/five_note_factorial_1m
for config in configs/five_note_factorial_1m/*_seed13_1m.json; do
  echo "smoke_config=${config}"
  python - <<PY
import json
from pathlib import Path
cfg=json.loads(Path("${config}").read_text())
assert cfg["resume_from_checkpoint"] is None
assert cfg["midi_min"] == 72 and cfg["midi_max"] == 76
assert cfg["expected_observation_dim"] == 301
assert cfg["expected_action_dim"] == 22
print(cfg["condition_id"], "ok")
PY
done
echo "smoke_test_five_note_factorial=ok"
