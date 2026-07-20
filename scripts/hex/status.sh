#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:-}"
SCRATCH="${2:-}"
[[ -n "${RUN_NAME}" ]] || { echo "Usage: $0 <run-name> [scratch-path]" >&2; exit 2; }

echo "run_name=${RUN_NAME}"
tmux has-session -t "${RUN_NAME}" 2>/dev/null && {
  echo "tmux=present"
  tmux capture-pane -pt "${RUN_NAME}" | tail -80 || true
} || echo "tmux=absent"

command -v hare >/dev/null 2>&1 && {
  echo "hare_ps:"
  hare ps -a | grep -F "${RUN_NAME}" || true
  echo "hare_logs_tail:"
  hare logs -n 80 "${RUN_NAME}" 2>/dev/null || true
  echo "hare_stats:"
  hare stats "${RUN_NAME}" 2>/dev/null || true
} || echo "hare=not_found"

command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
ps -eo pid,etime,%cpu,%mem,cmd | grep -F "${RUN_NAME}" | grep -v grep || true

if [[ -n "${SCRATCH}" ]]; then
  RUN_DIR="${SCRATCH}/runs/${RUN_NAME}"
  echo "run_dir=${RUN_DIR}"
  df -h "${SCRATCH}" || true
  echo "latest_log:"
  tail -80 "${RUN_DIR}/logs/train.log" 2>/dev/null || true
  echo "latest_checkpoint:"
  find "${RUN_DIR}" -name 'checkpoint_*_steps.pt' -type f 2>/dev/null | sort | tail -1 || true
fi
