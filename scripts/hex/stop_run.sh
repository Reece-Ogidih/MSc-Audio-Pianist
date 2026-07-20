#!/usr/bin/env bash
set -euo pipefail

RUN_NAME=""
SCRATCH=""
CONFIRM="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name) RUN_NAME="${2:?}"; shift 2 ;;
    --scratch) SCRATCH="${2:?}"; shift 2 ;;
    --confirm-stop) CONFIRM="true"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "${RUN_NAME}" ]] || { echo "--run-name is required" >&2; exit 2; }
echo "run_name=${RUN_NAME}"
if [[ "${CONFIRM}" != "true" ]]; then
  echo "Dry-run only. Re-run with --confirm-stop to stop."
  exit 0
fi

if tmux has-session -t "${RUN_NAME}" 2>/dev/null; then
  tmux send-keys -t "${RUN_NAME}" C-c
  sleep 10
fi

if command -v hare >/dev/null 2>&1; then
  hare stop "${RUN_NAME}" 2>/dev/null || true
  sleep 10
  if hare ps | grep -F "${RUN_NAME}" >/dev/null 2>&1; then
    echo "Container still appears to be running; using hare kill."
    hare kill "${RUN_NAME}"
  fi
fi

if [[ -n "${SCRATCH}" ]]; then
  find "${SCRATCH}/runs/${RUN_NAME}" -name 'checkpoint_*_steps.pt' -type f 2>/dev/null | sort | tail -5 || true
fi
