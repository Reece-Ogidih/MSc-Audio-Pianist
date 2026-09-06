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
  -e "PYTHONPATH=/app/src:/app/third_party/robopianist" \
  -e "MUJOCO_GL=egl" \
  -e "NUMBA_CACHE_DIR=/tmp/ala-numba-cache" \
  -e "XDG_CACHE_HOME=/tmp/ala-xdg-cache" \
  -v "${REPO_ROOT}:/app" \
  -v "${SCRATCH}:/workspace" \
  --workdir /app \
  "${IMAGE_TAG}" \
  python /app/scripts/hex/hex_runtime_smoke.py

test -f "${SCRATCH}/${SMOKE_MARKER}"
echo "smoke_marker=${SCRATCH}/${SMOKE_MARKER}"
