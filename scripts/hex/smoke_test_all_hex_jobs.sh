#!/usr/bin/env bash
set -euo pipefail
: "${HEX_GPU_INDEX:?Set HEX_GPU_INDEX}"
: "${HEX_SCRATCH:?Set HEX_SCRATCH}"
: "${HEX_IMAGE_TAG:?Set HEX_IMAGE_TAG}"
scripts/hex/smoke_test.sh --gpu "${HEX_GPU_INDEX}" --scratch "${HEX_SCRATCH}" --image "${HEX_IMAGE_TAG}"
echo "Run tiny local/Hex smoke commands with --timesteps 1000 before full launch."
