#!/usr/bin/env bash
set -euo pipefail

TAG="${1:-$(date +%Y%m%d)-droq}"
IMAGE_TAG="${USER}/msc-audio-pianist:${TAG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DOCKER_DIR="${REPO_ROOT}/docker/hex"

echo "image_tag=${IMAGE_TAG}"
echo "docker_context=${DOCKER_DIR}"
echo "This script is intended to run on the selected Hex node."
cd "${DOCKER_DIR}"
hare build -t "${IMAGE_TAG}" .
