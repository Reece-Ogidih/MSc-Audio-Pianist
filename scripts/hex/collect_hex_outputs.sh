#!/usr/bin/env bash
set -euo pipefail
: "${HEX_USER:?Set HEX_USER}"
: "${HEX_HOST:?Set HEX_HOST}"
: "${HEX_REMOTE_RUN_DIR:?Set HEX_REMOTE_RUN_DIR}"
: "${LOCAL_OUTPUT_DIR:?Set LOCAL_OUTPUT_DIR}"
mkdir -p "${LOCAL_OUTPUT_DIR}"
rsync -avP "${HEX_USER}@${HEX_HOST}:${HEX_REMOTE_RUN_DIR}/" "${LOCAL_OUTPUT_DIR}/"
