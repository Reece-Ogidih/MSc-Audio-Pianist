#!/usr/bin/env bash
set -euo pipefail

HEX_USER=""
NODE=""
REMOTE_RUN_DIRECTORY=""
LOCAL_OUTPUT_DIRECTORY=""
DRY_RUN="--dry-run"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hex-user) HEX_USER="${2:?}"; shift 2 ;;
    --node) NODE="${2:?}"; shift 2 ;;
    --remote-run-directory) REMOTE_RUN_DIRECTORY="${2:?}"; shift 2 ;;
    --local-output-directory) LOCAL_OUTPUT_DIRECTORY="${2:?}"; shift 2 ;;
    --execute) DRY_RUN=""; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "${HEX_USER}" ]] || { echo "--hex-user is required" >&2; exit 2; }
[[ -n "${NODE}" ]] || { echo "--node is required" >&2; exit 2; }
[[ -n "${REMOTE_RUN_DIRECTORY}" ]] || { echo "--remote-run-directory is required" >&2; exit 2; }
[[ -n "${LOCAL_OUTPUT_DIRECTORY}" ]] || { echo "--local-output-directory is required" >&2; exit 2; }

mkdir -p "${LOCAL_OUTPUT_DIRECTORY}"
remote="${HEX_USER}@${NODE}.cs.bath.ac.uk:${REMOTE_RUN_DIRECTORY}/"
echo "remote=${remote}"
echo "local_output=${LOCAL_OUTPUT_DIRECTORY}"
rsync -avP ${DRY_RUN} "${remote}" "${LOCAL_OUTPUT_DIRECTORY}/"
if [[ -n "${DRY_RUN}" ]]; then
  echo "Dry-run only. Re-run with --execute to copy results."
fi
