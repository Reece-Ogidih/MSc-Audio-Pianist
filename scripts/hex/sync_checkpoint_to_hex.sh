#!/usr/bin/env bash
set -euo pipefail

HEX_USER=""
NODE=""
REMOTE_DIRECTORY=""
CHECKPOINT=""
DRY_RUN="--dry-run"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hex-user) HEX_USER="${2:?}"; shift 2 ;;
    --node) NODE="${2:?}"; shift 2 ;;
    --remote-directory) REMOTE_DIRECTORY="${2:?}"; shift 2 ;;
    --checkpoint) CHECKPOINT="${2:?}"; shift 2 ;;
    --execute) DRY_RUN=""; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "${HEX_USER}" ]] || { echo "--hex-user is required" >&2; exit 2; }
[[ -n "${NODE}" ]] || { echo "--node is required" >&2; exit 2; }
[[ -n "${REMOTE_DIRECTORY}" ]] || { echo "--remote-directory is required" >&2; exit 2; }
[[ -f "${CHECKPOINT}" ]] || { echo "--checkpoint must be a file" >&2; exit 2; }

remote="${HEX_USER}@${NODE}.cs.bath.ac.uk:${REMOTE_DIRECTORY}/"
echo "local_sha256=$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
echo "remote=${remote}"
rsync -avP ${DRY_RUN} "${CHECKPOINT}" "${remote}"
if [[ -z "${DRY_RUN}" ]]; then
  ssh "${HEX_USER}@${NODE}.cs.bath.ac.uk" "sha256sum '${REMOTE_DIRECTORY}/$(basename "${CHECKPOINT}")'"
else
  echo "Dry-run only. Re-run with --execute to transfer and verify remote SHA-256."
fi
