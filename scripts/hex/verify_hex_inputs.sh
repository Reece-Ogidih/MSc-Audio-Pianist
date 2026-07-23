#!/usr/bin/env bash
set -euo pipefail

: "${HEX_SCRATCH:?Set HEX_SCRATCH}"
: "${HEX_IMAGE_TAG:?Set HEX_IMAGE_TAG}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

echo "git_commit=$(git rev-parse HEAD)"
echo "hostname=$(hostname)"
date --iso-8601=seconds
test -d "${HEX_SCRATCH}"
test -w "${HEX_SCRATCH}"
test -f configs/sac_fair_six_sequence_seed13_1m.json
test -f configs/droq_cleanliness_sensitive_v1_seed13_300k.json
test -f configs/droq_five_note_expansion_from_300k_seed13.json
if [[ -n "${HEX_DROQ_300K_CHECKPOINT:-}" ]]; then
  test -f "${HEX_DROQ_300K_CHECKPOINT}"
  if [[ -n "${HEX_DROQ_300K_SHA256:-}" ]]; then
    printf '%s  %s\n' "${HEX_DROQ_300K_SHA256}" "${HEX_DROQ_300K_CHECKPOINT}" | sha256sum -c -
  fi
fi
hare me || true
nvidia-smi || true
echo "image=${HEX_IMAGE_TAG}"
