#!/usr/bin/env bash
set -euo pipefail
: "${HEX_RUN_DIR:?Set HEX_RUN_DIR}"
find "${HEX_RUN_DIR}" -name 'checkpoint_*_steps.pt' -type f | sort
echo "Evaluate five-note sequences listed in configs/droq_five_note_expansion_from_300k_seed13.json."
