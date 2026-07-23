#!/usr/bin/env bash
set -euo pipefail

: "${HEX_SCRATCH:?Set HEX_SCRATCH}"
ROOT="${FIVE_NOTE_FACTORIAL_ROOT:-${HEX_SCRATCH}/five_note_factorial_1m}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
PACKAGE_ROOT="${ROOT}/packages/five_note_factorial_1m_results_compact_${STAMP}"
mkdir -p "${PACKAGE_ROOT}"

find "${ROOT}" -name 'full_checkpoint_*_steps.pt' -o -name 'full_checkpoint_*_steps.zip' -o -name 'rolling_latest_full*' > "${PACKAGE_ROOT}/full_checkpoint_inventory.txt" || true
find "${ROOT}" \( -path '*/lightweight_checkpoints/*' -o -path '*/evaluation/*' \) -type f > "${PACKAGE_ROOT}/lightweight_checkpoint_inventory.txt" || true
find "${ROOT}" -name 'checkpoint_*_steps.pt' -o -name 'checkpoint_*_steps.zip' | sort | sha256sum > "${PACKAGE_ROOT}/selected_checkpoint_hashes.txt" || true
grep -E 'full_checkpoint_(500000|1000000)_steps|rolling_latest_full' "${PACKAGE_ROOT}/full_checkpoint_inventory.txt" > "${PACKAGE_ROOT}/full_checkpoints_to_preserve_on_hex.txt" || true
cat > "${PACKAGE_ROOT}/transfer_manifest.txt" <<MANIFEST
Compact five-note factorial result package.
Includes resolved configs, CSV/JSON summaries, SVG plots, selected logs, resource summaries, lightweight policy checkpoints and checkpoint inventories.
Excludes replay buffers, optimiser-heavy full checkpoints, Docker caches, Python caches and raw container filesystems.
MANIFEST

for condition in droq_original droq_sensitive_v1 sac_original sac_sensitive_v1; do
  if [[ -f "${ROOT}/${condition}/evaluation/per_checkpoint_summary.csv" ]]; then
    latest="${ROOT}/${condition}"
  else
    latest="$(find "${ROOT}/${condition}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1 || true)"
  fi
  [[ -n "${latest}" ]] || continue
  mkdir -p "${PACKAGE_ROOT}/${condition}"
  cp -a "${latest}/resolved_training_config.json" "${latest}/five_note_curriculum_v1.json" "${latest}/factorial_manifest_seed13.json" "${PACKAGE_ROOT}/${condition}/" 2>/dev/null || true
  cp -a "${latest}/evaluation" "${PACKAGE_ROOT}/${condition}/evaluation" 2>/dev/null || true
  mkdir -p "${PACKAGE_ROOT}/${condition}/logs"
  cp -a "${latest}/logs/train.log" "${latest}/logs/evaluate.log" "${latest}/resource_usage.csv" "${PACKAGE_ROOT}/${condition}/logs/" 2>/dev/null || true
  if [[ -d "${latest}/output/lightweight_checkpoints" ]]; then
    find "${latest}/output/lightweight_checkpoints" -type f \( -name '*.pt' -o -name '*.zip' \) -print0 | xargs -0 -r -I{} cp --parents "{}" "${PACKAGE_ROOT}/${condition}/"
  fi
  if [[ -d "${latest}/lightweight_checkpoints" ]]; then
    find "${latest}/lightweight_checkpoints" -type f \( -name '*.pt' -o -name '*.zip' \) -print0 | xargs -0 -r -I{} cp --parents "{}" "${PACKAGE_ROOT}/${condition}/"
  fi
done

latest_aggregate="$(find "${ROOT}/aggregate" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1 || true)"
if [[ -n "${latest_aggregate}" ]]; then
  cp -a "${latest_aggregate}" "${PACKAGE_ROOT}/aggregate"
fi

tarball="${PACKAGE_ROOT}.tar.gz"
tar -C "$(dirname "${PACKAGE_ROOT}")" -czf "${tarball}" "$(basename "${PACKAGE_ROOT}")"
echo "package=${tarball}"
