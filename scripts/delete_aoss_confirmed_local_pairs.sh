#!/usr/bin/env bash
# Delete only local feature/sidecar pairs named by an AOSS-confirmed manifest.
#
# Safety invariants:
# - the manifest is immutable for the lifetime of this invocation;
# - every entry is a relative path for the requested timestep and layer set;
# - every entry must already be present in the process-wide AOSS done-index;
# - no glob, directory recursion, or ads-cli --delete-src operation is used;
# - an absent pair is harmless, while any pair still present after rm is fatal.
# The caller must build the manifest from an AOSS LIST intersection, never
# from local presence alone. This script deliberately performs no AOSS GET.
set -euo pipefail

FEATURE_ROOT="${FEATURE_ROOT:?set FEATURE_ROOT}"
T="${T:?set T}"
LAYERS="${LAYERS:-15 20 25}"
LAYERS="${LAYERS//,/ }"
LAYERS="${LAYERS//-/ }"
EXTRACT_DONE_INDEX="${EXTRACT_DONE_INDEX:?set EXTRACT_DONE_INDEX}"
CLEANUP_MANIFEST="${PRE_EXTRACT_CLEANUP_MANIFEST:?set PRE_EXTRACT_CLEANUP_MANIFEST}"
CLEANUP_EXPECTED_COUNT="${CLEANUP_EXPECTED_COUNT:-0}"
CLEANUP_DRY_RUN="${CLEANUP_DRY_RUN:-0}"

[[ -d "${FEATURE_ROOT}" ]] || { echo "ERROR: missing feature root: ${FEATURE_ROOT}" >&2; exit 1; }
[[ -f "${EXTRACT_DONE_INDEX}" ]] || { echo "ERROR: missing done-index: ${EXTRACT_DONE_INDEX}" >&2; exit 1; }
[[ -f "${CLEANUP_MANIFEST}" ]] || { echo "ERROR: missing cleanup manifest: ${CLEANUP_MANIFEST}" >&2; exit 1; }

LAYER_RE="$(awk '{$1=$1; gsub(/ /,"|"); print}' <<<"${LAYERS}")"
manifest_count="$(wc -l < "${CLEANUP_MANIFEST}")"
unique_count="$(LC_ALL=C sort -u "${CLEANUP_MANIFEST}" | wc -l)"
[[ "${manifest_count}" -eq "${unique_count}" ]] || {
  echo "ERROR: cleanup manifest contains duplicate entries" >&2
  exit 2
}
if [[ "${CLEANUP_EXPECTED_COUNT}" -gt 0 && "${manifest_count}" -ne "${CLEANUP_EXPECTED_COUNT}" ]]; then
  echo "ERROR: cleanup manifest count ${manifest_count} != expected ${CLEANUP_EXPECTED_COUNT}" >&2
  exit 2
fi

bad=0
while IFS= read -r rel; do
  [[ -n "${rel}" ]] || { bad=$((bad + 1)); continue; }
  if [[ ! "${rel}" =~ ^[^/]+/window_[0-9]+/prefix_[0-9]+/feature_t${T}_layer(${LAYER_RE})\.sft$ ]] ||
     [[ "${rel}" == *".."* ]]; then
    echo "ERROR: unsafe cleanup entry: ${rel}" >&2
    bad=$((bad + 1))
  fi
done < "${CLEANUP_MANIFEST}"
[[ "${bad}" -eq 0 ]] || exit 2

not_indexed="$({
  comm -23 \
    <(LC_ALL=C sort -u "${CLEANUP_MANIFEST}") \
    <(LC_ALL=C sort -u "${EXTRACT_DONE_INDEX}")
} | wc -l)"
[[ "${not_indexed}" -eq 0 ]] || {
  echo "ERROR: ${not_indexed} cleanup entries are absent from done-index" >&2
  exit 2
}

echo "pre-extract cleanup validated entries=${manifest_count} dry_run=${CLEANUP_DRY_RUN}"
[[ "${CLEANUP_DRY_RUN}" == "0" ]] || exit 0

deleted_sft=0
deleted_sidecar=0
already_absent=0
while IFS= read -r rel; do
  sft="${FEATURE_ROOT}/${rel}"
  sidecar="${sft}.manifest.json"
  had_sft=0
  had_sidecar=0
  [[ -e "${sft}" ]] && had_sft=1
  [[ -e "${sidecar}" ]] && had_sidecar=1
  rm -f -- "${sft}" "${sidecar}"
  [[ ! -e "${sft}" && ! -e "${sidecar}" ]] || {
    echo "ERROR: cleanup failed: ${rel}" >&2
    exit 1
  }
  deleted_sft=$((deleted_sft + had_sft))
  deleted_sidecar=$((deleted_sidecar + had_sidecar))
  if [[ "${had_sft}" -eq 0 && "${had_sidecar}" -eq 0 ]]; then
    already_absent=$((already_absent + 1))
  fi
done < "${CLEANUP_MANIFEST}"

echo "pre-extract cleanup complete entries=${manifest_count} deleted_sft=${deleted_sft} deleted_sidecar=${deleted_sidecar} already_absent=${already_absent}"
