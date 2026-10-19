#!/usr/bin/env bash
# Flush only complete canonical seeded-v1 Wan streaming t499 scenes left by an
# interrupted extractor.  Each scene is validated and committed independently;
# incomplete scenes are deliberately retained for the extractor to resume.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/umm/users/yinbaiqiao/InsScene-15K}"
FEATURE_NAMESPACE=FEAT_STREAMING_CANONICAL_V2_SEEDED_V1
FEATURE_ROOT="${DATASET_ROOT}/${FEATURE_NAMESPACE}/wan/scannetpp"
DONE_INDEX="${PROJECT_ROOT}/logs/tstep_canonical_v2_seeded_v1/aoss_done/${FEATURE_NAMESPACE}_wan_t499.txt"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/seeded_v1_wan_t499_complete_scene_flush_$(date -u +%Y%m%d_%H%M%S)}"
SYNC_LOOP="${PROJECT_ROOT}/scripts/run_tstep_sync_loop.sh"

[[ -d "${FEATURE_ROOT}" ]] || { echo "ERROR: missing feature root: ${FEATURE_ROOT}" >&2; exit 1; }
[[ -s "${SYNC_LOOP}" && ! -L "${SYNC_LOOP}" ]] || { echo "ERROR: invalid sync loop: ${SYNC_LOOP}" >&2; exit 1; }
[[ -f "${DONE_INDEX}" && ! -L "${DONE_INDEX}" ]] || { echo "ERROR: invalid done-index: ${DONE_INDEX}" >&2; exit 1; }
mkdir -p "${LOG_ROOT}"

mapfile -d '' scenes < <(
  find "${FEATURE_ROOT}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z
)

selected=0
for scene_root in "${scenes[@]}"; do
  [[ -f "${scene_root}/prefix_index.npy" && \
     -f "${scene_root}/prefix_index.npy.manifest.json" ]] || continue
  if ! find "${scene_root}" -type f -name 'feature_t499_layer*.sft' -print -quit | grep -q .; then
    continue
  fi
  selected=$((selected + 1))
  scene="$(basename "${scene_root}")"
  scene_log="${LOG_ROOT}/$(printf '%04d' "${selected}")_${scene}"
  mkdir -p "${scene_log}"
  stop_file="${scene_log}/extract.finished"
  : > "${stop_file}"
  echo "[$(date -Is)] FLUSH_START ordinal=${selected} scene=${scene} index_lines=$(wc -l < "${DONE_INDEX}")"
  FEATURE_KIND=FEAT_STREAMING \
  FEATURE_NAMESPACE="${FEATURE_NAMESPACE}" \
  VFM=wan \
  T=499 \
  EXTRACT_DONE_INDEX="${DONE_INDEX}" \
  LOG_DIR="${scene_log}" \
  STOP_FILE="${stop_file}" \
  DATASET_ROOT="${DATASET_ROOT}" \
  FEATURE_ROOT="${FEATURE_ROOT}" \
  FEATURE_SCAN_ROOT="${scene_root}" \
  SYNC_EVERY_SEC=0 \
  SYNC_EVERY_GIB=0 \
  SYNC_POLL_SEC=5 \
  SYNC_RETRY_SEC=10 \
  SYNC_FINAL_MAX_ATTEMPTS=3 \
    bash "${SYNC_LOOP}" 2>&1 | tee "${scene_log}/sync_loop.log"
  if find "${scene_root}" -type f \
      \( -name 'feature_t499_layer*.sft' -o \
         -name 'feature_t499_layer*.sft.manifest.json' -o \
         -name '.feature_t499_layer*.tmp-*' \) -print -quit | grep -q .; then
    echo "ERROR: scene still has matching local t499 state after successful sync: ${scene}" >&2
    exit 1
  fi
  echo "[$(date -Is)] FLUSH_COMPLETE ordinal=${selected} scene=${scene} index_lines=$(wc -l < "${DONE_INDEX}")"
done

printf 'selected_complete_scenes=%s\ndone_index_lines=%s\n' \
  "${selected}" "$(wc -l < "${DONE_INDEX}")" | tee "${LOG_ROOT}/SUMMARY.txt"
[[ "${selected}" -gt 0 ]] || { echo "ERROR: no complete local scenes selected" >&2; exit 1; }
