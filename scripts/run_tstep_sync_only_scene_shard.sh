#!/usr/bin/env bash
# Drain already-written timestep features by scene shard.  This wrapper only
# scopes FEATURE_SCAN_ROOT; run_tstep_sync_loop.sh still performs validation,
# AOSS upload, done-index merge, and source deletion.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/umm/users/yinbaiqiao/InsScene-15K}"
FEATURE_KIND="${FEATURE_KIND:?set FEATURE_KIND=FEAT_STREAMING|FEAT_TARGET}"
FEATURE_NAMESPACE="${FEATURE_NAMESPACE:?set FEATURE_NAMESPACE=<feature namespace>}"
VFM="${VFM:?set VFM=wan|cogvideox|minimax_h3}"
T="${T:?set T=<timestep>}"
SOURCE="${SOURCE:-scannetpp}"
EXTRACT_DONE_INDEX="${EXTRACT_DONE_INDEX:?set EXTRACT_DONE_INDEX=<done-index>}"
LOG_DIR="${LOG_DIR:?set LOG_DIR=<sync-only log dir>}"
SCENE_START="${SCENE_START:?set SCENE_START=<0-based inclusive>}"
SCENE_END="${SCENE_END:?set SCENE_END=<0-based exclusive>}"
SYNC_EVERY_SEC="${SYNC_EVERY_SEC:-1}"
SYNC_EVERY_GIB="${SYNC_EVERY_GIB:-1}"
SYNC_FINAL_MAX_ATTEMPTS="${SYNC_FINAL_MAX_ATTEMPTS:-2}"

case "${SCENE_START}:${SCENE_END}" in
  *[!0-9:]*|:*|*:) echo "ERROR: invalid scene range ${SCENE_START}:${SCENE_END}" >&2; exit 2 ;;
esac
(( SCENE_START >= 0 && SCENE_END > SCENE_START )) || {
  echo "ERROR: invalid scene range ${SCENE_START}:${SCENE_END}" >&2
  exit 2
}

FEATURE_ROOT="${FEATURE_ROOT:-${DATASET_ROOT}/${FEATURE_NAMESPACE}/${VFM}/${SOURCE}}"
[[ -d "${FEATURE_ROOT}" && ! -L "${FEATURE_ROOT}" ]] || {
  echo "ERROR: FEATURE_ROOT missing/non-dir/symlink: ${FEATURE_ROOT}" >&2
  exit 1
}

mkdir -p "${LOG_DIR}" "$(dirname "${EXTRACT_DONE_INDEX}")"
chmod +x "${PROJECT_ROOT}/scripts/run_tstep_sync_loop.sh"

mapfile -t SCENES < <(
  find "${FEATURE_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
    | grep -E '^[0-9a-f]{10}$' \
    | sort
)

total="${#SCENES[@]}"
(( SCENE_START < total )) || {
  echo "ERROR: SCENE_START=${SCENE_START} >= scene_count=${total}" >&2
  exit 2
}
(( SCENE_END <= total )) || SCENE_END="${total}"

summary="${LOG_DIR}/sync_only_scene_shard.summary.tsv"
printf 'scene_index\tscene_id\tstatus\tstarted_at\tfinished_at\n' > "${summary}"
echo "sync-only shard start vfm=${VFM} t=${T} range=[${SCENE_START}:${SCENE_END}) scenes_total=${total}"

for ((idx = SCENE_START; idx < SCENE_END; idx++)); do
  scene="${SCENES[$idx]}"
  scene_root="${FEATURE_ROOT}/${scene}"
  scene_log="${LOG_DIR}/scene_${idx}_${scene}"
  stop_file="${scene_log}/extract.finished"
  mkdir -p "${scene_log}"
  : > "${stop_file}"
  started="$(date -Is)"
  echo "[${started}] sync scene idx=${idx} id=${scene}"
  if PROJECT_ROOT="${PROJECT_ROOT}" \
      DATASET_ROOT="${DATASET_ROOT}" \
      FEATURE_KIND="${FEATURE_KIND}" \
      FEATURE_NAMESPACE="${FEATURE_NAMESPACE}" \
      VFM="${VFM}" \
      T="${T}" \
      SOURCE="${SOURCE}" \
      FEATURE_ROOT="${FEATURE_ROOT}" \
      FEATURE_SCAN_ROOT="${scene_root}" \
      EXTRACT_DONE_INDEX="${EXTRACT_DONE_INDEX}" \
      LOG_DIR="${scene_log}" \
      STOP_FILE="${stop_file}" \
      SYNC_DISJOINT_SCAN=1 \
      SYNC_EVERY_SEC="${SYNC_EVERY_SEC}" \
      SYNC_EVERY_GIB="${SYNC_EVERY_GIB}" \
      SYNC_FINAL_MAX_ATTEMPTS="${SYNC_FINAL_MAX_ATTEMPTS}" \
      bash "${PROJECT_ROOT}/scripts/run_tstep_sync_loop.sh"; then
    status=ok
  else
    status=failed
  fi
  finished="$(date -Is)"
  printf '%s\t%s\t%s\t%s\t%s\n' "${idx}" "${scene}" "${status}" "${started}" "${finished}" >> "${summary}"
  [[ "${status}" == "ok" ]] || exit 1
done

echo "sync-only shard complete range=[${SCENE_START}:${SCENE_END})"
