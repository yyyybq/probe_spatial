#!/usr/bin/env bash
# Sync InsScene feature caches to AOSS via ads-cli.
#
# Root cause of earlier failures: this host sets HTTP(S)_PROXY, but NO_PROXY did
# not include the internal AOSS endpoint, so ads-cli went through Squid and got
# 503 / CONNECT_FAIL HTML error pages. This script always sets NO_PROXY first.
#
# Modes:
#   MODE=scenes  (default; preferred — resilient on flaky AOSS links)
#     Per-scene sync under <FEATURE_DIR>/<VFM>/<SOURCE> with timeouts + retries.
#     Example:
#       VFM=vjepa2 bash scripts/sync_features_to_aoss.sh
#       VFM=minimax_h3 FEATURE_DIR=FEAT_STREAMING bash scripts/sync_features_to_aoss.sh
#
#   MODE=tree
#     Whole-directory sync of FEATURE_DIRS. Faster only when the link is healthy;
#     otherwise prefer MODE=scenes (large trees can stall with many failed objects).
#     Example:
#       MODE=tree FEATURE_DIRS=FEAT_TARGET THREADS=16 bash scripts/sync_features_to_aoss.sh
set -u -o pipefail

AOSS_DIR=${AOSS_DIR:-/mnt/umm/users/yinbaiqiao/aoss}
CONF=${CONF:-${AOSS_DIR}/petreloss.conf}
ADS_CLI=${ADS_CLI:-${AOSS_DIR}/ads-cli}
# Prefer AOSS_API_HOST. Bare HOST is often polluted by conda (e.g. x86_64-conda-linux-gnu).
AOSS_API_HOST=${AOSS_API_HOST:-aoss-internal.cn-fz-01.fjscmsapi-oss.com}
if [[ -n "${HOST:-}" && "${HOST}" != *conda* && "${HOST}" != x86_64* ]]; then
  AOSS_API_HOST=${HOST}
fi
HOST="${AOSS_API_HOST}"
BUCKET=${BUCKET:-baiqiao}
AOSS_HOST=${AOSS_HOST:-baiqiao.${HOST}}
DATASET_ROOT=${DATASET_ROOT:-/mnt/umm/users/yinbaiqiao/InsScene-15K}
MODE=${MODE:-scenes}

# Bypass corporate HTTP proxy for internal AOSS (required on this cluster).
export NO_PROXY="localhost,127.0.0.1,::1,${AOSS_HOST},.${HOST},${HOST}"
export no_proxy="${NO_PROXY}"

AK="$(awk '
  /^\[fj\]/ {in_fj=1; next}
  /^\[/ {in_fj=0}
  in_fj && /^access_key[[:space:]]*=/ {gsub(/^access_key[[:space:]]*=[[:space:]]*/, ""); print; exit}
' "${CONF}")"
SK="$(awk '
  /^\[fj\]/ {in_fj=1; next}
  /^\[/ {in_fj=0}
  in_fj && /^secret_key[[:space:]]*=/ {gsub(/^secret_key[[:space:]]*=[[:space:]]*/, ""); print; exit}
' "${CONF}")"
if [[ -z "${AK}" || -z "${SK}" ]]; then
  echo "ERROR: failed to read fj credentials from ${CONF}" >&2
  exit 1
fi
CRED="${AK}:${SK}"
DST_ROOT="s3://${CRED}@${BUCKET}.${HOST}/InsScene-15K"

sync_tree() {
  local THREADS=${THREADS:-16}
  local FEATURE_DIRS=${FEATURE_DIRS:-"FEAT_TARGET FEAT_STREAMING"}
  echo "START $(date -Is) mode=tree threads=${THREADS} dirs=${FEATURE_DIRS}"
  echo "NO_PROXY=${NO_PROXY}"
  local dir src dst
  for dir in ${FEATURE_DIRS}; do
    src="${DATASET_ROOT}/${dir}/"
    dst="${DST_ROOT}/${dir}/"
    if [[ ! -d "${src}" ]]; then
      echo "SKIP missing ${src}"
      continue
    fi
    echo "SYNC ${dir} $(date -Is)"
    echo "  src=${src}"
    echo "  dst=s3://${BUCKET}.${HOST}/InsScene-15K/${dir}/"
    "${ADS_CLI}" -p "${THREADS}" sync "${src}" "${dst}"
    echo "OK ${dir} $(date -Is)"
  done
  echo "DONE $(date -Is)"
}

sync_scenes() {
  local VFM=${VFM:-vjepa2}
  local THREADS=${THREADS:-1}
  local LISTERS=${LISTERS:-2}
  local BATCH_SIZE=${BATCH_SIZE:-50}
  local SCENE_TIMEOUT=${SCENE_TIMEOUT:-30m}
  local MAX_ATTEMPTS=${MAX_ATTEMPTS:-4}
  local SOURCE=${SOURCE:-scannetpp}
  local FEATURE_DIR=${FEATURE_DIR:-FEAT_TARGET}
  local SCENE_ROOT="${DATASET_ROOT}/${FEATURE_DIR}/${VFM}/${SOURCE}"
  local WORK_DIR=${WORK_DIR:-/mnt/umm/users/yinbaiqiao/probe_spatial/logs/${VFM}_aoss_batched}
  local DST_BASE="${DST_ROOT}/${FEATURE_DIR}/${VFM}/${SOURCE}"

  if [[ ! -d "${SCENE_ROOT}" ]]; then
    echo "ERROR: scene root missing: ${SCENE_ROOT}" >&2
    exit 1
  fi

  mkdir -p "${WORK_DIR}"/{logs,done,batches}
  touch "${WORK_DIR}/failed.txt"

  find "${SCENE_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
    | sort > "${WORK_DIR}/scenes.txt"
  rm -f "${WORK_DIR}"/batches/batch_*
  split -d -a 3 -l "${BATCH_SIZE}" "${WORK_DIR}/scenes.txt" "${WORK_DIR}/batches/batch_"

  echo "START $(date -Is) mode=scenes vfm=${VFM} threads=${THREADS} listers=${LISTERS}"
  echo "  scene_root=${SCENE_ROOT}"
  echo "  dst=s3://${BUCKET}.${HOST}/InsScene-15K/${FEATURE_DIR}/${VFM}/${SOURCE}/"
  echo "  work_dir=${WORK_DIR}"
  echo "  scenes=$(wc -l < "${WORK_DIR}/scenes.txt") done=$(find "${WORK_DIR}/done" -maxdepth 1 -type f | wc -l)"
  echo "NO_PROXY=${NO_PROXY}"

  local batch batch_name scene attempt success rc
  for batch in "${WORK_DIR}"/batches/batch_*; do
    [[ -f "${batch}" ]] || continue
    batch_name="$(basename "${batch}")"
    echo "[$(date -Is)] start ${batch_name}"

    while IFS= read -r scene; do
      [[ -n "${scene}" ]] || continue
      if [[ -f "${WORK_DIR}/done/${scene}" ]]; then
        continue
      fi

      success=0
      for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
        echo "[$(date -Is)] ${scene} attempt=${attempt}"
        timeout "${SCENE_TIMEOUT}" \
          "${ADS_CLI}" \
          -p "${THREADS}" \
          -l "${LISTERS}" \
          --conntimeout 120 \
          --timeout 600 \
          sync \
          "${SCENE_ROOT}/${scene}/" \
          "${DST_BASE}/${scene}/" \
          >>"${WORK_DIR}/logs/${scene}.log" 2>&1
        rc=$?

        if [[ ${rc} -eq 0 ]]; then
          touch "${WORK_DIR}/done/${scene}"
          success=1
          echo "[$(date -Is)] ${scene} success"
          break
        fi

        echo "[$(date -Is)] ${scene} failed rc=${rc}"
        sleep $((attempt * 60))
      done

      if [[ ${success} -ne 1 ]]; then
        echo "${scene}" >> "${WORK_DIR}/failed.txt"
        echo "[$(date -Is)] ${scene} permanently failed"
      fi
    done < "${batch}"

    echo "[$(date -Is)] finish ${batch_name}"
  done

  echo "DONE $(date -Is) done=$(find "${WORK_DIR}/done" -maxdepth 1 -type f | wc -l) failed=$(sort -u "${WORK_DIR}/failed.txt" | sed '/^$/d' | wc -l)"
}

case "${MODE}" in
  tree) sync_tree ;;
  scenes) sync_scenes ;;
  *)
    echo "ERROR: MODE must be 'scenes' or 'tree' (got: ${MODE})" >&2
    exit 1
    ;;
esac
