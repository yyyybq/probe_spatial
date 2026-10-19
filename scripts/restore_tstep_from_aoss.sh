#!/usr/bin/env bash
# Restore new-timestep features from AOSS to local so extract skip-existing works.
#
# Usage:
#   bash scripts/restore_tstep_from_aoss.sh wan_target
#   bash scripts/restore_tstep_from_aoss.sh minimax_stream
set -u -o pipefail

AOSS_DIR=${AOSS_DIR:-/mnt/umm/users/yinbaiqiao/aoss}
CONF=${CONF:-${AOSS_DIR}/petreloss.conf}
ADS_CLI=${ADS_CLI:-${AOSS_DIR}/ads-cli}
AOSS_API_HOST=${AOSS_API_HOST:-aoss-internal.cn-fz-01.fjscmsapi-oss.com}
HOST="${AOSS_API_HOST}"
BUCKET=${BUCKET:-baiqiao}
AOSS_HOST=${AOSS_HOST:-baiqiao.${HOST}}
DATASET_ROOT=${DATASET_ROOT:-/mnt/umm/users/yinbaiqiao/InsScene-15K}
THREADS=${THREADS:-32}
LISTERS=${LISTERS:-8}

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
SRC_ROOT="s3://${AK}:${SK}@${BUCKET}.${HOST}/InsScene-15K"

restore_one() {
  local kind="$1"
  local vfm="$2"
  local include="$3"
  local src="${SRC_ROOT}/${kind}/${vfm}/scannetpp/"
  local dst="${DATASET_ROOT}/${kind}/${vfm}/scannetpp/"
  mkdir -p "${dst}"
  echo "RESTORE ${kind}/${vfm} include=${include} $(date -Is)"
  echo "  src=s3://${BUCKET}.${HOST}/InsScene-15K/${kind}/${vfm}/scannetpp/"
  echo "  dst=${dst}"
  "${ADS_CLI}" \
    -p "${THREADS}" \
    -l "${LISTERS}" \
    --conntimeout 120 \
    --timeout 600 \
    --include "${include}" \
    sync \
    "${src}" \
    "${dst}"
  echo "RC=$? ${kind}/${vfm} $(date -Is)"
}

JOB="${1:-wan_target}"
case "${JOB}" in
  wan_target)     restore_one FEAT_TARGET wan 'feature_t(499|999)_' ;;
  wan_stream)     restore_one FEAT_STREAMING wan 'feature_t(499|999)_' ;;
  cogvideox_stream) restore_one FEAT_STREAMING cogvideox 'feature_t(499|999)_' ;;
  minimax_stream) restore_one FEAT_STREAMING minimax_h3 'feature_t(500|1000)_' ;;
  *)
    echo "unknown job ${JOB}" >&2
    exit 2
    ;;
esac
