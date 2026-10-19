#!/usr/bin/env bash
set -euo pipefail
umask 0022

CELL=${CELL:?CELL is required}
PROJECT_ROOT=/mnt/umm/users/yinbaiqiao/probe_spatial
PYTHON=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python
RESULT=${PROJECT_ROOT}/results/experimental/se3_wan_vjepa2_exact_v2_layers_20260916_r1

case "${CELL}" in
  wan_l0|wan_l5|wan_l10|wan_l15|wan_l20|wan_l25|wan_l29|vjepa2_l0|vjepa2_l5|vjepa2_l11|vjepa2_l17|vjepa2_l23) ;;
  *) echo "unregistered cell ${CELL}" >&2; exit 2 ;;
esac

export AOSS_API_HOST=aoss-internal.cn-fz-01.fjscmsapi-oss.com
export AOSS_HOST=baiqiao.${AOSS_API_HOST}
export NO_PROXY=localhost,127.0.0.1,::1,${AOSS_HOST},.${AOSS_API_HOST},${AOSS_API_HOST}
export AOSS_CONF=/mnt/umm/users/yinbaiqiao/aoss/petreloss.conf
export CONF=${AOSS_CONF}
export AOSS_CACHE_DIR=/mnt/local_cache/inscene/se3_wan_vjepa2_v2/${CELL}
export AOSS_FS_METRICS=${RESULT}/logs/${CELL}_aoss_metrics.jsonl

cd "${PROJECT_ROOT}"
mkdir -p "${RESULT}/logs" "${AOSS_CACHE_DIR}"
test -r "${RESULT}/data/common_manifest.jsonl"
test -r "${RESULT}/data/route_plan.jsonl"
"${PYTHON}" -m analyses.se3_protocol_v2.wan_vjepa2_v2_data materialize --cell "${CELL}" \
  >"${RESULT}/logs/${CELL}_materialize.log" 2>&1
"${PYTHON}" -m analyses.se3_protocol_v2.wan_vjepa2_v2_train --cell "${CELL}" --device cuda \
  >"${RESULT}/logs/${CELL}_train_eval.log" 2>&1
"${PYTHON}" -m analyses.se3_protocol_v2.wan_vjepa2_v2_aggregate \
  >"${RESULT}/logs/${CELL}_incremental_aggregate.log" 2>&1
