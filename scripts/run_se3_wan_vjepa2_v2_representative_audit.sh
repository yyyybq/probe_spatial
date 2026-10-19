#!/usr/bin/env bash
set -euo pipefail
umask 0022

PROJECT_ROOT=/mnt/umm/users/yinbaiqiao/probe_spatial
PYTHON=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python
RESULT=${PROJECT_ROOT}/results/experimental/se3_wan_vjepa2_exact_v2_layers_20260916_r1

export AOSS_API_HOST=aoss-internal.cn-fz-01.fjscmsapi-oss.com
export AOSS_HOST=baiqiao.${AOSS_API_HOST}
export NO_PROXY=localhost,127.0.0.1,::1,${AOSS_HOST},.${AOSS_API_HOST},${AOSS_API_HOST}
export AOSS_CONF=/mnt/umm/users/yinbaiqiao/aoss/petreloss.conf
export CONF=${AOSS_CONF}
export AOSS_CACHE_DIR=/mnt/local_cache/inscene/se3_wan_vjepa2_v2_representative
export AOSS_FS_METRICS=${RESULT}/logs/representative_aoss_metrics.jsonl

cd "${PROJECT_ROOT}"
mkdir -p "${RESULT}/logs" "${AOSS_CACHE_DIR}"
"${PYTHON}" -m analyses.se3_protocol_v2.wan_vjepa2_v2_data prepare

status=0
for cell in wan_l15 vjepa2_l11; do
  if "${PYTHON}" -m analyses.se3_protocol_v2.wan_vjepa2_v2_data materialize --cell "${cell}" \
      >"${RESULT}/logs/${cell}_materialize.log" 2>&1; then
    printf '%s\tPASS\n' "${cell}" | tee -a "${RESULT}/logs/representative_status.tsv"
  else
    printf '%s\tFAIL\n' "${cell}" | tee -a "${RESULT}/logs/representative_status.tsv"
    status=1
  fi
done
exit "${status}"
