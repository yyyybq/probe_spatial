#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/umm/users/yinbaiqiao/probe_spatial
OPT_ROOT=${ROOT}/runtime/tstep_probe_opt_v2/source
PYTHON=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python
OUT=${CANARY_OUT:?set CANARY_OUT}
mkdir -p "${OUT}/cases" "${OUT}/metrics"
nvidia-smi --query-gpu=uuid,name --format=csv,noheader > "${OUT}/metrics/gpu_identity.csv"
case_id=wan__t999__l15__p8__streaming_mask__train0__empty_mask
for impl in old optimized; do
  source="${ROOT}"
  if [[ "${impl}" == optimized ]]; then source="${OPT_ROOT}"; fi
  export AOSS_CACHE_DIR="/mnt/local_cache/inscene/tstep_probe_opt_v2_empty_mask/${impl}"
  export AOSS_FS_METRICS="${OUT}/metrics/${impl}__aoss.jsonl"
  "${PYTHON}" "${ROOT}/scripts/tstep_opt_campaign/canary_worker.py" \
    --implementation-root "${source}" --implementation-name "${impl}" \
    --task streaming_mask --backend wan --timestep 999 --layer 15 --prefix 8 \
    --split train --index 0 --train-steps 1 --empty-mask-fixture \
    --output "${OUT}/cases/${impl}__${case_id}.json" \
    > "${OUT}/cases/${impl}__${case_id}.log" 2>&1
done
"${PYTHON}" "${ROOT}/scripts/tstep_opt_campaign/compare_canary.py" \
  --directory "${OUT}/cases" --output "${OUT}/comparison.json" \
  --parameter-policy exact > "${OUT}/comparison.log" 2>&1
touch "${OUT}/CANARY_PASS"
