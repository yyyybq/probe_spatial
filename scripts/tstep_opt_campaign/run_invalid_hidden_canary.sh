#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/umm/users/yinbaiqiao/probe_spatial
OPT_ROOT=${OPT_ROOT:-${ROOT}/runtime/tstep_probe_opt_v2/source}
PYTHON=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python
OUT=${CANARY_OUT:?set CANARY_OUT}
INDEX=${CANARY_INDEX:-4}
mkdir -p "${OUT}/cases" "${OUT}/metrics"
nvidia-smi --query-gpu=uuid,name --format=csv,noheader > "${OUT}/metrics/gpu_identity.csv"
for task in ego_belief ego_belief_v2; do
  case_id="wan__t999__l15__p8__${task}__train${INDEX}__invalid"
  for impl in old optimized; do
    source="${ROOT}"
    if [[ "${impl}" == optimized ]]; then source="${OPT_ROOT}"; fi
    export AOSS_CACHE_DIR="/mnt/local_cache/inscene/tstep_probe_opt_v2_invalid/${impl}/${task}"
    export AOSS_FS_METRICS="${OUT}/metrics/${impl}__${case_id}__aoss.jsonl"
    export INSCENE_LABEL_CACHE_ROOT="${ROOT}/artifacts/tstep_probe_opt_v1/invalid_canary_label_sidecar"
    "${PYTHON}" "${ROOT}/scripts/tstep_opt_campaign/canary_worker.py" \
      --implementation-root "${source}" --implementation-name "${impl}" \
      --task "${task}" --backend wan --timestep 999 --layer 15 --prefix 8 \
      --split train --index "${INDEX}" --train-steps 1 \
      --output "${OUT}/cases/${impl}__${case_id}.json" \
      > "${OUT}/cases/${impl}__${case_id}.log" 2>&1
  done
done
"${PYTHON}" "${ROOT}/scripts/tstep_opt_campaign/compare_canary.py" \
  --directory "${OUT}/cases" --output "${OUT}/comparison.json" \
  --parameter-policy exact > "${OUT}/comparison.log" 2>&1
touch "${OUT}/CANARY_PASS"
