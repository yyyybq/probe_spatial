#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/umm/users/yinbaiqiao/probe_spatial
PYTHON=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python
OUT=${CANARY_OUT:?set CANARY_OUT}
mkdir -p "${OUT}/cases" "${OUT}/parameters" "${OUT}/metrics"
nvidia-smi --query-gpu=uuid,name --format=csv,noheader > "${OUT}/metrics/gpu_identity.csv"
for repeat in 2 3 4; do
  export AOSS_CACHE_DIR="/mnt/local_cache/inscene/tstep_probe_opt_v1_depth_repeat/${repeat}"
  export AOSS_FS_METRICS="${OUT}/metrics/old_repeat${repeat}__aoss.jsonl"
  "${PYTHON}" "${ROOT}/scripts/tstep_opt_campaign/canary_worker.py" \
    --implementation-root "${ROOT}" --implementation-name old \
    --task streaming_depth --backend wan --timestep 999 --layer 15 --prefix 8 \
    --split train --index 0 --train-steps 1 \
    --parameter-output "${OUT}/parameters/old_repeat${repeat}.pt" \
    --output "${OUT}/cases/old_repeat${repeat}.json" \
    > "${OUT}/cases/old_repeat${repeat}.log" 2>&1
done
touch "${OUT}/REPEATS_COMPLETE"
