#!/usr/bin/env bash
set -euo pipefail

MAIN_ROOT=${MAIN_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
OPT_ROOT=${OPT_ROOT:-${MAIN_ROOT}/runtime/tstep_probe_opt_v1/source}
PYTHON=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python
OUT=${CANARY_OUT:?set CANARY_OUT}
mkdir -p "${OUT}/cases" "${OUT}/parameters" "${OUT}/comparisons" "${OUT}/metrics"

nvidia-smi --query-gpu=uuid,name --format=csv,noheader > "${OUT}/metrics/gpu_identity.csv"
nvidia-smi dmon -s pucvmet -d 1 > "${OUT}/metrics/nvidia_dmon.log" 2>&1 &
dmon_pid=$!
vmstat 1 > "${OUT}/metrics/vmstat.log" 2>&1 &
vmstat_pid=$!
cleanup() {
  kill "${dmon_pid}" "${vmstat_pid}" 2>/dev/null || true
  wait "${dmon_pid}" "${vmstat_pid}" 2>/dev/null || true
}
trap cleanup EXIT

tasks=(streaming_depth streaming_mask ego_belief_v2)
for task in "${tasks[@]}"; do
  steps=3
  if [[ "${task}" == streaming_depth || "${task}" == streaming_mask ]]; then steps=1; fi
  case_id="wan__t999__l15__p8__${task}__train0"
  for impl in old old_repeat optimized; do
    root="${MAIN_ROOT}"
    impl_name=old
    if [[ "${impl}" == optimized ]]; then
      root="${OPT_ROOT}"
      impl_name=optimized
    fi
    export AOSS_CACHE_DIR="/mnt/local_cache/inscene/tstep_probe_opt_v1_param_canary/${impl}/${task}"
    export AOSS_FS_METRICS="${OUT}/metrics/${impl}__${case_id}__aoss.jsonl"
    "${PYTHON}" "${MAIN_ROOT}/scripts/tstep_opt_campaign/canary_worker.py" \
      --implementation-root "${root}" --implementation-name "${impl_name}" \
      --task "${task}" --backend wan --timestep 999 --layer 15 --prefix 8 \
      --split train --index 0 --train-steps "${steps}" \
      --parameter-output "${OUT}/parameters/${impl}__${case_id}.pt" \
      --output "${OUT}/cases/${impl}__${case_id}.json" \
      > "${OUT}/cases/${impl}__${case_id}.log" 2>&1
  done
  set +e
  "${PYTHON}" "${MAIN_ROOT}/scripts/tstep_opt_campaign/compare_parameters.py" \
    --reference "${OUT}/parameters/old__${case_id}.pt" \
    --candidate "${OUT}/parameters/old_repeat__${case_id}.pt" \
    --output "${OUT}/comparisons/old_repeat__${case_id}.json" \
    > "${OUT}/comparisons/old_repeat__${case_id}.log" 2>&1
  repeat_rc=$?
  "${PYTHON}" "${MAIN_ROOT}/scripts/tstep_opt_campaign/compare_parameters.py" \
    --reference "${OUT}/parameters/old__${case_id}.pt" \
    --candidate "${OUT}/parameters/optimized__${case_id}.pt" \
    --output "${OUT}/comparisons/optimized__${case_id}.json" \
    > "${OUT}/comparisons/optimized__${case_id}.log" 2>&1
  optimized_rc=$?
  set -e
  printf '%s %s %s\n' "${task}" "${repeat_rc}" "${optimized_rc}" >> "${OUT}/comparison_status.tsv"
done
touch "${OUT}/PARAMETER_CANARY_COMPLETE"
