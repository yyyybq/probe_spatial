#!/usr/bin/env bash
set -euo pipefail

MAIN_ROOT=${MAIN_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
OLD_ROOT=${OLD_ROOT:-${MAIN_ROOT}}
OPT_ROOT=${OPT_ROOT:-${MAIN_ROOT}/runtime/tstep_probe_opt_v1/source}
PYTHON=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python
OUT=${CANARY_OUT:?set CANARY_OUT}
BACKEND=${CANARY_BACKEND:?set CANARY_BACKEND}
mkdir -p "${OUT}/cases" "${OUT}/metrics"

case "${BACKEND}" in
  cogvideox) timestep=499; layer=20; prefix=12; split=val; index=0 ;;
  minimax_h3) timestep=1000; layer=25; prefix=24; split=train; index=${CANARY_INDEX:--1} ;;
  *) echo "unsupported backend ${BACKEND}" >&2; exit 2 ;;
esac

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

tasks=(streaming_depth streaming_mask view_consistency ego_belief ego_belief_v2 action_dynamics path_integration counterfactual)
for task in "${tasks[@]}"; do
  steps=1
  case_id="${BACKEND}__t${timestep}__l${layer}__p${prefix}__${task}__${split}${index}"
  for impl in old optimized; do
    root="${OLD_ROOT}"
    if [[ "${impl}" == optimized ]]; then root="${OPT_ROOT}"; fi
    export AOSS_CACHE_DIR="/mnt/local_cache/inscene/tstep_probe_opt_v1_backend_canary/${BACKEND}/${impl}/${task}"
    export AOSS_FS_METRICS="${OUT}/metrics/${impl}__${case_id}__aoss.jsonl"
    "${PYTHON}" "${MAIN_ROOT}/scripts/tstep_opt_campaign/canary_worker.py" \
      --implementation-root "${root}" --implementation-name "${impl}" \
      --task "${task}" --backend "${BACKEND}" --timestep "${timestep}" \
      --layer "${layer}" --prefix "${prefix}" --split "${split}" --index "${index}" \
      --train-steps "${steps}" --output "${OUT}/cases/${impl}__${case_id}.json" \
      > "${OUT}/cases/${impl}__${case_id}.log" 2>&1
  done
done

"${PYTHON}" "${MAIN_ROOT}/scripts/tstep_opt_campaign/compare_canary.py" \
  --directory "${OUT}/cases" --output "${OUT}/comparison.json" \
  --parameter-policy skip > "${OUT}/comparison.log" 2>&1
touch "${OUT}/CANARY_PASS"
