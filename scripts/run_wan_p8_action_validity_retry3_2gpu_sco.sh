#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
OUTPUT=${PROJECT_ROOT}/results/analysis/wan_p8_action_validity
FAILED_ARCHIVE=${OUTPUT}/failed_attempts/phase1_batch64_rank_tie_gate_20260825T0911Z
LAYERS=(10 29)
mkdir -p "${FAILED_ARCHIVE}" "${OUTPUT}/launcher"

for layer in "${LAYERS[@]}"; do
  mv "${OUTPUT}/layers/layer${layer}" "${FAILED_ARCHIVE}/"
  mv "${OUTPUT}/layers/layer${layer}.sco.log" "${FAILED_ARCHIVE}/"
  mv "${OUTPUT}/layers/layer${layer}.sco_status.json" "${FAILED_ARCHIVE}/"
done

pids=()
for index in "${!LAYERS[@]}"; do
  layer=${LAYERS[${index}]}
  echo "retry3 launching layer=${layer} cuda=${index}"
  CUDA_VISIBLE_DEVICES=${index} PROJECT_ROOT=${PROJECT_ROOT} LAYER=${layer} \
    bash "${PROJECT_ROOT}/scripts/run_wan_p8_action_validity_layer_sco.sh" &
  pids+=("$!")
done

rc=0
for index in "${!pids[@]}"; do
  if wait "${pids[${index}]}"; then
    echo "retry3 layer=${LAYERS[${index}]} complete"
  else
    layer_rc=$?
    echo "retry3 layer=${LAYERS[${index}]} failed rc=${layer_rc}" >&2
    rc=1
  fi
done
exit "${rc}"
