#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
OUTPUT=${PROJECT_ROOT}/results/analysis/wan_p8_action_validity
FAILED_ARCHIVE=${OUTPUT}/failed_attempts/phase1_prediction_batch_kernel_gate_20260825T0905Z
LAYERS=(0 10 15 25 29)
mkdir -p "${FAILED_ARCHIVE}" "${OUTPUT}/launcher"

for layer in "${LAYERS[@]}"; do
  mv "${OUTPUT}/layers/layer${layer}" "${FAILED_ARCHIVE}/"
  mv "${OUTPUT}/layers/layer${layer}.sco.log" "${FAILED_ARCHIVE}/"
  mv "${OUTPUT}/layers/layer${layer}.sco_status.json" "${FAILED_ARCHIVE}/"
done

pids=()
for index in "${!LAYERS[@]}"; do
  layer=${LAYERS[${index}]}
  echo "retry2 launching layer=${layer} cuda=${index}"
  CUDA_VISIBLE_DEVICES=${index} PROJECT_ROOT=${PROJECT_ROOT} LAYER=${layer} \
    bash "${PROJECT_ROOT}/scripts/run_wan_p8_action_validity_layer_sco.sh" &
  pids+=("$!")
done

rc=0
for index in "${!pids[@]}"; do
  if wait "${pids[${index}]}"; then
    echo "retry2 layer=${LAYERS[${index}]} complete"
  else
    layer_rc=$?
    echo "retry2 layer=${LAYERS[${index}]} failed rc=${layer_rc}" >&2
    rc=1
  fi
done
exit "${rc}"
