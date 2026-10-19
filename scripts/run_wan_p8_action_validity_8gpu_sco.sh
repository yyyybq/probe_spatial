#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
OUTPUT=${PROJECT_ROOT}/results/analysis/wan_p8_action_validity
LAYERS=(0 5 10 15 25 29)
FAILED_ARCHIVE=${FAILED_ARCHIVE:-${OUTPUT}/failed_attempts/phase1_target_cpu_reduction_gate_20260825T0900Z}
mkdir -p "${OUTPUT}/launcher" "${FAILED_ARCHIVE}"

# Preserve the first fail-closed attempt before creating retry outputs.
if [[ -e "${OUTPUT}/layers/layer0.sco_status.json" ]]; then
  for layer in "${LAYERS[@]}"; do
    mv "${OUTPUT}/layers/layer${layer}" "${FAILED_ARCHIVE}/"
    mv "${OUTPUT}/layers/layer${layer}.sco.log" "${FAILED_ARCHIVE}/"
    mv "${OUTPUT}/layers/layer${layer}.sco_status.json" "${FAILED_ARCHIVE}/"
  done
fi

pids=()
for index in "${!LAYERS[@]}"; do
  layer=${LAYERS[${index}]}
  echo "launching layer=${layer} cuda=${index}"
  CUDA_VISIBLE_DEVICES=${index} PROJECT_ROOT=${PROJECT_ROOT} LAYER=${layer} \
    bash "${PROJECT_ROOT}/scripts/run_wan_p8_action_validity_layer_sco.sh" &
  pids+=("$!")
done

rc=0
for index in "${!pids[@]}"; do
  if wait "${pids[${index}]}"; then
    echo "layer=${LAYERS[${index}]} complete"
  else
    layer_rc=$?
    echo "layer=${LAYERS[${index}]} failed rc=${layer_rc}" >&2
    rc=1
  fi
done
exit "${rc}"
