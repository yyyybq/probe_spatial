#!/usr/bin/env bash
set -euo pipefail

STAGE=${1:?usage: run_wan_p8_exact_canary.sh streaming|target|verify}
PROJECT=/mnt/umm/users/yinbaiqiao/probe_spatial
PY=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python
DATA_ROOT=/mnt/umm/users/yinbaiqiao/InsScene-15K/data
MODEL=/mnt/umm/users/yinbaiqiao/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780a534b6463e45facd96134c9f345acfa5b
MANIFEST=${PROJECT}/manifests/c_retrain_20260722/inscene_c_h124_prefix8.jsonl
SELECTION=${PROJECT}/reports/wan_p8_recovery/canary_selection_20260821_185059.txt
CANARY_TAG=wan_p8_manifest_exact_canary_20260821_185059_2075f3e
CANARY_ROOT=/mnt/umm/users/yinbaiqiao/InsScene-15K/FEATURES_VERSIONED/${CANARY_TAG}
REPORT_DIR=${PROJECT}/reports/wan_p8_recovery/${CANARY_TAG}
STREAMING_ROOT=${CANARY_ROOT}/streaming
TARGET_ROOT=${CANARY_ROOT}/target
LAYERS=(0 5 10 15 25 29)
GPUS=(0 1 2 3 6 7)
BOUNDS=(0 2 4 6 8 10 12)

mkdir -p "${REPORT_DIR}/logs" "${CANARY_ROOT}/provenance"

launch_extract() {
  local mode=$1
  local root=$2
  local -a pids=()
  : > "${REPORT_DIR}/logs/${mode}_pids.tsv"
  for shard in 0 1 2 3 4 5; do
    local start=${BOUNDS[$shard]}
    local end=${BOUNDS[$((shard + 1))]}
    local gpu=${GPUS[$shard]}
    local log=${REPORT_DIR}/logs/${mode}_shard${shard}_gpu${gpu}_${start}_${end}.log
    (
      cd "${PROJECT}"
      export CUDA_VISIBLE_DEVICES=${gpu}
      set +e
      "${PY}" scripts/extract_wan_manifest_exact.py \
        --mode "${mode}" \
        --manifest "${MANIFEST}" \
        --selection-file "${SELECTION}" \
        --data-root "${DATA_ROOT}" \
        --out-root "${root}" \
        --model-id "${MODEL}" \
        --layers "${LAYERS[@]}" \
        --target-horizon 1 \
        --seed 42 \
        --start "${start}" \
        --end "${end}"
      rc=$?
      set -e
      printf '{"stage":"%s","shard":%d,"gpu":%d,"start":%d,"end":%d,"rc":%d}\n' \
        "${mode}" "${shard}" "${gpu}" "${start}" "${end}" "${rc}" \
        > "${REPORT_DIR}/logs/${mode}_shard${shard}.status.json"
      exit "${rc}"
    ) > "${log}" 2>&1 &
    pids+=("$!")
    printf '%d\t%d\t%d\t%d\t%d\t%s\n' \
      "${shard}" "${gpu}" "${start}" "${end}" "$!" "${log}" \
      >> "${REPORT_DIR}/logs/${mode}_pids.tsv"
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "${mode} canary extraction failed" >&2
    return 2
  fi
}

verify_canary() {
  cd "${PROJECT}"
  "${PY}" scripts/verify_wan_p8_dataset_exact.py \
    --manifest "${MANIFEST}" \
    --selection-file "${SELECTION}" \
    --filtered-manifest "${REPORT_DIR}/canary_manifest_40.jsonl" \
    --data-root "${DATA_ROOT}" \
    --streaming-root "${STREAMING_ROOT}" \
    --target-root "${TARGET_ROOT}" \
    --out-json "${REPORT_DIR}/dataset_canary.json" \
    --num-workers 0 \
    > "${REPORT_DIR}/logs/dataset_canary.log" 2>&1
}

case "${STAGE}" in
  streaming) launch_extract streaming "${STREAMING_ROOT}" ;;
  target) launch_extract target "${TARGET_ROOT}" ;;
  verify) verify_canary ;;
  *) echo "unknown stage: ${STAGE}" >&2; exit 2 ;;
esac
