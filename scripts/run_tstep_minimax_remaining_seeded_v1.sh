#!/usr/bin/env bash
# Remaining seeded-v1 MiniMax-H3 extracts on one 8-GPU node:
# target t500 resume, then stream t1000, then target t1000.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}"
CHAIN_LOG_ROOT="${CHAIN_LOG_ROOT:?set CHAIN_LOG_ROOT}"
DONE_DIR="${DONE_DIR:-${PROJECT_ROOT}/logs/tstep_canonical_v2_seeded_v1/aoss_done}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/umm/users/yinbaiqiao/InsScene-15K}"
PROTOCOL=inscene_streaming_canonical_v2_obs28_h124
PROTOCOL_SHA=215c720c8537f4dfae6243bedcc7339fbb7e3ba8a6ca2dcaf04f49cd4b738e4f

cd "${PROJECT_ROOT}"
mkdir -p "${CHAIN_LOG_ROOT}"
chmod +x scripts/run_streaming_feat_extract_8gpu.sh scripts/run_tstep_sync_loop.sh

run_one() {
  local kind="$1" timestep="$2" namespace="$3"
  local stream=0 target=0 mode_tag
  if [[ "${kind}" == stream ]]; then
    stream=1
    mode_tag=stream
  else
    target=1
    mode_tag=target
  fi
  local done_index="${DONE_DIR}/${namespace}_minimax_h3_t${timestep}.txt"
  [[ -f "${done_index}" && ! -L "${done_index}" ]] || {
    echo "ERROR: missing done-index ${done_index}" >&2
    exit 1
  }
  chmod a+r -- "${done_index}" 2>/dev/null || true
  local step_log="${CHAIN_LOG_ROOT}/${mode_tag}_t${timestep}"
  echo "[$(date -Is)] START minimax_h3 ${mode_tag} t${timestep} index_lines=$(wc -l < "${done_index}" 2>/dev/null || echo unreadable)"
  CANONICAL_PROTOCOL="${PROTOCOL}" \
  CANONICAL_PROTOCOL_SHA256="${PROTOCOL_SHA}" \
  REQUIRE_H800=1 \
  FEATURE_NAMESPACE="${namespace}" \
  VFM=minimax_h3 \
  T="${timestep}" \
  EXTRACT_STREAM="${stream}" \
  EXTRACT_TARGET="${target}" \
  EXTRACT_DONE_INDEX="${done_index}" \
  DATASET_ROOT="${DATASET_ROOT}" \
  N_GPUS=8 \
  SCENE_START=0 \
  SCENE_END=682 \
  SYNC_EVERY_SEC="${SYNC_EVERY_SEC:-180}" \
  SYNC_EVERY_GIB="${SYNC_EVERY_GIB:-80}" \
  LOG_DIR="${step_log}" \
    bash scripts/run_streaming_feat_extract_8gpu.sh
  echo "[$(date -Is)] COMPLETE minimax_h3 ${mode_tag} t${timestep} index_lines=$(wc -l < "${done_index}" 2>/dev/null || echo unreadable)"
}

run_one target 500 FEAT_TARGET_CANONICAL_V2_SEEDED_V1
LOCK_DIR="${PROJECT_ROOT}/logs/tstep_canonical_v2_seeded_v1/step_locks"
if [[ -f "${LOCK_DIR}/FEAT_STREAMING_CANONICAL_V2_SEEDED_V1_minimax_h3_t1000.lock" || \
      -f "${LOCK_DIR}/FEAT_TARGET_CANONICAL_V2_SEEDED_V1_minimax_h3_t1000.lock" ]]; then
  echo "[$(date -Is)] MiniMax t1000 stream/target owned by dedicated parallel jobs; chain stops after target t500"
  exit 0
fi
run_one stream 1000 FEAT_STREAMING_CANONICAL_V2_SEEDED_V1
run_one target 1000 FEAT_TARGET_CANONICAL_V2_SEEDED_V1
echo "[$(date -Is)] MiniMax remaining seeded-v1 chain complete"
