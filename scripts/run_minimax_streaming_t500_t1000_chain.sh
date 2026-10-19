#!/usr/bin/env bash
# Run MiniMax-H3 streaming t500 and t1000 serially on one 8-GPU node.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}"
CHAIN_LOG_ROOT="${CHAIN_LOG_ROOT:?set CHAIN_LOG_ROOT=<job log directory>}"
DONE_DIR="${DONE_DIR:-${PROJECT_ROOT}/logs/tstep_resume4/aoss_done}"

cd "${PROJECT_ROOT}"
mkdir -p "${CHAIN_LOG_ROOT}"

for timestep in 500 1000; do
  done_index="${DONE_DIR}/FEAT_STREAMING_minimax_h3_t${timestep}.txt"
  [[ -f "${done_index}" ]] || {
    echo "ERROR: missing completion index: ${done_index}" >&2
    exit 1
  }
  echo "[$(date -Is)] starting MiniMax-H3 streaming t${timestep}; index_lines=$(wc -l < "${done_index}")"
  VFM=minimax_h3 \
  T="${timestep}" \
  EXTRACT_STREAM=1 \
  EXTRACT_TARGET=0 \
  EXTRACT_DONE_INDEX="${done_index}" \
  LOG_DIR="${CHAIN_LOG_ROOT}/t${timestep}_stream" \
    bash scripts/run_streaming_feat_extract_8gpu.sh
  echo "[$(date -Is)] completed MiniMax-H3 streaming t${timestep}"
done

echo "[$(date -Is)] MiniMax-H3 streaming t500 -> t1000 chain complete"
