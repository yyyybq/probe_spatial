#!/usr/bin/env bash
# Run the remaining timestep extraction campaign serially on one 8-GPU node.
# Each step invokes the accepted extractor/sync runner; set -e guarantees that
# a failed extraction, failed final sync, or residual matching local feature
# prevents every later step from starting.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}"
CAMPAIGN_LOG_ROOT="${CAMPAIGN_LOG_ROOT:?set CAMPAIGN_LOG_ROOT=<campaign log directory>}"
DONE_DIR="${DONE_DIR:-${PROJECT_ROOT}/logs/tstep_resume4/aoss_done}"
SYNC_EVERY_SEC="${SYNC_EVERY_SEC:-180}"
SYNC_EVERY_GIB="${SYNC_EVERY_GIB:-80}"

cd "${PROJECT_ROOT}"
mkdir -p "${CAMPAIGN_LOG_ROOT}"
CAMPAIGN_LOG="${CAMPAIGN_LOG_ROOT}/campaign.log"
STATE_FILE="${CAMPAIGN_LOG_ROOT}/campaign.state.tsv"
printf 'ordinal\tvfm\tmode\ttimestep\tstarted_at\tcompleted_at\tdone_index_lines\n' > "${STATE_FILE}"

require_min_bytes() {
  local path="$1" minimum="$2" size=0
  [[ -f "${path}" ]] && size="$(wc -c < "${path}")"
  if [[ "${size}" -lt "${minimum}" ]]; then
    echo "ERROR: required source looks truncated: ${path} (${size} < ${minimum} bytes)" >&2
    exit 1
  fi
}
require_min_bytes features/run_inscene15k.py 50000
require_min_bytes features/cogvideox/cogvideox_feature_i2v.py 10000
require_min_bytes features/minimax_h3/minimax_h3_feature.py 10000
require_min_bytes scripts/run_streaming_feat_extract_8gpu.sh 5000
require_min_bytes scripts/run_tstep_sync_loop.sh 5000

run_step() {
  local ordinal="$1" vfm="$2" mode="$3" timestep="$4"
  local feature_kind mode_tag extract_stream extract_target done_index step_log started completed
  case "${mode}" in
    streaming_prefix)
      feature_kind=FEAT_STREAMING
      mode_tag=stream
      extract_stream=1
      extract_target=0
      ;;
    target_isolated)
      feature_kind=FEAT_TARGET
      mode_tag=target
      extract_stream=0
      extract_target=1
      ;;
    *) echo "ERROR: invalid campaign mode: ${mode}" >&2; exit 2 ;;
  esac
  done_index="${DONE_DIR}/${feature_kind}_${vfm}_t${timestep}.txt"
  [[ -f "${done_index}" ]] || {
    echo "ERROR: missing completion index: ${done_index}" >&2
    exit 1
  }
  step_log="${CAMPAIGN_LOG_ROOT}/$(printf '%02d' "${ordinal}")_${vfm}_t${timestep}_${mode_tag}"
  started="$(date -Is)"
  echo "[${started}] START ordinal=${ordinal} vfm=${vfm} mode=${mode} t=${timestep} index_lines=$(wc -l < "${done_index}")" | tee -a "${CAMPAIGN_LOG}"
  VFM="${vfm}" \
  T="${timestep}" \
  EXTRACT_STREAM="${extract_stream}" \
  EXTRACT_TARGET="${extract_target}" \
  EXTRACT_DONE_INDEX="${done_index}" \
  LOG_DIR="${step_log}" \
  SYNC_EVERY_SEC="${SYNC_EVERY_SEC}" \
  SYNC_EVERY_GIB="${SYNC_EVERY_GIB}" \
    bash scripts/run_streaming_feat_extract_8gpu.sh
  completed="$(date -Is)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${ordinal}" "${vfm}" "${mode}" "${timestep}" "${started}" "${completed}" \
    "$(wc -l < "${done_index}")" >> "${STATE_FILE}"
  echo "[${completed}] COMPLETE ordinal=${ordinal} vfm=${vfm} mode=${mode} t=${timestep}" | tee -a "${CAMPAIGN_LOG}"
}

# Required order: MiniMax target, CogVideoX streaming, CogVideoX target.
run_step 1 minimax_h3 target_isolated 500
run_step 2 minimax_h3 target_isolated 1000
run_step 3 cogvideox streaming_prefix 499
run_step 4 cogvideox streaming_prefix 999
run_step 5 cogvideox target_isolated 499
run_step 6 cogvideox target_isolated 999

echo "[$(date -Is)] CAMPAIGN COMPLETE" | tee -a "${CAMPAIGN_LOG}"
