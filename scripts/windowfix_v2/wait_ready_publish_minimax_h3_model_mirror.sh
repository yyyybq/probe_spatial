#!/usr/bin/env bash
# Poll the MiniMax-H3 checkpoint directory until the mirror manifest validates,
# then publish it as an immutable AOSS model mirror.
set -euo pipefail

project_root="${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}"
model_root="${MODEL_ROOT:-/mnt/umm/users/yinbaiqiao/Consistency/minimax-h3/models/MiniMax-H3}"
python_bin="${PYTHON:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}"
log_path="${1:-${project_root}/logs/windowfix_v2_model_mirror/minimax_h3_wait_ready_publish_$(date -u +%Y%m%dT%H%M%SZ).log}"
interval_sec="${MINIMAX_H3_PUBLISH_POLL_SEC:-120}"

mkdir -p "$(dirname "${log_path}")"
scratch="${project_root}/artifacts/windowfix_v2/model_mirrors/minimax_h3/.readiness"
mkdir -p "${scratch}"

{
  echo "READY_WATCH_START $(date -u +%Y-%m-%dT%H:%M:%SZ) model_root=${model_root}"
  while true; do
    readiness_manifest="${scratch}/readiness_manifest.json"
    if "${python_bin}" "${project_root}/scripts/windowfix_v2/build_minimax_h3_model_mirror_manifest.py" \
        --model-root "${model_root}" \
        --output "${readiness_manifest}" >/dev/null 2>"${scratch}/last_error.log"; then
      echo "READY_VALIDATED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
      exec /bin/bash "${project_root}/scripts/windowfix_v2/publish_minimax_h3_model_mirror.sh"
    fi
    echo "NOT_READY $(date -u +%Y-%m-%dT%H:%M:%SZ) $(tr '\n' ' ' <"${scratch}/last_error.log" | cut -c1-500)"
    sleep "${interval_sec}"
  done
} >>"${log_path}" 2>&1
