#!/usr/bin/env bash
# Wait for an existing MiniMax-H3 download process, then publish the verified
# checkpoint as an immutable AOSS model mirror.
set -euo pipefail

download_pid="${1:?usage: watch_minimax_h3_download_and_publish.sh <download-pid> [log-path]}"
project_root="${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}"
log_path="${2:-${project_root}/logs/windowfix_v2_model_mirror/minimax_h3_download_publish_$(date -u +%Y%m%dT%H%M%SZ).log}"

mkdir -p "$(dirname "${log_path}")"
{
  echo "WATCH_START $(date -u +%Y-%m-%dT%H:%M:%SZ) download_pid=${download_pid}"
  while kill -0 "${download_pid}" 2>/dev/null; do
    sleep 30
  done
  echo "DOWNLOAD_PROCESS_EXITED $(date -u +%Y-%m-%dT%H:%M:%SZ) download_pid=${download_pid}"
  exec /bin/bash "${project_root}/scripts/windowfix_v2/publish_minimax_h3_model_mirror.sh"
} >>"${log_path}" 2>&1
