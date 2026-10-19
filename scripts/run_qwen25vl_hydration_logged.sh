#!/usr/bin/env bash
set -uo pipefail

project_root=/mnt/umm/users/yinbaiqiao/probe_spatial
run_root=${project_root}/artifacts/main_v2_probe_v2_816_20260917/qwen_snapshot_hydration_20260922_r2
log_path=${run_root}/hydrate.log
receipt_path=${run_root}/receipt.json

mkdir -p "${run_root}"
cd "${project_root}" || exit 2
{
  date -u +start_utc=%Y-%m-%dT%H:%M:%SZ
  /mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python \
    scripts/hydrate_qwen25vl_7b_snapshot.py \
    --receipt "${receipt_path}"
  rc=$?
  date -u +end_utc=%Y-%m-%dT%H:%M:%SZ
  echo "exit_code=${rc}"
  exit "${rc}"
} >"${log_path}" 2>&1
