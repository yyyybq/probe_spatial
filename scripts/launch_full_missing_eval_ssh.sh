#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/umm/users/yinbaiqiao/probe_spatial
SSH_KEY=/mnt/umm/users/yinbaiqiao/id_rsa
SSH_OPTS=(-i "${SSH_KEY}" -o UserKnownHostsFile=/tmp/known_hosts -o StrictHostKeyChecking=no)
LOG_DIR="${ROOT}/logs/remote_eval_20260817"

launch_remote() {
  local host="$1"
  local name="$2"
  local run_id="$3"
  local out_dir="$4"
  local shard_tsv="$5"
  local slots="$6"
  local gpus="$7"
  local runner="$8"
  local stdout_log="${LOG_DIR}/${host}_${name}.out"
  local pid_file="${LOG_DIR}/${host}_${name}.pid"

  ssh "${SSH_OPTS[@]}" "root@${host}" \
    "mkdir -p '${LOG_DIR}' && cd '${ROOT}' && nohup env RUN_ID='${run_id}' OUT_DIR='${out_dir}' SHARD_TSV='${shard_tsv}' SLOTS='${slots}' GPU_IDS_CSV='${gpus}' /usr/bin/python3 '${runner}' > '${stdout_log}' 2>&1 < /dev/null & echo \$! > '${pid_file}' && cat '${pid_file}'"
}

launch_remote \
  10.119.27.144 \
  full_diag_s00 \
  full_missing_eval_20260817 \
  "${ROOT}/logs/probing_jobs_full_missing_eval_20260817" \
  "${ROOT}/logs/probing_jobs_full_missing_eval_20260817/jobs/shard_00.tsv" \
  4 \
  0,1,2,3 \
  scripts/run_missing_eval_shard.py

launch_remote \
  10.119.28.240 \
  full_diag_s01 \
  full_missing_eval_20260817 \
  "${ROOT}/logs/probing_jobs_full_missing_eval_20260817" \
  "${ROOT}/logs/probing_jobs_full_missing_eval_20260817/jobs/shard_01.tsv" \
  4 \
  0,1,2,3 \
  scripts/run_missing_eval_shard.py

launch_remote \
  10.119.27.144 \
  full_a1_s00 \
  full_missing_a1_eval_20260817 \
  "${ROOT}/logs/probing_jobs_full_missing_a1_eval_20260817" \
  "${ROOT}/logs/probing_jobs_full_missing_a1_eval_20260817/jobs/shard_00.tsv" \
  4 \
  4,5,6,7 \
  scripts/run_missing_a1_eval_shard.py

launch_remote \
  10.119.28.240 \
  full_a1_s01 \
  full_missing_a1_eval_20260817 \
  "${ROOT}/logs/probing_jobs_full_missing_a1_eval_20260817" \
  "${ROOT}/logs/probing_jobs_full_missing_a1_eval_20260817/jobs/shard_01.tsv" \
  4 \
  4,5,6,7 \
  scripts/run_missing_a1_eval_shard.py
