#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/umm/users/yinbaiqiao/probe_spatial
LOG_DIR="${ROOT}/logs/remote_eval_20260817"
SSH_KEY=/mnt/umm/users/yinbaiqiao/id_rsa
SSH_OPTS=(-i "${SSH_KEY}" -o UserKnownHostsFile=/tmp/known_hosts -o StrictHostKeyChecking=no)

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

DIAG_OUT="${ROOT}/logs/probing_jobs_remaining_diag_4node_20260817"
A1_OUT="${ROOT}/logs/probing_jobs_remaining_a1_4node_20260817"

launch_remote 10.119.27.144 remaining_diag_s00 remaining_diag_4node_20260817 "${DIAG_OUT}" "${DIAG_OUT}/jobs/shard_00.tsv" 4 0,1,2,3 scripts/run_missing_eval_shard.py
launch_remote 10.119.28.240 remaining_diag_s01 remaining_diag_4node_20260817 "${DIAG_OUT}" "${DIAG_OUT}/jobs/shard_01.tsv" 4 0,1,2,3 scripts/run_missing_eval_shard.py
launch_remote 10.119.16.197 remaining_diag_s02 remaining_diag_4node_20260817 "${DIAG_OUT}" "${DIAG_OUT}/jobs/shard_02.tsv" 4 0,1,2,3 scripts/run_missing_eval_shard.py
launch_remote 10.119.16.197 remaining_diag_s03 remaining_diag_4node_20260817 "${DIAG_OUT}" "${DIAG_OUT}/jobs/shard_03.tsv" 4 4,5,6,7 scripts/run_missing_eval_shard.py

launch_remote 10.119.27.144 remaining_a1_s00 remaining_a1_4node_20260817 "${A1_OUT}" "${A1_OUT}/jobs/shard_00.tsv" 2 4,5 scripts/run_missing_a1_eval_shard.py
launch_remote 10.119.28.240 remaining_a1_s01 remaining_a1_4node_20260817 "${A1_OUT}" "${A1_OUT}/jobs/shard_01.tsv" 2 4,5 scripts/run_missing_a1_eval_shard.py
launch_remote 10.119.27.144 remaining_a1_s02 remaining_a1_4node_20260817 "${A1_OUT}" "${A1_OUT}/jobs/shard_02.tsv" 2 6,7 scripts/run_missing_a1_eval_shard.py
launch_remote 10.119.28.240 remaining_a1_s03 remaining_a1_4node_20260817 "${A1_OUT}" "${A1_OUT}/jobs/shard_03.tsv" 2 6,7 scripts/run_missing_a1_eval_shard.py
