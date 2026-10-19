#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 8 ]]; then
  echo "usage: $0 SNAPSHOT_ROOT PROJECT_ROOT RUN_DIR TASK MODEL CHECKPOINT SHA256 REFERENCE_SUMMARY" >&2
  exit 64
fi

SNAPSHOT_ROOT=$1
PROJECT_ROOT=$2
RUN_DIR=$3
TASK=$4
MODEL=$5
CHECKPOINT=$6
CHECKPOINT_SHA256=$7
REFERENCE_SUMMARY=$8
CELL="${TASK}_${MODEL}"

mkdir -p "${RUN_DIR}/launcher_logs"
exec > >(tee -a "${RUN_DIR}/launcher_logs/${CELL}.log") 2>&1

echo "P16_REPAIR_CELL=${CELL}"
echo "HOSTNAME=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "SNAPSHOT_ROOT=${SNAPSHOT_ROOT}"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "CHECKPOINT=${CHECKPOINT}"
echo "REFERENCE_SUMMARY=${REFERENCE_SUMMARY}"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader

export PYTHONPATH="${SNAPSHOT_ROOT}:${PYTHONPATH:-}"
export PROBE_VALIDITY_PROJECT_ROOT="${PROJECT_ROOT}"
export AOSS_CONF="/mnt/umm/users/yinbaiqiao/aoss/petreloss.conf"
export AOSS_API_HOST="aoss-internal.cn-fz-01.fjscmsapi-oss.com"
export NO_PROXY="localhost,127.0.0.1,::1,baiqiao.aoss-internal.cn-fz-01.fjscmsapi-oss.com,.aoss-internal.cn-fz-01.fjscmsapi-oss.com,aoss-internal.cn-fz-01.fjscmsapi-oss.com"

cd "${SNAPSHOT_ROOT}"
/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python \
  "${SNAPSHOT_ROOT}/analyses/probe_validity/run_probe_validity.py" \
  --p16-repair "${TASK}" \
  --model "${MODEL}" \
  --p16-output "${RUN_DIR}/cells/${CELL}" \
  --checkpoint "${CHECKPOINT}" \
  --checkpoint-sha256 "${CHECKPOINT_SHA256}" \
  --reference-summary "${REFERENCE_SUMMARY}"
