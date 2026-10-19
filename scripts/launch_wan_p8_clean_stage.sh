#!/usr/bin/env bash
set -euo pipefail

STAGE=${1:?usage: launch_wan_p8_clean_stage.sh <target|train>}
RUN_TAG=${RUN_TAG:?RUN_TAG is required}

PROJECT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
DATA_ROOT=${INSCENE_DATA_ROOT:-/mnt/umm/users/yinbaiqiao/InsScene-15K/data}
FEATURE_ROOT=/mnt/umm/users/yinbaiqiao/InsScene-15K/FEATURES_VERSIONED/${RUN_TAG}
STREAMING_ROOT=${FEATURE_ROOT}/streaming
TARGET_ROOT=${FEATURE_ROOT}/target
REPORT_DIR=${PROJECT}/reports/wan_p8_clean_rerun/${RUN_TAG}
PY=${PYTHON_BIN:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}
MODEL=/mnt/umm/users/yinbaiqiao/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780a534b6463e45facd96134c9f345acfa5b
MANIFEST=${PROJECT}/manifests/c_retrain_20260722/inscene_c_h124_prefix8.jsonl

mkdir -p "${REPORT_DIR}/logs"
export PYTHONPATH="${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
export PROJECT_ROOT="${PROJECT}"
export HF_HOME=${HF_HOME:-/mnt/umm/users/yinbaiqiao/.cache/huggingface}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-root}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1
mkdir -p "${MPLCONFIGDIR}"

GPUS=(0 1 2 3 6 7)
LAYERS=(0 5 10 15 25 29)

write_status() {
  local path=$1
  local stage=$2
  local layer_or_shard=$3
  local rc=$4
  local start_ts=$5
  local end_ts=$6
  printf '{"stage":"%s","id":"%s","rc":%s,"start":"%s","end":"%s","run_tag":"%s"}\n' \
    "${stage}" "${layer_or_shard}" "${rc}" "${start_ts}" "${end_ts}" "${RUN_TAG}" > "${path}"
}

launch_target() {
  local starts=(0 114 228 342 456 570)
  local ends=(114 228 342 456 570 682)
  : > "${REPORT_DIR}/logs/target_pids.txt"
  for i in 0 1 2 3 4 5; do
    local gpu=${GPUS[$i]}
    local start=${starts[$i]}
    local end=${ends[$i]}
    local log="${REPORT_DIR}/logs/target_shard${i}_gpu${gpu}_${start}_${end}.log"
    local status="${REPORT_DIR}/logs/target_shard${i}.status.json"
    (
      cd "${PROJECT}"
      start_ts=$(date -u +%FT%TZ)
      echo "START ${start_ts} stage=target shard=${i} gpu=${gpu} start=${start} end=${end}"
      set +e
      CUDA_VISIBLE_DEVICES=${gpu} "${PY}" -m features.run_inscene15k \
        --data-root "${DATA_ROOT}" \
        --out-root "${TARGET_ROOT}" \
        --vfm wan \
        --model-id "${MODEL}" \
        --prompt . \
        --mode target_isolated \
        --target-from-streaming-windows \
        --target-prefix-lengths 8 \
        --target-horizons 1 \
        --source scannetpp \
        --output-layers 0 5 10 15 25 29 \
        --prefix-lengths 8 \
        --prefix-max-len 24 \
        --streaming-motion-step 0.35 \
        --streaming-rotation-weight 0.5 \
        --streaming-max-windows-per-scene 4 \
        --start "${start}" \
        --end "${end}"
      rc=$?
      set -e
      end_ts=$(date -u +%FT%TZ)
      echo "END ${end_ts} stage=target shard=${i} gpu=${gpu} rc=${rc}"
      write_status "${status}" target "${i}" "${rc}" "${start_ts}" "${end_ts}"
      exit "${rc}"
    ) > "${log}" 2>&1 &
    echo -e "${i}\t${gpu}\t${start}\t${end}\t$!\t${log}" >> "${REPORT_DIR}/logs/target_pids.txt"
  done
  cat "${REPORT_DIR}/logs/target_pids.txt"
}

launch_train() {
  : > "${REPORT_DIR}/logs/train_pids.txt"
  for i in 0 1 2 3 4 5; do
    local layer=${LAYERS[$i]}
    local gpu=${GPUS[$i]}
    local name="${RUN_TAG}_canonical_clean_action_dynamics_wan_p8_layer${layer}"
    local log="${REPORT_DIR}/logs/train_layer${layer}.log"
    local status="${REPORT_DIR}/logs/train_layer${layer}.status.json"
    (
      cd "${PROJECT}"
      export INSCENE_DATA_ROOT="${DATA_ROOT}"
      export INSCENE_MANIFEST="${MANIFEST}"
      export INSCENE_STREAMING_FEAT_ROOT="${STREAMING_ROOT}"
      export INSCENE_TARGET_FEAT_ROOT="${TARGET_ROOT}"
      start_ts=$(date -u +%FT%TZ)
      echo "START ${start_ts} stage=train layer=${layer} gpu=${gpu} name=${name}"
      echo "manifest=${INSCENE_MANIFEST}"
      echo "streaming_root=${INSCENE_STREAMING_FEAT_ROOT}"
      echo "target_root=${INSCENE_TARGET_FEAT_ROOT}"
      set +e
      CUDA_VISIBLE_DEVICES=${gpu} "${PY}" vidfm3d/train.py \
        experiment=inscene15k_streaming/action_dynamics_wan_v1 \
        vfm_name=wan \
        video_channels=1536 \
        streaming_feat_root="${STREAMING_ROOT}" \
        target_feat_root="${TARGET_ROOT}" \
        feature_layer="${layer}" \
        feature_timestep=749 \
        prefix_len=8 \
        job_name="${name}" \
        paths.run_folder_name="${RUN_TAG}/${name}" \
        logger.wandb.name="${name}" \
        logger.wandb.group="${RUN_TAG}" \
        logger.wandb.offline=true \
        autoresume=false \
        ckpt_path=null \
        data.data_module.batch_size_per_device=8 \
        data.data_module.batch_size_per_device_val=8 \
        data.data_module.num_workers=4 \
        data.data_module.num_workers_val=2
      rc=$?
      set -e
      end_ts=$(date -u +%FT%TZ)
      echo "END ${end_ts} stage=train layer=${layer} gpu=${gpu} rc=${rc}"
      write_status "${status}" train "${layer}" "${rc}" "${start_ts}" "${end_ts}"
      exit "${rc}"
    ) > "${log}" 2>&1 &
    echo -e "${layer}\t${gpu}\t$!\t${log}" >> "${REPORT_DIR}/logs/train_pids.txt"
  done
  cat "${REPORT_DIR}/logs/train_pids.txt"
}

case "${STAGE}" in
  target) launch_target ;;
  train) launch_train ;;
  *) echo "unknown stage: ${STAGE}" >&2; exit 2 ;;
esac
