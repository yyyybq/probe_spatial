#!/usr/bin/env bash
# Slot launcher for multi-VFM probe fill jobs (TSV from build_probe_fill_jobs.py).
#
# TSV columns:
#   vfm probe prefix layer disk_layer video_channels feature_timestep batch_size manifest cfg
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
PYTHON=${PYTHON:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}
DATASET_ROOT=${DATASET_ROOT:-/mnt/umm/users/yinbaiqiao/InsScene-15K}
RUN_ID=${RUN_ID:-}
OUT_DIR=${OUT_DIR:-}
DRY_RUN=${DRY_RUN:-1}
SLOTS=${SLOTS:-16}
GPU_IDS_CSV=${GPU_IDS_CSV:-0,1,2,3,4,5,6,7}
NUM_WORKERS=${NUM_WORKERS:-4}
NUM_WORKERS_VAL=${NUM_WORKERS_VAL:-2}
ALLOW_MISSING_STREAMING=${ALLOW_MISSING_STREAMING:-1}
FEAT_BACKEND=${FEAT_BACKEND:-aoss}
AOSS_BUCKET_ROOT=${AOSS_BUCKET_ROOT:-aoss://baiqiao/InsScene-15K}

AOSS_HOST=${AOSS_HOST:-baiqiao.aoss-internal.cn-fz-01.fjscmsapi-oss.com}
export NO_PROXY="localhost,127.0.0.1,::1,${AOSS_HOST},.aoss-internal.cn-fz-01.fjscmsapi-oss.com,aoss-internal.cn-fz-01.fjscmsapi-oss.com"
export no_proxy="${NO_PROXY}"

if [[ -z "${OUT_DIR}" ]]; then
  if [[ -z "${RUN_ID}" ]]; then
    echo "[err] set OUT_DIR=... or RUN_ID=..." >&2
    exit 1
  fi
  OUT_DIR="${PROJECT_ROOT}/logs/probing_jobs_${RUN_ID}"
fi
if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="$(basename "${OUT_DIR}" | sed 's/^probing_jobs_//')"
fi

JOBS_TSV=${JOBS_TSV:-${OUT_DIR}/jobs/all.tsv}
if [[ ! -s "${JOBS_TSV}" ]]; then
  echo "[err] missing job list: ${JOBS_TSV}" >&2
  exit 1
fi

case "${FEAT_BACKEND}" in
  aoss)
    DATA_ROOT=${DATA_ROOT:-${AOSS_BUCKET_ROOT}/data}
    STREAM_ROOT=${STREAM_ROOT:-${AOSS_BUCKET_ROOT}/FEAT_STREAMING}
    TARGET_ROOT=${TARGET_ROOT:-${AOSS_BUCKET_ROOT}/FEAT_TARGET}
    ;;
  local)
    DATA_ROOT=${DATA_ROOT:-${DATASET_ROOT}/data}
    STREAM_ROOT=${STREAM_ROOT:-${DATASET_ROOT}/FEAT_STREAMING}
    TARGET_ROOT=${TARGET_ROOT:-${DATASET_ROOT}/FEAT_TARGET}
    ;;
  *)
    echo "[err] FEAT_BACKEND must be aoss|local" >&2
    exit 1
    ;;
esac

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}}"
mkdir -p "${MPLCONFIGDIR}"
unset INSCENE_STREAMING_FEAT_ROOT || true
unset INSCENE_TARGET_FEAT_ROOT || true
export INSCENE_DATA_ROOT="${DATA_ROOT}"

LOG_ROOT="${PROJECT_ROOT}/logs/inscene15k_streaming/${RUN_ID}/launch"
mkdir -p "${LOG_ROOT}" "${OUT_DIR}/launcher"
IFS=',' read -r -a GPU_IDS <<< "${GPU_IDS_CSV}"

needs_target() {
  case "$1" in
    action_dynamics|path_integration|counterfactual) return 0 ;;
    *) return 1 ;;
  esac
}

feat_postfix_for() {
  local vfm="$1" disk_layer="$2" t="$3"
  case "${vfm}" in
    vjepa2|videomae) printf '_layer%s' "${disk_layer}" ;;
    *) printf '_t%s_layer%s' "${t}" "${disk_layer}" ;;
  esac
}

active_jobs() { jobs -pr | wc -l; }

run_one() {
  local slot="$1" vfm="$2" probe="$3" prefix="$4" layer="$5"
  local disk_layer="$6" ch="$7" t="$8" bs_default="$9" manifest="${10}"
  local gpu="${GPU_IDS[$((slot % ${#GPU_IDS[@]}))]}"
  local tag="${probe}_${vfm}_p${prefix}_layer${layer}"
  local full_tag="${RUN_ID}_${tag}"
  local log="${LOG_ROOT}/${full_tag}.log"
  local run_folder="${RUN_ID}/${full_tag}"
  local cache_root="${AOSS_CACHE_ROOT:-/mnt/local_cache/inscene/aoss_cache}"
  if [[ ! -d "$(dirname "${cache_root}")" && ! -d "${cache_root}" ]]; then
    cache_root="/tmp/inscene_aoss_cache"
  fi
  local cache_dir="${cache_root}/${RUN_ID}/${full_tag}"
  local postfix
  postfix="$(feat_postfix_for "${vfm}" "${disk_layer}" "${t}")"
  local bs="${bs_default}"
  local bs_val="${bs_default}"
  if [[ "${probe}" == "streaming_depth" || "${probe}" == "streaming_mask" ]]; then
    if [[ -z "${bs}" || "${bs}" == "0" ]]; then
      bs=1
      bs_val=1
    fi
  fi

  local cmd=(
    env
    CUDA_VISIBLE_DEVICES="${gpu}"
    INSCENE_DATA_ROOT="${DATA_ROOT}"
    INSCENE_MANIFEST="${manifest}"
    AOSS_CACHE_DIR="${cache_dir}"
    AOSS_API_HOST="${AOSS_API_HOST:-aoss-internal.cn-fz-01.fjscmsapi-oss.com}"
    NO_PROXY="${NO_PROXY}"
    no_proxy="${no_proxy}"
    "${PYTHON}" vidfm3d/train.py
    "experiment=inscene15k_streaming/${probe}_${vfm}_v1"
    "vfm_name=${vfm}"
    "video_channels=${ch}"
    "streaming_feat_root=${STREAM_ROOT}"
    "feature_layer=${disk_layer}"
    "feat_postfix=${postfix}"
    "feature_timestep=${t}"
    "prefix_len=${prefix}"
    "job_name=${tag}"
    "paths.run_folder_name=${run_folder}"
    "logger.wandb.name=${full_tag}"
    "logger.wandb.group=${RUN_ID}"
    "logger.wandb.offline=true"
    "autoresume=false"
    "trainer.devices=[0]"
    "data.data_module.batch_size_per_device=${bs}"
    "data.data_module.batch_size_per_device_val=${bs_val}"
    "data.data_module.num_workers=${NUM_WORKERS}"
    "data.data_module.num_workers_val=${NUM_WORKERS_VAL}"
  )
  if needs_target "${probe}"; then
    cmd+=("target_feat_root=${TARGET_ROOT}")
  fi
  if [[ "${probe}" == "ego_belief_v2" || "${probe}" == "ego_belief" ]]; then
    cmd+=("streaming_hidden_prefix_lengths=[${prefix}]")
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[dry-run] gpu=%s ' "${gpu}"
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  mkdir -p "${cache_dir}"
  (
    set +e
    echo "BEGIN $(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) gpu=${gpu} vfm=${vfm} probe=${probe} prefix=${prefix} layer=${layer} disk_layer=${disk_layer}"
    echo "STREAM_ROOT=${STREAM_ROOT}"
    echo "TARGET_ROOT=${TARGET_ROOT}"
    echo "MANIFEST=${manifest}"
    "${cmd[@]}"
    rc=$?
    echo "END $(date -u +%Y-%m-%dT%H:%M:%SZ) rc=${rc}"
    exit "${rc}"
  ) > "${log}" 2>&1 &
}

echo "OUT_DIR=${OUT_DIR}"
echo "RUN_ID=${RUN_ID}"
echo "JOBS=$(wc -l < "${JOBS_TSV}")"
echo "DRY_RUN=${DRY_RUN} SLOTS=${SLOTS} GPUS=${GPU_IDS_CSV}"
echo "STREAM_ROOT=${STREAM_ROOT}"
echo "TARGET_ROOT=${TARGET_ROOT}"
echo "LOG_ROOT=${LOG_ROOT}"

slot=0
{
  read -r _header || true
  while IFS=$'\t' read -r vfm probe prefix layer disk_layer ch t bs_default manifest cfg; do
    [[ -z "${vfm}" || "${vfm}" == \#* ]] && continue
    while (( $(active_jobs) >= SLOTS )); do
      sleep 5
    done
    run_one "${slot}" "${vfm}" "${probe}" "${prefix}" "${layer}" "${disk_layer}" "${ch}" "${t}" "${bs_default}" "${manifest}"
    slot=$((slot + 1))
  done
} < "${JOBS_TSV}"

wait || true
echo "launcher_finished_at=$(date -Is) slots_launched=${slot}"
