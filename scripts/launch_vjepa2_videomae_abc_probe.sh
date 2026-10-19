#!/usr/bin/env bash
# Slot launcher for V-JEPA2 / VideoMAE multilayer ABC probing.
#
# Expects a prepared OUT_DIR from:
#   bash scripts/prepare_vjepa2_videomae_abc_probe.sh
#
# Default is DRY_RUN=1 (prints commands only). Set DRY_RUN=0 to train.
#
# Examples:
#   # Review only
#   RUN_ID=... OUT_DIR=logs/probing_jobs_... DRY_RUN=1 \
#     bash scripts/launch_vjepa2_videomae_abc_probe.sh
#
#   # Real launch on 8 GPUs, 2 slots/GPU
#   RUN_ID=... OUT_DIR=logs/probing_jobs_... DRY_RUN=0 \
#     GPU_IDS_CSV=0,1,2,3,4,5,6,7 SLOTS=16 \
#     bash scripts/launch_vjepa2_videomae_abc_probe.sh
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
PYTHON=${PYTHON:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}
DATASET_ROOT=${DATASET_ROOT:-/mnt/umm/users/yinbaiqiao/InsScene-15K}
RUN_ID=${RUN_ID:-}
OUT_DIR=${OUT_DIR:-}
DRY_RUN=${DRY_RUN:-1}
SLOTS=${SLOTS:-16}
GPU_IDS_CSV=${GPU_IDS_CSV:-0,1,2,3,4,5,6,7}
BATCH_SIZE=${BATCH_SIZE:-8}
BATCH_SIZE_VAL=${BATCH_SIZE_VAL:-8}
NUM_WORKERS=${NUM_WORKERS:-4}
NUM_WORKERS_VAL=${NUM_WORKERS_VAL:-2}
VIDEO_CHANNELS=${VIDEO_CHANNELS:-1024}
FEATURE_TIMESTEP=${FEATURE_TIMESTEP:-749}
ALLOW_MISSING_STREAMING=${ALLOW_MISSING_STREAMING:-0}
FEAT_BACKEND=${FEAT_BACKEND:-aoss}
AOSS_BUCKET_ROOT=${AOSS_BUCKET_ROOT:-aoss://baiqiao/InsScene-15K}

AOSS_HOST=${AOSS_HOST:-baiqiao.aoss-internal.cn-fz-01.fjscmsapi-oss.com}
export NO_PROXY="localhost,127.0.0.1,::1,${AOSS_HOST},.aoss-internal.cn-fz-01.fjscmsapi-oss.com,aoss-internal.cn-fz-01.fjscmsapi-oss.com"
export no_proxy="${NO_PROXY}"

if [[ -z "${OUT_DIR}" ]]; then
  if [[ -z "${RUN_ID}" ]]; then
    echo "[err] set OUT_DIR=... or RUN_ID=... (from prepare script)" >&2
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
  echo "      run: bash scripts/prepare_vjepa2_videomae_abc_probe.sh" >&2
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

# Avoid env roots shadowing Hydra overrides (see docs/c_retrain_session_report).
unset INSCENE_STREAMING_FEAT_ROOT || true
unset INSCENE_TARGET_FEAT_ROOT || true
export INSCENE_DATA_ROOT="${DATA_ROOT}"

LOG_ROOT="${PROJECT_ROOT}/logs/inscene15k_streaming/${RUN_ID}/launch"
mkdir -p "${LOG_ROOT}" "${OUT_DIR}/launcher"
IFS=',' read -r -a GPU_IDS <<< "${GPU_IDS_CSV}"

# Preflight: local streaming presence (AOSS existence is harder to assert here).
if [[ "${ALLOW_MISSING_STREAMING}" != "1" ]]; then
  for vfm in vjepa2 videomae; do
    if ! grep -q "^${vfm}"$'\t' "${JOBS_TSV}"; then
      continue
    fi
    st="${DATASET_ROOT}/FEAT_STREAMING/${vfm}/scannetpp"
    n=0
    [[ -d "${st}" ]] && n=$(find "${st}" -mindepth 1 -maxdepth 1 -type d | wc -l)
    if (( n < 1 )) && [[ "${FEAT_BACKEND}" == "local" ]]; then
      echo "[err] missing local streaming cache for ${vfm}: ${st}" >&2
      echo "      extract+upload FEAT_STREAMING first, or ALLOW_MISSING_STREAMING=1" >&2
      exit 2
    fi
    if (( n < 1 )) && [[ "${FEAT_BACKEND}" == "aoss" ]]; then
      echo "[err] local FEAT_STREAMING/${vfm} is empty; AOSS streaming for these VFMs" >&2
      echo "      was not uploaded with the FEAT_TARGET multilayer sync." >&2
      echo "      Extract+upload FEAT_STREAMING, or set ALLOW_MISSING_STREAMING=1 to force." >&2
      exit 2
    fi
  done
fi

needs_target() {
  case "$1" in
    action_dynamics|path_integration|counterfactual) return 0 ;;
    *) return 1 ;;
  esac
}

active_jobs() {
  jobs -pr | wc -l
}

run_one() {
  local slot="$1"
  local vfm="$2"
  local probe="$3"
  local prefix="$4"
  local layer="$5"
  local manifest="$6"
  local ckpt_path="${7:-}"
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

  local bs="${BATCH_SIZE}"
  local bs_val="${BATCH_SIZE_VAL}"
  if [[ "${probe}" == "streaming_mask" ]]; then
    bs=1
    bs_val=1
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
    "video_channels=${VIDEO_CHANNELS}"
    "streaming_feat_root=${STREAM_ROOT}"
    "feature_layer=${layer}"
    "feat_postfix=_layer${layer}"
    "feature_timestep=${FEATURE_TIMESTEP}"
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
  # Optional 7th TSV column: absolute path to a Lightning ckpt to resume.
  # Quote for Hydra: filenames often contain '=' (e.g. epoch=39-step=12200.ckpt).
  if [[ -n "${ckpt_path}" && "${ckpt_path}" != "-" && "${ckpt_path}" != "null" ]]; then
    if [[ ! -f "${ckpt_path}" ]]; then
      echo "[err] ckpt_path missing for ${tag}: ${ckpt_path}" >&2
      return 1
    fi
    cmd+=("ckpt_path='${ckpt_path}'")
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
    echo "BEGIN $(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) gpu=${gpu} vfm=${vfm} probe=${probe} prefix=${prefix} layer=${layer}"
    echo "STREAM_ROOT=${STREAM_ROOT}"
    echo "TARGET_ROOT=${TARGET_ROOT}"
    echo "MANIFEST=${manifest}"
    if [[ -n "${ckpt_path}" && "${ckpt_path}" != "-" && "${ckpt_path}" != "null" ]]; then
      echo "CKPT_PATH=${ckpt_path}"
    fi
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
launched=0
# TSV columns: vfm probe prefix layer manifest cfg [ckpt_path]
while IFS=$'\t' read -r vfm probe prefix layer manifest cfg ckpt_path _rest || [[ -n "${vfm:-}" ]]; do
  [[ -z "${vfm:-}" || "${vfm}" == \#* ]] && continue
  if [[ "${DRY_RUN}" != "1" ]]; then
    while (( $(active_jobs) >= SLOTS )); do
      sleep 20
    done
  fi
  run_one "${slot}" "${vfm}" "${probe}" "${prefix}" "${layer}" "${manifest}" "${ckpt_path:-}"
  slot=$((slot + 1))
  launched=$((launched + 1))
  if [[ "${DRY_RUN}" == "1" && "${launched}" -ge 8 ]]; then
    echo "[dry-run] showing first 8 jobs only; total=$(wc -l < "${JOBS_TSV}")"
    break
  fi
  if [[ "${DRY_RUN}" != "1" ]]; then
    sleep 2
  fi
done < "${JOBS_TSV}"

if [[ "${DRY_RUN}" != "1" ]]; then
  wait
  echo "All slots drained. Logs: ${LOG_ROOT}"
else
  echo "Dry-run done. Set DRY_RUN=0 to launch for real."
fi
