#!/usr/bin/env bash
# Run exactly one timestep/mode on 8 GPUs plus one CPU AOSS sync loop.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

require_min_bytes() {
  local path="$1" minimum="$2" size=0
  [[ -f "${path}" ]] && size="$(wc -c < "${path}")"
  if [[ "${size}" -lt "${minimum}" ]]; then
    echo "ERROR: required source looks truncated: ${path} (${size} < ${minimum} bytes)" >&2
    exit 1
  fi
}
require_min_bytes "features/run_inscene15k.py" 50000
require_min_bytes "features/minimax_h3/minimax_h3_feature.py" 10000
require_min_bytes "features/minimax_h3/extract_features.py" 1000
require_min_bytes "features/lingbot/lingbot_feature.py" 10000
require_min_bytes "features/lingbot/extract_features.py" 1000
require_min_bytes "scripts/run_tstep_sync_loop.sh" 5000
require_min_bytes "scripts/delete_aoss_confirmed_local_pairs.sh" 1000

VFM="${VFM:?set VFM=wan|cogvideox|minimax_h3}"
T="${T:?set T=<timestep>}"
LAYERS="${LAYERS:-15 20 25}"
LAYERS="${LAYERS//,/ }"
LAYERS="${LAYERS//-/ }"
EXTRACT_STREAM="${EXTRACT_STREAM:-1}"
EXTRACT_TARGET="${EXTRACT_TARGET:-0}"
if [[ $((EXTRACT_STREAM + EXTRACT_TARGET)) -ne 1 ]]; then
  echo "ERROR: select exactly one mode: EXTRACT_STREAM=${EXTRACT_STREAM} EXTRACT_TARGET=${EXTRACT_TARGET}" >&2
  exit 2
fi

PREFIX_LENGTHS="${PREFIX_LENGTHS:-8,12,16,24}"
STREAMING_WINDOW_OBS_LEN="${STREAMING_WINDOW_OBS_LEN:-28}"
CANONICAL_PROTOCOL="${CANONICAL_PROTOCOL:-}"
CANONICAL_PROTOCOL_SHA256="${CANONICAL_PROTOCOL_SHA256:-}"
REQUIRE_H800="${REQUIRE_H800:-0}"
SOURCE="${SOURCE:-scannetpp}"
N_GPUS="${N_GPUS:-8}"
SCENE_START="${SCENE_START:-0}"
SCENE_END="${SCENE_END:-}"
if [[ ! "${N_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: N_GPUS must be a positive integer" >&2
  exit 2
fi

# CUDA_VISIBLE_DEVICES may be an explicit physical-device allocation (for
# example "3,4,5,6,7").  Preserve that allocation when spawning workers:
# each worker must receive the corresponding physical id, not its logical
# loop index.  Without this mapping a one-GPU canary allocated to GPU 3 would
# accidentally reset itself to physical GPU 0.
REQUESTED_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
if [[ -n "${REQUESTED_CUDA_VISIBLE_DEVICES}" ]]; then
  IFS=',' read -r -a GPU_DEVICE_IDS <<< "${REQUESTED_CUDA_VISIBLE_DEVICES}"
  if [[ "${#GPU_DEVICE_IDS[@]}" -ne "${N_GPUS}" ]]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES (${REQUESTED_CUDA_VISIBLE_DEVICES}) must list exactly N_GPUS=${N_GPUS} devices" >&2
    exit 2
  fi
  for device_id in "${GPU_DEVICE_IDS[@]}"; do
    if [[ -z "${device_id}" ]]; then
      echo "ERROR: CUDA_VISIBLE_DEVICES contains an empty device id" >&2
      exit 2
    fi
  done
else
  GPU_DEVICE_IDS=()
  for gpu in $(seq 0 $((N_GPUS - 1))); do
    GPU_DEVICE_IDS+=("${gpu}")
  done
fi
DATASET_ROOT="${DATASET_ROOT:-/mnt/umm/users/yinbaiqiao/InsScene-15K}"
DATA_ROOT="${DATA_ROOT:-${DATASET_ROOT}/data}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/feat_extract_tstep_resume4_${VFM}_t${T}_$(date -u +%Y%m%d_%H%M%S)}"

if [[ "${EXTRACT_STREAM}" == "1" ]]; then
  FEATURE_KIND="FEAT_STREAMING"
  MODE="streaming_prefix"
else
  FEATURE_KIND="FEAT_TARGET"
  MODE="target_isolated"
fi
FEATURE_NAMESPACE="${FEATURE_NAMESPACE:-${FEATURE_KIND}}"
case "${FEATURE_KIND}:${FEATURE_NAMESPACE}" in
  FEAT_STREAMING:FEAT_STREAMING|\
  FEAT_STREAMING:FEAT_STREAMING_CANONICAL_V2|\
  FEAT_STREAMING:FEAT_STREAMING_CANONICAL_V2_SEEDED_V1|\
  FEAT_STREAMING:FEAT_STREAMING_CANONICAL_V2_SEEDED_V1_MINIMAX_BF16_V1|\
  FEAT_TARGET:FEAT_TARGET|\
  FEAT_TARGET:FEAT_TARGET_CANONICAL_V2|\
  FEAT_TARGET:FEAT_TARGET_CANONICAL_V2_SEEDED_V1|\
  FEAT_TARGET:FEAT_TARGET_CANONICAL_V2_SEEDED_V1_MINIMAX_BF16_V1) ;;
  *)
    echo "ERROR: namespace ${FEATURE_NAMESPACE} is incompatible with ${FEATURE_KIND}" >&2
    exit 2
    ;;
esac
export FEATURE_NAMESPACE
# Seeded-v1 contract namespaces: the 2026-09 fp16 originals and the MiniMax-only
# bf16 re-extraction (2026-10, storage contract MINIMAX_H3_FINITE_BF16_V1).
# Both share the frozen protocol/model/prompt/done-index gates below.
SEEDED_NAMESPACE=0
case "${FEATURE_NAMESPACE}" in
  *_CANONICAL_V2_SEEDED_V1|*_CANONICAL_V2_SEEDED_V1_MINIMAX_BF16_V1) SEEDED_NAMESPACE=1 ;;
esac
if [[ "${FEATURE_NAMESPACE}" == *_MINIMAX_BF16_V1 && "${VFM}" != "minimax_h3" ]]; then
  echo "ERROR: namespace ${FEATURE_NAMESPACE} is reserved for VFM=minimax_h3 (got ${VFM})" >&2
  exit 2
fi
if [[ "${FEATURE_NAMESPACE}" == *CANONICAL_V2* && "${STREAMING_WINDOW_OBS_LEN}" -ne 28 ]]; then
  echo "ERROR: canonical v2 namespace requires STREAMING_WINDOW_OBS_LEN=28" >&2
  exit 2
fi
case "${REQUIRE_H800}" in
  0|1) ;;
  *) echo "ERROR: REQUIRE_H800 must be 0 or 1" >&2; exit 2 ;;
esac
if [[ "${SEEDED_NAMESPACE}" == "1" ]]; then
  [[ "${CANONICAL_PROTOCOL}" == "inscene_streaming_canonical_v2_obs28_h124" ]] || {
    echo "ERROR: seeded-v1 namespace requires the canonical obs28/h124 protocol" >&2
    exit 2
  }
  [[ "${CANONICAL_PROTOCOL_SHA256}" == "215c720c8537f4dfae6243bedcc7339fbb7e3ba8a6ca2dcaf04f49cd4b738e4f" ]] || {
    echo "ERROR: seeded-v1 namespace requires the exact production canonical protocol SHA256" >&2
    exit 2
  }
fi
if [[ "${EXTRACT_STREAM}" == "1" ]]; then
  OUT_ROOT="${OUT_STREAM:-${DATASET_ROOT}/${FEATURE_NAMESPACE}}"
else
  OUT_ROOT="${OUT_TARGET:-${DATASET_ROOT}/${FEATURE_NAMESPACE}}"
fi
if [[ -z "${EXTRACT_DONE_INDEX:-}" ]]; then
  if [[ "${FEATURE_NAMESPACE}" != "${FEATURE_KIND}" ]]; then
    echo "ERROR: canonical FEATURE_NAMESPACE requires an explicit EXTRACT_DONE_INDEX" >&2
    exit 2
  fi
  EXTRACT_DONE_INDEX="${PROJECT_ROOT}/logs/tstep_resume4/aoss_done/${FEATURE_KIND}_${VFM}_t${T}.txt"
fi
export EXTRACT_DONE_INDEX
if [[ "${SEEDED_NAMESPACE}" == "1" ]]; then
  expected_done_name="${FEATURE_NAMESPACE}_${VFM}_t${T}.txt"
  [[ "$(basename -- "${EXTRACT_DONE_INDEX}")" == "${expected_done_name}" ]] || {
    echo "ERROR: seeded-v1 done-index must be named ${expected_done_name}: ${EXTRACT_DONE_INDEX}" >&2
    exit 2
  }
  [[ "$(basename -- "${OUT_ROOT%/}")" == "${FEATURE_NAMESPACE}" ]] || {
    echo "ERROR: seeded-v1 output root must end in ${FEATURE_NAMESPACE}: ${OUT_ROOT}" >&2
    exit 2
  }
fi

PY_PROBE="/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python"
PY_MINIMAX="/mnt/umm/users/yinbaiqiao/.conda/envs/minimax-h3/bin/python"
WAN_SNAPSHOT="/mnt/umm/users/yinbaiqiao/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780a534b6463e45facd96134c9f345acfa5b"
COGX_SNAPSHOT="/mnt/umm/users/yinbaiqiao/.cache/huggingface/hub/models--THUDM--CogVideoX-5b-I2V/snapshots/a6f0f4858a8395e7429d82493864ce92bf73af11"

case "${VFM}" in
  wan)
    PYTHON="${PYTHON:-${PY_PROBE}}"
    MODEL_ID="${MODEL_ID:-${WAN_SNAPSHOT}}"
    PROMPT="${PROMPT:-.}"
    export HF_HOME="${HF_HOME:-/mnt/umm/users/yinbaiqiao/.cache/huggingface}"
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
    ;;
  cogvideox)
    PYTHON="${PYTHON:-${PY_PROBE}}"
    MODEL_ID="${MODEL_ID:-${COGX_SNAPSHOT}}"
    PROMPT="${PROMPT:-}"
    export HF_HOME="${HF_HOME:-/mnt/umm/users/yinbaiqiao/.cache/huggingface}"
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
    ;;
  minimax_h3)
    PYTHON="${PYTHON:-${PY_MINIMAX}}"
    MODEL_ID="${MODEL_ID:-/mnt/umm/users/yinbaiqiao/Consistency/minimax-h3/models/MiniMax-H3}"
    PROMPT="${PROMPT:-.}"
    export HF_HOME="${HF_HOME:-/mnt/umm/shared_model/huggingface}"
    ;;
  *) echo "ERROR: unsupported VFM=${VFM}" >&2; exit 2 ;;
esac

if [[ "${SEEDED_NAMESPACE}" == "1" ]]; then
  case "${VFM}" in
    wan)
      EXPECTED_MODEL_ID="${WAN_SNAPSHOT}"
      EXPECTED_PROMPT="."
      ;;
    cogvideox)
      EXPECTED_MODEL_ID="${COGX_SNAPSHOT}"
      EXPECTED_PROMPT=""
      ;;
    minimax_h3)
      EXPECTED_MODEL_ID="/mnt/umm/users/yinbaiqiao/Consistency/minimax-h3/models/MiniMax-H3"
      EXPECTED_PROMPT="."
      ;;
  esac
  [[ "${MODEL_ID}" == "${EXPECTED_MODEL_ID}" ]] || {
    echo "ERROR: seeded-v1 ${VFM} MODEL_ID is frozen to ${EXPECTED_MODEL_ID}" >&2
    exit 2
  }
  [[ "${PROMPT}" == "${EXPECTED_PROMPT}" ]] || {
    echo "ERROR: seeded-v1 ${VFM} PROMPT differs from the frozen contract" >&2
    exit 2
  }
fi

[[ -x "${PYTHON}" ]] || { echo "ERROR: Python not executable: ${PYTHON}" >&2; exit 1; }
[[ -d "${MODEL_ID}" ]] || {
  echo "ERROR: frozen local model path is missing: ${MODEL_ID}" >&2
  exit 1
}
[[ -f "${EXTRACT_DONE_INDEX}" ]] || {
  echo "ERROR: completion index missing: ${EXTRACT_DONE_INDEX}" >&2
  exit 1
}
STEP_OWNER_LOCK="${STEP_OWNER_LOCK:-${PROJECT_ROOT}/logs/tstep_canonical_v2_seeded_v1/step_locks/${FEATURE_NAMESPACE}_${VFM}_t${T}.lock}"
if [[ -f "${STEP_OWNER_LOCK}" && ! -L "${STEP_OWNER_LOCK}" ]]; then
  step_owner="$(tr -d '\n' < "${STEP_OWNER_LOCK}" || true)"
  if [[ -n "${step_owner}" && "${step_owner}" != "${LOG_DIR}" ]]; then
    mkdir -p "${LOG_DIR}"
    echo "SKIP: ${VFM} t${T} ${FEATURE_NAMESPACE} is owned by ${step_owner}; this LOG_DIR=${LOG_DIR}"
    printf 'skipped_owned_by=%s\nthis_log_dir=%s\n' "${step_owner}" "${LOG_DIR}" > "${LOG_DIR}/skipped_other_owner.txt"
    exit 0
  fi
fi

export PATH="$(dirname "${PYTHON}"):${PATH}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
mkdir -p "${MPLCONFIGDIR}" "${OUT_ROOT}" "${LOG_DIR}"

if [[ -n "${PRE_EXTRACT_CLEANUP_MANIFEST:-}" ]]; then
  FEATURE_ROOT="${OUT_ROOT}/${VFM}/${SOURCE}" \
  T="${T}" \
  LAYERS="${LAYERS}" \
  EXTRACT_DONE_INDEX="${EXTRACT_DONE_INDEX}" \
  PRE_EXTRACT_CLEANUP_MANIFEST="${PRE_EXTRACT_CLEANUP_MANIFEST}" \
  CLEANUP_EXPECTED_COUNT="${CLEANUP_EXPECTED_COUNT:-0}" \
    bash "${PROJECT_ROOT}/scripts/delete_aoss_confirmed_local_pairs.sh" \
    | tee "${LOG_DIR}/pre_extract_cleanup.log"
fi

{
  echo "=============================================="
  echo " timestep feature extract 8-GPU + sync"
  echo " vfm        : ${VFM}"
  echo " python     : ${PYTHON}"
  echo " model      : ${MODEL_ID}"
  echo " t          : ${T}"
  echo " layers     : ${LAYERS}"
  echo " mode       : ${MODE}"
  echo " namespace  : ${FEATURE_NAMESPACE}"
  echo " protocol   : ${CANONICAL_PROTOCOL:-<legacy/non-canonical>}"
  echo " protocol sha256: ${CANONICAL_PROTOCOL_SHA256:-<legacy/non-canonical>}"
  echo " window obs : ${STREAMING_WINDOW_OBS_LEN}"
  echo " data_root  : ${DATA_ROOT}"
  echo " out_root   : ${OUT_ROOT}"
  echo " done_index : ${EXTRACT_DONE_INDEX} ($(wc -l < "${EXTRACT_DONE_INDEX}") lines)"
  echo " n_gpus     : ${N_GPUS}"
  echo " gpu devices: ${GPU_DEVICE_IDS[*]}"
  echo " scene span : ${SCENE_START}:${SCENE_END:-<all>}"
  echo " log_dir    : ${LOG_DIR}"
  echo " sync       : every ${SYNC_EVERY_SEC:-180}s or ${SYNC_EVERY_GIB:-80} GiB"
  echo "=============================================="
  "${PYTHON}" -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.device_count())'
} | tee "${LOG_DIR}/supervisor.log"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: configuration validated" | tee -a "${LOG_DIR}/supervisor.log"
  exit 0
fi

if [[ "${REQUIRE_H800}" == "1" ]]; then
  "${PYTHON}" - "${N_GPUS}" <<'PY' | tee -a "${LOG_DIR}/supervisor.log"
import sys
import torch

expected = int(sys.argv[1])
actual = torch.cuda.device_count()
if actual != expected:
    raise SystemExit(
        f"ERROR: H800 gate expected {expected} visible CUDA devices, found {actual}"
    )
names = [torch.cuda.get_device_name(index) for index in range(actual)]
unexpected = [name for name in names if "H800" not in name]
if unexpected:
    raise SystemExit(
        "ERROR: seeded-v1 extraction is H800-only; found " + ", ".join(unexpected)
    )
print(f"H800 gate PASS: {actual} visible CUDA device(s): {names}")
PY
fi

N_SCENES="$(
  "${PYTHON}" - <<PY
import os, sys
sys.path.insert(0, "${PROJECT_ROOT}")
from features.run_inscene15k import collect_scannetpp_scenes
root = os.path.join("${DATA_ROOT}", "processed_scannetpp_v2")
print(len(collect_scannetpp_scenes(root)))
PY
)"
if [[ "${N_SCENES}" -ne 682 ]]; then
  echo "ERROR: expected 682 ScanNet++ scenes, found ${N_SCENES}" >&2
  exit 1
fi
echo "Total scannetpp scenes: ${N_SCENES}" | tee -a "${LOG_DIR}/supervisor.log"
SCENE_END="${SCENE_END:-${N_SCENES}}"
if [[ ! "${SCENE_START}" =~ ^[0-9]+$ || ! "${SCENE_END}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: SCENE_START/SCENE_END must be non-negative integers" >&2
  exit 2
fi
if (( SCENE_START < 0 || SCENE_END <= SCENE_START || SCENE_END > N_SCENES )); then
  echo "ERROR: invalid scene span [${SCENE_START}:${SCENE_END}) for ${N_SCENES} scenes" >&2
  exit 2
fi
SCENE_COUNT=$((SCENE_END - SCENE_START))
echo "Selected scene span: [${SCENE_START}:${SCENE_END}) count=${SCENE_COUNT}" \
  | tee -a "${LOG_DIR}/supervisor.log"

read -r -a LAYER_ARGS <<<"${LAYERS}"
common_args=(
  -m features.run_inscene15k
  --vfm "${VFM}"
  --model-id "${MODEL_ID}"
  --data-root "${DATA_ROOT}"
  --source "${SOURCE}"
  --t "${T}"
  --output-layers "${LAYER_ARGS[@]}"
  --prefix-lengths "${PREFIX_LENGTHS}"
  --prefix-max-len 24
  --streaming-window-obs-len "${STREAMING_WINDOW_OBS_LEN}"
  --streaming-motion-step 0.35
  --streaming-rotation-weight 0.5
)
if [[ -n "${PROMPT}" ]]; then
  common_args+=(--prompt "${PROMPT}")
fi

STOP_FILE="${LOG_DIR}/extract.finished"
rm -f -- "${STOP_FILE}"
FEATURE_KIND="${FEATURE_KIND}" \
FEATURE_NAMESPACE="${FEATURE_NAMESPACE}" \
CANONICAL_PROTOCOL_SHA256="${CANONICAL_PROTOCOL_SHA256}" \
FEATURE_ROOT="${OUT_ROOT}/${VFM}/${SOURCE}" \
FEATURE_SCAN_ROOT="${SYNC_FEATURE_SCAN_ROOT:-${OUT_ROOT}/${VFM}/${SOURCE}}" \
VFM="${VFM}" \
T="${T}" \
LAYERS="${LAYERS}" \
DATASET_ROOT="${DATASET_ROOT}" \
SOURCE="${SOURCE}" \
EXTRACT_DONE_INDEX="${EXTRACT_DONE_INDEX}" \
LOG_DIR="${LOG_DIR}" \
STOP_FILE="${STOP_FILE}" \
SYNC_EVERY_SEC="${SYNC_EVERY_SEC:-180}" \
SYNC_EVERY_GIB="${SYNC_EVERY_GIB:-80}" \
SYNC_TRUST_EXTRACT="${SYNC_TRUST_EXTRACT:-1}" \
bash "${PROJECT_ROOT}/scripts/run_tstep_sync_loop.sh" \
  >"${LOG_DIR}/sync_loop.log" 2>&1 &
sync_pid=$!
echo "${sync_pid}" > "${LOG_DIR}/sync.pid"
sleep 2
if ! kill -0 "${sync_pid}" 2>/dev/null; then
  wait "${sync_pid}" || true
  echo "ERROR: sync loop failed during startup; refusing to launch GPUs" >&2
  exit 1
fi

pids=()
for gpu in $(seq 0 $((N_GPUS - 1))); do
  start=$((SCENE_START + SCENE_COUNT * gpu / N_GPUS))
  end=$((SCENE_START + SCENE_COUNT * (gpu + 1) / N_GPUS))
  log="${LOG_DIR}/gpu${gpu}_${start}_${end}.log"
  echo "[gpu ${gpu}] scenes [${start}:${end}) -> ${log}" | tee -a "${LOG_DIR}/supervisor.log"
  (
    set -euo pipefail
    physical_gpu="${GPU_DEVICE_IDS[$gpu]}"
    export CUDA_VISIBLE_DEVICES="${physical_gpu}"
    echo "[$(date -Is)] vfm=${VFM} t=${T} mode=${MODE} gpu=${gpu} physical_gpu=${physical_gpu} scenes=[${start}:${end})"
    mode_args=(--mode "${MODE}" --out-root "${OUT_ROOT}")
    if [[ "${MODE}" == "target_isolated" ]]; then
      mode_args+=(
        --target-from-streaming-windows
        --target-prefix-lengths "${PREFIX_LENGTHS}"
        --target-horizons "1,2,4"
      )
    fi
    "${PYTHON}" "${common_args[@]}" "${mode_args[@]}" \
      --start "${start}" --end "${end}"
    echo "[$(date -Is)] vfm=${VFM} t=${T} mode=${MODE} gpu=${gpu} DONE"
  ) >"${log}" 2>&1 &
  worker_pid=$!
  pids+=("${worker_pid}")
  echo "${worker_pid}" > "${LOG_DIR}/gpu${gpu}.pid"
done

SYNC_DIED_FILE="${LOG_DIR}/sync_died_early"
rm -f -- "${SYNC_DIED_FILE}"
(
  while kill -0 "${sync_pid}" 2>/dev/null; do
    sleep 15
  done
  if [[ ! -e "${STOP_FILE}" ]]; then
    touch "${SYNC_DIED_FILE}"
    echo "[$(date -Is)] sync loop died before extraction completed" >> "${LOG_DIR}/supervisor.log"
    for pid_file in "${LOG_DIR}"/gpu*.pid; do
      [[ -f "${pid_file}" ]] || continue
      worker_pid="$(cat "${pid_file}")"
      pkill -TERM -P "${worker_pid}" 2>/dev/null || true
      kill "${worker_pid}" 2>/dev/null || true
    done
  fi
) &
watchdog_pid=$!

extract_fail=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    extract_fail=1
  fi
done
touch "${STOP_FILE}"

sync_fail=0
if ! wait "${sync_pid}"; then
  sync_fail=1
fi
wait "${watchdog_pid}" || true
[[ -e "${SYNC_DIED_FILE}" ]] && sync_fail=1

local_left=0
LOCAL_ACCEPTANCE_ROOT="${SYNC_FEATURE_SCAN_ROOT:-${OUT_ROOT}/${VFM}/${SOURCE}}"
if [[ -d "${LOCAL_ACCEPTANCE_ROOT}" ]]; then
  local_left="$(
    find "${LOCAL_ACCEPTANCE_ROOT}" -type f -regextype posix-extended \
      -regex ".*/feature_t${T}_layer(15|20|25)\\.sft" | wc -l
  )"
fi
{
  echo "finished_at=$(date -Is) extract_fail=${extract_fail} sync_fail=${sync_fail}"
  echo "done_index_lines=$(wc -l < "${EXTRACT_DONE_INDEX}")"
  echo "local_matching_sft_left=${local_left}"
} | tee -a "${LOG_DIR}/supervisor.log"

if [[ "${extract_fail}" -ne 0 || "${sync_fail}" -ne 0 || "${local_left}" -ne 0 ]]; then
  echo "ERROR: extraction/sync acceptance failed; see ${LOG_DIR}" >&2
  exit 1
fi
echo "OK" | tee -a "${LOG_DIR}/supervisor.log"
