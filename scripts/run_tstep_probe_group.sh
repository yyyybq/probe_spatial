#!/usr/bin/env bash
# Train one (VFM, timestep, layer, prefix) group: one probe per GPU.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PROJECT_ROOT
cd "${PROJECT_ROOT}"

export PATH="/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON=${PYTHON:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}

GROUPS_TSV=${GROUPS_TSV:?set GROUPS_TSV}
GROUP_ID=${GROUP_ID:-${1:-}}
RUN_ID=${RUN_ID:?set RUN_ID}
[[ -n "${GROUP_ID}" ]] || { echo "[err] set GROUP_ID" >&2; exit 2; }

row="$(${PYTHON} - "${GROUPS_TSV}" "${GROUP_ID}" <<'PY'
import csv, sys
path, wanted = sys.argv[1:]
with open(path, newline="") as handle:
    rows = [row for row in csv.DictReader(handle, delimiter="\t") if row["group_id"] == wanted]
if len(rows) != 1:
    raise SystemExit(f"expected exactly one row for {wanted!r}, got {len(rows)}")
row = rows[0]
keys = [
    "status", "vfm", "source", "timestep", "layer", "prefix", "manifest", "tasks",
    "canonical_protocol", "canonical_obs_len", "streaming_namespace", "target_namespace",
    "strict_audit", "strict_audit_sha256", "strict_audit_ok",
]
missing = [key for key in keys if key not in row]
if missing:
    raise SystemExit(f"groups.tsv lacks strict-gate columns: {missing}")
print("\t".join(row[key] for key in keys))
PY
)"
IFS=$'\t' read -r status vfm source timestep layer prefix manifest tasks_csv canonical_protocol canonical_obs_len streaming_namespace target_namespace strict_audit_from_row strict_audit_sha256 strict_audit_ok <<< "${row}"
if [[ "${status}" == "wait_streaming" || -z "${tasks_csv}" ]]; then
  echo "[err] group is not trainable: ${GROUP_ID} status=${status}" >&2
  exit 3
fi
if [[ "${status}" != "ready_full" && "${ALLOW_STREAMING_ONLY:-0}" != "1" ]]; then
  echo "[err] group is ${status}; set ALLOW_STREAMING_ONLY=1 to run only A/B tasks" >&2
  exit 3
fi
if [[ "${strict_audit_ok}" != "true" ]]; then
  echo "[err] groups.tsv does not record a passing strict canonical audit" >&2
  exit 3
fi
if [[ "${canonical_protocol}" != "inscene_streaming_canonical_v2_obs28_h124" ||
      "${canonical_obs_len}" != "28" ||
      "${streaming_namespace}" != "FEAT_STREAMING_CANONICAL_V2_SEEDED_V1" ||
      "${target_namespace}" != "FEAT_TARGET_CANONICAL_V2_SEEDED_V1" ]]; then
  echo "[err] group is not bound to canonical-v2 seeded-v1" >&2
  exit 3
fi
if [[ -z "${INSCENE_STRICT_AUDIT:-}" ]]; then
  echo "[err] set INSCENE_STRICT_AUDIT to the full passing audit artifact: ${strict_audit_from_row}" >&2
  exit 3
fi
"${PYTHON}" scripts/audit_tstep_canonical_alignment.py verify-artifact \
  --artifact "${INSCENE_STRICT_AUDIT}" \
  --manifest "${manifest}" \
  --vfm "${vfm}" \
  --source "${source}" \
  --protocol-id "${canonical_protocol}" \
  --canonical-obs-len "${canonical_obs_len}" \
  --prefix "${prefix}" \
  --expected-artifact-sha256 "${strict_audit_sha256}"

DATA_ROOT=${DATA_ROOT:-/mnt/umm/users/yinbaiqiao/InsScene-15K/data}
LOCAL_SCENE_ROOT=${LOCAL_SCENE_ROOT:-${DATA_ROOT}/processed_scannetpp_v2/processed_scannetpp_v2}
TARGET_INDEX_ROOT=${TARGET_INDEX_ROOT:-/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_TARGET_CANONICAL_V2_SEEDED_V1}
STREAMING_EXCLUSIONS=${STREAMING_EXCLUSIONS:-${PROJECT_ROOT}/configs/experiment/inscene15k_streaming/tstep_canonical_v2_exclusions.txt}
AOSS_BUCKET_ROOT=${AOSS_BUCKET_ROOT:-aoss://baiqiao/InsScene-15K}
STREAM_ROOT=${STREAM_ROOT:-${AOSS_BUCKET_ROOT}/FEAT_STREAMING_CANONICAL_V2_SEEDED_V1}
TARGET_ROOT=${TARGET_ROOT:-${AOSS_BUCKET_ROOT}/FEAT_TARGET_CANONICAL_V2_SEEDED_V1}
LOG_ROOT=${LOG_ROOT:-${PROJECT_ROOT}/logs/tstep_probe_seeded_v1/${RUN_ID}}
GPU_IDS_CSV=${GPU_IDS_CSV:-0,1,2,3,4,5,6,7}
BATCH_SIZE=${BATCH_SIZE:-2}
BATCH_SIZE_VAL=${BATCH_SIZE_VAL:-2}
NUM_WORKERS=${NUM_WORKERS:-4}
NUM_WORKERS_VAL=${NUM_WORKERS_VAL:-2}
CACHE_MAX_GIB=${CACHE_MAX_GIB:-900}
CACHE_TARGET_GIB=${CACHE_TARGET_GIB:-700}
CACHE_MIN_AGE_SEC=${CACHE_MIN_AGE_SEC:-300}
DELETE_CACHE_ON_EXIT=${DELETE_CACHE_ON_EXIT:-1}
DRY_RUN=${DRY_RUN:-0}
AEC2_NAME=${AEC2_NAME:-zoetrope}

[[ "${AEC2_NAME}" == "zoetrope" ]] || {
  echo "[err] seeded-v1 probing must run in the Zoetrope pool; got AEC2_NAME=${AEC2_NAME}" >&2
  exit 3
}
[[ "${STREAM_ROOT%/}" == "${AOSS_BUCKET_ROOT%/}/${streaming_namespace}" ]] || {
  echo "[err] STREAM_ROOT must be the seeded-v1 AOSS namespace" >&2
  exit 3
}
[[ "${TARGET_ROOT%/}" == "${AOSS_BUCKET_ROOT%/}/${target_namespace}" ]] || {
  echo "[err] TARGET_ROOT must be the seeded-v1 AOSS namespace" >&2
  exit 3
}
[[ "$(basename -- "${TARGET_INDEX_ROOT%/}")" == "${target_namespace}" ]] || {
  echo "[err] TARGET_INDEX_ROOT must end in ${target_namespace}" >&2
  exit 3
}

case "${vfm}" in
  wan) video_channels=1536 ;;
  cogvideox) video_channels=3072 ;;
  minimax_h3) video_channels=5376 ;;
  *) echo "[err] unsupported vfm=${vfm}" >&2; exit 4 ;;
esac

if [[ -d /mnt/local_cache ]]; then
  cache_base=${AOSS_CACHE_BASE:-/mnt/local_cache/inscene/tstep_probe}
else
  cache_base=${AOSS_CACHE_BASE:-/tmp/inscene_tstep_probe_cache}
fi
CACHE_DIR="${cache_base}/${RUN_ID}/${GROUP_ID}"
mkdir -p "${CACHE_DIR}" "${LOG_ROOT}/groups/${GROUP_ID}" "${LOG_ROOT}/runs"
touch "${CACHE_DIR}/.tstep_probe_cache"

export INSCENE_DATA_ROOT="${DATA_ROOT}"
export INSCENE_LOCAL_SCENE_ROOT="${LOCAL_SCENE_ROOT}"
export INSCENE_MANIFEST="${manifest}"
export INSCENE_CANONICAL_PROTOCOL="${canonical_protocol}"
export INSCENE_CANONICAL_OBS_LEN="${canonical_obs_len}"
export INSCENE_STREAMING_EXCLUSIONS="${STREAMING_EXCLUSIONS}"
export INSCENE_STREAMING_FEAT_ROOT="${STREAM_ROOT}"
export INSCENE_TARGET_FEAT_ROOT="${TARGET_ROOT}"
export INSCENE_TARGET_INDEX_ROOT="${TARGET_INDEX_ROOT}"
export INSCENE_TARGET_LAYOUT=flat
export INSCENE_TRUST_EXPANDED_MANIFEST=1
export AOSS_CACHE_DIR="${CACHE_DIR}"
# Default 1 reproduces the 2026-09 campaigns. The B2 (ego_belief_v2) heads of
# CogVideoX/MiniMax diverged to NaN under fp16 pass-through; their 2026-10
# re-runs set PROBE_FP16_PASSTHROUGH=0 (features widened to fp32 in the loader).
export PROBE_FP16_PASSTHROUGH="${PROBE_FP16_PASSTHROUGH:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}}"
mkdir -p "${MPLCONFIGDIR}"

unset HOST || true
export AOSS_API_HOST="${AOSS_API_HOST:-aoss-internal.cn-fz-01.fjscmsapi-oss.com}"
export AOSS_HOST="${AOSS_HOST:-baiqiao.${AOSS_API_HOST}}"
export NO_PROXY="localhost,127.0.0.1,::1,${AOSS_HOST},.${AOSS_API_HOST},${AOSS_API_HOST}"
export no_proxy="${NO_PROXY}"
# SCO jobs run as root inside the container, where ~/.petreloss.conf does not
# exist; vidfm3d.utils.aoss_fs resolves its client config from AOSS_CONF first.
# (2026-10-16: 32 B2 re-runs failed at the first remote read without this.)
export AOSS_CONF="${AOSS_CONF:-/mnt/umm/users/yinbaiqiao/aoss/petreloss.conf}"
[[ -r "${AOSS_CONF}" ]] || { echo "[err] AOSS_CONF is not readable: ${AOSS_CONF}" >&2; exit 3; }

IFS=',' read -r -a gpu_ids <<< "${GPU_IDS_CSV}"
if [[ "${DRY_RUN}" != "1" ]]; then
  "${PYTHON}" - "${#gpu_ids[@]}" <<'PY'
import sys
import torch

expected = int(sys.argv[1])
actual = torch.cuda.device_count()
if actual != expected:
    raise SystemExit(f"H800 gate expected {expected} visible CUDA devices, found {actual}")
names = [torch.cuda.get_device_name(index) for index in range(actual)]
if any("H800" not in name for name in names):
    raise SystemExit(f"seeded-v1 probing is H800-only; visible devices={names}")
print(f"H800 gate PASS: {actual} visible CUDA device(s): {names}")
PY
fi

cache_log="${LOG_ROOT}/groups/${GROUP_ID}/cache_pruner.log"
"${PYTHON}" scripts/prune_tstep_probe_cache.py \
  --cache-dir "${CACHE_DIR}" \
  --max-gib "${CACHE_MAX_GIB}" \
  --target-gib "${CACHE_TARGET_GIB}" \
  --min-age-sec "${CACHE_MIN_AGE_SEC}" \
  > "${cache_log}" 2>&1 &
cache_pid=$!
pids=()
task_names=()

cleanup() {
  for pid in "${pids[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
  kill "${cache_pid}" 2>/dev/null || true
  wait "${cache_pid}" 2>/dev/null || true
  if [[ "${DELETE_CACHE_ON_EXIT}" == "1" ]]; then
    "${PYTHON}" scripts/prune_tstep_probe_cache.py \
      --cache-dir "${CACHE_DIR}" --purge >> "${cache_log}" 2>&1 || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

IFS=',' read -r -a tasks <<< "${tasks_csv}"
if (( ${#tasks[@]} > ${#gpu_ids[@]} )); then
  echo "[err] tasks=${#tasks[@]} exceeds GPUs=${#gpu_ids[@]}" >&2
  exit 5
fi

echo "group=${GROUP_ID} status=${status} host=$(hostname)"
echo "vfm=${vfm} timestep=${timestep} layer=${layer} prefix=${prefix}"
echo "tasks=${tasks_csv} manifest=${manifest}"
echo "strict_audit=${INSCENE_STRICT_AUDIT} protocol=${canonical_protocol} obs_len=${canonical_obs_len}"
echo "stream_root=${STREAM_ROOT} target_root=${TARGET_ROOT}"
echo "local_scene_root=${LOCAL_SCENE_ROOT} target_index_root=${TARGET_INDEX_ROOT}"
echo "cache_dir=${CACHE_DIR} cache_max_gib=${CACHE_MAX_GIB}"

for i in "${!tasks[@]}"; do
  task="${tasks[$i]}"
  gpu="${gpu_ids[$i]}"
  tag="tp_${vfm}_t${timestep}_l${layer}_p${prefix}_${task}"
  run_folder="tstep_probe_seeded_v1/${RUN_ID}/${tag}"
  run_dir="${PROJECT_ROOT}/logs/inscene15k_streaming/runs/${run_folder}"
  task_log="${LOG_ROOT}/groups/${GROUP_ID}/${task}.log"
  complete_marker="${run_dir}/TRAIN_COMPLETE"

  if [[ "${SKIP_COMPLETE:-1}" == "1" && -f "${complete_marker}" ]]; then
    echo "[skip] ${tag}: ${complete_marker}"
    continue
  fi

  task_bs="${BATCH_SIZE}"
  task_bs_val="${BATCH_SIZE_VAL}"
  if [[ "${task}" == "streaming_depth" || "${task}" == "streaming_movement" || "${task}" == "streaming_mask" ]]; then
    task_bs=1
    task_bs_val=1
  fi

  (
    set +e
    echo "BEGIN $(date -u +%Y-%m-%dT%H:%M:%SZ) gpu=${gpu} tag=${tag}"
    cmd=(
      env
      CUDA_VISIBLE_DEVICES="${gpu}"
      AOSS_FS_METRICS="${LOG_ROOT}/groups/${GROUP_ID}/${task}_aoss_metrics.jsonl"
      "${PYTHON}" vidfm3d/train.py
      "experiment=inscene15k_streaming/${task}_${vfm}_v1"
      "vfm_name=${vfm}"
      "video_channels=${video_channels}"
      "streaming_feat_root=${STREAM_ROOT}"
      "feature_layer=${layer}"
      "feature_timestep=${timestep}"
      "feat_postfix=_t${timestep}_layer${layer}"
      "prefix_len=${prefix}"
      "job_name=${tag}"
      "paths.run_folder_name=${run_folder}"
      "logger.wandb.name=${RUN_ID}_${tag}"
      "logger.wandb.group=${RUN_ID}"
      "logger.wandb.offline=true"
      "autoresume=true"
      "trainer.devices=[0]"
      "data.data_module.batch_size_per_device=${task_bs}"
      "data.data_module.batch_size_per_device_val=${task_bs_val}"
      "data.data_module.num_workers=${NUM_WORKERS}"
      "data.data_module.num_workers_val=${NUM_WORKERS_VAL}"
    )
    case "${task}" in
      action_dynamics|path_integration|counterfactual)
        cmd+=("target_feat_root=${TARGET_ROOT}")
        ;;
    esac
    if [[ "${DRY_RUN}" == "1" ]]; then
      printf '[dry-run] '
      printf '%q ' "${cmd[@]}"
      printf '\n'
      rc=0
    else
      "${cmd[@]}"
      rc=$?
    fi
    if (( rc == 0 )) && [[ "${DRY_RUN}" != "1" ]]; then
      mkdir -p "${run_dir}"
      touch "${complete_marker}"
    fi
    echo "END $(date -u +%Y-%m-%dT%H:%M:%SZ) rc=${rc}"
    exit "${rc}"
  ) > "${task_log}" 2>&1 &
  pids+=("$!")
  task_names+=("${task}")
done

failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[ok] ${task_names[$i]}"
  else
    rc=$?
    echo "[failed] ${task_names[$i]} rc=${rc}" >&2
    failed=1
  fi
done
pids=()

if (( failed != 0 )); then
  echo "[err] one or more probes failed; see ${LOG_ROOT}/groups/${GROUP_ID}" >&2
  exit 10
fi
echo "group_complete=${GROUP_ID} at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
