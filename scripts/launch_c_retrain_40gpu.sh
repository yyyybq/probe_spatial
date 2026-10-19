#!/usr/bin/env bash
# Launch C-series retraining across the available 5x8 H800 nodes.
#
# The launcher starts one supervisor per node.  Each supervisor runs a bounded
# local queue so jobs continue as earlier jobs finish without overloading AOSS.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
PYTHON=${PYTHON:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}
SSH_KEY=${SSH_KEY:-/mnt/umm/users/yinbaiqiao/id_rsa}
KNOWN_HOSTS=${KNOWN_HOSTS:-/tmp/known_hosts}
RUN_ID=${RUN_ID:-c_retrain_20260722}
SLOTS_PER_NODE=${SLOTS_PER_NODE:-10}
NUM_WORKERS=${NUM_WORKERS:-4}
NUM_WORKERS_VAL=${NUM_WORKERS_VAL:-2}
BATCH_SIZE=${BATCH_SIZE:-8}
BATCH_SIZE_VAL=${BATCH_SIZE_VAL:-8}
GPU_IDS_CSV=${GPU_IDS_CSV:-0,1,2,3,4,5,6,7}
STREAM_ROOT_BASE=${STREAM_ROOT_BASE:-fj:s3://baiqiao/InsScene-15K/FEAT_STREAMING_BY_PREFIX}
TARGET_ROOT_BASE=${TARGET_ROOT_BASE:-fj:s3://baiqiao/InsScene-15K/FEAT_TARGET_BY_PREFIX}

NODES=(
  10.119.17.135
  10.119.16.151
  10.119.31.225
  10.119.20.37
  10.119.19.88
)
if [[ -n "${NODES_CSV:-}" ]]; then
  IFS=',' read -r -a NODES <<< "${NODES_CSV}"
fi

VFMS=(wan cogvideox vjepa2 dino aether f3r)
PROBES=(action_dynamics path_integration counterfactual)
PREFIXES=(8 12 16 24)

manifest_for_prefix() {
  local prefix="$1"
  printf '%s/manifests/c_retrain_20260722/inscene_c_h124_prefix%s.jsonl' "${PROJECT_ROOT}" "${prefix}"
}

layer_for_vfm() {
  case "$1" in
    wan) printf '20' ;;
    cogvideox) printf '20' ;;
    vjepa2) printf '23' ;;
    dino) printf '0' ;;
    aether) printf '1' ;;
    f3r) printf '24' ;;
    videomae) printf '23' ;;
    qwen2_5_vl) printf '-1' ;;
    qwen2_5_vl_3b) printf '-1' ;;
    *) return 1 ;;
  esac
}

write_job_lists() {
  local out_dir="$1"
  rm -rf "${out_dir}/jobs"
  mkdir -p "${out_dir}/jobs"
  local idx=0
  if [[ -n "${JOB_LIST:-}" ]]; then
    while IFS=$'\t' read -r vfm probe prefix layer manifest; do
      [[ -z "${vfm:-}" || "${vfm}" == \#* ]] && continue
      local node_idx=$((idx % ${#NODES[@]}))
      local node="${NODES[$node_idx]}"
      printf '%s\t%s\t%s\t%s\t%s\n' "${vfm}" "${probe}" "${prefix}" "${layer}" "${manifest}" \
        >> "${out_dir}/jobs/${node}.tsv"
      idx=$((idx + 1))
    done < "${JOB_LIST}"
    return
  fi
  for vfm in "${VFMS[@]}"; do
    for prefix in "${PREFIXES[@]}"; do
      for probe in "${PROBES[@]}"; do
        local node_idx=$((idx % ${#NODES[@]}))
        local node="${NODES[$node_idx]}"
        local layer
        layer="$(layer_for_vfm "${vfm}")"
        local manifest
        manifest="$(manifest_for_prefix "${prefix}")"
        printf '%s\t%s\t%s\t%s\t%s\n' "${vfm}" "${probe}" "${prefix}" "${layer}" "${manifest}" \
          >> "${out_dir}/jobs/${node}.tsv"
        idx=$((idx + 1))
      done
    done
  done
}

launch_node() {
  local node="$1"
  local job_file="$2"
  local out_dir="$3"
  local remote_job_file="/tmp/${RUN_ID}_${node}.jobs.tsv"
  local remote_launcher="/tmp/${RUN_ID}_${node}.launcher.sh"

  scp -i "${SSH_KEY}" -o UserKnownHostsFile="${KNOWN_HOSTS}" -o StrictHostKeyChecking=accept-new \
    "${job_file}" "root@${node}:${remote_job_file}" >/dev/null

  ssh -i "${SSH_KEY}" -o UserKnownHostsFile="${KNOWN_HOSTS}" -o StrictHostKeyChecking=accept-new \
    -o BatchMode=yes "root@${node}" "cat > ${remote_launcher} && chmod +x ${remote_launcher}" <<'REMOTE_LAUNCHER'
#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$1"
PYTHON="$2"
RUN_ID="$3"
JOB_FILE="$4"
SLOTS_PER_NODE="$5"
NUM_WORKERS="$6"
NUM_WORKERS_VAL="$7"
BATCH_SIZE="$8"
BATCH_SIZE_VAL="$9"
GPU_IDS_CSV="${10}"
STREAM_ROOT_BASE="${11}"
TARGET_ROOT_BASE="${12}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export INSCENE_DATA_ROOT='fj:s3://baiqiao/InsScene-15K/data/processed_scannetpp_v2/processed_scannetpp_v2'
export STREAM_ROOT_BASE
unset INSCENE_STREAMING_FEAT_ROOT
export MPLCONFIGDIR="/tmp/matplotlib-${USER:-root}"
mkdir -p "${MPLCONFIGDIR}"

LOG_ROOT="${PROJECT_ROOT}/logs/inscene15k_streaming/${RUN_ID}/launch"
mkdir -p "${LOG_ROOT}"
IFS=',' read -r -a GPU_IDS <<< "${GPU_IDS_CSV}"
if (( ${#GPU_IDS[@]} == 0 )); then
  echo "No GPU ids supplied via GPU_IDS_CSV='${GPU_IDS_CSV}'" >&2
  exit 1
fi

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
  local gpu="${GPU_IDS[$((slot % ${#GPU_IDS[@]}))]}"
  local tag="${RUN_ID}_${probe}_${vfm}_p${prefix}_layer${layer}"
  local log="${LOG_ROOT}/${tag}.log"
  local target_root="${TARGET_ROOT_BASE}/${vfm}_prefix_${prefix}_m0p35_r0p5"
  local run_folder="${RUN_ID}/${tag}"
  local extra_train
  extra_train="autoresume=false data.data_module.batch_size_per_device=${BATCH_SIZE} data.data_module.batch_size_per_device_val=${BATCH_SIZE_VAL} data.data_module.num_workers=${NUM_WORKERS} data.data_module.num_workers_val=${NUM_WORKERS_VAL} logger.wandb.offline=true logger.wandb.group=${RUN_ID} paths.run_folder_name=${run_folder} logger.wandb.name=${tag}"

  (
    set +e
    export INSCENE_MANIFEST="${manifest}"
    export INSCENE_TARGET_FEAT_ROOT="${target_root}"
    export AOSS_CACHE_DIR="/mnt/local_cache/inscene/aoss_cache/${RUN_ID}/${tag}"
    mkdir -p "${AOSS_CACHE_DIR}"
    echo "BEGIN $(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) gpu=${gpu} vfm=${vfm} probe=${probe} prefix=${prefix} layer=${layer} manifest=${manifest}"
    env \
      PYTHON="${PYTHON}" \
      DEV="${gpu}" \
      VFM="${vfm}" \
      VFM_NAME="${vfm}" \
      PROBES="${probe}" \
      PREFIX_LENGTHS="${prefix}" \
      C_PREFIX_LENGTHS="${prefix}" \
      LAYERS="${layer}" \
      TRAIN=1 \
      EVAL=0 \
      EXTRACT_STREAMING=0 \
      EXTRACT_TARGETS=0 \
      TARGET_FEAT_ROOT="${target_root}" \
      STREAM_ROOT_BASE="${STREAM_ROOT_BASE}" \
      EXTRA_TRAIN="${extra_train}" \
      bash scripts/run_streaming_probe_sweep.sh
    rc=$?
    echo "END $(date -u +%Y-%m-%dT%H:%M:%SZ) rc=${rc}"
    exit "${rc}"
  ) > "${log}" 2>&1 &
}

slot=0
while IFS=$'\t' read -r vfm probe prefix layer manifest; do
  while (( $(active_jobs) >= SLOTS_PER_NODE )); do
    sleep 30
  done
  run_one "${slot}" "${vfm}" "${probe}" "${prefix}" "${layer}" "${manifest}"
  slot=$((slot + 1))
  sleep 3
done < "${JOB_FILE}"

wait
REMOTE_LAUNCHER

  ssh -n -f -i "${SSH_KEY}" -o UserKnownHostsFile="${KNOWN_HOSTS}" -o StrictHostKeyChecking=accept-new \
    -o BatchMode=yes "root@${node}" \
    "nohup ${remote_launcher} '${PROJECT_ROOT}' '${PYTHON}' '${RUN_ID}' '${remote_job_file}' '${SLOTS_PER_NODE}' '${NUM_WORKERS}' '${NUM_WORKERS_VAL}' '${BATCH_SIZE}' '${BATCH_SIZE_VAL}' '${GPU_IDS_CSV}' '${STREAM_ROOT_BASE}' '${TARGET_ROOT_BASE}' > '${out_dir}/${node}.supervisor.log' 2>&1 & echo \$! > '${out_dir}/${node}.supervisor.pid'"
}

main() {
  local out_dir="${PROJECT_ROOT}/logs/inscene15k_streaming/${RUN_ID}/launcher"
  mkdir -p "${out_dir}"
  write_job_lists "${out_dir}"

  for node in "${NODES[@]}"; do
    local job_file="${out_dir}/jobs/${node}.tsv"
    if [[ ! -s "${job_file}" ]]; then
      continue
    fi
    echo "Launching ${node}: $(wc -l < "${job_file}") jobs, slots=${SLOTS_PER_NODE}, gpus=${GPU_IDS_CSV}"
    launch_node "${node}" "${job_file}" "${out_dir}"
  done

  echo "Launcher state: ${out_dir}"
}

main "$@"
