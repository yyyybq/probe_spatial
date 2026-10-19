#!/usr/bin/env bash
# Prepare the 240-job ABC matrix and submit shard launchers to SCO ACP.
#
# Default: 5 × 8-GPU H800 nodes, 16 concurrent slots/node, AOSS raw features.
#
# Usage:
#   bash scripts/submit_vjepa2_videomae_abc_probe_sco.sh
#   NUM_SHARDS=3 bash scripts/submit_vjepa2_videomae_abc_probe_sco.sh zoetrope
#   DRY_SUBMIT=1 bash scripts/submit_vjepa2_videomae_abc_probe_sco.sh   # prepare only
set -euo pipefail

export PATH="${HOME}/.sco/bin:${PATH}"

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
cd "${PROJECT_ROOT}"

aec2_name=${1:-"zoetrope"}
NUM_SHARDS=${NUM_SHARDS:-5}
NPROC=${NPROC:-8}
SLOTS=${SLOTS:-16}
BATCH_SIZE=${BATCH_SIZE:-8}
BATCH_SIZE_VAL=${BATCH_SIZE_VAL:-8}
DRY_SUBMIT=${DRY_SUBMIT:-0}
JOB_PREFIX=${JOB_PREFIX:-bq_vm_vj2_abc}
RUN_ID=${RUN_ID:-vjepa2_videomae_abc_$(date -u +%Y%m%d_%H%M)}
OUT_DIR=${OUT_DIR:-${PROJECT_ROOT}/logs/probing_jobs_${RUN_ID}}
AUTO_MONITOR=${AUTO_MONITOR:-1}
MONITOR_INTERVAL_SEC=${MONITOR_INTERVAL_SEC:-60}
MONITOR_MAX_HOURS=${MONITOR_MAX_HOURS:-36}
SMOKE_HYDRA=${SMOKE_HYDRA:-1}
EXPECTED_JOBS=${EXPECTED_JOBS:-240}
PYTHON=${PYTHON:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}

container_image_url="registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617"
storage_mount="019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm"

require_local_files() {
  local missing=0
  local rel
  local required=(
    "vidfm3d/train.py"
    "vidfm3d/models/video_probe_module.py"
    "vidfm3d/models/probe_ext_module.py"
    "vidfm3d/models/components/dpt_head.py"
    "vidfm3d/utils/aoss_fs.py"
    "configs/callbacks/v3_checkpoint.yaml"
    "configs/train.yaml"
    "configs/experiment/inscene15k_streaming/streaming_mask_vjepa2_v1.yaml"
    "configs/experiment/inscene15k_streaming/streaming_mask_videomae_v1.yaml"
    "scripts/run_vjepa2_videomae_abc_probe_shard.sh"
    "scripts/launch_vjepa2_videomae_abc_probe.sh"
    "scripts/monitor_vjepa2_videomae_abc_probe.py"
  )
  echo "## hard preflight: required local files"
  for rel in "${required[@]}"; do
    if [[ -f "${PROJECT_ROOT}/${rel}" ]]; then
      echo "OK   ${rel}"
    else
      echo "MISS ${rel}"
      missing=1
    fi
  done
  if (( missing == 1 )); then
    echo "[err] refusing submit: restore missing files first (git checkout HEAD -- <path>)" >&2
    exit 2
  fi
}

smoke_hydra_compose() {
  # Catch Hydra MissingConfig / deleted callback yaml before spending GPUs.
  if [[ "${SMOKE_HYDRA}" != "1" ]]; then
    echo "[smoke] skipped (SMOKE_HYDRA=${SMOKE_HYDRA})"
    return 0
  fi
  echo "## hard preflight: hydra compose smoke"
  local cfg="experiment=inscene15k_streaming/streaming_depth_vjepa2_v1"
  if ! "${PYTHON}" - <<PY
import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pathlib import Path
cfg_dir = str(Path("${PROJECT_ROOT}") / "configs")
with initialize_config_dir(version_base="1.3", config_dir=cfg_dir):
    cfg = compose(config_name="train", overrides=["${cfg}", "trainer.devices=[0]"])
assert "callbacks" in cfg, cfg
print("smoke_ok", "${cfg}", "callbacks=", type(cfg.callbacks).__name__)
PY
  then
    echo "[err] hydra compose smoke failed; refusing SCO submit" >&2
    exit 3
  fi
}

if [ "${aec2_name}" = "zoetrope" ] || [ "${aec2_name}" = "h800" ]; then
  worker_spec="N4lS.Iq.I80."
elif [ "${aec2_name}" = "vigen" ] || [ "${aec2_name}" = "si" ]; then
  worker_spec="N6lS.Iq.I10."
else
  echo "Error: Invalid aec2_name '${aec2_name}'" >&2
  exit 1
fi

echo "=============================================="
echo " prepare + submit V-JEPA2/VideoMAE ABC probes"
echo " aec2       : ${aec2_name}"
echo " shards     : ${NUM_SHARDS} x ${NPROC}GPU (slots=${SLOTS})"
echo " batch_size : ${BATCH_SIZE}"
echo " RUN_ID     : ${RUN_ID}"
echo " OUT_DIR    : ${OUT_DIR}"
echo " DRY_SUBMIT : ${DRY_SUBMIT}"
echo " AUTO_MON   : ${AUTO_MONITOR}"
echo "=============================================="

require_local_files
smoke_hydra_compose

# 1) Prepare full 240-job matrix (AOSS raw roots).
FEAT_BACKEND=aoss \
RUN_ID="${RUN_ID}" \
OUT_DIR="${OUT_DIR}" \
BATCH_SIZE="${BATCH_SIZE}" \
BATCH_SIZE_VAL="${BATCH_SIZE_VAL}" \
bash scripts/prepare_vjepa2_videomae_abc_probe.sh

if grep -q '^MISS ' "${OUT_DIR}/PREFLIGHT.txt"; then
  echo "[err] PREFLIGHT still reports MISS entries; refusing submit:" >&2
  grep '^MISS ' "${OUT_DIR}/PREFLIGHT.txt" >&2 || true
  exit 2
fi

JOBS_TSV="${OUT_DIR}/jobs/all.tsv"
n_jobs=$(wc -l < "${JOBS_TSV}")
if (( n_jobs != EXPECTED_JOBS )); then
  echo "[err] expected ${EXPECTED_JOBS} jobs, got ${n_jobs}" >&2
  exit 1
fi

# 2) Split into shards.
mkdir -p "${OUT_DIR}/jobs/shards" "${OUT_DIR}/launcher"
rm -f "${OUT_DIR}/jobs/shards"/shard_*.tsv
split -d -a 2 -n "l/${NUM_SHARDS}" "${JOBS_TSV}" "${OUT_DIR}/jobs/shards/shard_"
# rename to .tsv
i=0
for f in "${OUT_DIR}/jobs/shards"/shard_*; do
  [[ -f "${f}" ]] || continue
  [[ "${f}" == *.tsv ]] && continue
  mv "${f}" "${f}.tsv"
  i=$((i + 1))
done
mapfile -t SHARDS < <(ls "${OUT_DIR}/jobs/shards"/shard_*.tsv | sort)
echo "created ${#SHARDS[@]} shards:"
for s in "${SHARDS[@]}"; do
  echo "  $(basename "${s}"): $(wc -l < "${s}") jobs"
done

chmod +x scripts/run_vjepa2_videomae_abc_probe_shard.sh scripts/launch_vjepa2_videomae_abc_probe.sh

if [[ "${DRY_SUBMIT}" == "1" ]]; then
  echo "DRY_SUBMIT=1: skipping sco create"
  exit 0
fi

# 3) Submit one SCO job per shard.
submit_log="${OUT_DIR}/launcher/sco_submit.log"
: > "${submit_log}"
idx=0
for shard in "${SHARDS[@]}"; do
  idx=$((idx + 1))
  job_name=$(printf '%s_s%02d' "${JOB_PREFIX}" "${idx}")
  shard_abs="$(readlink -f "${shard}")"
  cmd="cd ${PROJECT_ROOT} && OUT_DIR=${OUT_DIR} RUN_ID=${RUN_ID} SHARD_TSV=${shard_abs} SLOTS=${SLOTS} BATCH_SIZE=${BATCH_SIZE} BATCH_SIZE_VAL=${BATCH_SIZE_VAL} FEAT_BACKEND=aoss bash scripts/run_vjepa2_videomae_abc_probe_shard.sh"
  echo "[submit] ${job_name} shard=$(basename "${shard}") jobs=$(wc -l < "${shard}")" | tee -a "${submit_log}"
  sco acp jobs create \
    --workspace-name=aigc \
    --aec2-name="${aec2_name}" \
    --job-name="${job_name}" \
    --priority=HIGHEST \
    --container-image-url="${container_image_url}" \
    --storage-mount="${storage_mount}" \
    --training-framework=pytorch \
    --worker-nodes=1 \
    --worker-spec="${worker_spec}${NPROC}" \
    --command="${cmd}" \
    | tee -a "${submit_log}"
done

echo
echo "Submitted ${#SHARDS[@]} SCO jobs for RUN_ID=${RUN_ID}"

# Persist SCO job ids for the monitor / humans.
job_ids_file="${OUT_DIR}/launcher/sco_job_ids.txt"
grep -oE 'pt-[a-z0-9]+' "${submit_log}" | awk '!seen[$0]++' > "${job_ids_file}" || true
echo "SCO job ids: $(tr '\n' ' ' < "${job_ids_file}")"

if [[ "${AUTO_MONITOR}" == "1" ]]; then
  mkdir -p "${OUT_DIR}/launcher"
  mon_log="${OUT_DIR}/launcher/monitor.log"
  mon_pid_file="${OUT_DIR}/launcher/monitor.pid"
  nohup "${PYTHON}" scripts/monitor_vjepa2_videomae_abc_probe.py \
    --run-id "${RUN_ID}" \
    --out-dir "${OUT_DIR}" \
    --expected "${n_jobs}" \
    --interval-sec "${MONITOR_INTERVAL_SEC}" \
    --max-hours "${MONITOR_MAX_HOURS}" \
    > "${mon_log}" 2>&1 &
  echo $! > "${mon_pid_file}"
  echo "Started background monitor pid=$(cat "${mon_pid_file}")"
  echo "  progress : ${OUT_DIR}/launcher/PROGRESS.txt"
  echo "  alert    : ${OUT_DIR}/launcher/ALERT.txt   (created on failure storm)"
  echo "  monitor  : ${mon_log}"
fi

echo "Manual monitor:"
echo "  sco acp jobs list --workspace-name=aigc --page-size 20"
echo "  python scripts/monitor_vjepa2_videomae_abc_probe.py --run-id ${RUN_ID} --once"
echo "  ls logs/inscene15k_streaming/${RUN_ID}/launch/ | head"
echo "  tail -f ${OUT_DIR}/launcher/PROGRESS.txt"
