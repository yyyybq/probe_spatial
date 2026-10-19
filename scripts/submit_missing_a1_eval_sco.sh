#!/usr/bin/env bash
# Build and submit missing A1 streaming eval jobs to SCO.
set -euo pipefail

export PATH="${HOME}/.sco/bin:${PATH}"

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
PYTHON=${PYTHON:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}
RUN_ID=${RUN_ID:-missing_a1_eval_20260817}
OUT_DIR=${OUT_DIR:-${PROJECT_ROOT}/logs/probing_jobs_${RUN_ID}}
NUM_SHARDS=${NUM_SHARDS:-2}
SLOTS=${SLOTS:-8}
NPROC=${NPROC:-8}
DRY_SUBMIT=${DRY_SUBMIT:-0}
AEC2_NAME=${1:-zoetrope}
JOB_PREFIX=${JOB_PREFIX:-bq_missing_a1_eval}

container_image_url="registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617"
storage_mount="019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm"

case "${AEC2_NAME}" in
  zoetrope|h800) worker_spec="N4lS.Iq.I80." ;;
  vigen|si) worker_spec="N6lS.Iq.I10." ;;
  *) echo "[err] invalid AEC2_NAME=${AEC2_NAME}" >&2; exit 1 ;;
esac

cd "${PROJECT_ROOT}"
"${PYTHON}" scripts/build_missing_a1_eval_jobs.py --run-id "${RUN_ID}" --out-dir "${OUT_DIR}"

JOBS_TSV="${OUT_DIR}/jobs/eval_jobs.tsv"
n_jobs=$(( $(wc -l < "${JOBS_TSV}") - 1 ))
if (( n_jobs < 1 )); then
  echo "[ok] no missing A1 eval jobs"
  exit 0
fi

mkdir -p "${OUT_DIR}/jobs/shards" "${OUT_DIR}/launcher"
rm -f "${OUT_DIR}/jobs/shards"/shard_*.tsv
body="${OUT_DIR}/jobs/body.tsv"
tail -n +2 "${JOBS_TSV}" > "${body}"
split -d -a 2 -n "l/${NUM_SHARDS}" "${body}" "${OUT_DIR}/jobs/shards/shard_"
for shard in "${OUT_DIR}/jobs/shards"/shard_*; do
  [[ -f "${shard}" ]] || continue
  [[ "${shard}" == *.tsv ]] && continue
  {
    head -n 1 "${JOBS_TSV}"
    cat "${shard}"
  } > "${shard}.tsv"
  rm -f "${shard}"
done

mapfile -t SHARDS < <(ls "${OUT_DIR}/jobs/shards"/shard_*.tsv | sort)
echo "RUN_ID=${RUN_ID} jobs=${n_jobs} shards=${#SHARDS[@]} aec2=${AEC2_NAME}"

cat > "${OUT_DIR}/PLAN.txt" <<EOF
Missing A1 eval submission
==========================
RUN_ID: ${RUN_ID}
jobs: ${n_jobs}
shards: ${#SHARDS[@]}
slots per shard: ${SLOTS}

Manifest:
  ${JOBS_TSV}

Run locally:
  RUN_ID=${RUN_ID} OUT_DIR=${OUT_DIR} SHARD_TSV=<shard.tsv> SLOTS=${SLOTS} python scripts/run_missing_a1_eval_shard.py
EOF

if [[ "${DRY_SUBMIT}" == "1" ]]; then
  echo "DRY_SUBMIT=1: prepared manifest/shards only"
  exit 0
fi

submit_log="${OUT_DIR}/launcher/sco_submit.log"
: > "${submit_log}"
idx=0
for shard in "${SHARDS[@]}"; do
  idx=$((idx + 1))
  shard_abs="$(readlink -f "${shard}")"
  job_name="$(printf '%s_%s_s%02d' "${JOB_PREFIX}" "${RUN_ID}" "${idx}")"
  cmd="cd ${PROJECT_ROOT} && RUN_ID=${RUN_ID} OUT_DIR=${OUT_DIR} SHARD_TSV=${shard_abs} SLOTS=${SLOTS} GPU_IDS_CSV=0,1,2,3,4,5,6,7 python scripts/run_missing_a1_eval_shard.py"
  echo "[submit] ${job_name} shard=$(basename "${shard}") jobs=$(( $(wc -l < "${shard}") - 1 ))" | tee -a "${submit_log}"
  sco acp jobs create \
    --workspace-name=aigc \
    --aec2-name="${AEC2_NAME}" \
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

grep -oE 'pt-[a-z0-9]+' "${submit_log}" | awk '!seen[$0]++' > "${OUT_DIR}/launcher/sco_job_ids.txt" || true
echo "SCO job ids: $(tr '\n' ' ' < "${OUT_DIR}/launcher/sco_job_ids.txt")"
