#!/usr/bin/env bash
# Submit official tstep A1/A2/B2 held-out eval shards to SCO.
set -euo pipefail

export PATH="${HOME}/.sco/bin:${PATH}"

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
PYTHON=${PYTHON:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}
RUN_ID=${RUN_ID:-tstep_eval_a1a2b2_20260923_r1}
OUT_DIR=${OUT_DIR:-${PROJECT_ROOT}/artifacts/${RUN_ID}}
NUM_SHARDS=${NUM_SHARDS:-3}
SLOTS=${SLOTS:-8}
NPROC=${NPROC:-8}
DRY_SUBMIT=${DRY_SUBMIT:-0}
AEC2_NAME=${AEC2_NAME:-h800}
JOB_PREFIX=${JOB_PREFIX:-bq-tstep-eval-a1a2b2-0923}

container_image_url="registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617"
storage_mount="019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm"

case "${AEC2_NAME}" in
  zoetrope|h800) worker_spec="N4lS.Iq.I80." ;;
  vigen|si) worker_spec="N6lS.Iq.I10." ;;
  *) echo "[err] invalid AEC2_NAME=${AEC2_NAME}" >&2; exit 1 ;;
esac

cd "${PROJECT_ROOT}"
"${PYTHON}" analyses/tstep_canonical_v2/build_tstep_eval_manifest.py \
  --run-id "${RUN_ID}" --out-dir "${OUT_DIR}" --num-shards "${NUM_SHARDS}"

mapfile -t SHARDS < <(ls "${OUT_DIR}/jobs/shards"/shard_*.tsv | sort)
n_jobs=$(( $(wc -l < "${OUT_DIR}/jobs/eval_jobs.tsv") - 1 ))
echo "RUN_ID=${RUN_ID} jobs=${n_jobs} shards=${#SHARDS[@]} aec2=${AEC2_NAME}"

if [[ "${DRY_SUBMIT}" == "1" ]]; then
  echo "DRY_SUBMIT=1: manifest only"
  exit 0
fi

mkdir -p "${OUT_DIR}/launcher"
submit_log="${OUT_DIR}/launcher/sco_submit.log"
: > "${submit_log}"
idx=0
for shard in "${SHARDS[@]}"; do
  idx=$((idx + 1))
  shard_abs="$(readlink -f "${shard}")"
  job_name="$(printf '%s-s%02d' "${JOB_PREFIX}" "${idx}")"
  cmd="cd ${PROJECT_ROOT} && RUN_ID=${RUN_ID} OUT_DIR=${OUT_DIR} SHARD_TSV=${shard_abs} SLOTS=${SLOTS} GPU_IDS_CSV=0,1,2,3,4,5,6,7 ${PYTHON} analyses/tstep_canonical_v2/run_tstep_eval_shard.py"
  echo "[submit] ${job_name} shard=$(basename "${shard}") jobs=$(( $(wc -l < "${shard}") - 1 ))" | tee -a "${submit_log}"
  sco acp jobs create \
    --workspace-name=aigc \
    --aec2-name="${AEC2_NAME}" \
    --job-name="${job_name}" \
    --priority=HIGHEST \
    --quota-type=reserved \
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
