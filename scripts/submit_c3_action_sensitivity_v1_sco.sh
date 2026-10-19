#!/usr/bin/env bash
# Submit the prepared V1 C3-from-C1 evaluation-only sweep to SCO.
set -euo pipefail

export PATH="/mnt/umm/users/yinbaiqiao/.sco/bin:${PATH}"

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
PYTHON_BIN=${PYTHON_BIN:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}
C3_SWEEP_DIR=${C3_SWEEP_DIR:?set C3_SWEEP_DIR to the prepared sweep directory}
AEC2_NAME=${AEC2_NAME:-h800}
NUM_SHARDS=${NUM_SHARDS:-5}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
NPROC=${NPROC:-8}
DRY_SUBMIT=${DRY_SUBMIT:-0}
JOB_PREFIX=${JOB_PREFIX:-c3_c1_v1}
SUBMISSION_TAG=${SUBMISSION_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}

case "${AEC2_NAME}" in
  h800|zoetrope) worker_spec="N4lS.Iq.I80." ;;
  si|vigen) worker_spec="N6lS.Iq.I10." ;;
  *) echo "invalid AEC2_NAME=${AEC2_NAME}" >&2; exit 1 ;;
esac

manifest="${C3_SWEEP_DIR}/sweep_manifest.csv"
[[ -f "${manifest}" ]] || { echo "missing ${manifest}" >&2; exit 1; }
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/c3_action_sensitivity_v1.py" gate --manifest "${manifest}"
mkdir -p "${C3_SWEEP_DIR}/launcher"
submit_log="${C3_SWEEP_DIR}/launcher/sco_submit_${SUBMISSION_TAG}.log"
: > "${submit_log}"

if [[ "${DRY_SUBMIT}" == "1" ]]; then
  echo "DRY_SUBMIT=1 manifest=${manifest} shards=${NUM_SHARDS} gpus=${GPUS}"
  exit 0
fi

for ((shard=0; shard<NUM_SHARDS; shard++)); do
  job_name=$(printf '%s_s%02d' "${JOB_PREFIX}" "${shard}")
  cmd="cd ${PROJECT_ROOT} && PYTHON_BIN=${PYTHON_BIN} ${PYTHON_BIN} scripts/c3_action_sensitivity_v1.py run --manifest ${manifest} --shard-index ${shard} --num-shards ${NUM_SHARDS} --gpu-ids ${GPUS}"
  echo "[submit] ${job_name}" | tee -a "${submit_log}"
  sco acp jobs create \
    --workspace-name=aigc \
    --aec2-name="${AEC2_NAME}" \
    --job-name="${job_name}" \
    --priority=HIGHEST \
    --container-image-url="registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617" \
    --storage-mount="019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm" \
    --training-framework=pytorch \
    --worker-nodes=1 \
    --worker-spec="${worker_spec}${NPROC}" \
    --command="${cmd}" | tee -a "${submit_log}"
done

job_ids="${C3_SWEEP_DIR}/launcher/sco_job_ids_${SUBMISSION_TAG}.txt"
grep -oE 'pt-[a-z0-9]+' "${submit_log}" | awk '!seen[$0]++' > "${job_ids}" || true
echo "submitted job ids: $(tr '\n' ' ' < "${job_ids}")"
