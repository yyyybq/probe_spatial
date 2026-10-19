#!/usr/bin/env bash
set -euo pipefail

export PATH="${HOME}/.sco/bin:${PATH}"
PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
RUN_TAG=${RUN_TAG:?RUN_TAG is required}
FEATURE_ROOT=${FEATURE_ROOT:?FEATURE_ROOT is required}
REPORT_DIR=${PROJECT_ROOT}/reports/wan_p8_recovery/${RUN_TAG}
AEC2_NAME=${AEC2_NAME:-zoetrope}
JOB_NAME=${JOB_NAME:-bq_wanp8_full_gate_$(date -u +%m%d%H%M%S)}
CONTAINER_IMAGE=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617
STORAGE_MOUNT=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm
WORKER_SPEC=N4lS.Iq.I80.1

cd "${PROJECT_ROOT}"
for path in scripts/audit_wan_p8_exact_full.py scripts/verify_wan_p8_dataset_exact.py scripts/run_wan_p8_exact_full_gate_sco.sh; do
  [[ -f "${path}" ]] || { echo "missing ${path}" >&2; exit 2; }
done
for shard in 00 01 02 03 04 05; do
  python_status=${REPORT_DIR}/logs/shard_${shard}.status.json
  [[ -f "${python_status}" ]] || { echo "missing ${python_status}" >&2; exit 2; }
  grep -q '"rc":0' "${python_status}" || { echo "failed shard status ${python_status}" >&2; exit 2; }
done
[[ -d "${FEATURE_ROOT}/streaming" && -d "${FEATURE_ROOT}/target" ]] || { echo "missing feature roots" >&2; exit 2; }
[[ ! -e "${REPORT_DIR}/logs/full_gate.status.json" && ! -e "${REPORT_DIR}/full_gate_summary.json" ]] \
  || { echo "refusing existing gate outputs" >&2; exit 2; }

mkdir -p "${REPORT_DIR}/launcher"
command="cd ${PROJECT_ROOT} && PROJECT_ROOT=${PROJECT_ROOT} RUN_TAG=${RUN_TAG} FEATURE_ROOT=${FEATURE_ROOT} bash scripts/run_wan_p8_exact_full_gate_sco.sh"
printf '%q ' sco acp jobs create \
  --workspace-name=aigc --aec2-name="${AEC2_NAME}" --job-name="${JOB_NAME}" --priority=HIGHEST \
  --container-image-url="${CONTAINER_IMAGE}" --storage-mount="${STORAGE_MOUNT}" \
  --training-framework=pytorch --worker-nodes=1 --worker-spec="${WORKER_SPEC}" --command="${command}" \
  > "${REPORT_DIR}/launcher/full_gate_submit_command.txt"
printf '\n' >> "${REPORT_DIR}/launcher/full_gate_submit_command.txt"

sco acp jobs create \
  --workspace-name=aigc \
  --aec2-name="${AEC2_NAME}" \
  --job-name="${JOB_NAME}" \
  --priority=HIGHEST \
  --container-image-url="${CONTAINER_IMAGE}" \
  --storage-mount="${STORAGE_MOUNT}" \
  --training-framework=pytorch \
  --worker-nodes=1 \
  --worker-spec="${WORKER_SPEC}" \
  --command="${command}" \
  | tee "${REPORT_DIR}/launcher/full_gate_submit.log"
grep -oE 'pt-[a-z0-9]+' "${REPORT_DIR}/launcher/full_gate_submit.log" | head -n 1 \
  > "${REPORT_DIR}/launcher/full_gate_job_id.txt"
[[ -s "${REPORT_DIR}/launcher/full_gate_job_id.txt" ]] || { echo "missing gate job id" >&2; exit 3; }
printf 'JOB_ID=%s\nRUN_TAG=%s\n' "$(cat "${REPORT_DIR}/launcher/full_gate_job_id.txt")" "${RUN_TAG}"
