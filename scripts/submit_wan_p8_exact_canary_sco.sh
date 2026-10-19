#!/usr/bin/env bash
set -euo pipefail

export PATH="${HOME}/.sco/bin:${PATH}"

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
PY=${PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}
CANARY_TAG=${CANARY_TAG:-wan_p8_manifest_exact_canary_$(date -u +%Y%m%d_%H%M%S)_$(git -C "${PROJECT_ROOT}" rev-parse --short HEAD)}
FEATURE_ROOT=${FEATURE_ROOT:-/mnt/umm/users/yinbaiqiao/InsScene-15K/FEATURES_VERSIONED/${CANARY_TAG}}
REPORT_DIR=${PROJECT_ROOT}/reports/wan_p8_recovery/${CANARY_TAG}
MANIFEST=${PROJECT_ROOT}/manifests/c_retrain_20260722/inscene_c_h124_prefix8.jsonl
SELECTION=${PROJECT_ROOT}/reports/wan_p8_recovery/canary_selection_20260821_185059.txt
MODEL=/mnt/umm/users/yinbaiqiao/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780a534b6463e45facd96134c9f345acfa5b
AEC2_NAME=${AEC2_NAME:-zoetrope}
JOB_NAME=${JOB_NAME:-bq_wan_p8_exact_canary_$(date -u +%m%d%H%M%S)}
CONTAINER_IMAGE=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617
STORAGE_MOUNT=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm
WORKER_SPEC=N4lS.Iq.I80.1

cd "${PROJECT_ROOT}"

required=(
  scripts/extract_wan_manifest_exact.py
  scripts/verify_wan_p8_dataset_exact.py
  scripts/audit_wan_p8_exact_canary.py
  scripts/run_wan_p8_exact_canary_sco.sh
  "${MANIFEST}"
  "${SELECTION}"
)
for path in "${required[@]}"; do
  [[ -f "${path}" ]] || { echo "missing required file: ${path}" >&2; exit 2; }
done

actual_manifest_sha=$(sha256sum "${MANIFEST}" | awk '{print $1}')
[[ "${actual_manifest_sha}" == "772194f077c45cc85a942f17f46f50103125ae8f5b1c983c08881c1759607e0c" ]] \
  || { echo "canonical manifest hash mismatch: ${actual_manifest_sha}" >&2; exit 2; }
[[ ! -e "${FEATURE_ROOT}" ]] || { echo "refusing existing FEATURE_ROOT=${FEATURE_ROOT}" >&2; exit 2; }
[[ ! -e "${REPORT_DIR}" ]] || { echo "refusing existing REPORT_DIR=${REPORT_DIR}" >&2; exit 2; }

"${PY}" scripts/extract_wan_manifest_exact.py \
  --mode streaming \
  --manifest "${MANIFEST}" \
  --selection-file "${SELECTION}" \
  --data-root /mnt/umm/users/yinbaiqiao/InsScene-15K/data \
  --out-root /tmp/wan_p8_submit_preflight \
  --model-id "${MODEL}" \
  --layers 0 5 10 15 25 29 \
  --target-horizon 1 \
  --seed 42 \
  --validate-only

mkdir -p "${REPORT_DIR}/launcher"
git -C "${PROJECT_ROOT}" rev-parse HEAD > "${REPORT_DIR}/launcher/git_commit.txt"
git -C "${PROJECT_ROOT}" status --short > "${REPORT_DIR}/launcher/git_status_short.txt"
git -C "${PROJECT_ROOT}" diff --binary HEAD > "${REPORT_DIR}/launcher/git_dirty.diff"
command="cd ${PROJECT_ROOT} && PROJECT_ROOT=${PROJECT_ROOT} CANARY_TAG=${CANARY_TAG} FEATURE_ROOT=${FEATURE_ROOT} SCO_JOB_NAME=${JOB_NAME} bash scripts/run_wan_p8_exact_canary_sco.sh"
submit_log=${REPORT_DIR}/launcher/sco_submit.log
submit_command=${REPORT_DIR}/launcher/submit_command.txt
job_ids_file=${REPORT_DIR}/launcher/sco_job_ids.txt

printf 'CANARY_TAG=%s\nFEATURE_ROOT=%s\nREPORT_DIR=%s\nJOB_NAME=%s\nAEC2_NAME=%s\nWORKER_SPEC=%s\nCONTAINER_IMAGE=%s\nSTORAGE_MOUNT=%s\nCOMMAND=%s\n' \
  "${CANARY_TAG}" "${FEATURE_ROOT}" "${REPORT_DIR}" "${JOB_NAME}" "${AEC2_NAME}" \
  "${WORKER_SPEC}" "${CONTAINER_IMAGE}" "${STORAGE_MOUNT}" "${command}" \
  > "${REPORT_DIR}/launcher/submission_plan.txt"

printf '%q ' sco acp jobs create \
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
  > "${submit_command}"
printf '\n' >> "${submit_command}"

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
  | tee "${submit_log}"

grep -oE 'pt-[a-z0-9]+' "${submit_log}" | awk '!seen[$0]++' > "${job_ids_file}" || true
job_id=$(head -n 1 "${job_ids_file}")
[[ -n "${job_id}" ]] || { echo "SCO submission returned no job ID" >&2; exit 3; }

printf 'JOB_ID=%s\nCANARY_TAG=%s\nFEATURE_ROOT=%s\nREPORT_DIR=%s\n' \
  "${job_id}" "${CANARY_TAG}" "${FEATURE_ROOT}" "${REPORT_DIR}"
