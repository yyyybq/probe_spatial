#!/usr/bin/env bash
set -euo pipefail

export PATH="${HOME}/.sco/bin:${PATH}"
PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
OUTPUT=${PROJECT_ROOT}/results/analysis/wan_p8_action_validity
CONTAINER_IMAGE=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617
STORAGE_MOUNT=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm
LEDGER=${OUTPUT}/phase1_8gpu_sco_job.tsv

[[ ! -e "${LEDGER}" ]] || { echo "refusing existing 8-GPU ledger" >&2; exit 2; }
job_name="bq_wanp8_valid_8gpu_$(date -u +%m%d%H%M%S)"
command="cd ${PROJECT_ROOT} && PROJECT_ROOT=${PROJECT_ROOT} bash scripts/run_wan_p8_action_validity_8gpu_sco.sh"
output=$(sco acp jobs create \
  --workspace-name=aigc \
  --aec2-name=zoetrope \
  --job-name="${job_name}" \
  --priority=HIGHEST \
  --container-image-url="${CONTAINER_IMAGE}" \
  --storage-mount="${STORAGE_MOUNT}" \
  --training-framework=pytorch \
  --worker-nodes=1 \
  --worker-spec=N4lS.Iq.I80.8 \
  --command="${command}")
job_id=$(sed -n 's/^job id : //p' <<<"${output}" | tail -n 1)
[[ -n "${job_id}" ]] || { echo "failed to parse job id: ${output}" >&2; exit 2; }
printf 'job_id\tjob_name\n%s\t%s\n' "${job_id}" "${job_name}" > "${LEDGER}"
printf '%s\n' "${output}"
