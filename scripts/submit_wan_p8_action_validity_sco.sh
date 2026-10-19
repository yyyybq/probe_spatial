#!/usr/bin/env bash
set -euo pipefail

export PATH="${HOME}/.sco/bin:${PATH}"
PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
OUTPUT=${PROJECT_ROOT}/results/analysis/wan_p8_action_validity
CONTAINER_IMAGE=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617
STORAGE_MOUNT=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm
LAYERS=(0 5 10 15 25 29)
JOBS=${OUTPUT}/phase1_sco_jobs.tsv

[[ -f "${OUTPUT}/phase0_audit.json" ]] || { echo "missing Phase 0 audit" >&2; exit 2; }
[[ ! -e "${JOBS}" ]] || { echo "refusing existing job ledger" >&2; exit 2; }
printf 'layer\tjob_id\tjob_name\n' > "${JOBS}"
for layer in "${LAYERS[@]}"; do
  job_name="bq_wanp8_valid_l${layer}_$(date -u +%m%d%H%M%S)"
  command="cd ${PROJECT_ROOT} && PROJECT_ROOT=${PROJECT_ROOT} LAYER=${layer} bash scripts/run_wan_p8_action_validity_layer_sco.sh"
  output=$(sco acp jobs create \
    --workspace-name=aigc \
    --aec2-name=zoetrope \
    --job-name="${job_name}" \
    --priority=HIGHEST \
    --container-image-url="${CONTAINER_IMAGE}" \
    --storage-mount="${STORAGE_MOUNT}" \
    --training-framework=pytorch \
    --worker-nodes=1 \
    --worker-spec=N4lS.Iq.I80.1 \
    --command="${command}")
  job_id=$(sed -n 's/^job id : //p' <<<"${output}" | tail -n 1)
  [[ -n "${job_id}" ]] || { echo "failed to parse SCO job id: ${output}" >&2; exit 2; }
  printf '%s\t%s\t%s\n' "${layer}" "${job_id}" "${job_name}" >> "${JOBS}"
  printf '%s\n' "${output}"
done
echo "PHASE1 SUBMISSION COMPLETE"
