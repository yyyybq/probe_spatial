#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/mnt/umm/users/yinbaiqiao/probe_spatial
RESULT=${PROJECT_ROOT}/results/experimental/se3_wan_vjepa2_exact_v2_layers_20260916_r1
SCO=/mnt/umm/users/yinbaiqiao/.sco/bin/sco
mkdir -p "${RESULT}/launcher"

job_name=bq-se3-v2-reps-$(date -u +%m%d%H%M%S)
command="cd ${PROJECT_ROOT} && bash scripts/run_se3_wan_vjepa2_v2_representative_audit.sh"
output=$("${SCO}" acp jobs create \
  --workspace-name=aigc \
  --aec2-name=zoetrope \
  --job-name="${job_name}" \
  --priority=HIGHEST \
  --quota-type=reserved \
  --container-image-url=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617 \
  --storage-mount=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm \
  --training-framework=pytorch \
  --worker-nodes=1 \
  --worker-spec=N4lS.Iq.I80.1 \
  --command="${command}")
printf '%s\n' "${output}" | tee "${RESULT}/launcher/representative_submit.log"
job_id=$(printf '%s\n' "${output}" | grep -oE 'pt-[a-z0-9]+' | head -n 1)
test -n "${job_id}"
printf 'job_id\tjob_name\n%s\t%s\n' "${job_id}" "${job_name}" > "${RESULT}/launcher/representative_job.tsv"
