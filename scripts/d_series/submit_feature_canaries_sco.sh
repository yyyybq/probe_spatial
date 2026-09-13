#!/usr/bin/env bash
set -euo pipefail

export PATH="${HOME}/.sco/bin:${PATH}"
PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
RUN_TAG=${RUN_TAG:-dseries_feature_canary_$(date -u +%Y%m%d_%H%M%S)}
JOB_NAME=${JOB_NAME:-bq_dseries_canary_$(date -u +%m%d%H%M%S)}
REPORT=${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/runs/${RUN_TAG}
mkdir -p "${REPORT}/launcher"

cd "${PROJECT_ROOT}"
test "$(sha256sum experiments/d_series_scannetpp_v1/data/sample_manifest.jsonl | awk '{print $1}')" = \
  781e25848de48fa5219aad876bd10cb8f9445328b145a19193a6f6b2ab5efcc9
for path in scripts/d_series/extract_vjepa2.py scripts/d_series/extract_dinov2.py \
  scripts/d_series/extract_wan.py scripts/d_series/run_feature_canaries_sco.sh; do
  test -f "${path}"
done

command="cd ${PROJECT_ROOT} && PROJECT_ROOT=${PROJECT_ROOT} RUN_TAG=${RUN_TAG} bash scripts/d_series/run_feature_canaries_sco.sh"
printf '%s\n' "${command}" > "${REPORT}/launcher/command.txt"
printf '%s\n' "${JOB_NAME}" > "${REPORT}/launcher/job_name.txt"
sco acp jobs create \
  --workspace-name=aigc \
  --aec2-name=zoetrope \
  --job-name="${JOB_NAME}" \
  --priority=HIGHEST \
  --container-image-url=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617 \
  --storage-mount=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm \
  --training-framework=pytorch \
  --worker-nodes=1 \
  --worker-spec=N4lS.Iq.I80.4 \
  --command="${command}" | tee "${REPORT}/launcher/sco_submit.log"
grep -oE 'pt-[a-z0-9]+' "${REPORT}/launcher/sco_submit.log" | awk '!seen[$0]++' > "${REPORT}/launcher/job_ids.txt"
job_id=$(head -1 "${REPORT}/launcher/job_ids.txt")
test -n "${job_id}"
printf 'RUN_TAG=%s\nJOB_ID=%s\nREPORT=%s\n' "${RUN_TAG}" "${job_id}" "${REPORT}"
