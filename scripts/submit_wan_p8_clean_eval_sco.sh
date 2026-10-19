#!/usr/bin/env bash
set -euo pipefail

export PATH="${HOME}/.sco/bin:${PATH}"
PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
RUN_TAG=${RUN_TAG:?RUN_TAG is required}
EVAL_RUN_TAG=${EVAL_RUN_TAG:-${RUN_TAG}_eval_staging}
FEATURE_ROOT=${FEATURE_ROOT:-/mnt/umm/users/yinbaiqiao/InsScene-15K/FEATURES_VERSIONED/${RUN_TAG}}
REPORT_DIR=${PROJECT_ROOT}/reports/wan_p8_recovery/${RUN_TAG}
PY=${PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}
LAYERS=(0 5 10 15 25 29)
CONTAINER_IMAGE=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617
STORAGE_MOUNT=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm

cd "${PROJECT_ROOT}"
"${PY}" - "${REPORT_DIR}/train_audit.json" <<'PY'
import json, sys
audit = json.load(open(sys.argv[1]))
assert audit.get('ok') is True
assert all(entry.get('status') == 'CLEAN_COMPLETE' for entry in audit.get('layers', {}).values())
print('ALL TRAINING CLEAN GATE PASS')
PY
[[ ! -e "${PROJECT_ROOT}/logs/inscene15k_streaming/runs/${EVAL_RUN_TAG}" ]] || {
  echo "refusing existing eval run tag: ${EVAL_RUN_TAG}" >&2
  exit 2
}
for layer in "${LAYERS[@]}"; do
  [[ ! -e "${REPORT_DIR}/logs/eval_layer${layer}.status.json" ]] || {
    echo "refusing existing eval status for layer ${layer}" >&2
    exit 2
  }
done

mkdir -p "${REPORT_DIR}/provenance/evaluation" "${REPORT_DIR}/launcher"
cp scripts/run_wan_p8_clean_eval_sco.sh "${REPORT_DIR}/provenance/evaluation/"
cp scripts/audit_wan_p8_clean_rerun.py "${REPORT_DIR}/provenance/evaluation/"
sha256sum \
  "${REPORT_DIR}/provenance/evaluation/run_wan_p8_clean_eval_sco.sh" \
  "${REPORT_DIR}/provenance/evaluation/audit_wan_p8_clean_rerun.py" \
  > "${REPORT_DIR}/provenance/evaluation/sha256.txt"

jobs="${REPORT_DIR}/launcher/eval_sco_job_ids.tsv"
[[ ! -e "${jobs}" ]] || { echo "refusing existing job record: ${jobs}" >&2; exit 2; }
printf 'layer\tjob_id\tjob_name\n' > "${jobs}"
for layer in "${LAYERS[@]}"; do
  job_name="bq_wanp8_eval_l${layer}_$(date -u +%m%d%H%M%S)"
  command="cd ${PROJECT_ROOT} && PROJECT_ROOT=${PROJECT_ROOT} RUN_TAG=${RUN_TAG} EVAL_RUN_TAG=${EVAL_RUN_TAG} FEATURE_ROOT=${FEATURE_ROOT} LAYER=${layer} bash scripts/run_wan_p8_clean_eval_sco.sh"
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
  printf '%s\t%s\t%s\n' "${layer}" "${job_id}" "${job_name}" >> "${jobs}"
  printf '%s\n' "${output}"
done

echo "EVAL SUBMISSION COMPLETE eval_run_tag=${EVAL_RUN_TAG}"
