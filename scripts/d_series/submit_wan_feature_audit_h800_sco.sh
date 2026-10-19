#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
SCO=/mnt/umm/users/yinbaiqiao/.sco/bin/sco
RUN_TAG=${RUN_TAG:-dseries_wan_feature_audit_$(date -u +%Y%m%d_%H%M%S)_h800_r1}
FEATURE_ROOT=${FEATURE_ROOT:-/mnt/umm/users/yinbaiqiao/InsScene-15K/D_SERIES_SCANNETPP_V1_WAN_FEATURES_R1}
REPORT=${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/runs/${RUN_TAG}/launcher
cd "${PROJECT_ROOT}"

[[ ! -e "${REPORT}/job_ids.txt" ]] || { echo "refusing duplicate audit submission: ${REPORT}" >&2; exit 2; }
mkdir -p "${REPORT}"
PROJECT_ROOT="${PROJECT_ROOT}" PROVENANCE_DIR="${REPORT}" bash scripts/d_series/capture_source_provenance.sh
base_commit=$(cat "${REPORT}/base_commit.txt")
git_dirty=$(cat "${REPORT}/git_dirty.txt")
job_name="bq-ds-wan-audit-$(date -u +%m%d%H%M%S)"
command="cd ${PROJECT_ROOT} && PROJECT_ROOT=${PROJECT_ROOT} RUN_TAG=${RUN_TAG} FEATURE_ROOT=${FEATURE_ROOT} D_SERIES_BASE_COMMIT=${base_commit} D_SERIES_GIT_DIRTY=${git_dirty} bash scripts/d_series/run_wan_feature_audit_sco.sh"
printf '%s\n' "${command}" > "${REPORT}/command.txt"
printf '%s\n' "${job_name}" > "${REPORT}/job_name.txt"
"${SCO}" acp jobs create --workspace-name=aigc --aec2-name=h800 --job-name="${job_name}" --priority=HIGHEST \
  --container-image-url=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617 \
  --storage-mount=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm --training-framework=pytorch \
  --worker-nodes=1 --worker-spec=N4lS.Iq.I80.1 --command="${command}" | tee "${REPORT}/sco_submit.log"
grep -oE 'pt-[a-z0-9]+' "${REPORT}/sco_submit.log" | awk '!seen[$0]++' > "${REPORT}/job_ids.txt"
test -s "${REPORT}/job_ids.txt"
printf 'RUN_TAG=%s\nJOB_ID=%s\nGPU_COUNT=1\nAUDIT=%s\nLOG_ROOT=%s\n' "${RUN_TAG}" "$(head -1 "${REPORT}/job_ids.txt")" \
  "${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/runs/${RUN_TAG}/wan21_t2v_1p3b/feature_audit.json" \
  "${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/runs/${RUN_TAG}/wan21_t2v_1p3b"
