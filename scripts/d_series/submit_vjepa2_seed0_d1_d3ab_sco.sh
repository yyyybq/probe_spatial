#!/usr/bin/env bash
set -euo pipefail

export PATH="${HOME}/.sco/bin:${PATH}"
PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
AEC2_NAME=${AEC2_NAME:-h800}
FEATURE_RUN=${FEATURE_RUN:?FEATURE_RUN is required}
FEATURE_ROOT=${FEATURE_ROOT:?FEATURE_ROOT is required}
RUN_TAG=${RUN_TAG:-dseries_vjepa2_seed0_$(date -u +%Y%m%d_%H%M%S)}
OUTPUT_ROOT=${OUTPUT_ROOT:-${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/diagnostic_runs/${RUN_TAG}}
FEATURE_REPORT=${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/runs/${FEATURE_RUN}/vjepa2_vitl256
FEATURE_AUDIT=${FEATURE_REPORT}/feature_audit.json
FEATURE_AUDIT_RECOVERY=${FEATURE_REPORT}/feature_audit_recovery_status.json
REPORT=${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/runs/${RUN_TAG}/launcher
PY=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python
cd "${PROJECT_ROOT}"

"${PY}" - "${FEATURE_REPORT}/status.json" "${FEATURE_AUDIT}" "${FEATURE_AUDIT_RECOVERY}" <<'PY'
import json, sys
status, audit = (json.load(open(path)) for path in sys.argv[1:3])
if status["failed"] != 0:
    recovery = json.load(open(sys.argv[3]))
    assert recovery["status"] == "RECOVERED_COMPLETE_VALID", recovery
    assert recovery["original_status_failed"] == 1, recovery
assert audit["status"] == "COMPLETE_VALID", audit
assert audit["resolved"] == audit["expected"], audit
for role, expected in {"prefix": 2721, "isolated_target": 5442, "native_target": 5442, "native_pred": 5442}.items():
    assert audit["roles"][role] == {"complete": expected, "expected": expected}
PY
test ! -e "${OUTPUT_ROOT}" || { echo "refusing existing output ${OUTPUT_ROOT}" >&2; exit 2; }
mkdir -p "${REPORT}"
PROJECT_ROOT="${PROJECT_ROOT}" PROVENANCE_DIR="${REPORT}" bash scripts/d_series/capture_source_provenance.sh
base_commit=$(cat "${REPORT}/base_commit.txt"); git_dirty=$(cat "${REPORT}/git_dirty.txt")
job_name="bq_dseries_vjepa2_s0_$(date -u +%m%d%H%M%S)"
command="cd ${PROJECT_ROOT} && PROJECT_ROOT=${PROJECT_ROOT} RUN_TAG=${RUN_TAG} FEATURE_ROOT=${FEATURE_ROOT} FEATURE_AUDIT=${FEATURE_AUDIT} OUTPUT_ROOT=${OUTPUT_ROOT} D_SERIES_BASE_COMMIT=${base_commit} D_SERIES_GIT_DIRTY=${git_dirty} bash scripts/d_series/run_vjepa2_seed0_d1_d3ab_sco.sh"
printf '%s\n' "${command}" > "${REPORT}/command.txt"; printf '%s\n' "${job_name}" > "${REPORT}/job_name.txt"
sco acp jobs create --workspace-name=aigc --aec2-name="${AEC2_NAME}" --job-name="${job_name}" --priority=HIGHEST \
  --container-image-url=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617 \
  --storage-mount=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm --training-framework=pytorch \
  --worker-nodes=1 --worker-spec=N4lS.Iq.I80.4 --command="${command}" | tee "${REPORT}/sco_submit.log"
grep -oE 'pt-[a-z0-9]+' "${REPORT}/sco_submit.log" | awk '!seen[$0]++' > "${REPORT}/job_ids.txt"
test -s "${REPORT}/job_ids.txt"
printf 'RUN_TAG=%s\nJOB_ID=%s\nOUTPUT_ROOT=%s\n' "${RUN_TAG}" "$(head -1 "${REPORT}/job_ids.txt")" "${OUTPUT_ROOT}"
