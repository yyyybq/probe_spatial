#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
SCO=/mnt/umm/users/yinbaiqiao/.sco/bin/sco
AEC2_NAME=${AEC2_NAME:-zoetrope}
FEATURE_RUN=${FEATURE_RUN:?FEATURE_RUN is required}
FEATURE_ROOT=${FEATURE_ROOT:?FEATURE_ROOT is required}
CAMPAIGN_TAG=${CAMPAIGN_TAG:-$(date -u +%Y%m%d_%H%M%S)_zoetrope_r1}
FEATURE_REPORT=${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/runs/${FEATURE_RUN}/vjepa2_vitl256
FEATURE_AUDIT=${FEATURE_REPORT}/feature_audit.json
FEATURE_AUDIT_RECOVERY=${FEATURE_REPORT}/feature_audit_recovery_status.json
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

for seed in 1 2; do
  run_tag="dseries_vjepa2_seed${seed}_${CAMPAIGN_TAG}"
  output_root="${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/diagnostic_runs/${run_tag}"
  report="${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/runs/${run_tag}/launcher"
  mkdir -p "${report}"
  PROJECT_ROOT="${PROJECT_ROOT}" PROVENANCE_DIR="${report}" bash scripts/d_series/capture_source_provenance.sh
  base_commit=$(cat "${report}/base_commit.txt")
  git_dirty=$(cat "${report}/git_dirty.txt")
  job_name="bq-ds-vj-s${seed}-$(date -u +%m%d%H%M%S)"
  command="cd ${PROJECT_ROOT} && PROJECT_ROOT=${PROJECT_ROOT} RUN_TAG=${run_tag} SEED=${seed} FEATURE_ROOT=${FEATURE_ROOT} FEATURE_AUDIT=${FEATURE_AUDIT} OUTPUT_ROOT=${output_root} D_SERIES_BASE_COMMIT=${base_commit} D_SERIES_GIT_DIRTY=${git_dirty} bash scripts/d_series/run_vjepa2_seed_serial_sco.sh"
  printf '%s\n' "${command}" > "${report}/command.txt"
  printf '%s\n' "${job_name}" > "${report}/job_name.txt"
  "${SCO}" acp jobs create --workspace-name=aigc --aec2-name="${AEC2_NAME}" \
    --job-name="${job_name}" --priority=HIGHEST \
    --container-image-url=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617 \
    --storage-mount=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm --training-framework=pytorch \
    --worker-nodes=1 --worker-spec=N4lS.Iq.I80.1 --command="${command}" | tee "${report}/sco_submit.log"
  grep -oE 'pt-[a-z0-9]+' "${report}/sco_submit.log" | awk '!seen[$0]++' > "${report}/job_ids.txt"
  test -s "${report}/job_ids.txt"
  printf 'RUN_TAG=%s\nSEED=%s\nJOB_ID=%s\nGPU_COUNT=1\nOUTPUT_ROOT=%s\nLOG_ROOT=%s\n' \
    "${run_tag}" "${seed}" "$(head -1 "${report}/job_ids.txt")" "${output_root}" \
    "${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/runs/${run_tag}/seed${seed}_serial"
done
