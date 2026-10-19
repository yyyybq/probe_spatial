#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
SCO=/mnt/umm/users/yinbaiqiao/.sco/bin/sco
AEC2_NAME=${AEC2_NAME:-h800}
FEATURE_ROOT=${FEATURE_ROOT:-/mnt/umm/users/yinbaiqiao/InsScene-15K/D_SERIES_SCANNETPP_V1_WAN_FEATURES_R1}
FEATURE_AUDIT=${FEATURE_AUDIT:?FEATURE_AUDIT is required}
CAMPAIGN_TAG=${CAMPAIGN_TAG:-$(date -u +%Y%m%d_%H%M%S)_h800_r1}
PY=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python
cd "${PROJECT_ROOT}"

"${PY}" - "${FEATURE_AUDIT}" <<'PY'
import json, pathlib, sys
audit = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert audit["status"] == "COMPLETE_VALID", audit
assert audit["model"] == "wan21_t2v_1p3b", audit
assert audit["manifest_sha256"] == "781e25848de48fa5219aad876bd10cb8f9445328b145a19193a6f6b2ab5efcc9", audit
assert audit["roles"] == {"prefix": {"expected": 2721, "complete": 2721},
                           "isolated_target": {"expected": 5442, "complete": 5442}}, audit
PY

for seed in 0 1 2; do
  run_tag="dseries_wan_seed${seed}_${CAMPAIGN_TAG}"
  output_root="${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/diagnostic_runs/${run_tag}"
  report="${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/runs/${run_tag}/launcher"
  [[ ! -e "${output_root}" ]] || { echo "refusing existing output root: ${output_root}" >&2; exit 2; }
  [[ ! -e "${report}/job_ids.txt" ]] || { echo "refusing duplicate seed submission: ${report}" >&2; exit 2; }
  mkdir -p "${report}"
  PROJECT_ROOT="${PROJECT_ROOT}" PROVENANCE_DIR="${report}" bash scripts/d_series/capture_source_provenance.sh
  base_commit=$(cat "${report}/base_commit.txt")
  git_dirty=$(cat "${report}/git_dirty.txt")
  job_name="bq-ds-wan-s${seed}-$(date -u +%m%d%H%M%S)"
  command="cd ${PROJECT_ROOT} && PROJECT_ROOT=${PROJECT_ROOT} RUN_TAG=${run_tag} SEED=${seed} FEATURE_ROOT=${FEATURE_ROOT} FEATURE_AUDIT=${FEATURE_AUDIT} OUTPUT_ROOT=${output_root} D_SERIES_BASE_COMMIT=${base_commit} D_SERIES_GIT_DIRTY=${git_dirty} bash scripts/d_series/run_wan_seed_serial_sco.sh"
  printf '%s\n' "${command}" > "${report}/command.txt"
  printf '%s\n' "${job_name}" > "${report}/job_name.txt"
  "${SCO}" acp jobs create --workspace-name=aigc --aec2-name="${AEC2_NAME}" --job-name="${job_name}" --priority=HIGHEST \
    --container-image-url=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617 \
    --storage-mount=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm --training-framework=pytorch \
    --worker-nodes=1 --worker-spec=N4lS.Iq.I80.1 --command="${command}" | tee "${report}/sco_submit.log"
  grep -oE 'pt-[a-z0-9]+' "${report}/sco_submit.log" | awk '!seen[$0]++' > "${report}/job_ids.txt"
  test -s "${report}/job_ids.txt"
  printf 'RUN_TAG=%s\nSEED=%s\nJOB_ID=%s\nGPU_COUNT=1\nOUTPUT_ROOT=%s\nLOG_ROOT=%s\n' \
    "${run_tag}" "${seed}" "$(head -1 "${report}/job_ids.txt")" "${output_root}" \
    "${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/runs/${run_tag}/seed${seed}_serial"
done
