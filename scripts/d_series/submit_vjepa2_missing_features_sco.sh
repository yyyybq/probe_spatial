#!/usr/bin/env bash
set -euo pipefail

export PATH="${HOME}/.sco/bin:${PATH}"
PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
CANARY_RUN=${CANARY_RUN:?CANARY_RUN must name the passing V-JEPA2 canary}
RUN_TAG=${RUN_TAG:-dseries_vjepa2_missing_features_$(date -u +%Y%m%d_%H%M%S)}
FEATURE_ROOT=${FEATURE_ROOT:-/mnt/umm/users/yinbaiqiao/InsScene-15K/D_SERIES_SCANNETPP_V1_VJEPA2_FEATURES_R1}
AEC2_NAME=${AEC2_NAME:-h800}
REPORT=${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/runs/${RUN_TAG}/launcher
CANARY=${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/runs/${CANARY_RUN}
CANARY_FEATURES=${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/features_canary/${CANARY_RUN}
PY=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python
cd "${PROJECT_ROOT}"

"${PY}" - "${CANARY}" "${CANARY_FEATURES}" <<'PY'
import json, sys
from pathlib import Path
report, features = Path(sys.argv[1]), Path(sys.argv[2])
status = json.loads((report / "status.json").read_text())
assert status["vjepa2_v2_rc"] == status["vjepa2_native_rc"] == 0, status
assert json.loads((report / "vjepa2_v2_pooled_canary.json").read_text())["passed"] is True
assert json.loads((report / "vjepa2_native_spatial_overfit.json").read_text())["passed"] is True
audit = json.loads((features / "vjepa2_audit_0_1.json").read_text())
assert audit["windows_requested"] == audit["windows_completed"] == 1
assert audit["local_files_only"] is True
assert audit["horizon_token_mapping"]["distinct"] is True
inv = audit["future_invariance"]
assert inv["variants"] == ["real_future", "zeros", "noise", "other_scene"]
assert inv["prediction_cache_used"] is False and inv["independent_predictor_calls"] == 4
assert max(inv["max_abs_difference"].values()) <= 1e-5
PY
test ! -e "${FEATURE_ROOT}" || { echo "refusing existing feature root ${FEATURE_ROOT}" >&2; exit 2; }
mkdir -p "${REPORT}"
PROJECT_ROOT="${PROJECT_ROOT}" PROVENANCE_DIR="${REPORT}" bash scripts/d_series/capture_source_provenance.sh
base_commit=$(cat "${REPORT}/base_commit.txt")
git_dirty=$(cat "${REPORT}/git_dirty.txt")
job_name="bq_dseries_vjepa2_fill_$(date -u +%m%d%H%M%S)"
command="cd ${PROJECT_ROOT} && PROJECT_ROOT=${PROJECT_ROOT} RUN_TAG=${RUN_TAG} FEATURE_ROOT=${FEATURE_ROOT} NPROC=8 D_SERIES_BASE_COMMIT=${base_commit} D_SERIES_GIT_DIRTY=${git_dirty} bash scripts/d_series/run_vjepa2_missing_features_sco.sh"
printf '%s\n' "${command}" > "${REPORT}/command.txt"
printf '%s\n' "${job_name}" > "${REPORT}/job_name.txt"
sco acp jobs create --workspace-name=aigc --aec2-name="${AEC2_NAME}" --job-name="${job_name}" --priority=HIGHEST \
  --container-image-url=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617 \
  --storage-mount=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm --training-framework=pytorch \
  --worker-nodes=1 --worker-spec=N4lS.Iq.I80.8 --command="${command}" | tee "${REPORT}/sco_submit.log"
grep -oE 'pt-[a-z0-9]+' "${REPORT}/sco_submit.log" | awk '!seen[$0]++' > "${REPORT}/job_ids.txt"
test -s "${REPORT}/job_ids.txt"
printf 'RUN_TAG=%s\nJOB_ID=%s\nFEATURE_ROOT=%s\nAEC2_NAME=%s\nPRIORITY=HIGHEST\n' \
  "${RUN_TAG}" "$(head -1 "${REPORT}/job_ids.txt")" "${FEATURE_ROOT}" "${AEC2_NAME}"
