#!/usr/bin/env bash
set -euo pipefail

export PATH="${HOME}/.sco/bin:${PATH}"
PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
CANARY_RUN=${CANARY_RUN:?CANARY_RUN must name a passing canary run directory}
RUN_TAG=${RUN_TAG:-dseries_features_full_$(date -u +%Y%m%d_%H%M%S)}
FEATURE_ROOT=${FEATURE_ROOT:-/mnt/umm/users/yinbaiqiao/InsScene-15K/D_SERIES_SCANNETPP_V1_FEATURES}
CANARY_STATUS=${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/runs/${CANARY_RUN}/status.json
CANARY_REPORT=${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/runs/${CANARY_RUN}
CANARY_FEATURES=${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/features_canary/${CANARY_RUN}
REPORT=${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/runs/${RUN_TAG}/launcher
PY=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python
cd "${PROJECT_ROOT}"

"${PY}" - "${CANARY_STATUS}" "${CANARY_REPORT}" "${CANARY_FEATURES}" <<'PY'
import json, sys
from pathlib import Path
value = json.load(open(sys.argv[1]))
assert value["vjepa2_rc"] == value["dinov2_rc"] == value["wan_rc"] == 0, value
report, features = Path(sys.argv[2]), Path(sys.argv[3])
for name in ("vjepa2_native_spatial_overfit.json", "dinov2_spatial_overfit.json", "wan_spatial_overfit.json"):
    result = json.load(open(report / name)); assert result["passed"], result
for name in ("vjepa2_audit_0_1.json", "dinov2_audit_0_1.json", "wan_audit_0_1.json"):
    audit = json.load(open(features / name))
    assert audit["windows_requested"] == audit["windows_completed"] == 1, audit
vjepa = json.load(open(features / "vjepa2_audit_0_1.json"))["future_invariance"]
assert vjepa["variants"] == ["real_future", "zeros", "noise", "other_scene"], vjepa
assert max(vjepa["max_abs_difference"].values()) <= 1e-5, vjepa
PY
test "$(sha256sum experiments/d_series_scannetpp_v1/data/sample_manifest.jsonl | awk '{print $1}')" = \
  781e25848de48fa5219aad876bd10cb8f9445328b145a19193a6f6b2ab5efcc9
mkdir -p "${REPORT}"
: > "${REPORT}/sco_submit.log"
for model in vjepa2_vitl256 dinov2_vitl14 wan21_t2v_1p3b; do
  job_name="bq_dseries_${model}_$(date -u +%m%d%H%M%S)"
  command="cd ${PROJECT_ROOT} && PROJECT_ROOT=${PROJECT_ROOT} RUN_TAG=${RUN_TAG} MODEL=${model} FEATURE_ROOT=${FEATURE_ROOT} NPROC=8 bash scripts/d_series/run_feature_shards_sco.sh"
  printf '%s\t%s\n' "${model}" "${command}" >> "${REPORT}/commands.tsv"
  sco acp jobs create --workspace-name=aigc --aec2-name=zoetrope --job-name="${job_name}" --priority=HIGHEST \
    --container-image-url=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617 \
    --storage-mount=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm --training-framework=pytorch \
    --worker-nodes=1 --worker-spec=N4lS.Iq.I80.8 --command="${command}" | tee -a "${REPORT}/sco_submit.log"
done
grep -oE 'pt-[a-z0-9]+' "${REPORT}/sco_submit.log" | awk '!seen[$0]++' > "${REPORT}/job_ids.txt"
test "$(wc -l < "${REPORT}/job_ids.txt")" -eq 3
printf 'RUN_TAG=%s\nFEATURE_ROOT=%s\nJOB_IDS=%s\n' "${RUN_TAG}" "${FEATURE_ROOT}" "$(tr '\n' ' ' < "${REPORT}/job_ids.txt")"
