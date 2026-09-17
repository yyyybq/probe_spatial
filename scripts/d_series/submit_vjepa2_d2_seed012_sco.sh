#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
SCO=/mnt/umm/users/yinbaiqiao/.sco/bin/sco
AEC2_NAME=${AEC2_NAME:-zoetrope}
FEATURE_RUN=${FEATURE_RUN:?FEATURE_RUN is required}
FEATURE_ROOT=${FEATURE_ROOT:?FEATURE_ROOT is required}
D1_RUN_ROOT_S0=${D1_RUN_ROOT_S0:?D1_RUN_ROOT_S0 is required}
D1_RUN_ROOT_S1=${D1_RUN_ROOT_S1:?D1_RUN_ROOT_S1 is required}
D1_RUN_ROOT_S2=${D1_RUN_ROOT_S2:?D1_RUN_ROOT_S2 is required}
CAMPAIGN_TAG=${CAMPAIGN_TAG:-$(date -u +%Y%m%d_%H%M%S)_zoetrope_r1}
FEATURE_REPORT=${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/runs/${FEATURE_RUN}/vjepa2_vitl256
FEATURE_AUDIT=${FEATURE_REPORT}/feature_audit.json
PY=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python
cd "${PROJECT_ROOT}"

"${PY}" - "${FEATURE_AUDIT}" <<'PY'
import json, sys
audit = json.load(open(sys.argv[1]))
assert audit["status"] == "COMPLETE_VALID", audit
assert audit["manifest_sha256"] == "781e25848de48fa5219aad876bd10cb8f9445328b145a19193a6f6b2ab5efcc9", audit
assert audit["model"] == "vjepa2_vitl256", audit
assert audit["roles"]["isolated_target"] == {"complete": 5442, "expected": 5442}, audit
assert audit["roles"]["prefix"] == {"complete": 2721, "expected": 2721}, audit
PY

for seed in 0 1 2; do
  root_name="D1_RUN_ROOT_S${seed}"
  d1_root=${!root_name}
  d1_checkpoint="${d1_root}/d1_isolated_pooled_s${seed}/checkpoint.pt"
  "${PY}" - "${d1_checkpoint}" "${seed}" <<'PY'
import json, pathlib, sys
checkpoint, seed = pathlib.Path(sys.argv[1]), int(sys.argv[2])
receipt = json.loads((checkpoint.parent / "run_receipt.json").read_text())
assert checkpoint.is_file() and checkpoint.stat().st_size > 0
expected = {"status": "complete_development_evaluation", "experiment": "D1", "model": "vjepa2_vitl256",
            "reference": "isolated_target", "branch": "pooled", "seed": seed,
            "manifest_sha256": "781e25848de48fa5219aad876bd10cb8f9445328b145a19193a6f6b2ab5efcc9"}
assert all(receipt.get(key) == value for key, value in expected.items()), receipt
PY
  run_tag="dseries_vjepa2_d2_seed${seed}_${CAMPAIGN_TAG}"
  output_dir="${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/diagnostic_runs/${run_tag}/d2_vjepa2_pooled_s${seed}"
  report="${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/runs/${run_tag}/launcher"
  mkdir -p "${report}"
  PROJECT_ROOT="${PROJECT_ROOT}" PROVENANCE_DIR="${report}" bash scripts/d_series/capture_source_provenance.sh
  base_commit=$(cat "${report}/base_commit.txt")
  git_dirty=$(cat "${report}/git_dirty.txt")
  job_name="bq-ds-vj-d2-s${seed}-$(date -u +%m%d%H%M%S)"
  command="cd ${PROJECT_ROOT} && PROJECT_ROOT=${PROJECT_ROOT} RUN_TAG=${run_tag} SEED=${seed} FEATURE_ROOT=${FEATURE_ROOT} FEATURE_AUDIT=${FEATURE_AUDIT} D1_CHECKPOINT=${d1_checkpoint} OUTPUT_DIR=${output_dir} D_SERIES_BASE_COMMIT=${base_commit} D_SERIES_GIT_DIRTY=${git_dirty} bash scripts/d_series/run_vjepa2_d2_seed_sco.sh"
  printf '%s\n' "${command}" > "${report}/command.txt"
  printf '%s\n' "${job_name}" > "${report}/job_name.txt"
  "${SCO}" acp jobs create --workspace-name=aigc --aec2-name="${AEC2_NAME}" \
    --job-name="${job_name}" --priority=HIGHEST \
    --container-image-url=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617 \
    --storage-mount=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm --training-framework=pytorch \
    --worker-nodes=1 --worker-spec=N4lS.Iq.I80.1 --command="${command}" | tee "${report}/sco_submit.log"
  grep -oE 'pt-[a-z0-9]+' "${report}/sco_submit.log" | awk '!seen[$0]++' > "${report}/job_ids.txt"
  test -s "${report}/job_ids.txt"
  printf 'RUN_TAG=%s\nSEED=%s\nJOB_ID=%s\nGPU_COUNT=1\nOUTPUT_DIR=%s\nLOG_ROOT=%s\n' \
    "${run_tag}" "${seed}" "$(head -1 "${report}/job_ids.txt")" "${output_dir}" \
    "${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/runs/${run_tag}/seed${seed}_d2"
done
