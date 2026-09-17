#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
SCO=/mnt/umm/users/yinbaiqiao/.sco/bin/sco
AEC2_NAME=${AEC2_NAME:-zoetrope}
RUN_TAG=${RUN_TAG:?RUN_TAG is required}
AUDIT_DIR=${AUDIT_DIR:?AUDIT_DIR is required}
FEATURE_ROOT=${FEATURE_ROOT:?FEATURE_ROOT is required}
NUM_SHARDS=${NUM_SHARDS:-6}
PY=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python
REPORT=${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/runs/${RUN_TAG}
LAUNCHER=${REPORT}/launcher
cd "${PROJECT_ROOT}"

test -f "${AUDIT_DIR}/audit_summary.json"
"${PY}" - "${AUDIT_DIR}/audit_summary.json" "${NUM_SHARDS}" <<'PY'
import json, pathlib, sys
value = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert value["status"] == "COMPLETE_EXACT_IDENTITY_ZERO_NUMERICAL_REUSE", value
assert value["window_count"] == 2721 and value["role_count"] == 8163, value
assert value["reusable_role_counts"] == {"prefix": 0, "h2": 0, "h4": 0}, value
assert value["num_shards"] == int(sys.argv[2]), value
assert sum(item["window_count"] for item in value["shards"]) == 2721, value
PY

mkdir -p "${LAUNCHER}"
PROJECT_ROOT="${PROJECT_ROOT}" PROVENANCE_DIR="${LAUNCHER}" bash scripts/d_series/capture_source_provenance.sh
base_commit=$(cat "${LAUNCHER}/base_commit.txt")
git_dirty=$(cat "${LAUNCHER}/git_dirty.txt")
: > "${LAUNCHER}/submissions.tsv"

for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  shard_id=$(printf '%02d' "${shard}")
  selection="${AUDIT_DIR}/shard_${shard_id}_parent_ids.txt"
  test -s "${selection}"
  selection_sha=$(sha256sum "${selection}" | awk '{print $1}')
  job_name="bq-ds-wan-p8-s${shard_id}-$(date -u +%m%d%H%M%S)"
  command="cd ${PROJECT_ROOT} && PROJECT_ROOT=${PROJECT_ROOT} RUN_TAG=${RUN_TAG} SHARD_ID=${shard_id} SELECTION_FILE=${selection} SELECTION_SHA256=${selection_sha} FEATURE_ROOT=${FEATURE_ROOT} D_SERIES_BASE_COMMIT=${base_commit} D_SERIES_GIT_DIRTY=${git_dirty} bash scripts/d_series/run_wan_gap_shard_sco.sh"
  printf '%s\n' "${command}" > "${LAUNCHER}/command_shard_${shard_id}.txt"
  "${SCO}" acp jobs create --workspace-name=aigc --aec2-name="${AEC2_NAME}" \
    --job-name="${job_name}" --priority=HIGHEST \
    --container-image-url=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617 \
    --storage-mount=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm --training-framework=pytorch \
    --worker-nodes=1 --worker-spec=N4lS.Iq.I80.1 --command="${command}" \
    | tee "${LAUNCHER}/sco_submit_shard_${shard_id}.log"
  job_id=$(grep -oE 'pt-[a-z0-9]+' "${LAUNCHER}/sco_submit_shard_${shard_id}.log" | head -1)
  test -n "${job_id}"
  printf '%s\t%s\t%s\t1\t%s\t%s\n' "${shard_id}" "${job_id}" "${job_name}" \
    "$(grep -cve '^$' "${selection}")" "${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/runs/${RUN_TAG}/shard_${shard_id}" \
    | tee -a "${LAUNCHER}/submissions.tsv"
done
