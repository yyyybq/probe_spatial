#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
STAGE=${STAGE:?STAGE must be d1 or d2d3}
RUN_TAG=${RUN_TAG:?RUN_TAG is required}
FEATURE_ROOT=${FEATURE_ROOT:-/mnt/umm/users/yinbaiqiao/InsScene-15K/D_SERIES_SCANNETPP_V1_FEATURES}
FORMAL_RUN_ROOT=${FORMAL_RUN_ROOT:-${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/formal_runs}
PY=${PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}
REPORT=${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/runs/${RUN_TAG}/${STAGE}
PROVENANCE=${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/runs/${RUN_TAG}/launcher
SCENE_ROOT=/mnt/umm/users/yinbaiqiao/InsScene-15K/data/processed_scannetpp_v2/processed_scannetpp_v2

mkdir -p "${REPORT}"
date -u +%FT%TZ > "${REPORT}/started_utc.txt"
nvidia-smi > "${REPORT}/nvidia_smi.txt"
for name in base_commit.txt git_dirty.txt working_tree.patch source_snapshot.tar.gz source_hashes.sha256; do
  test -f "${PROVENANCE}/${name}"
  cp "${PROVENANCE}/${name}" "${REPORT}/${name}"
done
test -s "${REPORT}/base_commit.txt"
test -s "${REPORT}/source_snapshot.tar.gz"
export D_SERIES_BASE_COMMIT=${D_SERIES_BASE_COMMIT:-$(cat "${REPORT}/base_commit.txt")}
export D_SERIES_GIT_DIRTY=${D_SERIES_GIT_DIRTY:-$(cat "${REPORT}/git_dirty.txt")}
cd "${PROJECT_ROOT}"

"${PY}" scripts/d_series/audit_protocol.py --feature-root "${FEATURE_ROOT}" --run-root "${FORMAL_RUN_ROOT}"
"${PY}" - experiments/d_series_scannetpp_v1/audits/protocol_audit.json <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
assert report["feature_status"], "feature audit is empty"
failed = {key: value for key, value in report["feature_status"].items() if value["complete"] != value["expected"]}
assert not failed, failed
PY

"${PY}" scripts/d_series/run_training_campaign.py --stage "${STAGE}" --feature-root "${FEATURE_ROOT}" \
  --run-root "${FORMAL_RUN_ROOT}" --scene-root "${SCENE_ROOT}" --report-dir "${REPORT}"
