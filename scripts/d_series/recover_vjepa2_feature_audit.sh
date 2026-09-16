#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
RUN_TAG=${RUN_TAG:?RUN_TAG is required}
FEATURE_ROOT=${FEATURE_ROOT:?FEATURE_ROOT is required}
PY=${PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}
REPORT=${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/runs/${RUN_TAG}/vjepa2_vitl256
SNAPSHOT=${REPORT}/source_snapshot.tar.gz

test -f "${SNAPSHOT}"
test -f "${REPORT}/status.json"
grep -q 'ModuleNotFoundError: No module named '\''scripts'\''' "${REPORT}/feature_audit.log"
"${PY}" - "${REPORT}/status.json" <<'PY'
import json, sys
status = json.load(open(sys.argv[1]))
assert status["failed"] == 1, status
assert status["processes"] == 8, status
PY

FROZEN_ROOT=$(mktemp -d "/tmp/dseries_${RUN_TAG}_audit_recovery.XXXXXX")
trap 'rm -rf -- "${FROZEN_ROOT}"' EXIT
tar -xzf "${SNAPSHOT}" -C "${FROZEN_ROOT}"
sha256sum "${SNAPSHOT}" > "${REPORT}/feature_audit_recovery_snapshot.sha256"
date -u +%FT%TZ > "${REPORT}/feature_audit_recovery_started_utc.txt"
cd "${FROZEN_ROOT}"
PYTHONPATH="${FROZEN_ROOT}" "${PY}" scripts/d_series/audit_vjepa2_d_features.py \
  --manifest experiments/d_series_scannetpp_v1/data/sample_manifest.jsonl \
  --feature-root "${FEATURE_ROOT}" --output "${REPORT}/feature_audit.json" \
  > "${REPORT}/feature_audit_recovery.log" 2>&1
ended_utc=$(date -u +%FT%TZ)
snapshot_sha256=$(sha256sum "${SNAPSHOT}" | awk '{print $1}')
printf '{"status":"RECOVERED_COMPLETE_VALID","original_status_failed":1,"recovery_kind":"audit_only_import_path_fix","source_snapshot_sha256":"%s","ended_utc":"%s"}\n' \
  "${snapshot_sha256}" "${ended_utc}" > "${REPORT}/feature_audit_recovery_status.json"
