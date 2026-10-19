#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
RUN_TAG=${RUN_TAG:?RUN_TAG is required}
FEATURE_ROOT=${FEATURE_ROOT:?FEATURE_ROOT is required}
PY=${PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}
REPORT=${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/runs/${RUN_TAG}/wan21_t2v_1p3b
PROVENANCE=${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/runs/${RUN_TAG}/launcher
AUDIT=${REPORT}/feature_audit.json

mkdir -p "${REPORT}"
date -u +%FT%TZ > "${REPORT}/started_utc.txt"
nvidia-smi > "${REPORT}/nvidia_smi.txt"
"${PY}" - <<'PY'
import torch
assert torch.cuda.device_count() == 1, f"expected exactly one visible GPU, got {torch.cuda.device_count()}"
print(torch.cuda.get_device_name(0))
PY
for name in base_commit.txt git_dirty.txt working_tree.patch source_snapshot.tar.gz source_hashes.sha256; do
  test -f "${PROVENANCE}/${name}"
  cp "${PROVENANCE}/${name}" "${REPORT}/${name}"
done
export D_SERIES_BASE_COMMIT=${D_SERIES_BASE_COMMIT:-$(cat "${REPORT}/base_commit.txt")}
export D_SERIES_GIT_DIRTY=${D_SERIES_GIT_DIRTY:-$(cat "${REPORT}/git_dirty.txt")}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0

FROZEN_ROOT=$(mktemp -d "/tmp/dseries_${RUN_TAG}_audit_source.XXXXXX")
cleanup() { rm -rf -- "${FROZEN_ROOT}"; }
trap cleanup EXIT
tar -xzf "${REPORT}/source_snapshot.tar.gz" -C "${FROZEN_ROOT}"
printf '%s\n' "${FROZEN_ROOT}" > "${REPORT}/executed_source_root.txt"
sha256sum "${REPORT}/source_snapshot.tar.gz" > "${REPORT}/executed_source_snapshot.sha256"
cd "${FROZEN_ROOT}"
export PYTHONPATH="${FROZEN_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -e "${AUDIT}" ]]; then
  "${PY}" - "${AUDIT}" <<'PY'
import json, pathlib, sys
audit = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert audit["status"] == "COMPLETE_VALID", audit
assert audit["model"] == "wan21_t2v_1p3b", audit
assert audit["roles"] == {"prefix": {"expected": 2721, "complete": 2721},
                           "isolated_target": {"expected": 5442, "complete": 5442}}, audit
PY
  printf '%s\tSKIPPED_COMPLETE\twan_feature_audit\n' "$(date -u +%FT%TZ)" | tee "${REPORT}/stage_events.tsv"
  printf '{"stage":"wan_feature_audit","failed":0,"resume_skipped":true,"ended_utc":"%s"}\n' "$(date -u +%FT%TZ)" > "${REPORT}/status.json"
  exit 0
fi

printf '%s\tSTARTED\twan_feature_audit\n' "$(date -u +%FT%TZ)" | tee "${REPORT}/stage_events.tsv"
if ! "${PY}" scripts/d_series/audit_wan_d_features.py \
  --manifest experiments/d_series_scannetpp_v1/data/sample_manifest.jsonl \
  --feature-root "${FEATURE_ROOT}" --output "${AUDIT}" > "${REPORT}/feature_audit.log" 2>&1; then
  printf '%s\tFAILED\twan_feature_audit\n' "$(date -u +%FT%TZ)" | tee -a "${REPORT}/stage_events.tsv" >&2
  printf '{"stage":"wan_feature_audit","failed":1,"ended_utc":"%s"}\n' "$(date -u +%FT%TZ)" > "${REPORT}/status.json"
  exit 1
fi
printf '%s\tCOMPLETE\twan_feature_audit\n' "$(date -u +%FT%TZ)" | tee -a "${REPORT}/stage_events.tsv"
printf '{"stage":"wan_feature_audit","failed":0,"ended_utc":"%s"}\n' "$(date -u +%FT%TZ)" > "${REPORT}/status.json"
