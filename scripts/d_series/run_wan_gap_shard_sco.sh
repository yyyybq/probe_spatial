#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
RUN_TAG=${RUN_TAG:?RUN_TAG is required}
SHARD_ID=${SHARD_ID:?SHARD_ID is required}
SELECTION_FILE=${SELECTION_FILE:?SELECTION_FILE is required}
SELECTION_SHA256=${SELECTION_SHA256:?SELECTION_SHA256 is required}
FEATURE_ROOT=${FEATURE_ROOT:?FEATURE_ROOT is required}
PY=${PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}
SCENE_ROOT=${SCENE_ROOT:-/mnt/umm/users/yinbaiqiao/InsScene-15K/data/processed_scannetpp_v2/processed_scannetpp_v2}
REPORT=${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/runs/${RUN_TAG}/shard_${SHARD_ID}
PROVENANCE=${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/runs/${RUN_TAG}/launcher

mkdir -p "${REPORT}" "${FEATURE_ROOT}"
date -u +%FT%TZ > "${REPORT}/started_utc.txt"
nvidia-smi > "${REPORT}/nvidia_smi.txt"
"${PY}" - <<'PY'
import torch
assert torch.cuda.device_count() == 1, f"expected one visible GPU, got {torch.cuda.device_count()}"
print(torch.cuda.get_device_name(0))
PY
for name in base_commit.txt git_dirty.txt working_tree.patch source_snapshot.tar.gz source_hashes.sha256; do
  test -f "${PROVENANCE}/${name}"
  cp "${PROVENANCE}/${name}" "${REPORT}/${name}"
done
test "$(sha256sum "${SELECTION_FILE}" | awk '{print $1}')" = "${SELECTION_SHA256}"
export D_SERIES_BASE_COMMIT=${D_SERIES_BASE_COMMIT:-$(cat "${REPORT}/base_commit.txt")}
export D_SERIES_GIT_DIRTY=${D_SERIES_GIT_DIRTY:-$(cat "${REPORT}/git_dirty.txt")}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0

FROZEN_ROOT=$(mktemp -d "/tmp/dseries_${RUN_TAG}_s${SHARD_ID}_source.XXXXXX")
cleanup() { rm -rf -- "${FROZEN_ROOT}"; }
trap cleanup EXIT
tar -xzf "${REPORT}/source_snapshot.tar.gz" -C "${FROZEN_ROOT}"
printf '%s\n' "${FROZEN_ROOT}" > "${REPORT}/executed_source_root.txt"
sha256sum "${REPORT}/source_snapshot.tar.gz" > "${REPORT}/executed_source_snapshot.sha256"
cd "${FROZEN_ROOT}"
export PYTHONPATH="${FROZEN_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

printf '%s\tSTARTED\twan_gap_shard_%s\n' "$(date -u +%FT%TZ)" "${SHARD_ID}" | tee "${REPORT}/stage_events.tsv"
set +e
"${PY}" scripts/d_series/extract_wan.py \
  --manifest experiments/d_series_scannetpp_v1/data/sample_manifest.jsonl \
  --scene-root "${SCENE_ROOT}" --output-root "${FEATURE_ROOT}" \
  --selection-file "${SELECTION_FILE}" --shard-id "${SHARD_ID}" \
  > "${REPORT}/extract.log" 2>&1
rc=$?
set -e
if (( rc != 0 )); then
  printf '%s\tFAILED\twan_gap_shard_%s\n' "$(date -u +%FT%TZ)" "${SHARD_ID}" | tee -a "${REPORT}/stage_events.tsv" >&2
  printf '{"stage":"wan_gap_shard_%s","shard":%s,"failed":1,"rc":%s,"ended_utc":"%s"}\n' \
    "${SHARD_ID}" "$((10#${SHARD_ID}))" "${rc}" "$(date -u +%FT%TZ)" > "${REPORT}/status.json"
  exit "${rc}"
fi

audit="${FEATURE_ROOT}/wan_shard_${SHARD_ID}.json"
"${PY}" - "${audit}" "${SELECTION_FILE}" <<'PY'
import json, pathlib, sys
audit_path, selection = map(pathlib.Path, sys.argv[1:])
value = json.loads(audit_path.read_text())
n = len([line for line in selection.read_text().splitlines() if line.strip()])
assert value["windows_requested"] == n and value["windows_completed"] == n, value
counts = value["counts"]
for role in ("prefix", "h2", "h4"):
    assert counts.get(f"{role}_extracted", 0) + counts.get(f"{role}_skipped_complete", 0) == n, (role, counts, n)
PY
printf '%s\tCOMPLETE\twan_gap_shard_%s\n' "$(date -u +%FT%TZ)" "${SHARD_ID}" | tee -a "${REPORT}/stage_events.tsv"
printf '{"stage":"wan_gap_shard_%s","shard":%s,"failed":0,"windows":%s,"ended_utc":"%s"}\n' \
  "${SHARD_ID}" "$((10#${SHARD_ID}))" "$(grep -cve '^$' "${SELECTION_FILE}")" "$(date -u +%FT%TZ)" > "${REPORT}/status.json"
