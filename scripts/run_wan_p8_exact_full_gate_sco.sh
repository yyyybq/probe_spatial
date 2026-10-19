#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
RUN_TAG=${RUN_TAG:?RUN_TAG is required}
FEATURE_ROOT=${FEATURE_ROOT:?FEATURE_ROOT is required}
PY=${PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}
MANIFEST=${PROJECT_ROOT}/manifests/c_retrain_20260722/inscene_c_h124_prefix8.jsonl
DATA_ROOT=/mnt/umm/users/yinbaiqiao/InsScene-15K/data
REPORT_DIR=${PROJECT_ROOT}/reports/wan_p8_recovery/${RUN_TAG}
SOURCE_PROVENANCE=${REPORT_DIR}/provenance
FEATURE_PROVENANCE=${FEATURE_ROOT}/provenance
STREAMING_ROOT=${FEATURE_ROOT}/streaming
TARGET_ROOT=${FEATURE_ROOT}/target
LOG=${REPORT_DIR}/logs/full_gate.log
STATUS=${REPORT_DIR}/logs/full_gate.status.json

[[ ! -e "${STATUS}" ]] || { echo "refusing existing gate status ${STATUS}" >&2; exit 2; }
exec > >(tee -a "${LOG}") 2>&1
started=$(date -u +%FT%TZ)
finish() {
  rc=$?
  trap - EXIT
  ended=$(date -u +%FT%TZ)
  printf '{"run_tag":"%s","started":"%s","ended":"%s","rc":%d}\n' \
    "${RUN_TAG}" "${started}" "${ended}" "${rc}" > "${STATUS}"
  exit "${rc}"
}
trap finish EXIT

cd "${PROJECT_ROOT}"
for shard in 00 01 02 03 04 05; do
  "${PY}" - "${REPORT_DIR}/logs/shard_${shard}.status.json" <<'PY'
import json, sys
status = json.load(open(sys.argv[1]))
assert status.get("rc") == 0, status
PY
done

mkdir -p "${FEATURE_PROVENANCE}/shards"
for frozen in \
  canonical_manifest.jsonl extract_wan_manifest_exact.py prepare_wan_p8_exact_full.py \
  run_wan_p8_exact_full_shard_sco.sh canary_report.md canary_summary.json \
  canary_runtime_status.json git_commit.txt git_status_short.txt git_dirty.diff \
  sha256.txt wan_model_snapshot.txt; do
  cp "${SOURCE_PROVENANCE}/${frozen}" "${FEATURE_PROVENANCE}/${frozen}"
done
cp "${SOURCE_PROVENANCE}/shards/"* "${FEATURE_PROVENANCE}/shards/"
cp scripts/audit_wan_p8_exact_full.py "${FEATURE_PROVENANCE}/audit_wan_p8_exact_full.py"
cp scripts/verify_wan_p8_dataset_exact.py "${FEATURE_PROVENANCE}/verify_wan_p8_dataset_exact.py"
sha256sum "${FEATURE_PROVENANCE}/audit_wan_p8_exact_full.py" "${FEATURE_PROVENANCE}/verify_wan_p8_dataset_exact.py" \
  > "${FEATURE_PROVENANCE}/full_gate_source_sha256.txt"

"${PY}" scripts/audit_wan_p8_exact_full.py \
  --manifest "${MANIFEST}" \
  --streaming-root "${STREAMING_ROOT}" \
  --target-root "${TARGET_ROOT}" \
  --shard-inventory "${SOURCE_PROVENANCE}/shards/full_shard_inventory.json" \
  --status-dir "${REPORT_DIR}/logs" \
  --out-json "${REPORT_DIR}/full_feature_audit.json" \
  --check-finite

"${PY}" scripts/verify_wan_p8_dataset_exact.py \
  --manifest "${MANIFEST}" \
  --filtered-manifest "${REPORT_DIR}/full_dataset_manifest.jsonl" \
  --data-root "${DATA_ROOT}" \
  --streaming-root "${STREAMING_ROOT}" \
  --target-root "${TARGET_ROOT}" \
  --out-json "${REPORT_DIR}/full_dataset_dryrun.json" \
  --num-workers 0

RUN_TAG="${RUN_TAG}" REPORT_DIR="${REPORT_DIR}" FEATURE_ROOT="${FEATURE_ROOT}" "${PY}" - <<'PY'
import hashlib, json, os
from pathlib import Path

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

report = Path(os.environ["REPORT_DIR"])
feature = Path(os.environ["FEATURE_ROOT"]).resolve()
audit = json.load(open(report / "full_feature_audit.json"))
dataset = json.load(open(report / "full_dataset_dryrun.json"))
manifest_hash = "772194f077c45cc85a942f17f46f50103125ae8f5b1c983c08881c1759607e0c"
ok = (
    audit.get("ok") is True
    and dataset.get("ok") is True
    and dataset.get("selected_rows") == 2724
    and dataset.get("canonical_manifest_sha256") == manifest_hash
    and dataset.get("filtered_manifest_sha256") == manifest_hash
    and Path(dataset.get("streaming_root", "")).resolve() == feature / "streaming"
    and Path(dataset.get("target_root", "")).resolve() == feature / "target"
)
result = {
    "run_tag": os.environ["RUN_TAG"],
    "ok": ok,
    "canonical_manifest_sha256": manifest_hash,
    "feature_root": str(feature),
    "streaming_root": str(feature / "streaming"),
    "target_root": str(feature / "target"),
    "feature_audit": str(report / "full_feature_audit.json"),
    "feature_audit_sha256": sha(report / "full_feature_audit.json"),
    "dataset_dryrun": str(report / "full_dataset_dryrun.json"),
    "dataset_dryrun_sha256": sha(report / "full_dataset_dryrun.json"),
    "streaming_exact_raw": audit.get("streaming_exact_raw"),
    "target_exact_raw": audit.get("target_exact_raw"),
    "finite_files_checked": audit.get("finite_files_checked"),
    "dataset_layers": dataset.get("layers"),
}
(report / "full_gate_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
(report / "full_gate_report.md").write_text(f"""# Wan p8 Full Feature Gate

```text
RUN_TAG: {result['run_tag']}
STATUS: {'PASS' if ok else 'FAIL'}
CANONICAL_MANIFEST_SHA256: {manifest_hash}
STREAMING_EXACT_RAW: {result['streaming_exact_raw']}/2724
TARGET_EXACT_RAW: {result['target_exact_raw']}/2724
FINITE_FILES_CHECKED: {result['finite_files_checked']}
DATASET_ROWS_BY_LAYER: {','.join(str(dataset.get('layers', {}).get(str(layer), {}).get('checked')) for layer in [0,5,10,15,25,29])}
FEATURE_ROOT: {feature}
```
""")
if not ok:
    raise SystemExit(2)
PY

echo "FULL FEATURE GATE PASS ${RUN_TAG}"
