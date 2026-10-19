#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
CANARY_TAG=${CANARY_TAG:?CANARY_TAG is required}
FEATURE_ROOT=${FEATURE_ROOT:?FEATURE_ROOT is required}
PY=${PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}
DATA_ROOT=/mnt/umm/users/yinbaiqiao/InsScene-15K/data
MODEL=/mnt/umm/users/yinbaiqiao/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780a534b6463e45facd96134c9f345acfa5b
MANIFEST=${PROJECT_ROOT}/manifests/c_retrain_20260722/inscene_c_h124_prefix8.jsonl
SELECTION=${PROJECT_ROOT}/reports/wan_p8_recovery/canary_selection_20260821_185059.txt
REPORT_DIR=${PROJECT_ROOT}/reports/wan_p8_recovery/${CANARY_TAG}
LAUNCHER_DIR=${REPORT_DIR}/launcher
STREAMING_ROOT=${FEATURE_ROOT}/streaming
TARGET_ROOT=${FEATURE_ROOT}/target
PROVENANCE_ROOT=${FEATURE_ROOT}/provenance
LOG=${REPORT_DIR}/canary_job.log

[[ ! -e "${FEATURE_ROOT}" ]] || {
  echo "refusing existing FEATURE_ROOT=${FEATURE_ROOT}" >&2
  exit 2
}
mkdir -p "${REPORT_DIR}" "${STREAMING_ROOT}" "${TARGET_ROOT}" "${PROVENANCE_ROOT}"
exec > >(tee -a "${LOG}") 2>&1

started=$(date -u +%FT%TZ)
status_file=${REPORT_DIR}/sco_runtime_status.json
finish() {
  rc=$?
  trap - EXIT
  ended=$(date -u +%FT%TZ)
  printf '{"canary_tag":"%s","started":"%s","ended":"%s","rc":%d}\n' \
    "${CANARY_TAG}" "${started}" "${ended}" "${rc}" > "${status_file}"
  exit "${rc}"
}
trap finish EXIT

cd "${PROJECT_ROOT}"
echo "START ${started} CANARY_TAG=${CANARY_TAG} FEATURE_ROOT=${FEATURE_ROOT}"
nvidia-smi
"${PY}" - <<'PY'
import json, platform, sys
import diffusers, safetensors, torch, transformers
print(json.dumps({
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
    "diffusers": diffusers.__version__,
    "transformers": transformers.__version__,
    "safetensors": safetensors.__version__,
}, indent=2))
PY

cp "${MANIFEST}" "${PROVENANCE_ROOT}/canonical_manifest.jsonl"
cp scripts/extract_wan_manifest_exact.py "${PROVENANCE_ROOT}/extract_wan_manifest_exact.py"
cp scripts/verify_wan_p8_dataset_exact.py "${PROVENANCE_ROOT}/verify_wan_p8_dataset_exact.py"
cp scripts/audit_wan_p8_exact_canary.py "${PROVENANCE_ROOT}/audit_wan_p8_exact_canary.py"
cp "${SELECTION}" "${PROVENANCE_ROOT}/canary_selection.txt"
for snapshot in git_commit.txt git_status_short.txt git_dirty.diff; do
  [[ -f "${LAUNCHER_DIR}/${snapshot}" ]] || {
    echo "missing launcher provenance snapshot: ${LAUNCHER_DIR}/${snapshot}" >&2
    exit 2
  }
  cp "${LAUNCHER_DIR}/${snapshot}" "${PROVENANCE_ROOT}/${snapshot}"
done
sha256sum \
  "${PROVENANCE_ROOT}/canonical_manifest.jsonl" \
  "${PROVENANCE_ROOT}/extract_wan_manifest_exact.py" \
  "${PROVENANCE_ROOT}/verify_wan_p8_dataset_exact.py" \
  "${PROVENANCE_ROOT}/audit_wan_p8_exact_canary.py" \
  "${PROVENANCE_ROOT}/canary_selection.txt" \
  "${PROVENANCE_ROOT}/git_commit.txt" \
  "${PROVENANCE_ROOT}/git_status_short.txt" \
  "${PROVENANCE_ROOT}/git_dirty.diff" \
  > "${PROVENANCE_ROOT}/sha256.txt"

common_args=(
  --manifest "${MANIFEST}"
  --selection-file "${SELECTION}"
  --data-root "${DATA_ROOT}"
  --model-id "${MODEL}"
  --layers 0 5 10 15 25 29
  --target-horizon 1
  --seed 42
)

printf '%q ' "${PY}" scripts/extract_wan_manifest_exact.py --mode streaming --out-root "${STREAMING_ROOT}" "${common_args[@]}" > "${PROVENANCE_ROOT}/streaming_command.txt"
printf '\n' >> "${PROVENANCE_ROOT}/streaming_command.txt"
"${PY}" scripts/extract_wan_manifest_exact.py \
  --mode streaming \
  --out-root "${STREAMING_ROOT}" \
  "${common_args[@]}"

printf '%q ' "${PY}" scripts/extract_wan_manifest_exact.py --mode target --out-root "${TARGET_ROOT}" "${common_args[@]}" > "${PROVENANCE_ROOT}/target_command.txt"
printf '\n' >> "${PROVENANCE_ROOT}/target_command.txt"
"${PY}" scripts/extract_wan_manifest_exact.py \
  --mode target \
  --out-root "${TARGET_ROOT}" \
  "${common_args[@]}"

"${PY}" scripts/verify_wan_p8_dataset_exact.py \
  --manifest "${MANIFEST}" \
  --selection-file "${SELECTION}" \
  --filtered-manifest "${REPORT_DIR}/canary_manifest_40.jsonl" \
  --data-root "${DATA_ROOT}" \
  --streaming-root "${STREAMING_ROOT}" \
  --target-root "${TARGET_ROOT}" \
  --out-json "${REPORT_DIR}/dataset_canary.json" \
  --num-workers 0

"${PY}" scripts/audit_wan_p8_exact_canary.py \
  --canary-tag "${CANARY_TAG}" \
  --manifest "${MANIFEST}" \
  --selection-file "${SELECTION}" \
  --streaming-root "${STREAMING_ROOT}" \
  --target-root "${TARGET_ROOT}" \
  --dataset-json "${REPORT_DIR}/dataset_canary.json" \
  --out-json "${REPORT_DIR}/canary_summary.json" \
  --out-md "${REPORT_DIR}/canary_report.md"

echo "CANARY PASS ${CANARY_TAG}"
