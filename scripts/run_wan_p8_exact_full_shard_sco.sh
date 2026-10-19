#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
RUN_TAG=${RUN_TAG:?RUN_TAG is required}
FEATURE_ROOT=${FEATURE_ROOT:?FEATURE_ROOT is required}
SHARD_ID=${SHARD_ID:?SHARD_ID is required}
PY=${PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}
DATA_ROOT=/mnt/umm/users/yinbaiqiao/InsScene-15K/data
MODEL=/mnt/umm/users/yinbaiqiao/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780a534b6463e45facd96134c9f345acfa5b
MANIFEST=${PROJECT_ROOT}/manifests/c_retrain_20260722/inscene_c_h124_prefix8.jsonl
REPORT_DIR=${PROJECT_ROOT}/reports/wan_p8_recovery/${RUN_TAG}
SOURCE_PROVENANCE=${REPORT_DIR}/provenance
SHARD_NAME=$(printf 'shard_%02d' "${SHARD_ID}")
SHARD_PROVENANCE=${FEATURE_ROOT}/provenance/shards/${SHARD_NAME}
SELECTION=${SOURCE_PROVENANCE}/shards/${SHARD_NAME}_sample_ids.txt
INVENTORY=${SOURCE_PROVENANCE}/shards/${SHARD_NAME}_inventory.json
STREAMING_ROOT=${FEATURE_ROOT}/streaming
TARGET_ROOT=${FEATURE_ROOT}/target
LOG=${REPORT_DIR}/logs/${SHARD_NAME}.log
STATUS=${REPORT_DIR}/logs/${SHARD_NAME}.status.json

mkdir -p "${REPORT_DIR}/logs"
[[ -f "${SELECTION}" && -f "${INVENTORY}" ]] || {
  echo "missing frozen shard inputs for ${SHARD_NAME}" >&2
  exit 2
}
[[ ! -e "${STATUS}" ]] || { echo "refusing existing status ${STATUS}" >&2; exit 2; }
exec > >(tee -a "${LOG}") 2>&1

started=$(date -u +%FT%TZ)
finish() {
  rc=$?
  trap - EXIT
  ended=$(date -u +%FT%TZ)
  printf '{"run_tag":"%s","shard":%d,"started":"%s","ended":"%s","rc":%d}\n' \
    "${RUN_TAG}" "${SHARD_ID}" "${started}" "${ended}" "${rc}" > "${STATUS}"
  exit "${rc}"
}
trap finish EXIT

cd "${PROJECT_ROOT}"
echo "START ${started} RUN_TAG=${RUN_TAG} SHARD=${SHARD_ID} FEATURE_ROOT=${FEATURE_ROOT}"
nvidia-smi
mkdir -p "${STREAMING_ROOT}" "${TARGET_ROOT}" "${SHARD_PROVENANCE}"
for frozen in \
  canonical_manifest.jsonl \
  extract_wan_manifest_exact.py \
  git_commit.txt \
  git_status_short.txt \
  git_dirty.diff \
  sha256.txt \
  wan_model_snapshot.txt; do
  [[ -f "${SOURCE_PROVENANCE}/${frozen}" ]] || {
    echo "missing frozen provenance: ${SOURCE_PROVENANCE}/${frozen}" >&2
    exit 2
  }
  cp "${SOURCE_PROVENANCE}/${frozen}" "${SHARD_PROVENANCE}/${frozen}"
done
cp "${SELECTION}" "${SHARD_PROVENANCE}/${SHARD_NAME}_sample_ids.txt"
cp "${INVENTORY}" "${SHARD_PROVENANCE}/${SHARD_NAME}_inventory.json"

SHARD_ID="${SHARD_ID}" "${PY}" - <<'PY' > "${SHARD_PROVENANCE}/environment.json"
import json, os, platform, sys
import diffusers, safetensors, torch, transformers
print(json.dumps({
    "shard": int(os.environ["SHARD_ID"]),
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

SHARD_ID="${SHARD_ID}" FEATURE_ROOT="${FEATURE_ROOT}" INVENTORY="${INVENTORY}" "${PY}" - <<'PY'
import json, os
from pathlib import Path

root = Path(os.environ["FEATURE_ROOT"])
shard = int(os.environ["SHARD_ID"])
inventory = json.loads(Path(os.environ["INVENTORY"]).read_text())
problems = []
for scene in inventory["scenes"]:
    for kind in ("streaming", "target"):
        path = root / kind / "wan" / "scannetpp" / scene
        if path.exists():
            problems.append(str(path))
if problems:
    raise SystemExit("Refusing existing shard scene outputs: " + ", ".join(problems[:5]))
print(json.dumps({"preflight": "ok", "shard": shard, "scenes": inventory["scene_count"], "samples": inventory["sample_count"]}))
PY

common_args=(
  --manifest "${MANIFEST}"
  --selection-file "${SELECTION}"
  --data-root "${DATA_ROOT}"
  --model-id "${MODEL}"
  --layers 0 5 10 15 25 29
  --target-horizon 1
  --seed 42
)

printf '%q ' "${PY}" scripts/extract_wan_manifest_exact.py --mode streaming --out-root "${STREAMING_ROOT}" "${common_args[@]}" \
  > "${SHARD_PROVENANCE}/streaming_command.txt"
printf '\n' >> "${SHARD_PROVENANCE}/streaming_command.txt"
"${PY}" scripts/extract_wan_manifest_exact.py --mode streaming --out-root "${STREAMING_ROOT}" "${common_args[@]}"

printf '%q ' "${PY}" scripts/extract_wan_manifest_exact.py --mode target --out-root "${TARGET_ROOT}" "${common_args[@]}" \
  > "${SHARD_PROVENANCE}/target_command.txt"
printf '\n' >> "${SHARD_PROVENANCE}/target_command.txt"
"${PY}" scripts/extract_wan_manifest_exact.py --mode target --out-root "${TARGET_ROOT}" "${common_args[@]}"

echo "SHARD PASS ${SHARD_NAME}"
