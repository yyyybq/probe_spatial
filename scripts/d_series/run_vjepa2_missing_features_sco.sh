#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
RUN_TAG=${RUN_TAG:?RUN_TAG is required}
FEATURE_ROOT=${FEATURE_ROOT:?FEATURE_ROOT is required}
NPROC=${NPROC:-8}
PY=${PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}
SCENE_ROOT=/mnt/umm/users/yinbaiqiao/InsScene-15K/data/processed_scannetpp_v2/processed_scannetpp_v2
MODEL_SOURCE=${PROJECT_ROOT}/model_cache/models--facebook--vjepa2-vitl-fpc64-256/snapshots/b3c1679b7c34d3255ef3547f27c7b226aefab26f
REPORT=${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/runs/${RUN_TAG}/vjepa2_vitl256
PROVENANCE=${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/runs/${RUN_TAG}/launcher
TOTAL_WINDOWS=2721

mkdir -p "${REPORT}"
date -u +%FT%TZ > "${REPORT}/started_utc.txt"
nvidia-smi > "${REPORT}/nvidia_smi.txt"
for name in base_commit.txt git_dirty.txt working_tree.patch source_snapshot.tar.gz source_hashes.sha256; do
  test -f "${PROVENANCE}/${name}"
  cp "${PROVENANCE}/${name}" "${REPORT}/${name}"
done
export D_SERIES_BASE_COMMIT=${D_SERIES_BASE_COMMIT:-$(cat "${REPORT}/base_commit.txt")}
export D_SERIES_GIT_DIRTY=${D_SERIES_GIT_DIRTY:-$(cat "${REPORT}/git_dirty.txt")}

FROZEN_ROOT=$(mktemp -d "/tmp/dseries_${RUN_TAG}_source.XXXXXX")
tar -xzf "${REPORT}/source_snapshot.tar.gz" -C "${FROZEN_ROOT}"
printf '%s\n' "${FROZEN_ROOT}" > "${REPORT}/executed_source_root.txt"
sha256sum "${REPORT}/source_snapshot.tar.gz" > "${REPORT}/executed_source_snapshot.sha256"
cd "${FROZEN_ROOT}"

chunk=$(( (TOTAL_WINDOWS + NPROC - 1) / NPROC ))
pids=()
for ((gpu=0; gpu<NPROC; gpu++)); do
  start=$((gpu * chunk)); count=${chunk}
  (( start < TOTAL_WINDOWS )) || break
  if (( start + count > TOTAL_WINDOWS )); then count=$((TOTAL_WINDOWS - start)); fi
  CUDA_VISIBLE_DEVICES=${gpu} "${PY}" scripts/d_series/extract_vjepa2.py \
    --scene-root "${SCENE_ROOT}" --output-root "${FEATURE_ROOT}" --model-source "${MODEL_SOURCE}" \
    --window-start "${start}" --max-windows "${count}" --skip-order-shuffle --skip-future-invariance \
    > "${REPORT}/gpu${gpu}_${start}_${count}.log" 2>&1 &
  pids+=("$!")
done
set +e; failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=$((failed + 1)); done
set -e
if (( failed == 0 )); then
  PYTHONPATH="${FROZEN_ROOT}" "${PY}" scripts/d_series/audit_vjepa2_d_features.py \
    --manifest experiments/d_series_scannetpp_v1/data/sample_manifest.jsonl \
    --feature-root "${FEATURE_ROOT}" --output "${REPORT}/feature_audit.json" \
    > "${REPORT}/feature_audit.log" 2>&1 || failed=$((failed + 1))
fi
printf '{"model":"vjepa2_vitl256","scope":"missing_role_fill","processes":%d,"failed":%d,"ended_utc":"%s"}\n' \
  "${#pids[@]}" "${failed}" "$(date -u +%FT%TZ)" > "${REPORT}/status.json"
(( failed == 0 ))
