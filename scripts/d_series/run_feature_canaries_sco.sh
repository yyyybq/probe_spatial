#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
RUN_TAG=${RUN_TAG:?RUN_TAG is required}
PY=${PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}
SCENE_ROOT=/mnt/umm/users/yinbaiqiao/InsScene-15K/data/processed_scannetpp_v2/processed_scannetpp_v2
OUT=${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/features_canary/${RUN_TAG}
REPORT=${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/runs/${RUN_TAG}
PROVENANCE=${REPORT}/launcher

[[ ! -e "${OUT}" ]] || { echo "refusing existing output ${OUT}" >&2; exit 2; }
mkdir -p "${OUT}" "${REPORT}"
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

FROZEN_ROOT=$(mktemp -d "/tmp/dseries_${RUN_TAG}_source.XXXXXX")
tar -xzf "${REPORT}/source_snapshot.tar.gz" -C "${FROZEN_ROOT}"
printf '%s\n' "${FROZEN_ROOT}" > "${REPORT}/executed_source_root.txt"
sha256sum "${REPORT}/source_snapshot.tar.gz" > "${REPORT}/executed_source_snapshot.sha256"
cd "${FROZEN_ROOT}"
set +e
CUDA_VISIBLE_DEVICES=0 "${PY}" scripts/d_series/vjepa2_v2_canary.py \
    --manifest experiments/d_series_scannetpp_v1/data/sample_manifest.jsonl \
    --mapping experiments/d_series_scannetpp_v1/v2_reuse_mapping.csv \
    --coverage-receipt results/independent_actionability_algebra_p8_final_v1_20260913/coverage/vjepa2_coverage_receipt.json \
    --scene-root "${SCENE_ROOT}" --output "${REPORT}/vjepa2_v2_pooled_canary.json" \
    > "${REPORT}/vjepa2_v2.log" 2>&1
rc_v2=$?
( CUDA_VISIBLE_DEVICES=0 "${PY}" scripts/d_series/extract_vjepa2.py \
    --scene-root "${SCENE_ROOT}" --output-root "${OUT}" --canary-only \
    --model-source "${PROJECT_ROOT}/model_cache/models--facebook--vjepa2-vitl-fpc64-256/snapshots/b3c1679b7c34d3255ef3547f27c7b226aefab26f" && \
  CUDA_VISIBLE_DEVICES=0 "${PY}" scripts/d_series/overfit_canary.py --model vjepa2_vitl256 \
    --reference native_target --branch spatial --scene-root "${SCENE_ROOT}" --feature-root "${OUT}" \
    --output "${REPORT}/vjepa2_native_spatial_overfit.json" ) \
  > "${REPORT}/vjepa2.log" 2>&1 &
pid_vjepa=$!
wait "${pid_vjepa}"; rc_vjepa=$?
set -e
ended=$(date -u +%FT%TZ)
printf '{"run_tag":"%s","ended_utc":"%s","scope":"vjepa2_independent","vjepa2_v2_rc":%d,"vjepa2_native_rc":%d}\n' \
  "${RUN_TAG}" "${ended}" "${rc_v2}" "${rc_vjepa}" > "${REPORT}/status.json"
cp "${REPORT}/source_hashes.sha256" "${REPORT}/launcher_hashes.sha256"
(( rc_v2 == 0 && rc_vjepa == 0 ))
