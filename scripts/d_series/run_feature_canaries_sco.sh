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

cd "${PROJECT_ROOT}"
set +e
( CUDA_VISIBLE_DEVICES=0 "${PY}" scripts/d_series/extract_vjepa2.py \
    --scene-root "${SCENE_ROOT}" --output-root "${OUT}" --canary-only && \
  CUDA_VISIBLE_DEVICES=0 "${PY}" scripts/d_series/overfit_canary.py --model vjepa2_vitl256 \
    --reference native_target --branch spatial --scene-root "${SCENE_ROOT}" --feature-root "${OUT}" \
    --output "${REPORT}/vjepa2_native_spatial_overfit.json" ) \
  > "${REPORT}/vjepa2.log" 2>&1 &
pid_vjepa=$!
( CUDA_VISIBLE_DEVICES=1 "${PY}" scripts/d_series/extract_dinov2.py \
    --scene-root "${SCENE_ROOT}" --output-root "${OUT}" --canary-only && \
  CUDA_VISIBLE_DEVICES=1 "${PY}" scripts/d_series/overfit_canary.py --model dinov2_vitl14 \
    --branch spatial --scene-root "${SCENE_ROOT}" --feature-root "${OUT}" \
    --output "${REPORT}/dinov2_spatial_overfit.json" ) \
  > "${REPORT}/dinov2.log" 2>&1 &
pid_dino=$!
( CUDA_VISIBLE_DEVICES=2 WAN_T2V_DTYPE=bf16 "${PY}" scripts/d_series/extract_wan.py \
    --scene-root "${SCENE_ROOT}" --output-root "${OUT}" --canary-only && \
  CUDA_VISIBLE_DEVICES=2 "${PY}" scripts/d_series/overfit_canary.py --model wan21_t2v_1p3b \
    --branch spatial --scene-root "${SCENE_ROOT}" --feature-root "${OUT}" \
    --output "${REPORT}/wan_spatial_overfit.json" ) \
  > "${REPORT}/wan.log" 2>&1 &
pid_wan=$!

wait "${pid_vjepa}"; rc_vjepa=$?
wait "${pid_dino}"; rc_dino=$?
wait "${pid_wan}"; rc_wan=$?
set -e
ended=$(date -u +%FT%TZ)
printf '{"run_tag":"%s","ended_utc":"%s","vjepa2_rc":%d,"dinov2_rc":%d,"wan_rc":%d}\n' \
  "${RUN_TAG}" "${ended}" "${rc_vjepa}" "${rc_dino}" "${rc_wan}" > "${REPORT}/status.json"
cp "${REPORT}/source_hashes.sha256" "${REPORT}/launcher_hashes.sha256"
(( rc_vjepa == 0 && rc_dino == 0 && rc_wan == 0 ))
