#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/mnt/umm/users/yinbaiqiao/probe_spatial
REPORT_DIR=${PROJECT_ROOT}/reports/gate_recovery_20260827T051224Z_2075f3eb
PY=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python
SMOKE_ROOT=${REPORT_DIR}/smoke_data
SCENE_ROOT=${SMOKE_ROOT}/processed_scannetpp_v2/processed_scannetpp_v2
STRICT_OUT=${REPORT_DIR}/manifests/strict_b2_smoke

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}"
export MPLCONFIGDIR="${REPORT_DIR}/mpl_gpu"
export AOSS_CONF=/mnt/umm/users/yinbaiqiao/aoss/petreloss.conf
export AOSS_CACHE_DIR="${REPORT_DIR}/aoss_cache_gpu"
export AOSS_API_HOST=aoss-internal.cn-fz-01.fjscmsapi-oss.com
export NO_PROXY='localhost,127.0.0.1,::1,baiqiao.aoss-internal.cn-fz-01.fjscmsapi-oss.com,.aoss-internal.cn-fz-01.fjscmsapi-oss.com,aoss-internal.cn-fz-01.fjscmsapi-oss.com'
export no_proxy="${NO_PROXY}"

mkdir -p "${REPORT_DIR}/logs" "${REPORT_DIR}/manifests" "${SCENE_ROOT}" "${MPLCONFIGDIR}" "${AOSS_CACHE_DIR}"

timeout 300s "${PY}" "${REPORT_DIR}/gpu_aoss_gate.py"

# Phase 6 is intentionally below the prerequisite gate and is unreachable on failure.
for scene in 00777c41d4 00a231a370 00dd871005 020312de8d 027cd6ea0f; do
  source_scene=/mnt/umm/users/yinbaiqiao/InsScene-15K/data/processed_scannetpp_v2/processed_scannetpp_v2/${scene}
  [[ -d "${source_scene}" ]]
  [[ -L "${SCENE_ROOT}/${scene}" || ! -e "${SCENE_ROOT}/${scene}" ]]
  ln -sfn "${source_scene}" "${SCENE_ROOT}/${scene}"
done

"${PY}" "${REPORT_DIR}/make_smoke_split.py"
mkdir -p "${STRICT_OUT}"
"${PY}" scripts/build_b2_strict_manifest.py \
  --data-root "${SMOKE_ROOT}" \
  --split-manifest "${REPORT_DIR}/manifests/strict_b2_smoke_split.json" \
  --output-dir "${STRICT_OUT}" \
  --run-tag gate_recovery_smoke_20260827T051224Z_2075f3eb \
  --splits train \
  --max-windows-per-scene 1
"${PY}" "${REPORT_DIR}/validate_strict_smoke.py"
