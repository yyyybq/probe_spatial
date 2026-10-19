#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
PY=${PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}
OUTPUT=${PROJECT_ROOT}/results/analysis/wan_p12_l20_action_validity
STATUS=${OUTPUT}/layer20.sco_status.json
LOG=${OUTPUT}/layer20.sco.log

[[ -f "${OUTPUT}/prepare_status.json" ]] || { echo "missing prepare status" >&2; exit 2; }
[[ ! -e "${OUTPUT}/layers/layer20" ]] || { echo "refusing existing layer output" >&2; exit 2; }
[[ ! -e "${STATUS}" ]] || { echo "refusing existing status" >&2; exit 2; }

mkdir -p "${OUTPUT}" /tmp/matplotlib-wan-p12-validity
exec > >(tee -a "${LOG}") 2>&1
started=$(date -u +%FT%TZ)
finish() {
  rc=$?
  trap - EXIT
  ended=$(date -u +%FT%TZ)
  printf '{"prefix":12,"layer":20,"started":"%s","ended":"%s","rc":%d}\n' \
    "${started}" "${ended}" "${rc}" > "${STATUS}"
  exit "${rc}"
}
trap finish EXIT

export PYTHONPATH=${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR=/tmp/matplotlib-wan-p12-validity
export PYTHONUNBUFFERED=1
cd "${PROJECT_ROOT}"

"${PY}" scripts/run_wan_p12_l20_action_validity.py \
  --analysis-dir "${OUTPUT}" \
  --batch-size 64

echo "WAN P12/L20 PHASE1 INTERVENTIONS PASS"
