#!/usr/bin/env bash
set -euo pipefail
umask 0022

PROJECT_ROOT=/mnt/umm/users/yinbaiqiao/probe_spatial
PYTHON=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python
RESULT=${PROJECT_ROOT}/results/experimental/se3_wan_vjepa2_exact_v2_p1_20260917_r1

export MPLCONFIGDIR=/tmp/matplotlib-se3-wan-vjepa2-p1
export PYTHONUNBUFFERED=1
cd "${PROJECT_ROOT}"
mkdir -p "${RESULT}/logs" "${MPLCONFIGDIR}"

"${PYTHON}" -m pytest -q \
  tests/test_wan_vjepa2_p1.py \
  tests/test_wan_vjepa2_v2_structural.py \
  >"${RESULT}/logs/tests.log" 2>&1

"${PYTHON}" -m analyses.se3_protocol_v2.wan_vjepa2_p1 all --device cuda \
  >"${RESULT}/logs/run_all.log" 2>&1
