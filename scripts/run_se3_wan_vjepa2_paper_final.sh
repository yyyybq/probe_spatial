#!/usr/bin/env bash
set -euo pipefail
umask 0022

PROJECT_ROOT=/mnt/umm/users/yinbaiqiao/probe_spatial
PYTHON=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python
OUTPUT=${PROJECT_ROOT}/results/experimental/se3_wan_vjepa2_paper_final_20260917_r1

export MPLCONFIGDIR=/tmp/mpl-se3-wan-vjepa2-paper-final
export PYTHONUNBUFFERED=1
cd "${PROJECT_ROOT}"
mkdir -p "${MPLCONFIGDIR}"

"${PYTHON}" -m py_compile \
  analyses/se3_protocol_v2/wan_vjepa2_paper_final.py
"${PYTHON}" -m analyses.se3_protocol_v2.wan_vjepa2_paper_final \
  --output "${OUTPUT}"
