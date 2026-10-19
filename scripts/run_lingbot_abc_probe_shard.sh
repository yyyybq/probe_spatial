#!/usr/bin/env bash
# Run one shard of the LingBot-Video ABC probe matrix on an SCO 8-GPU worker.
#
# Usage (inside SCO worker):
#   SHARD_TSV=... OUT_DIR=... RUN_ID=... bash scripts/run_lingbot_abc_probe_shard.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PATH="/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHON="${PYTHON:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/mnt/umm/users/yinbaiqiao/.cache/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
mkdir -p "${MPLCONFIGDIR}"

# Do not inherit conda HOST pollution into AOSS helpers / sync tools.
unset HOST || true
export AOSS_API_HOST="${AOSS_API_HOST:-aoss-internal.cn-fz-01.fjscmsapi-oss.com}"
export AOSS_HOST="${AOSS_HOST:-baiqiao.${AOSS_API_HOST}}"
export NO_PROXY="localhost,127.0.0.1,::1,${AOSS_HOST},.${AOSS_API_HOST},${AOSS_API_HOST}"
export no_proxy="${NO_PROXY}"

OUT_DIR=${OUT_DIR:?set OUT_DIR}
RUN_ID=${RUN_ID:?set RUN_ID}
SHARD_TSV=${SHARD_TSV:?set SHARD_TSV}
SLOTS=${SLOTS:-16}
GPU_IDS_CSV=${GPU_IDS_CSV:-0,1,2,3,4,5,6,7}
BATCH_SIZE=${BATCH_SIZE:-2}
BATCH_SIZE_VAL=${BATCH_SIZE_VAL:-2}
NUM_WORKERS=${NUM_WORKERS:-4}
NUM_WORKERS_VAL=${NUM_WORKERS_VAL:-2}
VIDEO_CHANNELS=${VIDEO_CHANNELS:-2048}
FEATURE_TIMESTEP=${FEATURE_TIMESTEP:-250}
FEAT_BACKEND=${FEAT_BACKEND:-aoss}
ALLOW_MISSING_STREAMING=${ALLOW_MISSING_STREAMING:-1}
AOSS_CACHE_ROOT=${AOSS_CACHE_ROOT:-/mnt/local_cache/inscene/aoss_cache}
if [[ ! -d /mnt/local_cache ]]; then
  AOSS_CACHE_ROOT=/tmp/inscene_aoss_cache
fi
export AOSS_CACHE_ROOT

echo "=============================================="
echo " LingBot-Video ABC probe shard"
echo " host       : $(hostname)"
echo " cuda       : $(${PYTHON} -c 'import torch; print(torch.cuda.device_count())')"
echo " OUT_DIR    : ${OUT_DIR}"
echo " RUN_ID     : ${RUN_ID}"
echo " SHARD_TSV  : ${SHARD_TSV}"
echo " jobs       : $(wc -l < "${SHARD_TSV}")"
echo " SLOTS      : ${SLOTS}"
echo " GPUS       : ${GPU_IDS_CSV}"
echo " FEAT       : ${FEAT_BACKEND}"
echo " CACHE_ROOT : ${AOSS_CACHE_ROOT}"
echo "=============================================="

# Sanity: aoss_fs must import for AOSS raw roots.
"${PYTHON}" - <<'PY'
import importlib
from pathlib import Path
mod = importlib.import_module("vidfm3d.utils.aoss_fs")
src = Path(mod.__file__)
if not hasattr(mod, "is_remote"):
    for stale in src.with_name("__pycache__").glob("aoss_fs*.pyc"):
        stale.unlink(missing_ok=True)
    mod = importlib.reload(mod)
if not hasattr(mod, "is_remote"):
    raise SystemExit(f"[err] aoss_fs missing is_remote at {src} size={src.stat().st_size}")
print("aoss_fs", mod.__file__, "size", src.stat().st_size)
print("is_remote", mod.is_remote("aoss://baiqiao/InsScene-15K/FEAT_STREAMING"))
PY

DRY_RUN=0 \
OUT_DIR="${OUT_DIR}" \
RUN_ID="${RUN_ID}" \
JOBS_TSV="${SHARD_TSV}" \
SLOTS="${SLOTS}" \
GPU_IDS_CSV="${GPU_IDS_CSV}" \
BATCH_SIZE="${BATCH_SIZE}" \
BATCH_SIZE_VAL="${BATCH_SIZE_VAL}" \
NUM_WORKERS="${NUM_WORKERS}" \
NUM_WORKERS_VAL="${NUM_WORKERS_VAL}" \
VIDEO_CHANNELS="${VIDEO_CHANNELS}" \
FEATURE_TIMESTEP="${FEATURE_TIMESTEP}" \
FEAT_BACKEND="${FEAT_BACKEND}" \
ALLOW_MISSING_STREAMING="${ALLOW_MISSING_STREAMING}" \
AOSS_CACHE_ROOT="${AOSS_CACHE_ROOT}" \
bash "${PROJECT_ROOT}/scripts/launch_lingbot_abc_probe.sh"

echo "shard_finished_at=$(date -Is)"
