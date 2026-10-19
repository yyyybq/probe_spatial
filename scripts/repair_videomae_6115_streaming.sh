#!/usr/bin/env bash
# Repair videomae FEAT_STREAMING for scene 6115eddb86, then AOSS-sync that scene only.
#
# Fills:
#   window_0000-0003 × prefix 8/12/16 × layers 0,5,11,17,23
#   + prefix_index.npy
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
cd "${PROJECT_ROOT}"

export PATH="/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-/mnt/umm/users/yinbaiqiao/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
mkdir -p "${MPLCONFIGDIR}"

PYTHON=${PYTHON:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}
DATA_ROOT=${DATA_ROOT:-/mnt/umm/users/yinbaiqiao/InsScene-15K/data}
OUT_STREAM=${OUT_STREAM:-/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_STREAMING}
SCENE_ID=${SCENE_ID:-6115eddb86}
SCENE_START=${SCENE_START:-342}
SCENE_END=${SCENE_END:-343}
SNAP=${SNAP:-/mnt/umm/users/yinbaiqiao/.cache/huggingface/hub/models--MCG-NJU--videomae-large/snapshots/12da269a02d3e1fbbb7011e610e591fea8061dca}
LOG_DIR=${LOG_DIR:-${PROJECT_ROOT}/logs/videomae_6115_repair_$(date -u +%Y%m%d_%H%M%S)}
mkdir -p "${LOG_DIR}"

echo "=============================================="
echo " repair videomae streaming scene=${SCENE_ID}"
echo " host=$(hostname) cuda=$("${PYTHON}" -c 'import torch; print(torch.cuda.device_count())')"
echo " start:end=${SCENE_START}:${SCENE_END}"
echo " out=${OUT_STREAM}"
echo " log=${LOG_DIR}"
echo "=============================================="

# 1) Extract (resume-safe; only missing prefixes/layers are rewritten).
"${PYTHON}" -u -m features.run_inscene15k \
  --vfm videomae \
  --model-id "${SNAP}" \
  --mode streaming_prefix \
  --data-root "${DATA_ROOT}" \
  --out-root "${OUT_STREAM}" \
  --source scannetpp \
  --t 749 \
  --output-layers 0 5 11 17 23 \
  --prefix-lengths 8,12,16 \
  --prefix-max-len 24 \
  --streaming-motion-step 0.35 \
  --streaming-rotation-weight 0.5 \
  --streaming-max-windows-per-scene 4 \
  --start "${SCENE_START}" \
  --end "${SCENE_END}" \
  --no-cache-checksum \
  2>&1 | tee "${LOG_DIR}/extract.log"

# 2) Verify local completeness for p8/12/16.
"${PYTHON}" -u - <<PY | tee "${LOG_DIR}/verify.txt"
from pathlib import Path
scene = Path("${OUT_STREAM}") / "videomae" / "scannetpp" / "${SCENE_ID}"
layers = [
    "feature_layer0.sft",
    "feature_layer5.sft",
    "feature_layer11.sft",
    "feature_layer17.sft",
    "feature_layer23.sft",
]
prefs = ["prefix_000007", "prefix_000011", "prefix_000015"]
miss = []
for wi in range(4):
    w = scene / f"window_{wi:04d}"
    if not w.is_dir():
        miss.append(f"window_{wi:04d}/DIR_MISSING")
        continue
    for pref in prefs:
        for f in layers:
            if not (w / pref / f).exists():
                miss.append(f"window_{wi:04d}/{pref}/{f}")
idx = scene / "prefix_index.npy"
print(f"scene={scene}")
print(f"prefix_index={idx.exists()} size={idx.stat().st_size if idx.exists() else 0}")
print(f"miss_count={len(miss)}")
for m in miss:
    print("MISS", m)
if miss or not idx.exists():
    raise SystemExit(2)
print("VERIFY_OK")
PY

# 3) AOSS sync this scene only.
AOSS_DIR=${AOSS_DIR:-/mnt/umm/users/yinbaiqiao/aoss}
CONF=${CONF:-${AOSS_DIR}/petreloss.conf}
ADS_CLI=${ADS_CLI:-${AOSS_DIR}/ads-cli}
AOSS_API_HOST=${AOSS_API_HOST:-aoss-internal.cn-fz-01.fjscmsapi-oss.com}
unset HOST || true
AOSS_HOST=${AOSS_HOST:-baiqiao.${AOSS_API_HOST}}
export NO_PROXY="localhost,127.0.0.1,::1,${AOSS_HOST},.${AOSS_API_HOST},${AOSS_API_HOST}"
export no_proxy="${NO_PROXY}"

AK="$(awk '
  /^\[fj\]/ {in_fj=1; next}
  /^\[/ {in_fj=0}
  in_fj && /^access_key[[:space:]]*=/ {gsub(/^access_key[[:space:]]*=[[:space:]]*/, ""); print; exit}
' "${CONF}")"
SK="$(awk '
  /^\[fj\]/ {in_fj=1; next}
  /^\[/ {in_fj=0}
  in_fj && /^secret_key[[:space:]]*=/ {gsub(/^secret_key[[:space:]]*=[[:space:]]*/, ""); print; exit}
' "${CONF}")"
if [[ -z "${AK}" || -z "${SK}" ]]; then
  echo "[err] failed to read AOSS credentials from ${CONF}" >&2
  exit 3
fi

SRC="${OUT_STREAM}/videomae/scannetpp/${SCENE_ID}/"
DST="s3://${AK}:${SK}@baiqiao.${AOSS_API_HOST}/InsScene-15K/FEAT_STREAMING/videomae/scannetpp/${SCENE_ID}/"
echo "AOSS sync ${SRC} -> s3://baiqiao.${AOSS_API_HOST}/InsScene-15K/FEAT_STREAMING/videomae/scannetpp/${SCENE_ID}/"
timeout 30m "${ADS_CLI}" -p 1 -l 2 --conntimeout 120 --timeout 600 sync "${SRC}" "${DST}" \
  2>&1 | tee "${LOG_DIR}/aoss_sync.log"

echo "REPAIR_DONE scene=${SCENE_ID} log=${LOG_DIR}"
