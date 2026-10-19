#!/usr/bin/env bash
# Gapfill FEAT_STREAMING window_0003 for scene 48da625e8c on VFMs that only have
# w0-w2 (obs=24). Aligns windows with videomae by extracting with prefix lengths
# 8,12,16 (obs=16 -> 4 windows), then merges prefix_index so existing AOSS w0-w2
# records are preserved and only window_0003 is uploaded.
#
# Usage:
#   bash scripts/gapfill_48da_window0003.sh              # all missing VFMs
#   VFMS="vjepa2 wan" bash scripts/gapfill_48da_window0003.sh
#   SKIP_EXTRACT=1 bash scripts/gapfill_48da_window0003.sh  # merge+sync only
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
cd "${PROJECT_ROOT}"

DATA_ROOT=${DATA_ROOT:-/mnt/umm/users/yinbaiqiao/InsScene-15K/data}
OUT_STREAM=${OUT_STREAM:-/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_STREAMING}
SCENE_ID=${SCENE_ID:-48da625e8c}
# Filtered scannetpp list (collect_scannetpp_scenes) has 682 scenes; 48da625e8c is idx 276.
# Raw dir listing index 277 is wrong (skips .vscode + c24f94007b).
SCENE_START=${SCENE_START:-276}
SCENE_END=${SCENE_END:-277}
SOURCE=${SOURCE:-scannetpp}
PREFIX_LENGTHS=${PREFIX_LENGTHS:-8,12,16}
VFMS=${VFMS:-vjepa2 wan cogvideox lingbot minimax_h3}
SKIP_EXTRACT=${SKIP_EXTRACT:-0}
SKIP_SYNC=${SKIP_SYNC:-0}
LOG_DIR=${LOG_DIR:-${PROJECT_ROOT}/logs/gapfill_48da_w3_$(date -u +%Y%m%d_%H%M%S)}
mkdir -p "${LOG_DIR}"

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-/mnt/umm/users/yinbaiqiao/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
mkdir -p "${MPLCONFIGDIR}"

PY_PROBE=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python
PY_LINGBOT=/mnt/umm/users/yinbaiqiao/.conda/envs/lingbot-video/bin/python
PY_MINIMAX=/mnt/umm/users/yinbaiqiao/.conda/envs/minimax-h3/bin/python

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

vfm_python() {
  case "$1" in
    lingbot) echo "${PY_LINGBOT}" ;;
    minimax_h3) echo "${PY_MINIMAX}" ;;
    *) echo "${PY_PROBE}" ;;
  esac
}

vfm_layers() {
  case "$1" in
    vjepa2) echo "0 5 11 17 23" ;;
    wan) echo "0 5 10 15 25 29" ;;
    cogvideox) echo "0 5 10 15 25 30 35 40" ;;
    lingbot) echo "10 20 30 40 47" ;;
    minimax_h3) echo "10 20 30 40 49" ;;
    *) echo ""; return 1 ;;
  esac
}

vfm_extra_args() {
  case "$1" in
    wan)
      echo --model-id /mnt/umm/users/yinbaiqiao/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780a534b6463e45facd96134c9f345acfa5b --t 749 --prompt .
      ;;
    cogvideox)
      # Must end with "I2V" so run_inscene15k selects the I2V feature path.
      # Local snapshot is already in HF cache; disable hub-offline for this call.
      echo --model-id THUDM/CogVideoX-5b-I2V --t 749
      ;;
    lingbot)
      echo --model-id /mnt/umm/users/yinbaiqiao/Consistency/lingbot-video/models/lingbot-video-moe-30b-a3b --t 250 --prompt .
      ;;
    minimax_h3)
      echo --model-id /mnt/umm/users/yinbaiqiao/Consistency/minimax-h3/models/MiniMax-H3 --t 250 --prompt .
      ;;
    vjepa2)
      echo --t 0
      ;;
  esac
}

merge_prefix_index() {
  local vfm="$1"
  local scene_dir="${OUT_STREAM}/${vfm}/${SOURCE}/${SCENE_ID}"
  local local_idx="${scene_dir}/prefix_index.npy"
  local aoss_uri="aoss://baiqiao/InsScene-15K/FEAT_STREAMING/${vfm}/${SOURCE}/${SCENE_ID}/prefix_index.npy"
  "${PY_PROBE}" - <<PY
import os, shutil, numpy as np
from pathlib import Path
from vidfm3d.utils.aoss_fs import load_npy

scene = Path("${scene_dir}")
local_path = scene / "prefix_index.npy"
bak = scene / "prefix_index.npy.gapfill_bak"
aoss_uri = "${aoss_uri}"

old = load_npy(aoss_uri, allow_pickle=True)
new = np.load(local_path, allow_pickle=True)

def as_dict(r):
    return r.item() if hasattr(r, "item") else dict(r)

old_keep = [r for r in old if int(as_dict(r).get("window_id", -1)) != 3]
new_w3 = [r for r in new if int(as_dict(r).get("window_id", -1)) == 3]
if not new_w3:
    raise SystemExit(f"no window_id=3 records in local index for ${vfm}")

merged = np.array(old_keep + new_w3, dtype=object)
if bak.exists():
    bak.unlink()
shutil.copy2(local_path, bak)
np.save(local_path, merged, allow_pickle=True)
print(f"${vfm}: merged prefix_index old_keep={len(old_keep)} + w3={len(new_w3)} -> {len(merged)}")
for r in new_w3:
    d = as_dict(r)
    wi = d.get("window_indices") or d.get("indices")
    print(f"  w3 tail={d.get('tail')} obs_start={d.get('obs_start')} win=[{wi[0]}..{wi[-1]}] len={len(wi)}")
PY
}

verify_local_w3() {
  local vfm="$1"
  local layers
  layers="$(vfm_layers "${vfm}")"
  local scene_dir="${OUT_STREAM}/${vfm}/${SOURCE}/${SCENE_ID}"
  "${PY_PROBE}" - <<PY
from pathlib import Path
scene = Path("${scene_dir}")
w3 = scene / "window_0003"
prefs = ["prefix_000007", "prefix_000011", "prefix_000015"]
# map layer ids -> filenames present under one existing prefix if possible
layer_ids = [int(x) for x in "${layers}".split()]
miss = []
# discover naming from any existing feature under w0 if present, else guess
sample = None
for p in (scene / "window_0000" / "prefix_000007").glob("feature_*.sft"):
    sample = p.name
    break
if sample is None:
    # fall back: look inside w3 after extract
    for p in (w3 / "prefix_000007").glob("feature_*.sft"):
        sample = p.name
        break

def fname(layer):
    if sample is None:
        return f"feature_layer{layer}.sft"
    import re
    return re.sub(r"layer\d+", f"layer{layer}", sample)

if not w3.is_dir():
    raise SystemExit("window_0003 missing")
for pref in prefs:
    for lid in layer_ids:
        f = w3 / pref / fname(lid)
        if not f.exists():
            miss.append(str(f.relative_to(scene)))
idx = scene / "prefix_index.npy"
print(f"${vfm}: w3_ok miss={len(miss)} idx={idx.exists()}")
for m in miss[:20]:
    print("MISS", m)
if miss or not idx.exists():
    raise SystemExit(2)
print("VERIFY_OK")
PY
}

sync_w3_only() {
  local vfm="$1"
  local scene_dir="${OUT_STREAM}/${vfm}/${SOURCE}/${SCENE_ID}"
  local remote_base="s3://${AK}:${SK}@baiqiao.${AOSS_API_HOST}/InsScene-15K/FEAT_STREAMING/${vfm}/${SOURCE}/${SCENE_ID}"
  local log="${LOG_DIR}/aoss_sync_${vfm}.log"
  local staging="${LOG_DIR}/aoss_staging_${vfm}"

  echo "[sync] ${vfm} window_0003 + prefix_index.npy"
  # New window dir: sync is fine (objects absent on dst).
  timeout 30m "${ADS_CLI}" -p 1 -l 2 --conntimeout 120 --timeout 600 \
    sync "${scene_dir}/window_0003/" "${remote_base}/window_0003/" \
    2>&1 | tee -a "${log}"

  # prefix_index.npy already exists on AOSS and must be overwritten -> use copy.
  rm -rf "${staging}"
  mkdir -p "${staging}"
  cp -a "${scene_dir}/prefix_index.npy" "${staging}/"
  if [[ -f "${scene_dir}/prefix_index.npy.manifest.json" ]]; then
    cp -a "${scene_dir}/prefix_index.npy.manifest.json" "${staging}/"
  fi
  timeout 10m "${ADS_CLI}" -p 1 -l 2 --conntimeout 120 --timeout 600 \
    copy "${staging}/" "${remote_base}/" \
    2>&1 | tee -a "${log}"
}

extract_one() {
  local vfm="$1"
  local py layers
  py="$(vfm_python "${vfm}")"
  layers="$(vfm_layers "${vfm}")"
  local log="${LOG_DIR}/extract_${vfm}.log"
  echo "=============================================="
  echo " extract ${vfm} scene=${SCENE_ID} [${SCENE_START}:${SCENE_END})"
  echo " python=${py}"
  echo " layers=${layers}"
  echo " prefixes=${PREFIX_LENGTHS} (obs=max -> 16 windows like videomae)"
  echo " log=${log}"
  echo "=============================================="

  # Cog I2V needs model_id ending in "I2V"; allow HF cache hits without network.
  local hub_offline_restore="${HF_HUB_OFFLINE-}"
  local transformers_offline_restore="${TRANSFORMERS_OFFLINE-}"
  if [[ "${vfm}" == "cogvideox" ]]; then
    export HF_HUB_OFFLINE=0
    export TRANSFORMERS_OFFLINE=0
    export HF_HUB_DISABLE_TELEMETRY=1
  fi

  # shellcheck disable=SC2046
  "${py}" -u -m features.run_inscene15k \
    --vfm "${vfm}" \
    --mode streaming_prefix \
    --data-root "${DATA_ROOT}" \
    --out-root "${OUT_STREAM}" \
    --source "${SOURCE}" \
    --output-layers ${layers} \
    --prefix-lengths "${PREFIX_LENGTHS}" \
    --prefix-max-len 24 \
    --streaming-motion-step 0.35 \
    --streaming-rotation-weight 0.5 \
    --streaming-max-windows-per-scene 4 \
    --start "${SCENE_START}" \
    --end "${SCENE_END}" \
    --no-cache-checksum \
    $(vfm_extra_args "${vfm}") \
    2>&1 | tee "${log}"

  if [[ "${vfm}" == "cogvideox" ]]; then
    if [[ -n "${hub_offline_restore}" ]]; then export HF_HUB_OFFLINE="${hub_offline_restore}"; else unset HF_HUB_OFFLINE; fi
    if [[ -n "${transformers_offline_restore}" ]]; then export TRANSFORMERS_OFFLINE="${transformers_offline_restore}"; else unset TRANSFORMERS_OFFLINE; fi
  fi

  local scene_dir="${OUT_STREAM}/${vfm}/${SOURCE}/${SCENE_ID}"
  if [[ ! -d "${scene_dir}" ]]; then
    echo "[err] expected extract output missing: ${scene_dir}" >&2
    echo "[err] check SCENE_START/END against filtered collect_scannetpp_scenes index" >&2
    rg -n "scannetpp/" "${log}" | tail -5 >&2 || true
    exit 4
  fi
  if ! rg -q "scannetpp/${SCENE_ID}:" "${log}"; then
    echo "[err] extract log did not process ${SCENE_ID}; abort before merge/sync" >&2
    rg -n "scannetpp/" "${log}" | tail -10 >&2 || true
    exit 4
  fi
  if rg -q "FAILED scannetpp/${SCENE_ID}" "${log}"; then
    echo "[err] extract reported FAILED for ${SCENE_ID}" >&2
    exit 4
  fi

  # Drop locally regenerated w0-w2 so we never accidentally sync them over AOSS.
  for wi in 0 1 2; do
    rm -rf "${scene_dir}/window_$(printf '%04d' "${wi}")"
  done
}

echo "gapfill start host=$(hostname) log=${LOG_DIR} vfms=${VFMS}"
for vfm in ${VFMS}; do
  if [[ "${SKIP_EXTRACT}" != "1" ]]; then
    extract_one "${vfm}"
  fi
  merge_prefix_index "${vfm}"
  verify_local_w3 "${vfm}"
  if [[ "${SKIP_SYNC}" != "1" ]]; then
    sync_w3_only "${vfm}"
  fi
done

echo "GAPFILL_DONE log=${LOG_DIR}"
export GAPFILL_VFMS="${VFMS}"
"${PY_PROBE}" - <<'PY'
import os
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1,baiqiao.aoss-internal.cn-fz-01.fjscmsapi-oss.com,aoss-internal.cn-fz-01.fjscmsapi-oss.com")
from vidfm3d.utils.aoss_fs import load_npy, exists
base = "aoss://baiqiao/InsScene-15K/FEAT_STREAMING"
vfms = os.environ.get("GAPFILL_VFMS", "").split()
for vfm in vfms:
    root = f"{base}/{vfm}/scannetpp/48da625e8c"
    idx = load_npy(f"{root}/prefix_index.npy", allow_pickle=True)
    wids = sorted({
        (r.item() if hasattr(r, "item") else r).get("window_id")
        for r in idx
    })
    has_w3 = exists(f"{root}/window_0003")
    print(f"AOSS {vfm}: windows={wids} has_window_0003={has_w3} n={len(idx)}")
PY
