#!/usr/bin/env bash
# Shard V-JEPA2 + VideoMAE streaming_prefix multilayer extraction across 8 GPUs.
#
# Layout (default):
#   GPU 0-3 -> vjepa2  (layers 0,5,11,17,23)
#   GPU 4-7 -> videomae (layers 0,5,11,17,23)
#
# Usage (on an 8-GPU SCO worker):
#   bash scripts/run_vjepa2_videomae_streaming_8gpu_extract.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PATH="/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/mnt/umm/users/yinbaiqiao/.cache/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
mkdir -p "${MPLCONFIGDIR}"

PYTHON="${PYTHON:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/umm/users/yinbaiqiao/InsScene-15K/data}"
OUT_STREAM="${OUT_STREAM:-/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_STREAMING}"
LAYERS_VJEPA2="${LAYERS_VJEPA2:-0 5 11 17 23}"
LAYERS_VIDEOMAE="${LAYERS_VIDEOMAE:-0 5 11 17 23}"
T="${T:-749}"
PREFIX_LENGTHS="${PREFIX_LENGTHS:-8,12,16,24}"
SOURCE="${SOURCE:-scannetpp}"
N_GPUS="${N_GPUS:-8}"
GPUS_PER_VFM="${GPUS_PER_VFM:-4}"
ONLY_VFMS="${ONLY_VFMS:-vjepa2,videomae}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/vjepa2_videomae_streaming_extract_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${OUT_STREAM}" "${LOG_DIR}"
echo "log_dir=${LOG_DIR}" | tee "${LOG_DIR}/supervisor.log"

{
  echo "=============================================="
  echo " V-JEPA2 + VideoMAE streaming_prefix extract"
  echo " python     : ${PYTHON}"
  echo " data_root  : ${DATA_ROOT}"
  echo " out_stream : ${OUT_STREAM}"
  echo " layers_vj  : ${LAYERS_VJEPA2}"
  echo " layers_vm  : ${LAYERS_VIDEOMAE}"
  echo " prefixes   : ${PREFIX_LENGTHS}"
  echo " only_vfms  : ${ONLY_VFMS}"
  echo " n_gpus     : ${N_GPUS} (per_vfm=${GPUS_PER_VFM})"
  echo " log_dir    : ${LOG_DIR}"
  echo "=============================================="
  which "${PYTHON}"
  "${PYTHON}" -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.device_count())'
  "${PYTHON}" -c 'from features.run_inscene15k import videomae_forward; print("videomae_forward: ok")'
} | tee -a "${LOG_DIR}/supervisor.log"

N_SCENES="$(
  "${PYTHON}" - <<PY
import os, sys
sys.path.insert(0, "${PROJECT_ROOT}")
from features.run_inscene15k import collect_scannetpp_scenes
root = os.path.join("${DATA_ROOT}", "processed_scannetpp_v2")
print(len(collect_scannetpp_scenes(root)))
PY
)"
echo "Total scannetpp scenes: ${N_SCENES}" | tee -a "${LOG_DIR}/supervisor.log"
echo "${N_SCENES}" > "${LOG_DIR}/n_scenes.txt"

want_vfm() {
  local vfm="$1"
  [[ ",${ONLY_VFMS}," == *",${vfm},"* ]]
}

n_wanted=0
want_vfm vjepa2 && n_wanted=$((n_wanted + 1))
want_vfm videomae && n_wanted=$((n_wanted + 1))
if (( n_wanted < 1 )); then
  echo "[err] ONLY_VFMS=${ONLY_VFMS} selected no models" | tee -a "${LOG_DIR}/supervisor.log" >&2
  exit 1
fi
if (( N_GPUS < GPUS_PER_VFM * n_wanted )); then
  echo "[err] need at least $((GPUS_PER_VFM * n_wanted)) GPUs for ONLY_VFMS=${ONLY_VFMS}, have ${N_GPUS}" | tee -a "${LOG_DIR}/supervisor.log" >&2
  exit 1
fi

launch_vfm() {
  local vfm="$1"
  local layers="$2"
  local gpu_begin="$3"
  local n_shards="$4"
  local gpu_end=$((gpu_begin + n_shards))
  local i=0
  local gpu start end log
  for ((gpu=gpu_begin; gpu<gpu_end; gpu++)); do
    start=$(( N_SCENES * i / n_shards ))
    end=$(( N_SCENES * (i + 1) / n_shards ))
    i=$((i + 1))
    if [[ "${start}" -ge "${end}" ]]; then
      echo "[${vfm} gpu ${gpu}] empty shard, skip" | tee -a "${LOG_DIR}/supervisor.log"
      continue
    fi
    log="${LOG_DIR}/${vfm}_gpu${gpu}_${start}_${end}.log"
    echo "[${vfm} gpu ${gpu}] scenes [${start}:${end}) -> ${log}" | tee -a "${LOG_DIR}/supervisor.log"
    (
      set -euo pipefail
      export CUDA_VISIBLE_DEVICES="${gpu}"
      export OMP_NUM_THREADS=4
      echo "[$(date -Is)] vfm=${vfm} gpu=${gpu} begin streaming_prefix [${start}:${end}) layers=${layers}"
      "${PYTHON}" -m features.run_inscene15k \
        --vfm "${vfm}" \
        --mode streaming_prefix \
        --data-root "${DATA_ROOT}" \
        --out-root "${OUT_STREAM}" \
        --source "${SOURCE}" \
        --t "${T}" \
        --output-layers ${layers} \
        --prefix-lengths "${PREFIX_LENGTHS}" \
        --prefix-max-len 24 \
        --streaming-motion-step 0.35 \
        --streaming-rotation-weight 0.5 \
        --start "${start}" \
        --end "${end}" \
        --no-cache-checksum
      echo "[$(date -Is)] vfm=${vfm} gpu=${gpu} finished=ok"
    ) > "${log}" 2>&1 &
    echo $! > "${LOG_DIR}/${vfm}_gpu${gpu}.pid"
  done
}

gpu_cursor=0
if want_vfm vjepa2; then
  launch_vfm "vjepa2" "${LAYERS_VJEPA2}" "${gpu_cursor}" "${GPUS_PER_VFM}"
  gpu_cursor=$((gpu_cursor + GPUS_PER_VFM))
fi
if want_vfm videomae; then
  launch_vfm "videomae" "${LAYERS_VIDEOMAE}" "${gpu_cursor}" "${GPUS_PER_VFM}"
fi

fail=0
for pidfile in "${LOG_DIR}"/*.pid; do
  [[ -f "${pidfile}" ]] || continue
  pid="$(cat "${pidfile}")"
  if ! wait "${pid}"; then
    echo "[err] worker pid=${pid} failed (${pidfile})" | tee -a "${LOG_DIR}/supervisor.log"
    fail=1
  fi
done

echo "finished_at=$(date -Is) fail=${fail} log_dir=${LOG_DIR}" | tee -a "${LOG_DIR}/supervisor.log"

"${PYTHON}" - <<PY | tee -a "${LOG_DIR}/supervisor.log"
from pathlib import Path
root = Path("${OUT_STREAM}")
layers_map = {
  "vjepa2": [0,5,11,17,23],
  "videomae": [0,5,11,17,23],
}
for vfm, layers in layers_map.items():
  base = root / vfm / "scannetpp"
  if not base.is_dir():
    print(vfm, "scene_dirs=0")
    continue
  scenes = [p for p in base.iterdir() if p.is_dir()]
  complete = 0
  for sc in scenes:
    prefs = [p for p in sc.iterdir() if p.is_dir() and p.name.startswith("prefix_")]
    ok = bool(prefs)
    for pref in prefs:
      for layer in layers:
        if not (pref / f"feature_layer{layer}.sft").exists():
          ok = False
          break
      if not ok:
        break
    if ok:
      complete += 1
  print(f"{vfm} scene_dirs={len(scenes)} complete_all_prefix_layers={complete}")
PY

exit "${fail}"
