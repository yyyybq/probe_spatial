#!/usr/bin/env bash
# Shard MiniMax-H3 streaming_prefix + target_isolated extraction across 8 GPUs.
#
# Intended to run inside an SCO 8-GPU worker. Each GPU owns a scene range and
# extracts layers 10/20/30/40/50(=49) for all streaming prefixes, then targets.
#
# Usage (on the node):
#   bash scripts/run_minimax_h3_8gpu_extract.sh
#
# Env overrides:
#   DATA_ROOT OUT_STREAM OUT_TARGET MODEL_ID LAYERS T PREFIX_LENGTHS N_GPUS

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PATH="/mnt/umm/users/yinbaiqiao/.conda/envs/minimax-h3/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-/mnt/umm/shared_model/huggingface/}"

PYTHON="${PYTHON:-/mnt/umm/users/yinbaiqiao/.conda/envs/minimax-h3/bin/python}"
MODEL_ID="${MODEL_ID:-/mnt/umm/users/yinbaiqiao/Consistency/minimax-h3/models/MiniMax-H3}"
DATA_ROOT="${DATA_ROOT:-/mnt/umm/users/yinbaiqiao/InsScene-15K/data}"
OUT_STREAM="${OUT_STREAM:-/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_STREAMING}"
OUT_TARGET="${OUT_TARGET:-/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_TARGET}"
LAYERS="${LAYERS:-10 20 30 40 50}"
T="${T:-250}"
PREFIX_LENGTHS="${PREFIX_LENGTHS:-8,12,16,24}"
SOURCE="${SOURCE:-scannetpp}"
N_GPUS="${N_GPUS:-8}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/minimax_h3_feat_extract_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${OUT_STREAM}" "${OUT_TARGET}" "${LOG_DIR}"

echo "=============================================="
echo " MiniMax-H3 8-GPU feature extraction"
echo " python     : ${PYTHON}"
echo " model      : ${MODEL_ID}"
echo " data_root  : ${DATA_ROOT}"
echo " out_stream : ${OUT_STREAM}"
echo " out_target : ${OUT_TARGET}"
echo " layers     : ${LAYERS}"
echo " t          : ${T}"
echo " prefixes   : ${PREFIX_LENGTHS}"
echo " n_gpus     : ${N_GPUS}"
echo " log_dir    : ${LOG_DIR}"
echo "=============================================="
which python
python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.device_count())'

# Discover scene count with the same collector as the extractor.
N_SCENES="$(
  "${PYTHON}" - <<PY
import os, sys
sys.path.insert(0, "${PROJECT_ROOT}")
from features.run_inscene15k import collect_scannetpp_scenes
root = os.path.join("${DATA_ROOT}", "processed_scannetpp_v2")
print(len(collect_scannetpp_scenes(root)))
PY
)"
echo "Total scannetpp scenes: ${N_SCENES}"

pids=()
for gpu in $(seq 0 $((N_GPUS - 1))); do
  start=$(( N_SCENES * gpu / N_GPUS ))
  end=$(( N_SCENES * (gpu + 1) / N_GPUS ))
  if [[ "${start}" -ge "${end}" ]]; then
    echo "[gpu ${gpu}] empty shard, skip"
    continue
  fi
  log="${LOG_DIR}/gpu${gpu}_${start}_${end}.log"
  echo "[gpu ${gpu}] scenes [${start}:${end}) -> ${log}"
  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="${gpu}"
    echo "[$(date -Is)] gpu=${gpu} start streaming_prefix [${start}:${end})"
    "${PYTHON}" -m features.run_inscene15k \
      --vfm minimax_h3 \
      --model-id "${MODEL_ID}" \
      --mode streaming_prefix \
      --data-root "${DATA_ROOT}" \
      --out-root "${OUT_STREAM}" \
      --source "${SOURCE}" \
      --t "${T}" \
      --output-layers ${LAYERS} \
      --prefix-lengths "${PREFIX_LENGTHS}" \
      --prefix-max-len 24 \
      --prompt "." \
      --start "${start}" \
      --end "${end}"
    echo "[$(date -Is)] gpu=${gpu} start target_isolated [${start}:${end})"
    "${PYTHON}" -m features.run_inscene15k \
      --vfm minimax_h3 \
      --model-id "${MODEL_ID}" \
      --mode target_isolated \
      --data-root "${DATA_ROOT}" \
      --out-root "${OUT_TARGET}" \
      --source "${SOURCE}" \
      --t "${T}" \
      --output-layers ${LAYERS} \
      --prefix-lengths "${PREFIX_LENGTHS}" \
      --prefix-max-len 24 \
      --target-from-streaming-windows \
      --target-prefix-lengths "${PREFIX_LENGTHS}" \
      --target-horizons "1,2,4" \
      --prompt "." \
      --start "${start}" \
      --end "${end}"
    echo "[$(date -Is)] gpu=${gpu} DONE"
  ) >"${log}" 2>&1 &
  pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    fail=1
  fi
done

echo "All GPU workers finished. logs: ${LOG_DIR}"
if [[ "${fail}" -ne 0 ]]; then
  echo "ERROR: at least one GPU worker failed" >&2
  exit 1
fi
echo "OK"
