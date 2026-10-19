#!/bin/bash
# Extract MiniMax-H3 streaming_prefix + target_isolated features (Wan-style).
#
# Requires the minimax-h3 conda env (Diffusers modular MiniMax-H3) and local weights.
#
# Usage:
#   DATA_ROOT=/path/to/InsScene-15K/data \
#   OUT_STREAM=/path/to/FEAT_STREAMING \
#   OUT_TARGET=/path/to/FEAT_TARGET \
#   bash scripts/run_minimax_h3_streaming_extract.sh
#
# Optional:
#   LAYERS="10 20 30 40 50"  # default; 50 maps to last block (49)
#   T=250                    # continuous t=0.250
#   START=0 END=10
#   CUDA_VISIBLE_DEVICES=0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${PYTHON:-/mnt/umm/users/yinbaiqiao/.conda/envs/minimax-h3/bin/python}"
MODEL_ID="${MODEL_ID:-/mnt/umm/users/yinbaiqiao/Consistency/minimax-h3/models/MiniMax-H3}"
DATA_ROOT="${DATA_ROOT:-${INSCENE_DATA_ROOT:-}/data}"
OUT_STREAM="${OUT_STREAM:-${INSCENE_STREAMING_FEAT_ROOT:-}}"
OUT_TARGET="${OUT_TARGET:-${INSCENE_TARGET_FEAT_ROOT:-}}"
LAYERS="${LAYERS:-10 20 30 40 50}"
T="${T:-250}"
PREFIX_LENGTHS="${PREFIX_LENGTHS:-8,12,16,24}"
SOURCE="${SOURCE:-scannetpp}"
START="${START:-0}"
END="${END:-}"

if [[ -z "${DATA_ROOT}" || -z "${OUT_STREAM}" || -z "${OUT_TARGET}" ]]; then
  echo "ERROR: set DATA_ROOT, OUT_STREAM, OUT_TARGET (or INSCENE_* env vars)." >&2
  exit 1
fi

COMMON=(
  -m features.run_inscene15k
  --vfm minimax_h3
  --model-id "${MODEL_ID}"
  --data-root "${DATA_ROOT}"
  --source "${SOURCE}"
  --t "${T}"
  --output-layers ${LAYERS}
  --prefix-lengths "${PREFIX_LENGTHS}"
  --prefix-max-len 24
  --start "${START}"
  --prompt "."
)
if [[ -n "${END}" ]]; then
  COMMON+=(--end "${END}")
fi

echo "[1/2] streaming_prefix -> ${OUT_STREAM}"
"${PYTHON}" "${COMMON[@]}" \
  --mode streaming_prefix \
  --out-root "${OUT_STREAM}"

echo "[2/2] target_isolated -> ${OUT_TARGET}"
"${PYTHON}" "${COMMON[@]}" \
  --mode target_isolated \
  --out-root "${OUT_TARGET}" \
  --target-from-streaming-windows \
  --target-prefix-lengths "${PREFIX_LENGTHS}" \
  --target-horizons "1,2,4"

echo "Done."
