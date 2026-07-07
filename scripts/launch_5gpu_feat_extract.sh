#!/bin/bash
# Launch all 5 feature extraction modes in parallel across 5 H200 GPUs (wan only).
# Each GPU handles one extraction mode; all modes run concurrently.
#
# Usage:
#   bash scripts/launch_5gpu_feat_extract.sh
#
# Logs go to logs/feat_extract_{mode}.log
# Resume-safe: already-extracted scenes are skipped automatically.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET_ROOT="${DATASET_ROOT:-/scratch/by2593/project/InsScene-15K}"
DATA_ROOT="${DATASET_ROOT}/data"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

VFM="${VFM:-wan}"
MODEL_ID="${MODEL_ID:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}"
LAYERS="${LAYERS:-20}"
T="${T:-749}"
PYTHON="${PYTHON:-conda run -n vidfm3d python}"

echo "======================================================"
echo " probe_spatial: 5-GPU parallel feature extraction"
echo " VFM       : ${VFM}"
echo " MODEL_ID  : ${MODEL_ID}"
echo " DATA_ROOT : ${DATA_ROOT}"
echo " DATASET   : ${DATASET_ROOT}"
echo "======================================================"

cd "${PROJECT_ROOT}"

# ── GPU 0: normal features (A1/A2/A3/B1/B2) ──────────────────────────────────
echo "[GPU 0] Starting NORMAL extraction..."
CUDA_VISIBLE_DEVICES=0 ${PYTHON} -m features.run_inscene15k \
    --vfm "${VFM}" --model-id "${MODEL_ID}" \
    --data-root "${DATA_ROOT}" \
    --out-root "${DATASET_ROOT}/FEAT" \
    --t "${T}" --output-layers ${LAYERS} \
    > "${LOG_DIR}/feat_normal.log" 2>&1 &
PID_NORMAL=$!
echo "  PID: ${PID_NORMAL}  log: logs/feat_normal.log"

# ── GPU 1: shuffled features (A3 abnormal branch) ────────────────────────────
echo "[GPU 1] Starting SHUFFLED extraction..."
CUDA_VISIBLE_DEVICES=1 ${PYTHON} -m features.run_inscene15k \
    --vfm "${VFM}" --model-id "${MODEL_ID}" \
    --mode shuffled \
    --data-root "${DATA_ROOT}" \
    --out-root "${DATASET_ROOT}/FEAT_SHUFFLED" \
    --t "${T}" --output-layers ${LAYERS} \
    > "${LOG_DIR}/feat_shuffled.log" 2>&1 &
PID_SHUFFLED=$!
echo "  PID: ${PID_SHUFFLED}  log: logs/feat_shuffled.log"

# ── GPU 2: context_segment features (C1/C2/C3 causal inputs) ─────────────────
echo "[GPU 2] Starting CONTEXT_SEGMENT extraction..."
CUDA_VISIBLE_DEVICES=2 ${PYTHON} -m features.run_inscene15k \
    --vfm "${VFM}" --model-id "${MODEL_ID}" \
    --mode context_segment \
    --data-root "${DATA_ROOT}" \
    --out-root "${DATASET_ROOT}/FEAT_CONTEXT" \
    --t "${T}" --output-layers ${LAYERS} --context-len 76 \
    > "${LOG_DIR}/feat_context.log" 2>&1 &
PID_CONTEXT=$!
echo "  PID: ${PID_CONTEXT}  log: logs/feat_context.log"

# ── GPU 3: target_isolated features (C1/C2/C3 targets) ───────────────────────
echo "[GPU 3] Starting TARGET_ISOLATED extraction..."
CUDA_VISIBLE_DEVICES=3 ${PYTHON} -m features.run_inscene15k \
    --vfm "${VFM}" --model-id "${MODEL_ID}" \
    --mode target_isolated \
    --data-root "${DATA_ROOT}" \
    --out-root "${DATASET_ROOT}/FEAT_TARGET" \
    --t "${T}" --output-layers ${LAYERS} --num-targets 0 \
    > "${LOG_DIR}/feat_target.log" 2>&1 &
PID_TARGET=$!
echo "  PID: ${PID_TARGET}  log: logs/feat_target.log"

# ── GPU 4: streaming_prefix features (streaming A/B/C probes) ────────────────
echo "[GPU 4] Starting STREAMING_PREFIX extraction..."
CUDA_VISIBLE_DEVICES=4 ${PYTHON} -m features.run_inscene15k \
    --vfm "${VFM}" --model-id "${MODEL_ID}" \
    --mode streaming_prefix \
    --data-root "${DATA_ROOT}" \
    --out-root "${DATASET_ROOT}/FEAT_STREAMING" \
    --t "${T}" --output-layers ${LAYERS} \
    --prefix-lengths "4,8,16,32,64" --prefix-max-len 64 \
    > "${LOG_DIR}/feat_streaming.log" 2>&1 &
PID_STREAMING=$!
echo "  PID: ${PID_STREAMING}  log: logs/feat_streaming.log"

echo ""
echo "All 5 extractors launched. PIDs:"
echo "  normal=${PID_NORMAL} shuffled=${PID_SHUFFLED} context=${PID_CONTEXT}"
echo "  target=${PID_TARGET} streaming=${PID_STREAMING}"
echo ""
echo "Monitor with:"
echo "  tail -f logs/feat_normal.log"
echo "  tail -f logs/feat_shuffled.log logs/feat_context.log logs/feat_target.log logs/feat_streaming.log"

# Wait for all and report
wait ${PID_NORMAL}   && echo "[DONE] normal"    || echo "[FAIL] normal (rc=$?)"
wait ${PID_SHUFFLED} && echo "[DONE] shuffled"  || echo "[FAIL] shuffled (rc=$?)"
wait ${PID_CONTEXT}  && echo "[DONE] context"   || echo "[FAIL] context (rc=$?)"
wait ${PID_TARGET}   && echo "[DONE] target"    || echo "[FAIL] target (rc=$?)"
wait ${PID_STREAMING}&& echo "[DONE] streaming" || echo "[FAIL] streaming (rc=$?)"

echo "======================================================"
echo "All feature extraction jobs complete."
echo "======================================================"
