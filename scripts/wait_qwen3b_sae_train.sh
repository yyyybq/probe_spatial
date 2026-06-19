#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/nas/baiqiao/probe_spatial}
FEAT_DIR=${FEAT_DIR:-${INSCENE_MLLM_FEAT_ROOT:?set FEAT_DIR or INSCENE_MLLM_FEAT_ROOT}/qwen2_5_vl_3b}
TOTAL=${TOTAL:-2371}
CHECK_INTERVAL=${CHECK_INTERVAL:-300}
GPU=${GPU:-4}
PYTHON=${PYTHON:-/data/baiqiao/miniconda3/envs/vidfm3d/bin/python}
LOG_DIR=${LOG_DIR:-$ROOT/logs/qwen_probe}
TRAIN_LOG=${TRAIN_LOG:-$LOG_DIR/train_sae_qwen2_5vl_3b.log}

mkdir -p "$LOG_DIR"
echo "[$(date)] Waiting for $TOTAL Qwen2.5-VL-3B feature files under $FEAT_DIR"

while true; do
  n=$(find "$FEAT_DIR" -name 'feature_layer-1.sft' 2>/dev/null | wc -l)
  echo "[$(date)] features_ready=$n/$TOTAL"
  if [ "$n" -ge "$TOTAL" ]; then
    break
  fi
  sleep "$CHECK_INTERVAL"
done

echo "[$(date)] Feature extraction complete; starting SAE probe training on GPU $GPU"
cd "$ROOT"
MPLCONFIGDIR=/tmp/matplotlib-qwen3b \
CUDA_VISIBLE_DEVICES="$GPU" \
"$PYTHON" vidfm3d/train.py \
  experiment=inscene15k_ext/sae_qwen2_5vl_3b_v1 \
  logger.wandb.offline=true \
  > "$TRAIN_LOG" 2>&1
echo "[$(date)] SAE probe training finished with exit code $?"
