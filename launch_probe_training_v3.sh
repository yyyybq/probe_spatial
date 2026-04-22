#!/bin/bash
# v3 training: no point head, window_size=200, 100 epochs
# 3 VFMs on GPUs 3, 5, 6 (single-GPU each)

set -e
cd /nas/baiqiao/VidFM3D
source /data/baiqiao/miniconda3/etc/profile.d/conda.sh
conda activate vidfm3d
export WANDB_MODE=offline

echo "=== Starting v3 training (no point head, window_size=200, 100 epochs) ==="
echo "Start time: $(date)"

# WAN on GPU 3
echo "[$(date)] Launching WAN on GPU 3..."
CUDA_VISIBLE_DEVICES=3 nohup python vidfm3d/train.py \
    experiment=inscene15k/wan_probe_v3 \
    > logs/train_wan_v3.log 2>&1 &
PID_WAN=$!
echo "  WAN PID: $PID_WAN"

# CogVideoX on GPU 5
echo "[$(date)] Launching CogVideoX on GPU 5..."
CUDA_VISIBLE_DEVICES=5 nohup python vidfm3d/train.py \
    experiment=inscene15k/cogvideox_probe_v3 \
    > logs/train_cogvideox_v3.log 2>&1 &
PID_COG=$!
echo "  CogVideoX PID: $PID_COG"

# V-JEPA2 on GPU 6
echo "[$(date)] Launching V-JEPA2 on GPU 6..."
CUDA_VISIBLE_DEVICES=6 nohup python vidfm3d/train.py \
    experiment=inscene15k/vjepa2_probe_v3 \
    > logs/train_vjepa2_v3.log 2>&1 &
PID_VJEPA=$!
echo "  V-JEPA2 PID: $PID_VJEPA"

echo ""
echo "All jobs launched."
echo "  WAN:       PID=$PID_WAN,  GPU=3, log=logs/train_wan_v3.log"
echo "  CogVideoX: PID=$PID_COG,  GPU=5, log=logs/train_cogvideox_v3.log"
echo "  V-JEPA2:   PID=$PID_VJEPA, GPU=6, log=logs/train_vjepa2_v3.log"
echo ""
echo "Monitor: tail -f logs/train_wan_v3.log logs/train_cogvideox_v3.log logs/train_vjepa2_v3.log"
