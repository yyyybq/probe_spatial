#!/bin/bash
# Resume v3 training to 100 epochs - clean launch
set -e
cd /nas/baiqiao/VidFM3D
source /data/baiqiao/miniconda3/etc/profile.d/conda.sh
conda activate vidfm3d
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Resuming v3 training at $(date) ==="

# WAN on GPU 3 (batch_size=3)
echo "Launching WAN on GPU 3..."
CUDA_VISIBLE_DEVICES=3 python vidfm3d/train.py experiment=inscene15k/wan_probe_v3 \
    > logs/train_wan_v3_resume.log 2>&1 &
PID_WAN=$!

# CogVideoX on GPU 5 (batch_size=4)
echo "Launching CogVideoX on GPU 5..."
CUDA_VISIBLE_DEVICES=5 python vidfm3d/train.py experiment=inscene15k/cogvideox_probe_v3 \
    > logs/train_cogvideox_v3_resume.log 2>&1 &
PID_COG=$!

# V-JEPA2 on GPU 6 (batch_size=8)
echo "Launching V-JEPA2 on GPU 6..."
CUDA_VISIBLE_DEVICES=6 python vidfm3d/train.py experiment=inscene15k/vjepa2_probe_v3 \
    > logs/train_vjepa2_v3_resume.log 2>&1 &
PID_VJ=$!

echo "PIDs: WAN=$PID_WAN CogVideoX=$PID_COG V-JEPA2=$PID_VJ"
echo "Logs: logs/train_{wan,cogvideox,vjepa2}_v3_resume.log"

# Wait for all
wait $PID_WAN $PID_COG $PID_VJ
echo "All done at $(date)"
