#!/bin/bash
# Resume all 3 v3 training jobs with checkpoint fix + every_n_epochs=5
set -e
cd /nas/baiqiao/VidFM3D
source /data/baiqiao/miniconda3/etc/profile.d/conda.sh
conda activate vidfm3d
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Resuming ALL v3 training at $(date) ==="

# WAN on GPU 3
echo "Launching WAN on GPU 3..."
CUDA_VISIBLE_DEVICES=3 python vidfm3d/train.py experiment=inscene15k/wan_probe_v3 \
    > logs/train_wan_v3_resume3.log 2>&1 &
PID_WAN=$!

# CogVideoX on GPU 5
echo "Launching CogVideoX on GPU 5..."
CUDA_VISIBLE_DEVICES=5 python vidfm3d/train.py experiment=inscene15k/cogvideox_probe_v3 \
    > logs/train_cogvideox_v3_resume3.log 2>&1 &
PID_COG=$!

# V-JEPA2 on GPU 6
echo "Launching V-JEPA2 on GPU 6..."
CUDA_VISIBLE_DEVICES=6 python vidfm3d/train.py experiment=inscene15k/vjepa2_probe_v3 \
    > logs/train_vjepa2_v3_resume3.log 2>&1 &
PID_VJ=$!

echo "PIDs: WAN=$PID_WAN CogVideoX=$PID_COG V-JEPA2=$PID_VJ"
echo "Waiting for all processes..."
wait
echo "All done at $(date)"
