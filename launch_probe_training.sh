#!/bin/bash
# Launch 3 probe training runs on single GPUs (NCCL multi-GPU broken on this machine).
# WAN:       GPU 3  (video_channels=1536, batch=4)
# CogVideoX: GPU 5  (video_channels=3072, batch=4)
# V-JEPA2:   GPU 6  (video_channels=1024, batch=8)

set -e
PYTHON=/data/baiqiao/miniconda3/envs/vidfm3d/bin/python
WORKDIR=/nas/baiqiao/VidFM3D
LOG_DIR=$WORKDIR/logs/inscene15k/runs
CONDA_PREFIX=/data/baiqiao/miniconda3/envs/vidfm3d

export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline

cd $WORKDIR

echo "=== Launching WAN probe on GPU 3 ==="
CUDA_VISIBLE_DEVICES=3 \
nohup $PYTHON vidfm3d/train.py experiment=inscene15k/wan_probe \
    > $LOG_DIR/wan_probe.log 2>&1 &
echo "  PID: $!"

echo "=== Launching CogVideoX probe on GPU 5 ==="
CUDA_VISIBLE_DEVICES=5 \
nohup $PYTHON vidfm3d/train.py experiment=inscene15k/cogvideox_probe \
    > $LOG_DIR/cogvideox_probe.log 2>&1 &
echo "  PID: $!"

echo "=== Launching V-JEPA2 probe on GPU 6 ==="
CUDA_VISIBLE_DEVICES=6 \
nohup $PYTHON vidfm3d/train.py experiment=inscene15k/vjepa2_probe \
    > $LOG_DIR/vjepa2_probe.log 2>&1 &
echo "  PID: $!"

echo ""
echo "=== All 3 training runs launched (single GPU each) ==="
echo "Monitor logs:"
echo "  tail -f $LOG_DIR/wan_probe.log"
echo "  tail -f $LOG_DIR/cogvideox_probe.log"
echo "  tail -f $LOG_DIR/vjepa2_probe.log"
