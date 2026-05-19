#!/usr/bin/env bash
# Sweep training of the four diagnostic probes across one VFM.
#   bash scripts/run_diag_sweep.sh wan
#   bash scripts/run_diag_sweep.sh cogvideox
#   bash scripts/run_diag_sweep.sh vjepa2
set -euo pipefail
VFM=${1:-wan}
DEV=${2:-0}
EXTRA=${3:-}

for probe in view_consistency ego_belief action_dynamics abnormal; do
    cfg="inscene15k_ext/${probe}_${VFM}_v1"
    echo "==== ${cfg} ===="
    CUDA_VISIBLE_DEVICES=${DEV} python vidfm3d/train.py \
        experiment=${cfg} \
        ${EXTRA}
done
