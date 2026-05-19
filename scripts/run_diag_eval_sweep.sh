#!/usr/bin/env bash
# Run eval_diag.py on every (probe, vfm) run that has a last.ckpt under logs/
#   bash scripts/run_diag_eval_sweep.sh
set -euo pipefail

LOGS_ROOT=${LOGS_ROOT:-logs}
SPLIT=${SPLIT:-val}

for vfm in wan cogvideox vjepa2; do
    for probe in view_consistency ego_belief action_dynamics abnormal; do
        run="inscene15k_ext_${probe}_${vfm}_v1"
        # find the latest ckpt under any matching run dir
        ckpt=$(ls -1t ${LOGS_ROOT}/**/${run}/checkpoints/last.ckpt 2>/dev/null | head -n1 || true)
        if [[ -z "${ckpt}" ]]; then
            ckpt=$(ls -1t ${LOGS_ROOT}/${run}/checkpoints/last.ckpt 2>/dev/null | head -n1 || true)
        fi
        if [[ -z "${ckpt}" ]]; then
            echo "[skip] no ckpt for ${run}"
            continue
        fi
        echo "==== eval ${run} ===="
        python vidfm3d/eval_diag.py \
            experiment=inscene15k_ext/${probe}_${vfm}_v1 \
            ckpt_path=${ckpt} \
            eval_split=${SPLIT} \
            train=false test=false
    done
done

# Aggregate
python vidfm3d/eval_diag_compare.py \
    --runs $(ls -d ${LOGS_ROOT}/*inscene15k_ext_*_v1 2>/dev/null) \
    --split ${SPLIT} \
    --output comparison_${SPLIT}.csv
