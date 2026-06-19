#!/usr/bin/env bash
# Example: sbatch --array=0-38%100 slurm/train_diag_array.sbatch
# 13 configs x 3 seeds = 39 independent one-GPU jobs.
#SBATCH --job-name=probe-diag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --signal=B:USR1@120

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:?export PROJECT_ROOT before sbatch}
PYTHON=${PYTHON:?export PYTHON before sbatch}
INSCENE_SPLIT_MANIFEST=${INSCENE_SPLIT_MANIFEST:-${PROJECT_ROOT}/configs/splits/inscene15k_v1.json}
INSCENE_DATA_ROOT=${INSCENE_DATA_ROOT:?export INSCENE_DATA_ROOT before sbatch}
INSCENE_FEAT_ROOT=${INSCENE_FEAT_ROOT:?export INSCENE_FEAT_ROOT before sbatch}
INSCENE_TARGET_FEAT_ROOT=${INSCENE_TARGET_FEAT_ROOT:?export INSCENE_TARGET_FEAT_ROOT before sbatch}
INSCENE_SHUFFLED_FEAT_ROOT=${INSCENE_SHUFFLED_FEAT_ROOT:?export INSCENE_SHUFFLED_FEAT_ROOT before sbatch}
export PROJECT_ROOT PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export INSCENE_SPLIT_MANIFEST INSCENE_DATA_ROOT INSCENE_FEAT_ROOT
export INSCENE_TARGET_FEAT_ROOT INSCENE_SHUFFLED_FEAT_ROOT
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${USER}"
mkdir -p "${MPLCONFIGDIR}" "${PROJECT_ROOT}/logs/slurm"
cd "${PROJECT_ROOT}"

configs=(
  view_consistency_wan_v1 view_consistency_cogvideox_v1 view_consistency_vjepa2_v1
  abnormal_wan_v1 abnormal_cogvideox_v1 abnormal_vjepa2_v1
  ego_belief_wan_v1 ego_belief_cogvideox_v1 ego_belief_vjepa2_v1
  ego_belief_v2_wan_v1
  action_dynamics_wan_v1 action_dynamics_cogvideox_v1 action_dynamics_vjepa2_v1
)
seeds=(41 42 43)
task=${SLURM_ARRAY_TASK_ID:?submit as an array}
config=${configs[$((task / ${#seeds[@]}))]}
seed=${seeds[$((task % ${#seeds[@]}))]}
job="${config}_seed${seed}"

srun "${PYTHON}" vidfm3d/train.py \
  "experiment=inscene15k_ext/${config}" \
  "seed=${seed}" \
  "job_name=${job}" \
  "paths.run_folder_name=inscene15k_ext_${job}" \
  "logger.wandb.name=${job}" \
  "trainer.devices=1" \
  "trainer.strategy=auto"
