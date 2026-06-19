#!/usr/bin/env bash
# Example:
#   export PROJECT_ROOT=/shared/probe_spatial PYTHON=/env/bin/python
#   export DATA_ROOT=/shared/InsScene-15K/data OUT_ROOT=/shared/InsScene-15K/FEAT
#   export TOTAL_SCENES=9530 VFM=wan MODE=normal LAYERS="20"
#   sbatch --array=0-99%100 slurm/extract_feature_array.sbatch
#SBATCH --job-name=probe-feat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --signal=B:USR1@120

set -euo pipefail
: "${PROJECT_ROOT:?}" "${PYTHON:?}" "${DATA_ROOT:?}" "${OUT_ROOT:?}" "${TOTAL_SCENES:?}"
VFM=${VFM:-wan}
MODE=${MODE:-normal}
LAYERS=${LAYERS:-20}
SHUFFLE_SEED=${SHUFFLE_SEED:-42}
NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT:-100}
SHARD=${SLURM_ARRAY_TASK_ID:?}
CHUNK=$(((TOTAL_SCENES + NUM_SHARDS - 1) / NUM_SHARDS))
START=$((SHARD * CHUNK))
END=$((START + CHUNK))
(( END > TOTAL_SCENES )) && END=${TOTAL_SCENES}
(( START >= END )) && exit 0

export PROJECT_ROOT PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${USER}"
mkdir -p "${MPLCONFIGDIR}" "${PROJECT_ROOT}/logs/slurm"
cd "${PROJECT_ROOT}"
read -r -a layer_args <<< "${LAYERS}"
srun "${PYTHON}" -m features.run_inscene15k \
  --vfm "${VFM}" --mode "${MODE}" \
  --data-root "${DATA_ROOT}" --out-root "${OUT_ROOT}" \
  --output-layers "${layer_args[@]}" --shuffle-seed "${SHUFFLE_SEED}" \
  --start "${START}" --end "${END}"
