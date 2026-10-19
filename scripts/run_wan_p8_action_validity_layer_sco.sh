#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
RUN_TAG=wan_p8_action_dynamics_canonical_manifest_exact_20260821_200604_2075f3e
LAYER=${LAYER:?LAYER is required}
PY=${PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}
MANIFEST=${PROJECT_ROOT}/manifests/c_retrain_20260722/inscene_c_h124_prefix8.jsonl
FEATURE_ROOT=/mnt/umm/users/yinbaiqiao/InsScene-15K/FEATURES_VERSIONED/${RUN_TAG}
OUTPUT=${PROJECT_ROOT}/results/analysis/wan_p8_action_validity
MAPPINGS=${OUTPUT}/permutations/index.json

case " ${LAYER} " in
  " 0 "|" 5 "|" 10 "|" 15 "|" 25 "|" 29 ") ;;
  *) echo "invalid layer ${LAYER}" >&2; exit 2 ;;
esac

declare -A WANDB_ID CKPT_SHA
WANDB_ID[0]=jspcgevf
WANDB_ID[5]=kttf6gu0
WANDB_ID[10]=2oefz7nm
WANDB_ID[15]=npn9jxyk
WANDB_ID[25]=jo3b0g4f
WANDB_ID[29]=nmawq03h
CKPT_SHA[0]=014ee02384b1b54cbd5ebee3d8b0ff133e3eb8ff949459f5e7bda70af705c13b
CKPT_SHA[5]=9bc4a459008871199945cc287a65bb62c8906ee66aa4d10edd288380bdb4a093
CKPT_SHA[10]=d4fd33f068a396d5a024ac212f4f452a8cc43a1f12a9aeab8d29290243ef2343
CKPT_SHA[15]=bcfc7ccf508cba6b99dbe41e01a134a146d3ec557c147d639ad9a3f885043ed2
CKPT_SHA[25]=c4bd2643fbf6a0f175e411849364967b4c2ed5dfef6a5b2ef3818a3a6c7ad1c3
CKPT_SHA[29]=3712d78b5058008849a37cee01aa832947bdd2a7a9dab38f85fd05ac3336b35b

NAME=${RUN_TAG}_canonical_clean_action_dynamics_wan_p8_layer${LAYER}
CKPT=${PROJECT_ROOT}/logs/inscene15k_streaming/runs/${RUN_TAG}/${NAME}/checkpoints/last.ckpt
EXISTING=${PROJECT_ROOT}/logs/inscene15k_streaming/runs/${RUN_TAG}_eval_staging/${NAME}/eval/val_predictions.pt
STATUS=${OUTPUT}/layers/layer${LAYER}.sco_status.json
LOG=${OUTPUT}/layers/layer${LAYER}.sco.log

[[ ! -e "${OUTPUT}/layers/layer${LAYER}" ]] || { echo "refusing existing layer output" >&2; exit 2; }
[[ ! -e "${STATUS}" ]] || { echo "refusing existing status" >&2; exit 2; }
mkdir -p "${OUTPUT}/layers" /tmp/matplotlib-wan-action
exec > >(tee -a "${LOG}") 2>&1
started=$(date -u +%FT%TZ)
finish() {
  rc=$?
  trap - EXIT
  ended=$(date -u +%FT%TZ)
  printf '{"layer":%d,"started":"%s","ended":"%s","rc":%d}\n' "${LAYER}" "${started}" "${ended}" "${rc}" > "${STATUS}"
  exit "${rc}"
}
trap finish EXIT

export PYTHONPATH=${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}
export INSCENE_DATA_ROOT=/mnt/umm/users/yinbaiqiao/InsScene-15K/data
export INSCENE_MANIFEST=${MANIFEST}
export INSCENE_STREAMING_FEAT_ROOT=${FEATURE_ROOT}/streaming
export INSCENE_TARGET_FEAT_ROOT=${FEATURE_ROOT}/target
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLCONFIGDIR=/tmp/matplotlib-wan-action PYTHONUNBUFFERED=1
cd "${PROJECT_ROOT}"

"${PY}" scripts/run_wan_p8_action_validity_layer.py \
  experiment=inscene15k_streaming/action_dynamics_wan_v1 \
  seed=42 train=false test=false autoresume=false \
  vfm_name=wan video_channels=1536 \
  streaming_feat_root="${FEATURE_ROOT}/streaming" \
  target_feat_root="${FEATURE_ROOT}/target" \
  feature_layer="${LAYER}" feature_timestep=749 prefix_len=8 \
  data.data_module.batch_size_per_device_val=4 \
  data.data_module.num_workers_val=2 \
  ckpt_path="${CKPT}" \
  +analysis_output_dir="${OUTPUT}" \
  +analysis_mapping_index="${MAPPINGS}" \
  +analysis_existing_predictions="${EXISTING}" \
  +analysis_manifest="${MANIFEST}" \
  +analysis_checkpoint_sha256="${CKPT_SHA[${LAYER}]}" \
  +analysis_batch_size=64

echo "PHASE1 LAYER PASS layer=${LAYER}"
