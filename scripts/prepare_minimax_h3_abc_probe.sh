#!/usr/bin/env bash
# Prepare MiniMax-H3 multilayer ABC probing jobs for review.
#
# Probe matrix (ABC):
#   A1 geometry = streaming_depth  (depth MAE + pose via camera head; one job)
#   A1 mask     = streaming_mask   (identity loss; optional, not in default PROBES)
#   A2=view_consistency  B2=ego_belief_v2
#   C1=action_dynamics  C2=path_integration
#   C3 is evaluated from the frozen C1 head; it is not a train job.
# Pose is the camera head inside streaming_depth. Do not add streaming_camera.
# Layers: 10 20 30 40 50(=disk 49)
# Prefixes: 8 12 16 24
# Total default jobs: 1 VFM x 5 probes x 4 prefixes x 5 layers = 100
#
# Features are expected on AOSS raw trees:
#   aoss://baiqiao/InsScene-15K/FEAT_STREAMING/minimax_h3
#   aoss://baiqiao/InsScene-15K/FEAT_TARGET/minimax_h3
#
# Usage:
#   bash scripts/prepare_minimax_h3_abc_probe.sh
#   FEAT_BACKEND=aoss bash scripts/prepare_minimax_h3_abc_probe.sh
#   PROBES="action_dynamics path_integration" \
#     bash scripts/prepare_minimax_h3_abc_probe.sh
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
DATASET_ROOT=${DATASET_ROOT:-/mnt/umm/users/yinbaiqiao/InsScene-15K}
RUN_ID=${RUN_ID:-minimax_h3_abc_$(date -u +%Y%m%d_%H%M)}
OUT_DIR=${OUT_DIR:-${PROJECT_ROOT}/logs/probing_jobs_${RUN_ID}}

VFMS=${VFMS:-"minimax_h3"}
PROBES=${PROBES:-"streaming_depth view_consistency ego_belief_v2 action_dynamics path_integration"}
if [[ " ${PROBES} " == *" counterfactual "* ]]; then
  echo "[err] C3/counterfactual is evaluation-only; train C1/action_dynamics and run c3_action_sensitivity evaluation" >&2
  exit 2
fi
LAYERS=${LAYERS:-"10 20 30 40 50"}
PREFIXES=${PREFIXES:-"8 12 16 24"}
BATCH_SIZE=${BATCH_SIZE:-2}
BATCH_SIZE_VAL=${BATCH_SIZE_VAL:-2}
NUM_WORKERS=${NUM_WORKERS:-4}
NUM_WORKERS_VAL=${NUM_WORKERS_VAL:-2}
VIDEO_CHANNELS=${VIDEO_CHANNELS:-5376}
FEATURE_TIMESTEP=${FEATURE_TIMESTEP:-250}

# Feat backend: local raw trees or AOSS raw trees (aoss:// scheme used by aoss_fs).
FEAT_BACKEND=${FEAT_BACKEND:-aoss}
AOSS_BUCKET_ROOT=${AOSS_BUCKET_ROOT:-aoss://baiqiao/InsScene-15K}
LOCAL_STREAM_ROOT=${LOCAL_STREAM_ROOT:-${DATASET_ROOT}/FEAT_STREAMING}
LOCAL_TARGET_ROOT=${LOCAL_TARGET_ROOT:-${DATASET_ROOT}/FEAT_TARGET}
MANIFEST_DIR=${MANIFEST_DIR:-${PROJECT_ROOT}/logs/probing_jobs_20260724_1808/manifests_filtered}

case "${FEAT_BACKEND}" in
  aoss)
    DATA_ROOT=${DATA_ROOT:-${AOSS_BUCKET_ROOT}/data}
    STREAM_ROOT=${STREAM_ROOT:-${AOSS_BUCKET_ROOT}/FEAT_STREAMING}
    TARGET_ROOT=${TARGET_ROOT:-${AOSS_BUCKET_ROOT}/FEAT_TARGET}
    ;;
  local)
    DATA_ROOT=${DATA_ROOT:-${DATASET_ROOT}/data}
    STREAM_ROOT=${STREAM_ROOT:-${LOCAL_STREAM_ROOT}}
    TARGET_ROOT=${TARGET_ROOT:-${LOCAL_TARGET_ROOT}}
    ;;
  *)
    echo "[err] FEAT_BACKEND must be aoss|local (got ${FEAT_BACKEND})" >&2
    exit 1
    ;;
esac

mkdir -p "${OUT_DIR}"/{jobs,dryrun,configs_snapshot}

manifest_for_prefix() {
  local prefix="$1"
  local m="${MANIFEST_DIR}/inscene_c_h124_prefix${prefix}_no48da_w003.jsonl"
  if [[ ! -f "${m}" ]]; then
    m="${PROJECT_ROOT}/manifests/c_retrain_20260722/inscene_c_h124_prefix${prefix}.jsonl"
  fi
  printf '%s' "${m}"
}

needs_target() {
  case "$1" in
    action_dynamics|path_integration|counterfactual) return 0 ;;
    *) return 1 ;;
  esac
}

# CLI layer 50 was remapped to disk layer 49 during extraction.
disk_layer_for() {
  local layer="$1"
  if [[ "${layer}" == "50" ]]; then
    printf '49'
  else
    printf '%s' "${layer}"
  fi
}

echo "RUN_ID=${RUN_ID}"
echo "OUT_DIR=${OUT_DIR}"
echo "FEAT_BACKEND=${FEAT_BACKEND}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "STREAM_ROOT=${STREAM_ROOT}"
echo "TARGET_ROOT=${TARGET_ROOT}"
echo "VFMS=${VFMS}"
echo "PROBES=${PROBES}"
echo "LAYERS=${LAYERS}"
echo "PREFIXES=${PREFIXES}"
echo "BATCH_SIZE=${BATCH_SIZE}"

# ---- preflight ----
preflight_log="${OUT_DIR}/PREFLIGHT.txt"
{
  echo "generated_at_utc=$(date -u -Is)"
  echo "feat_backend=${FEAT_BACKEND}"
  echo "stream_root=${STREAM_ROOT}"
  echo "target_root=${TARGET_ROOT}"
  echo
  echo "## local FEAT trees"
  for vfm in ${VFMS}; do
    st="${LOCAL_STREAM_ROOT}/${vfm}/scannetpp"
    tg="${LOCAL_TARGET_ROOT}/${vfm}/scannetpp"
    st_n=0; tg_n=0
    [[ -d "${st}" ]] && st_n=$(find "${st}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    [[ -d "${tg}" ]] && tg_n=$(find "${tg}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    echo "local_streaming ${vfm}: dir=${st} scenes=${st_n}"
    echo "local_target    ${vfm}: dir=${tg} scenes=${tg_n}"
  done
  echo
  echo "## required code/config files (local working tree)"
  required_files=(
    "vidfm3d/train.py"
    "vidfm3d/models/video_probe_module.py"
    "vidfm3d/models/probe_ext_module.py"
    "vidfm3d/models/components/dpt_head.py"
    "vidfm3d/utils/aoss_fs.py"
    "vidfm3d/utils/feature_layers.py"
    "configs/callbacks/v3_checkpoint.yaml"
    "configs/train.yaml"
  )
  missing_required=0
  for rel in "${required_files[@]}"; do
    if [[ -f "${PROJECT_ROOT}/${rel}" ]]; then
      echo "OK   ${rel}"
    else
      echo "MISS ${rel}"
      missing_required=1
    fi
  done
  if (( missing_required == 1 )); then
    echo "BLOCKER: required training files missing in working tree."
    echo "         Restore with: git checkout HEAD -- <paths>"
  fi
  echo
  echo "## config presence"
  for vfm in ${VFMS}; do
    for probe in ${PROBES}; do
      cfg="${PROJECT_ROOT}/configs/experiment/inscene15k_streaming/${probe}_${vfm}_v1.yaml"
      if [[ -f "${cfg}" ]]; then
        echo "OK   ${cfg}"
      else
        echo "MISS ${cfg}"
      fi
    done
  done
  echo
  echo "## manifests"
  for prefix in ${PREFIXES}; do
    m="$(manifest_for_prefix "${prefix}")"
    if [[ -f "${m}" ]]; then
      echo "OK   prefix=${prefix} ${m}"
    else
      echo "MISS prefix=${prefix} ${m}"
    fi
  done
  echo
  echo "## notes"
  echo "MiniMax-H3 features are expected on AOSS (FEAT_STREAMING + FEAT_TARGET)."
  echo "Local empty caches are OK when FEAT_BACKEND=aoss; launch defaults"
  echo "ALLOW_MISSING_STREAMING=1 for this VFM."
  echo "Layer CLI 50 maps to disk feature_t250_layer49.sft."
} | tee "${preflight_log}"

# ---- job list ----
# TSV columns: vfm probe prefix layer disk_layer manifest cfg
jobs_tsv="${OUT_DIR}/jobs/all.tsv"
: > "${jobs_tsv}"
n_jobs=0
for vfm in ${VFMS}; do
  for probe in ${PROBES}; do
    cfg="configs/experiment/inscene15k_streaming/${probe}_${vfm}_v1.yaml"
    if [[ ! -f "${PROJECT_ROOT}/${cfg}" ]]; then
      echo "[warn] skip missing config: ${cfg}" >&2
      continue
    fi
    for prefix in ${PREFIXES}; do
      manifest="$(manifest_for_prefix "${prefix}")"
      for layer in ${LAYERS}; do
        disk_layer="$(disk_layer_for "${layer}")"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
          "${vfm}" "${probe}" "${prefix}" "${layer}" "${disk_layer}" "${manifest}" "${cfg}" \
          >> "${jobs_tsv}"
        n_jobs=$((n_jobs + 1))
      done
    done
  done
done

# ---- dry-run command examples (first 3 jobs) ----
dry="${OUT_DIR}/dryrun/example_train_commands.sh"
{
  echo "#!/usr/bin/env bash"
  echo "# Example train commands generated for review."
  echo "set -euo pipefail"
  echo "cd ${PROJECT_ROOT}"
  echo "source /mnt/umm/users/yinbaiqiao/probe_spatial_env.sh"
  echo
  echo "# Bypass HTTP proxy for internal AOSS (required on this cluster)."
  echo "export NO_PROXY='localhost,127.0.0.1,::1,baiqiao.aoss-internal.cn-fz-01.fjscmsapi-oss.com,.aoss-internal.cn-fz-01.fjscmsapi-oss.com,aoss-internal.cn-fz-01.fjscmsapi-oss.com'"
  echo "export no_proxy=\"\${NO_PROXY}\""
  echo "export INSCENE_DATA_ROOT='${DATA_ROOT}'"
  echo "unset INSCENE_STREAMING_FEAT_ROOT   # let Hydra override win"
  echo "unset INSCENE_TARGET_FEAT_ROOT"
  echo
  n=0
  while IFS=$'\t' read -r vfm probe prefix layer disk_layer manifest cfg; do
    n=$((n + 1))
    (( n > 3 )) && break
    tag="${probe}_${vfm}_p${prefix}_layer${layer}"
    bs="${BATCH_SIZE}"
    if [[ "${probe}" == "streaming_depth" || "${probe}" == "streaming_mask" ]]; then
      bs=1
    fi
    echo "# --- example ${n}: ${tag} (disk_layer=${disk_layer}) ---"
    echo "export INSCENE_MANIFEST='${manifest}'"
    echo "export CUDA_VISIBLE_DEVICES=0"
    echo "python vidfm3d/train.py \\"
    echo "  experiment=inscene15k_streaming/${probe}_${vfm}_v1 \\"
    echo "  vfm_name=${vfm} \\"
    echo "  video_channels=${VIDEO_CHANNELS} \\"
    echo "  streaming_feat_root=${STREAM_ROOT} \\"
    echo "  feature_layer=${disk_layer} \\"
    echo "  feat_postfix=_t${FEATURE_TIMESTEP}_layer${disk_layer} \\"
    echo "  feature_timestep=${FEATURE_TIMESTEP} \\"
    echo "  prefix_len=${prefix} \\"
    echo "  job_name=${tag} \\"
    echo "  paths.run_folder_name=${RUN_ID}/${RUN_ID}_${tag} \\"
    echo "  logger.wandb.name=${RUN_ID}_${tag} \\"
    echo "  logger.wandb.group=${RUN_ID} \\"
    echo "  logger.wandb.offline=true \\"
    echo "  autoresume=false \\"
    echo "  data.data_module.batch_size_per_device=${bs} \\"
    echo "  data.data_module.batch_size_per_device_val=${bs} \\"
    echo "  data.data_module.num_workers=${NUM_WORKERS} \\"
    echo "  data.data_module.num_workers_val=${NUM_WORKERS_VAL} \\"
    if needs_target "${probe}"; then
      echo "  target_feat_root=${TARGET_ROOT} \\"
    fi
    if [[ "${probe}" == "ego_belief_v2" || "${probe}" == "ego_belief" ]]; then
      echo "  streaming_hidden_prefix_lengths=[${prefix}] \\"
    fi
    echo "  #"
    echo
  done < "${jobs_tsv}"
} > "${dry}"
chmod +x "${dry}"

# snapshot of key knobs
cat > "${OUT_DIR}/PLAN.txt" <<EOF
MiniMax-H3 multilayer ABC probing plan
======================================

Run id: ${RUN_ID}
Jobs:   ${n_jobs}
Matrix: VFMS={${VFMS}} PROBES={${PROBES}} LAYERS={${LAYERS}} PREFIXES={${PREFIXES}}
Batch:  ${BATCH_SIZE} (A1 streaming_depth/streaming_mask forced to 1) / workers ${NUM_WORKERS}
Feat:   backend=${FEAT_BACKEND}
        streaming=${STREAM_ROOT}
        target=${TARGET_ROOT}
        data=${DATA_ROOT}
        channels=${VIDEO_CHANNELS} timestep=${FEATURE_TIMESTEP}

Layer note:
  job label layer=50 uses disk feature_t250_layer49.sft

Files:
  ${jobs_tsv}
  ${preflight_log}
  ${dry}
  scripts/launch_minimax_h3_abc_probe.sh

Launch:
  RUN_ID=${RUN_ID} OUT_DIR=${OUT_DIR} \\
    ALLOW_MISSING_STREAMING=1 GPU_IDS_CSV=0,1,2,3,4,5,6,7 SLOTS=16 \\
    bash scripts/launch_minimax_h3_abc_probe.sh
EOF

cp "${PROJECT_ROOT}/scripts/run_streaming_probe_sweep.sh" "${OUT_DIR}/configs_snapshot/" 2>/dev/null || true
echo
echo "Prepared ${n_jobs} jobs."
echo "Review:"
echo "  ${OUT_DIR}/PLAN.txt"
echo "  ${OUT_DIR}/PREFLIGHT.txt"
echo "  ${OUT_DIR}/jobs/all.tsv"
echo "  ${OUT_DIR}/dryrun/example_train_commands.sh"
