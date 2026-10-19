#!/usr/bin/env bash
# Submit one quota-safe resume4 feature-extraction job.
set -euo pipefail

export PATH="${HOME}/.sco/bin:${PATH}"
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}"
cd "${PROJECT_ROOT}"

for guarded in \
  features/run_inscene15k.py \
  scripts/run_streaming_feat_extract_8gpu.sh \
  scripts/run_tstep_sync_loop.sh \
  scripts/delete_aoss_confirmed_local_pairs.sh; do
  size=0
  [[ -f "${guarded}" ]] && size="$(wc -c < "${guarded}")"
  if [[ "${size}" -lt 3000 ]]; then
    echo "ERROR: refusing submit; source looks truncated: ${guarded} (${size} bytes)" >&2
    exit 1
  fi
done

VFM="${VFM:-wan}"
T="${T:-499}"
EXTRACT_STREAM="${EXTRACT_STREAM:-0}"
EXTRACT_TARGET="${EXTRACT_TARGET:-1}"
LAYERS="${LAYERS:-15,20,25}"
NPROC="${NPROC:-8}"
AEC2_NAME="${AEC2_NAME:-h800}"
DRY_SUBMIT="${DRY_SUBMIT:-0}"
SYNC_EVERY_SEC="${SYNC_EVERY_SEC:-180}"
SYNC_EVERY_GIB="${SYNC_EVERY_GIB:-80}"
PRE_EXTRACT_CLEANUP_MANIFEST="${PRE_EXTRACT_CLEANUP_MANIFEST:-}"
CLEANUP_EXPECTED_COUNT="${CLEANUP_EXPECTED_COUNT:-0}"

if [[ $((EXTRACT_STREAM + EXTRACT_TARGET)) -ne 1 ]]; then
  echo "ERROR: submit exactly one mode at a time" >&2
  exit 2
fi
case "${VFM}" in
  wan) tag="wan" ;;
  cogvideox) tag="cgx" ;;
  minimax_h3) tag="mmh3" ;;
  *) echo "ERROR: unsupported VFM=${VFM}" >&2; exit 2 ;;
esac
if [[ "${EXTRACT_STREAM}" == "1" ]]; then
  mode_tag="stream"
  feature_kind="FEAT_STREAMING"
else
  mode_tag="target"
  feature_kind="FEAT_TARGET"
fi

case "${AEC2_NAME}" in
  zoetrope|h800) worker_spec="N4lS.Iq.I80." ;;
  vigen|si) worker_spec="N6lS.Iq.I10." ;;
  *) echo "ERROR: invalid AEC2_NAME=${AEC2_NAME}" >&2; exit 2 ;;
esac

RUN_ID="${RUN_ID:-tstep_resume4_$(date -u +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/logs/feat_extract_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/${VFM}_t${T}_${mode_tag}}"
EXTRACT_DONE_INDEX="${EXTRACT_DONE_INDEX:-${PROJECT_ROOT}/logs/tstep_resume4/aoss_done/${feature_kind}_${VFM}_t${T}.txt}"
JOB_NAME="${JOB_NAME:-bq_tstep_r4_${tag}_t${T}_${mode_tag}}"

[[ -f "${EXTRACT_DONE_INDEX}" ]] || {
  echo "ERROR: missing AOSS done index: ${EXTRACT_DONE_INDEX}" >&2
  exit 1
}
if [[ -n "${PRE_EXTRACT_CLEANUP_MANIFEST}" ]]; then
  [[ -f "${PRE_EXTRACT_CLEANUP_MANIFEST}" ]] || {
    echo "ERROR: missing cleanup manifest: ${PRE_EXTRACT_CLEANUP_MANIFEST}" >&2
    exit 1
  }
fi
[[ "${NPROC}" -eq 8 ]] || {
  echo "ERROR: resume4 extraction requires NPROC=8" >&2
  exit 2
}

container_image_url="registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617"
storage_mount="019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm"
mkdir -p "${OUT_DIR}/launcher" "${LOG_DIR}"
chmod +x \
  "${PROJECT_ROOT}/scripts/run_streaming_feat_extract_8gpu.sh" \
  "${PROJECT_ROOT}/scripts/run_tstep_sync_loop.sh"

cmd="cd ${PROJECT_ROOT} && VFM=${VFM} T=${T} LAYERS=${LAYERS} EXTRACT_STREAM=${EXTRACT_STREAM} EXTRACT_TARGET=${EXTRACT_TARGET} EXTRACT_DONE_INDEX=${EXTRACT_DONE_INDEX} SYNC_EVERY_SEC=${SYNC_EVERY_SEC} SYNC_EVERY_GIB=${SYNC_EVERY_GIB} PRE_EXTRACT_CLEANUP_MANIFEST=${PRE_EXTRACT_CLEANUP_MANIFEST} CLEANUP_EXPECTED_COUNT=${CLEANUP_EXPECTED_COUNT} LOG_DIR=${LOG_DIR} bash scripts/run_streaming_feat_extract_8gpu.sh"

cat > "${OUT_DIR}/PLAN.txt" <<EOF
RUN_ID=${RUN_ID}
job_name=${JOB_NAME}
vfm=${VFM}
t=${T}
mode=${mode_tag}
layers=${LAYERS}
done_index=${EXTRACT_DONE_INDEX}
done_index_lines=$(wc -l < "${EXTRACT_DONE_INDEX}")
sync_every_sec=${SYNC_EVERY_SEC}
sync_every_gib=${SYNC_EVERY_GIB}
pre_extract_cleanup_manifest=${PRE_EXTRACT_CLEANUP_MANIFEST}
cleanup_expected_count=${CLEANUP_EXPECTED_COUNT}
log_dir=${LOG_DIR}
EOF
printf 'job_name\tvfm\tt\tmode\tlayers\tdone_index\tlog_dir\n' > "${OUT_DIR}/launcher/job_map.tsv"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "${JOB_NAME}" "${VFM}" "${T}" "${mode_tag}" "${LAYERS}" \
  "${EXTRACT_DONE_INDEX}" "${LOG_DIR}" >> "${OUT_DIR}/launcher/job_map.tsv"

if [[ "${DRY_SUBMIT}" == "1" ]]; then
  echo "DRY_SUBMIT=1"
  cat "${OUT_DIR}/PLAN.txt"
  echo "command=${cmd}"
  exit 0
fi

submit_log="${OUT_DIR}/launcher/sco_submit.log"
sco acp jobs create \
  --workspace-name=aigc \
  --aec2-name="${AEC2_NAME}" \
  --job-name="${JOB_NAME}" \
  --priority=HIGHEST \
  --container-image-url="${container_image_url}" \
  --storage-mount="${storage_mount}" \
  --training-framework=pytorch \
  --worker-nodes=1 \
  --worker-spec="${worker_spec}${NPROC}" \
  --command="${cmd}" \
  | tee "${submit_log}"

job_id="$(grep -oE 'pt-[a-z0-9]+' "${submit_log}" | head -n 1)"
[[ -n "${job_id}" ]] || { echo "ERROR: SCO did not return a job id" >&2; exit 1; }
printf '%s\n' "${job_id}" > "${OUT_DIR}/launcher/sco_job_id.txt"
echo "Submitted ${JOB_NAME}: ${job_id}"
echo "Logs: ${LOG_DIR}"
