#!/usr/bin/env bash
# Submit the strictly serial remaining timestep extraction campaign to one 8-GPU SCO node.
set -euo pipefail

export PATH="${HOME}/.sco/bin:${PATH}"
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}"
RUN_ID="${RUN_ID:-tstep_campaign_r7_$(date -u +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/logs/feat_extract_${RUN_ID}}"
CAMPAIGN_LOG_ROOT="${CAMPAIGN_LOG_ROOT:-${OUT_DIR}/campaign}"
JOB_NAME="${JOB_NAME:-bq_tstep_r7_mmh3target_cgx}"
SYNC_EVERY_SEC="${SYNC_EVERY_SEC:-180}"
SYNC_EVERY_GIB="${SYNC_EVERY_GIB:-80}"
DRY_SUBMIT="${DRY_SUBMIT:-0}"

cd "${PROJECT_ROOT}"
for guarded in \
  features/run_inscene15k.py \
  features/cogvideox/cogvideox_feature_i2v.py \
  features/minimax_h3/minimax_h3_feature.py \
  scripts/run_streaming_feat_extract_8gpu.sh \
  scripts/run_tstep_sync_loop.sh \
  scripts/run_tstep_extract_campaign_r7.sh; do
  size=0
  [[ -f "${guarded}" ]] && size="$(wc -c < "${guarded}")"
  if [[ "${size}" -lt 3000 ]]; then
    echo "ERROR: refusing submit; source looks truncated: ${guarded} (${size} bytes)" >&2
    exit 1
  fi
done

for index in \
  logs/tstep_resume4/aoss_done/FEAT_TARGET_minimax_h3_t500.txt \
  logs/tstep_resume4/aoss_done/FEAT_TARGET_minimax_h3_t1000.txt \
  logs/tstep_resume4/aoss_done/FEAT_STREAMING_cogvideox_t499.txt \
  logs/tstep_resume4/aoss_done/FEAT_STREAMING_cogvideox_t999.txt \
  logs/tstep_resume4/aoss_done/FEAT_TARGET_cogvideox_t499.txt \
  logs/tstep_resume4/aoss_done/FEAT_TARGET_cogvideox_t999.txt; do
  [[ -f "${index}" ]] || { echo "ERROR: missing done-index: ${index}" >&2; exit 1; }
done

mkdir -p "${OUT_DIR}/launcher" "${CAMPAIGN_LOG_ROOT}"
chmod +x scripts/run_tstep_extract_campaign_r7.sh scripts/run_streaming_feat_extract_8gpu.sh scripts/run_tstep_sync_loop.sh

command="cd ${PROJECT_ROOT} && CAMPAIGN_LOG_ROOT=${CAMPAIGN_LOG_ROOT} SYNC_EVERY_SEC=${SYNC_EVERY_SEC} SYNC_EVERY_GIB=${SYNC_EVERY_GIB} bash scripts/run_tstep_extract_campaign_r7.sh"
cat > "${OUT_DIR}/PLAN.txt" <<EOF
RUN_ID=${RUN_ID}
job_name=${JOB_NAME}
order=minimax_target_t500,minimax_target_t1000,cogvideox_stream_t499,cogvideox_stream_t999,cogvideox_target_t499,cogvideox_target_t999
campaign_log_root=${CAMPAIGN_LOG_ROOT}
sync_every_sec=${SYNC_EVERY_SEC}
sync_every_gib=${SYNC_EVERY_GIB}
EOF

if [[ "${DRY_SUBMIT}" == "1" ]]; then
  echo "DRY_SUBMIT=1"
  cat "${OUT_DIR}/PLAN.txt"
  echo "command=${command}"
  exit 0
fi

submit_log="${OUT_DIR}/launcher/sco_submit.log"
sco acp jobs create \
  --workspace-name=aigc \
  --aec2-name=h800 \
  --job-name="${JOB_NAME}" \
  --priority=HIGHEST \
  --container-image-url=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617 \
  --storage-mount=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm \
  --training-framework=pytorch \
  --worker-nodes=1 \
  --worker-spec=N4lS.Iq.I80.8 \
  --command="${command}" \
  | tee "${submit_log}"
job_id="$(grep -oE 'pt-[a-z0-9]+' "${submit_log}" | head -n 1)"
[[ -n "${job_id}" ]] || { echo "ERROR: SCO did not return a job id" >&2; exit 1; }
printf '%s\n' "${job_id}" > "${OUT_DIR}/launcher/sco_job_id.txt"
echo "Submitted ${JOB_NAME}: ${job_id}"
echo "Campaign logs: ${CAMPAIGN_LOG_ROOT}"
