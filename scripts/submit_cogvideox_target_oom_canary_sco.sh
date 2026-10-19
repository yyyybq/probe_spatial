#!/usr/bin/env bash
# One-scene H800 canary for the CogVideoX target CPU-materialization fix.
set -euo pipefail

export PATH="${HOME}/.sco/bin:${PATH}"
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/umm/users/yinbaiqiao/InsScene-15K}"
FEATURE_NAMESPACE=FEAT_TARGET_CANONICAL_V2_SEEDED_V1
VFM=cogvideox
T=499
SCENE_START=357
SCENE_END=358
EXPECTED_SCENE=64672b5bf5
DONE_INDEX="${PROJECT_ROOT}/logs/tstep_canonical_v2_seeded_v1/aoss_done/${FEATURE_NAMESPACE}_${VFM}_t${T}.txt"
OUT_ROOT="${DATASET_ROOT}/${FEATURE_NAMESPACE}"
SCENE_ROOT="${OUT_ROOT}/${VFM}/scannetpp/${EXPECTED_SCENE}"
RUN_ID="${RUN_ID:-seeded_v1_cgx_target_oom_canary_$(date -u +%Y%m%d_%H%M%S)}"
LOG_DIR="${PROJECT_ROOT}/logs/${RUN_ID}"
DRY_SUBMIT="${DRY_SUBMIT:-0}"

cd "${PROJECT_ROOT}"
for path in features/run_inscene15k.py scripts/run_streaming_feat_extract_8gpu.sh scripts/run_tstep_sync_loop.sh; do
  [[ -s "${path}" && ! -L "${path}" ]] || { echo "ERROR: invalid source ${path}" >&2; exit 1; }
done
[[ -f "${DONE_INDEX}" && ! -L "${DONE_INDEX}" ]] || { echo "ERROR: invalid done-index" >&2; exit 1; }
if rg -q "^${EXPECTED_SCENE}/" "${DONE_INDEX}"; then
  echo "ERROR: canary scene already appears in production done-index: ${EXPECTED_SCENE}" >&2
  exit 1
fi
if [[ -d "${SCENE_ROOT}" ]] && find "${SCENE_ROOT}" -type f \
    \( -name "feature_t${T}_layer*.sft" -o -name ".feature_t${T}_layer*.tmp-*" \) \
    -print -quit | grep -q .; then
  echo "ERROR: refusing to mix canary with pre-existing local target state: ${SCENE_ROOT}" >&2
  exit 1
fi
mkdir -p "${LOG_DIR}/launcher"
chmod +x scripts/run_streaming_feat_extract_8gpu.sh scripts/run_tstep_sync_loop.sh

cmd="cd ${PROJECT_ROOT} && VFM=${VFM} T=${T} EXTRACT_STREAM=0 EXTRACT_TARGET=1 FEATURE_NAMESPACE=${FEATURE_NAMESPACE} CANONICAL_PROTOCOL=inscene_streaming_canonical_v2_obs28_h124 CANONICAL_PROTOCOL_SHA256=215c720c8537f4dfae6243bedcc7339fbb7e3ba8a6ca2dcaf04f49cd4b738e4f EXTRACT_DONE_INDEX=${DONE_INDEX} N_GPUS=1 REQUIRE_H800=1 SCENE_START=${SCENE_START} SCENE_END=${SCENE_END} SYNC_FEATURE_SCAN_ROOT=${SCENE_ROOT} SYNC_EVERY_SEC=60 SYNC_EVERY_GIB=10 LOG_DIR=${LOG_DIR} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash scripts/run_streaming_feat_extract_8gpu.sh"
{
  echo "job_name=bq_sv1_cgx_target_t499_oom_canary"
  echo "scene=${EXPECTED_SCENE}"
  echo "scene_range=${SCENE_START}:${SCENE_END}"
  echo "done_index=${DONE_INDEX}"
  echo "extractor_sha256=$(sha256sum features/run_inscene15k.py | awk '{print $1}')"
  echo "runner_sha256=$(sha256sum scripts/run_streaming_feat_extract_8gpu.sh | awk '{print $1}')"
  echo "sync_sha256=$(sha256sum scripts/run_tstep_sync_loop.sh | awk '{print $1}')"
  echo "command=${cmd}"
} > "${LOG_DIR}/launcher/PLAN.txt"

if [[ "${DRY_SUBMIT}" == 1 ]]; then
  cat "${LOG_DIR}/launcher/PLAN.txt"
  exit 0
fi

sco acp jobs create \
  --workspace-name=aigc \
  --aec2-name=h800 \
  --job-name=bq_sv1_cgx_target_t499_oom_canary \
  --priority=HIGHEST \
  --container-image-url=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617 \
  --storage-mount=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm \
  --training-framework=pytorch \
  --worker-nodes=1 \
  --worker-spec=N4lS.Iq.I80.1 \
  --command="${cmd}" \
  | tee "${LOG_DIR}/launcher/sco_submit.log"
job_id="$(grep -oE 'pt-[a-z0-9]+' "${LOG_DIR}/launcher/sco_submit.log" | head -1)"
[[ -n "${job_id}" ]] || { echo "ERROR: SCO did not return job id" >&2; exit 1; }
printf '%s\n' "${job_id}" > "${LOG_DIR}/launcher/sco_job_id.txt"
echo "Submitted CogVideoX target OOM canary: ${job_id}"

