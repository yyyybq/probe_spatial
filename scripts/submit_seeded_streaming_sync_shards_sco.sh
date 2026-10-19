#!/usr/bin/env bash
# Submit one CPU-heavy, 1xH800 sync-only drain for Wan or CogVideoX streaming.
set -euo pipefail

export PATH="${HOME}/.sco/bin:${PATH}"
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}"
cd "${PROJECT_ROOT}"

VFM="${VFM:?set VFM=wan|cogvideox}"
T="${T:-499}"
SHARD_WORKERS="${SHARD_WORKERS:-8}"
SCENE_START="${SCENE_START:-0}"
SCENE_END="${SCENE_END:-}"
AEC2_NAME="${AEC2_NAME:-h800}"
DRY_SUBMIT="${DRY_SUBMIT:-0}"
RUN_ID="${RUN_ID:-seeded_v1_${VFM}_t${T}_sync_drain_$(date -u +%Y%m%d_%H%M%S)}"
LOG_ROOT="${PROJECT_ROOT}/logs/${RUN_ID}"

[[ "${AEC2_NAME}" == h800 ]] || { echo "ERROR: sync drain is H800-only" >&2; exit 2; }
case "${VFM}:${T}" in
  wan:499) tag=wan ;;
  wan:999) tag=wan999 ;;
  cogvideox:499) tag=cgx ;;
  cogvideox:999) tag=cgx999 ;;
  *) echo "ERROR: only wan:499 and cogvideox:499 are supported" >&2; exit 2 ;;
esac
for path in scripts/run_seeded_streaming_sync_shards.sh scripts/run_tstep_sync_loop.sh; do
  [[ -s "${path}" && ! -L "${path}" ]] || { echo "ERROR: invalid source ${path}" >&2; exit 1; }
done
mkdir -p "${LOG_ROOT}/launcher"
chmod +x scripts/run_seeded_streaming_sync_shards.sh scripts/run_tstep_sync_loop.sh

JOB_NAME="bq_sv1_${tag}_t${T}_sync_drain"
cmd="cd ${PROJECT_ROOT} && VFM=${VFM} T=${T} SHARD_WORKERS=${SHARD_WORKERS} SCENE_START=${SCENE_START} SCENE_END=${SCENE_END} RUN_ID=${RUN_ID} LOG_ROOT=${LOG_ROOT} bash scripts/run_seeded_streaming_sync_shards.sh"
{
  echo "job_name=${JOB_NAME}"
  echo "vfm=${VFM}"
  echo "t=${T}"
  echo "scene_span=${SCENE_START}:${SCENE_END:-<all-existing-scene-roots>}"
  echo "shard_workers=${SHARD_WORKERS}"
  echo "sync_sha256=$(sha256sum scripts/run_tstep_sync_loop.sh | awk '{print $1}')"
  echo "runner_sha256=$(sha256sum scripts/run_seeded_streaming_sync_shards.sh | awk '{print $1}')"
  echo "command=${cmd}"
} > "${LOG_ROOT}/launcher/PLAN.txt"

if [[ "${DRY_SUBMIT}" == 1 ]]; then
  cat "${LOG_ROOT}/launcher/PLAN.txt"
  exit 0
fi

sco acp jobs create \
  --workspace-name=aigc \
  --aec2-name=h800 \
  --job-name="${JOB_NAME}" \
  --priority=HIGHEST \
  --container-image-url=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617 \
  --storage-mount=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm \
  --training-framework=pytorch \
  --worker-nodes=1 \
  --worker-spec=N4lS.Iq.I80.1 \
  --command="${cmd}" \
  | tee "${LOG_ROOT}/launcher/sco_submit.log"

job_id="$(grep -oE 'pt-[a-z0-9]+' "${LOG_ROOT}/launcher/sco_submit.log" | head -1)"
[[ -n "${job_id}" ]] || { echo "ERROR: SCO did not return job id" >&2; exit 1; }
printf '%s\n' "${job_id}" > "${LOG_ROOT}/launcher/sco_job_id.txt"
echo "Submitted ${JOB_NAME}: ${job_id}"
