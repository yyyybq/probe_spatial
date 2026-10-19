#!/usr/bin/env bash
# Submit the one-scene canonical-v2 Wan alignment canary to one H800 GPU.
set -euo pipefail

export PATH="${HOME}/.sco/bin:${PATH}"
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}"
PROTOCOL_SPEC="${PROTOCOL_SPEC:-${PROJECT_ROOT}/protocols/inscene_streaming_canonical_v2.json}"
MANIFEST_DIR="${MANIFEST_DIR:-${PROJECT_ROOT}/artifacts/tstep_canonical_v2/manifests}"
RUN_ID="${RUN_ID:-tstep_canonical_v2_seeded_v1_canary_$(date -u +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/logs/${RUN_ID}}"
CANARY_LOG_ROOT="${CANARY_LOG_ROOT:-${OUT_DIR}/canary}"
DONE_DIR="${DONE_DIR:-${OUT_DIR}/aoss_done}"
JOB_NAME="${JOB_NAME:-bq_tstep_v2_sv1_wan_t499_canary}"
PYTHON="${PYTHON:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}"
AEC2_NAME="${AEC2_NAME:-h800}"
DRY_SUBMIT="${DRY_SUBMIT:-0}"
CANONICAL_PROTOCOL=inscene_streaming_canonical_v2_obs28_h124
CANONICAL_PROTOCOL_SHA256=215c720c8537f4dfae6243bedcc7339fbb7e3ba8a6ca2dcaf04f49cd4b738e4f

[[ "${AEC2_NAME}" == "h800" ]] || {
  echo "ERROR: seeded-v1 canary must run on AEC2_NAME=h800 (got ${AEC2_NAME})" >&2
  exit 2
}
WORKER_SPEC="N4lS.Iq.I80.1"

cd "${PROJECT_ROOT}"
require_min_bytes() {
  local path="$1" minimum="$2" size=0
  [[ -f "${path}" ]] && size="$(wc -c < "${path}")"
  if [[ "${size}" -lt "${minimum}" ]]; then
    echo "ERROR: refusing canary submit; source looks truncated: ${path} (${size} < ${minimum} bytes)" >&2
    exit 1
  fi
}
require_min_bytes features/run_inscene15k.py 50000
require_min_bytes vidfm3d/utils/temporal_sampling.py 5000
require_min_bytes features/noise_seed.py 5000
require_min_bytes features/wan/wan_feature.py 10000
require_min_bytes features/wan/wan_feature_i2v.py 10000
require_min_bytes features/wan/extract_features.py 5000
require_min_bytes features/cogvideox/cogvideox_feature.py 10000
require_min_bytes features/cogvideox/cogvideox_feature_i2v.py 10000
require_min_bytes features/cogvideox/extract_features.py 5000
require_min_bytes features/minimax_h3/minimax_h3_feature.py 10000
require_min_bytes features/minimax_h3/extract_features.py 1000
require_min_bytes scripts/run_streaming_feat_extract_8gpu.sh 5000
require_min_bytes scripts/run_tstep_sync_loop.sh 5000
require_min_bytes scripts/run_tstep_canonical_v2_canary.sh 5000
require_min_bytes scripts/audit_tstep_canonical_alignment.py 5000
require_min_bytes scripts/build_tstep_canonical_manifests.py 5000
require_min_bytes scripts/build_tstep_canonical_v2_canary_receipt.py 3000
require_min_bytes scripts/verify_tstep_canonical_v2_canary_receipt.py 10000

for invalid_marker in \
  reports/tstep_canonical_v2/INVALID_R1_20260830.md \
  logs/feat_extract_tstep_campaign_r7_20260828/INVALID \
  logs/tstep_probe_campaigns/tstep_probe_wan_r1_20260828/INVALID; do
  [[ -s "${invalid_marker}" ]] || {
    echo "ERROR: frozen legacy INVALID marker missing/empty: ${invalid_marker}" >&2
    exit 1
  }
done
[[ -x "${PYTHON}" ]] || { echo "ERROR: Python not executable: ${PYTHON}" >&2; exit 1; }
"${PYTHON}" -c '
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
assert p.get("canonical_protocol") == "inscene_streaming_canonical_v2_obs28_h124"
assert p.get("canonical_protocol_sha256") == "215c720c8537f4dfae6243bedcc7339fbb7e3ba8a6ca2dcaf04f49cd4b738e4f"
assert p.get("canonical_window", {}).get("observations") == 28
assert p.get("streaming", {}).get("prefix_lengths") == [8, 12, 16, 24]
assert p.get("target", {}).get("future_horizons") == [1, 2, 4]
assert p.get("target", {}).get("include_anchors") is True
assert p.get("feature_namespaces", {}).get("streaming") == "FEAT_STREAMING_CANONICAL_V2_SEEDED_V1"
assert p.get("feature_namespaces", {}).get("target") == "FEAT_TARGET_CANONICAL_V2_SEEDED_V1"
assert p.get("feature_generation", {}).get("contract") == "inscene_deterministic_noise_v1"
' "${PROTOCOL_SPEC}" || {
  echo "ERROR: canonical-v2 protocol spec validation failed: ${PROTOCOL_SPEC}" >&2
  exit 1
}
"${PYTHON}" scripts/build_tstep_canonical_manifests.py audit \
  --manifest-dir "${MANIFEST_DIR}" \
  --prefixes "8 12 16 24" \
  --expected-shared-units 2721

mkdir -p "${OUT_DIR}/launcher" "${CANARY_LOG_ROOT}"
mkdir -p "${DONE_DIR}"
chmod 0775 "${DONE_DIR}"
for done_index in \
  "${DONE_DIR}/FEAT_STREAMING_CANONICAL_V2_SEEDED_V1_wan_t499.txt" \
  "${DONE_DIR}/FEAT_TARGET_CANONICAL_V2_SEEDED_V1_wan_t499.txt"; do
  [[ ! -L "${done_index}" ]] || {
    echo "ERROR: run-specific done-index must not be a symlink: ${done_index}" >&2
    exit 1
  }
  if [[ -e "${done_index}" ]]; then
    [[ -f "${done_index}" && ! -s "${done_index}" ]] || {
      echo "ERROR: refusing non-empty/non-regular run-specific done-index: ${done_index}" >&2
      exit 1
    }
  else
    touch "${done_index}"
  fi
  chmod 0664 "${done_index}"
done
chmod +x \
  scripts/run_tstep_canonical_v2_canary.sh \
  scripts/run_streaming_feat_extract_8gpu.sh \
  scripts/run_tstep_sync_loop.sh \
  scripts/build_tstep_canonical_v2_canary_receipt.py \
  scripts/verify_tstep_canonical_v2_canary_receipt.py

PROJECT_ROOT="${PROJECT_ROOT}" \
CANONICAL_PROTOCOL="${CANONICAL_PROTOCOL}" \
CANONICAL_PROTOCOL_SHA256="${CANONICAL_PROTOCOL_SHA256}" \
MANIFEST_DIR="${MANIFEST_DIR}" \
DONE_DIR="${DONE_DIR}" \
PROTOCOL_SPEC="${PROTOCOL_SPEC}" \
CANARY_LOG_ROOT="${OUT_DIR}/launcher/preflight" \
AEC2_NAME="${AEC2_NAME}" \
SCENE_START=9 \
SCENE_END=10 \
EXPECTED_SCENE=033d0b9343 \
DRY_RUN=1 \
  bash scripts/run_tstep_canonical_v2_canary.sh

remote=(
  env
  "PROJECT_ROOT=${PROJECT_ROOT}"
  "CANONICAL_PROTOCOL=${CANONICAL_PROTOCOL}"
  "CANONICAL_PROTOCOL_SHA256=${CANONICAL_PROTOCOL_SHA256}"
  "MANIFEST_DIR=${MANIFEST_DIR}"
  "DONE_DIR=${DONE_DIR}"
  "PROTOCOL_SPEC=${PROTOCOL_SPEC}"
  "CANARY_LOG_ROOT=${CANARY_LOG_ROOT}"
  "AEC2_NAME=${AEC2_NAME}"
  SCENE_START=9
  SCENE_END=10
  EXPECTED_SCENE=033d0b9343
  bash scripts/run_tstep_canonical_v2_canary.sh
)
printf -v remote_command '%q ' "${remote[@]}"
printf -v project_q '%q' "${PROJECT_ROOT}"
command="cd ${project_q} && ${remote_command}"

{
  printf 'RUN_ID=%s\n' "${RUN_ID}"
  printf 'job_name=%s\n' "${JOB_NAME}"
  printf 'worker=%s:%s\n' "${AEC2_NAME}" "${WORKER_SPEC}"
  printf 'scene_range=9:10\n'
  printf 'expected_scene=033d0b9343\n'
  printf 'vfm=wan\n'
  printf 'timestep=499\n'
  printf 'modes=streaming_prefix,target_isolated\n'
  printf 'layers=15,20,25\n'
  printf 'canonical_protocol=%s\n' "${CANONICAL_PROTOCOL}"
  printf 'canonical_protocol_sha256=%s\n' "${CANONICAL_PROTOCOL_SHA256}"
  printf 'protocol_spec=%s\n' "${PROTOCOL_SPEC}"
  printf 'manifest_dir=%s\n' "${MANIFEST_DIR}"
  printf 'done_dir=%s\n' "${DONE_DIR}"
  printf 'canary_log_root=%s\n' "${CANARY_LOG_ROOT}"
  printf 'canary_receipt=%s\n' "${CANARY_LOG_ROOT}/CANARY_RECEIPT.json"
} > "${OUT_DIR}/PLAN.txt"

if [[ "${DRY_SUBMIT}" == "1" ]]; then
  echo "DRY_SUBMIT=1"
  sed -n '1,120p' "${OUT_DIR}/PLAN.txt"
  echo "command=${command}"
  exit 0
fi
[[ "${DRY_SUBMIT}" == "0" ]] || { echo "ERROR: DRY_SUBMIT must be 0 or 1" >&2; exit 2; }

submit_log="${OUT_DIR}/launcher/sco_submit.log"
sco acp jobs create \
  --workspace-name=aigc \
  --aec2-name="${AEC2_NAME}" \
  --job-name="${JOB_NAME}" \
  --priority=HIGHEST \
  --container-image-url=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617 \
  --storage-mount=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm \
  --training-framework=pytorch \
  --worker-nodes=1 \
  --worker-spec="${WORKER_SPEC}" \
  --command="${command}" \
  | tee "${submit_log}"
job_id="$(grep -oE 'pt-[a-z0-9]+' "${submit_log}" | head -n 1)"
[[ -n "${job_id}" ]] || { echo "ERROR: SCO did not return a job id" >&2; exit 1; }
printf '%s\n' "${job_id}" > "${OUT_DIR}/launcher/sco_job_id.txt"
echo "Submitted ${JOB_NAME}: ${job_id}"
echo "Canary logs: ${CANARY_LOG_ROOT}"
