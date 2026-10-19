#!/usr/bin/env bash
# Submit the canonical-v2 extraction campaign to exactly one 8-GPU node.
set -euo pipefail

export PATH="${HOME}/.sco/bin:${PATH}"
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}"
PROTOCOL_SPEC="${PROTOCOL_SPEC:-${PROJECT_ROOT}/protocols/inscene_streaming_canonical_v2.json}"
MANIFEST_DIR="${MANIFEST_DIR:-${PROJECT_ROOT}/artifacts/tstep_canonical_v2/manifests}"
DONE_DIR="${DONE_DIR:-${PROJECT_ROOT}/logs/tstep_canonical_v2_seeded_v1/aoss_done}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/umm/users/yinbaiqiao/InsScene-15K}"
CANARY_RECEIPT_MANIFEST="${CANARY_RECEIPT_MANIFEST:-${CANARY_PASS_FILE:-}}"
[[ -n "${CANARY_RECEIPT_MANIFEST}" ]] || {
  echo "ERROR: set CANARY_RECEIPT_MANIFEST=<successful canonical-v2 CANARY_RECEIPT.json>" >&2
  exit 2
}
RUN_ID="${RUN_ID:-tstep_canonical_v2_seeded_v1_$(date -u +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/logs/feat_extract_${RUN_ID}}"
CAMPAIGN_LOG_ROOT="${CAMPAIGN_LOG_ROOT:-${OUT_DIR}/campaign}"
CAMPAIGN_LOCK_FILE="${CAMPAIGN_LOCK_FILE:-${PROJECT_ROOT}/logs/tstep_canonical_v2_seeded_v1/tstep_canonical_v2_seeded_v1_full_campaign.lock}"
JOB_NAME="${JOB_NAME:-bq_tstep_v2_seeded_v1_all_8gpu}"
PYTHON="${PYTHON:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}"
AEC2_NAME="${AEC2_NAME:-h800}"
SYNC_EVERY_SEC="${SYNC_EVERY_SEC:-180}"
SYNC_EVERY_GIB="${SYNC_EVERY_GIB:-80}"
STEPS="${STEPS:-}"
START_AT="${START_AT:-1}"
DRY_SUBMIT="${DRY_SUBMIT:-0}"

CANONICAL_PROTOCOL=inscene_streaming_canonical_v2_obs28_h124
CANONICAL_PROTOCOL_SHA256=215c720c8537f4dfae6243bedcc7339fbb7e3ba8a6ca2dcaf04f49cd4b738e4f
STREAMING_WINDOW_OBS_LEN=28
STREAMING_FEATURE_NAMESPACE=FEAT_STREAMING_CANONICAL_V2_SEEDED_V1
TARGET_FEATURE_NAMESPACE=FEAT_TARGET_CANONICAL_V2_SEEDED_V1

[[ "${AEC2_NAME}" == "h800" ]] || {
  echo "ERROR: seeded-v1 full extraction must run on AEC2_NAME=h800 (got ${AEC2_NAME})" >&2
  exit 2
}
WORKER_SPEC="N4lS.Iq.I80.8"

cd "${PROJECT_ROOT}"
require_min_bytes() {
  local path="$1" minimum="$2" size=0
  [[ -f "${path}" ]] && size="$(wc -c < "${path}")"
  if [[ "${size}" -lt "${minimum}" ]]; then
    echo "ERROR: refusing submit; source looks truncated: ${path} (${size} < ${minimum} bytes)" >&2
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
require_min_bytes scripts/run_tstep_extract_campaign_v2.sh 5000
require_min_bytes scripts/build_tstep_canonical_manifests.py 5000
require_min_bytes scripts/verify_tstep_canonical_v2_canary_receipt.py 20000

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
[[ -s "${CANARY_RECEIPT_MANIFEST}" && ! -L "${CANARY_RECEIPT_MANIFEST}" ]] || {
  echo "ERROR: canonical-v2 canary receipt is missing, empty, or a symlink: ${CANARY_RECEIPT_MANIFEST}" >&2
  exit 1
}
if find "$(dirname "${CANARY_RECEIPT_MANIFEST}")" -maxdepth 1 -type f -name 'INVALID*' -print -quit \
  | grep -q .; then
  echo "ERROR: canonical-v2 canary receipt directory is frozen INVALID: $(dirname "${CANARY_RECEIPT_MANIFEST}")" >&2
  exit 1
fi

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

# The plain-text CANARY_PASS is informational only. This verifier replays the
# four scene-level audits, checks the fresh canary's isolated done indexes and
# upload/delete logs, rejects INVALID directories, and binds current sources.
"${PYTHON}" scripts/verify_tstep_canonical_v2_canary_receipt.py \
  --receipt "${CANARY_RECEIPT_MANIFEST}" \
  --project-root "${PROJECT_ROOT}" \
  --protocol-spec "${PROTOCOL_SPEC}" \
  --manifest-dir "${MANIFEST_DIR}" \
  --dataset-root "${DATASET_ROOT}" \
  --expected-scene 033d0b9343

sha256_file() {
  sha256sum -- "$1" | awk '{print $1}'
}
EXPECTED_PROTOCOL_SPEC_SHA256="$(sha256_file "${PROTOCOL_SPEC}")"
EXPECTED_CAMPAIGN_RUNNER_SHA256="$(sha256_file scripts/run_tstep_extract_campaign_v2.sh)"
EXPECTED_EXTRACTOR_SHA256="$(sha256_file features/run_inscene15k.py)"
EXPECTED_TEMPORAL_SAMPLING_SHA256="$(sha256_file vidfm3d/utils/temporal_sampling.py)"
EXPECTED_NOISE_SEED_HELPER_SHA256="$(sha256_file features/noise_seed.py)"
EXPECTED_WAN_FEATURE_SHA256="$(sha256_file features/wan/wan_feature.py)"
EXPECTED_WAN_I2V_FEATURE_SHA256="$(sha256_file features/wan/wan_feature_i2v.py)"
EXPECTED_WAN_WRAPPER_SHA256="$(sha256_file features/wan/extract_features.py)"
EXPECTED_COGVIDEOX_FEATURE_SHA256="$(sha256_file features/cogvideox/cogvideox_feature.py)"
EXPECTED_COGVIDEOX_I2V_FEATURE_SHA256="$(sha256_file features/cogvideox/cogvideox_feature_i2v.py)"
EXPECTED_COGVIDEOX_WRAPPER_SHA256="$(sha256_file features/cogvideox/extract_features.py)"
EXPECTED_MINIMAX_H3_FEATURE_SHA256="$(sha256_file features/minimax_h3/minimax_h3_feature.py)"
EXPECTED_MINIMAX_H3_WRAPPER_SHA256="$(sha256_file features/minimax_h3/extract_features.py)"
EXPECTED_EXTRACT_RUNNER_SHA256="$(sha256_file scripts/run_streaming_feat_extract_8gpu.sh)"
EXPECTED_SYNC_LOOP_SHA256="$(sha256_file scripts/run_tstep_sync_loop.sh)"
EXPECTED_MANIFEST_P8_SHA256="$(sha256_file "${MANIFEST_DIR}/inscene_c_h124_prefix8.jsonl")"
EXPECTED_MANIFEST_P12_SHA256="$(sha256_file "${MANIFEST_DIR}/inscene_c_h124_prefix12.jsonl")"
EXPECTED_MANIFEST_P16_SHA256="$(sha256_file "${MANIFEST_DIR}/inscene_c_h124_prefix16.jsonl")"
EXPECTED_MANIFEST_P24_SHA256="$(sha256_file "${MANIFEST_DIR}/inscene_c_h124_prefix24.jsonl")"
CANARY_RECEIPT_SHA256="$(sha256_file "${CANARY_RECEIPT_MANIFEST}")"

mkdir -p "${OUT_DIR}/launcher" "${CAMPAIGN_LOG_ROOT}"
chmod +x \
  scripts/run_tstep_extract_campaign_v2.sh \
  scripts/run_streaming_feat_extract_8gpu.sh \
  scripts/run_tstep_sync_loop.sh

# Exercise the exact step filter and every per-step canonical done-index before
# allocating a node. The delegated runner exits before model loading/sync.
PROJECT_ROOT="${PROJECT_ROOT}" \
CANONICAL_PROTOCOL="${CANONICAL_PROTOCOL}" \
CANONICAL_PROTOCOL_SHA256="${CANONICAL_PROTOCOL_SHA256}" \
STREAMING_WINDOW_OBS_LEN="${STREAMING_WINDOW_OBS_LEN}" \
STREAMING_FEATURE_NAMESPACE="${STREAMING_FEATURE_NAMESPACE}" \
TARGET_FEATURE_NAMESPACE="${TARGET_FEATURE_NAMESPACE}" \
DONE_DIR="${DONE_DIR}" \
DATASET_ROOT="${DATASET_ROOT}" \
SOURCE=scannetpp \
PROTOCOL_SPEC="${PROTOCOL_SPEC}" \
MANIFEST_DIR="${MANIFEST_DIR}" \
GATE_PYTHON="${PYTHON}" \
AEC2_NAME="${AEC2_NAME}" \
CAMPAIGN_LOCK_FILE="${CAMPAIGN_LOCK_FILE}" \
EXPECTED_PROTOCOL_SPEC_SHA256="${EXPECTED_PROTOCOL_SPEC_SHA256}" \
EXPECTED_CAMPAIGN_RUNNER_SHA256="${EXPECTED_CAMPAIGN_RUNNER_SHA256}" \
EXPECTED_EXTRACTOR_SHA256="${EXPECTED_EXTRACTOR_SHA256}" \
EXPECTED_TEMPORAL_SAMPLING_SHA256="${EXPECTED_TEMPORAL_SAMPLING_SHA256}" \
EXPECTED_NOISE_SEED_HELPER_SHA256="${EXPECTED_NOISE_SEED_HELPER_SHA256}" \
EXPECTED_WAN_FEATURE_SHA256="${EXPECTED_WAN_FEATURE_SHA256}" \
EXPECTED_WAN_I2V_FEATURE_SHA256="${EXPECTED_WAN_I2V_FEATURE_SHA256}" \
EXPECTED_WAN_WRAPPER_SHA256="${EXPECTED_WAN_WRAPPER_SHA256}" \
EXPECTED_COGVIDEOX_FEATURE_SHA256="${EXPECTED_COGVIDEOX_FEATURE_SHA256}" \
EXPECTED_COGVIDEOX_I2V_FEATURE_SHA256="${EXPECTED_COGVIDEOX_I2V_FEATURE_SHA256}" \
EXPECTED_COGVIDEOX_WRAPPER_SHA256="${EXPECTED_COGVIDEOX_WRAPPER_SHA256}" \
EXPECTED_MINIMAX_H3_FEATURE_SHA256="${EXPECTED_MINIMAX_H3_FEATURE_SHA256}" \
EXPECTED_MINIMAX_H3_WRAPPER_SHA256="${EXPECTED_MINIMAX_H3_WRAPPER_SHA256}" \
EXPECTED_EXTRACT_RUNNER_SHA256="${EXPECTED_EXTRACT_RUNNER_SHA256}" \
EXPECTED_SYNC_LOOP_SHA256="${EXPECTED_SYNC_LOOP_SHA256}" \
EXPECTED_MANIFEST_P8_SHA256="${EXPECTED_MANIFEST_P8_SHA256}" \
EXPECTED_MANIFEST_P12_SHA256="${EXPECTED_MANIFEST_P12_SHA256}" \
EXPECTED_MANIFEST_P16_SHA256="${EXPECTED_MANIFEST_P16_SHA256}" \
EXPECTED_MANIFEST_P24_SHA256="${EXPECTED_MANIFEST_P24_SHA256}" \
CAMPAIGN_LOG_ROOT="${OUT_DIR}/launcher/preflight" \
SYNC_EVERY_SEC="${SYNC_EVERY_SEC}" \
SYNC_EVERY_GIB="${SYNC_EVERY_GIB}" \
STEPS="${STEPS}" \
START_AT="${START_AT}" \
DRY_RUN=1 \
  bash scripts/run_tstep_extract_campaign_v2.sh

remote=(
  env
  "PROJECT_ROOT=${PROJECT_ROOT}"
  "CANONICAL_PROTOCOL=${CANONICAL_PROTOCOL}"
  "CANONICAL_PROTOCOL_SHA256=${CANONICAL_PROTOCOL_SHA256}"
  "STREAMING_WINDOW_OBS_LEN=${STREAMING_WINDOW_OBS_LEN}"
  "STREAMING_FEATURE_NAMESPACE=${STREAMING_FEATURE_NAMESPACE}"
  "TARGET_FEATURE_NAMESPACE=${TARGET_FEATURE_NAMESPACE}"
  "DONE_DIR=${DONE_DIR}"
  "DATASET_ROOT=${DATASET_ROOT}"
  "SOURCE=scannetpp"
  "PROTOCOL_SPEC=${PROTOCOL_SPEC}"
  "MANIFEST_DIR=${MANIFEST_DIR}"
  "GATE_PYTHON=${PYTHON}"
  "AEC2_NAME=${AEC2_NAME}"
  "CAMPAIGN_LOCK_FILE=${CAMPAIGN_LOCK_FILE}"
  "EXPECTED_PROTOCOL_SPEC_SHA256=${EXPECTED_PROTOCOL_SPEC_SHA256}"
  "EXPECTED_CAMPAIGN_RUNNER_SHA256=${EXPECTED_CAMPAIGN_RUNNER_SHA256}"
  "EXPECTED_EXTRACTOR_SHA256=${EXPECTED_EXTRACTOR_SHA256}"
  "EXPECTED_TEMPORAL_SAMPLING_SHA256=${EXPECTED_TEMPORAL_SAMPLING_SHA256}"
  "EXPECTED_NOISE_SEED_HELPER_SHA256=${EXPECTED_NOISE_SEED_HELPER_SHA256}"
  "EXPECTED_WAN_FEATURE_SHA256=${EXPECTED_WAN_FEATURE_SHA256}"
  "EXPECTED_WAN_I2V_FEATURE_SHA256=${EXPECTED_WAN_I2V_FEATURE_SHA256}"
  "EXPECTED_WAN_WRAPPER_SHA256=${EXPECTED_WAN_WRAPPER_SHA256}"
  "EXPECTED_COGVIDEOX_FEATURE_SHA256=${EXPECTED_COGVIDEOX_FEATURE_SHA256}"
  "EXPECTED_COGVIDEOX_I2V_FEATURE_SHA256=${EXPECTED_COGVIDEOX_I2V_FEATURE_SHA256}"
  "EXPECTED_COGVIDEOX_WRAPPER_SHA256=${EXPECTED_COGVIDEOX_WRAPPER_SHA256}"
  "EXPECTED_MINIMAX_H3_FEATURE_SHA256=${EXPECTED_MINIMAX_H3_FEATURE_SHA256}"
  "EXPECTED_MINIMAX_H3_WRAPPER_SHA256=${EXPECTED_MINIMAX_H3_WRAPPER_SHA256}"
  "EXPECTED_EXTRACT_RUNNER_SHA256=${EXPECTED_EXTRACT_RUNNER_SHA256}"
  "EXPECTED_SYNC_LOOP_SHA256=${EXPECTED_SYNC_LOOP_SHA256}"
  "EXPECTED_MANIFEST_P8_SHA256=${EXPECTED_MANIFEST_P8_SHA256}"
  "EXPECTED_MANIFEST_P12_SHA256=${EXPECTED_MANIFEST_P12_SHA256}"
  "EXPECTED_MANIFEST_P16_SHA256=${EXPECTED_MANIFEST_P16_SHA256}"
  "EXPECTED_MANIFEST_P24_SHA256=${EXPECTED_MANIFEST_P24_SHA256}"
  "CAMPAIGN_LOG_ROOT=${CAMPAIGN_LOG_ROOT}"
  "SYNC_EVERY_SEC=${SYNC_EVERY_SEC}"
  "SYNC_EVERY_GIB=${SYNC_EVERY_GIB}"
  "STEPS=${STEPS}"
  "START_AT=${START_AT}"
  bash scripts/run_tstep_extract_campaign_v2.sh
)
printf -v remote_command '%q ' "${remote[@]}"
printf -v project_q '%q' "${PROJECT_ROOT}"
printf -v campaign_runner_q '%q' 'scripts/run_tstep_extract_campaign_v2.sh'
printf -v campaign_hash_line_q '%q' \
  "${EXPECTED_CAMPAIGN_RUNNER_SHA256}  scripts/run_tstep_extract_campaign_v2.sh"
# This immutable SCO command checks the shared campaign runner before invoking
# it. An empty/truncated queued-worktree script must not turn into exit code 0.
command="cd ${project_q} && test \"\$(sha256sum -- ${campaign_runner_q})\" = ${campaign_hash_line_q} && ${remote_command}"

{
  printf 'RUN_ID=%s\n' "${RUN_ID}"
  printf 'job_name=%s\n' "${JOB_NAME}"
  printf 'worker=%s:%s\n' "${AEC2_NAME}" "${WORKER_SPEC}"
  printf 'protocol=%s\n' "${CANONICAL_PROTOCOL}"
  printf 'protocol_sha256=%s\n' "${CANONICAL_PROTOCOL_SHA256}"
  printf 'protocol_spec=%s\n' "${PROTOCOL_SPEC}"
  printf 'manifest_dir=%s\n' "${MANIFEST_DIR}"
  printf 'canary_receipt_manifest=%s\n' "${CANARY_RECEIPT_MANIFEST}"
  printf 'canary_receipt_sha256=%s\n' "${CANARY_RECEIPT_SHA256}"
  printf 'streaming_namespace=%s\n' "${STREAMING_FEATURE_NAMESPACE}"
  printf 'target_namespace=%s\n' "${TARGET_FEATURE_NAMESPACE}"
  printf 'done_dir=%s\n' "${DONE_DIR}"
  printf 'dataset_root=%s\n' "${DATASET_ROOT}"
  printf 'steps=%s\n' "${STEPS:-<all>}"
  printf 'start_at=%s\n' "${START_AT}"
  printf 'campaign_log_root=%s\n' "${CAMPAIGN_LOG_ROOT}"
  printf 'campaign_lock_file=%s\n' "${CAMPAIGN_LOCK_FILE}"
  printf 'protocol_spec_sha256=%s\n' "${EXPECTED_PROTOCOL_SPEC_SHA256}"
  printf 'campaign_runner_sha256=%s\n' "${EXPECTED_CAMPAIGN_RUNNER_SHA256}"
  printf 'extractor_sha256=%s\n' "${EXPECTED_EXTRACTOR_SHA256}"
  printf 'temporal_sampling_sha256=%s\n' "${EXPECTED_TEMPORAL_SAMPLING_SHA256}"
  printf 'noise_seed_helper_sha256=%s\n' "${EXPECTED_NOISE_SEED_HELPER_SHA256}"
  printf 'wan_feature_sha256=%s\n' "${EXPECTED_WAN_FEATURE_SHA256}"
  printf 'wan_i2v_feature_sha256=%s\n' "${EXPECTED_WAN_I2V_FEATURE_SHA256}"
  printf 'wan_wrapper_sha256=%s\n' "${EXPECTED_WAN_WRAPPER_SHA256}"
  printf 'cogvideox_feature_sha256=%s\n' "${EXPECTED_COGVIDEOX_FEATURE_SHA256}"
  printf 'cogvideox_i2v_feature_sha256=%s\n' "${EXPECTED_COGVIDEOX_I2V_FEATURE_SHA256}"
  printf 'cogvideox_wrapper_sha256=%s\n' "${EXPECTED_COGVIDEOX_WRAPPER_SHA256}"
  printf 'minimax_h3_feature_sha256=%s\n' "${EXPECTED_MINIMAX_H3_FEATURE_SHA256}"
  printf 'minimax_h3_wrapper_sha256=%s\n' "${EXPECTED_MINIMAX_H3_WRAPPER_SHA256}"
  printf 'extract_runner_sha256=%s\n' "${EXPECTED_EXTRACT_RUNNER_SHA256}"
  printf 'sync_loop_sha256=%s\n' "${EXPECTED_SYNC_LOOP_SHA256}"
  printf 'manifest_p8_sha256=%s\n' "${EXPECTED_MANIFEST_P8_SHA256}"
  printf 'manifest_p12_sha256=%s\n' "${EXPECTED_MANIFEST_P12_SHA256}"
  printf 'manifest_p16_sha256=%s\n' "${EXPECTED_MANIFEST_P16_SHA256}"
  printf 'manifest_p24_sha256=%s\n' "${EXPECTED_MANIFEST_P24_SHA256}"
  printf 'sync_every_sec=%s\n' "${SYNC_EVERY_SEC}"
  printf 'sync_every_gib=%s\n' "${SYNC_EVERY_GIB}"
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
echo "Campaign logs: ${CAMPAIGN_LOG_ROOT}"
