#!/usr/bin/env bash
# Extract and strictly audit one deterministic canonical-v2 Wan scene.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/umm/users/yinbaiqiao/InsScene-15K}"
DATA_ROOT="${DATA_ROOT:-${DATASET_ROOT}/data}"
MANIFEST_DIR="${MANIFEST_DIR:-${PROJECT_ROOT}/artifacts/tstep_canonical_v2/manifests}"
CANARY_LOG_ROOT="${CANARY_LOG_ROOT:?set CANARY_LOG_ROOT=<canonical-v2 canary log directory>}"
DONE_DIR="${DONE_DIR:-$(dirname "${CANARY_LOG_ROOT}")/aoss_done}"
PROTOCOL_SPEC="${PROTOCOL_SPEC:-${PROJECT_ROOT}/protocols/inscene_streaming_canonical_v2.json}"
PYTHON="${PYTHON:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}"
CANONICAL_PROTOCOL="${CANONICAL_PROTOCOL:-inscene_streaming_canonical_v2_obs28_h124}"
CANONICAL_PROTOCOL_SHA256="${CANONICAL_PROTOCOL_SHA256:-215c720c8537f4dfae6243bedcc7339fbb7e3ba8a6ca2dcaf04f49cd4b738e4f}"
AEC2_NAME="${AEC2_NAME:-h800}"
SCENE_START="${SCENE_START:-9}"
SCENE_END="${SCENE_END:-10}"
EXPECTED_SCENE="${EXPECTED_SCENE:-033d0b9343}"
SYNC_EVERY_SEC="${SYNC_EVERY_SEC:-60}"
SYNC_EVERY_GIB="${SYNC_EVERY_GIB:-20}"
DRY_RUN="${DRY_RUN:-0}"

STREAMING_NAMESPACE=FEAT_STREAMING_CANONICAL_V2_SEEDED_V1
TARGET_NAMESPACE=FEAT_TARGET_CANONICAL_V2_SEEDED_V1
STREAMING_DONE_INDEX="${DONE_DIR}/${STREAMING_NAMESPACE}_wan_t499.txt"
TARGET_DONE_INDEX="${DONE_DIR}/${TARGET_NAMESPACE}_wan_t499.txt"
RUNNER="${PROJECT_ROOT}/scripts/run_streaming_feat_extract_8gpu.sh"
AUDITOR="${PROJECT_ROOT}/scripts/audit_tstep_canonical_alignment.py"
RECEIPT_BUILDER="${PROJECT_ROOT}/scripts/build_tstep_canonical_v2_canary_receipt.py"
RECEIPT_VERIFIER="${PROJECT_ROOT}/scripts/verify_tstep_canonical_v2_canary_receipt.py"
START_SNAPSHOT="${CANARY_LOG_ROOT}/.CANARY_INPUTS_START.json"
CANARY_RECEIPT="${CANARY_LOG_ROOT}/CANARY_RECEIPT.json"
CAMPAIGN_LOCK="${PROJECT_ROOT}/logs/tstep_canonical_v2_seeded_v1/tstep_canonical_v2_seeded_v1_full_campaign.lock"

[[ "${CANONICAL_PROTOCOL}" == "inscene_streaming_canonical_v2_obs28_h124" ]] || {
  echo "ERROR: invalid canonical protocol: ${CANONICAL_PROTOCOL}" >&2
  exit 2
}
[[ "${CANONICAL_PROTOCOL_SHA256}" == "215c720c8537f4dfae6243bedcc7339fbb7e3ba8a6ca2dcaf04f49cd4b738e4f" ]] || {
  echo "ERROR: invalid canonical protocol SHA256: ${CANONICAL_PROTOCOL_SHA256}" >&2
  exit 2
}
[[ "${AEC2_NAME}" == "h800" ]] || {
  echo "ERROR: seeded-v1 canary is H800-only (AEC2_NAME=${AEC2_NAME})" >&2
  exit 2
}
[[ "${SCENE_START}" == "9" && "${SCENE_END}" == "10" ]] || {
  echo "ERROR: production canary is pinned to SCENE_START=9 SCENE_END=10" >&2
  exit 2
}
case "${DRY_RUN}" in 0|1) ;; *) echo "ERROR: DRY_RUN must be 0 or 1" >&2; exit 2 ;; esac
[[ -x "${PYTHON}" ]] || { echo "ERROR: Python not executable: ${PYTHON}" >&2; exit 1; }
[[ -s "${RUNNER}" ]] || { echo "ERROR: missing extraction runner: ${RUNNER}" >&2; exit 1; }
[[ -s "${AUDITOR}" ]] || { echo "ERROR: missing strict auditor: ${AUDITOR}" >&2; exit 1; }
[[ -s "${RECEIPT_BUILDER}" ]] || { echo "ERROR: missing receipt builder: ${RECEIPT_BUILDER}" >&2; exit 1; }
[[ -s "${RECEIPT_VERIFIER}" ]] || { echo "ERROR: missing receipt verifier: ${RECEIPT_VERIFIER}" >&2; exit 1; }
[[ -s "${PROTOCOL_SPEC}" ]] || { echo "ERROR: missing protocol spec: ${PROTOCOL_SPEC}" >&2; exit 1; }
command -v flock >/dev/null 2>&1 || { echo "ERROR: flock is required" >&2; exit 1; }

mkdir -p "$(dirname "${CAMPAIGN_LOCK}")" "${CANARY_LOG_ROOT}" "${DONE_DIR}"
exec 9>"${CAMPAIGN_LOCK}"
flock -n 9 || {
  echo "ERROR: canonical-v2 canary/full campaign lock is held: ${CAMPAIGN_LOCK}" >&2
  exit 1
}
chmod 0755 "${CANARY_LOG_ROOT}"
if find "${CANARY_LOG_ROOT}" -maxdepth 1 -name 'INVALID*' -print -quit | grep -q .; then
  echo "ERROR: canary directory contains INVALID marker: ${CANARY_LOG_ROOT}" >&2
  exit 1
fi
for stale_receipt in "${START_SNAPSHOT}" "${CANARY_RECEIPT}" "${CANARY_LOG_ROOT}/CANARY_PASS"; do
  [[ ! -e "${stale_receipt}" && ! -L "${stale_receipt}" ]] || {
    echo "ERROR: refusing to overwrite canary evidence: ${stale_receipt}" >&2
    exit 1
  }
done
for interface in SCENE_START SCENE_END; do
  grep -q "${interface}" "${RUNNER}" || {
    echo "ERROR: extraction runner lacks required ${interface} interface; refusing an all-scene canary" >&2
    exit 1
  }
done
for done_index in "${STREAMING_DONE_INDEX}" "${TARGET_DONE_INDEX}"; do
  [[ -f "${done_index}" ]] || {
    echo "ERROR: missing canonical-v2 done-index (legacy indexes are forbidden): ${done_index}" >&2
    exit 1
  }
  [[ ! -L "${done_index}" ]] || {
    echo "ERROR: canonical-v2 done-index must not be a symlink: ${done_index}" >&2
    exit 1
  }
  done_count="$(awk 'NF { count += 1 } END { print count + 0 }' "${done_index}")"
  [[ "${done_count}" -eq 0 ]] || {
    echo "ERROR: fresh canary requires an empty run-specific done-index: ${done_index} (${done_count})" >&2
    exit 1
  }
done

for local_root in \
  "${DATASET_ROOT}/${STREAMING_NAMESPACE}/wan/scannetpp/${EXPECTED_SCENE}" \
  "${DATASET_ROOT}/${TARGET_NAMESPACE}/wan/scannetpp/${EXPECTED_SCENE}"; do
  if [[ -d "${local_root}" ]] && find "${local_root}" -type f \
      \( -name '*.sft' -o -name '*.sft.manifest.json' \) -print -quit | grep -q .; then
    echo "ERROR: fresh canary requires zero local feature tensors/sidecars: ${local_root}" >&2
    exit 1
  fi
done

cd "${PROJECT_ROOT}"
"${PYTHON}" - "${PROTOCOL_SPEC}" "${CANONICAL_PROTOCOL}" "${CANONICAL_PROTOCOL_SHA256}" <<'PY'
import json
import sys

path, protocol, protocol_sha256 = sys.argv[1:]
value = json.load(open(path, encoding="utf-8"))
if value.get("canonical_protocol") != protocol:
    raise SystemExit("ERROR: protocol spec canonical_protocol mismatch")
if value.get("canonical_protocol_sha256") != protocol_sha256:
    raise SystemExit("ERROR: protocol spec canonical_protocol_sha256 mismatch")
PY
"${PYTHON}" scripts/build_tstep_canonical_manifests.py audit \
  --manifest-dir "${MANIFEST_DIR}" \
  --prefixes "8 12 16 24" \
  --expected-shared-units 2721

# Resolve the deterministic pinned scene without loading all trajectory arrays
# during submit preflight. The extraction runner performs the full image/mask/
# metadata inventory again on the allocated worker.
SCENE_ROOT="${DATA_ROOT}/processed_scannetpp_v2"
[[ -d "${SCENE_ROOT}/processed_scannetpp_v2" ]] && \
  SCENE_ROOT="${SCENE_ROOT}/processed_scannetpp_v2"
mapfile -t SCENE_DIRS < <(
  while IFS= read -r scene_dir; do
    [[ -d "${scene_dir}/images" ]] || continue
    [[ -d "${scene_dir}/refined_ins_ids" ]] || continue
    [[ -f "${scene_dir}/scene_iphone_metadata.npz" ]] || continue
    basename "${scene_dir}"
  done < <(find "${SCENE_ROOT}" -mindepth 1 -maxdepth 1 -type d -print) \
    | LC_ALL=C sort
)
[[ "${#SCENE_DIRS[@]}" -eq 682 ]] || {
  echo "ERROR: expected 682 scene directories, found ${#SCENE_DIRS[@]}" >&2
  exit 1
}
actual_scene="${SCENE_DIRS[SCENE_START]}"
[[ "${actual_scene}" == "${EXPECTED_SCENE}" ]] || {
  echo "ERROR: scene index ${SCENE_START} changed: expected ${EXPECTED_SCENE}, got ${actual_scene}" >&2
  exit 1
}
for required_scene_path in \
  "${SCENE_ROOT}/${actual_scene}/images" \
  "${SCENE_ROOT}/${actual_scene}/refined_ins_ids" \
  "${SCENE_ROOT}/${actual_scene}/scene_iphone_metadata.npz"; do
  [[ -e "${required_scene_path}" ]] || {
    echo "ERROR: canary scene input missing: ${required_scene_path}" >&2
    exit 1
  }
done

mkdir -p "${CANARY_LOG_ROOT}/audit"
chmod 0755 "${CANARY_LOG_ROOT}/audit"
echo "canary_scene=${actual_scene} range=[${SCENE_START}:${SCENE_END}) protocol=${CANONICAL_PROTOCOL} dry_run=${DRY_RUN}"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "streaming_done_index=${STREAMING_DONE_INDEX}"
  echo "target_done_index=${TARGET_DONE_INDEX}"
  echo "streaming_root=${DATASET_ROOT}/${STREAMING_NAMESPACE}"
  echo "target_root=${DATASET_ROOT}/${TARGET_NAMESPACE}"
  echo "DRY_RUN=1: source, scene-range, manifest, and canonical done-index gates passed"
  exit 0
fi

"${PYTHON}" "${RECEIPT_BUILDER}" snapshot \
  --project-root "${PROJECT_ROOT}" \
  --protocol-spec "${PROTOCOL_SPEC}" \
  --manifest-dir "${MANIFEST_DIR}" \
  --done-dir "${DONE_DIR}" \
  --dataset-root "${DATASET_ROOT}" \
  --expected-scene "${EXPECTED_SCENE}" \
  --output "${START_SNAPSHOT}"

run_mode() {
  local mode="$1" namespace done_index extract_stream extract_target
  case "${mode}" in
    stream)
      namespace="${STREAMING_NAMESPACE}"
      done_index="${STREAMING_DONE_INDEX}"
      extract_stream=1
      extract_target=0
      ;;
    target)
      namespace="${TARGET_NAMESPACE}"
      done_index="${TARGET_DONE_INDEX}"
      extract_stream=0
      extract_target=1
      ;;
    *) echo "ERROR: invalid canary mode: ${mode}" >&2; exit 2 ;;
  esac
  CANONICAL_PROTOCOL="${CANONICAL_PROTOCOL}" \
  CANONICAL_PROTOCOL_SHA256="${CANONICAL_PROTOCOL_SHA256}" \
  REQUIRE_H800=1 \
  STREAMING_WINDOW_OBS_LEN=28 \
  FEATURE_NAMESPACE="${namespace}" \
  VFM=wan \
  T=499 \
  LAYERS="15 20 25" \
  EXTRACT_STREAM="${extract_stream}" \
  EXTRACT_TARGET="${extract_target}" \
  EXTRACT_DONE_INDEX="${done_index}" \
  DATASET_ROOT="${DATASET_ROOT}" \
  DATA_ROOT="${DATA_ROOT}" \
  OUT_STREAM="${DATASET_ROOT}/${namespace}" \
  OUT_TARGET="${DATASET_ROOT}/${namespace}" \
  SYNC_FEATURE_SCAN_ROOT="${DATASET_ROOT}/${namespace}/wan/scannetpp/${EXPECTED_SCENE}" \
  N_GPUS=1 \
  SCENE_START="${SCENE_START}" \
  SCENE_END="${SCENE_END}" \
  LOG_DIR="${CANARY_LOG_ROOT}/${mode}" \
  SYNC_EVERY_SEC="${SYNC_EVERY_SEC}" \
  SYNC_EVERY_GIB="${SYNC_EVERY_GIB}" \
    bash "${RUNNER}"
}

run_mode stream
run_mode target

# Each per-prefix manifest must independently agree with the same prefix index
# records and with the target union (anchor plus h=1/2/4) for this scene.
for prefix in 8 12 16 24; do
  manifest="${MANIFEST_DIR}/inscene_c_h124_prefix${prefix}.jsonl"
  artifact="${CANARY_LOG_ROOT}/audit/wan_scene_${EXPECTED_SCENE}_p${prefix}.json"
  "${PYTHON}" "${AUDITOR}" audit \
    --manifest "${manifest}" \
    --streaming-index-root "${DATASET_ROOT}/${STREAMING_NAMESPACE}" \
    --target-index-root "${DATASET_ROOT}/${TARGET_NAMESPACE}" \
    --vfm wan \
    --source scannetpp \
    --protocol-id "${CANONICAL_PROTOCOL}" \
    --canonical-obs-len 28 \
    --prefixes "8 12 16 24" \
    --horizons "1 2 4" \
    --scene-ids "${EXPECTED_SCENE}" \
    --output "${artifact}"
  chmod 0644 "${artifact}"
done

"${PYTHON}" "${RECEIPT_BUILDER}" finalize \
  --project-root "${PROJECT_ROOT}" \
  --protocol-spec "${PROTOCOL_SPEC}" \
  --manifest-dir "${MANIFEST_DIR}" \
  --done-dir "${DONE_DIR}" \
  --dataset-root "${DATASET_ROOT}" \
  --expected-scene "${EXPECTED_SCENE}" \
  --start-snapshot "${START_SNAPSHOT}" \
  --receipt "${CANARY_RECEIPT}"
chmod 0644 "${START_SNAPSHOT}" "${CANARY_RECEIPT}"
"${PYTHON}" "${RECEIPT_VERIFIER}" \
  --receipt "${CANARY_RECEIPT}" \
  --project-root "${PROJECT_ROOT}" \
  --protocol-spec "${PROTOCOL_SPEC}" \
  --manifest-dir "${MANIFEST_DIR}" \
  --done-dir "${DONE_DIR}" \
  --dataset-root "${DATASET_ROOT}" \
  --expected-scene "${EXPECTED_SCENE}"

receipt_sha256="$(sha256sum "${CANARY_RECEIPT}" | awk '{print $1}')"
printf '%s\n' "PASS (human-readable only; authority=CANARY_RECEIPT.json) scene=${EXPECTED_SCENE} protocol=${CANONICAL_PROTOCOL} prefixes=8,12,16,24 modes=stream,target layers=15,20,25 t=499 receipt_sha256=${receipt_sha256}" \
  > "${CANARY_LOG_ROOT}/CANARY_PASS"
chmod 0644 "${CANARY_LOG_ROOT}/CANARY_PASS"
echo "CANARY RECEIPT PASS: ${CANARY_RECEIPT} sha256=${receipt_sha256}"
