#!/usr/bin/env bash
set -euo pipefail

export PATH="${HOME}/.sco/bin:${PATH}"

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
PY=${PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}
RUN_TAG=${RUN_TAG:-wan_p8_action_dynamics_canonical_manifest_exact_$(date -u +%Y%m%d_%H%M%S)_$(git -C "${PROJECT_ROOT}" rev-parse --short HEAD)}
FEATURE_ROOT=${FEATURE_ROOT:-/mnt/umm/users/yinbaiqiao/InsScene-15K/FEATURES_VERSIONED/${RUN_TAG}}
REPORT_DIR=${PROJECT_ROOT}/reports/wan_p8_recovery/${RUN_TAG}
MANIFEST=${PROJECT_ROOT}/manifests/c_retrain_20260722/inscene_c_h124_prefix8.jsonl
CANARY_TAG=wan_p8_manifest_exact_canary_20260821_193845_2075f3e
CANARY_REPORT=${PROJECT_ROOT}/reports/wan_p8_recovery/${CANARY_TAG}/canary_report.md
CANARY_SUMMARY=${PROJECT_ROOT}/reports/wan_p8_recovery/${CANARY_TAG}/canary_summary.json
CANARY_RUNTIME=${PROJECT_ROOT}/reports/wan_p8_recovery/${CANARY_TAG}/sco_runtime_status.json
MODEL=/mnt/umm/users/yinbaiqiao/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780a534b6463e45facd96134c9f345acfa5b
AEC2_NAME=${AEC2_NAME:-zoetrope}
CONTAINER_IMAGE=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617
STORAGE_MOUNT=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm
WORKER_SPEC=N4lS.Iq.I80.1

cd "${PROJECT_ROOT}"
required=(
  scripts/extract_wan_manifest_exact.py
  scripts/prepare_wan_p8_exact_full.py
  scripts/run_wan_p8_exact_full_shard_sco.sh
  "${MANIFEST}"
  "${CANARY_REPORT}"
  "${CANARY_SUMMARY}"
  "${CANARY_RUNTIME}"
  "${MODEL}/model_index.json"
)
for path in "${required[@]}"; do
  [[ -f "${path}" ]] || { echo "missing required file: ${path}" >&2; exit 2; }
done

actual_manifest_sha=$(sha256sum "${MANIFEST}" | awk '{print $1}')
[[ "${actual_manifest_sha}" == "772194f077c45cc85a942f17f46f50103125ae8f5b1c983c08881c1759607e0c" ]] \
  || { echo "canonical manifest hash mismatch: ${actual_manifest_sha}" >&2; exit 2; }
"${PY}" - "${CANARY_SUMMARY}" "${CANARY_RUNTIME}" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))
runtime = json.load(open(sys.argv[2]))
assert summary.get("ok") is True, summary.get("problems")
assert summary.get("streaming_exact_raw") == 40
assert summary.get("target_exact_raw") == 40
assert summary.get("problem_count") == 0
assert runtime.get("rc") == 0
print("canary gate: PASS")
PY

[[ ! -e "${FEATURE_ROOT}" ]] || { echo "refusing existing FEATURE_ROOT=${FEATURE_ROOT}" >&2; exit 2; }
[[ ! -e "${REPORT_DIR}" ]] || { echo "refusing existing REPORT_DIR=${REPORT_DIR}" >&2; exit 2; }
df -h /mnt/umm/users/yinbaiqiao/InsScene-15K
df -i /mnt/umm/users/yinbaiqiao/InsScene-15K

mkdir -p "${REPORT_DIR}/launcher" "${REPORT_DIR}/logs" "${REPORT_DIR}/provenance/shards"
PROVENANCE_ROOT=${REPORT_DIR}/provenance
git -C "${PROJECT_ROOT}" rev-parse HEAD > "${PROVENANCE_ROOT}/git_commit.txt"
git -C "${PROJECT_ROOT}" status --short > "${PROVENANCE_ROOT}/git_status_short.txt"
git -C "${PROJECT_ROOT}" diff --binary HEAD > "${PROVENANCE_ROOT}/git_dirty.diff"
cp "${MANIFEST}" "${PROVENANCE_ROOT}/canonical_manifest.jsonl"
cp scripts/extract_wan_manifest_exact.py "${PROVENANCE_ROOT}/extract_wan_manifest_exact.py"
cp scripts/prepare_wan_p8_exact_full.py "${PROVENANCE_ROOT}/prepare_wan_p8_exact_full.py"
cp scripts/run_wan_p8_exact_full_shard_sco.sh "${PROVENANCE_ROOT}/run_wan_p8_exact_full_shard_sco.sh"
cp "${CANARY_REPORT}" "${PROVENANCE_ROOT}/canary_report.md"
cp "${CANARY_SUMMARY}" "${PROVENANCE_ROOT}/canary_summary.json"
cp "${CANARY_RUNTIME}" "${PROVENANCE_ROOT}/canary_runtime_status.json"
printf '%s\n' "${MODEL}" > "${PROVENANCE_ROOT}/wan_model_snapshot.txt"
"${PY}" scripts/prepare_wan_p8_exact_full.py \
  --manifest "${MANIFEST}" \
  --out-dir "${PROVENANCE_ROOT}/shards"
sha256sum \
  "${PROVENANCE_ROOT}/canonical_manifest.jsonl" \
  "${PROVENANCE_ROOT}/extract_wan_manifest_exact.py" \
  "${PROVENANCE_ROOT}/prepare_wan_p8_exact_full.py" \
  "${PROVENANCE_ROOT}/run_wan_p8_exact_full_shard_sco.sh" \
  "${PROVENANCE_ROOT}/canary_report.md" \
  "${PROVENANCE_ROOT}/canary_summary.json" \
  "${PROVENANCE_ROOT}/canary_runtime_status.json" \
  "${PROVENANCE_ROOT}/shards/sample_to_raw_id_mapping.jsonl" \
  "${PROVENANCE_ROOT}/shards/full_shard_inventory.json" \
  > "${PROVENANCE_ROOT}/sha256.txt"
printf 'FEATURE_ROOT_ABSENT_AT_SUBMIT=true\nFEATURE_ROOT=%s\n' "${FEATURE_ROOT}" \
  > "${REPORT_DIR}/launcher/initial_feature_root_state.txt"

submit_log=${REPORT_DIR}/launcher/sco_submit.log
job_ids_file=${REPORT_DIR}/launcher/sco_job_ids.tsv
printf 'shard\tjob_id\tjob_name\n' > "${job_ids_file}"
printf 'RUN_TAG=%s\nFEATURE_ROOT=%s\nREPORT_DIR=%s\nAEC2_NAME=%s\nWORKER_SPEC=%s\nCONTAINER_IMAGE=%s\nSTORAGE_MOUNT=%s\n' \
  "${RUN_TAG}" "${FEATURE_ROOT}" "${REPORT_DIR}" "${AEC2_NAME}" "${WORKER_SPEC}" "${CONTAINER_IMAGE}" "${STORAGE_MOUNT}" \
  > "${REPORT_DIR}/launcher/submission_plan.txt"

for shard in 0 1 2 3 4 5; do
  shard_name=$(printf 's%02d' "${shard}")
  job_name="bq_wanp8_exact_${shard_name}_$(date -u +%m%d%H%M%S)"
  command="cd ${PROJECT_ROOT} && PROJECT_ROOT=${PROJECT_ROOT} RUN_TAG=${RUN_TAG} FEATURE_ROOT=${FEATURE_ROOT} SHARD_ID=${shard} bash scripts/run_wan_p8_exact_full_shard_sco.sh"
  command_file=${REPORT_DIR}/launcher/submit_command_${shard_name}.txt
  printf '%q ' sco acp jobs create \
    --workspace-name=aigc \
    --aec2-name="${AEC2_NAME}" \
    --job-name="${job_name}" \
    --priority=HIGHEST \
    --container-image-url="${CONTAINER_IMAGE}" \
    --storage-mount="${STORAGE_MOUNT}" \
    --training-framework=pytorch \
    --worker-nodes=1 \
    --worker-spec="${WORKER_SPEC}" \
    --command="${command}" \
    > "${command_file}"
  printf '\n' >> "${command_file}"

  output=$(sco acp jobs create \
    --workspace-name=aigc \
    --aec2-name="${AEC2_NAME}" \
    --job-name="${job_name}" \
    --priority=HIGHEST \
    --container-image-url="${CONTAINER_IMAGE}" \
    --storage-mount="${STORAGE_MOUNT}" \
    --training-framework=pytorch \
    --worker-nodes=1 \
    --worker-spec="${WORKER_SPEC}" \
    --command="${command}")
  printf '%s\n' "${output}" | tee -a "${submit_log}"
  job_id=$(printf '%s\n' "${output}" | grep -oE 'pt-[a-z0-9]+' | head -n 1)
  [[ -n "${job_id}" ]] || { echo "SCO submission returned no job ID for shard ${shard}" >&2; exit 3; }
  printf '%d\t%s\t%s\n' "${shard}" "${job_id}" "${job_name}" | tee -a "${job_ids_file}"
done

printf 'RUN_TAG=%s\nFEATURE_ROOT=%s\nREPORT_DIR=%s\n' "${RUN_TAG}" "${FEATURE_ROOT}" "${REPORT_DIR}"
cat "${job_ids_file}"
