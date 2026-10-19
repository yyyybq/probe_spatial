#!/usr/bin/env bash
set -euo pipefail

export PATH="${HOME}/.sco/bin:${PATH}"
PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
RUN_TAG=${RUN_TAG:?RUN_TAG is required}
FEATURE_ROOT=${FEATURE_ROOT:?FEATURE_ROOT is required}
PY=${PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}
REPORT_DIR=${PROJECT_ROOT}/reports/wan_p8_recovery/${RUN_TAG}
TRAIN_PROVENANCE=${REPORT_DIR}/provenance/training
AEC2_NAME=${AEC2_NAME:-zoetrope}
CONTAINER_IMAGE=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617
STORAGE_MOUNT=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm
WORKER_SPEC=N4lS.Iq.I80.1
LAYERS=(0 5 10 15 25 29)

cd "${PROJECT_ROOT}"
for path in scripts/prepare_wan_p8_clean_training.py scripts/run_wan_p8_clean_train_sco.sh "${REPORT_DIR}/full_gate_summary.json" "${REPORT_DIR}/logs/full_gate.status.json"; do
  [[ -f "${path}" ]] || { echo "missing ${path}" >&2; exit 2; }
done
grep -q '"rc":0' "${REPORT_DIR}/logs/full_gate.status.json" || { echo "full gate rc is not zero" >&2; exit 2; }
"${PY}" - "${REPORT_DIR}/full_gate_summary.json" "${FEATURE_ROOT}" <<'PY'
import json, sys
from pathlib import Path
x=json.load(open(sys.argv[1])); root=Path(sys.argv[2]).resolve()
assert x.get('ok') is True
assert x.get('canonical_manifest_sha256') == '772194f077c45cc85a942f17f46f50103125ae8f5b1c983c08881c1759607e0c'
assert x.get('streaming_exact_raw') == 2724 and x.get('target_exact_raw') == 2724
assert x.get('finite_files_checked') == 20430
assert Path(x['feature_root']).resolve() == root
assert all(v.get('ok') is True and v.get('checked') == 2724 and not v.get('problems') for v in x.get('dataset_layers',{}).values())
print('FULL TRAIN SUBMISSION GATE PASS')
PY

for layer in "${LAYERS[@]}"; do
  name=${RUN_TAG}_canonical_clean_action_dynamics_wan_p8_layer${layer}
  run_dir=${PROJECT_ROOT}/logs/inscene15k_streaming/runs/${RUN_TAG}/${name}
  [[ ! -e "${run_dir}" ]] || { echo "refusing existing run dir ${run_dir}" >&2; exit 2; }
  [[ ! -e "${REPORT_DIR}/logs/train_layer${layer}.status.json" ]] || { echo "refusing existing layer status ${layer}" >&2; exit 2; }
done

mkdir -p "${TRAIN_PROVENANCE}" "${REPORT_DIR}/launcher"
"${PY}" scripts/prepare_wan_p8_clean_training.py \
  --repo-root "${PROJECT_ROOT}" \
  --out-json "${TRAIN_PROVENANCE}/historical_training_contract.json"
git -C "${PROJECT_ROOT}" rev-parse HEAD > "${TRAIN_PROVENANCE}/git_commit.txt"
git -C "${PROJECT_ROOT}" status --short > "${TRAIN_PROVENANCE}/git_status_short.txt"
git -C "${PROJECT_ROOT}" diff --binary HEAD > "${TRAIN_PROVENANCE}/git_dirty.diff"
cp scripts/prepare_wan_p8_clean_training.py "${TRAIN_PROVENANCE}/prepare_wan_p8_clean_training.py"
cp scripts/run_wan_p8_clean_train_sco.sh "${TRAIN_PROVENANCE}/run_wan_p8_clean_train_sco.sh"
sha256sum "${TRAIN_PROVENANCE}/historical_training_contract.json" "${TRAIN_PROVENANCE}/prepare_wan_p8_clean_training.py" "${TRAIN_PROVENANCE}/run_wan_p8_clean_train_sco.sh" \
  > "${TRAIN_PROVENANCE}/sha256.txt"

job_ids=${REPORT_DIR}/launcher/train_sco_job_ids.tsv
submit_log=${REPORT_DIR}/launcher/train_sco_submit.log
printf 'layer\tjob_id\tjob_name\n' > "${job_ids}"
for layer in "${LAYERS[@]}"; do
  job_name="bq_wanp8_train_l${layer}_$(date -u +%m%d%H%M%S)"
  command="cd ${PROJECT_ROOT} && PROJECT_ROOT=${PROJECT_ROOT} RUN_TAG=${RUN_TAG} FEATURE_ROOT=${FEATURE_ROOT} LAYER=${layer} bash scripts/run_wan_p8_clean_train_sco.sh"
  printf '%q ' sco acp jobs create \
    --workspace-name=aigc --aec2-name="${AEC2_NAME}" --job-name="${job_name}" --priority=HIGHEST \
    --container-image-url="${CONTAINER_IMAGE}" --storage-mount="${STORAGE_MOUNT}" \
    --training-framework=pytorch --worker-nodes=1 --worker-spec="${WORKER_SPEC}" --command="${command}" \
    > "${REPORT_DIR}/launcher/train_submit_command_layer${layer}.txt"
  printf '\n' >> "${REPORT_DIR}/launcher/train_submit_command_layer${layer}.txt"
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
  [[ -n "${job_id}" ]] || { echo "missing job id for layer ${layer}" >&2; exit 3; }
  printf '%d\t%s\t%s\n' "${layer}" "${job_id}" "${job_name}" | tee -a "${job_ids}"
done
cat "${job_ids}"
