#!/usr/bin/env bash
set -euo pipefail

export PATH="${HOME}/.sco/bin:${PATH}"
PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
STAGE=${STAGE:?STAGE must be d1 or d2d3}
RUN_TAG=${RUN_TAG:-dseries_training_${STAGE}_$(date -u +%Y%m%d_%H%M%S)}
FEATURE_ROOT=${FEATURE_ROOT:-/mnt/umm/users/yinbaiqiao/InsScene-15K/D_SERIES_SCANNETPP_V1_FEATURES}
FORMAL_RUN_ROOT=${FORMAL_RUN_ROOT:-${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/formal_runs}
FEATURE_RUN=${FEATURE_RUN:-}
REPORT=${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/runs/${RUN_TAG}/launcher
cd "${PROJECT_ROOT}"

case "${STAGE}" in
  d1)
    [[ -n "${FEATURE_RUN}" ]] || { echo "FEATURE_RUN must identify the completed full-feature run" >&2; exit 2; }
    for model in vjepa2_vitl256 dinov2_vitl14 wan21_t2v_1p3b; do
      status="experiments/d_series_scannetpp_v1/runs/${FEATURE_RUN}/${model}/status.json"
      test -f "${status}"
      /mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python - "${status}" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
assert value["failed"] == 0, value
PY
    done
    ;;
  d2d3) ;;
  *) echo "STAGE must be d1 or d2d3" >&2; exit 2 ;;
esac

mkdir -p "${REPORT}"
PROJECT_ROOT="${PROJECT_ROOT}" PROVENANCE_DIR="${REPORT}" \
  bash scripts/d_series/capture_source_provenance.sh
base_commit=$(cat "${REPORT}/base_commit.txt")
git_dirty=$(cat "${REPORT}/git_dirty.txt")
job_name="bq_dseries_train_${STAGE}_$(date -u +%m%d%H%M%S)"
command="cd ${PROJECT_ROOT} && PROJECT_ROOT=${PROJECT_ROOT} STAGE=${STAGE} RUN_TAG=${RUN_TAG} FEATURE_ROOT=${FEATURE_ROOT} FORMAL_RUN_ROOT=${FORMAL_RUN_ROOT} D_SERIES_BASE_COMMIT=${base_commit} D_SERIES_GIT_DIRTY=${git_dirty} bash scripts/d_series/run_training_stage_sco.sh"
printf '%s\n' "${command}" > "${REPORT}/command.txt"
printf '%s\n' "${job_name}" > "${REPORT}/job_name.txt"
sco acp jobs create --workspace-name=aigc --aec2-name=zoetrope --job-name="${job_name}" --priority=HIGHEST \
  --container-image-url=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617 \
  --storage-mount=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm --training-framework=pytorch \
  --worker-nodes=1 --worker-spec=N4lS.Iq.I80.8 --command="${command}" | tee "${REPORT}/sco_submit.log"
grep -oE 'pt-[a-z0-9]+' "${REPORT}/sco_submit.log" | awk '!seen[$0]++' > "${REPORT}/job_ids.txt"
test -s "${REPORT}/job_ids.txt"
printf 'RUN_TAG=%s\nJOB_ID=%s\n' "${RUN_TAG}" "$(head -1 "${REPORT}/job_ids.txt")"
