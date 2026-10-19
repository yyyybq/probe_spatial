#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
RUN_TAG=${RUN_TAG:?RUN_TAG is required}
EVAL_RUN_TAG=${EVAL_RUN_TAG:-${RUN_TAG}_eval_staging}
FEATURE_ROOT=${FEATURE_ROOT:-/mnt/umm/users/yinbaiqiao/InsScene-15K/FEATURES_VERSIONED/${RUN_TAG}}
PY=${PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}
REPORT_DIR=${PROJECT_ROOT}/reports/wan_p8_recovery/${RUN_TAG}
LOG=${REPORT_DIR}/logs/completion_controller.log
STATUS=${REPORT_DIR}/logs/completion_controller.status.json
TOOL_HASHES=${REPORT_DIR}/provenance/completion_controller_tooling.sha256
MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS:-43200}
POLL_SECONDS=${POLL_SECONDS:-120}
LAYERS=(0 5 10 15 25 29)

[[ ! -e "${STATUS}" ]] || { echo "refusing existing controller status: ${STATUS}" >&2; exit 2; }
exec > >(tee -a "${LOG}") 2>&1
started=$(date -u +%FT%TZ)
finish() {
  rc=$?
  trap - EXIT
  ended=$(date -u +%FT%TZ)
  printf '{"run_tag":"%s","eval_run_tag":"%s","started":"%s","ended":"%s","rc":%d}\n' \
    "${RUN_TAG}" "${EVAL_RUN_TAG}" "${started}" "${ended}" "${rc}" > "${STATUS}"
  exit "${rc}"
}
trap finish EXIT

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
verify_tooling() {
  [[ -f "${TOOL_HASHES}" ]] || { echo "missing frozen tooling hashes" >&2; exit 2; }
  sha256sum -c "${TOOL_HASHES}"
}
verify_tooling
echo "CONTROLLER START ${started} run_tag=${RUN_TAG} eval_run_tag=${EVAL_RUN_TAG}"
deadline=$(( $(date +%s) + MAX_WAIT_SECONDS ))
while true; do
  complete=0
  for layer in "${LAYERS[@]}"; do
    train_status="${REPORT_DIR}/logs/train_layer${layer}.status.json"
    if [[ -f "${train_status}" ]]; then
      rc=$("${PY}" -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["rc"]))' "${train_status}")
      [[ "${rc}" == "0" ]] || { echo "TRAIN STATUS FAILED layer=${layer} rc=${rc}" >&2; exit 2; }
      complete=$((complete + 1))
    fi
  done
  echo "TRAIN STATUS POLL $(date -u +%FT%TZ) complete=${complete}/6"
  [[ ${complete} -eq 6 ]] && break
  [[ $(date +%s) -lt ${deadline} ]] || { echo "training wait timeout" >&2; exit 2; }
  sleep "${POLL_SECONDS}"
done

verify_tooling
"${PY}" scripts/audit_wan_p8_clean_rerun.py train \
  --run-tag "${RUN_TAG}" \
  --report-dir "${REPORT_DIR}" \
  --out-json "${REPORT_DIR}/train_audit.json"
echo "ALL TRAINING CLEAN AUDIT PASS"

verify_tooling
for layer in "${LAYERS[@]}"; do
  PROJECT_ROOT="${PROJECT_ROOT}" RUN_TAG="${RUN_TAG}" EVAL_RUN_TAG="${EVAL_RUN_TAG}" \
    FEATURE_ROOT="${FEATURE_ROOT}" LAYER="${layer}" \
    bash scripts/run_wan_p8_clean_eval_sco.sh
done

"${PY}" scripts/audit_wan_p8_clean_rerun.py eval \
  --run-tag "${RUN_TAG}" \
  --eval-run-tag "${EVAL_RUN_TAG}" \
  --report-dir "${REPORT_DIR}" \
  --train-audit "${REPORT_DIR}/train_audit.json" \
  --out-json "${REPORT_DIR}/eval_audit.json"
echo "ALL STAGING EVAL AUDIT PASS"

verify_tooling
"${PY}" scripts/promote_wan_p8_clean_eval.py \
  --run-tag "${RUN_TAG}" \
  --eval-run-tag "${EVAL_RUN_TAG}" \
  --eval-audit "${REPORT_DIR}/eval_audit.json" \
  --report-dir "${REPORT_DIR}"
echo "REGISTRY CANONICAL PROMOTION PASS"

"${PY}" scripts/render_wan_p8_clean_final_report.py \
  --run-tag "${RUN_TAG}" \
  --eval-run-tag "${EVAL_RUN_TAG}" \
  --report-dir "${REPORT_DIR}" \
  --out "${PROJECT_ROOT}/reports/wan_p8_canonical_clean_rerun_${RUN_TAG}.md"
echo "CONTROLLER COMPLETE"
