#!/usr/bin/env bash
# Render canonical-v2 extraction shards and SCO create commands. Never submits.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}"
SPECS="${SPECS:?set SPECS=<cache-audited canonical-v2 spec CSV/JSON>}"
RUN_TAG="${RUN_TAG:-windowfix_v2_$(date -u +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/artifacts/windowfix_v2/sco_plans/${RUN_TAG}}"
PROTOCOL="${PROTOCOL:-${PROJECT_ROOT}/protocols/inscene_streaming_canonical_v2.json}"
CANONICAL_SOURCE="${CANONICAL_SOURCE:?set CANONICAL_SOURCE=<frozen prefix24/obs28 JSONL>}"
NAMESPACE_LOCK="${NAMESPACE_LOCK:-${PROJECT_ROOT}/artifacts/windowfix_v2/canonical_namespace_lock.json}"
LABEL_AUDIT="${LABEL_AUDIT:-${PROJECT_ROOT}/artifacts/windowfix_v2/canonical_v2_compact_source.audit.json}"
BACKEND_CANARY_DIR="${BACKEND_CANARY_DIR:-}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/umm/users/yinbaiqiao/InsScene-15K}"
ALLOW_PENDING="${ALLOW_PENDING:-0}"

[[ -x "${PYTHON}" ]] || { echo "ERROR: Python is not executable: ${PYTHON}" >&2; exit 1; }
[[ -f "${SPECS}" && ! -L "${SPECS}" ]] || {
  echo "ERROR: SPECS must be a regular non-symlink file: ${SPECS}" >&2
  exit 1
}
case "${ALLOW_PENDING}" in 0|1) ;; *) echo "ERROR: ALLOW_PENDING must be 0 or 1" >&2; exit 2 ;; esac

args=(
  "${PYTHON}" "${SCRIPT_DIR}/prepare_sco_feature_jobs.py"
  --specs "${SPECS}"
  --protocol "${PROTOCOL}"
  --canonical-source "${CANONICAL_SOURCE}"
  --namespace-lock "${NAMESPACE_LOCK}"
  --label-audit "${LABEL_AUDIT}"
  --output-dir "${OUTPUT_DIR}"
  --dataset-root "${DATASET_ROOT}"
  --project-root "${PROJECT_ROOT}"
)
if [[ "${ALLOW_PENDING}" == "1" ]]; then
  args+=(--allow-pending)
fi
if [[ -n "${BACKEND_CANARY_DIR}" ]]; then
  args+=(--backend-canary-dir "${BACKEND_CANARY_DIR}")
fi

"${args[@]}"
echo "Rendered only; no SCO command was run: ${OUTPUT_DIR}"
echo "Review: ${OUTPUT_DIR}/summary.json"
echo "Review: ${OUTPUT_DIR}/sco_create_commands.txt"
