#!/usr/bin/env bash
# Drain validated seeded-v1 streaming caches scene by scene.  Each worker owns
# a disjoint contiguous range of the sorted scene universe, so no process ever
# scans the full feature tree.  Incomplete/temporary scene state is retained.
set -u -o pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/umm/users/yinbaiqiao/InsScene-15K}"
VFM="${VFM:?set VFM=wan|cogvideox}"
T="${T:-499}"
SCENE_START="${SCENE_START:-0}"
SCENE_END="${SCENE_END:-}"
SHARD_WORKERS="${SHARD_WORKERS:-8}"
RUN_ID="${RUN_ID:-seeded_stream_sync_drain_$(date -u +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/${RUN_ID}}"
FEATURE_NAMESPACE=FEAT_STREAMING_CANONICAL_V2_SEEDED_V1
FEATURE_ROOT="${DATASET_ROOT}/${FEATURE_NAMESPACE}/${VFM}/scannetpp"
DONE_INDEX="${PROJECT_ROOT}/logs/tstep_canonical_v2_seeded_v1/aoss_done/${FEATURE_NAMESPACE}_${VFM}_t${T}.txt"
SYNC_LOOP="${PROJECT_ROOT}/scripts/run_tstep_sync_loop.sh"

case "${VFM}:${T}" in
  wan:499|wan:999|cogvideox:499|cogvideox:999) ;;
  *) echo "ERROR: this drain only accepts wan/cogvideox with T=499 or T=999" >&2; exit 2 ;;
esac
for value in "${SCENE_START}" "${SHARD_WORKERS}"; do
  [[ "${value}" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid integer ${value}" >&2; exit 2; }
done
(( SHARD_WORKERS > 0 )) || { echo "ERROR: SHARD_WORKERS must be positive" >&2; exit 2; }
[[ -d "${FEATURE_ROOT}" ]] || { echo "ERROR: missing feature root ${FEATURE_ROOT}" >&2; exit 1; }
[[ -s "${SYNC_LOOP}" && -x "${SYNC_LOOP}" && ! -L "${SYNC_LOOP}" ]] || {
  echo "ERROR: invalid sync loop ${SYNC_LOOP}" >&2; exit 1;
}
[[ -f "${DONE_INDEX}" && ! -L "${DONE_INDEX}" ]] || {
  echo "ERROR: invalid done-index ${DONE_INDEX}" >&2; exit 1;
}
chmod a+r -- "${DONE_INDEX}" 2>/dev/null || true
[[ -r "${DONE_INDEX}" ]] || {
  echo "ERROR: done-index is unreadable: ${DONE_INDEX}" >&2
  exit 1
}
if [[ -z "${SYNC_PYTHON:-}" ]]; then
  if [[ -x /mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python ]]; then
    SYNC_PYTHON=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python
  else
    SYNC_PYTHON=python3
  fi
fi
mkdir -p "${LOG_ROOT}/workers" "${LOG_ROOT}/inventory"

mapfile -d '' SCENES < <(
  find "${FEATURE_ROOT}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z
)
N_SCENES="${#SCENES[@]}"
SCENE_END="${SCENE_END:-${N_SCENES}}"
[[ "${SCENE_END}" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid SCENE_END" >&2; exit 2; }
if (( SCENE_START < 0 || SCENE_END <= SCENE_START || SCENE_END > N_SCENES )); then
  echo "ERROR: invalid scene span [${SCENE_START}:${SCENE_END}) for ${N_SCENES} roots" >&2
  exit 2
fi

SYNC_SHA256="$(sha256sum "${SYNC_LOOP}" | awk '{print $1}')"
{
  echo "run_id=${RUN_ID}"
  echo "vfm=${VFM}"
  echo "t=${T}"
  echo "scene_span=${SCENE_START}:${SCENE_END}"
  echo "scene_roots=${N_SCENES}"
  echo "shard_workers=${SHARD_WORKERS}"
  echo "feature_root=${FEATURE_ROOT}"
  echo "done_index=${DONE_INDEX}"
  echo "done_index_start=$(wc -l < "${DONE_INDEX}" 2>/dev/null || echo unreadable)"
  echo "sync_python=${SYNC_PYTHON}"
  echo "sync_sha256=${SYNC_SHA256}"
} | tee "${LOG_ROOT}/PLAN.txt"

scene_has_publishable_state() {
  local scene_root="$1"
  [[ -f "${scene_root}/prefix_index.npy" && \
     -f "${scene_root}/prefix_index.npy.manifest.json" ]] || return 1
  find "${scene_root}" -type f -name "feature_t${T}_layer*.sft" -print -quit | grep -q .
}

scene_has_incomplete_state() {
  local scene_root="$1"
  /mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python - \
    "${scene_root}" "${T}" <<'PY'
import re
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
timestep = sys.argv[2]
pat = re.compile(rf"feature_t{re.escape(timestep)}_layer(15|20|25)\.sft$")
groups = defaultdict(set)
bad = False
for path in root.rglob("*"):
    if not path.is_file():
        continue
    name = path.name
    lower = name.lower()
    if "tmp" in lower and (f"feature_t{timestep}_" in lower or "prefix_index" in lower):
        bad = True
    match = pat.fullmatch(name)
    if not match:
        continue
    groups[path.parent].add(int(match.group(1)))
    if not Path(str(path) + ".manifest.json").is_file():
        bad = True
if any(layers != {15, 20, 25} for layers in groups.values()):
    bad = True
raise SystemExit(0 if bad else 1)
PY
}

drain_worker() {
  local worker="$1" start="$2" end="$3"
  local worker_root="${LOG_ROOT}/workers/$(printf '%02d' "${worker}")_${start}_${end}"
  mkdir -p "${worker_root}"
  local selected=0 cleared=0 retained=0 failed=0 before after rc scene_root scene scene_log
  for ((idx=start; idx<end; idx++)); do
    scene_root="${SCENES[idx]}"
    scene="$(basename "${scene_root}")"
    scene_has_publishable_state "${scene_root}" || continue
    selected=$((selected + 1))
    if scene_has_incomplete_state "${scene_root}"; then
      echo "${scene}" >> "${worker_root}/retained_incomplete.txt"
      retained=$((retained + 1))
      continue
    fi
    scene_log="${worker_root}/${scene}"
    mkdir -p "${scene_log}"
    touch "${scene_log}/extract.finished"
    before="$(find "${scene_root}" -type f -name "feature_t${T}_layer*.sft" | wc -l)"
    FEATURE_KIND=FEAT_STREAMING \
    FEATURE_NAMESPACE="${FEATURE_NAMESPACE}" \
    VFM="${VFM}" \
    T="${T}" \
    EXTRACT_DONE_INDEX="${DONE_INDEX}" \
    LOG_DIR="${scene_log}" \
    STOP_FILE="${scene_log}/extract.finished" \
    DATASET_ROOT="${DATASET_ROOT}" \
    FEATURE_ROOT="${FEATURE_ROOT}" \
    FEATURE_SCAN_ROOT="${scene_root}" \
    SYNC_PYTHON="${SYNC_PYTHON}" \
    SYNC_DISJOINT_SCAN=1 \
    SYNC_TRUST_EXTRACT="${SYNC_TRUST_EXTRACT:-1}" \
    SYNC_EVERY_SEC=0 \
    SYNC_EVERY_GIB=0 \
    SYNC_POLL_SEC=2 \
    SYNC_RETRY_SEC=2 \
    SYNC_FINAL_MAX_ATTEMPTS=1 \
      bash "${SYNC_LOOP}" >"${scene_log}/sync_loop.log" 2>&1
    rc=$?
    after="$(find "${scene_root}" -type f -name "feature_t${T}_layer*.sft" | wc -l)"
    if (( rc == 0 && after == 0 )); then
      cleared=$((cleared + 1))
    else
      printf '%s\trc=%s\tbefore=%s\tafter=%s\n' \
        "${scene}" "${rc}" "${before}" "${after}" >> "${worker_root}/failed_or_retained.tsv"
      failed=$((failed + 1))
    fi
  done
  printf 'worker=%s\trange=%s:%s\tselected=%s\tcleared=%s\tretained=%s\tfailed=%s\n' \
    "${worker}" "${start}" "${end}" "${selected}" "${cleared}" "${retained}" "${failed}" \
    | tee "${worker_root}/SUMMARY.txt"
  (( failed == 0 ))
}

SPAN=$((SCENE_END - SCENE_START))
pids=()
for ((worker=0; worker<SHARD_WORKERS; worker++)); do
  start=$((SCENE_START + SPAN * worker / SHARD_WORKERS))
  end=$((SCENE_START + SPAN * (worker + 1) / SHARD_WORKERS))
  drain_worker "${worker}" "${start}" "${end}" &
  pids+=("$!")
done

failures=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failures=$((failures + 1))
done
{
  echo "worker_failures=${failures}"
  echo "done_index_end=$(wc -l < "${DONE_INDEX}" 2>/dev/null || echo unreadable)"
  find "${LOG_ROOT}/workers" -name SUMMARY.txt -type f -exec cat {} + | sort
} | tee "${LOG_ROOT}/SUMMARY.txt"
(( failures == 0 ))
