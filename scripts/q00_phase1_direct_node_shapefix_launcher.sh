#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/umm/users/yinbaiqiao/probe_spatial
BUNDLE=${ROOT}/reports/q00_phase1_runs/q00_phase1_wave_b_e_direct_shapecontract_20260830T040218Z_2075f3eb
RUN_TAG=q00_phase1_wave_b_e_direct_shapecontract_20260830T040218Z_2075f3eb
PY=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python
EXPECTED_HOST=pt-314e0f9747cc44dc9e91dc07705c52a4-worker-0
EXPECTED_BUNDLE_SHA=ed5ee041e18efd45ede9c40ff79e488fcf3c1a9ec2bc7a2fa3ff918a6df0cd5c
DIRECT_ROOT=${ROOT}/reports/q00_phase1_direct_node/${RUN_TAG}
LOCK_ROOT=${DIRECT_ROOT}/locks
LOG_ROOT=${DIRECT_ROOT}/logs
CACHE_ROOT=/mnt/local_cache/inscene/aoss_cache/${RUN_TAG}/shared
ATTEMPT_TAG=${DIRECT_ATTEMPT:-direct1}

die() {
  echo "DIRECT_NODE_FAIL: $*" >&2
  exit 2
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

check_base() {
  [[ "$(hostname)" == "${EXPECTED_HOST}" ]] || die "unexpected host $(hostname)"
  [[ -x "${PY}" ]] || die "missing frozen Python ${PY}"
  [[ -f "${BUNDLE}/bundle.json" ]] || die "missing bundle.json"
  [[ "$(sha256_file "${BUNDLE}/bundle.json")" == "${EXPECTED_BUNDLE_SHA}" ]] || die "bundle SHA mismatch"
  [[ -f "${ROOT}/scripts/q00_phase1_run_eval_shard.py" ]] || die "missing runner"
  command -v nvidia-smi >/dev/null || die "nvidia-smi unavailable"
  command -v flock >/dev/null || die "flock unavailable"
  mkdir -p "${LOCK_ROOT}" "${LOG_ROOT}" "${CACHE_ROOT}"
  PYTHONPATH="${ROOT}" "${PY}" -B -c 'import vidfm3d' \
    || die "repo PYTHONPATH cannot import vidfm3d"
}

check_gpu() {
  local gpu=$1
  [[ "${gpu}" =~ ^[0-9]+$ ]] || die "invalid GPU id ${gpu}"
  local gpu_row uuid used free
  gpu_row=$(nvidia-smi --query-gpu=index,uuid,memory.used,memory.free \
    --format=csv,noheader,nounits \
    | awk -F, -v target="${gpu}" '$1 + 0 == target {print}')
  [[ -n "${gpu_row}" ]] || die "GPU ${gpu} not found"
  IFS=',' read -r _ uuid used free <<<"${gpu_row}"
  uuid=${uuid// /}
  used=${used// /}
  free=${free// /}
  (( free >= 24000 )) || die "GPU ${gpu} has insufficient free memory: free_memory_mib=${free}"
  local process_uuid pid stat
  local -a active_holders=()
  while IFS=',' read -r process_uuid pid; do
    process_uuid=${process_uuid// /}
    pid=${pid// /}
    [[ "${process_uuid}" == "${uuid}" && -n "${pid}" ]] || continue
    stat=$(ps -o stat= -p "${pid}" 2>/dev/null | tr -d ' ' || true)
    [[ "${stat}" == Z* ]] || active_holders+=("${pid}:${stat:-missing}")
  done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits)
  (( ${#active_holders[@]} == 0 )) \
    || die "GPU ${gpu} has active compute holders: ${active_holders[*]}"
  echo "GPU_GATE gpu=${gpu} used_mib=${used} free_mib=${free} active_holders=0 zombie_holders_allowed=true"
  nvidia-smi --query-gpu=index,name,memory.total,memory.free,uuid \
    --format=csv,noheader | awk -F, -v target="${gpu}" '$1 + 0 == target {print}'
}

run_shard() {
  local label=$1
  local shard=$2
  local gpu_csv=$3
  local slots=$4
  local log=${LOG_ROOT}/${label}_${ATTEMPT_TAG}.log
  local lock=${LOCK_ROOT}/${label}_${ATTEMPT_TAG}.lock
  [[ -f "${shard}" ]] || die "missing shard ${shard}"
  [[ ! -e "${log}" ]] || die "refusing existing direct-node log ${log}"
  IFS=',' read -r -a gpu_ids <<<"${gpu_csv}"
  [[ "${#gpu_ids[@]}" -eq "${slots}" ]] || die "slots/GPU count mismatch"
  local gpu
  for gpu in "${gpu_ids[@]}"; do
    check_gpu "${gpu}"
  done
  {
    flock -n 9 || die "duplicate direct-node writer for ${label}"
    echo "DIRECT_NODE_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "HOST=$(hostname)"
    echo "BUNDLE_SHA=$(sha256_file "${BUNDLE}/bundle.json")"
    echo "SHARD=${shard}"
    echo "SHARD_SHA=$(sha256_file "${shard}")"
    echo "GPU_IDS_CSV=${gpu_csv}"
    echo "SLOTS=${slots}"
    printf 'CLI='
    printf '%q ' env \
      RUN_TAG="${RUN_TAG}" BUNDLE="${BUNDLE}" SHARD_TSV="${shard}" \
      SLOTS="${slots}" GPU_IDS_CSV="${gpu_csv}" PYTHON_BIN="${PY}" \
      PROJECT_ROOT="${ROOT}" PYTHONPATH="${ROOT}" \
      "${PY}" -B scripts/q00_phase1_run_eval_shard.py
    printf '\n'
    cd "${ROOT}"
    set +e
    env \
      RUN_TAG="${RUN_TAG}" \
      BUNDLE="${BUNDLE}" \
      SHARD_TSV="${shard}" \
      SLOTS="${slots}" \
      GPU_IDS_CSV="${gpu_csv}" \
      PYTHON_BIN="${PY}" \
      PROJECT_ROOT="${ROOT}" \
      PYTHONPATH="${ROOT}" \
      "${PY}" -B scripts/q00_phase1_run_eval_shard.py
    rc=$?
    set -e
    echo "DIRECT_NODE_END=$(date -u +%Y-%m-%dT%H:%M:%SZ) rc=${rc}"
    exit "${rc}"
  } 9>"${lock}" >"${log}" 2>&1
}

run_python_gate() {
  local label=$1
  shift
  local log=${LOG_ROOT}/${label}_${ATTEMPT_TAG}.log
  local lock=${LOCK_ROOT}/${label}_${ATTEMPT_TAG}.lock
  [[ ! -e "${log}" ]] || die "refusing existing gate log ${log}"
  {
    flock -n 9 || die "duplicate gate writer for ${label}"
    echo "DIRECT_NODE_GATE_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'CLI='
    printf '%q ' env PYTHONPATH="${ROOT}" "${PY}" -B "$@"
    printf '\n'
    cd "${ROOT}"
    set +e
    env PYTHONPATH="${ROOT}" "${PY}" -B "$@"
    rc=$?
    set -e
    echo "DIRECT_NODE_GATE_END=$(date -u +%Y-%m-%dT%H:%M:%SZ) rc=${rc}"
    exit "${rc}"
  } 9>"${lock}" >"${log}" 2>&1
}

main() {
  check_base
  local action=${1:-}
  case "${action}" in
    canaries)
      set +e
      "${BASH_SOURCE[0]}" canary-lingbot 4 &
      local p_lingbot=$!
      "${BASH_SOURCE[0]}" canary-minimax 5 &
      local p_minimax=$!
      "${BASH_SOURCE[0]}" canary-wan 6 &
      local p_wan=$!
      wait "${p_lingbot}"; local rc_lingbot=$?
      wait "${p_minimax}"; local rc_minimax=$?
      wait "${p_wan}"; local rc_wan=$?
      set -e
      echo "CANARY_RESULTS lingbot=${rc_lingbot} minimax_h3=${rc_minimax} wan=${rc_wan}"
      (( rc_lingbot == 0 && rc_minimax == 0 && rc_wan == 0 )) || exit 2
      ;;
    canary-lingbot)
      run_shard canary_lingbot "${BUNDLE}/inputs/canary_shards/canary_lingbot.tsv" "${2:?GPU required}" 1
      ;;
    canary-minimax)
      run_shard canary_minimax_h3 "${BUNDLE}/inputs/canary_shards/canary_minimax_h3.tsv" "${2:?GPU required}" 1
      ;;
    canary-wan)
      run_shard canary_wan "${BUNDLE}/inputs/canary_shards/canary_wan.tsv" "${2:?GPU required}" 1
      ;;
    remaining)
      local gpu_csv=${2:?comma-separated GPUs required}
      IFS=',' read -r -a gpu_ids <<<"${gpu_csv}"
      run_shard wave_b_remaining_96 "${BUNDLE}/inputs/wave_b_remaining_96.tsv" "${gpu_csv}" "${#gpu_ids[@]}"
      ;;
    audit-canary)
      run_python_gate audit_canary scripts/q00_phase1_audit_eval.py --bundle "${BUNDLE}" --scope canary
      ;;
    audit-full)
      run_python_gate audit_full scripts/q00_phase1_audit_eval.py --bundle "${BUNDLE}" --scope full
      ;;
    promote)
      run_python_gate promote scripts/q00_phase1_promote.py --bundle "${BUNDLE}"
      ;;
    finalize)
      run_python_gate finalize scripts/q00_phase1_finalize.py --bundle "${BUNDLE}"
      ;;
    *)
      die "usage: $0 {canaries|canary-lingbot GPU|canary-minimax GPU|canary-wan GPU|remaining GPU_CSV|audit-canary|audit-full|promote|finalize}"
      ;;
  esac
}

main "$@"
