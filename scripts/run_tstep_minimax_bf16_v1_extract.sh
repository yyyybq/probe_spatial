#!/usr/bin/env bash
# Run one (timestep, mode) step of the MiniMax-H3 bf16 seeded-v1 re-extraction
# from the frozen runtime (runtime/tstep_minimax_bf16_v1_extractor_r2; r1 is the
# superseded freeze used by canary attempts 1-2).
#
# Stage 1/2/3 of docs/MINIMAX_REEXTRACT_RETRAIN_PLAN_20261016.md.  The frozen
# copy of scripts/run_streaming_feat_extract_8gpu.sh does the actual work (GPU
# workers + AOSS sync loop); this wrapper only pins the contract:
#   namespace   FEAT_{STREAMING,TARGET}_CANONICAL_V2_SEEDED_V1_MINIMAX_BF16_V1
#   protocol    inscene_streaming_canonical_v2_obs28_h124 (production SHA256)
#   layers      20 25          (the fp16 seeded-v1 layers that overflowed)
#   storage     MINIMAX_H3_FINITE_BF16_V1 (enforced inside the frozen extractor)
#   gates       H800 only, frozen model path/prompt, done-index naming,
#               sync loop in validating mode (SYNC_TRUST_EXTRACT=0)
# and refuses to run unless the frozen tree hashes to EXPECTED_SOURCE_TREE_SHA256.
#
# Required env:  EXPECTED_SOURCE_TREE_SHA256  T=500|1000  MODE=stream|target  RUN_TAG
# Optional env:  N_GPUS (8)  SCENE_START  SCENE_END  SYNC_TRUST_EXTRACT (0)  DRY_RUN
set -euo pipefail

ROOT=/mnt/umm/users/yinbaiqiao/probe_spatial
RELEASE=${ROOT}/runtime/tstep_minimax_bf16_v1_extractor_r2
SOURCE=${RELEASE}/source
EXPECTED_SOURCE_TREE_SHA256=${EXPECTED_SOURCE_TREE_SHA256:?EXPECTED_SOURCE_TREE_SHA256 is required}
T=${T:?T is required (500|1000)}
MODE=${MODE:?MODE is required (stream|target)}
RUN_TAG=${RUN_TAG:?RUN_TAG is required (e.g. canary_20261016)}
N_GPUS=${N_GPUS:-8}

case "${T}" in
  500|1000) ;;
  *) echo "ERROR: T must be 500 or 1000 (seeded-v1 MiniMax timesteps), got ${T}" >&2; exit 2 ;;
esac
case "${MODE}" in
  stream)
    EXTRACT_STREAM=1; EXTRACT_TARGET=0
    NAMESPACE=FEAT_STREAMING_CANONICAL_V2_SEEDED_V1_MINIMAX_BF16_V1
    ;;
  target)
    EXTRACT_STREAM=0; EXTRACT_TARGET=1
    NAMESPACE=FEAT_TARGET_CANONICAL_V2_SEEDED_V1_MINIMAX_BF16_V1
    ;;
  *) echo "ERROR: MODE must be stream or target, got ${MODE}" >&2; exit 2 ;;
esac
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "ERROR: unsafe RUN_TAG" >&2; exit 2; }

manifest="${RELEASE}/source_manifest.json"
[[ -f "${manifest}" && -f "${RELEASE}/source.sha256" ]] || {
  echo "ERROR: frozen runtime is incomplete: ${RELEASE}" >&2
  exit 3
}
actual_tree_sha256=$(SOURCE="${SOURCE}" python3 - <<'PY'
import os
from hashlib import sha256
from pathlib import Path
root = Path(os.environ["SOURCE"])
files = sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
d = sha256()
for p in files:
    rel = p.relative_to(root).as_posix().encode()
    d.update(len(rel).to_bytes(8, "big")); d.update(rel); d.update(sha256(p.read_bytes()).digest())
print(d.hexdigest())
PY
)
[[ "${actual_tree_sha256}" == "${EXPECTED_SOURCE_TREE_SHA256}" ]] || {
  echo "ERROR: frozen source tree mismatch: ${actual_tree_sha256} != ${EXPECTED_SOURCE_TREE_SHA256}" >&2
  exit 3
}
(cd "${SOURCE}" && sha256sum --status -c "${RELEASE}/source.sha256") || {
  echo "ERROR: frozen source.sha256 verification failed" >&2
  exit 3
}
recorded_tree_sha256=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['source_tree_sha256'])" "${manifest}")
[[ "${recorded_tree_sha256}" == "${EXPECTED_SOURCE_TREE_SHA256}" ]] || {
  echo "ERROR: source_manifest.json records a different tree digest" >&2
  exit 3
}

LOG_DIR="${ROOT}/logs/tstep_minimax_bf16_v1/${RUN_TAG}/minimax_h3_t${T}_${MODE}"
EXTRACT_DONE_INDEX="${ROOT}/logs/tstep_canonical_v2_seeded_v1/aoss_done/${NAMESPACE}_minimax_h3_t${T}.txt"
STEP_OWNER_LOCK="${ROOT}/logs/tstep_canonical_v2_seeded_v1/step_locks/${NAMESPACE}_minimax_h3_t${T}.lock"
[[ -f "${EXTRACT_DONE_INDEX}" && ! -L "${EXTRACT_DONE_INDEX}" ]] || {
  echo "ERROR: done-index must pre-exist (create it empty on purpose): ${EXTRACT_DONE_INDEX}" >&2
  exit 3
}
mkdir -p "${LOG_DIR}"
# The frozen sync loop starts with `cd FEATURE_ROOT` (OUT_ROOT/<vfm>/<source>);
# a brand-new namespace has no such directory yet, and the loop then dies before
# any GPU worker starts (canary attempt 1, job pt-6g2hggcx, 2026-10-16).
mkdir -p "/mnt/umm/users/yinbaiqiao/InsScene-15K/${NAMESPACE}/minimax_h3/scannetpp"
{
  echo "frozen_runtime=${RELEASE}"
  echo "source_tree_sha256=${actual_tree_sha256}"
  echo "namespace=${NAMESPACE} t=${T} mode=${MODE} layers=20 25 n_gpus=${N_GPUS}"
  echo "scene_span=${SCENE_START:-0}:${SCENE_END:-<all>}"
  echo "done_index=${EXTRACT_DONE_INDEX}"
  echo "step_owner_lock=${STEP_OWNER_LOCK}"
  echo "sync_trust_extract=${SYNC_TRUST_EXTRACT:-0}"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname)"
} | tee "${LOG_DIR}/wrapper.txt"

# Keep the frozen tree read-only in practice: the upload client writes its
# record into LOG_DIR (scripts/ads_cli_in_logdir.sh) and Python writes no .pyc.
[[ -x "${ROOT}/scripts/ads_cli_in_logdir.sh" ]] || { echo "ERROR: ads_cli_in_logdir.sh is missing" >&2; exit 3; }
export PROJECT_ROOT="${SOURCE}"
# Jobs run as root in the container (no ~/.petreloss.conf).
export AOSS_CONF="${AOSS_CONF:-/mnt/umm/users/yinbaiqiao/aoss/petreloss.conf}"
[[ -r "${AOSS_CONF}" ]] || { echo "ERROR: AOSS_CONF is not readable: ${AOSS_CONF}" >&2; exit 3; }
exec env \
  VFM=minimax_h3 \
  T="${T}" \
  LAYERS="20 25" \
  EXTRACT_STREAM="${EXTRACT_STREAM}" \
  EXTRACT_TARGET="${EXTRACT_TARGET}" \
  FEATURE_NAMESPACE="${NAMESPACE}" \
  CANONICAL_PROTOCOL=inscene_streaming_canonical_v2_obs28_h124 \
  CANONICAL_PROTOCOL_SHA256=215c720c8537f4dfae6243bedcc7339fbb7e3ba8a6ca2dcaf04f49cd4b738e4f \
  STREAMING_WINDOW_OBS_LEN=28 \
  PREFIX_LENGTHS=8,12,16,24 \
  REQUIRE_H800=1 \
  N_GPUS="${N_GPUS}" \
  SCENE_START="${SCENE_START:-0}" \
  SCENE_END="${SCENE_END:-}" \
  EXTRACT_DONE_INDEX="${EXTRACT_DONE_INDEX}" \
  STEP_OWNER_LOCK="${STEP_OWNER_LOCK}" \
  LOG_DIR="${LOG_DIR}" \
  SYNC_TRUST_EXTRACT="${SYNC_TRUST_EXTRACT:-0}" \
  ADS_CLI="${ROOT}/scripts/ads_cli_in_logdir.sh" \
  PYTHONDONTWRITEBYTECODE=1 \
  DRY_RUN="${DRY_RUN:-0}" \
  bash "${SOURCE}/scripts/run_streaming_feat_extract_8gpu.sh"
