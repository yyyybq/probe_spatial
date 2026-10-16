#!/usr/bin/env bash
# One single-GPU production shard of the MiniMax-H3 bf16 seeded-v1 re-extraction
# (stages 2/3 of docs/MINIMAX_REEXTRACT_RETRAIN_PLAN_20261016.md: 2 timesteps x
# 2 modes x 8 shards = 32 one-GPU jobs), run from the frozen runtime
# runtime/tstep_minimax_bf16_v1_extractor_r2.
#
# Same contract as scripts/run_tstep_minimax_bf16_v1_extract.sh (the canary
# wrapper): namespace *_SEEDED_V1_MINIMAX_BF16_V1, canonical-v2 protocol SHA,
# layers 20 25, H800 only, frozen model/prompt, validating sync loop.
#
# Why per-shard state: the frozen sync loop merges its done-index without a
# lock and its final acceptance check counts every feature file left under its
# local root.  Several jobs sharing one (namespace, timestep) root/index would
# lose index entries and fail each other's acceptance.  Each shard therefore owns
#   local root   InsScene-15K/tstep_minimax_bf16_v1_shards/<shard>/<NAMESPACE>
#   done-index   logs/tstep_minimax_bf16_v1/<RUN_TAG>/shards/<shard>/aoss_done/<NAMESPACE>_minimax_h3_t<T>.txt
#   step lock    logs/tstep_minimax_bf16_v1/<RUN_TAG>/shards/<shard>/step.lock
# and publishes to the shared AOSS namespace (disjoint scene keys).  Shard
# indices and keep_local structural files are consolidated after all shards
# pass (see the plan's stage 3 exit criteria).
#
# Scene span of shard i of K over the 682 collected ScanNet++ scenes:
#   [682*i//K, 682*(i+1)//K), except shard 0 starts at scene 1 because the
#   stage-1 canary already published scene 0 into the canonical index.
#
# Required env: EXPECTED_SOURCE_TREE_SHA256 T=500|1000 MODE=stream|target
#               SHARD_INDEX SHARD_COUNT=8 RUN_TAG
# Optional env: SYNC_TRUST_EXTRACT (0) DRY_RUN AOSS_CONF
set -euo pipefail

ROOT=/mnt/umm/users/yinbaiqiao/probe_spatial
RELEASE=${ROOT}/runtime/tstep_minimax_bf16_v1_extractor_r2
SOURCE=${RELEASE}/source
DATASET_ROOT=/mnt/umm/users/yinbaiqiao/InsScene-15K
N_SCENES_EXPECTED=682
EXPECTED_SOURCE_TREE_SHA256=${EXPECTED_SOURCE_TREE_SHA256:?EXPECTED_SOURCE_TREE_SHA256 is required}
T=${T:?T is required (500|1000)}
MODE=${MODE:?MODE is required (stream|target)}
SHARD_INDEX=${SHARD_INDEX:?SHARD_INDEX is required}
SHARD_COUNT=${SHARD_COUNT:?SHARD_COUNT is required}
RUN_TAG=${RUN_TAG:?RUN_TAG is required}

case "${T}" in
  500|1000) ;;
  *) echo "ERROR: T must be 500 or 1000, got ${T}" >&2; exit 2 ;;
esac
case "${MODE}" in
  stream) EXTRACT_STREAM=1; EXTRACT_TARGET=0; NAMESPACE=FEAT_STREAMING_CANONICAL_V2_SEEDED_V1_MINIMAX_BF16_V1 ;;
  target) EXTRACT_STREAM=0; EXTRACT_TARGET=1; NAMESPACE=FEAT_TARGET_CANONICAL_V2_SEEDED_V1_MINIMAX_BF16_V1 ;;
  *) echo "ERROR: MODE must be stream or target, got ${MODE}" >&2; exit 2 ;;
esac
[[ "${SHARD_COUNT}" =~ ^[1-9][0-9]*$ && "${SHARD_INDEX}" =~ ^[0-9]+$ ]] || {
  echo "ERROR: SHARD_INDEX/SHARD_COUNT must be integers" >&2; exit 2;
}
(( SHARD_INDEX < SHARD_COUNT )) || { echo "ERROR: SHARD_INDEX >= SHARD_COUNT" >&2; exit 2; }
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "ERROR: unsafe RUN_TAG" >&2; exit 2; }

scene_start=$(( N_SCENES_EXPECTED * SHARD_INDEX / SHARD_COUNT ))
scene_end=$(( N_SCENES_EXPECTED * (SHARD_INDEX + 1) / SHARD_COUNT ))
(( scene_start == 0 )) && scene_start=1   # scene 0 = stage-1 canary
(( scene_end > scene_start )) || { echo "ERROR: empty shard span" >&2; exit 2; }

manifest="${RELEASE}/source_manifest.json"
[[ -f "${manifest}" && -f "${RELEASE}/source.sha256" ]] || { echo "ERROR: frozen runtime is incomplete" >&2; exit 3; }
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
  echo "ERROR: frozen source tree mismatch: ${actual_tree_sha256} != ${EXPECTED_SOURCE_TREE_SHA256}" >&2; exit 3;
}
(cd "${SOURCE}" && sha256sum --status -c "${RELEASE}/source.sha256") || { echo "ERROR: source.sha256 check failed" >&2; exit 3; }

shard_id="t${T}_${MODE}_s${SHARD_INDEX}of${SHARD_COUNT}"
state_dir="${ROOT}/logs/tstep_minimax_bf16_v1/${RUN_TAG}/shards/${shard_id}"
LOG_DIR="${ROOT}/logs/tstep_minimax_bf16_v1/${RUN_TAG}/minimax_h3_${shard_id}"
EXTRACT_DONE_INDEX="${state_dir}/aoss_done/${NAMESPACE}_minimax_h3_t${T}.txt"
STEP_OWNER_LOCK="${state_dir}/step.lock"
OUT_ROOT_SHARD="${DATASET_ROOT}/tstep_minimax_bf16_v1_shards/${shard_id}/${NAMESPACE}"
[[ -f "${EXTRACT_DONE_INDEX}" && ! -L "${EXTRACT_DONE_INDEX}" ]] || {
  echo "ERROR: shard done-index must pre-exist: ${EXTRACT_DONE_INDEX}" >&2; exit 3;
}
[[ -f "${STEP_OWNER_LOCK}" && "$(tr -d '\n' < "${STEP_OWNER_LOCK}")" == "${LOG_DIR}" ]] || {
  echo "ERROR: step lock ${STEP_OWNER_LOCK} must exist and name ${LOG_DIR}" >&2; exit 3;
}
export AOSS_CONF="${AOSS_CONF:-/mnt/umm/users/yinbaiqiao/aoss/petreloss.conf}"
[[ -r "${AOSS_CONF}" ]] || { echo "ERROR: AOSS_CONF is not readable: ${AOSS_CONF}" >&2; exit 3; }
# The frozen sync loop starts with `cd FEATURE_ROOT` (OUT_ROOT/<vfm>/<source>),
# which must exist before the launcher starts it.
mkdir -p "${LOG_DIR}" "${OUT_ROOT_SHARD}/minimax_h3/scannetpp"
{
  echo "frozen_runtime=${RELEASE}"
  echo "source_tree_sha256=${actual_tree_sha256}"
  echo "namespace=${NAMESPACE} t=${T} mode=${MODE} layers=20 25 shard=${SHARD_INDEX}/${SHARD_COUNT}"
  echo "scene_span=${scene_start}:${scene_end}"
  echo "local_root=${OUT_ROOT_SHARD}"
  echo "done_index=${EXTRACT_DONE_INDEX}"
  echo "step_owner_lock=${STEP_OWNER_LOCK}"
  echo "sync_trust_extract=${SYNC_TRUST_EXTRACT:-0}"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname)"
} | tee "${LOG_DIR}/wrapper.txt"

# Keep the frozen tree read-only in practice: the upload client writes its
# record into LOG_DIR (scripts/ads_cli_in_logdir.sh) and Python writes no .pyc.
[[ -x "${ROOT}/scripts/ads_cli_in_logdir.sh" ]] || { echo "ERROR: ads_cli_in_logdir.sh is missing" >&2; exit 3; }
export PROJECT_ROOT="${SOURCE}"
exec env \
  VFM=minimax_h3 \
  T="${T}" \
  LAYERS="20 25" \
  EXTRACT_STREAM="${EXTRACT_STREAM}" \
  EXTRACT_TARGET="${EXTRACT_TARGET}" \
  FEATURE_NAMESPACE="${NAMESPACE}" \
  OUT_STREAM="${OUT_ROOT_SHARD}" \
  OUT_TARGET="${OUT_ROOT_SHARD}" \
  CANONICAL_PROTOCOL=inscene_streaming_canonical_v2_obs28_h124 \
  CANONICAL_PROTOCOL_SHA256=215c720c8537f4dfae6243bedcc7339fbb7e3ba8a6ca2dcaf04f49cd4b738e4f \
  STREAMING_WINDOW_OBS_LEN=28 \
  PREFIX_LENGTHS=8,12,16,24 \
  REQUIRE_H800=1 \
  N_GPUS=1 \
  SCENE_START="${scene_start}" \
  SCENE_END="${scene_end}" \
  EXTRACT_DONE_INDEX="${EXTRACT_DONE_INDEX}" \
  STEP_OWNER_LOCK="${STEP_OWNER_LOCK}" \
  LOG_DIR="${LOG_DIR}" \
  SYNC_TRUST_EXTRACT="${SYNC_TRUST_EXTRACT:-0}" \
  ADS_CLI="${ROOT}/scripts/ads_cli_in_logdir.sh" \
  PYTHONDONTWRITEBYTECODE=1 \
  DRY_RUN="${DRY_RUN:-0}" \
  bash "${SOURCE}/scripts/run_streaming_feat_extract_8gpu.sh"
