#!/usr/bin/env bash
# Run the canonical-v2 timestep extraction campaign serially on one 8-GPU node.
#
# STEPS is an optional comma/space separated list of exact step ids printed by
# `DRY_RUN=1`. START_AT accepts either a 1-based campaign ordinal or a step id.
# Every selected step uses a versioned feature namespace and a versioned AOSS
# done-index. set -e makes extraction, upload, deletion, and final-sync failure
# stop the campaign before the next timestep can start.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}"
CANONICAL_PROTOCOL="${CANONICAL_PROTOCOL:-inscene_streaming_canonical_v2_obs28_h124}"
CANONICAL_PROTOCOL_SHA256="${CANONICAL_PROTOCOL_SHA256:-215c720c8537f4dfae6243bedcc7339fbb7e3ba8a6ca2dcaf04f49cd4b738e4f}"
STREAMING_WINDOW_OBS_LEN="${STREAMING_WINDOW_OBS_LEN:-28}"
STREAMING_FEATURE_NAMESPACE="${STREAMING_FEATURE_NAMESPACE:-FEAT_STREAMING_CANONICAL_V2_SEEDED_V1}"
TARGET_FEATURE_NAMESPACE="${TARGET_FEATURE_NAMESPACE:-FEAT_TARGET_CANONICAL_V2_SEEDED_V1}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/umm/users/yinbaiqiao/InsScene-15K}"
SOURCE="${SOURCE:-scannetpp}"
DONE_DIR="${DONE_DIR:-${PROJECT_ROOT}/logs/tstep_canonical_v2_seeded_v1/aoss_done}"
CAMPAIGN_LOG_ROOT="${CAMPAIGN_LOG_ROOT:?set CAMPAIGN_LOG_ROOT=<canonical-v2 campaign log directory>}"
CAMPAIGN_LOCK_FILE="${CAMPAIGN_LOCK_FILE:-${PROJECT_ROOT}/logs/tstep_canonical_v2_seeded_v1/tstep_canonical_v2_seeded_v1_full_campaign.lock}"
PROTOCOL_SPEC="${PROTOCOL_SPEC:-${PROJECT_ROOT}/protocols/inscene_streaming_canonical_v2.json}"
MANIFEST_DIR="${MANIFEST_DIR:-${PROJECT_ROOT}/artifacts/tstep_canonical_v2/manifests}"
GATE_PYTHON="${GATE_PYTHON:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}"
AEC2_NAME="${AEC2_NAME:-h800}"
SYNC_EVERY_SEC="${SYNC_EVERY_SEC:-180}"
SYNC_EVERY_GIB="${SYNC_EVERY_GIB:-80}"
STEPS="${STEPS:-}"
START_AT="${START_AT:-1}"
DRY_RUN="${DRY_RUN:-0}"

EXPECTED_PROTOCOL="inscene_streaming_canonical_v2_obs28_h124"
[[ "${CANONICAL_PROTOCOL}" == "${EXPECTED_PROTOCOL}" ]] || {
  echo "ERROR: canonical protocol must be ${EXPECTED_PROTOCOL}" >&2
  exit 2
}
[[ "${CANONICAL_PROTOCOL_SHA256}" == "215c720c8537f4dfae6243bedcc7339fbb7e3ba8a6ca2dcaf04f49cd4b738e4f" ]] || {
  echo "ERROR: canonical protocol SHA256 must be 215c720c8537f4dfae6243bedcc7339fbb7e3ba8a6ca2dcaf04f49cd4b738e4f" >&2
  exit 2
}
[[ "${STREAMING_WINDOW_OBS_LEN}" == "28" ]] || {
  echo "ERROR: canonical-v2 requires STREAMING_WINDOW_OBS_LEN=28" >&2
  exit 2
}
[[ "${STREAMING_FEATURE_NAMESPACE}" == "FEAT_STREAMING_CANONICAL_V2_SEEDED_V1" ]] || {
  echo "ERROR: invalid canonical streaming namespace: ${STREAMING_FEATURE_NAMESPACE}" >&2
  exit 2
}
[[ "${TARGET_FEATURE_NAMESPACE}" == "FEAT_TARGET_CANONICAL_V2_SEEDED_V1" ]] || {
  echo "ERROR: invalid canonical target namespace: ${TARGET_FEATURE_NAMESPACE}" >&2
  exit 2
}
[[ "${AEC2_NAME}" == "h800" ]] || {
  echo "ERROR: seeded-v1 full extraction is H800-only (AEC2_NAME=${AEC2_NAME})" >&2
  exit 2
}
[[ "${SOURCE}" == "scannetpp" ]] || {
  echo "ERROR: canonical-v2 full campaign source must be scannetpp" >&2
  exit 2
}
case "${DRY_RUN}" in 0|1) ;; *) echo "ERROR: DRY_RUN must be 0 or 1" >&2; exit 2 ;; esac

cd "${PROJECT_ROOT}"
mkdir -p "${CAMPAIGN_LOG_ROOT}"
CAMPAIGN_LOG="${CAMPAIGN_LOG_ROOT}/campaign.log"
STATE_FILE="${CAMPAIGN_LOG_ROOT}/campaign.state.tsv"

require_min_bytes() {
  local path="$1" minimum="$2" size=0
  [[ -f "${path}" ]] && size="$(wc -c < "${path}")"
  if [[ "${size}" -lt "${minimum}" ]]; then
    echo "ERROR: required source looks truncated: ${path} (${size} < ${minimum} bytes)" >&2
    exit 1
  fi
}
require_min_bytes features/run_inscene15k.py 50000
require_min_bytes vidfm3d/utils/temporal_sampling.py 5000
require_min_bytes features/noise_seed.py 5000
require_min_bytes features/wan/wan_feature.py 10000
require_min_bytes features/wan/wan_feature_i2v.py 10000
require_min_bytes features/wan/extract_features.py 5000
require_min_bytes features/cogvideox/cogvideox_feature.py 10000
require_min_bytes features/cogvideox/cogvideox_feature_i2v.py 10000
require_min_bytes features/cogvideox/extract_features.py 5000
require_min_bytes features/minimax_h3/minimax_h3_feature.py 10000
require_min_bytes features/minimax_h3/extract_features.py 1000
require_min_bytes scripts/run_streaming_feat_extract_8gpu.sh 5000
require_min_bytes scripts/run_tstep_sync_loop.sh 5000

[[ -x "${GATE_PYTHON}" ]] || {
  echo "ERROR: completion-gate Python is not executable: ${GATE_PYTHON}" >&2
  exit 1
}
"${GATE_PYTHON}" - "${PROTOCOL_SPEC}" "${CANONICAL_PROTOCOL}" "${CANONICAL_PROTOCOL_SHA256}" <<'PY'
import json
import sys

path, protocol, protocol_sha256 = sys.argv[1:]
value = json.load(open(path, encoding="utf-8"))
if value.get("canonical_protocol") != protocol:
    raise SystemExit("ERROR: protocol spec canonical_protocol mismatch")
if value.get("canonical_protocol_sha256") != protocol_sha256:
    raise SystemExit("ERROR: protocol spec canonical_protocol_sha256 mismatch")
PY

sha256_file() {
  sha256sum -- "$1" | awk '{print $1}'
}

require_bound_file() {
  local label="$1" path="$2" expected="$3" actual
  if [[ ! "${expected}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "ERROR: missing/invalid canary SHA binding for ${label}" >&2
    exit 1
  fi
  [[ -f "${path}" && ! -L "${path}" ]] || {
    echo "ERROR: bound ${label} is missing or a symlink: ${path}" >&2
    exit 1
  }
  actual="$(sha256_file "${path}")"
  [[ "${actual}" == "${expected}" ]] || {
    echo "ERROR: ${label} changed after the accepted canary: ${path}" >&2
    echo "  expected=${expected}" >&2
    echo "  actual=${actual}" >&2
    exit 1
  }
}

verify_canary_source_bindings() {
  # A queued SCO job executes a shared worktree. Recheck every canary-bound
  # source and manifest at worker start and again before each expensive step.
  require_bound_file protocol_spec "${PROTOCOL_SPEC}" "${EXPECTED_PROTOCOL_SPEC_SHA256:-}"
  require_bound_file campaign_runner "${PROJECT_ROOT}/scripts/run_tstep_extract_campaign_v2.sh" "${EXPECTED_CAMPAIGN_RUNNER_SHA256:-}"
  require_bound_file extractor "${PROJECT_ROOT}/features/run_inscene15k.py" "${EXPECTED_EXTRACTOR_SHA256:-}"
  require_bound_file temporal_sampling "${PROJECT_ROOT}/vidfm3d/utils/temporal_sampling.py" "${EXPECTED_TEMPORAL_SAMPLING_SHA256:-}"
  require_bound_file noise_seed_helper "${PROJECT_ROOT}/features/noise_seed.py" "${EXPECTED_NOISE_SEED_HELPER_SHA256:-}"
  require_bound_file wan_feature "${PROJECT_ROOT}/features/wan/wan_feature.py" "${EXPECTED_WAN_FEATURE_SHA256:-}"
  require_bound_file wan_i2v_feature "${PROJECT_ROOT}/features/wan/wan_feature_i2v.py" "${EXPECTED_WAN_I2V_FEATURE_SHA256:-}"
  require_bound_file wan_wrapper "${PROJECT_ROOT}/features/wan/extract_features.py" "${EXPECTED_WAN_WRAPPER_SHA256:-}"
  require_bound_file cogvideox_feature "${PROJECT_ROOT}/features/cogvideox/cogvideox_feature.py" "${EXPECTED_COGVIDEOX_FEATURE_SHA256:-}"
  require_bound_file cogvideox_i2v_feature "${PROJECT_ROOT}/features/cogvideox/cogvideox_feature_i2v.py" "${EXPECTED_COGVIDEOX_I2V_FEATURE_SHA256:-}"
  require_bound_file cogvideox_wrapper "${PROJECT_ROOT}/features/cogvideox/extract_features.py" "${EXPECTED_COGVIDEOX_WRAPPER_SHA256:-}"
  require_bound_file minimax_h3_feature "${PROJECT_ROOT}/features/minimax_h3/minimax_h3_feature.py" "${EXPECTED_MINIMAX_H3_FEATURE_SHA256:-}"
  require_bound_file minimax_h3_wrapper "${PROJECT_ROOT}/features/minimax_h3/extract_features.py" "${EXPECTED_MINIMAX_H3_WRAPPER_SHA256:-}"
  require_bound_file extract_runner "${PROJECT_ROOT}/scripts/run_streaming_feat_extract_8gpu.sh" "${EXPECTED_EXTRACT_RUNNER_SHA256:-}"
  require_bound_file sync_loop "${PROJECT_ROOT}/scripts/run_tstep_sync_loop.sh" "${EXPECTED_SYNC_LOOP_SHA256:-}"
  require_bound_file manifest_p8 "${MANIFEST_DIR}/inscene_c_h124_prefix8.jsonl" "${EXPECTED_MANIFEST_P8_SHA256:-}"
  require_bound_file manifest_p12 "${MANIFEST_DIR}/inscene_c_h124_prefix12.jsonl" "${EXPECTED_MANIFEST_P12_SHA256:-}"
  require_bound_file manifest_p16 "${MANIFEST_DIR}/inscene_c_h124_prefix16.jsonl" "${EXPECTED_MANIFEST_P16_SHA256:-}"
  require_bound_file manifest_p24 "${MANIFEST_DIR}/inscene_c_h124_prefix24.jsonl" "${EXPECTED_MANIFEST_P24_SHA256:-}"
}

if [[ "${DRY_RUN}" == "0" ]]; then
  mkdir -p "$(dirname "${CAMPAIGN_LOCK_FILE}")"
  [[ ! -L "${CAMPAIGN_LOCK_FILE}" ]] || {
    echo "ERROR: campaign lock path must not be a symlink: ${CAMPAIGN_LOCK_FILE}" >&2
    exit 1
  }
  exec {CAMPAIGN_LOCK_FD}>>"${CAMPAIGN_LOCK_FILE}"
  if ! flock -n "${CAMPAIGN_LOCK_FD}"; then
    echo "ERROR: another canonical-v2 extraction job holds ${CAMPAIGN_LOCK_FILE}" >&2
    exit 1
  fi
  printf 'acquired_at=%s pid=%s log_root=%s\n' \
    "$(date -Is)" "$$" "${CAMPAIGN_LOG_ROOT}" >&"${CAMPAIGN_LOCK_FD}"
  verify_canary_source_bindings
fi

if [[ ! -s "${STATE_FILE}" ]]; then
  printf 'ordinal\tstep_id\tvfm\tmode\ttimestep\tstarted_at\tcompleted_at\tdone_index_lines\n' \
    > "${STATE_FILE}"
fi

STEP_IDS=(
  wan_stream_t499
  wan_stream_t999
  wan_target_t499
  wan_target_t999
  minimax_h3_stream_t500
  minimax_h3_stream_t1000
  minimax_h3_target_t500
  minimax_h3_target_t1000
  cogvideox_stream_t499
  cogvideox_stream_t999
  cogvideox_target_t499
  cogvideox_target_t999
)
STEP_VFMS=(
  wan wan wan wan
  minimax_h3 minimax_h3 minimax_h3 minimax_h3
  cogvideox cogvideox cogvideox cogvideox
)
STEP_MODES=(
  streaming_prefix streaming_prefix target_isolated target_isolated
  streaming_prefix streaming_prefix target_isolated target_isolated
  streaming_prefix streaming_prefix target_isolated target_isolated
)
STEP_TIMESTEPS=(
  499 999 499 999
  500 1000 500 1000
  499 999 499 999
)

contains_step_id() {
  local wanted="$1" item
  for item in "${STEP_IDS[@]}"; do
    [[ "${item}" == "${wanted}" ]] && return 0
  done
  return 1
}

normalize_words() {
  tr ',' ' ' <<<"$1" | xargs
}

FILTER_WORDS="$(normalize_words "${STEPS}")"
if [[ -n "${FILTER_WORDS}" ]]; then
  for requested in ${FILTER_WORDS}; do
    contains_step_id "${requested}" || {
      echo "ERROR: unknown STEPS entry: ${requested}" >&2
      echo "Known steps: ${STEP_IDS[*]}" >&2
      exit 2
    }
  done
fi

if [[ "${START_AT}" =~ ^[0-9]+$ ]]; then
  START_ORDINAL="${START_AT}"
  if (( START_ORDINAL < 1 || START_ORDINAL > ${#STEP_IDS[@]} )); then
    echo "ERROR: START_AT ordinal must be in [1,${#STEP_IDS[@]}]" >&2
    exit 2
  fi
else
  contains_step_id "${START_AT}" || {
    echo "ERROR: unknown START_AT step id: ${START_AT}" >&2
    exit 2
  }
  START_ORDINAL=0
  for i in "${!STEP_IDS[@]}"; do
    if [[ "${STEP_IDS[$i]}" == "${START_AT}" ]]; then
      START_ORDINAL=$((i + 1))
      break
    fi
  done
fi

is_selected() {
  local step_id="$1" requested
  [[ -z "${FILTER_WORDS}" ]] && return 0
  for requested in ${FILTER_WORDS}; do
    [[ "${requested}" == "${step_id}" ]] && return 0
  done
  return 1
}

verify_step_completion() {
  local mode="$1" vfm="$2" timestep="$3" done_index="$4" feature_root="$5"
  local expected_keys expected_units expected_structures
  case "${mode}" in
    streaming_prefix)
      expected_keys=32652
      expected_units=10884
      expected_structures=681
      ;;
    target_isolated)
      expected_keys=2043
      expected_units=681
      expected_structures=681
      ;;
    *) echo "ERROR: invalid completion-gate mode: ${mode}" >&2; return 2 ;;
  esac

  "${GATE_PYTHON}" - \
    "${done_index}" "${feature_root}" "${mode}" "${vfm}" "${timestep}" \
    "${expected_keys}" "${expected_units}" "${expected_structures}" \
    "${CANONICAL_PROTOCOL}" "${CANONICAL_PROTOCOL_SHA256}" \
    "${MANIFEST_DIR}" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath

import numpy as np

(
    done_index_raw,
    feature_root_raw,
    mode,
    vfm,
    timestep_raw,
    expected_keys_raw,
    expected_units_raw,
    expected_structures_raw,
    protocol,
    expected_protocol_hash,
    manifest_dir_raw,
) = sys.argv[1:]
done_index = Path(done_index_raw)
feature_root = Path(feature_root_raw)
manifest_dir = Path(manifest_dir_raw)
timestep = int(timestep_raw)
expected_keys = int(expected_keys_raw)
expected_units = int(expected_units_raw)
expected_structures = int(expected_structures_raw)
expected_layers = {15, 20, 25}
expected_prefix_tails = {7, 11, 15, 23}
feature_generation_contract = "inscene_deterministic_noise_v1"
noise_seed_scheme = "inscene_noise_seed_v1"
minimax_noise_seed_scheme = "minimax_h3_fixed_seed_v1"
if mode == "streaming_prefix":
    expected_namespace = "FEAT_STREAMING_CANONICAL_V2_SEEDED_V1"
elif mode == "target_isolated":
    expected_namespace = "FEAT_TARGET_CANONICAL_V2_SEEDED_V1"
else:
    raise SystemExit(f"ERROR: invalid completion-gate mode: {mode}")


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: canonical-v2 completion gate failed: {message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> dict:
    if not path.is_file() or path.is_symlink():
        fail(f"missing/non-regular {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"unreadable {label} {path}: {exc!r}")
    if not isinstance(value, dict):
        fail(f"{label} is not a JSON object: {path}")
    return value


def validate_generation(value: dict, label: str, path: Path) -> None:
    expected = {
        "feature_namespace": expected_namespace,
        "feature_generation_contract": feature_generation_contract,
    }
    if vfm in {"wan", "cogvideox"}:
        expected["noise_seed_scheme"] = noise_seed_scheme
    elif vfm == "minimax_h3":
        expected.update(
            {
                "noise_seed_scheme": minimax_noise_seed_scheme,
                "noise_seed": 0,
                "noise_seed_scope": "reset_per_forward",
            }
        )
    else:
        fail(f"unsupported canonical VFM: {vfm!r}")
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            fail(
                f"{label} generation mismatch {path}: {key} "
                f"expected={wanted!r} actual={value.get(key)!r}"
            )


def validate_index_sidecar(index_path: Path) -> tuple[dict, str]:
    sidecar_path = Path(f"{index_path}.manifest.json")
    sidecar = read_json(sidecar_path, "index sidecar")
    actual_hash = sha256_file(index_path)
    expected = {
        "file": index_path.name,
        "size_bytes": index_path.stat().st_size,
        "sha256": actual_hash,
        "canonical_protocol": protocol,
        "canonical_obs_len": 28,
    }
    for key, value in expected.items():
        if sidecar.get(key) != value:
            fail(
                f"index sidecar mismatch {sidecar_path}: {key} "
                f"expected={value!r} actual={sidecar.get(key)!r}"
            )
    protocol_hash = str(sidecar.get("canonical_protocol_sha256", ""))
    if protocol_hash != expected_protocol_hash:
        fail(
            f"canonical protocol SHA mismatch in {sidecar_path}: "
            f"expected={expected_protocol_hash!r} actual={protocol_hash!r}"
        )
    validate_generation(sidecar, "index sidecar", sidecar_path)
    return sidecar, protocol_hash


if not done_index.is_file() or done_index.is_symlink():
    fail(f"done-index is missing or a symlink: {done_index}")
if not feature_root.is_dir() or feature_root.is_symlink():
    fail(f"local structural-index root is missing or a symlink: {feature_root}")
if (
    feature_root.name != "scannetpp"
    or feature_root.parent.name != vfm
    or feature_root.parent.parent.name != expected_namespace
):
    fail(
        f"structural-index root is outside exact seeded namespace "
        f"{expected_namespace!r}: {feature_root}"
    )

if not manifest_dir.is_dir() or manifest_dir.is_symlink():
    fail(f"canonical manifest directory is missing or a symlink: {manifest_dir}")
expected_manifest_units: dict[str, tuple[list[int], list[int]]] = {}
expected_manifest_targets: dict[str, set[int]] = {}
manifest_scenes: set[str] = set()
for prefix in (8, 12, 16, 24):
    manifest_path = manifest_dir / f"inscene_c_h124_prefix{prefix}.jsonl"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        fail(f"missing/non-regular canonical manifest: {manifest_path}")
    row_count = 0
    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
                scene = str(row["scene_id"])
                window_id = int(row["streaming_window_id"])
                tail = int(row["streaming_tail"])
                row_prefix = int(row["prefix_length"])
                window = [int(value) for value in row["streaming_window_indices"]]
                indices = [int(value) for value in row["streaming_indices"]]
            except Exception as exc:
                fail(f"invalid canonical manifest row {manifest_path}:{line_number}: {exc!r}")
            expected_row = {
                "source": "scannetpp",
                "canonical_protocol": protocol,
                "canonical_obs_len": 28,
                "prefix_length": prefix,
                "streaming_tail": prefix - 1,
                "streaming_feature_root": (
                    "aoss://baiqiao/InsScene-15K/"
                    "FEAT_STREAMING_CANONICAL_V2_SEEDED_V1"
                ),
                "target_feature_root": (
                    "aoss://baiqiao/InsScene-15K/"
                    "FEAT_TARGET_CANONICAL_V2_SEEDED_V1"
                ),
            }
            actual_row = {
                "source": row.get("source"),
                "canonical_protocol": row.get("canonical_protocol"),
                "canonical_obs_len": row.get("canonical_obs_len"),
                "prefix_length": row_prefix,
                "streaming_tail": tail,
                "streaming_feature_root": row.get("streaming_feature_root"),
                "target_feature_root": row.get("target_feature_root"),
            }
            if actual_row != expected_row:
                fail(
                    f"canonical manifest contract mismatch {manifest_path}:{line_number}: "
                    f"expected={expected_row!r} actual={actual_row!r}"
                )
            if len(window) != 28 or indices != window[:prefix]:
                fail(f"canonical window/prefix mismatch {manifest_path}:{line_number}")
            unit = f"{scene}/window_{window_id:04d}/prefix_{tail:06d}"
            if unit in expected_manifest_units:
                fail(f"duplicate canonical manifest unit: {unit!r}")
            expected_manifest_units[unit] = (window, indices)
            manifest_scenes.add(scene)
            scene_targets = expected_manifest_targets.setdefault(scene, set())
            for offset in (0, 1, 2, 4):
                position = tail + offset
                if position >= len(window):
                    fail(
                        f"canonical target position outside window "
                        f"{manifest_path}:{line_number}: {position}"
                    )
                scene_targets.add(window[position])
            row_count += 1
    if row_count != 2721:
        fail(f"manifest rows={row_count}, expected=2721: {manifest_path}")
if len(expected_manifest_units) != 10884 or len(manifest_scenes) != 681:
    fail(
        f"canonical manifest universe units={len(expected_manifest_units)} "
        f"scenes={len(manifest_scenes)}, expected=10884/681"
    )

raw_lines = done_index.read_text(encoding="utf-8").splitlines()
if any(not line.strip() for line in raw_lines):
    fail(f"done-index contains blank lines: {done_index}")
lines = [line.strip().replace("\\", "/") for line in raw_lines]
if len(lines) != len(set(lines)):
    fail(f"done-index contains duplicate keys: {done_index}")
if len(lines) != expected_keys:
    fail(f"done-index keys={len(lines)}, expected={expected_keys}: {done_index}")

if mode == "streaming_prefix":
    key_re = re.compile(
        rf"^(?P<scene>[^/]+)/window_(?P<window>[0-9]{{4}})/"
        rf"prefix_(?P<tail>[0-9]{{6}})/feature_t{timestep}_layer"
        rf"(?P<layer>15|20|25)\.sft$"
    )
else:
    key_re = re.compile(
        rf"^(?P<scene>[^/]+)/feature_t{timestep}_layer(?P<layer>15|20|25)\.sft$"
    )

unit_layers: dict[str, set[int]] = {}
for rel in lines:
    pure = PurePosixPath(rel)
    if pure.is_absolute() or ".." in pure.parts:
        fail(f"unsafe done-index key: {rel!r}")
    match = key_re.fullmatch(rel)
    if match is None:
        fail(f"malformed/wrong-timestep done-index key: {rel!r}")
    if mode == "streaming_prefix":
        tail = int(match.group("tail"))
        if tail not in expected_prefix_tails:
            fail(f"unexpected canonical prefix tail in key: {rel!r}")
    unit = rel.rsplit("/", 1)[0]
    unit_layers.setdefault(unit, set()).add(int(match.group("layer")))

partials = sorted(unit for unit, layers in unit_layers.items() if layers != expected_layers)
if partials:
    fail(f"{len(partials)} partial units; first={partials[0]!r}")
if len(unit_layers) != expected_units:
    fail(f"complete units={len(unit_layers)}, expected={expected_units}")

completed_units = set(unit_layers)
if mode == "streaming_prefix":
    index_paths = sorted(feature_root.glob("*/prefix_index.npy"))
    if len(index_paths) != expected_structures:
        fail(f"prefix_index files={len(index_paths)}, expected={expected_structures}")
    structural_units: set[str] = set()
    for index_path in index_paths:
        if index_path.parent.parent != feature_root:
            fail(f"prefix index escaped scene layout: {index_path}")
        scene = index_path.parent.name
        _sidecar, protocol_hash = validate_index_sidecar(index_path)
        try:
            records = np.load(index_path, allow_pickle=True).tolist()
        except Exception as exc:
            fail(f"unreadable prefix index {index_path}: {exc!r}")
        if not isinstance(records, list) or not records:
            fail(f"empty/non-list prefix index: {index_path}")
        scene_units: set[str] = set()
        for record in records:
            if not isinstance(record, dict):
                fail(f"non-object prefix record in {index_path}")
            try:
                window_id = int(record["window_id"])
                tail = int(record["tail"])
                indices = [int(value) for value in record["indices"]]
                window_indices = [int(value) for value in record["window_indices"]]
            except Exception as exc:
                fail(f"invalid prefix record in {index_path}: {exc!r}")
            if window_id < 0 or tail not in expected_prefix_tails:
                fail(f"invalid window/tail in {index_path}: {window_id}/{tail}")
            if len(indices) != tail + 1 or len(window_indices) != 28:
                fail(f"invalid canonical index lengths in {index_path}: window={window_id} tail={tail}")
            if indices != window_indices[: tail + 1]:
                fail(f"prefix is not canonical-window prefix in {index_path}: window={window_id} tail={tail}")
            if record.get("canonical_protocol") != protocol or int(record.get("canonical_obs_len", -1)) != 28:
                fail(f"prefix record protocol mismatch in {index_path}")
            if record.get("canonical_protocol_sha256") != protocol_hash:
                fail(f"prefix record/sidecar protocol SHA mismatch in {index_path}")
            unit = f"{scene}/window_{window_id:04d}/prefix_{tail:06d}"
            expected_record = expected_manifest_units.get(unit)
            if expected_record is None:
                fail(f"prefix record is absent from canonical manifests: {unit!r}")
            if (window_indices, indices) != expected_record:
                fail(
                    f"prefix record differs from canonical manifest {unit!r}: "
                    f"expected={expected_record!r} actual={(window_indices, indices)!r}"
                )
            scene_units.add(unit)
        if len(scene_units) != len(records):
            fail(f"duplicate structural units in {index_path}")
        structural_units.update(scene_units)
    if structural_units != completed_units:
        missing = sorted(completed_units - structural_units)
        extra = sorted(structural_units - completed_units)
        fail(
            f"prefix-index coverage mismatch missing={len(missing)} extra={len(extra)} "
            f"first_missing={missing[:1]} first_extra={extra[:1]}"
        )
    if structural_units != set(expected_manifest_units):
        fail("prefix-index universe does not exactly match the four canonical manifests")
else:
    index_paths = sorted(feature_root.glob("*/target_indices.npy"))
    if len(index_paths) != expected_structures:
        fail(f"target_indices files={len(index_paths)}, expected={expected_structures}")
    structural_units: set[str] = set()
    for index_path in index_paths:
        if index_path.parent.parent != feature_root:
            fail(f"target index escaped scene layout: {index_path}")
        scene = index_path.parent.name
        _sidecar, protocol_hash = validate_index_sidecar(index_path)
        try:
            target_values = np.load(index_path, allow_pickle=False)
        except Exception as exc:
            fail(f"unreadable target index {index_path}: {exc!r}")
        if target_values.ndim != 1 or target_values.dtype.kind not in "iu" or target_values.size == 0:
            fail(f"invalid target index shape/dtype: {index_path}")
        as_ints = [int(value) for value in target_values.tolist()]
        if len(as_ints) != len(set(as_ints)) or any(value < 0 for value in as_ints):
            fail(f"duplicate/negative target ids: {index_path}")
        expected_target_values = sorted(expected_manifest_targets.get(scene, set()))
        if as_ints != expected_target_values:
            fail(
                f"target index differs from canonical manifest union for {scene}: "
                f"expected={expected_target_values!r} actual={as_ints!r}"
            )
        meta_path = index_path.with_name("target_indices.meta.json")
        meta = read_json(meta_path, "target index metadata")
        expected_meta = {
            "protocol": protocol,
            "protocol_sha256": protocol_hash,
            "obs_len": 28,
            "prefixes": [8, 12, 16, 24],
            "horizons": [1, 2, 4],
            "include_anchors": True,
            "target_count": len(as_ints),
            "target_indices_sha256": sha256_file(index_path),
        }
        for key, value in expected_meta.items():
            if meta.get(key) != value:
                fail(
                    f"target metadata mismatch {meta_path}: {key} "
                    f"expected={value!r} actual={meta.get(key)!r}"
                )
        validate_generation(meta, "target index metadata", meta_path)
        structural_units.add(scene)
    if structural_units != completed_units:
        missing = sorted(completed_units - structural_units)
        extra = sorted(structural_units - completed_units)
        fail(
            f"target-index coverage mismatch missing={len(missing)} extra={len(extra)} "
            f"first_missing={missing[:1]} first_extra={extra[:1]}"
        )
    if structural_units != manifest_scenes:
        fail("target-index scene universe does not exactly match canonical manifests")

print(
    "COMPLETION_GATE PASS "
    f"mode={mode} vfm={vfm} t={timestep} keys={len(lines)} "
    f"units={len(completed_units)} structural={expected_structures}"
)
PY
}

run_step() {
  local ordinal="$1" step_id="$2" vfm="$3" mode="$4" timestep="$5"
  local feature_kind feature_namespace out_root mode_tag extract_stream extract_target
  local done_index step_log started completed
  case "${mode}" in
    streaming_prefix)
      feature_kind=FEAT_STREAMING
      feature_namespace="${STREAMING_FEATURE_NAMESPACE}"
      out_root="${DATASET_ROOT}/${STREAMING_FEATURE_NAMESPACE}"
      mode_tag=stream
      extract_stream=1
      extract_target=0
      ;;
    target_isolated)
      feature_kind=FEAT_TARGET
      feature_namespace="${TARGET_FEATURE_NAMESPACE}"
      out_root="${DATASET_ROOT}/${TARGET_FEATURE_NAMESPACE}"
      mode_tag=target
      extract_stream=0
      extract_target=1
      ;;
    *) echo "ERROR: invalid campaign mode: ${mode}" >&2; exit 2 ;;
  esac
  done_index="${DONE_DIR}/${feature_namespace}_${vfm}_t${timestep}.txt"
  [[ -f "${done_index}" ]] || {
    echo "ERROR: missing canonical-v2 AOSS done-index: ${done_index}" >&2
    exit 1
  }
  [[ ! -L "${done_index}" ]] || {
    echo "ERROR: done-index must not be a symlink: ${done_index}" >&2
    exit 1
  }
  step_log="${CAMPAIGN_LOG_ROOT}/$(printf '%02d' "${ordinal}")_${step_id}"
  started="$(date -Is)"
  echo "[${started}] START ordinal=${ordinal} step=${step_id} vfm=${vfm} mode=${mode} t=${timestep} namespace=${feature_namespace} index_lines=$(wc -l < "${done_index}")" \
    | tee -a "${CAMPAIGN_LOG}"
  if [[ "${DRY_RUN}" == "0" ]]; then
    verify_canary_source_bindings
  fi
  CANONICAL_PROTOCOL="${CANONICAL_PROTOCOL}" \
  CANONICAL_PROTOCOL_SHA256="${CANONICAL_PROTOCOL_SHA256}" \
  REQUIRE_H800=1 \
  STREAMING_WINDOW_OBS_LEN="${STREAMING_WINDOW_OBS_LEN}" \
  FEATURE_NAMESPACE="${feature_namespace}" \
  VFM="${vfm}" \
  T="${timestep}" \
  EXTRACT_STREAM="${extract_stream}" \
  EXTRACT_TARGET="${extract_target}" \
  EXTRACT_DONE_INDEX="${done_index}" \
  DATASET_ROOT="${DATASET_ROOT}" \
  SOURCE="${SOURCE}" \
  OUT_STREAM="${out_root}" \
  OUT_TARGET="${out_root}" \
  LOG_DIR="${step_log}" \
  SYNC_EVERY_SEC="${SYNC_EVERY_SEC}" \
  SYNC_EVERY_GIB="${SYNC_EVERY_GIB}" \
  DRY_RUN="${DRY_RUN}" \
    bash scripts/run_streaming_feat_extract_8gpu.sh
  completed="$(date -Is)"
  if [[ "${DRY_RUN}" == "0" ]]; then
    verify_canary_source_bindings
    verify_step_completion \
      "${mode}" "${vfm}" "${timestep}" "${done_index}" \
      "${out_root}/${vfm}/${SOURCE}" \
      | tee -a "${CAMPAIGN_LOG}" "${step_log}/completion_gate.log"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${ordinal}" "${step_id}" "${vfm}" "${mode}" "${timestep}" \
      "${started}" "${completed}" "$(wc -l < "${done_index}")" >> "${STATE_FILE}"
  fi
  echo "[${completed}] COMPLETE ordinal=${ordinal} step=${step_id} dry_run=${DRY_RUN}" \
    | tee -a "${CAMPAIGN_LOG}"
}

selected_count=0
for i in "${!STEP_IDS[@]}"; do
  ordinal=$((i + 1))
  step_id="${STEP_IDS[$i]}"
  (( ordinal >= START_ORDINAL )) || continue
  is_selected "${step_id}" || continue
  selected_count=$((selected_count + 1))
  run_step \
    "${ordinal}" "${step_id}" "${STEP_VFMS[$i]}" \
    "${STEP_MODES[$i]}" "${STEP_TIMESTEPS[$i]}"
done

if [[ "${selected_count}" -eq 0 ]]; then
  echo "ERROR: STEPS/START_AT selected no campaign steps" >&2
  exit 2
fi
echo "[$(date -Is)] CAMPAIGN COMPLETE selected=${selected_count} dry_run=${DRY_RUN}" \
  | tee -a "${CAMPAIGN_LOG}"
