#!/usr/bin/env bash
# Publish complete local timestep units to AOSS, then delete only after the
# done-index commit.  Writer temps and hardlink races stay fail-closed.
#
# SYNC_TRUST_EXTRACT=1 (production default) inventories by complete file
# presence and inode/size.  It does not reopen safetensors, rehash payloads,
# or replay sidecar contracts.  Set 0 to keep the historical validator.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}"
FEATURE_KIND="${FEATURE_KIND:?set FEATURE_KIND=FEAT_STREAMING|FEAT_TARGET}"
FEATURE_NAMESPACE="${FEATURE_NAMESPACE:?set FEATURE_NAMESPACE}"
FEATURE_ROOT="${FEATURE_ROOT:?set FEATURE_ROOT}"
VFM="${VFM:?set VFM}"
T="${T:?set T}"
LAYERS="${LAYERS:-15 20 25}"
LAYERS="${LAYERS//,/ }"
EXTRACT_DONE_INDEX="${EXTRACT_DONE_INDEX:?set EXTRACT_DONE_INDEX}"
LOG_DIR="${LOG_DIR:?set LOG_DIR}"
STOP_FILE="${STOP_FILE:?set STOP_FILE}"
FEATURE_SCAN_ROOT="${FEATURE_SCAN_ROOT:-${FEATURE_ROOT}}"
SOURCE="${SOURCE:-scannetpp}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/umm/users/yinbaiqiao/InsScene-15K}"
AOSS_DIR="${AOSS_DIR:-/mnt/umm/users/yinbaiqiao/aoss}"
CONF="${CONF:-${AOSS_DIR}/petreloss.conf}"
ADS_CLI="${ADS_CLI:-${AOSS_DIR}/ads-cli}"
# Keep one receipt-producing sync loop per VFM/source, but allow its AOSS
# client to transfer independent staged objects concurrently.  Multiple sync
# loops are unsafe because they share the receipt, done index, and cleanup
# scope; ADS_THREADS is the safe concurrency boundary.
ADS_THREADS="${ADS_THREADS:-1}"
ADS_LISTERS="${ADS_LISTERS:-2}"
if [[ -z "${SYNC_PYTHON:-}" ]]; then
  if [[ -x /mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python ]]; then
    SYNC_PYTHON=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python
  else
    SYNC_PYTHON=python3
  fi
fi
SYNC_EVERY_SEC="${SYNC_EVERY_SEC:-180}"
SYNC_EVERY_GIB="${SYNC_EVERY_GIB:-80}"
SYNC_POLL_SEC="${SYNC_POLL_SEC:-30}"
SYNC_RETRY_SEC="${SYNC_RETRY_SEC:-5}"
SYNC_MAX_FAILURES="${SYNC_MAX_FAILURES:-8}"
SYNC_FINAL_MAX_ATTEMPTS="${SYNC_FINAL_MAX_ATTEMPTS:-8}"
SYNC_DISJOINT_SCAN="${SYNC_DISJOINT_SCAN:-0}"
SYNC_TRUST_EXTRACT="${SYNC_TRUST_EXTRACT:-1}"
SYNC_INDEX_MERGE_ATTEMPTS="${SYNC_INDEX_MERGE_ATTEMPTS:-6}"
GAP_ADAPTER="${GAP_ADAPTER:-}"
GAP_SYNC_MAX_SCENES="${GAP_SYNC_MAX_SCENES:-4}"
SEEDED_PROTOCOL_SHA256="215c720c8537f4dfae6243bedcc7339fbb7e3ba8a6ca2dcaf04f49cd4b738e4f"
WAN_SNAPSHOT="/mnt/umm/users/yinbaiqiao/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780a534b6463e45facd96134c9f345acfa5b"
COGX_SNAPSHOT="/mnt/umm/users/yinbaiqiao/.cache/huggingface/hub/models--THUDM--CogVideoX-5b-I2V/snapshots/a6f0f4858a8395e7429d82493864ce92bf73af11"
MINIMAX_MODEL="/mnt/umm/users/yinbaiqiao/Consistency/minimax-h3/models/MiniMax-H3"

case "${FEATURE_KIND}" in
  FEAT_STREAMING|FEAT_TARGET) ;;
  *) echo "ERROR: FEATURE_KIND must be FEAT_STREAMING or FEAT_TARGET" >&2; exit 2 ;;
esac
[[ "${SYNC_DISJOINT_SCAN}" =~ ^[01]$ ]] || {
  echo "ERROR: SYNC_DISJOINT_SCAN must be 0 or 1" >&2
  exit 2
}
[[ "${SYNC_TRUST_EXTRACT}" =~ ^[01]$ ]] || {
  echo "ERROR: SYNC_TRUST_EXTRACT must be 0 or 1" >&2
  exit 2
}
for value in "${T}" "${SYNC_EVERY_SEC}" "${SYNC_EVERY_GIB}" "${SYNC_POLL_SEC}" \
             "${SYNC_RETRY_SEC}" "${SYNC_MAX_FAILURES}" "${SYNC_FINAL_MAX_ATTEMPTS}"; do
  [[ "${value}" =~ ^[0-9]+$ ]] || {
    echo "ERROR: numeric setting is invalid: ${value}" >&2
    exit 2
  }
done
for value in "${ADS_THREADS}" "${ADS_LISTERS}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: ADS transfer/list concurrency must be a positive integer: ${value}" >&2
    exit 2
  }
done

FEATURE_ROOT="$(cd "${FEATURE_ROOT}" && pwd)"
FEATURE_SCAN_ROOT="$(cd "${FEATURE_SCAN_ROOT}" && pwd)"
case "${FEATURE_SCAN_ROOT}" in
  "${FEATURE_ROOT}"|"${FEATURE_ROOT}"/*) ;;
  *)
    echo "ERROR: FEATURE_SCAN_ROOT escapes FEATURE_ROOT" >&2
    exit 2
    ;;
esac

mkdir -p "${LOG_DIR}" "$(dirname -- "${EXTRACT_DONE_INDEX}")"
[[ -f "${EXTRACT_DONE_INDEX}" ]] || : > "${EXTRACT_DONE_INDEX}"
[[ ! -L "${EXTRACT_DONE_INDEX}" ]] || {
  echo "ERROR: done-index must not be a symlink" >&2
  exit 2
}
chmod a+r -- "${EXTRACT_DONE_INDEX}" 2>/dev/null || true
[[ -r "${EXTRACT_DONE_INDEX}" ]] || {
  echo "ERROR: done-index is unreadable: ${EXTRACT_DONE_INDEX}" >&2
  exit 2
}

if [[ "${FEATURE_NAMESPACE}" == *_CANONICAL_V2_SEEDED_V1 || "${FEATURE_NAMESPACE}" == *_CANONICAL_V2_SEEDED_V1_MINIMAX_BF16_V1 ]]; then
  case "${VFM}" in
    wan) frozen_model="${WAN_SNAPSHOT}"; frozen_prompt="." ;;
    cogvideox) frozen_model="${COGX_SNAPSHOT}"; frozen_prompt="" ;;
    minimax_h3) frozen_model="${MINIMAX_MODEL}"; frozen_prompt="." ;;
    *) frozen_model=""; frozen_prompt="" ;;
  esac
  if [[ -n "${EXPECTED_MODEL_ID:-}" && -n "${frozen_model}" && "${EXPECTED_MODEL_ID}" != "${frozen_model}" ]]; then
    echo "ERROR: model_id may not be overridden (seeded-v1 ${VFM} is frozen to ${frozen_model})" >&2
    exit 2
  fi
  if [[ -n "${EXPECTED_PROMPT+x}" && "${EXPECTED_PROMPT}" != "${frozen_prompt}" ]]; then
    echo "ERROR: prompt may not be overridden (seeded-v1 ${VFM} prompt is frozen)" >&2
    exit 2
  fi
  EXPECTED_MODEL_ID="${EXPECTED_MODEL_ID:-${frozen_model}}"
  if [[ -z "${EXPECTED_PROMPT+x}" ]]; then
    EXPECTED_PROMPT="${frozen_prompt}"
  fi
fi

if [[ -n "${GAP_ADAPTER}" ]]; then
  [[ -f "${GAP_ADAPTER}" && ! -L "${GAP_ADAPTER}" ]] || {
    echo "ERROR: GAP_ADAPTER is missing or a symlink: ${GAP_ADAPTER}" >&2
    exit 2
  }
  actual_adapter_sha="$(sha256sum -- "${GAP_ADAPTER}" | awk '{print $1}')"
  [[ "${EXPECTED_GAP_ADAPTER_SHA256:-}" == "${actual_adapter_sha}" ]] || {
    echo "ERROR: GAP_ADAPTER sha256 mismatch" >&2
    exit 2
  }
fi

AOSS_API_HOST="${AOSS_API_HOST:-aoss-internal.cn-fz-01.fjscmsapi-oss.com}"
if [[ -n "${HOST:-}" && "${HOST}" != *conda* && "${HOST}" != x86_64* ]]; then
  AOSS_API_HOST="${HOST}"
fi
BUCKET="${BUCKET:-baiqiao}"
AOSS_HOST="${AOSS_HOST:-${BUCKET}.${AOSS_API_HOST}}"
export NO_PROXY="localhost,127.0.0.1,::1,${AOSS_HOST},.${AOSS_API_HOST},${AOSS_API_HOST}"
export no_proxy="${NO_PROXY}"

echo "tstep sync loop kind=${FEATURE_KIND} ns=${FEATURE_NAMESPACE} vfm=${VFM} t=${T}"
echo "  feature_root=${FEATURE_ROOT}"
echo "  feature_scan_root=${FEATURE_SCAN_ROOT}"
echo "  cadence=${SYNC_EVERY_SEC}s threshold=${SYNC_EVERY_GIB}GiB poll=${SYNC_POLL_SEC}s"
echo "  trust_extract=${SYNC_TRUST_EXTRACT} (1=stat-only inventory; 0=open/hash/sidecar contract)"
echo "  sync_python=${SYNC_PYTHON}"

read_ads_credentials() {
  AK="$(awk '
    /^\[fj\]/ {in_fj=1; next}
    /^\[/ {in_fj=0}
    in_fj && /^access_key[[:space:]]*=/ {gsub(/^access_key[[:space:]]*=[[:space:]]*/, ""); print; exit}
  ' "${CONF}")"
  SK="$(awk '
    /^\[fj\]/ {in_fj=1; next}
    /^\[/ {in_fj=0}
    in_fj && /^secret_key[[:space:]]*=/ {gsub(/^secret_key[[:space:]]*=[[:space:]]*/, ""); print; exit}
  ' "${CONF}")"
  if [[ -z "${AK}" || -z "${SK}" ]]; then
    echo "ERROR: failed to read fj credentials from ${CONF}" >&2
    return 1
  fi
}

scan_atomic_temps() {
  local listing="$1"
  if [[ -n "${GAP_ADAPTER}" ]]; then
    "${SYNC_PYTHON}" "${GAP_ADAPTER}" atomic-temps \
      --root "${FEATURE_ROOT}" \
      --worklist "${CANONICAL_GAP_WORKLIST}" \
      --worklist-sha256 "${CANONICAL_GAP_WORKLIST_SHA256}" \
      --output "${listing}"
    return 0
  fi
  FEATURE_ROOT="${FEATURE_ROOT}" \
  FEATURE_SCAN_ROOT="${FEATURE_SCAN_ROOT}" \
  T="${T}" \
  "${SYNC_PYTHON}" - "${listing}" <<'PY'
import os
import sys
from pathlib import Path

root = Path(os.environ["FEATURE_SCAN_ROOT"]).resolve()
feature_root = Path(os.environ["FEATURE_ROOT"]).resolve()
timestep = os.environ["T"]
out = Path(sys.argv[1])
found = []
for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
    dirnames[:] = [name for name in dirnames if not name.startswith(".tstep_sync_staging_")]
    parent = Path(dirpath)
    for name in filenames:
        lower = name.lower()
        if "tmp" not in lower:
            continue
        if not (
            f"feature_t{timestep}_" in lower
            or "prefix_index" in lower
            or "target_indices" in lower
        ):
            continue
        path = parent / name
        if path.is_symlink() or not path.is_file():
            continue
        found.append(path.relative_to(feature_root).as_posix())
out.write_text("".join(f"{item}\n" for item in sorted(found)), encoding="utf-8")
PY
}

snapshot_candidates() {
  local snapshot="$1" invalid="$2"
  if [[ -n "${GAP_ADAPTER}" ]]; then
    local mode="${GAP_MODE:?set GAP_MODE}"
    local protocol_sha="${GAP_PROTOCOL_SHA256:?set GAP_PROTOCOL_SHA256}"
    local prompt="${EXPECTED_PROMPT-}"
    local height="${EXPECTED_HEIGHT:?set EXPECTED_HEIGHT}"
    local width="${EXPECTED_WIDTH:?set EXPECTED_WIDTH}"
    local frames="${EXPECTED_NUM_FRAMES:?set EXPECTED_NUM_FRAMES}"
    local model_id="${EXPECTED_MODEL_ID:?set EXPECTED_MODEL_ID}"
    "${SYNC_PYTHON}" "${GAP_ADAPTER}" snapshot \
      --root "${FEATURE_ROOT}" \
      --scan-root "${FEATURE_SCAN_ROOT}" \
      --worklist "${CANONICAL_GAP_WORKLIST}" \
      --worklist-sha256 "${CANONICAL_GAP_WORKLIST_SHA256}" \
      --mode "${mode}" \
      --vfm "${VFM}" \
      --model-id "${model_id}" \
      --timestep "${T}" \
      --namespace "${FEATURE_NAMESPACE}" \
      --layers ${LAYERS} \
      --protocol-sha256 "${protocol_sha}" \
      --prompt "${prompt}" \
      --resize "${height}" "${width}" \
      --num-frames "${frames}" \
      --max-scenes "${GAP_SYNC_MAX_SCENES}" \
      --output "${snapshot}" \
      --invalid-output "${invalid}"
    return 0
  fi
  local snap_env=(
    "FEATURE_KIND=${FEATURE_KIND}"
    "FEATURE_NAMESPACE=${FEATURE_NAMESPACE}"
    "FEATURE_ROOT=${FEATURE_ROOT}"
    "FEATURE_SCAN_ROOT=${FEATURE_SCAN_ROOT}"
    "VFM=${VFM}"
    "T=${T}"
    "LAYERS=${LAYERS}"
    "EXTRACT_DONE_INDEX=${EXTRACT_DONE_INDEX}"
    "SYNC_TRUST_EXTRACT=${SYNC_TRUST_EXTRACT}"
    "EXPECTED_MODEL_ID=${EXPECTED_MODEL_ID:-}"
    "PROJECT_ROOT=${PROJECT_ROOT}"
    "SYNC_PENDING_FILE=${snapshot%.snapshot.jsonl}.pending.txt"
  )
  if [[ -n "${EXPECTED_PROMPT+x}" ]]; then
    snap_env+=("EXPECTED_PROMPT=${EXPECTED_PROMPT}")
  fi
  env "${snap_env[@]}" \
  "${SYNC_PYTHON}" - "${FEATURE_ROOT}" "${FEATURE_SCAN_ROOT}" "snapshot" \
    >"${snapshot}" 2>"${invalid}" <<'PY'
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

trust = os.environ.get("SYNC_TRUST_EXTRACT", "1") == "1"
if not trust:
    import numpy as np
    from safetensors import safe_open

    PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from features.noise_seed import (  # noqa: E402
        DETERMINISTIC_NOISE_CONTRACT,
        MINIMAX_FIXED_NOISE_SEED,
        MINIMAX_NOISE_SEED_SCHEME,
        NOISE_SEED_SCHEME,
        canonical_noise_sample,
    )

SEEDED_PROTOCOL = "215c720c8537f4dfae6243bedcc7339fbb7e3ba8a6ca2dcaf04f49cd4b738e4f"
PROTOCOL = "inscene_streaming_canonical_v2_obs28_h124"
VFM_INPUT = {
    "wan": {"prompt": ".", "resize": [480, 832], "num_frames": 81},
    "cogvideox": {"prompt": "", "resize": [480, 720], "num_frames": 97},
    "minimax_h3": {"prompt": ".", "resize": [480, 832], "num_frames": 73},
}


class SnapshotError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signature(path: Path) -> list[int]:
    stat = path.stat()
    return [int(stat.st_dev), int(stat.st_ino), int(stat.st_size)]


def load_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SnapshotError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise SnapshotError(f"{label} is not an object: {path}")
    return value


def relative_to(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise SnapshotError(f"path escapes feature root: {path}") from exc


def row(rel: str, path: Path, sidecar: Path | None, *, unit: str, layer: int, structural: bool) -> dict:
    if path.is_symlink() or not path.is_file():
        raise SnapshotError(f"missing/non-regular {'structural index' if structural else 'feature'}: {rel}")
    payload = {
        "rel": rel,
        "sft_stat": signature(path),
        "sidecar_stat": signature(sidecar) if sidecar is not None else None,
        "unit": unit,
        "layer": int(layer),
        "structural": bool(structural),
        "keep_local": bool(structural),
    }
    return payload


root = Path(os.environ["FEATURE_ROOT"]).resolve()
scan_root = Path(os.environ["FEATURE_SCAN_ROOT"]).resolve()
kind = os.environ["FEATURE_KIND"]
namespace = os.environ["FEATURE_NAMESPACE"]
vfm = os.environ["VFM"]
timestep = int(os.environ["T"])
layers = [int(item) for item in os.environ["LAYERS"].split()]
done = {
    line.strip()
    for line in Path(os.environ["EXTRACT_DONE_INDEX"]).read_text(encoding="utf-8").splitlines()
    if line.strip()
}
expected_model = os.environ.get("EXPECTED_MODEL_ID") or None
expected_prompt = os.environ["EXPECTED_PROMPT"] if "EXPECTED_PROMPT" in os.environ else None
contract = VFM_INPUT[vfm]
seeded = namespace.endswith(("_CANONICAL_V2_SEEDED_V1", "_CANONICAL_V2_SEEDED_V1_MINIMAX_BF16_V1"))
bf16_contract = namespace.endswith("_MINIMAX_BF16_V1")
mode = "streaming_prefix" if kind == "FEAT_STREAMING" else "target_isolated"
feat_pat = f"feature_t{timestep}_layer"
units: dict[str, dict[int, Path]] = defaultdict(dict)
errors: list[str] = []
# Units whose features are complete but whose scene-level structural index has
# not been written yet.  The extractor writes prefix_index.npy (streaming) and
# target_indices.* (target) only after the scene's last feature, so while it
# runs this is "not yet publishable", not invalid.  The bash loop fails the run
# if any unit is still pending after extraction has stopped.
pending: list[str] = []

for dirpath, dirnames, filenames in os.walk(scan_root, followlinks=False):
    dirnames[:] = [name for name in dirnames if not name.startswith(".tstep_sync_staging_")]
    parent = Path(dirpath)
    for name in filenames:
        if "tmp" in name.lower() or not name.endswith(".sft"):
            continue
        if not name.startswith(feat_pat) or not name.endswith(".sft"):
            continue
        try:
            layer = int(name[len(feat_pat) : -4])
        except ValueError:
            continue
        if layer not in layers:
            continue
        path = parent / name
        if path.is_symlink() or not path.is_file():
            continue
        rel_parent = relative_to(root, parent)
        if kind == "FEAT_STREAMING":
            parts = Path(rel_parent).parts
            if len(parts) < 3 or not parts[-2].startswith("window_") or not parts[-1].startswith("prefix_"):
                continue
            unit = rel_parent
        else:
            unit = Path(rel_parent).parts[0] if rel_parent != "." else rel_parent
            if Path(rel_parent).parts[1:]:
                continue
        units[unit][layer] = path

wanted = set(layers)
snapshot: list[dict] = []
ready_bytes = 0

MINIMAX_BF16_STORAGE_CONTRACT = "MINIMAX_H3_FINITE_BF16_V1"


def validate_storage_contract(meta: dict, stored_dtype: str, path: Path) -> None:
    """Fail closed on the storage dtype a namespace promises.

    ``*_MINIMAX_BF16_V1`` namespaces hold finite bf16 payloads (fp16 overflowed
    to inf for MiniMax-H3 layers >= 20); every other namespace keeps fp16.
    """
    if bf16_contract:
        if meta.get("feature_storage_contract") != MINIMAX_BF16_STORAGE_CONTRACT:
            raise SnapshotError(f"feature sidecar storage contract mismatch: {path.name}")
        if meta.get("feature_storage_dtype") != "torch.bfloat16":
            raise SnapshotError(f"feature sidecar storage dtype mismatch: {path.name}")
        if stored_dtype not in {"BF16", "BFLOAT16"}:
            raise SnapshotError(f"bf16 namespace holds {stored_dtype} payload: {path.name}")
        if not meta.get("finite_check"):
            raise SnapshotError(f"feature sidecar lacks a finite_check record: {path.name}")
    elif stored_dtype not in {"F16", "FLOAT16"}:
        raise SnapshotError(f"fp16 namespace holds {stored_dtype} payload: {path.name}")


def validate_feature(path: Path, sidecar: Path, *, scene: str, window: int | None, tail: int | None) -> None:
    meta = load_json(sidecar, "feature sidecar")
    digest = sha256_file(path)
    size = path.stat().st_size
    if meta.get("sha256") != digest:
        raise SnapshotError(f"feature sidecar SHA256 mismatch: {path.name}")
    if meta.get("size_bytes") != size or meta.get("file") != path.name:
        raise SnapshotError(f"feature sidecar file/size mismatch: {path.name}")
    if expected_model is not None and meta.get("model_id") != expected_model:
        raise SnapshotError(f"feature sidecar model_id {meta.get('model_id')!r} != {expected_model!r}")
    if meta.get("vfm") != vfm:
        raise SnapshotError(f"feature sidecar vfm mismatch: {path.name}")
    if meta.get("mode") != mode:
        raise SnapshotError(f"feature sidecar mode mismatch: {path.name}")
    if int(meta.get("timestep", -1)) != timestep:
        raise SnapshotError(f"feature sidecar timestep mismatch: {path.name}")
    if expected_prompt is not None and meta.get("prompt") != expected_prompt:
        raise SnapshotError(f"feature sidecar prompt mismatch: {path.name}")
    if meta.get("prompt") != contract["prompt"] or meta.get("resize") != contract["resize"] or meta.get("num_frames") != contract["num_frames"]:
        raise SnapshotError(f"feature sidecar input contract mismatch: {path.name}")
    if meta.get("canonical_protocol") != PROTOCOL or int(meta.get("canonical_obs_len", -1)) != 28:
        raise SnapshotError(f"feature sidecar protocol mismatch: {path.name}")
    if seeded and meta.get("canonical_protocol_sha256") != SEEDED_PROTOCOL:
        raise SnapshotError("seeded-v1 canonical protocol SHA256 mismatch")
    if meta.get("feature_namespace") not in (None, namespace) and meta.get("feature_namespace") != namespace:
        raise SnapshotError(f"feature sidecar feature_namespace {meta.get('feature_namespace')!r}")
    if seeded and meta.get("feature_namespace") != namespace:
        raise SnapshotError(f"feature sidecar feature_namespace {meta.get('feature_namespace')!r}")
    try:
        with safe_open(str(path), framework="np", device="cpu") as handle:
            if "feat" not in handle.keys():
                raise SnapshotError(f"safetensors lacks feat tensor: {path.name}")
            feat_view = handle.get_slice("feat")
            shape = list(feat_view.get_shape())
            stored_dtype = str(feat_view.get_dtype()).upper()
    except SnapshotError:
        raise
    except Exception as exc:
        raise SnapshotError(f"invalid safetensors {path.name}: {exc}") from exc
    if meta.get("tensors", {}).get("feat", {}).get("shape") != shape:
        raise SnapshotError(f"safetensors shape/sidecar mismatch: {path.name}")
    validate_storage_contract(meta, stored_dtype, path)
    if not seeded:
        return
    if meta.get("feature_generation_contract") != DETERMINISTIC_NOISE_CONTRACT:
        raise SnapshotError(f"feature sidecar generation contract mismatch: {path.name}")
    if vfm == "minimax_h3":
        seed = meta.get("noise_seed")
        if type(seed) is not int or seed != MINIMAX_FIXED_NOISE_SEED:
            raise SnapshotError("expected integer 0")
        if meta.get("noise_seed_scheme") != MINIMAX_NOISE_SEED_SCHEME:
            raise SnapshotError("feature sidecar noise_seed_scheme mismatch")
        return
    protocol = {
        "protocol": PROTOCOL,
        "protocol_sha256": SEEDED_PROTOCOL,
    }
    if mode == "streaming_prefix":
        expected = [
            canonical_noise_sample(
                protocol=protocol,
                vfm=vfm,
                model_id=str(meta.get("model_id")),
                scene=scene,
                sample_kind="streaming_prefix",
                window_id=window,
                prefix_len=(tail or 0) + 1,
            )
        ]
    else:
        raise SnapshotError("target seed validation requires target indices")
    if meta.get("noise_seed_samples") != expected:
        raise SnapshotError("feature sidecar noise_seed_samples mismatch")


def validate_prefix(path: Path, sidecar: Path, *, window: int, tail: int) -> None:
    if path.is_symlink() or not path.is_file():
        raise SnapshotError("missing/non-regular structural index")
    if sidecar.is_symlink() or not sidecar.is_file():
        raise SnapshotError("missing/non-regular structural index")
    if trust:
        return
    meta = load_json(sidecar, "structural sidecar")
    if seeded and meta.get("feature_namespace") != namespace:
        raise SnapshotError(f"structural sidecar feature_namespace {meta.get('feature_namespace')!r}")
    if seeded and meta.get("canonical_protocol_sha256") != SEEDED_PROTOCOL:
        raise SnapshotError("seeded-v1 canonical protocol SHA256 mismatch")
    if meta.get("sha256") != sha256_file(path) or meta.get("size_bytes") != path.stat().st_size:
        raise SnapshotError("structural sidecar SHA256 mismatch")
    try:
        records = np.load(path, allow_pickle=True).tolist()
    except Exception as exc:
        raise SnapshotError(f"cannot load prefix index: {exc}") from exc
    found = False
    for raw in records:
        if not isinstance(raw, dict):
            continue
        if int(raw.get("window_id", -1)) == window and int(raw.get("tail", -999)) == tail:
            found = True
            if seeded and raw.get("canonical_protocol_sha256") != SEEDED_PROTOCOL:
                raise SnapshotError("seeded-v1 canonical protocol SHA256 mismatch")
            break
    if not found:
        raise SnapshotError(f"prefix record missing for window={window} tail={tail}")


def validate_target_structure(scene_dir: Path, feature_paths: list[Path]) -> None:
    index = scene_dir / "target_indices.npy"
    sidecar = Path(f"{index}.manifest.json")
    meta_path = scene_dir / "target_indices.meta.json"
    if index.is_symlink() or not index.is_file() or sidecar.is_symlink() or not sidecar.is_file() or meta_path.is_symlink() or not meta_path.is_file():
        raise SnapshotError("missing/non-regular structural index")
    if trust:
        return
    index_meta = load_json(sidecar, "structural sidecar")
    if seeded and index_meta.get("feature_namespace") != namespace:
        raise SnapshotError(f"structural sidecar feature_namespace {index_meta.get('feature_namespace')!r}")
    if index_meta.get("sha256") != sha256_file(index):
        raise SnapshotError("structural sidecar SHA256 mismatch")
    meta = load_json(meta_path, "target meta")
    if seeded and meta.get("protocol_sha256") != SEEDED_PROTOCOL:
        raise SnapshotError("seeded-v1 canonical protocol SHA256 mismatch")
    targets = np.load(index).astype(np.int64).tolist()
    if meta.get("target_indices_sha256") != sha256_file(index):
        raise SnapshotError("target meta digest mismatch")
    protocol = {"protocol": PROTOCOL, "protocol_sha256": SEEDED_PROTOCOL}
    expected_samples = [
        canonical_noise_sample(
            protocol=protocol,
            vfm=vfm,
            model_id=str(expected_model or ""),
            scene=scene_dir.name,
            sample_kind="target_isolated",
            frame_index=int(frame),
        )
        for frame in targets
    ] if seeded and vfm in {"wan", "cogvideox"} else None
    for path in feature_paths:
        sidecar_path = Path(f"{path}.manifest.json")
        meta_feat = load_json(sidecar_path, "feature sidecar")
        digest = sha256_file(path)
        if meta_feat.get("sha256") != digest:
            raise SnapshotError(f"feature sidecar SHA256 mismatch: {path.name}")
        if expected_model is not None and meta_feat.get("model_id") != expected_model:
            raise SnapshotError(f"feature sidecar model_id {meta_feat.get('model_id')!r}")
        if seeded and meta_feat.get("canonical_protocol_sha256") != SEEDED_PROTOCOL:
            raise SnapshotError("seeded-v1 canonical protocol SHA256 mismatch")
        if seeded and meta_feat.get("feature_namespace") != namespace:
            raise SnapshotError(f"feature sidecar feature_namespace {meta_feat.get('feature_namespace')!r}")
        with safe_open(str(path), framework="np", device="cpu") as handle:
            feat_view = handle.get_slice("feat")
            shape = list(feat_view.get_shape())
            stored_dtype = str(feat_view.get_dtype()).upper()
        if shape[0] != len(targets):
            raise SnapshotError(f"target feature row count mismatch: {path.name}")
        validate_storage_contract(meta_feat, stored_dtype, path)
        if expected_samples is not None and meta_feat.get("noise_seed_samples") != expected_samples:
            raise SnapshotError("feature sidecar noise_seed_samples mismatch")
        if vfm == "minimax_h3" and seeded:
            seed = meta_feat.get("noise_seed")
            if type(seed) is not int or seed != MINIMAX_FIXED_NOISE_SEED:
                raise SnapshotError("expected integer 0")


for unit, layer_map in sorted(units.items()):
    if set(layer_map) != wanted:
        continue
    keys = [relative_to(root, layer_map[layer]) for layer in layers]
    if keys and set(keys).issubset(done):
        continue
    try:
        if kind == "FEAT_STREAMING":
            parts = Path(unit).parts
            scene = parts[0]
            window = int(parts[-2].split("_", 1)[1])
            tail = int(parts[-1].split("_", 1)[1])
            index = root / scene / "prefix_index.npy"
            index_side = Path(f"{index}.manifest.json")
            if not (index.exists() or index.is_symlink()) or not (index_side.exists() or index_side.is_symlink()):
                pending.append(f"{unit}: awaiting {relative_to(root, index)} (missing/non-regular structural index)")
                continue
            if index.is_symlink() or not index.is_file() or index_side.is_symlink() or not index_side.is_file():
                raise SnapshotError("missing/non-regular structural index")
            for layer in layers:
                path = layer_map[layer]
                sidecar = Path(f"{path}.manifest.json")
                if sidecar.is_symlink() or not sidecar.is_file():
                    raise SnapshotError(f"missing feature sidecar: {path.name}")
                if not trust:
                    validate_feature(path, sidecar, scene=scene, window=window, tail=tail)
            if not trust:
                validate_prefix(index, index_side, window=window, tail=tail)
            for layer in layers:
                path = layer_map[layer]
                sidecar = Path(f"{path}.manifest.json")
                snapshot.append(row(relative_to(root, path), path, sidecar, unit=unit, layer=layer, structural=False))
                ready_bytes += path.stat().st_size
            snapshot.append(row(relative_to(root, index), index, index_side, unit=unit, layer=-1, structural=True))
        else:
            scene_dir = root / unit
            target_index = scene_dir / "target_indices.npy"
            structural = (
                target_index,
                Path(f"{target_index}.manifest.json"),
                scene_dir / "target_indices.meta.json",
            )
            if any(not (item.exists() or item.is_symlink()) for item in structural):
                pending.append(f"{unit}: awaiting target_indices (missing/non-regular structural index)")
                continue
            paths = [layer_map[layer] for layer in layers]
            for path in paths:
                sidecar = Path(f"{path}.manifest.json")
                if sidecar.is_symlink() or not sidecar.is_file():
                    raise SnapshotError(f"missing feature sidecar: {path.name}")
            validate_target_structure(scene_dir, paths)
            for layer in layers:
                path = layer_map[layer]
                sidecar = Path(f"{path}.manifest.json")
                if not trust:
                    # streaming-style checks that validate_target_structure already did for seeds
                    pass
                snapshot.append(row(relative_to(root, path), path, sidecar, unit=unit, layer=layer, structural=False))
                ready_bytes += path.stat().st_size
            index = scene_dir / "target_indices.npy"
            index_side = Path(f"{index}.manifest.json")
            meta_path = scene_dir / "target_indices.meta.json"
            snapshot.append(row(relative_to(root, index), index, index_side, unit=unit, layer=-1, structural=True))
            snapshot.append(row(relative_to(root, meta_path), meta_path, None, unit=unit, layer=-2, structural=True))
    except SnapshotError as exc:
        errors.append(str(exc))

pending_file = os.environ.get("SYNC_PENDING_FILE")
if pending_file:
    Path(pending_file).write_text("".join(f"{item}\n" for item in pending), encoding="utf-8")
if errors:
    sys.stderr.write("".join(f"{item}\n" for item in errors))
    raise SystemExit(3)
for item in snapshot:
    item["ready_bytes"] = ready_bytes
    print(json.dumps(item, sort_keys=True))
PY
}

stage_snapshot() {
  local snapshot="$1" staging="$2"
  FEATURE_ROOT="${FEATURE_ROOT}" \
  "${SYNC_PYTHON}" - "${FEATURE_ROOT}" "${snapshot}" "${staging}" "stage" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
snapshot = Path(sys.argv[2])
staging = Path(sys.argv[3]).resolve()
rows = [json.loads(line) for line in snapshot.read_text(encoding="utf-8").splitlines() if line.strip()]
staging.mkdir(parents=True, exist_ok=True)

def signature(path: Path) -> list[int]:
    stat = path.stat()
    return [int(stat.st_dev), int(stat.st_ino), int(stat.st_size)]

def link_one(src: Path, dest: Path, expected: list[int] | None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    if src.is_symlink() or not src.is_file():
        raise SystemExit("source disappeared before hardlink staging")
    if expected is not None and signature(src) != expected:
        print("source changed during hardlink staging; safely deferring", flush=True)
        raise SystemExit(75)
    os.link(src, dest)
    if expected is not None and signature(dest) != expected:
        print("source changed during hardlink staging; safely deferring", flush=True)
        raise SystemExit(75)

for row in rows:
    rel = row["rel"]
    if Path(rel).is_absolute() or ".." in Path(rel).parts:
        raise SystemExit(f"unsafe snapshot path: {rel}")
    src = root / rel
    dest = staging / rel
    link_one(src, dest, row.get("sft_stat"))
    sidecar_stat = row.get("sidecar_stat")
    if sidecar_stat is not None:
        link_one(Path(f"{src}.manifest.json"), Path(f"{dest}.manifest.json"), sidecar_stat)
PY
}

changed_units() {
  local snapshot="$1"
  FEATURE_ROOT="${FEATURE_ROOT}" \
  "${SYNC_PYTHON}" - "${FEATURE_ROOT}" "${snapshot}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
rows = [json.loads(line) for line in Path(sys.argv[2]).read_text(encoding="utf-8").splitlines() if line.strip()]

def signature(path: Path) -> list[int] | None:
    if not path.is_file() or path.is_symlink():
        return None
    stat = path.stat()
    return [int(stat.st_dev), int(stat.st_ino), int(stat.st_size)]

changed = set()
stable = set()
for row in rows:
    unit = row["unit"]
    src = root / row["rel"]
    if signature(src) != row.get("sft_stat"):
        changed.add(unit)
        continue
    sidecar_stat = row.get("sidecar_stat")
    if sidecar_stat is not None and signature(Path(f"{src}.manifest.json")) != sidecar_stat:
        changed.add(unit)
        continue
    stable.add(unit)
for unit in sorted(changed):
    print(unit)
if changed and not (stable - changed):
    raise SystemExit(17)
PY
}

parse_ads() {
  local log="$1"
  local found=0 failed=0
  local line
  line="$(tr -d '\000' < "${log}" | tr -d '\r' | grep -E 'Found:' | tail -1 || true)"
  if [[ -n "${line}" ]]; then
    found="$(sed -n 's/.*Found:[[:space:]]*\([0-9][0-9]*\).*/\1/p' <<<"${line}")"
    failed="$(sed -n 's/.*failed:[[:space:]]*\([0-9][0-9]*\).*/\1/p' <<<"${line}")"
  fi
  printf '%s %s\n' "${found:-0}" "${failed:-0}"
}

merge_done_index() {
  local keys_file="$1"
  local tmp="${EXTRACT_DONE_INDEX}.tmp-$$"
  {
    cat -- "${EXTRACT_DONE_INDEX}"
    cat -- "${keys_file}"
  } | LC_ALL=C sort -u > "${tmp}"
  mv -f -- "${tmp}" "${EXTRACT_DONE_INDEX}"
}

delete_published() {
  local snapshot="$1" changed_file="$2"
  FEATURE_ROOT="${FEATURE_ROOT}" \
  "${SYNC_PYTHON}" - "${FEATURE_ROOT}" "${snapshot}" "${changed_file}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
changed = {
    line.strip()
    for line in Path(sys.argv[3]).read_text(encoding="utf-8").splitlines()
    if line.strip()
}
rows = [json.loads(line) for line in Path(sys.argv[2]).read_text(encoding="utf-8").splitlines() if line.strip()]
for row in rows:
    if row.get("keep_local") or row.get("structural") or row["unit"] in changed:
        continue
    path = root / row["rel"]
    sidecar = Path(f"{path}.manifest.json")
    if path.is_file() and not path.is_symlink():
        path.unlink()
    if sidecar.is_file() and not sidecar.is_symlink():
        sidecar.unlink()
PY
}

snapshot_keys() {
  local snapshot="$1" keys="$2"
  FEATURE_KIND="${FEATURE_KIND}" \
  GAP_ADAPTER="${GAP_ADAPTER}" \
  "${SYNC_PYTHON}" - "${snapshot}" "${keys}" <<'PY'
import json
import os
import sys
from pathlib import Path

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if line.strip()]
gap = bool(os.environ.get("GAP_ADAPTER"))
keys = []
for row in rows:
    rel = row["rel"]
    if gap or rel.endswith(".sft"):
        keys.append(rel)
Path(sys.argv[2]).write_text("".join(f"{item}\n" for item in keys), encoding="utf-8")
PY
}

ready_bytes_from_snapshot() {
  local snapshot="$1"
  "${SYNC_PYTHON}" - "${snapshot}" <<'PY'
import json
import sys
from pathlib import Path
rows = [json.loads(line) for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if line.strip()]
if not rows:
    print(0)
elif "ready_bytes" in rows[0]:
    print(int(rows[0]["ready_bytes"]))
else:
    print(sum(int((row.get("sft_stat") or [0, 0, 0])[2]) for row in rows if not row.get("structural")))
PY
}

commit_gap_receipt() {
  local snapshot="$1" keys="$2" round="$3" ads_rc="$4" found="$5" expected="$6" failed="$7"
  local remote="s3://${BUCKET}.${AOSS_API_HOST}/InsScene-15K/${FEATURE_NAMESPACE}/${VFM}/${SOURCE}/"
  "${SYNC_PYTHON}" "${GAP_ADAPTER}" commit-receipt \
    --receipt "${GAP_REMOTE_RECEIPT}" \
    --shard "${GAP_SHARD_PATH}" \
    --shard-file-sha256 "${GAP_SHARD_FILE_SHA256}" \
    --shard-content-sha256 "${GAP_SHARD_CONTENT_SHA256}" \
    --snapshot "${snapshot}" \
    --keys "${keys}" \
    --done-index "${EXTRACT_DONE_INDEX}" \
    --round "${round}" \
    --ads-rc "${ads_rc}" \
    --ads-found "${found}" \
    --ads-expected "${expected}" \
    --ads-failed "${failed}" \
    --remote-base-uri "${remote}"
  "${SYNC_PYTHON}" "${GAP_ADAPTER}" cleanup-confirmed \
    --root "${FEATURE_ROOT}" \
    --worklist "${CANONICAL_GAP_WORKLIST}" \
    --worklist-sha256 "${CANONICAL_GAP_WORKLIST_SHA256}" \
    --receipt "${GAP_REMOTE_RECEIPT}" \
    --shard-content-sha256 "${GAP_SHARD_CONTENT_SHA256}"
}

round=0
consecutive_failures=0
final_failures=0
last_sync_epoch=0
stopped=0

while true; do
  if [[ -e "${STOP_FILE}" ]]; then
    stopped=1
  fi
  if [[ -n "${GAP_ADAPTER}" && -f "${GAP_REMOTE_RECEIPT:-}" ]]; then
    "${SYNC_PYTHON}" "${GAP_ADAPTER}" cleanup-confirmed \
      --root "${FEATURE_ROOT}" \
      --worklist "${CANONICAL_GAP_WORKLIST}" \
      --worklist-sha256 "${CANONICAL_GAP_WORKLIST_SHA256}" \
      --receipt "${GAP_REMOTE_RECEIPT}" \
      --shard-content-sha256 "${GAP_SHARD_CONTENT_SHA256}" \
      >/dev/null
  fi
  temps="${LOG_DIR}/atomic_temps.txt"
  scan_atomic_temps "${temps}"
  if [[ -s "${temps}" ]]; then
    if (( stopped )); then
      echo "before snapshot final atomic temp files remain; safely deferring"
      final_failures=$((final_failures + 1))
      if (( final_failures >= SYNC_FINAL_MAX_ATTEMPTS )); then
        echo "final sync failed ${final_failures} times; giving up" >&2
        exit 1
      fi
      sleep "${SYNC_RETRY_SEC}"
      continue
    fi
    sleep "${SYNC_POLL_SEC}"
    continue
  fi

  round=$((round + 1))
  printf -v round_id '%04d' "${round}"
  snapshot="${LOG_DIR}/round_${round_id}.snapshot.jsonl"
  invalid="${LOG_DIR}/round_${round_id}.invalid.txt"
  : > "${snapshot}"
  : > "${invalid}"
  set +e
  snapshot_candidates "${snapshot}" "${invalid}"
  snap_rc=$?
  set -e
  if [[ -s "${invalid}" ]]; then
    cat "${invalid}"
  fi
  if (( snap_rc == 3 || snap_rc != 0 )); then
    if [[ -s "${invalid}" ]] || (( snap_rc == 3 )); then
      exit 1
    fi
    echo "ERROR: snapshot failed rc=${snap_rc}" >&2
    exit 1
  fi
  if [[ -s "${invalid}" && -z "${GAP_ADAPTER}" ]]; then
    exit 1
  fi
  # Stream units need feature + prefix_index. Extract writes the feature first,
  # so a live snapshot can be incomplete-only. Fail-closing here killed LingBot
  # mid-load. After extract stops, leftover scraps (index without payload) are
  # not publishable; treat that as drained rather than FAILED.
  if [[ -n "${GAP_ADAPTER}" && -s "${invalid}" && ! -s "${snapshot}" ]]; then
    n_invalid=$(wc -l < "${invalid}" | tr -d ' ')
    if (( stopped )); then
      echo "gap snapshot produced only incomplete units (${n_invalid}); nothing publishable remains"
      exit 0
    fi
    echo "gap snapshot has ${n_invalid} incomplete units and 0 complete; waiting for extract"
    sleep "${SYNC_POLL_SEC}"
    continue
  fi

  pending_units="${snapshot%.snapshot.jsonl}.pending.txt"
  if [[ ! -s "${snapshot}" ]]; then
    if (( stopped )); then
      if [[ -s "${pending_units}" ]]; then
        echo "ERROR: extraction stopped but $(wc -l < "${pending_units}") unit(s) never got their structural index:" >&2
        head -20 -- "${pending_units}" >&2
        exit 1
      fi
      echo "sync complete; no remaining publishable units"
      exit 0
    fi
    sleep "${SYNC_POLL_SEC}"
    continue
  fi

  ready_bytes="$(ready_bytes_from_snapshot "${snapshot}")"
  now="$(date +%s)"
  elapsed=$((now - last_sync_epoch))
  threshold_bytes=$((SYNC_EVERY_GIB * 1024 * 1024 * 1024))
  if (( !stopped )); then
    if (( SYNC_EVERY_SEC > 0 && elapsed < SYNC_EVERY_SEC && ready_bytes < threshold_bytes )); then
      sleep "${SYNC_POLL_SEC}"
      continue
    fi
    if (( SYNC_EVERY_GIB > 0 && ready_bytes < threshold_bytes && elapsed < SYNC_EVERY_SEC )); then
      sleep "${SYNC_POLL_SEC}"
      continue
    fi
  fi

  echo "round=${round_id} candidates=$(wc -l < "${snapshot}") ready_bytes=${ready_bytes}"
  staging="${FEATURE_ROOT}/.tstep_sync_staging_${round_id}_$$"
  rm -rf -- "${staging}"
  set +e
  stage_snapshot "${snapshot}" "${staging}"
  stage_rc=$?
  set -e
  if (( stage_rc == 75 )); then
    rm -rf -- "${staging}"
    if (( stopped )); then
      final_failures=$((final_failures + 1))
      if (( final_failures >= SYNC_FINAL_MAX_ATTEMPTS )); then
        echo "final sync failed ${final_failures} times; giving up" >&2
        exit 1
      fi
    fi
    sleep "${SYNC_RETRY_SEC}"
    continue
  fi
  if (( stage_rc != 0 )); then
    rm -rf -- "${staging}"
    echo "ERROR: hardlink staging failed rc=${stage_rc}" >&2
    exit 1
  fi

  read_ads_credentials
  dest="s3://${AK}:${SK}@${AOSS_HOST}/InsScene-15K/${FEATURE_NAMESPACE}/${VFM}/${SOURCE}/"
  ads_log="${LOG_DIR}/round_${round_id}.ads.log"
  set +e
  "${ADS_CLI}" -p "${ADS_THREADS}" -l "${ADS_LISTERS}" --conntimeout 120 --timeout 600 \
    sync "${staging}/" "${dest}" >"${ads_log}" 2>&1
  ads_rc=$?
  set -e
  read -r found failed < <(parse_ads "${ads_log}")
  expected="$(find "${staging}" -type f | wc -l | tr -d ' ')"
  cat "${ads_log}"
  rm -rf -- "${staging}"
  if (( ads_rc != 0 || failed != 0 || found != expected )); then
    echo "upload coverage not proven found=${found} expected=${expected} failed=${failed} ads_rc=${ads_rc}"
    if (( stopped )); then
      final_failures=$((final_failures + 1))
      if (( final_failures >= SYNC_FINAL_MAX_ATTEMPTS )); then
        echo "final sync failed ${final_failures} times; giving up" >&2
        exit 1
      fi
      sleep "${SYNC_RETRY_SEC}"
      continue
    fi
    consecutive_failures=$((consecutive_failures + 1))
    if (( consecutive_failures >= SYNC_MAX_FAILURES )); then
      echo "sync failed ${consecutive_failures} consecutive times" >&2
      exit 1
    fi
    sleep "${SYNC_RETRY_SEC}"
    continue
  fi

  changed_file="${LOG_DIR}/round_${round_id}.changed.txt"
  : > "${changed_file}"
  set +e
  changed_units "${snapshot}" > "${changed_file}"
  change_rc=$?
  set -e
  if (( change_rc == 17 )); then
    echo "all snapshotted units changed; retaining sources"
    if (( stopped )); then
      final_failures=$((final_failures + 1))
      if (( final_failures >= SYNC_FINAL_MAX_ATTEMPTS )); then
        echo "final sync failed ${final_failures} times; giving up" >&2
        exit 1
      fi
    fi
    sleep "${SYNC_RETRY_SEC}"
    continue
  fi

  keys_file="${LOG_DIR}/round_${round_id}.keys.txt"
  snapshot_keys "${snapshot}" "${keys_file}"
  if [[ -s "${changed_file}" ]]; then
    FEATURE_ROOT="${FEATURE_ROOT}" \
    "${SYNC_PYTHON}" - "${keys_file}" "${snapshot}" "${changed_file}" <<'PY'
import json
import sys
from pathlib import Path
keys = Path(sys.argv[1])
changed = {line.strip() for line in Path(sys.argv[3]).read_text().splitlines() if line.strip()}
rows = [json.loads(line) for line in Path(sys.argv[2]).read_text().splitlines() if line.strip()]
keep = [row["rel"] for row in rows if row["unit"] not in changed and row["rel"].endswith(".sft")]
keys.write_text("".join(f"{item}\n" for item in keep), encoding="utf-8")
PY
  fi
  if [[ ! -s "${keys_file}" ]]; then
    echo "all snapshotted units changed; retaining sources"
    sleep "${SYNC_RETRY_SEC}"
    continue
  fi

  set +e
  merge_done_index "${keys_file}"
  merge_rc=$?
  set -e
  if (( merge_rc != 0 )); then
    echo "done-index merge failed; retaining all source pairs" >&2
    exit 1
  fi

  if [[ -n "${GAP_ADAPTER}" ]]; then
    commit_gap_receipt "${snapshot}" "${keys_file}" "${round_id}" "${ads_rc}" "${found}" "${expected}" "${failed}"
  else
    delete_published "${snapshot}" "${changed_file}"
  fi
  last_sync_epoch="$(date +%s)"
  consecutive_failures=0
  if (( stopped )); then
    # Keep draining remaining units after a successful final-round publish.
    continue
  fi
done
