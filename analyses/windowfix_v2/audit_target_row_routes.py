#!/usr/bin/env python3
"""Audit raw-ID routes for a canonical-v2 aggregate target merge.

The input plan names a frozen canonical p24/obs28 source and, for every
physical target layer/scene, zero or more aggregate source payloads.  Sources
may be legacy target caches or gap-only delta files produced from a
``CANONICAL_V2_GAP_WORKLIST_V1`` worklist.

This program is read-only with respect to feature roots.  It validates source
indexes, safetensors headers, sidecar identity, and exact raw-ID ownership.  It
emits auditable row routes plus gap worklists; it never copies tensors and it
never writes a ``*_SEEDED_V1`` namespace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import struct
import sys
import tempfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyses.windowfix_v2.target_row_source import (
    RemoteSourcePending,
    SourceReadError,
    basename as source_basename,
    is_remote as source_is_remote,
    object_size as source_object_size,
    read_json as read_source_json,
    read_npy_int64,
    sha256_object,
    tensor_layout,
)


SCHEMA_VERSION = 1
CONTRACT = "CANONICAL_V2_TARGET_ROW_MERGE_V1"
PLAN_TYPE = "canonical_v2_target_row_merge_plan"
AUDIT_TYPE = "canonical_v2_target_row_route_audit"
GAP_CONTRACT = "CANONICAL_V2_GAP_WORKLIST_V1"
GAP_TARGET_NAMESPACE = "FEAT_TARGET_CANONICAL_V2_GAP_DELTA_V1"
CANONICAL_PROTOCOL = "inscene_streaming_canonical_v2_obs28_h124"
MIXED_TARGET_NAMESPACE = "FEAT_TARGET_CANONICAL_V2_MIXED_V1"
PREFIXES = (8, 12, 16, 24)
HORIZONS = (1, 2, 4)
HEX = frozenset("0123456789abcdef")
DINO_PREFIX_SOURCE_PRIORITIES = {
    "dino_prefix24_aoss": 0,
    "dino_prefix16_aoss": 1,
    "dino_prefix12_aoss": 2,
    "dino_prefix8_aoss": 3,
}

_DTYPE_ALIASES = {
    "F16": "torch.float16",
    "float16": "torch.float16",
    "torch.float16": "torch.float16",
    "BF16": "torch.bfloat16",
    "bfloat16": "torch.bfloat16",
    "torch.bfloat16": "torch.bfloat16",
    "F32": "torch.float32",
    "float32": "torch.float32",
    "torch.float32": "torch.float32",
    "F64": "torch.float64",
    "float64": "torch.float64",
    "torch.float64": "torch.float64",
    "I64": "torch.int64",
    "int64": "torch.int64",
    "torch.int64": "torch.int64",
    "I32": "torch.int32",
    "int32": "torch.int32",
    "torch.int32": "torch.int32",
    "I16": "torch.int16",
    "int16": "torch.int16",
    "torch.int16": "torch.int16",
    "I8": "torch.int8",
    "int8": "torch.int8",
    "torch.int8": "torch.int8",
    "U8": "torch.uint8",
    "uint8": "torch.uint8",
    "torch.uint8": "torch.uint8",
    "BOOL": "torch.bool",
    "bool": "torch.bool",
    "torch.bool": "torch.bool",
}
_DTYPE_BYTES = {
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}


class RouteAuditError(ValueError):
    """Raised when the merge plan itself is unsafe or malformed."""


class SourceGateTimeout(BaseException):
    """One remote source gate exceeded its operator-supplied deadline.

    It bypasses ordinary HTTP-client ``except Exception`` retry handlers and
    is caught explicitly by the outer audit loop.  Otherwise a client may
    swallow the first alarm and continue blocking indefinitely.
    """


@contextmanager
def _source_deadline(seconds: float | None):
    """Interrupt a blocking source gate on Unix main threads.

    The CLI defaults to not touching remote sources at all.  This deadline is
    an additional fail-closed guard for ``--remote-policy gate`` workers.
    """
    if seconds is None or seconds <= 0 or not hasattr(signal, "setitimer"):
        yield
        return
    try:
        previous_handler = signal.getsignal(signal.SIGALRM)
        previous_timer = signal.getitimer(signal.ITIMER_REAL)
    except (ValueError, AttributeError):
        # Non-main-thread library callers cannot install signal handlers.  The
        # read remains valid, but callers that require a hard deadline should
        # use the CLI in its own process.
        yield
        return

    def _timeout(_signum, _frame):
        raise SourceGateTimeout(f"remote source gate timed out after {seconds:g}s")

    signal.signal(signal.SIGALRM, _timeout)
    signal.siginterrupt(signal.SIGALRM, True)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.siginterrupt(signal.SIGALRM, False)
        signal.signal(signal.SIGALRM, previous_handler)


@dataclass(frozen=True)
class TensorInfo:
    shape: tuple[int, ...]
    dtype: str
    safetensors_dtype: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _is_sha(value: Any) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and set(text) <= HEX


def _regular(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise RouteAuditError(f"{label} must be an absolute regular non-symlink file: {path}")
    return path


def read_json(path: Path, label: str) -> dict[str, Any]:
    _regular(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RouteAuditError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RouteAuditError(f"{label} must contain a JSON object: {path}")
    return value


def _safe_component(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text or text in {".", ".."} or "/" in text or "\\" in text:
        raise RouteAuditError(f"unsafe {field}: {text!r}")
    return text


def _normal_dtype(value: Any) -> str:
    text = str(value or "")
    if text not in _DTYPE_ALIASES:
        raise RouteAuditError(f"unsupported tensor dtype: {text!r}")
    return _DTYPE_ALIASES[text]


def _lookup(value: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = value
    for component in dotted_key.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise KeyError(dotted_key)
        current = current[component]
    return current


def _reject_symlink_ancestors(path: Path, label: str) -> None:
    current = path
    while True:
        if current.exists() and current.is_symlink():
            raise RouteAuditError(f"{label} contains a symlink component: {current}")
        if current.parent == current:
            return
        current = current.parent


def validate_destination_root(value: Any) -> Path:
    root = Path(str(value or ""))
    if not root.is_absolute():
        raise RouteAuditError("destination_root must be an absolute local path")
    if root.name != MIXED_TARGET_NAMESPACE:
        raise RouteAuditError(
            f"destination_root must end in {MIXED_TARGET_NAMESPACE}, found {root}"
        )
    if any("SEEDED_V1" in component for component in root.parts):
        raise RouteAuditError("target row merge must never write a SEEDED_V1 namespace")
    _reject_symlink_ancestors(root, "destination_root")
    return root


def safetensors_tensor_info(path: Path, tensor_key: str) -> TensorInfo:
    """Validate one safetensors tensor header, offsets, shape, and byte span."""
    _regular(path, "feature payload")
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise RouteAuditError(f"truncated safetensors length: {path}")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size <= 1 or header_size > min(max(file_size - 8, 0), 128 * 1024 * 1024):
            raise RouteAuditError(f"invalid safetensors header length: {path}")
        try:
            header = json.loads(handle.read(header_size).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RouteAuditError(f"invalid safetensors header JSON: {path}") from exc
    tensor = header.get(tensor_key)
    if not isinstance(tensor, dict):
        raise RouteAuditError(f"safetensors tensor {tensor_key!r} is missing: {path}")
    shape_value = tensor.get("shape")
    if not isinstance(shape_value, list) or not shape_value:
        raise RouteAuditError(f"safetensors shape is missing: {path}")
    shape = tuple(int(item) for item in shape_value)
    if any(item < 0 for item in shape):
        raise RouteAuditError(f"safetensors shape contains a negative dimension: {path}")
    dtype_code = str(tensor.get("dtype", ""))
    if dtype_code not in _DTYPE_BYTES:
        raise RouteAuditError(f"unsupported safetensors dtype {dtype_code!r}: {path}")
    offsets = tensor.get("data_offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or any(not isinstance(item, int) for item in offsets)
        or offsets[0] < 0
        or offsets[1] < offsets[0]
    ):
        raise RouteAuditError(f"invalid safetensors data_offsets: {path}")
    data_area = file_size - 8 - header_size
    if offsets[1] > data_area:
        raise RouteAuditError(f"safetensors data_offsets exceed payload: {path}")
    elements = int(np.prod(shape, dtype=np.int64))
    if offsets[1] - offsets[0] != elements * _DTYPE_BYTES[dtype_code]:
        raise RouteAuditError(f"safetensors byte length disagrees with shape/dtype: {path}")
    return TensorInfo(shape, _normal_dtype(dtype_code), dtype_code)


def canonical_target_unions(
    source_path: Path, expected_sha256: str
) -> tuple[dict[tuple[str, str], tuple[int, ...]], dict[str, Any]]:
    _regular(source_path, "canonical p24 source")
    actual_sha = sha256_file(source_path)
    if not _is_sha(expected_sha256) or actual_sha != expected_sha256:
        raise RouteAuditError("canonical p24 source SHA256 mismatch")
    targets: dict[tuple[str, str], set[int]] = defaultdict(set)
    units: set[tuple[str, str, int]] = set()
    rows = 0
    with source_path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
                source = str(record["source"])
                scene = str(record["scene_id"])
                window_id = int(record["streaming_window_id"])
                window = tuple(int(value) for value in record["streaming_window_indices"])
                prefix24 = tuple(int(value) for value in record["streaming_indices"])
            except Exception as exc:
                raise RouteAuditError(
                    f"canonical p24 source line {line_number} is malformed: {exc}"
                ) from exc
            if len(window) != 28 or any(right <= left for left, right in zip(window, window[1:])):
                raise RouteAuditError(f"canonical source line {line_number} is not strict obs28")
            if len(prefix24) != 24 or prefix24 != window[:24]:
                raise RouteAuditError(f"canonical source line {line_number} is not p24=obs28[:24]")
            unit = (source, scene, window_id)
            if unit in units:
                raise RouteAuditError(f"duplicate canonical window: {unit}")
            units.add(unit)
            for prefix in PREFIXES:
                positions = (prefix - 1,) + tuple(prefix - 1 + horizon for horizon in HORIZONS)
                targets[(source, scene)].update(window[position] for position in positions)
            rows += 1
    frozen = {key: tuple(sorted(value)) for key, value in sorted(targets.items())}
    if not frozen:
        raise RouteAuditError("canonical p24 source is empty")
    universe_payload = [
        {"source": key[0], "scene_id": key[1], "target_raw_ids": list(values)}
        for key, values in frozen.items()
    ]
    metadata = {
        "path": str(source_path),
        "sha256": actual_sha,
        "windows": rows,
        "scenes": len(frozen),
        "target_rows": sum(len(value) for value in frozen.values()),
        "target_universe_sha256": canonical_json_sha(universe_payload),
    }
    return frozen, metadata


def _validate_sidecar_file_binding(
    sidecar: Mapping[str, Any], payload: str, *, verify_sha: bool
) -> tuple[str | None, bool]:
    if sidecar.get("file") not in (None, source_basename(payload)):
        raise RouteAuditError(f"sidecar file field disagrees with {payload}")
    size = sidecar.get("size_bytes")
    actual_size = source_object_size(payload, size)
    if size is not None and int(size) != actual_size:
        raise RouteAuditError(f"sidecar size_bytes disagrees with {payload}")
    declared = sidecar.get("sha256")
    if declared not in (None, "") and not _is_sha(declared):
        raise RouteAuditError(f"sidecar has malformed SHA256 for {payload}")
    # Full remote payload hashing would defeat row-range reuse.  Remote rows
    # are instead hashed individually by the merge writer.  Local payloads
    # retain the stronger whole-file gate.
    if verify_sha and not source_is_remote(payload):
        actual = sha256_object(payload)
        if declared not in (None, "") and actual != str(declared).lower():
            raise RouteAuditError(f"sidecar SHA256 disagrees with {payload}")
        return actual, True
    return (str(declared).lower() if _is_sha(declared) else None), False


def _validate_index_sidecar(
    path: str,
    sidecar_path: str,
    raw_ids: tuple[int, ...],
    *,
    verify_sha: bool,
) -> tuple[dict[str, Any], str | None, bool]:
    sidecar = read_source_json(sidecar_path, "target index sidecar")
    # Index objects are small and are always fully read, including on AOSS.
    digest, verified = _validate_sidecar_file_binding(sidecar, path, verify_sha=False)
    actual_digest = sha256_object(path)
    if digest is not None and digest != actual_digest:
        raise RouteAuditError(f"index sidecar SHA256 disagrees with {path}")
    digest, verified = actual_digest, True
    if sidecar.get("shape") is not None and sidecar.get("shape") != [len(raw_ids)]:
        raise RouteAuditError(f"index sidecar shape disagrees with {path}")
    if sidecar.get("dtype") is not None and str(sidecar["dtype"]) not in {
        "int64",
        "numpy.int64",
        "torch.int64",
    }:
        raise RouteAuditError(f"index sidecar dtype is not int64: {sidecar_path}")
    if sidecar.get("target_raw_ids") is not None:
        declared_ids = tuple(int(value) for value in sidecar["target_raw_ids"])
        if declared_ids != raw_ids:
            raise RouteAuditError(f"index sidecar target_raw_ids disagree with {path}")
    return sidecar, digest, verified


def _validate_gap_worklist(
    source: Mapping[str, Any],
    *,
    canonical_protocol: str,
    canonical_source_sha256: str,
    unit: Mapping[str, Any],
    raw_ids: tuple[int, ...],
) -> dict[str, Any]:
    declared_file_sha = source.get("gap_worklist_file_sha256")
    declared_content_sha = source.get("gap_worklist_content_sha256")
    if not _is_sha(declared_file_sha) or not _is_sha(declared_content_sha):
        raise RouteAuditError(
            "gap source declaration requires worklist file/content SHA256"
        )
    worklist_path = Path(str(source.get("gap_worklist_path", "")))
    worklist = read_json(worklist_path, "gap worklist")
    file_sha = sha256_file(worklist_path)
    if str(declared_file_sha).lower() != file_sha:
        raise RouteAuditError("gap worklist file SHA differs from source declaration")
    if worklist.get("contract") != GAP_CONTRACT or worklist.get("schema_version") != 1:
        raise RouteAuditError("gap worklist contract/schema mismatch")
    recorded_content_sha = worklist.get("content_sha256")
    actual_content_sha = canonical_json_sha(
        {key: value for key, value in worklist.items() if key != "content_sha256"}
    )
    if recorded_content_sha != actual_content_sha:
        raise RouteAuditError("gap worklist content SHA mismatch")
    if str(declared_content_sha).lower() != actual_content_sha:
        raise RouteAuditError("gap worklist content SHA differs from source declaration")
    expected_top = {
        "canonical_protocol": canonical_protocol,
        "canonical_source_sha256": canonical_source_sha256,
        "mode": "target_isolated",
        "feature_namespace": GAP_TARGET_NAMESPACE,
        "vfm": unit["vfm"],
        "model_id": unit["model_id"],
        "timestep": unit["timestep"],
        "source": unit["source"],
        "prompt_contract": unit["prompt_contract"],
        "preprocess_identity": unit["preprocess_identity"],
    }
    for field, expected in expected_top.items():
        if worklist.get(field) != expected:
            raise RouteAuditError(f"gap worklist {field} mismatch")
    layer = int(unit["layer"])
    if layer not in {int(value) for value in worklist.get("output_layers", [])}:
        raise RouteAuditError("gap worklist does not declare the requested output layer")
    allowed: set[int] = set()
    for entry in worklist.get("entries", []):
        if not isinstance(entry, dict):
            raise RouteAuditError("gap worklist entry is not an object")
        if str(entry.get("scene")) != unit["scene_id"]:
            continue
        layers = {int(value) for value in entry.get("layers", [])}
        if layer in layers:
            allowed.add(int(entry["frame_index"]))
    unexpected = sorted(set(raw_ids) - allowed)
    if unexpected:
        raise RouteAuditError(
            f"gap delta contains rows absent from bound worklist for this scene/layer: {unexpected[:8]}"
        )
    return {
        "path": str(worklist_path),
        "file_sha256": file_sha,
        "content_sha256": actual_content_sha,
    }


def _source_gate(
    source_spec: Mapping[str, Any],
    unit: Mapping[str, Any],
    *,
    expected_raw_ids: tuple[int, ...],
    canonical_protocol: str,
    canonical_source_sha256: str,
    output_root: Path,
    verify_source_sha: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_id = _safe_component(source_spec.get("source_id"), "source_id")
    source_priority = int(source_spec.get("priority", 0))
    if source_priority < 0:
        raise RouteAuditError(f"source priority must be non-negative: {source_id}")
    kind = str(source_spec.get("kind", ""))
    if kind not in {"legacy", "gap_delta"}:
        raise RouteAuditError(f"unsupported source kind for {source_id}: {kind!r}")
    feature = str(source_spec.get("feature_path", ""))
    index = str(source_spec.get("index_path", ""))
    feature_sidecar_path = str(
        source_spec.get("feature_sidecar_path") or f"{feature}.manifest.json"
    )
    index_sidecar_path = str(
        source_spec.get("index_sidecar_path") or f"{index}.manifest.json"
    )
    if not feature or not index:
        raise RouteAuditError("source feature_path/index_path are required")
    for location, label in (
        (feature, "source feature"),
        (index, "source target index"),
        (feature_sidecar_path, "source feature sidecar"),
        (index_sidecar_path, "source target index sidecar"),
    ):
        if not source_is_remote(location):
            _regular(Path(location), label)
    if not source_is_remote(feature):
        try:
            Path(feature).relative_to(output_root)
        except ValueError:
            pass
        else:
            raise RouteAuditError("source payload may not live inside the full mixed destination")

    raw_ids = read_npy_int64(index)
    index_sidecar, index_sha, index_sha_verified = _validate_index_sidecar(
        index,
        index_sidecar_path,
        raw_ids,
        verify_sha=verify_source_sha,
    )
    feature_sidecar = read_source_json(feature_sidecar_path, "feature sidecar")
    feature_sha, feature_sha_verified = _validate_sidecar_file_binding(
        feature_sidecar, feature, verify_sha=verify_source_sha
    )
    tensor_key = str(unit["tensor_key"])
    tensor = tensor_layout(
        feature, tensor_key, declared_size=feature_sidecar.get("size_bytes")
    )
    if tensor.shape[0] != len(raw_ids):
        raise RouteAuditError("feature row count does not match target index length")
    if tensor.shape[1:] != tuple(unit["expected_tail_shape"]):
        raise RouteAuditError(
            f"feature tail shape mismatch: {tensor.shape[1:]} != {tuple(unit['expected_tail_shape'])}"
        )
    if tensor.dtype != unit["expected_dtype"]:
        raise RouteAuditError(
            f"feature dtype mismatch: {tensor.dtype} != {unit['expected_dtype']}"
        )

    if feature_sidecar.get("vfm") not in (None, unit["vfm"]):
        raise RouteAuditError("feature sidecar VFM mismatch")
    sidecar_model = feature_sidecar.get("model_id")
    if sidecar_model not in unit["accepted_model_ids"]:
        raise RouteAuditError(
            f"feature sidecar model_id mismatch: {sidecar_model!r}"
        )
    layer_ok = False
    if feature_sidecar.get("layer") is not None:
        layer_ok = int(feature_sidecar["layer"]) == int(unit["layer"])
    if isinstance(feature_sidecar.get("layers"), list):
        layer_ok = layer_ok or int(unit["layer"]) in {
            int(value) for value in feature_sidecar["layers"]
        }
    # Historical aggregate target sidecars often omitted layer/layers.  In
    # that one legacy case, bind the layer through an explicit basename copied
    # from the frozen physical spec.  Never infer it from arbitrary path text.
    if not layer_ok and kind == "legacy":
        expected_name = source_spec.get("expected_source_filename")
        if expected_name is not None:
            expected_name = _safe_component(expected_name, "expected_source_filename")
            layer_ok = source_basename(feature) == expected_name
    if not layer_ok:
        raise RouteAuditError("feature sidecar does not bind the expected layer")
    if feature_sidecar.get("timestep") != unit["timestep"]:
        raise RouteAuditError("feature sidecar timestep mismatch")
    accepted_modes = set(unit["accepted_modes"])
    if feature_sidecar.get("mode") not in accepted_modes:
        raise RouteAuditError("feature sidecar mode mismatch")

    prompt_contract = unit["prompt_contract"]
    if prompt_contract["mode"] == "exact":
        if "prompt" not in feature_sidecar or feature_sidecar["prompt"] != prompt_contract["value"]:
            raise RouteAuditError("feature sidecar prompt mismatch")
    for key, expected in unit["preprocess_identity"].items():
        try:
            actual = _lookup(feature_sidecar, key)
        except KeyError as exc:
            raise RouteAuditError(f"feature sidecar lacks preprocessing field {key!r}") from exc
        if actual != expected:
            raise RouteAuditError(f"feature sidecar preprocessing mismatch for {key!r}")
    tensors = feature_sidecar.get("tensors")
    if isinstance(tensors, dict) and isinstance(tensors.get(tensor_key), dict):
        declared_tensor = tensors[tensor_key]
        if declared_tensor.get("shape") is not None and tuple(
            int(value) for value in declared_tensor["shape"]
        ) != tensor.shape:
            raise RouteAuditError("feature sidecar tensor shape mismatch")
        if declared_tensor.get("dtype") is not None and _normal_dtype(
            declared_tensor["dtype"]
        ) != tensor.dtype:
            raise RouteAuditError("feature sidecar tensor dtype mismatch")

    gap_binding = None
    if kind == "gap_delta":
        gap_binding = _validate_gap_worklist(
            source_spec,
            canonical_protocol=canonical_protocol,
            canonical_source_sha256=canonical_source_sha256,
            unit=unit,
            raw_ids=raw_ids,
        )
        actual_index_sha = index_sha if index_sha_verified else sha256_object(index)
        for sidecar_name, sidecar in (
            ("feature", feature_sidecar),
            ("index", index_sidecar),
        ):
            required = {
                "feature_namespace": GAP_TARGET_NAMESPACE,
                "gap_contract": GAP_CONTRACT,
                "gap_worklist_file_sha256": gap_binding["file_sha256"],
                "gap_worklist_content_sha256": gap_binding["content_sha256"],
                "canonical_source_sha256": canonical_source_sha256,
                "vfm": unit["vfm"],
                "model_id": unit["model_id"],
                "timestep": unit["timestep"],
                "layer": unit["layer"],
                "target_indices_sha256": actual_index_sha,
            }
            for field, expected in required.items():
                if sidecar.get(field) != expected:
                    raise RouteAuditError(f"gap {sidecar_name} sidecar {field} mismatch")
    expected_set = set(expected_raw_ids)
    routes = []
    for source_row_index, raw_id in enumerate(raw_ids):
        if raw_id not in expected_set:
            continue
        routes.append(
            {
                "schema_version": SCHEMA_VERSION,
                "contract": CONTRACT,
                "unit_kind": "target_row",
                "spec_id": unit["spec_id"],
                "vfm": unit["vfm"],
                "model_id": unit["model_id"],
                "layer": unit["layer"],
                "timestep": unit["timestep"],
                "tensor_key": tensor_key,
                "source": unit["source"],
                "scene_id": unit["scene_id"],
                "target_raw_id": raw_id,
                "source_id": source_id,
                "source_priority": source_priority,
                "source_kind": kind,
                "source_row_index": source_row_index,
                "source_feature_path": feature,
                "source_index_path": index,
                "source_feature_sidecar_path": feature_sidecar_path,
                "source_index_sidecar_path": index_sidecar_path,
                "source_payload_sha256": feature_sha,
                "source_payload_sha256_verified": feature_sha_verified,
                "source_index_sha256": index_sha,
                "source_index_sha256_verified": index_sha_verified,
                "source_tensor_header_sha256": tensor.header_sha256,
                "source_size_bytes": tensor.file_size,
                "source_tensor_data_start": tensor.tensor_data_start,
                "source_tensor_row_nbytes": tensor.row_nbytes,
                "source_safetensors_dtype": tensor.safetensors_dtype,
            }
        )
    evidence = {
        "source_id": source_id,
        "source_priority": source_priority,
        "kind": kind,
        "feature_path": feature,
        "index_path": index,
        "remote": source_is_remote(feature),
        "rows": len(raw_ids),
        "canonical_rows": len(routes),
        "ignored_noncanonical_rows": len(set(raw_ids) - expected_set),
        "tensor_shape": list(tensor.shape),
        "tensor_dtype": tensor.dtype,
        "tensor_header_sha256": tensor.header_sha256,
        "tensor_row_nbytes": tensor.row_nbytes,
        "payload_sha256": feature_sha,
        "payload_sha256_verified": feature_sha_verified,
        "index_sha256": index_sha,
        "index_sha256_verified": index_sha_verified,
        "gap_worklist": gap_binding,
    }
    return routes, evidence


def _normalize_unit(
    raw: Mapping[str, Any],
    *,
    canonical_targets: Mapping[tuple[str, str], tuple[int, ...]],
    output_root: Path,
) -> dict[str, Any]:
    unit = dict(raw)
    unit["spec_id"] = _safe_component(unit.get("spec_id"), "spec_id")
    unit["vfm"] = _safe_component(unit.get("vfm"), "vfm")
    unit["source"] = _safe_component(unit.get("source"), "source")
    unit["scene_id"] = _safe_component(unit.get("scene_id"), "scene_id")
    unit["model_id"] = str(unit.get("model_id", "")).strip()
    if not unit["model_id"]:
        raise RouteAuditError(f"{unit['spec_id']}: model_id is required")
    unit["layer"] = int(unit["layer"])
    unit["timestep"] = int(unit["timestep"])
    unit["tensor_key"] = _safe_component(unit.get("tensor_key", "feat"), "tensor_key")
    tail = unit.get("expected_tail_shape")
    if not isinstance(tail, list) or not tail or any(int(value) <= 0 for value in tail):
        raise RouteAuditError(f"{unit['spec_id']}: expected_tail_shape must be positive")
    unit["expected_tail_shape"] = [int(value) for value in tail]
    unit["expected_dtype"] = _normal_dtype(unit.get("expected_dtype"))
    accepted_models = unit.get("accepted_model_ids", [unit["model_id"]])
    if not isinstance(accepted_models, list) or not accepted_models:
        raise RouteAuditError(f"{unit['spec_id']}: accepted_model_ids must be non-empty")
    unit["accepted_model_ids"] = [str(value) for value in accepted_models]
    if unit["model_id"] not in unit["accepted_model_ids"]:
        raise RouteAuditError(f"{unit['spec_id']}: accepted_model_ids omits model_id")
    accepted_modes = unit.get("accepted_modes")
    if not isinstance(accepted_modes, list) or not accepted_modes:
        raise RouteAuditError(f"{unit['spec_id']}: accepted_modes must be explicit")
    unit["accepted_modes"] = [str(value) for value in accepted_modes]
    prompt = unit.get("prompt_contract")
    if not isinstance(prompt, dict) or prompt.get("mode") not in {"exact", "not_applicable"}:
        raise RouteAuditError(f"{unit['spec_id']}: explicit prompt_contract is required")
    if set(prompt) != {"mode", "value"}:
        raise RouteAuditError(
            f"{unit['spec_id']}: prompt_contract must contain exactly mode/value"
        )
    if prompt["mode"] == "exact" and not isinstance(prompt.get("value"), str):
        raise RouteAuditError(f"{unit['spec_id']}: exact prompt_contract lacks a string value")
    if prompt["mode"] == "not_applicable" and prompt.get("value") is not None:
        raise RouteAuditError(
            f"{unit['spec_id']}: not_applicable prompt_contract requires value=null"
        )
    unit["prompt_contract"] = prompt
    preprocess = unit.get("preprocess_identity")
    if not isinstance(preprocess, dict) or set(preprocess) != {"resize", "num_frames"}:
        raise RouteAuditError(
            f"{unit['spec_id']}: preprocess_identity must be exactly resize/num_frames"
        )
    resize = preprocess.get("resize")
    if (
        not isinstance(resize, list)
        or len(resize) != 2
        or any(isinstance(value, bool) or int(value) <= 0 for value in resize)
    ):
        raise RouteAuditError(f"{unit['spec_id']}: resize must be two positive integers")
    num_frames = preprocess.get("num_frames")
    if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames <= 0:
        raise RouteAuditError(f"{unit['spec_id']}: num_frames must be a positive integer")
    unit["preprocess_identity"] = {
        "resize": [int(value) for value in resize],
        "num_frames": int(num_frames),
    }
    destination_filename = _safe_component(
        unit.get("destination_filename"), "destination_filename"
    )
    if not destination_filename.endswith(".sft"):
        raise RouteAuditError(f"{unit['spec_id']}: destination_filename must end in .sft")
    unit["destination_filename"] = destination_filename
    duplicate_policy = str(unit.get("duplicate_owner_policy", "forbid"))
    if duplicate_policy not in {"forbid", "require_bitwise_equal_then_priority"}:
        raise RouteAuditError(
            f"{unit['spec_id']}: unsupported duplicate_owner_policy {duplicate_policy!r}"
        )
    unit["duplicate_owner_policy"] = duplicate_policy
    unit["destination_path"] = str(
        output_root / unit["vfm"] / unit["source"] / unit["scene_id"] / destination_filename
    )
    key = (unit["source"], unit["scene_id"])
    if key not in canonical_targets:
        raise RouteAuditError(f"{unit['spec_id']}: scene absent from canonical source: {key}")
    expected = canonical_targets[key]
    if unit.get("expected_raw_ids") is not None:
        declared = tuple(int(value) for value in unit["expected_raw_ids"])
        if declared != expected:
            raise RouteAuditError(f"{unit['spec_id']}: expected_raw_ids differ from canonical union")
    unit["expected_raw_ids"] = list(expected)
    sources = unit.get("sources", [])
    if not isinstance(sources, list):
        raise RouteAuditError(f"{unit['spec_id']}: sources must be a list")
    if any(not isinstance(source, dict) for source in sources):
        raise RouteAuditError(f"{unit['spec_id']}: every source must be an object")
    if duplicate_policy == "require_bitwise_equal_then_priority":
        if unit["vfm"] != "dino":
            raise RouteAuditError(
                "require_bitwise_equal_then_priority is restricted to vfm=dino"
            )
        if not sources:
            raise RouteAuditError("DINO conditional duplicate policy requires legacy prefix sources")
        for source in sources:
            if source.get("kind") != "legacy":
                raise RouteAuditError(
                    "DINO conditional duplicate policy accepts only legacy prefix sources"
                )
            source_id = str(source.get("source_id", ""))
            if source_id not in DINO_PREFIX_SOURCE_PRIORITIES:
                raise RouteAuditError(
                    f"forged/non-prefix DINO conditional source_id: {source_id!r}"
                )
            if int(source.get("priority", -1)) != DINO_PREFIX_SOURCE_PRIORITIES[source_id]:
                raise RouteAuditError(
                    f"DINO source priority mismatch for {source_id!r}"
                )
    unit["sources"] = sources
    return unit


def _missing_reason(source_failures: Sequence[Mapping[str, Any]]) -> str:
    return "source_gate_failed" if source_failures else "canonical_target_raw_id_unrouted"


def audit_plan(
    plan_path: Path,
    *,
    verify_source_sha: bool,
    remote_policy: str = "gate",
    source_timeout_seconds: float | None = None,
) -> dict[str, Any]:
    if remote_policy not in {"pending", "gate"}:
        raise RouteAuditError(f"unsupported remote_policy: {remote_policy!r}")
    plan = read_json(plan_path, "target row merge plan")
    if plan.get("schema_version") != SCHEMA_VERSION or plan.get("plan_type") != PLAN_TYPE:
        raise RouteAuditError("target row merge plan schema/type mismatch")
    if plan.get("contract") != CONTRACT:
        raise RouteAuditError("target row merge plan contract mismatch")
    recorded_plan_sha = plan.get("content_sha256")
    actual_plan_sha = canonical_json_sha(
        {key: value for key, value in plan.items() if key != "content_sha256"}
    )
    if recorded_plan_sha != actual_plan_sha:
        raise RouteAuditError("target row merge plan content SHA mismatch")
    if plan.get("canonical_protocol") != CANONICAL_PROTOCOL:
        raise RouteAuditError("target row merge plan canonical protocol mismatch")
    protocol_path = Path(str(plan.get("protocol_file", "")))
    protocol = read_json(protocol_path, "canonical protocol file")
    protocol_file_sha = sha256_file(protocol_path)
    if plan.get("protocol_file_sha256") != protocol_file_sha:
        raise RouteAuditError("canonical protocol file SHA mismatch")
    if protocol.get("canonical_protocol") != CANONICAL_PROTOCOL:
        raise RouteAuditError("canonical protocol file identity mismatch")
    canonical_source_spec = plan.get("canonical_source")
    if not isinstance(canonical_source_spec, dict):
        raise RouteAuditError("plan lacks canonical_source binding")
    canonical_source = Path(str(canonical_source_spec.get("path", "")))
    canonical_targets, canonical_metadata = canonical_target_unions(
        canonical_source, str(canonical_source_spec.get("sha256", ""))
    )
    output_root = validate_destination_root(plan.get("destination_root"))

    units_value = plan.get("units")
    if not isinstance(units_value, list) or not units_value:
        raise RouteAuditError("target row merge plan has no units")
    normalized_units: list[dict[str, Any]] = []
    seen_units: set[tuple[Any, ...]] = set()
    seen_destinations: set[str] = set()
    for raw_unit in units_value:
        if not isinstance(raw_unit, dict):
            raise RouteAuditError("target row merge unit is not an object")
        unit = _normalize_unit(
            raw_unit, canonical_targets=canonical_targets, output_root=output_root
        )
        identity = (
            unit["vfm"],
            unit["model_id"],
            unit["layer"],
            unit["timestep"],
            unit["source"],
            unit["scene_id"],
        )
        if identity in seen_units:
            raise RouteAuditError(f"duplicate target merge unit: {identity}")
        if unit["destination_path"] in seen_destinations:
            raise RouteAuditError(f"duplicate destination path: {unit['destination_path']}")
        seen_units.add(identity)
        seen_destinations.add(unit["destination_path"])
        normalized_units.append(unit)

    routes: list[dict[str, Any]] = []
    missing_units: list[dict[str, Any]] = []
    pending_units: list[dict[str, Any]] = []
    unit_reports: list[dict[str, Any]] = []
    hard_errors: list[str] = []
    for unit in normalized_units:
        expected = tuple(unit["expected_raw_ids"])
        candidates: dict[int, list[dict[str, Any]]] = defaultdict(list)
        source_evidence: list[dict[str, Any]] = []
        source_failures: list[dict[str, Any]] = []
        source_pending: list[dict[str, Any]] = []
        source_ids: set[str] = set()
        for source_spec in unit["sources"]:
            if not isinstance(source_spec, dict):
                hard_errors.append(f"{unit['spec_id']}: source declaration is not an object")
                continue
            source_id = str(source_spec.get("source_id", ""))
            if source_id in source_ids:
                hard_errors.append(f"{unit['spec_id']}: duplicate source_id {source_id!r}")
                continue
            source_ids.add(source_id)
            feature_location = str(source_spec.get("feature_path", ""))
            if source_is_remote(feature_location) and remote_policy == "pending":
                source_pending.append(
                    {
                        "source_id": source_id or None,
                        "kind": source_spec.get("kind"),
                        "status": "PENDING_AOSS_GATE",
                        "reason": "remote_policy=pending; remote source was not contacted",
                    }
                )
                continue
            try:
                deadline = source_timeout_seconds if source_is_remote(feature_location) else None
                with _source_deadline(deadline):
                    source_routes, evidence = _source_gate(
                        source_spec,
                        unit,
                        expected_raw_ids=expected,
                        canonical_protocol=CANONICAL_PROTOCOL,
                        canonical_source_sha256=canonical_metadata["sha256"],
                        output_root=output_root,
                        verify_source_sha=verify_source_sha,
                    )
            except SourceGateTimeout as exc:
                source_pending.append(
                    {
                        "source_id": source_id or None,
                        "kind": source_spec.get("kind"),
                        "status": "PENDING_AOSS_GATE",
                        "reason": str(exc),
                    }
                )
                continue
            except RemoteSourcePending as exc:
                source_pending.append(
                    {
                        "source_id": source_id or None,
                        "kind": source_spec.get("kind"),
                        "status": "PENDING_AOSS_GATE",
                        "reason": str(exc),
                    }
                )
                continue
            except (OSError, KeyError, TypeError, ValueError, RouteAuditError) as exc:
                source_failures.append(
                    {
                        "source_id": source_id or None,
                        "kind": source_spec.get("kind"),
                        "reason": str(exc),
                    }
                )
                continue
            source_evidence.append(evidence)
            for route in source_routes:
                candidates[int(route["target_raw_id"])].append(route)

        duplicate_ids = sorted(raw_id for raw_id, owners in candidates.items() if len(owners) > 1)
        duplicate_policy = unit["duplicate_owner_policy"]
        conditional_duplicate_ids = {
            raw_id
            for raw_id in duplicate_ids
            if duplicate_policy == "require_bitwise_equal_then_priority"
            and all(
                owner.get("source_kind") == "legacy"
                and owner.get("source_id") in DINO_PREFIX_SOURCE_PRIORITIES
                for owner in candidates[raw_id]
            )
        }
        forbidden_duplicate_ids = sorted(set(duplicate_ids) - conditional_duplicate_ids)
        if forbidden_duplicate_ids:
            hard_errors.append(
                f"{unit['spec_id']}:{unit['scene_id']}: forbidden duplicate raw-ID owners: "
                f"{forbidden_duplicate_ids[:16]}"
            )
        multi_source_pending = bool(source_pending) and len(unit["sources"]) > 1
        selected: list[dict[str, Any]] = []
        if not multi_source_pending:
            for canonical_row_index, raw_id in enumerate(expected):
                owners = candidates.get(raw_id, [])
                if len(owners) == 1 or (
                    len(owners) > 1 and raw_id in conditional_duplicate_ids
                ):
                    ordered = sorted(
                        owners,
                        key=lambda row: (
                            int(row.get("source_priority", 0)),
                            str(row["source_id"]),
                        ),
                    )
                    route = dict(ordered[0])
                    if len(ordered) > 1:
                        route["alternate_owners"] = [dict(owner) for owner in ordered[1:]]
                        route["duplicate_owner_policy"] = duplicate_policy
                    route["canonical_row_index"] = canonical_row_index
                    route["destination_path"] = unit["destination_path"]
                    selected.append(route)
        missing = [raw_id for raw_id in expected if len(candidates.get(raw_id, [])) == 0]
        pending_raw_ids = list(expected) if multi_source_pending else (
            missing if source_pending else []
        )
        definite_missing = [] if source_pending else missing
        if pending_raw_ids:
            pending_units.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "contract": CONTRACT,
                    "unit_kind": "target_pending",
                    "status": "PENDING_AOSS_GATE",
                    "spec_id": unit["spec_id"],
                    "vfm": unit["vfm"],
                    "model_id": unit["model_id"],
                    "layer": unit["layer"],
                    "timestep": unit["timestep"],
                    "source": unit["source"],
                    "scene_id": unit["scene_id"],
                    "pending_raw_ids": pending_raw_ids,
                    "destination_path": unit["destination_path"],
                    "source_gate_pending": source_pending,
                    "definitive_source_failures": source_failures,
                }
            )
        if definite_missing:
            missing_units.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "contract": CONTRACT,
                    "unit_kind": "target_gap",
                    "status": "REEXTRACT_MISSING",
                    "reason_codes": [_missing_reason(source_failures)],
                    "spec_id": unit["spec_id"],
                    "vfm": unit["vfm"],
                    "model_id": unit["model_id"],
                    "layer": unit["layer"],
                    "timestep": unit["timestep"],
                    "tensor_key": unit["tensor_key"],
                    "source": unit["source"],
                    "scene_id": unit["scene_id"],
                    "missing_raw_ids": definite_missing,
                    "expected_tail_shape": unit["expected_tail_shape"],
                    "expected_dtype": unit["expected_dtype"],
                    "prompt_contract": unit["prompt_contract"],
                    "preprocess_identity": unit["preprocess_identity"],
                    "destination_path": unit["destination_path"],
                    "canonical_source_sha256": canonical_metadata["sha256"],
                    "source_gate_failures": source_failures,
                }
            )
        routes.extend(selected)
        unit_reports.append(
            {
                "spec_id": unit["spec_id"],
                "vfm": unit["vfm"],
                "model_id": unit["model_id"],
                "layer": unit["layer"],
                "timestep": unit["timestep"],
                "source": unit["source"],
                "scene_id": unit["scene_id"],
                "destination_path": unit["destination_path"],
                "expected_rows": len(expected),
                "routed_rows": len(selected),
                "missing_rows": len(definite_missing),
                "pending_rows": len(pending_raw_ids),
                "duplicate_rows": len(duplicate_ids),
                "conditional_duplicate_rows": len(conditional_duplicate_ids),
                "forbidden_duplicate_rows": len(forbidden_duplicate_ids),
                "duplicate_owner_policy": duplicate_policy,
                "multi_source_gate_pending": multi_source_pending,
                "sources_passed": len(source_evidence),
                "sources_failed": len(source_failures),
                "sources_pending": len(source_pending),
                "source_evidence": source_evidence,
                "source_failures": source_failures,
                "source_pending": source_pending,
            }
        )

    routes.sort(
        key=lambda row: (
            row["vfm"], row["model_id"], row["timestep"], row["layer"],
            row["source"], row["scene_id"], row["canonical_row_index"],
        )
    )
    missing_units.sort(
        key=lambda row: (
            row["vfm"], row["model_id"], row["timestep"], row["layer"],
            row["source"], row["scene_id"],
        )
    )
    pending_units.sort(
        key=lambda row: (
            row["vfm"], row["model_id"], row["timestep"], row["layer"],
            row["source"], row["scene_id"],
        )
    )
    status = (
        "FAIL"
        if hard_errors
        else "PENDING"
        if pending_units
        else "INCOMPLETE"
        if missing_units
        else "PASS"
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "audit_type": AUDIT_TYPE,
        "contract": CONTRACT,
        "status": status,
        "ready_to_materialize": status == "PASS",
        "dry_run": True,
        "feature_payload_written": False,
        "seeded_namespace_written": False,
        "canonical_protocol": CANONICAL_PROTOCOL,
        "canonical_protocol_sha256": protocol.get("canonical_protocol_sha256"),
        "protocol_file": str(protocol_path),
        "protocol_file_sha256": protocol_file_sha,
        "canonical_source": canonical_metadata,
        "plan": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "auditor_source": str(Path(__file__).resolve()),
        "auditor_source_sha256": sha256_file(Path(__file__).resolve()),
        "source_reader_source": str(
            (PROJECT_ROOT / "analyses/windowfix_v2/target_row_source.py").resolve()
        ),
        "source_reader_source_sha256": sha256_file(
            (PROJECT_ROOT / "analyses/windowfix_v2/target_row_source.py").resolve()
        ),
        "destination_root": str(output_root),
        "destination_namespace": MIXED_TARGET_NAMESPACE,
        "verify_source_sha": verify_source_sha,
        "remote_policy": remote_policy,
        "source_timeout_seconds": source_timeout_seconds,
        "counts": {
            "units": len(normalized_units),
            "expected_rows": sum(item["expected_rows"] for item in unit_reports),
            "routed_rows": len(routes),
            "missing_rows": sum(len(item["missing_raw_ids"]) for item in missing_units),
            "pending_rows": sum(len(item["pending_raw_ids"]) for item in pending_units),
            "duplicate_rows": sum(item["duplicate_rows"] for item in unit_reports),
            "duplicate_rows_requiring_bitwise_check": sum(
                item["conditional_duplicate_rows"] for item in unit_reports
            ),
            "source_payloads_passed": sum(item["sources_passed"] for item in unit_reports),
            "source_payloads_full_sha_verified": sum(
                1
                for item in unit_reports
                for evidence in item["source_evidence"]
                if evidence.get("payload_sha256_verified") is True
            ),
            "source_payloads_header_only_unverified": sum(
                1
                for item in unit_reports
                for evidence in item["source_evidence"]
                if evidence.get("payload_sha256_verified") is not True
            ),
            "source_payloads_failed": sum(item["sources_failed"] for item in unit_reports),
            "source_payloads_pending": sum(item["sources_pending"] for item in unit_reports),
            "hard_errors": len(hard_errors),
            "routes_by_kind": dict(Counter(row["source_kind"] for row in routes)),
        },
        "units": unit_reports,
        "hard_errors": hard_errors,
        "hard_errors_truncated": False,
    }
    report["content_sha256"] = canonical_json_sha(
        {key: value for key, value in report.items() if key != "content_sha256"}
    )
    return {
        "report": report,
        "routes": routes,
        "missing_units": missing_units,
        "pending_units": pending_units,
        "normalized_units": normalized_units,
        "protocol": protocol,
    }


def _jsonl(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
        for row in rows
    )


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _slug(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-.")
    return value[:96] or "target-gap"


def build_gap_worklists(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    report = result["report"]
    groups: dict[tuple[str, str, int, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in result["missing_units"]:
        prompt_sha = canonical_json_sha(item["prompt_contract"])
        preprocess_sha = canonical_json_sha(item["preprocess_identity"])
        key = (
            item["vfm"], item["model_id"], int(item["timestep"]), item["source"],
            prompt_sha, preprocess_sha,
        )
        groups[key].append(item)
    worklists: list[dict[str, Any]] = []
    for key, missing in sorted(groups.items()):
        vfm, model_id, timestep, source, _prompt_sha, _preprocess_sha = key
        layers = sorted({int(item["layer"]) for item in missing})
        entries_by_frame: dict[tuple[str, int], dict[str, Any]] = {}
        for item in missing:
            for raw_id in item["missing_raw_ids"]:
                frame_key = (item["scene_id"], int(raw_id))
                entry = entries_by_frame.setdefault(
                    frame_key,
                    {
                        "scene": item["scene_id"],
                        "frame_index": int(raw_id),
                        "layers": [],
                        "reason": item["reason_codes"][0],
                    },
                )
                entry["layers"].append(int(item["layer"]))
        entries = []
        for frame_key in sorted(entries_by_frame):
            entry = entries_by_frame[frame_key]
            entry["layers"] = sorted(set(entry["layers"]))
            entries.append(entry)
        first = missing[0]
        worklist = {
            "schema_version": 1,
            "contract": GAP_CONTRACT,
            "canonical_protocol": CANONICAL_PROTOCOL,
            "protocol_sha256": report.get("canonical_protocol_sha256"),
            "protocol_file_sha256": report["protocol_file_sha256"],
            "canonical_source_sha256": report["canonical_source"]["sha256"],
            "mode": "target_isolated",
            "vfm": vfm,
            "model_id": model_id,
            "timestep": timestep,
            "feature_namespace": "FEAT_TARGET_CANONICAL_V2_GAP_DELTA_V1",
            "output_layers": layers,
            "source": source,
            "prompt_contract": first["prompt_contract"],
            "preprocess_identity": first["preprocess_identity"],
            "entries": entries,
            "counts": {
                "entries": len(entries),
                "layer_items": sum(len(entry["layers"]) for entry in entries),
                "scenes": len({entry["scene"] for entry in entries}),
            },
        }
        worklist["content_sha256"] = canonical_json_sha(worklist)
        worklists.append(worklist)
    return worklists


def write_outputs(output_dir: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    if output_dir.exists() or output_dir.is_symlink():
        raise RouteAuditError(
            f"audit output directory already exists; overwrite forbidden: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    routes_payload = _jsonl(result["routes"]).encode()
    missing_payload = _jsonl(result["missing_units"]).encode()
    pending_payload = _jsonl(result.get("pending_units", [])).encode()
    routes_path = output_dir / "target_row_routes.jsonl"
    missing_path = output_dir / "missing_target_units.jsonl"
    pending_path = output_dir / "pending_target_units.jsonl"
    _write_atomic(routes_path, routes_payload)
    _write_atomic(missing_path, missing_payload)
    _write_atomic(pending_path, pending_payload)

    gap_bindings = []
    for worklist in build_gap_worklists(result):
        name = _slug(
            f"{worklist['vfm']}-t{worklist['timestep']}-"
            f"{canonical_json_sha([worklist['model_id'], worklist['source'], worklist['output_layers']])[:12]}"
        ) + ".json"
        path = output_dir / "gap_worklists" / name
        payload = (
            json.dumps(worklist, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode()
        _write_atomic(path, payload)
        gap_bindings.append(
            {
                "path": str(path),
                "file_sha256": hashlib.sha256(payload).hexdigest(),
                "content_sha256": worklist["content_sha256"],
                "counts": worklist["counts"],
            }
        )

    report = dict(result["report"])
    report["outputs"] = {
        "target_row_routes": {
            "path": str(routes_path),
            "records": len(result["routes"]),
            "sha256": hashlib.sha256(routes_payload).hexdigest(),
        },
        "missing_target_units": {
            "path": str(missing_path),
            "records": len(result["missing_units"]),
            "sha256": hashlib.sha256(missing_payload).hexdigest(),
        },
        "pending_target_units": {
            "path": str(pending_path),
            "records": len(result.get("pending_units", [])),
            "sha256": hashlib.sha256(pending_payload).hexdigest(),
        },
        "gap_worklists": gap_bindings,
    }
    report["content_sha256"] = canonical_json_sha(
        {key: value for key, value in report.items() if key != "content_sha256"}
    )
    report_path = output_dir / "target_row_route_audit.json"
    _write_atomic(
        report_path,
        (json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(),
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--verify-source-sha", action="store_true")
    parser.add_argument(
        "--remote-policy",
        choices=("pending", "gate"),
        default="pending",
        help="pending never contacts AOSS; gate performs fresh remote source checks",
    )
    parser.add_argument(
        "--source-timeout-seconds",
        type=float,
        default=30.0,
        help="per-remote-source deadline under --remote-policy gate",
    )
    args = parser.parse_args(argv)
    try:
        result = audit_plan(
            args.plan.resolve(),
            verify_source_sha=args.verify_source_sha,
            remote_policy=args.remote_policy,
            source_timeout_seconds=args.source_timeout_seconds,
        )
        report = write_outputs(args.output_dir.resolve(), result)
    except RouteAuditError as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "ready_to_materialize": report["ready_to_materialize"],
                "counts": report["counts"],
                "output": str(args.output_dir.resolve()),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    return 2 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
