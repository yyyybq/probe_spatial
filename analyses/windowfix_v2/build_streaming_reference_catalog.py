#!/usr/bin/env python3
"""Build the record-level canonical-v2 streaming reuse catalogue.

This is an inventory/audit tool, not a copier and not an extractor.  It joins
the frozen 160 feature specs (40 physical feature identities x four prefixes)
to the 10,884 canonical streaming records, and classifies all 435,360 units.

The only rows emitted as immediately routable are local Wan p8 payloads that
pass a fresh ``LEGACY_EXACT_REFERENCE_V1`` gate.  That gate reads the JSON
sidecar and safetensors header, but deliberately does not read/hash the tensor
body.  Semantically matching AOSS rows remain ``PENDING_REMOTE_GATE`` until a
fresh remote receipt proves the same checks.  This distinction is intentional:
an inventory or an old training log is useful evidence, but is not current
payload evidence.

Historical ``prefix_index.npy`` files are trusted, operator-controlled inputs
and therefore loaded with ``allow_pickle=True``.  Their bytes (and every
sidecar that is consumed) are SHA-bound in the output receipt catalogue.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import struct
import sys
import tempfile
from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, TextIO

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.windowfix_v2.run_backend_determinism_canary import (  # noqa: E402
    backend_provenance_binding,
    canary_runtime_environment,
    validate_qwen_streaming_tensor_contract,
)

PROTOCOL = ROOT / "protocols/inscene_streaming_canonical_v2.json"
REFERENCE_POLICY = ROOT / "protocols/inscene_feature_reference_legacy_exact_v1.json"
INVENTORY = (
    ROOT
    / "artifacts/windowfix_v2/feature_inventory/"
    "canonical_v2_feature_specs.windowfix_v2_b34c099e010d.json"
)
MANIFEST_TEMPLATE = (
    ROOT / "artifacts/tstep_canonical_v2/manifests/inscene_c_h124_prefix{prefix}.jsonl"
)
CANONICAL_PARENT_SOURCE = (
    ROOT / "manifests/c_retrain_20260722/inscene_c_h124_prefix24.jsonl"
)
CANONICAL_NAMESPACE_LOCK = (
    ROOT / "artifacts/windowfix_v2/canonical_namespace_lock.02e17c40b59a.json"
)
DEDICATED_INDEX_ROOT = Path("/tmp/default8_full_indexes")
SHARED_INDEX_ROOT = Path(
    "/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_STREAMING/wan/scannetpp"
)
WAN_VERSIONED_ROOT = Path(
    "/mnt/umm/users/yinbaiqiao/InsScene-15K/FEATURES_VERSIONED/"
    "wan_p8_action_dynamics_canonical_manifest_exact_20260821_200604_2075f3e/"
    "streaming/wan/scannetpp"
)
WAN_REPORT_ROOT = (
    ROOT
    / "reports/wan_p8_recovery/"
    "wan_p8_action_dynamics_canonical_manifest_exact_20260821_200604_2075f3e"
)

PREFIXES = (8, 12, 16, 24)
EXPECTED_SPECS = 160
EXPECTED_RECORDS_PER_PREFIX = 2721
EXPECTED_UNIVERSE = 435_360
EXPECTED_RAW_EXACT = 257_228
EXPECTED_SEMANTIC_CANDIDATES = 217_396
EXPECTED_LOCAL_REFERENCES = 5_298
EXPECTED_PENDING = 212_098
EXPECTED_IDENTITY_MISMATCH = 39_832
EXPECTED_RAW_MISMATCH = 178_132
FROZEN_PROTOCOL_FILE_SHA256 = (
    "02e17c40b59ae5075a23641cb41ce5c46c2d7cda6333d40c48d988dffd106c7d"
)
FROZEN_PROTOCOL_SEMANTIC_SHA256 = (
    "215c720c8537f4dfae6243bedcc7339fbb7e3ba8a6ca2dcaf04f49cd4b738e4f"
)
CANONICAL_PARENT_SOURCE_SHA256 = (
    "ebc496eb0587cda372ccf29a7c5244ef454096151f73c8eed9511967ded1902d"
)
CANONICAL_UNIVERSE_SHA256 = (
    "4771e2859c43889bd5fb75f5876bcf493758f4187336c9bd7eeca5432acbac9b"
)

SCHEMA = "CANONICAL_V2_STREAMING_ROUTE_CATALOG_V1"
REFERENCE_NAMESPACE = "LEGACY_EXACT_REFERENCE_V1"
OUTPUT_NAMESPACE = "FEAT_STREAMING_CANONICAL_V2_MIXED_REFERENCE_V1"
STATUS_REFERENCE = "REFERENCE_SOURCE_READ_ONLY"
STATUS_PENDING = "PENDING_REMOTE_GATE"
STATUS_IDENTITY = "REEXTRACT_IDENTITY_MISMATCH"
STATUS_RAW = "REEXTRACT_RAW_ID_MISMATCH"
STATUS_UNVERIFIED = "REEXTRACT_UNVERIFIED"

SHARED_VFMS = frozenset(
    {"cogvideox", "lingbot", "minimax_h3", "videomae", "vjepa2"}
)
DEDICATED_VFMS = frozenset(
    {"aether", "cogvideox", "dino", "f3r", "qwen2_5_vl", "wan"}
)
WAN_SHARED_LAYERS = frozenset({0, 5, 10, 15, 25, 29})
HEX64 = re.compile(r"^[0-9a-f]{64}$")
WAN_SNAPSHOT = re.compile(
    r"(?:^|/)models--Wan-AI--Wan2\.1-T2V-1\.3B-Diffusers/"
    r"snapshots/[0-9a-f]+/?$"
)

DTYPE_ALIASES = {
    "F64": "float64",
    "F32": "float32",
    "F16": "float16",
    "BF16": "bfloat16",
    "I64": "int64",
    "I32": "int32",
    "I16": "int16",
    "I8": "int8",
    "U8": "uint8",
    "BOOL": "bool",
}
DTYPE_BYTES = {
    "float64": 8,
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int64": 8,
    "int32": 4,
    "int16": 2,
    "int8": 1,
    "uint8": 1,
    "bool": 1,
}

PROMPT_CONTRACTS = {
    "aether": {"mode": "exact", "value": ""},
    "cogvideox": {"mode": "exact", "value": ""},
    "dino": {"mode": "not_applicable"},
    "f3r": {"mode": "not_applicable"},
    "lingbot": {"mode": "exact", "value": "."},
    "minimax_h3": {"mode": "exact", "value": "."},
    "qwen2_5_vl": {
        "mode": "exact",
        "value": "Describe the spatial layout of this scene.",
    },
    "videomae": {"mode": "not_applicable"},
    "vjepa2": {"mode": "not_applicable"},
    "wan": {"mode": "exact", "value": "."},
}
PREPROCESS_CONTRACTS = {
    "aether": {"resize": [480, 720], "num_frames": 81},
    "cogvideox": {"resize": [480, 720], "num_frames": 97},
    "dino": {"resize": [420, 728], "num_frames": 64},
    "f3r": {"resize": [288, 512], "num_frames": 64},
    "lingbot": {"resize": [480, 832], "num_frames": 49},
    "minimax_h3": {"resize": [480, 832], "num_frames": 73},
    # Qwen is the one no-pad streaming backend: its real input frame count is
    # the selected prefix length.  ``preprocess_contract`` replaces this
    # sentinel per record; keeping a fixed value here would create false
    # provenance for p12/p16/p24.
    "qwen2_5_vl": {"resize": [336, 336], "num_frames": None},
    "videomae": {"resize": [224, 224], "num_frames": 16},
    "vjepa2": {"resize": [256, 256], "num_frames": 64},
    "wan": {"resize": [480, 832], "num_frames": 81},
}

# Exact canonical streaming payload shapes.  These are the extractor output
# contracts after prefix frames have been padded to each backend's fixed input
# length.  Aether's two 41-frame clips are kept as the backend's native 5-D
# layout.  Qwen is deliberately unresolved because its temporal/grid shape is
# emitted by the runtime processor and the model/processor revision is not yet
# canary-bound; it may not pass a reference gate until a new immutable
# catalogue is built with a canary-proven exact shape.
EXPECTED_STREAMING_SHAPES: dict[str, list[int] | None] = {
    "aether": [2, 11, 30, 45, 3072],
    "cogvideox": [2, 13, 30, 45, 3072],
    "dino": [64, 30, 52, 1024],
    "f3r": [64, 18, 32, 1024],
    "lingbot": [13, 30, 52, 2048],
    "minimax_h3": [19, 15, 26, 5376],
    "qwen2_5_vl": None,
    "videomae": [8, 14, 14, 1024],
    "vjepa2": [32, 16, 16, 1024],
    "wan": [21, 30, 52, 1536],
}


def expected_shapes_by_prefix(
    qwen_contract: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, list[int]] | None]:
    """Expand fixed backends and a canary-resolved Qwen contract by prefix."""
    result: dict[str, dict[str, list[int]] | None] = {}
    for vfm, shape in EXPECTED_STREAMING_SHAPES.items():
        if vfm == "qwen2_5_vl":
            result[vfm] = (
                {
                    str(prefix): [int(value) for value in qwen_contract[str(prefix)]["shape"]]
                    for prefix in PREFIXES
                }
                if qwen_contract is not None
                else None
            )
        else:
            if shape is None:  # pragma: no cover - frozen registry invariant
                raise CatalogError(f"unexpected unresolved fixed shape: {vfm}")
            result[vfm] = {str(prefix): list(shape) for prefix in PREFIXES}
    return result


def qwen_contract_from_canary_receipt(
    path: Path,
    *,
    protocol_file_sha256: str,
) -> tuple[dict[str, dict[str, Any]], Path]:
    """Certify a Qwen receipt before allowing it to resolve catalog shapes."""
    if path.is_symlink() or not path.is_file():
        raise CatalogError(f"Qwen shape canary receipt must be regular: {path}")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise CatalogError("Qwen shape canary receipt must be an object")
    extractor = ROOT / "features/run_inscene15k_mllm.py"
    runner = ROOT / "scripts/windowfix_v2/run_backend_determinism_canary.py"
    renderer = ROOT / "scripts/windowfix_v2/prepare_gap_backend_canary_jobs.py"
    expected = {
        "schema_version": 2,
        "audit_type": "canonical_v2_backend_cross_process_order_canary",
        "status": "pass",
        "submission_performed": False,
        "vfm": "qwen2_5_vl",
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "layer": -1,
        "timestep": 749,
        "protocol": "inscene_streaming_canonical_v2_obs28_h124",
        "protocol_file_sha256": protocol_file_sha256,
        "extractor": str(extractor),
        "extractor_sha256": sha256_file(extractor),
        "canary_script": str(runner),
        "canary_script_sha256": sha256_file(runner),
        "runtime_prompt": "Describe the spatial layout of this scene.",
        "runtime_resize": [336, 336],
        "runtime_num_frames": 24,
        "backend_provenance_revalidated_after_runs": True,
    }
    bad = [key for key, value in expected.items() if receipt.get(key) != value]
    if bad:
        raise CatalogError(f"Qwen shape canary receipt identity mismatch: {bad}")
    try:
        provenance = backend_provenance_binding("qwen2_5_vl", ROOT)
        runtime_environment = canary_runtime_environment("qwen2_5_vl", provenance)
    except RuntimeError as exc:
        raise CatalogError(str(exc)) from exc
    if (
        receipt.get("backend_provenance") != provenance
        or receipt.get("runtime_environment") != runtime_environment
    ):
        raise CatalogError("Qwen shape canary backend provenance is stale")

    spec_binding = receipt.get("canary_spec_binding")
    if not isinstance(spec_binding, dict):
        raise CatalogError("Qwen shape canary has no spec binding")
    spec_path = Path(str(spec_binding.get("path", "")))
    if spec_path.is_symlink() or not spec_path.is_file():
        raise CatalogError(f"Qwen canary spec must be regular: {spec_path}")
    if sha256_file(spec_path) != spec_binding.get("file_sha256"):
        raise CatalogError("Qwen canary spec file SHA mismatch")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise CatalogError("Qwen canary spec must be an object")
    unhashed = dict(spec)
    stored_content_sha = unhashed.pop("content_sha256", None)
    if (
        stored_content_sha != sha256_json(unhashed)
        or stored_content_sha != spec_binding.get("content_sha256")
    ):
        raise CatalogError("Qwen canary spec content SHA mismatch")
    spec_expected = {
        "schema_version": 2,
        "audit_type": "canonical_v2_backend_canary_execution_spec",
        "status": "READY_FOR_REVIEW_NOT_SUBMITTED",
        "submission_performed": False,
        "vfm": "qwen2_5_vl",
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "layer": -1,
        "timestep": 749,
        "protocol_file_sha256": protocol_file_sha256,
        "extractor": str(extractor),
        "extractor_sha256": sha256_file(extractor),
        "canary_runner": str(runner),
        "canary_runner_sha256": sha256_file(runner),
        "canary_renderer": str(renderer),
        "canary_renderer_sha256": sha256_file(renderer),
        "backend_provenance": provenance,
        "runtime_environment": runtime_environment,
        "runtime_contract": {
            "prompt": "Describe the spatial layout of this scene.",
            "resize": [336, 336],
            "num_frames": 24,
            "python": "/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python",
            "sample_index": 1,
        },
        "receipt": str(path.resolve()),
    }
    spec_bad = [key for key, value in spec_expected.items() if spec.get(key) != value]
    if spec_bad:
        raise CatalogError(f"Qwen canary spec identity mismatch: {spec_bad}")
    gap = receipt.get("gap_binding")
    comparisons = receipt.get("results")
    if (
        not isinstance(gap, dict)
        or gap.get("canonical_source_sha256") != CANONICAL_PARENT_SOURCE_SHA256
        or gap.get("selected_scene_layer_covered") is not True
    ):
        raise CatalogError("Qwen shape canary is not bound to the canonical gap universe")
    comparison_ok = (
        isinstance(comparisons, dict)
        and set(comparisons) == {"streaming_prefix", "target_isolated"}
        and all(
            isinstance(value, dict)
            and int(value.get("comparison", {}).get("safetensors_compared", 0)) > 0
            and int(value.get("comparison", {}).get("files_compared", 0))
            >= int(value.get("comparison", {}).get("safetensors_compared", 0))
            and isinstance(value.get("comparison", {}).get("comparison_sha256"), str)
            and len(value["comparison"]["comparison_sha256"]) == 64
            for value in comparisons.values()
        )
        and comparisons["streaming_prefix"]["comparison"].get("scene")
        == comparisons["target_isolated"]["comparison"].get("scene")
        == gap.get("selected_scene")
    )
    if not comparison_ok:
        raise CatalogError("Qwen shape canary cross-process comparisons are incomplete")
    try:
        contract = validate_qwen_streaming_tensor_contract(
            receipt.get("qwen_streaming_tensor_contract")
        )
    except RuntimeError as exc:
        raise CatalogError(str(exc)) from exc
    return contract, spec_path


def preprocess_contract(vfm: str, prefix: int) -> dict[str, Any]:
    value = dict(PREPROCESS_CONTRACTS[vfm])
    if vfm == "qwen2_5_vl":
        value.update(num_frames=int(prefix), temporal_padding="none")
    else:
        value["temporal_padding"] = "repeat_tail"
    return value


class CatalogError(RuntimeError):
    """Raised when evidence cannot support an unambiguous catalogue."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def json_load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CatalogError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CatalogError(f"expected JSON object: {path}")
    return value


def add_file_receipt(
    receipts: list[dict[str, Any]], path: Path, *, role: str, full_hash: bool = True
) -> dict[str, Any]:
    stat = path.stat()
    receipt: dict[str, Any] = {
        "role": role,
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "hash_scope": "full_file" if full_hash else "not_hashed",
    }
    if full_hash:
        receipt["sha256"] = sha256_file(path)
    receipts.append(receipt)
    return receipt


def load_canonical(path: Path, prefix: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            value = json.loads(line)
            raw_ids = tuple(int(item) for item in value["streaming_indices"])
            row = {
                "scene_id": str(value["scene_id"]),
                "sample_id": str(value["sample_id"]),
                "split": str(value["split"]),
                "window_id": int(value["streaming_window_id"]),
                "tail": int(value["streaming_tail"]),
                "prefix": int(value["prefix_length"]),
                "raw_ids": raw_ids,
            }
            if row["prefix"] != prefix or row["tail"] != prefix - 1:
                raise CatalogError(f"{path}:{line_number}: prefix/tail mismatch")
            if len(raw_ids) != prefix:
                raise CatalogError(f"{path}:{line_number}: raw ID length mismatch")
            key = (row["scene_id"], row["window_id"], row["tail"])
            if key in seen:
                raise CatalogError(f"{path}:{line_number}: duplicate key {key}")
            seen.add(key)
            rows.append(row)
    if len(rows) != EXPECTED_RECORDS_PER_PREFIX:
        raise CatalogError(
            f"{path}: expected {EXPECTED_RECORDS_PER_PREFIX} rows, found {len(rows)}"
        )
    return rows


def load_index(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    """Load one trusted historical object-array index."""

    try:
        values = np.load(path, allow_pickle=True).tolist()
    except Exception as exc:  # pragma: no cover - error text is enough for field triage
        raise CatalogError(f"cannot load historical index {path}: {exc}") from exc
    if isinstance(values, dict):
        values = [values]
    if not isinstance(values, list):
        raise CatalogError(f"historical index is not list-like: {path}")
    records: dict[tuple[int, int], dict[str, Any]] = {}
    for offset, value in enumerate(values):
        if not isinstance(value, dict):
            raise CatalogError(f"{path}: record {offset} is not an object")
        window_id = int(value.get("window_id", value.get("streaming_window_id")))
        tail_value = value.get("tail", value.get("streaming_tail"))
        if tail_value is None:
            tail_value = int(value.get("prefix_length")) - 1
        key = (window_id, int(tail_value))
        if key in records:
            raise CatalogError(f"{path}: duplicate index key {key}")
        records[key] = value
    return records


def index_raw_ids(record: Mapping[str, Any] | None) -> tuple[int, ...] | None:
    if record is None:
        return None
    value = record.get(
        "actual_input_raw_ids", record.get("streaming_indices", record.get("indices"))
    )
    if not isinstance(value, (list, tuple, np.ndarray)):
        return None
    return tuple(int(item) for item in value)


class IndexStore:
    """Lazy index reader with byte receipts and stable evidence labels."""

    def __init__(self, receipts: list[dict[str, Any]]) -> None:
        self.receipts = receipts
        self.cache: dict[Path, dict[tuple[int, int], dict[str, Any]]] = {}

    def get(self, path: Path, *, role: str) -> dict[tuple[int, int], dict[str, Any]]:
        if path not in self.cache:
            if not path.is_file():
                raise CatalogError(f"required index is missing: {path}")
            add_file_receipt(self.receipts, path, role=role)
            sidecar = Path(f"{path}.manifest.json")
            if sidecar.is_file():
                add_file_receipt(self.receipts, sidecar, role=f"{role}_sidecar")
            self.cache[path] = load_index(path)
        return self.cache[path]


def normalize_dtype(value: Any) -> str:
    text = str(value).strip()
    text = text.removeprefix("torch.").removeprefix("numpy.").removeprefix("np.")
    return DTYPE_ALIASES.get(text.upper(), text.lower())


def read_safetensors_header(path: Path, tensor_key: str = "feat") -> dict[str, Any]:
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        length_bytes = handle.read(8)
        if len(length_bytes) != 8:
            raise ValueError("truncated safetensors header length")
        header_size = struct.unpack("<Q", length_bytes)[0]
        if header_size <= 1 or header_size > min(file_size - 8, 128 * 1024 * 1024):
            raise ValueError("invalid safetensors header length")
        header_bytes = handle.read(header_size)
        if len(header_bytes) != header_size:
            raise ValueError("truncated safetensors header")
    header = json.loads(header_bytes.decode("utf-8"))
    tensor = header.get(tensor_key)
    if not isinstance(tensor, dict):
        raise ValueError(f"tensor {tensor_key!r} missing")
    shape = tuple(int(item) for item in tensor.get("shape", ()))
    if not shape or any(item < 0 for item in shape):
        raise ValueError("invalid tensor shape")
    dtype = normalize_dtype(tensor.get("dtype"))
    if dtype not in DTYPE_BYTES:
        raise ValueError("unsupported tensor dtype")
    offsets = tensor.get("data_offsets")
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise ValueError("invalid tensor offsets")
    start, end = int(offsets[0]), int(offsets[1])
    payload_size = file_size - 8 - header_size
    if start < 0 or end < start or end > payload_size:
        raise ValueError("tensor offsets outside payload")
    elements = 1
    for dimension in shape:
        elements *= dimension
    if end - start != elements * DTYPE_BYTES[dtype]:
        raise ValueError("tensor byte length disagrees with shape/dtype")
    return {
        "shape": list(shape),
        "dtype": dtype,
        "header_size_bytes": header_size,
        "header_sha256": sha256_bytes(length_bytes + header_bytes),
        "data_offsets": [start, end],
    }


def expected_wan_model(value: Any) -> bool:
    text = str(value).strip().rstrip("/")
    return text == "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" or bool(
        WAN_SNAPSHOT.search(text)
    )


def gate_wan_reference(
    *,
    path: Path,
    spec: Mapping[str, Any],
    canonical: Mapping[str, Any],
    receipts: list[dict[str, Any]],
) -> tuple[bool, dict[str, Any], list[str]]:
    """Fresh stat + sidecar + header gate; never hash the feature body."""

    failures: list[str] = []
    evidence: dict[str, Any] = {
        "gate_policy": REFERENCE_NAMESPACE,
        "payload_full_read_performed": False,
        "byte_level_certified": False,
    }
    sidecar_path = Path(f"{path}.manifest.json")
    if path.is_symlink() or not path.is_file():
        failures.append("source_missing_or_symlink")
        return False, evidence, failures
    if sidecar_path.is_symlink() or not sidecar_path.is_file():
        failures.append("sidecar_missing_or_symlink")
        return False, evidence, failures

    stat = path.stat()
    evidence["source_stat"] = {
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
    }
    try:
        sidecar_bytes = sidecar_path.read_bytes()
        sidecar = json.loads(sidecar_bytes)
        if not isinstance(sidecar, dict):
            raise TypeError("sidecar must be an object")
        sidecar_sha = sha256_bytes(sidecar_bytes)
        sidecar_stat = sidecar_path.stat()
        receipts.append(
            {
                "role": "wan_reference_feature_sidecar",
                "path": str(sidecar_path),
                "size_bytes": sidecar_stat.st_size,
                "mtime_ns": sidecar_stat.st_mtime_ns,
                "hash_scope": "full_file",
                "sha256": sidecar_sha,
            }
        )
        evidence["sidecar_uri"] = str(sidecar_path)
        evidence["sidecar_sha256"] = sidecar_sha
    except Exception as exc:
        failures.append(f"sidecar_unreadable:{type(exc).__name__}")
        return False, evidence, failures

    layer = int(spec["feature_layer"])
    expected_shape = EXPECTED_STREAMING_SHAPES["wan"]
    if expected_shape is None:  # pragma: no cover - frozen registry invariant
        raise CatalogError("Wan exact shape contract is unresolved")
    checks = {
        "file_name": sidecar.get("file") == path.name,
        "file_size": int(sidecar.get("size_bytes", -1)) == stat.st_size,
        "scene_id": str(sidecar.get("scene_id", "")) == canonical["scene_id"],
        "sample_id": str(sidecar.get("sample_id", "")) == canonical["sample_id"],
        "split": str(sidecar.get("split", "")) == canonical["split"],
        "window_id": int(sidecar.get("window_id", -1)) == canonical["window_id"],
        "tail": int(sidecar.get("tail", -1)) == canonical["tail"],
        "raw_ids": tuple(int(x) for x in sidecar.get("actual_input_raw_ids", ()))
        == canonical["raw_ids"],
        "timestep": int(sidecar.get("timestep", -1)) == int(spec["timestep"]),
        "layer": layer in [int(x) for x in sidecar.get("layers", ())],
        "model": expected_wan_model(sidecar.get("model_id")),
        "prompt": sidecar.get("prompt") == ".",
        "resize": sidecar.get("resize") == [480, 832],
        "mode": sidecar.get("mode") == "manifest_exact_streaming",
        "frame_index_schema": sidecar.get("frame_index_schema")
        == "canonical_manifest_exact_v1",
        "motion_window_generator_used": sidecar.get("motion_window_generator_used")
        is False,
    }
    declared_hash = str(sidecar.get("sha256", "")).lower()
    checks["declared_payload_sha256_syntax"] = bool(HEX64.fullmatch(declared_hash))
    evidence["declared_payload_sha256_unverified"] = (
        declared_hash if checks["declared_payload_sha256_syntax"] else None
    )

    try:
        header = read_safetensors_header(path)
        evidence["safetensors_header"] = header
        tensor = sidecar.get("tensors", {}).get("feat", {})
        checks["sidecar_shape_equals_header"] = tensor.get("shape") == header["shape"]
        checks["header_shape_equals_expected"] = header["shape"] == expected_shape
        checks["dtype"] = normalize_dtype(tensor.get("dtype")) == header["dtype"]
        receipts.append(
            {
                "role": "wan_reference_payload_header",
                "path": str(path),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "hash_scope": "safetensors_header_only",
                "sha256": header["header_sha256"],
            }
        )
    except Exception as exc:
        checks["header"] = False
        failures.append(f"payload_header_invalid:{type(exc).__name__}")

    evidence["checks"] = checks
    failures.extend(f"check_failed:{key}" for key, passed in checks.items() if not passed)
    return not failures, evidence, failures


def source_for(
    *, spec: Mapping[str, Any], canonical: Mapping[str, Any], source_kind: str
) -> str:
    vfm = str(spec["vfm"])
    prefix = int(spec["prefix"])
    scene = canonical["scene_id"]
    window_id = canonical["window_id"]
    tail = canonical["tail"]
    name = spec["feature_filename"]
    suffix = f"{vfm}/scannetpp/{scene}/window_{window_id:04d}/prefix_{tail:06d}/{name}"
    if source_kind == "shared_aoss":
        return f"aoss://baiqiao/InsScene-15K/FEAT_STREAMING/{suffix}"
    if source_kind == "dedicated_aoss":
        return (
            "aoss://baiqiao/InsScene-15K/FEAT_STREAMING_BY_PREFIX/"
            f"{vfm}_prefix_{prefix}_m0p35_r0p5/{suffix}"
        )
    if source_kind == "wan_versioned_local":
        return str(
            WAN_VERSIONED_ROOT
            / scene
            / f"window_{window_id:04d}"
            / f"prefix_{tail:06d}"
            / name
        )
    raise CatalogError(f"unknown source kind: {source_kind}")


def source_index_for(
    *, spec: Mapping[str, Any], canonical: Mapping[str, Any], source_kind: str
) -> str:
    vfm = str(spec["vfm"])
    prefix = int(spec["prefix"])
    scene = canonical["scene_id"]
    if source_kind == "shared_aoss":
        return (
            "aoss://baiqiao/InsScene-15K/FEAT_STREAMING/"
            f"{vfm}/scannetpp/{scene}/prefix_index.npy"
        )
    if source_kind == "dedicated_aoss":
        return (
            "aoss://baiqiao/InsScene-15K/FEAT_STREAMING_BY_PREFIX/"
            f"{vfm}_prefix_{prefix}_m0p35_r0p5/{vfm}/scannetpp/"
            f"{scene}/prefix_index.npy"
        )
    raise CatalogError(f"no remote source index for {source_kind}")


def old_record(
    *,
    store: IndexStore,
    spec: Mapping[str, Any],
    canonical: Mapping[str, Any],
    source_kind: str,
) -> tuple[Mapping[str, Any] | None, str]:
    scene = canonical["scene_id"]
    key = (canonical["window_id"], canonical["tail"])
    prefix = int(spec["prefix"])
    vfm = str(spec["vfm"])
    if source_kind == "shared_aoss":
        path = SHARED_INDEX_ROOT / scene / "prefix_index.npy"
        label = "shared_obs24_protocol_index"
    elif source_kind == "dedicated_aoss":
        path = DEDICATED_INDEX_ROOT / f"{vfm}_p{prefix}" / scene / "prefix_index.npy"
        label = "dedicated_prefix_protocol_index"
    elif source_kind == "wan_versioned_local":
        path = WAN_VERSIONED_ROOT / scene / "prefix_index.npy"
        label = "wan_versioned_manifest_exact_index"
    else:
        raise CatalogError(f"unknown source kind: {source_kind}")
    return store.get(path, role=label).get(key), str(path)


def select_protocol_source(spec: Mapping[str, Any]) -> str:
    vfm = str(spec["vfm"])
    layer = int(spec["feature_layer"])
    if vfm == "wan":
        return "dedicated_aoss" if layer == 20 else "shared_aoss"
    if vfm == "cogvideox":
        return "dedicated_aoss" if layer == 20 else "shared_aoss"
    if vfm in SHARED_VFMS:
        return "shared_aoss"
    if vfm in DEDICATED_VFMS:
        return "dedicated_aoss"
    raise CatalogError(f"no historical source mapping for {vfm}")


def base_route(spec: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    raw_ids = list(row["raw_ids"])
    return {
        "schema": SCHEMA,
        "namespace": OUTPUT_NAMESPACE,
        "reference_policy": REFERENCE_NAMESPACE,
        "unit_id": f"{spec['spec_id']}:{row['sample_id']}",
        "spec_id": spec["spec_id"],
        "vfm": spec["vfm"],
        "model_id": spec["model_id"],
        "feature_layer": int(spec["feature_layer"]),
        "timestep": int(spec["timestep"]),
        "feature_filename": spec["feature_filename"],
        "prefix": int(spec["prefix"]),
        "scene_id": row["scene_id"],
        "sample_id": row["sample_id"],
        "split": row["split"],
        "window_id": row["window_id"],
        "tail": row["tail"],
        "canonical_raw_ids": raw_ids,
        "canonical_raw_ids_sha256": sha256_json(raw_ids),
    }


def classify_unit(
    *,
    store: IndexStore,
    spec: Mapping[str, Any],
    canonical: Mapping[str, Any],
    receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    route = base_route(spec, canonical)
    protocol_source = select_protocol_source(spec)
    record, index_path = old_record(
        store=store, spec=spec, canonical=canonical, source_kind=protocol_source
    )
    legacy_ids = index_raw_ids(record)
    raw_exact = legacy_ids == canonical["raw_ids"]
    route.update(
        {
            "protocol_source_kind": protocol_source,
            "protocol_index_evidence": index_path,
            "legacy_raw_ids": list(legacy_ids) if legacy_ids is not None else None,
            "raw_ids_exact": raw_exact,
            "semantic_candidate": False,
            "payload_hash_performed": False,
            "byte_level_certified": False,
        }
    )
    if not raw_exact:
        route.update(
            status=STATUS_RAW,
            reason_codes=["NO_EXACT_LEGACY_RAW_ID_SOURCE"],
            action="REEXTRACT",
        )
        return route

    vfm = str(spec["vfm"])
    layer = int(spec["feature_layer"])
    prefix = int(spec["prefix"])
    if vfm != "wan":
        route.update(
            status=STATUS_PENDING,
            reason_codes=["SEMANTIC_IDENTITY_MATCH", "REMOTE_PAYLOAD_GATE_NOT_FRESH"],
            action="WAIT_FOR_REMOTE_GATE",
            semantic_candidate=True,
            source_kind=protocol_source,
            source_uri=source_for(
                spec=spec, canonical=canonical, source_kind=protocol_source
            ),
            gate_requirements=[
                "source_exists",
                "sidecar_exists",
                "file_and_size_match",
                "raw_ids_match",
                "model_layer_timestep_match",
                "prompt_preprocess_match_or_not_applicable",
                "shape_dtype_match",
                "safetensors_header_valid",
            ],
        )
        return route

    if layer == 20 or prefix != 8:
        route.update(
            status=STATUS_IDENTITY,
            reason_codes=["WAN_RAW_IDS_EXACT_BUT_PROMPT_OR_EXTRACTOR_IDENTITY_MISMATCH"],
            action="REEXTRACT",
            source_kind=protocol_source,
            source_uri=source_for(
                spec=spec, canonical=canonical, source_kind=protocol_source
            ),
        )
        return route

    versioned, versioned_index = old_record(
        store=store,
        spec=spec,
        canonical=canonical,
        source_kind="wan_versioned_local",
    )
    if index_raw_ids(versioned) != canonical["raw_ids"]:
        route.update(
            status=STATUS_IDENTITY,
            reason_codes=["WAN_SHARED_PROMPT_MISMATCH", "VERSIONED_P8_RAW_IDS_NOT_EXACT"],
            action="REEXTRACT",
            source_kind=protocol_source,
            source_uri=source_for(
                spec=spec, canonical=canonical, source_kind=protocol_source
            ),
        )
        return route

    route["semantic_candidate"] = True
    route["raw_candidate_source_kind"] = route["protocol_source_kind"]
    route["raw_candidate_index_evidence"] = route["protocol_index_evidence"]
    route["protocol_source_kind"] = "wan_versioned_local"
    route["protocol_index_evidence"] = versioned_index
    route["source_kind"] = "wan_versioned_local"
    route["source_uri"] = source_for(
        spec=spec, canonical=canonical, source_kind="wan_versioned_local"
    )
    passed, gate, failures = gate_wan_reference(
        path=Path(route["source_uri"]),
        spec=spec,
        canonical=canonical,
        receipts=receipts,
    )
    route["reference_gate"] = gate
    if passed:
        route.update(
            status=STATUS_REFERENCE,
            reason_codes=["LEGACY_EXACT_REFERENCE_V1_FRESH_GATE_PASS"],
            action="REFERENCE_SOURCE_READ_ONLY",
        )
    else:
        route.update(
            status=STATUS_UNVERIFIED,
            reason_codes=["LOCAL_REFERENCE_GATE_FAILED", *failures],
            action="REEXTRACT",
        )
    return route


def open_deterministic_gzip(path: Path) -> TextIO:
    raw = path.open("wb")
    zipped = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    return __import__("io").TextIOWrapper(zipped, encoding="utf-8", newline="\n")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def counter_tree(counter: Counter[tuple[Any, ...]], names: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        {**dict(zip(names, key)), "count": value}
        for key, value in sorted(counter.items())
    ]


def build(args: argparse.Namespace) -> Path:
    protocol = json_load(args.protocol)
    current_protocol_file_sha = sha256_file(args.protocol)
    reference_policy = json_load(args.reference_policy)
    if reference_policy.get("contract") != REFERENCE_NAMESPACE:
        raise CatalogError("reference policy contract is not LEGACY_EXACT_REFERENCE_V1")
    if (
        reference_policy.get("route_namespaces", {}).get("streaming")
        != OUTPUT_NAMESPACE
    ):
        raise CatalogError(
            "reference policy streaming namespace changed: "
            f"{reference_policy.get('route_namespaces', {}).get('streaming')!r}"
        )
    inventory = json_load(args.inventory)
    lineage_protocol_file_sha = str(
        inventory.get("source", {}).get("protocol_file_sha256", "")
    )
    inventory_semantic_sha = str(
        inventory.get("canonical_protocol", {}).get("protocol_sha256", "")
    )
    if lineage_protocol_file_sha != FROZEN_PROTOCOL_FILE_SHA256:
        raise CatalogError(
            "inventory is not bound to the frozen 02e17c protocol lineage: "
            f"{lineage_protocol_file_sha}"
        )
    if inventory_semantic_sha != FROZEN_PROTOCOL_SEMANTIC_SHA256:
        raise CatalogError(f"inventory semantic protocol changed: {inventory_semantic_sha}")
    if str(protocol.get("canonical_protocol_sha256", "")) != inventory_semantic_sha:
        raise CatalogError(
            "current protocol has a different semantic fingerprint: "
            f"{protocol.get('canonical_protocol_sha256')!r}"
        )

    # The four prefix manifests are projections of one physical canonical
    # p24 source.  Bind that parent explicitly: merely hashing the derived
    # JSONL files would not prove that they all came from the same universe.
    namespace_lock = json_load(args.namespace_lock)
    lock_source = namespace_lock.get("canonical_source", {})
    if not isinstance(lock_source, dict):
        raise CatalogError("namespace lock canonical_source is not an object")
    if lock_source.get("sha256") != CANONICAL_PARENT_SOURCE_SHA256:
        raise CatalogError(
            "namespace lock canonical parent changed: "
            f"{lock_source.get('sha256')!r}"
        )
    if namespace_lock.get("canonical_universe_sha256") != CANONICAL_UNIVERSE_SHA256:
        raise CatalogError(
            "namespace lock canonical universe changed: "
            f"{namespace_lock.get('canonical_universe_sha256')!r}"
        )
    parent_source_sha = sha256_file(args.canonical_parent_source)
    if parent_source_sha != CANONICAL_PARENT_SOURCE_SHA256:
        raise CatalogError(
            "canonical parent source bytes changed: "
            f"expected {CANONICAL_PARENT_SOURCE_SHA256}, found {parent_source_sha}"
        )
    specs = inventory.get("specs")
    if not isinstance(specs, list) or len(specs) != EXPECTED_SPECS:
        raise CatalogError(f"expected {EXPECTED_SPECS} feature specs")

    output = args.output_dir or (
        ROOT
        / f"artifacts/windowfix_v2/reference_overlay_{lineage_protocol_file_sha[:12]}"
    )
    output = Path(output)
    if output.exists():
        raise CatalogError(f"output already exists (refusing to overwrite): {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    canonical: dict[int, list[dict[str, Any]]] = {}
    input_receipts: list[dict[str, Any]] = []
    qwen_tensor_contract: dict[str, dict[str, Any]] | None = None
    qwen_canary_spec: Path | None = None
    if args.qwen_shape_canary_receipt is not None:
        qwen_tensor_contract, qwen_canary_spec = qwen_contract_from_canary_receipt(
            args.qwen_shape_canary_receipt.resolve(),
            protocol_file_sha256=current_protocol_file_sha,
        )
    shapes_by_prefix = expected_shapes_by_prefix(qwen_tensor_contract)
    direct_inputs = [
        (args.protocol, "canonical_protocol"),
        (args.reference_policy, "reference_policy"),
        (args.inventory, "feature_inventory"),
        (args.namespace_lock, "canonical_namespace_lock"),
        (args.canonical_parent_source, "canonical_parent_source_p24"),
        (Path(__file__), "catalog_builder"),
        (ROOT / "logs/aoss_inventory_20260812.json", "historical_aoss_inventory"),
        (
            ROOT / "logs/lingbot_aoss_sync/streaming_scenes_supervisor.log",
            "lingbot_upload_completion_log",
        ),
        (WAN_REPORT_ROOT / "full_feature_audit.json", "wan_prior_full_tensor_audit"),
        (
            WAN_REPORT_ROOT / "provenance/extractor_config.json",
            "wan_extractor_config",
        ),
        (ROOT / "manifests/c_retrain_20260722/inscene_c_h124_prefix8.jsonl", "wan_source_manifest"),
        (ROOT / "scripts/extract_wan_manifest_exact.py", "wan_extractor_script"),
    ]
    if args.qwen_shape_canary_receipt is not None:
        direct_inputs.extend(
            [
                (
                    args.qwen_shape_canary_receipt.resolve(),
                    "qwen_backend_shape_canary_receipt",
                ),
                (qwen_canary_spec, "qwen_backend_shape_canary_spec"),
            ]
        )
    for path, role in direct_inputs:
        if path is None:  # pragma: no cover - guarded by receipt validation
            raise CatalogError(f"required evidence input missing for {role}")
        if not path.is_file():
            raise CatalogError(f"required evidence input missing: {path}")
        add_file_receipt(input_receipts, path, role=role)
    for prefix in PREFIXES:
        path = Path(str(args.manifest_template).format(prefix=prefix))
        manifest_receipt = add_file_receipt(
            input_receipts, path, role=f"canonical_manifest_p{prefix}"
        )
        meta_path = Path(f"{path}.meta.json")
        meta_receipt = add_file_receipt(
            input_receipts, meta_path, role=f"canonical_manifest_meta_p{prefix}"
        )
        meta = json_load(meta_path)
        if int(meta.get("prefix_length", -1)) != prefix:
            raise CatalogError(f"{meta_path}: prefix_length does not equal {prefix}")
        if meta.get("sha256") != manifest_receipt.get("sha256"):
            raise CatalogError(
                f"{meta_path}: derived manifest SHA does not bind {path}"
            )
        meta_source = meta.get("source", {})
        if not isinstance(meta_source, dict):
            raise CatalogError(f"{meta_path}: source is not an object")
        meta_parent_sha = meta_source.get("canonical_manifest_sha256")
        if meta_parent_sha != CANONICAL_PARENT_SOURCE_SHA256:
            raise CatalogError(
                f"{meta_path}: canonical parent SHA changed: {meta_parent_sha!r}"
            )
        if meta.get("universe_sha256") != CANONICAL_UNIVERSE_SHA256:
            raise CatalogError(
                f"{meta_path}: canonical universe SHA changed: "
                f"{meta.get('universe_sha256')!r}"
            )
        # This explicit p24 check prevents a self-consistent family of derived
        # manifests from silently drifting away from the namespace lock.
        if prefix == 24 and meta_parent_sha != lock_source.get("sha256"):
            raise CatalogError(
                f"{meta_path}: p24 parent differs from namespace lock source"
            )
        canonical[prefix] = load_canonical(path, prefix)

    created_at = datetime.now(timezone.utc).isoformat()
    status_counts: Counter[str] = Counter()
    candidate_counts: Counter[tuple[str, int, str]] = Counter()
    model_prefix_status: Counter[tuple[str, int, str]] = Counter()
    spec_status: Counter[tuple[str, str]] = Counter()
    reason_counts: Counter[str] = Counter()
    raw_exact_count = 0
    semantic_count = 0

    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as tmp:
        stage = Path(tmp)
        store = IndexStore(input_receipts)
        paths = {
            "universe": stage / "streaming_universe.routes.jsonl.gz",
            "candidates": stage / "streaming_semantic_candidates.routes.jsonl.gz",
            "references": stage / "streaming_reference_routes.jsonl",
            "pending": stage / "streaming_pending_remote_gate.routes.jsonl.gz",
            "reextract": stage / "streaming_reextract.routes.jsonl.gz",
            "requests": stage / "remote_gate_requests.jsonl.gz",
        }
        with ExitStack() as stack:
            universe = stack.enter_context(open_deterministic_gzip(paths["universe"]))
            candidates = stack.enter_context(open_deterministic_gzip(paths["candidates"]))
            references = stack.enter_context(paths["references"].open("w", encoding="utf-8"))
            pending = stack.enter_context(open_deterministic_gzip(paths["pending"]))
            reextract = stack.enter_context(open_deterministic_gzip(paths["reextract"]))
            requests = stack.enter_context(open_deterministic_gzip(paths["requests"]))

            for spec in sorted(
                specs,
                key=lambda x: (
                    str(x["vfm"]),
                    int(x["feature_layer"]),
                    int(x["timestep"]),
                    int(x["prefix"]),
                ),
            ):
                prefix = int(spec["prefix"])
                for row in canonical[prefix]:
                    route = classify_unit(
                        store=store,
                        spec=spec,
                        canonical=row,
                        receipts=input_receipts,
                    )
                    encoded = canonical_json(route) + "\n"
                    universe.write(encoded)
                    status = route["status"]
                    status_counts[status] += 1
                    model_prefix_status[(route["vfm"], prefix, status)] += 1
                    spec_status[(route["spec_id"], status)] += 1
                    for reason in route["reason_codes"]:
                        reason_counts[reason] += 1
                    if route["raw_ids_exact"]:
                        raw_exact_count += 1
                    if route["semantic_candidate"]:
                        semantic_count += 1
                        candidates.write(encoded)
                        candidate_counts[(route["vfm"], prefix, status)] += 1
                    if status == STATUS_REFERENCE:
                        references.write(encoded)
                    elif status == STATUS_PENDING:
                        pending.write(encoded)
                        vfm = str(route["vfm"])
                        per_prefix_shapes = shapes_by_prefix[vfm]
                        expected_shape = (
                            per_prefix_shapes[str(prefix)]
                            if per_prefix_shapes is not None
                            else None
                        )
                        expected_dtype = (
                            qwen_tensor_contract[str(prefix)]["dtype"]
                            if vfm == "qwen2_5_vl" and qwen_tensor_contract is not None
                            else "float16"
                        )
                        preprocess = {
                            "frame_index_schema": "image_mask_intersection_v1",
                            **preprocess_contract(vfm, prefix),
                        }
                        request = {
                            "schema": "LEGACY_EXACT_REFERENCE_REMOTE_GATE_REQUEST_V1",
                            "unit_id": route["unit_id"],
                            "source_uri": route["source_uri"],
                            "sidecar_uri": f"{route['source_uri']}.manifest.json",
                            "source_index_uri": source_index_for(
                                spec=spec,
                                canonical=row,
                                source_kind=route["source_kind"],
                            ),
                            "source_index_record": {
                                "window_id": route["window_id"],
                                "tail": route["tail"],
                            },
                            "expected": {
                                "vfm": vfm,
                                "model_id": route["model_id"],
                                "feature_layer": route["feature_layer"],
                                "timestep": route["timestep"],
                                "feature_filename": route["feature_filename"],
                                "mode": "streaming_prefix",
                                "prompt_contract": PROMPT_CONTRACTS[vfm],
                                "preprocess_identity": preprocess,
                                "frame_index_schema": preprocess["frame_index_schema"],
                                "tensor_key": "feat",
                                "expected_shape": expected_shape,
                                "shape_contract": (
                                    "sidecar_and_safetensors_header_both_equal_exact_expected_shape"
                                    if expected_shape is not None
                                    else "pending_backend_shape_canary_no_reference_promotion"
                                ),
                                "expected_dtype": expected_dtype,
                                "dtype_contract": (
                                    "sidecar_and_safetensors_header_both_equal_expected_dtype"
                                ),
                                "raw_ids": route["canonical_raw_ids"],
                                "raw_ids_sha256": route["canonical_raw_ids_sha256"],
                            },
                            "required_checks": route["gate_requirements"],
                        }
                        requests.write(canonical_json(request) + "\n")
                    else:
                        reextract.write(encoded)

        total = sum(status_counts.values())
        expected_status = {
            STATUS_REFERENCE: EXPECTED_LOCAL_REFERENCES,
            STATUS_PENDING: EXPECTED_PENDING,
            STATUS_IDENTITY: EXPECTED_IDENTITY_MISMATCH,
            STATUS_RAW: EXPECTED_RAW_MISMATCH,
        }
        if total != EXPECTED_UNIVERSE:
            raise CatalogError(f"universe count changed: {total}")
        if raw_exact_count != EXPECTED_RAW_EXACT:
            raise CatalogError(f"raw-exact count changed: {raw_exact_count}")
        if semantic_count != EXPECTED_SEMANTIC_CANDIDATES:
            raise CatalogError(f"semantic candidate count changed: {semantic_count}")
        if dict(status_counts) != expected_status:
            raise CatalogError(
                f"fresh status counts differ from frozen expectation: "
                f"expected {expected_status}, found {dict(status_counts)}"
            )

        # Every consumed source file/sidecar/header has an individual receipt.
        # Deduplicate identical index receipts reached by several layer specs.
        unique_receipts: dict[tuple[str, str, str], dict[str, Any]] = {}
        for receipt in input_receipts:
            key = (receipt["role"], receipt["path"], receipt["hash_scope"])
            unique_receipts[key] = receipt
        ordered_receipts = sorted(
            unique_receipts.values(), key=lambda x: (x["path"], x["role"], x["hash_scope"])
        )
        receipt_path = stage / "input_sha_receipts.jsonl.gz"
        aggregate = hashlib.sha256()
        with open_deterministic_gzip(receipt_path) as handle:
            for receipt in ordered_receipts:
                line = canonical_json(receipt) + "\n"
                handle.write(line)
                aggregate.update(line.encode("utf-8"))

        candidate_rows = counter_tree(
            candidate_counts, ("vfm", "prefix", "status")
        )
        all_rows = counter_tree(
            model_prefix_status, ("vfm", "prefix", "status")
        )
        spec_rows = counter_tree(spec_status, ("spec_id", "status"))
        write_csv(
            stage / "candidate_counts_by_model_prefix_status.csv",
            candidate_rows,
            ["vfm", "prefix", "status", "count"],
        )
        write_csv(
            stage / "universe_counts_by_model_prefix_status.csv",
            all_rows,
            ["vfm", "prefix", "status", "count"],
        )
        write_csv(
            stage / "counts_by_spec_status.csv",
            spec_rows,
            ["spec_id", "status", "count"],
        )

        summary = {
            "schema": SCHEMA,
            "created_at": created_at,
            "canonical_protocol_sha256": inventory_semantic_sha,
            "canonical_protocol_lineage_file_sha256": lineage_protocol_file_sha,
            "current_protocol_file_sha256_observed": current_protocol_file_sha,
            "namespace": OUTPUT_NAMESPACE,
            "canonical_parent_source": str(args.canonical_parent_source),
            "canonical_parent_source_sha256": CANONICAL_PARENT_SOURCE_SHA256,
            "canonical_universe_sha256": CANONICAL_UNIVERSE_SHA256,
            "canonical_namespace_lock": str(args.namespace_lock),
            "reference_policy": REFERENCE_NAMESPACE,
            "expected_streaming_shapes": {
                **EXPECTED_STREAMING_SHAPES,
                "qwen2_5_vl": (
                    {
                        prefix: value["shape"]
                        for prefix, value in qwen_tensor_contract.items()
                    }
                    if qwen_tensor_contract is not None
                    else None
                ),
            },
            "expected_streaming_shapes_by_prefix": shapes_by_prefix,
            "qwen_shape_canary": (
                {
                    "receipt": str(args.qwen_shape_canary_receipt.resolve()),
                    "receipt_sha256": sha256_file(
                        args.qwen_shape_canary_receipt.resolve()
                    ),
                    "spec": str(qwen_canary_spec),
                    "spec_sha256": sha256_file(qwen_canary_spec),
                    "status": "PASS_BOUND",
                }
                if args.qwen_shape_canary_receipt is not None
                and qwen_canary_spec is not None
                else {
                    "status": "PENDING_REQUIRED",
                    "reason": "no certified Qwen backend canary receipt supplied",
                }
            ),
            "claims": {
                "payload_full_read_performed": False,
                "byte_level_certified": False,
                "seeded_v1_claimed": False,
                "feature_payloads_copied": 0,
                "jobs_submitted": 0,
            },
            "counts": {
                "feature_specs": len(specs),
                "canonical_records": sum(len(x) for x in canonical.values()),
                "universe": total,
                "raw_id_exact": raw_exact_count,
                "semantic_candidates": semantic_count,
                "status": dict(sorted(status_counts.items())),
            },
            "status_semantics": {
                STATUS_REFERENCE: (
                    "fresh local exists+sidecar+identity+shape/dtype+safetensors-header "
                    "gate passed; source is referenced read-only; body SHA was not recomputed"
                ),
                STATUS_PENDING: (
                    "raw IDs and semantic identity are candidates, but current remote "
                    "payload/sidecar/header evidence has not been received"
                ),
                STATUS_IDENTITY: "raw IDs match, but extractor/prompt identity does not",
                STATUS_RAW: "no selected legacy source has exact canonical raw IDs",
                STATUS_UNVERIFIED: "a candidate failed a required fresh payload gate",
            },
            "input_receipts": {
                "file": receipt_path.name,
                "count": len(ordered_receipts),
                "canonical_jsonl_sha256": aggregate.hexdigest(),
            },
            "remote_policy": {
                "fresh_remote_reads_performed": 0,
                "pending_units": status_counts[STATUS_PENDING],
                "note": "Historical AOSS inventories/uploads never promote a row to REFERENCE.",
            },
            "ignored_alternative_indexes": [
                str(DEDICATED_INDEX_ROOT / "videomae_p{8,12,16,24}"),
                str(DEDICATED_INDEX_ROOT / "vjepa2_p{8,12,16,24}"),
            ],
            "reason_counts": dict(sorted(reason_counts.items())),
        }
        write_json(stage / "summary.json", summary)

        output_files: list[dict[str, Any]] = []
        for path in sorted(stage.iterdir()):
            if path.name == "catalog_manifest.json":
                continue
            output_files.append(
                {
                    "file": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        manifest = {
            "schema": "CANONICAL_V2_STREAMING_ROUTE_CATALOG_MANIFEST_V1",
            "created_at": created_at,
            "protocol_lineage_file_sha256": lineage_protocol_file_sha,
            "current_protocol_file_sha256_observed": current_protocol_file_sha,
            "canonical_parent_source_sha256": CANONICAL_PARENT_SOURCE_SHA256,
            "canonical_universe_sha256": CANONICAL_UNIVERSE_SHA256,
            "outputs": output_files,
            "note": "catalog_manifest.json is excluded from its own non-circular digest set",
        }
        write_json(stage / "catalog_manifest.json", manifest)
        os.rename(stage, output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--reference-policy", type=Path, default=REFERENCE_POLICY)
    parser.add_argument("--inventory", type=Path, default=INVENTORY)
    parser.add_argument("--manifest-template", type=Path, default=MANIFEST_TEMPLATE)
    parser.add_argument(
        "--canonical-parent-source", type=Path, default=CANONICAL_PARENT_SOURCE
    )
    parser.add_argument("--namespace-lock", type=Path, default=CANONICAL_NAMESPACE_LOCK)
    parser.add_argument(
        "--qwen-shape-canary-receipt",
        type=Path,
        help=(
            "Optional schema-v2 PASS receipt. Without it Qwen expected shapes "
            "remain unresolved and remote reference promotion fails closed."
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    try:
        output = build(parse_args())
    except CatalogError as exc:
        raise SystemExit(f"ERROR: {exc}")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
