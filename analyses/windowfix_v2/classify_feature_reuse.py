#!/usr/bin/env python3
"""Classify legacy streaming features for canonical-v2 reuse.

The classifier is intentionally read-only.  A record is ``REUSE_EXACT`` only
when all of the following are proved from local artifacts:

* the legacy ``prefix_index.npy`` record has exactly the canonical raw IDs;
* the feature payload and its ``.manifest.json`` sidecar both exist;
* model/VFM, layer, timestep, tensor shape and dtype match the feature spec;
* sidecar file name, byte size and SHA256 match the payload; and
* the safetensors header agrees with the declared tensor shape and dtype.

Remote roots are reported as ``UNVERIFIED`` rather than guessed missing.  The
tool never copies, uploads, deletes, or submits anything.

``--payload-hash-policy legacy-import`` is the deliberately separate relaxed
policy used by ``LEGACY_EXACT_IMPORT_V1``.  It still reads and hashes the whole
payload, but does not require a historical SHA to have been written.  A valid
historical SHA that disagrees with the bytes remains a hard mismatch.  This
policy also requires the feature specification to pin ``prompt`` and a
preprocessing identity (``resize`` or ``preprocess*``) in ``expected_sidecar``.

``--payload-hash-policy reference`` is the faster, zero-copy
``LEGACY_EXACT_REFERENCE_V1`` policy.  It validates the safetensors header and
file size but deliberately does not read/hash tensor data.  A declared SHA is
recorded as unverified evidence only.  Therefore this mode certifies temporal
and physical identity, not byte integrity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import struct
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse

import numpy as np


SCHEMA_VERSION = 1
DEFAULT_PROTOCOL = "inscene_streaming_canonical_v2_obs28_h124"
STATUSES = (
    "REUSE_EXACT",
    "REEXTRACT_MISMATCH",
    "REEXTRACT_MISSING",
    "UNVERIFIED",
)
REMOTE_SCHEMES = frozenset({"aoss", "s3", "fj", "http", "https", "oss"})
HEX_RE = re.compile(r"^[0-9a-f]{64}$")


class InputError(ValueError):
    """Raised when an input inventory or manifest is ambiguous."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _read_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise InputError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                if not isinstance(value, dict):
                    raise InputError(f"{path}:{line_number}: expected a JSON object")
                rows.append(value)
        return rows
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            for key in ("items", "specs", "roots", "rows"):
                if isinstance(value.get(key), list):
                    value = value[key]
                    break
        if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
            raise InputError(f"{path}: expected a JSON list of objects")
        return list(value)
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open(encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)]
    raise InputError(f"unsupported table format for {path}; use JSON, JSONL, CSV, or TSV")


def _required_text(row: Mapping[str, Any], field: str, where: str) -> str:
    value = str(row.get(field, "")).strip()
    if not value:
        raise InputError(f"{where}: missing {field}")
    return value


def _optional_int(value: Any, *, field: str, where: str) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    if isinstance(value, bool):
        raise InputError(f"{where}: {field} must be an integer or null")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise InputError(f"{where}: {field} must be an integer or null") from exc
    if isinstance(value, float) and float(value) != result:
        raise InputError(f"{where}: {field} must be integral")
    return result


def _string_list(value: Any, *, field: str, where: str) -> tuple[str, ...]:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ()
        if stripped.startswith("["):
            value = json.loads(stripped)
        else:
            value = stripped.replace(",", " ").split()
    if not isinstance(value, (list, tuple)):
        raise InputError(f"{where}: {field} must be a list")
    result = tuple(str(item).strip() for item in value if str(item).strip())
    if len(result) != len(set(result)):
        raise InputError(f"{where}: {field} contains duplicates")
    return result


def _shape(value: Any, *, where: str) -> tuple[int, ...] | None:
    if value is None or str(value).strip() == "":
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            value = json.loads(stripped)
        else:
            value = re.split(r"[x, ]+", stripped)
    if not isinstance(value, (list, tuple)):
        raise InputError(f"{where}: expected_shape must be a list")
    try:
        result = tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise InputError(f"{where}: expected_shape must contain integers") from exc
    if not result or any(item < 0 for item in result):
        raise InputError(f"{where}: expected_shape must be nonempty and nonnegative")
    return result


def _json_object(value: Any, *, field: str, where: str) -> dict[str, Any]:
    if value is None or str(value).strip() == "":
        return {}
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise InputError(f"{where}: {field} must be a JSON object")
    return dict(value)


def _is_remote(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme.lower() in REMOTE_SCHEMES or "://" in value


@dataclass(frozen=True)
class LegacyRoot:
    root_id: str
    index_scene_root: str
    feature_scene_root: str
    index_path_template: str = "{scene}/prefix_index.npy"
    feature_path_template: str = (
        "{scene}/window_{window_id:04d}/prefix_{tail:06d}/{feature_name}"
    )
    feature_name_template: str = "feature_t{timestep}_layer{layer}.sft"

    @property
    def is_local(self) -> bool:
        return not _is_remote(self.index_scene_root) and not _is_remote(self.feature_scene_root)


@dataclass(frozen=True)
class FeatureSpec:
    spec_id: str
    root_ids: tuple[str, ...]
    prefix: int
    vfm: str
    model_id: str | None
    layer: int | None
    timestep: int | None
    expected_shape: tuple[int, ...] | None
    expected_dtype: str | None
    tensor_key: str
    feature_name: str | None
    feature_name_template: str | None
    expected_mode: str | None
    expected_sidecar: dict[str, Any]

    def incomplete_identity_fields(self) -> list[str]:
        missing: list[str] = []
        if not self.model_id:
            missing.append("model_id")
        if self.layer is None:
            missing.append("layer")
        if self.expected_shape is None:
            missing.append("expected_shape")
        if not self.expected_dtype:
            missing.append("expected_dtype")
        return missing


def load_roots(path: Path) -> dict[str, LegacyRoot]:
    roots: dict[str, LegacyRoot] = {}
    for offset, row in enumerate(_read_rows(path), 1):
        where = f"{path}:row {offset}"
        root_id = _required_text(row, "legacy_root_id", where)
        if root_id in roots:
            raise InputError(f"{where}: duplicate legacy_root_id={root_id!r}")
        generic = str(row.get("scene_root", "")).strip()
        index_root = str(row.get("index_scene_root", generic)).strip()
        feature_root = str(row.get("feature_scene_root", generic)).strip()
        if not index_root or not feature_root:
            raise InputError(
                f"{where}: provide scene_root or both index_scene_root and feature_scene_root"
            )
        roots[root_id] = LegacyRoot(
            root_id=root_id,
            index_scene_root=index_root,
            feature_scene_root=feature_root,
            index_path_template=str(
                row.get("index_path_template", "{scene}/prefix_index.npy")
            ).strip(),
            feature_path_template=str(
                row.get(
                    "feature_path_template",
                    "{scene}/window_{window_id:04d}/prefix_{tail:06d}/{feature_name}",
                )
            ).strip(),
            feature_name_template=str(
                row.get("feature_name_template", "feature_t{timestep}_layer{layer}.sft")
            ).strip(),
        )
    if not roots:
        raise InputError(f"{path}: no legacy roots")
    return roots


def load_specs(path: Path, roots: Mapping[str, LegacyRoot]) -> list[FeatureSpec]:
    specs: list[FeatureSpec] = []
    seen: set[str] = set()
    for offset, row in enumerate(_read_rows(path), 1):
        where = f"{path}:row {offset}"
        spec_id = _required_text(row, "spec_id", where)
        if spec_id in seen:
            raise InputError(f"{where}: duplicate spec_id={spec_id!r}")
        seen.add(spec_id)
        root_value = row.get("legacy_root_ids", row.get("legacy_root_id"))
        root_ids = _string_list(root_value, field="legacy_root_ids", where=where)
        if not root_ids:
            raise InputError(f"{where}: no legacy_root_id(s)")
        unknown = sorted(set(root_ids) - set(roots))
        if unknown:
            raise InputError(f"{where}: unknown legacy roots: {unknown}")
        prefix = _optional_int(row.get("prefix"), field="prefix", where=where)
        if prefix is None or prefix <= 0:
            raise InputError(f"{where}: prefix must be positive")
        shape = _shape(
            row.get("expected_shape", row.get("feature_shape", row.get("shape"))),
            where=where,
        )
        dtype_value = row.get("expected_dtype", row.get("dtype"))
        dtype = str(dtype_value).strip() if dtype_value is not None else None
        if not dtype:
            dtype = None
        model_value = row.get("model_id")
        mode_value = row.get("expected_mode", row.get("mode"))
        feature_name = (
            str(row.get("feature_name", row.get("feature_filename", ""))).strip()
            or None
        )
        feature_template = str(row.get("feature_name_template", "")).strip() or None
        specs.append(
            FeatureSpec(
                spec_id=spec_id,
                root_ids=root_ids,
                prefix=prefix,
                vfm=_required_text(row, "vfm", where),
                model_id=(
                    str(model_value).strip()
                    if model_value is not None and str(model_value).strip()
                    else None
                ),
                layer=_optional_int(
                    row.get("layer", row.get("feature_layer")), field="layer", where=where
                ),
                timestep=_optional_int(row.get("timestep"), field="timestep", where=where),
                expected_shape=shape,
                expected_dtype=dtype,
                tensor_key=str(row.get("tensor_key", "feat")).strip() or "feat",
                feature_name=feature_name,
                feature_name_template=feature_template,
                expected_mode=(
                    str(mode_value).strip()
                    if mode_value is not None and str(mode_value).strip()
                    else None
                ),
                expected_sidecar=_json_object(
                    row.get("expected_sidecar"), field="expected_sidecar", where=where
                ),
            )
        )
    if not specs:
        raise InputError(f"{path}: no feature specs")
    return specs


def _manifest_scene(row: Mapping[str, Any], where: str) -> str:
    value = row.get("scene_id")
    if not value and row.get("scene_dir"):
        value = Path(str(row["scene_dir"]).rstrip("/")).name
    scene = str(value or "").strip()
    if not scene or "/" in scene or "\\" in scene or scene in {".", ".."}:
        raise InputError(f"{where}: invalid scene_id={scene!r}")
    return scene


def load_manifest(path: Path, prefix: int, protocol: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys: set[tuple[str, int, int]] = set()
    for line_number, row in enumerate(_read_rows(path), 1):
        where = f"{path}:row {line_number}"
        scene = _manifest_scene(row, where)
        try:
            window_id = int(row["streaming_window_id"])
            raw_ids = tuple(int(item) for item in row["streaming_indices"])
        except (KeyError, TypeError, ValueError) as exc:
            raise InputError(f"{where}: invalid streaming window identity") from exc
        actual_prefix = int(row.get("prefix_length", len(raw_ids)))
        tail = int(row.get("streaming_tail", actual_prefix - 1))
        if actual_prefix != prefix or len(raw_ids) != prefix or tail != prefix - 1:
            raise InputError(
                f"{where}: expected prefix/tail {prefix}/{prefix - 1}, found "
                f"{actual_prefix}/{tail} with {len(raw_ids)} raw IDs"
            )
        if row.get("canonical_protocol") != protocol:
            raise InputError(
                f"{where}: canonical_protocol={row.get('canonical_protocol')!r}, "
                f"expected {protocol!r}"
            )
        if any(right <= left for left, right in zip(raw_ids, raw_ids[1:])):
            raise InputError(f"{where}: streaming_indices must be strictly increasing")
        key = (scene, window_id, tail)
        if key in keys:
            raise InputError(f"{where}: duplicate canonical key {key}")
        keys.add(key)
        rows.append(
            {
                "scene_id": scene,
                "window_id": window_id,
                "tail": tail,
                "raw_ids": raw_ids,
                "sample_id": str(row.get("sample_id", "")),
                "source": str(row.get("source", "")),
            }
        )
    if not rows:
        raise InputError(f"{path}: empty manifest")
    return rows


def _safe_relative_path(root: str, rendered: str, *, label: str) -> Path:
    base = Path(root).resolve()
    relative = Path(rendered)
    if relative.is_absolute():
        raise InputError(f"{label}: template rendered an absolute path")
    candidate = (base / relative).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise InputError(f"{label}: template escaped its scene root") from exc
    return candidate


def _format_template(template: str, values: Mapping[str, Any], *, label: str) -> str:
    try:
        return template.format(**values)
    except (KeyError, ValueError, TypeError) as exc:
        raise InputError(f"{label}: cannot render template {template!r}: {exc}") from exc


def _record_key(record: Mapping[str, Any]) -> tuple[int, int]:
    window = record.get("window_id", record.get("streaming_window_id"))
    tail = record.get("tail", record.get("streaming_tail"))
    if tail is None and record.get("prefix_length") is not None:
        tail = int(record["prefix_length"]) - 1
    return int(window), int(tail)


@dataclass
class IndexResult:
    status: str
    reason: str
    records: dict[tuple[int, int], dict[str, Any]]
    path: str


class IndexCache:
    def __init__(self, *, historical_sha_required: bool = True) -> None:
        self._values: dict[tuple[str, str], IndexResult] = {}
        self._historical_sha_required = historical_sha_required

    def get(self, root: LegacyRoot, scene: str) -> IndexResult:
        key = (root.root_id, scene)
        if key not in self._values:
            self._values[key] = self._load(root, scene)
        return self._values[key]

    def _load(self, root: LegacyRoot, scene: str) -> IndexResult:
        if _is_remote(root.index_scene_root):
            return IndexResult("UNVERIFIED", "remote_index_root", {}, root.index_scene_root)
        rendered = _format_template(
            root.index_path_template, {"scene": scene}, label=f"root {root.root_id} index"
        )
        path = _safe_relative_path(
            root.index_scene_root, rendered, label=f"root {root.root_id} index"
        )
        if not path.is_file():
            return IndexResult("REEXTRACT_MISSING", "prefix_index_missing", {}, str(path))
        try:
            values = np.load(path, allow_pickle=True).tolist()
            if isinstance(values, dict):
                values = [values]
            if not isinstance(values, list):
                raise TypeError(f"expected list-like object array, got {type(values).__name__}")
            records: dict[tuple[int, int], dict[str, Any]] = {}
            for offset, record in enumerate(values):
                if not isinstance(record, dict):
                    raise TypeError(f"record {offset} is not a dict")
                record_key = _record_key(record)
                if record_key in records:
                    raise ValueError(f"duplicate record key {record_key}")
                records[record_key] = record
        except Exception as exc:
            return IndexResult(
                "REEXTRACT_MISMATCH",
                f"prefix_index_unreadable:{type(exc).__name__}",
                {},
                str(path),
            )

        sidecar_path = Path(f"{path}.manifest.json")
        if sidecar_path.is_file():
            try:
                sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                if not isinstance(sidecar, dict):
                    raise TypeError("sidecar is not an object")
                if sidecar.get("file") != path.name:
                    raise ValueError("index sidecar file mismatch")
                if int(sidecar.get("size_bytes", -1)) != path.stat().st_size:
                    raise ValueError("index sidecar size mismatch")
                expected_hash = str(sidecar.get("sha256", "")).lower()
                if HEX_RE.fullmatch(expected_hash):
                    if sha256_file(path) != expected_hash:
                        raise ValueError("index sidecar sha256 mismatch")
                elif self._historical_sha_required:
                    raise ValueError("index sidecar sha256 missing or invalid")
            except Exception as exc:
                return IndexResult(
                    "REEXTRACT_MISMATCH",
                    f"prefix_index_sidecar_invalid:{type(exc).__name__}",
                    {},
                    str(path),
                )
        return IndexResult("OK", "prefix_index_loaded", records, str(path))


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


def _normalize_dtype(value: Any) -> str:
    text = str(value).strip()
    text = text.removeprefix("torch.").removeprefix("numpy.").removeprefix("np.")
    upper = text.upper()
    return DTYPE_ALIASES.get(upper, text.lower())


def _safetensors_tensor(path: Path, tensor_key: str) -> tuple[tuple[int, ...], str]:
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        size_bytes = handle.read(8)
        if len(size_bytes) != 8:
            raise ValueError("truncated safetensors length")
        header_size = struct.unpack("<Q", size_bytes)[0]
        if header_size <= 1 or header_size > min(file_size - 8, 128 * 1024 * 1024):
            raise ValueError("invalid safetensors header length")
        header = json.loads(handle.read(header_size).decode("utf-8"))
    tensor = header.get(tensor_key)
    if not isinstance(tensor, dict):
        raise ValueError(f"safetensors tensor {tensor_key!r} is missing")
    shape = tensor.get("shape")
    if not isinstance(shape, list):
        raise ValueError("safetensors shape is missing")
    parsed_shape = tuple(int(item) for item in shape)
    if any(item < 0 for item in parsed_shape):
        raise ValueError("safetensors shape contains a negative dimension")
    dtype = _normalize_dtype(tensor.get("dtype"))
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"unsupported safetensors dtype {tensor.get('dtype')!r}")
    offsets = tensor.get("data_offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or any(isinstance(item, bool) for item in offsets)
    ):
        raise ValueError("invalid safetensors data_offsets")
    start, end = (int(item) for item in offsets)
    payload_size = file_size - 8 - header_size
    if start < 0 or end < start or end > payload_size:
        raise ValueError("safetensors data_offsets exceed the payload")
    elements = 1
    for dimension in parsed_shape:
        elements *= dimension
    if end - start != elements * DTYPE_BYTES[dtype]:
        raise ValueError("safetensors tensor byte length disagrees with shape/dtype")
    return parsed_shape, dtype


def _raw_ids(record: Mapping[str, Any]) -> tuple[int, ...]:
    value = record.get("streaming_indices", record.get("indices"))
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise TypeError("record has no list-like streaming_indices/indices")
    return tuple(int(item) for item in value)


def _deep_get(value: Mapping[str, Any], dotted_key: str) -> tuple[bool, Any]:
    current: Any = value
    for component in dotted_key.split("."):
        if not isinstance(current, Mapping) or component not in current:
            return False, None
        current = current[component]
    return True, current


def _identity_value(meta: Mapping[str, Any], keys: Sequence[str]) -> tuple[bool, Any]:
    for key in keys:
        if key in meta:
            return True, meta[key]
    return False, None


def _filename_proves_integer(filename: str, field: str, expected: int) -> bool:
    pattern = re.compile(rf"(?:^|[_-]){re.escape(field)}_?(\d+)(?=[_.-]|$)")
    match = pattern.search(filename)
    return bool(match and int(match.group(1)) == expected)


@dataclass
class CandidateResult:
    status: str
    reasons: tuple[str, ...]
    root_id: str
    index_path: str = ""
    feature_path: str = ""
    observed_raw_ids: tuple[int, ...] | None = None
    source_payload_sha256: str | None = None
    source_size_bytes: int | None = None
    declared_source_payload_sha256: str | None = None
    payload_sha256_verified: bool = False


def _legacy_import_identity_contract_errors(spec: FeatureSpec) -> tuple[str, ...]:
    """Return missing generation-identity declarations for relaxed imports.

    A relaxed import may omit old RNG/extractor provenance, but it must not
    silently mix prompts or preprocessing.  ``expected_sidecar`` is the
    caller's explicit identity contract and can use dotted keys.
    """

    fields = set(spec.expected_sidecar)
    errors: list[str] = []
    if "prompt" not in fields:
        errors.append("spec_missing_expected_prompt_contract")
    has_explicit_preprocess = any(
        field == "preprocess"
        or field.startswith("preprocess.")
        or field.startswith("preprocess_")
        for field in fields
    )
    has_legacy_preprocess_bundle = {"frame_index_schema", "resize"} <= fields
    if not has_explicit_preprocess and not has_legacy_preprocess_bundle:
        errors.append("spec_missing_expected_preprocess_contract")
    return tuple(errors)


def _feature_name(spec: FeatureSpec, root: LegacyRoot, values: Mapping[str, Any]) -> str:
    if spec.feature_name:
        result = spec.feature_name
    else:
        template = spec.feature_name_template or root.feature_name_template
        result = _format_template(template, values, label=f"spec {spec.spec_id} feature name")
    if Path(result).name != result or result in {".", ".."}:
        raise InputError(f"spec {spec.spec_id}: feature_name must be a basename, found {result!r}")
    return result


def evaluate_candidate(
    *,
    spec: FeatureSpec,
    root: LegacyRoot,
    canonical: Mapping[str, Any],
    indexes: IndexCache,
    payload_hash_policy: str,
) -> CandidateResult:
    scene = str(canonical["scene_id"])
    index = indexes.get(root, scene)
    if index.status != "OK":
        return CandidateResult(index.status, (index.reason,), root.root_id, index.path)

    key = (int(canonical["window_id"]), int(canonical["tail"]))
    record = index.records.get(key)
    if record is None:
        return CandidateResult(
            "REEXTRACT_MISSING", ("prefix_record_missing",), root.root_id, index.path
        )
    try:
        actual_raw_ids = _raw_ids(record)
    except Exception as exc:
        return CandidateResult(
            "REEXTRACT_MISMATCH",
            (f"prefix_record_invalid:{type(exc).__name__}",),
            root.root_id,
            index.path,
        )
    if actual_raw_ids != tuple(canonical["raw_ids"]):
        return CandidateResult(
            "REEXTRACT_MISMATCH",
            ("raw_ids_mismatch",),
            root.root_id,
            index.path,
            observed_raw_ids=actual_raw_ids,
        )

    incomplete = spec.incomplete_identity_fields()
    if incomplete:
        return CandidateResult(
            "UNVERIFIED",
            tuple(f"spec_missing_{field}" for field in incomplete),
            root.root_id,
            index.path,
        )
    if payload_hash_policy in {"legacy-import", "reference"}:
        contract_errors = _legacy_import_identity_contract_errors(spec)
        if contract_errors:
            return CandidateResult(
                "UNVERIFIED",
                contract_errors,
                root.root_id,
                index.path,
            )
    if _is_remote(root.feature_scene_root):
        return CandidateResult(
            "UNVERIFIED",
            ("remote_feature_root",),
            root.root_id,
            index.path,
            root.feature_scene_root,
        )

    values = {
        "scene": scene,
        "window_id": int(canonical["window_id"]),
        "tail": int(canonical["tail"]),
        "prefix": spec.prefix,
        "vfm": spec.vfm,
        "model_id": spec.model_id,
        "layer": spec.layer,
        "timestep": spec.timestep,
        "spec_id": spec.spec_id,
    }
    try:
        feature_name = _feature_name(spec, root, values)
        values["feature_name"] = feature_name
        rendered = _format_template(
            root.feature_path_template, values, label=f"root {root.root_id} feature"
        )
        feature_path = _safe_relative_path(
            root.feature_scene_root, rendered, label=f"root {root.root_id} feature"
        )
    except InputError:
        raise
    if not feature_path.is_file():
        return CandidateResult(
            "REEXTRACT_MISSING",
            ("feature_payload_missing",),
            root.root_id,
            index.path,
            str(feature_path),
        )
    sidecar_path = Path(f"{feature_path}.manifest.json")
    if not sidecar_path.is_file():
        return CandidateResult(
            "REEXTRACT_MISSING",
            ("feature_sidecar_missing",),
            root.root_id,
            index.path,
            str(feature_path),
        )
    try:
        meta = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            raise TypeError("feature sidecar is not an object")
    except Exception as exc:
        return CandidateResult(
            "REEXTRACT_MISMATCH",
            (f"feature_sidecar_unreadable:{type(exc).__name__}",),
            root.root_id,
            index.path,
            str(feature_path),
        )

    reasons: list[str] = []
    if meta.get("file") != feature_path.name:
        reasons.append("sidecar_file_mismatch")
    try:
        if int(meta.get("size_bytes", -1)) != feature_path.stat().st_size:
            reasons.append("sidecar_size_mismatch")
    except (TypeError, ValueError, OSError):
        reasons.append("sidecar_size_invalid")

    found, actual_vfm = _identity_value(meta, ("vfm",))
    if not found:
        reasons.append("sidecar_vfm_missing")
    elif str(actual_vfm) != spec.vfm:
        reasons.append("sidecar_vfm_mismatch")
    found, actual_model = _identity_value(meta, ("model_id", "model_name", "model"))
    if not found:
        reasons.append("sidecar_model_id_missing")
    elif str(actual_model) != spec.model_id:
        reasons.append("sidecar_model_id_mismatch")

    found, actual_layer = _identity_value(meta, ("layer", "layer_idx"))
    if found:
        try:
            if int(actual_layer) != spec.layer:
                reasons.append("sidecar_layer_mismatch")
        except (TypeError, ValueError):
            reasons.append("sidecar_layer_invalid")
    elif not _filename_proves_integer(feature_path.name, "layer", int(spec.layer)):
        reasons.append("sidecar_layer_unproven")

    found, actual_timestep = _identity_value(meta, ("timestep", "t"))
    if spec.timestep is None:
        if found and actual_timestep not in (None, "", "none", "None"):
            reasons.append("sidecar_timestep_unexpected")
    elif not found:
        reasons.append("sidecar_timestep_missing")
    else:
        try:
            if int(actual_timestep) != spec.timestep:
                reasons.append("sidecar_timestep_mismatch")
        except (TypeError, ValueError):
            reasons.append("sidecar_timestep_invalid")
    if spec.expected_mode is not None:
        if meta.get("mode") != spec.expected_mode:
            reasons.append("sidecar_mode_mismatch")

    for field, expected in sorted(spec.expected_sidecar.items()):
        found, actual = _deep_get(meta, field)
        if not found:
            reasons.append(f"sidecar_expected_field_missing:{field}")
        elif actual != expected:
            reasons.append(f"sidecar_expected_field_mismatch:{field}")

    tensors = meta.get("tensors")
    tensor_meta = tensors.get(spec.tensor_key) if isinstance(tensors, dict) else None
    if not isinstance(tensor_meta, dict):
        reasons.append("sidecar_tensor_missing")
    else:
        try:
            sidecar_shape = tuple(int(item) for item in tensor_meta["shape"])
            if sidecar_shape != spec.expected_shape:
                reasons.append("sidecar_shape_mismatch")
        except Exception:
            reasons.append("sidecar_shape_invalid")
        if _normalize_dtype(tensor_meta.get("dtype")) != _normalize_dtype(spec.expected_dtype):
            reasons.append("sidecar_dtype_mismatch")

    try:
        actual_shape, actual_dtype = _safetensors_tensor(feature_path, spec.tensor_key)
        if actual_shape != spec.expected_shape:
            reasons.append("payload_shape_mismatch")
        if actual_dtype != _normalize_dtype(spec.expected_dtype):
            reasons.append("payload_dtype_mismatch")
    except Exception as exc:
        reasons.append(f"payload_header_invalid:{type(exc).__name__}")

    expected_hash = str(meta.get("sha256", "")).lower()
    declared_hash = expected_hash if HEX_RE.fullmatch(expected_hash) else None
    actual_hash: str | None = None
    if payload_hash_policy in {"full", "legacy-import"}:
        try:
            actual_hash = sha256_file(feature_path)
        except OSError:
            reasons.append("payload_sha256_unreadable")
    if HEX_RE.fullmatch(expected_hash):
        if actual_hash is not None and actual_hash != expected_hash:
            reasons.append("payload_sha256_mismatch")
    elif payload_hash_policy == "full":
        reasons.append("sidecar_sha256_invalid")

    if reasons:
        return CandidateResult(
            "REEXTRACT_MISMATCH",
            tuple(sorted(set(reasons))),
            root.root_id,
            index.path,
            str(feature_path),
            source_payload_sha256=actual_hash,
            source_size_bytes=feature_path.stat().st_size,
            declared_source_payload_sha256=declared_hash,
            payload_sha256_verified=(
                actual_hash is not None
                and declared_hash is not None
                and actual_hash == declared_hash
            ),
        )
    if payload_hash_policy == "sidecar":
        return CandidateResult(
            "UNVERIFIED",
            ("payload_sha256_not_computed",),
            root.root_id,
            index.path,
            str(feature_path),
            source_size_bytes=feature_path.stat().st_size,
            declared_source_payload_sha256=declared_hash,
        )
    if payload_hash_policy == "legacy-import":
        exact_reason = "legacy_exact_import_v1_identity_checks_passed"
    elif payload_hash_policy == "reference":
        exact_reason = "legacy_exact_reference_v1_identity_checks_passed"
    else:
        exact_reason = "all_identity_checks_passed"
    return CandidateResult(
        "REUSE_EXACT",
        (exact_reason,),
        root.root_id,
        index.path,
        str(feature_path),
        source_payload_sha256=actual_hash,
        source_size_bytes=feature_path.stat().st_size,
        declared_source_payload_sha256=declared_hash,
        payload_sha256_verified=(
            actual_hash is not None
            and declared_hash is not None
            and actual_hash == declared_hash
        ),
    )


def _choose_candidate(results: Sequence[CandidateResult]) -> CandidateResult:
    for status in ("REUSE_EXACT", "UNVERIFIED", "REEXTRACT_MISMATCH", "REEXTRACT_MISSING"):
        matches = [result for result in results if result.status == status]
        if matches:
            return matches[0]
    raise AssertionError("candidate result list is empty")


def classify(
    *,
    specs: Sequence[FeatureSpec],
    roots: Mapping[str, LegacyRoot],
    manifests: Mapping[int, tuple[Path, Sequence[Mapping[str, Any]]]],
    protocol: str,
    payload_hash_policy: str = "full",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if payload_hash_policy not in {"full", "sidecar", "legacy-import", "reference"}:
        raise InputError(f"invalid payload_hash_policy={payload_hash_policy!r}")
    indexes = IndexCache(
        historical_sha_required=payload_hash_policy not in {"legacy-import", "reference"}
    )
    details: list[dict[str, Any]] = []
    spec_summaries: list[dict[str, Any]] = []
    total_statuses: Counter[str] = Counter()
    for spec in specs:
        if spec.prefix not in manifests:
            raise InputError(f"spec {spec.spec_id}: no manifest for prefix {spec.prefix}")
        manifest_path, canonical_rows = manifests[spec.prefix]
        statuses: Counter[str] = Counter()
        reasons: Counter[str] = Counter()
        for canonical in canonical_rows:
            candidates = [
                evaluate_candidate(
                    spec=spec,
                    root=roots[root_id],
                    canonical=canonical,
                    indexes=indexes,
                    payload_hash_policy=payload_hash_policy,
                )
                for root_id in spec.root_ids
            ]
            chosen = _choose_candidate(candidates)
            statuses[chosen.status] += 1
            total_statuses[chosen.status] += 1
            reasons.update(chosen.reasons)
            details.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "spec_id": spec.spec_id,
                    "prefix": spec.prefix,
                    "vfm": spec.vfm,
                    "model_id": spec.model_id,
                    "layer": spec.layer,
                    "timestep": spec.timestep,
                    "tensor_key": spec.tensor_key,
                    "expected_sidecar_identity": spec.expected_sidecar,
                    "scene_id": canonical["scene_id"],
                    "sample_id": canonical["sample_id"],
                    "window_id": canonical["window_id"],
                    "tail": canonical["tail"],
                    "raw_ids": list(canonical["raw_ids"]),
                    "status": chosen.status,
                    "reason_codes": list(chosen.reasons),
                    "legacy_root_id": chosen.root_id,
                    "index_path": chosen.index_path,
                    "feature_path": chosen.feature_path,
                    "source_payload_sha256": chosen.source_payload_sha256,
                    "source_size_bytes": chosen.source_size_bytes,
                    "declared_source_payload_sha256": (
                        chosen.declared_source_payload_sha256
                    ),
                    "payload_sha256_verified": chosen.payload_sha256_verified,
                    "observed_raw_ids": (
                        list(chosen.observed_raw_ids)
                        if chosen.observed_raw_ids is not None
                        else None
                    ),
                    "candidate_statuses": {
                        item.root_id: {
                            "status": item.status,
                            "reason_codes": list(item.reasons),
                            "observed_raw_ids": (
                                list(item.observed_raw_ids)
                                if item.observed_raw_ids is not None
                                else None
                            ),
                        }
                        for item in candidates
                    },
                    "manifest": str(manifest_path),
                    "canonical_protocol": protocol,
                }
            )
        spec_summaries.append(
            {
                "spec_id": spec.spec_id,
                "prefix": spec.prefix,
                "vfm": spec.vfm,
                "model_id": spec.model_id,
                "layer": spec.layer,
                "timestep": spec.timestep,
                "legacy_root_ids": list(spec.root_ids),
                "records": len(canonical_rows),
                "status_counts": {status: statuses.get(status, 0) for status in STATUSES},
                "reason_counts": dict(sorted(reasons.items())),
                "all_reuse_exact": statuses.get("REUSE_EXACT", 0) == len(canonical_rows),
            }
        )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "audit_type": "canonical_v2_feature_reuse_classification",
        "read_only": True,
        "canonical_protocol": protocol,
        "payload_hash_policy": payload_hash_policy,
        "reuse_exact_requires_full_payload_sha256": payload_hash_policy in {
            "full",
            "legacy-import",
        },
        "historical_sidecar_sha256_required": payload_hash_policy not in {
            "legacy-import",
            "reference",
        },
        "payload_full_read_performed": payload_hash_policy in {"full", "legacy-import"},
        "byte_level_certified": payload_hash_policy in {"full", "legacy-import"},
        "generation_contract": (
            "LEGACY_EXACT_REFERENCE_V1"
            if payload_hash_policy == "reference"
            else (
                "LEGACY_EXACT_IMPORT_V1"
                if payload_hash_policy == "legacy-import"
                else "HISTORICAL_SHA_BOUND_V1"
            )
        ),
        "spec_count": len(specs),
        "record_count": len(details),
        "status_counts": {status: total_statuses.get(status, 0) for status in STATUSES},
        "all_reuse_exact": total_statuses.get("REUSE_EXACT", 0) == len(details),
        "specs": spec_summaries,
    }
    return details, summary


def write_outputs(
    output_dir: Path,
    details: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    by_status: dict[str, list[Mapping[str, Any]]] = {status: [] for status in STATUSES}
    for row in details:
        by_status[str(row["status"])].append(row)
    for status, rows in by_status.items():
        name = status.lower() + ".jsonl"
        text = "".join(_compact_json(row) + "\n" for row in rows)
        _write_text_atomic(output_dir / name, text)
    _write_text_atomic(
        output_dir / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )

    columns = [
        "spec_id",
        "prefix",
        "vfm",
        "model_id",
        "layer",
        "timestep",
        "records",
        *STATUSES,
        "all_reuse_exact",
        "legacy_root_ids",
        "reason_counts",
    ]
    rows: list[dict[str, Any]] = []
    for spec in summary["specs"]:
        rows.append(
            {
                "spec_id": spec["spec_id"],
                "prefix": spec["prefix"],
                "vfm": spec["vfm"],
                "model_id": spec["model_id"] or "",
                "layer": "" if spec["layer"] is None else spec["layer"],
                "timestep": "" if spec["timestep"] is None else spec["timestep"],
                "records": spec["records"],
                **spec["status_counts"],
                "all_reuse_exact": str(spec["all_reuse_exact"]).lower(),
                "legacy_root_ids": _compact_json(spec["legacy_root_ids"]),
                "reason_counts": _compact_json(spec["reason_counts"]),
            }
        )
    buffer: list[str] = []
    # csv.writer requires a file-like object; StringIO keeps the final write atomic.
    import io

    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    buffer.append(handle.getvalue())
    _write_text_atomic(output_dir / "spec_summary.csv", "".join(buffer))


def _manifest_arguments(values: Iterable[str], manifest_dir: Path | None) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for value in values:
        if "=" not in value:
            raise InputError(f"--manifest must be PREFIX=PATH, found {value!r}")
        raw_prefix, raw_path = value.split("=", 1)
        prefix = int(raw_prefix)
        if prefix in result:
            raise InputError(f"duplicate manifest for prefix {prefix}")
        result[prefix] = Path(raw_path)
    if manifest_dir is not None:
        for prefix in (8, 12, 16, 24):
            candidate = manifest_dir / f"inscene_c_h124_prefix{prefix}.jsonl"
            if candidate.is_file() and prefix not in result:
                result[prefix] = candidate
    if not result:
        raise InputError("provide --manifest PREFIX=PATH or --manifest-dir")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--specs", type=Path, required=True, help="Feature spec JSON/JSONL/CSV/TSV"
    )
    parser.add_argument("--legacy-roots", type=Path, required=True, help="Legacy root inventory")
    parser.add_argument("--manifest", action="append", default=[], metavar="PREFIX=PATH")
    parser.add_argument("--manifest-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--payload-hash-policy",
        choices=("full", "sidecar", "legacy-import", "reference"),
        default="full",
        help=(
            "sidecar skips payload hashing and can never emit REUSE_EXACT; "
            "legacy-import recomputes SHA but permits a missing historical SHA; "
            "reference emits non-byte-certified exact routes without reading tensor data"
        ),
    )
    parser.add_argument(
        "--require-all-exact",
        action="store_true",
        help="Exit 2 after writing reports if any record is not REUSE_EXACT",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    roots = load_roots(args.legacy_roots)
    specs = load_specs(args.specs, roots)
    manifest_paths = _manifest_arguments(args.manifest, args.manifest_dir)
    manifests: dict[int, tuple[Path, Sequence[Mapping[str, Any]]]] = {}
    for prefix, path in sorted(manifest_paths.items()):
        manifests[prefix] = (path.resolve(), load_manifest(path, prefix, args.protocol))
    details, summary = classify(
        specs=specs,
        roots=roots,
        manifests=manifests,
        protocol=args.protocol,
        payload_hash_policy=args.payload_hash_policy,
    )
    summary = dict(summary)
    summary["inputs"] = {
        "specs": {"path": str(args.specs.resolve()), "sha256": sha256_file(args.specs)},
        "legacy_roots": {
            "path": str(args.legacy_roots.resolve()),
            "sha256": sha256_file(args.legacy_roots),
        },
        "manifests": {
            str(prefix): {"path": str(path), "sha256": sha256_file(path)}
            for prefix, (path, _rows) in sorted(manifests.items())
        },
    }
    write_outputs(args.output_dir, details, summary)
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "spec_count",
                    "record_count",
                    "status_counts",
                    "all_reuse_exact",
                )
            },
            sort_keys=True,
        )
    )
    if args.require_all_exact and not summary["all_reuse_exact"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
