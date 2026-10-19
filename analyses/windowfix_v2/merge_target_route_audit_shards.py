#!/usr/bin/env python3
"""Merge hash-bound target-route audit shards into one parent-plan audit.

The remote target audit renderer partitions the immutable target merge plan by
VFM.  Each worker consequently emits a valid audit of its *shard* plan.  This
program verifies that the reviewed shards are an exact, disjoint partition of
the parent plan, verifies every worker binding and output byte, and publishes a
new immutable audit bound to the parent plan.  Definite gaps are regrouped into
fresh unified worklists; pending remote gates remain pending and are never
silently converted into extraction work.

No feature or AOSS object is opened.  Only the local canonical manifest,
review artifacts, and completed audit outputs are read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyses.windowfix_v2.audit_target_row_routes import (
    AUDIT_TYPE,
    CANONICAL_PROTOCOL,
    CONTRACT,
    PLAN_TYPE,
    SCHEMA_VERSION,
    _normalize_unit,
    _slug,
    build_gap_worklists,
    canonical_json_sha,
    canonical_target_unions,
    sha256_file,
    validate_destination_root,
)


REVIEW_TYPE = "canonical_v2_target_route_audit_sco_review"
BINDING_TYPE = "canonical_v2_target_route_audit_worker_binding"
BINDING_STATUS = "REVIEW_ONLY_NOT_EXECUTED"
BINDING_SCHEMA_ID = "canonical_v2_target_route_audit_worker_binding_v1.schema.json"
SAFE_AUDIT_STATUSES = {"PASS", "INCOMPLETE", "PENDING"}
HEX = frozenset("0123456789abcdef")


class ShardMergeError(ValueError):
    """A reviewed shard, worker binding, or audit output is not exact-safe."""


def _is_sha(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and not (set(text) - HEX)


def _regular(path: Path, label: str, *, absolute: bool = True) -> Path:
    if absolute and not path.is_absolute():
        raise ShardMergeError(f"{label} must be absolute: {path}")
    if path.is_symlink() or not path.is_file():
        raise ShardMergeError(f"{label} must be a regular non-symlink file: {path}")
    return path.resolve()


def _json(path: Path, label: str, *, absolute: bool = True) -> dict[str, Any]:
    path = _regular(path, label, absolute=absolute)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ShardMergeError(f"cannot parse {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ShardMergeError(f"{label} must contain a JSON object: {path}")
    return value


def _verified_content(value: Mapping[str, Any], label: str) -> str:
    recorded = value.get("content_sha256")
    actual = canonical_json_sha(
        {key: item for key, item in value.items() if key != "content_sha256"}
    )
    if not _is_sha(recorded) or recorded != actual:
        raise ShardMergeError(f"{label} content SHA256 mismatch")
    return actual


def _bound_file(value: Any, label: str, *, sha_field: str = "sha256") -> Path:
    if not isinstance(value, dict):
        raise ShardMergeError(f"{label} binding must be an object")
    path = _regular(Path(str(value.get("path", ""))), label)
    expected = value.get(sha_field)
    if not _is_sha(expected) or sha256_file(path) != expected:
        raise ShardMergeError(f"{label} file SHA256 mismatch")
    return path


def _unit_key(value: Mapping[str, Any]) -> tuple[Any, ...]:
    try:
        return (
            str(value["spec_id"]),
            str(value["vfm"]),
            str(value["model_id"]),
            int(value["timestep"]),
            int(value["layer"]),
            str(value["source"]),
            str(value["scene_id"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ShardMergeError(f"malformed target unit identity: {value!r}") from exc


def _route_sort_key(value: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(value["vfm"]),
        str(value["model_id"]),
        int(value["timestep"]),
        int(value["layer"]),
        str(value["source"]),
        str(value["scene_id"]),
        int(value["canonical_row_index"]),
    )


def _unit_sort_key(value: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(value["vfm"]),
        str(value["model_id"]),
        int(value["timestep"]),
        int(value["layer"]),
        str(value["source"]),
        str(value["scene_id"]),
    )


def _jsonl_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode()


def _iter_bound_jsonl(
    binding: Any,
    *,
    expected_path: Path,
    label: str,
) -> Iterator[dict[str, Any]]:
    if not isinstance(binding, dict):
        raise ShardMergeError(f"{label} output binding must be an object")
    path = _regular(Path(str(binding.get("path", ""))), label)
    if path != expected_path.resolve():
        raise ShardMergeError(f"{label} path escapes the reviewed audit output")
    expected_sha = binding.get("sha256")
    if not _is_sha(expected_sha) or sha256_file(path) != expected_sha:
        raise ShardMergeError(f"{label} SHA256 mismatch")
    expected_records = binding.get("records")
    if isinstance(expected_records, bool) or not isinstance(expected_records, int):
        raise ShardMergeError(f"{label} record count is malformed")
    records = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ShardMergeError(
                    f"{label} line {line_number} is invalid JSON"
                ) from exc
            if not isinstance(value, dict):
                raise ShardMergeError(f"{label} line {line_number} is not an object")
            records += 1
            yield value
    if records != expected_records:
        raise ShardMergeError(
            f"{label} record count mismatch: expected {expected_records}, found {records}"
        )


def _validate_parent_plan(plan_path: Path) -> dict[str, Any]:
    plan = _json(plan_path, "parent target merge plan")
    if (
        plan.get("schema_version") != SCHEMA_VERSION
        or plan.get("plan_type") != PLAN_TYPE
        or plan.get("contract") != CONTRACT
        or plan.get("canonical_protocol") != CANONICAL_PROTOCOL
    ):
        raise ShardMergeError("parent target merge plan contract/type mismatch")
    _verified_content(plan, "parent target merge plan")
    protocol_path = _regular(Path(str(plan.get("protocol_file", ""))), "protocol file")
    if sha256_file(protocol_path) != plan.get("protocol_file_sha256"):
        raise ShardMergeError("parent plan protocol file SHA256 mismatch")
    protocol = _json(protocol_path, "protocol file")
    if protocol.get("canonical_protocol") != CANONICAL_PROTOCOL:
        raise ShardMergeError("canonical protocol identity mismatch")
    canonical = plan.get("canonical_source")
    if not isinstance(canonical, dict):
        raise ShardMergeError("parent plan lacks canonical_source binding")
    canonical_path = Path(str(canonical.get("path", "")))
    # This is deliberately local-only.  The merge operation must never turn
    # into another AOSS gate.
    canonical_path = _regular(canonical_path, "canonical source")
    targets, canonical_metadata = canonical_target_unions(
        canonical_path, str(canonical.get("sha256", ""))
    )
    destination_root = validate_destination_root(plan.get("destination_root"))
    raw_units = plan.get("units")
    if not isinstance(raw_units, list) or not raw_units:
        raise ShardMergeError("parent plan contains no units")
    normalized: list[dict[str, Any]] = []
    unit_map: dict[tuple[Any, ...], dict[str, Any]] = {}
    for raw in raw_units:
        if not isinstance(raw, dict):
            raise ShardMergeError("parent plan contains a non-object unit")
        try:
            unit = _normalize_unit(
                raw, canonical_targets=targets, output_root=destination_root
            )
        except Exception as exc:
            raise ShardMergeError(f"invalid parent target unit: {exc}") from exc
        key = _unit_key(unit)
        if key in unit_map:
            raise ShardMergeError(f"duplicate parent target unit: {key}")
        unit_map[key] = unit
        normalized.append(unit)
    return {
        "plan": plan,
        "plan_file_sha256": sha256_file(plan_path),
        "protocol": protocol,
        "protocol_path": protocol_path,
        "canonical_metadata": canonical_metadata,
        "targets": targets,
        "destination_root": destination_root,
        "normalized_units": normalized,
        "unit_map": unit_map,
    }


def _validate_review_manifest(
    manifest_path: Path,
    *,
    plan_path: Path,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
) -> dict[str, Any]:
    manifest = _json(manifest_path, "target audit review manifest")
    _verified_content(manifest, "target audit review manifest")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("plan_type") != REVIEW_TYPE
        or manifest.get("status") != "READY_FOR_REVIEW_NOT_SUBMITTED"
        or manifest.get("submission_performed") is not False
        or manifest.get("feature_payload_written") is not False
        or manifest.get("remote_policy") != "gate"
        or manifest.get("verify_local_source_sha") is not True
    ):
        raise ShardMergeError("review manifest safety contract mismatch")
    timeout = manifest.get("source_timeout_seconds")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ShardMergeError("review manifest source timeout is invalid")
    source_plan = manifest.get("source_plan")
    expected_plan = {
        "path": str(plan_path),
        "file_sha256": plan_file_sha256,
        "content_sha256": plan["content_sha256"],
    }
    if source_plan != expected_plan:
        raise ShardMergeError("review manifest does not bind the parent plan")

    official = {
        "auditor": PROJECT_ROOT / "analyses/windowfix_v2/audit_target_row_routes.py",
        "source_reader": PROJECT_ROOT / "analyses/windowfix_v2/target_row_source.py",
        "worker_wrapper": PROJECT_ROOT
        / "scripts/windowfix_v2/run_target_route_audit_worker.py",
        "worker_binding_schema": PROJECT_ROOT
        / "scripts/windowfix_v2/target_route_audit_worker_binding_v1.schema.json",
    }
    bound_paths: dict[str, Path] = {}
    for label, expected_path in official.items():
        bound = _bound_file(manifest.get(label), f"manifest {label}")
        if bound != expected_path.resolve():
            raise ShardMergeError(f"manifest {label} is not the repository implementation")
        bound_paths[label] = bound
    schema = _json(bound_paths["worker_binding_schema"], "worker binding schema")
    if schema.get("$id") != BINDING_SCHEMA_ID:
        raise ShardMergeError("worker binding schema identity mismatch")
    if manifest["worker_binding_schema"].get("schema_id") != BINDING_SCHEMA_ID:
        raise ShardMergeError("review manifest binding schema ID mismatch")
    if bound_paths["source_reader"] != bound_paths["auditor"].with_name(
        "target_row_source.py"
    ):
        raise ShardMergeError("reviewed auditor would import a different source reader")
    _bound_file(manifest.get("worker_commands"), "review worker commands")
    _bound_file(manifest.get("sco_create_commands"), "review SCO commands")

    shards = manifest.get("shards")
    counts = manifest.get("counts")
    if not isinstance(shards, list) or not shards or not isinstance(counts, dict):
        raise ShardMergeError("review manifest shards/counts are malformed")
    if (
        counts.get("shards") != len(shards)
        or counts.get("worker_binding_receipts") != len(shards)
        or counts.get("units") != len(plan["units"])
        or counts.get("sco_commands_rendered") != len(shards)
        or counts.get("sco_commands_executed") != 0
    ):
        raise ShardMergeError("review manifest aggregate counts mismatch")
    return {"manifest": manifest, "paths": bound_paths, "timeout": float(timeout)}


def _validate_binding_receipt(
    shard_record: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    plan_path: Path,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    shard_path: Path,
    shard: Mapping[str, Any],
    shard_file_sha256: str,
    reviewed_paths: Mapping[str, Path],
) -> tuple[dict[str, Any], Path]:
    receipt_path = _regular(
        Path(str(shard_record.get("worker_binding_receipt", ""))),
        "worker binding receipt",
    )
    if sha256_file(receipt_path) != shard_record.get(
        "worker_binding_receipt_file_sha256"
    ):
        raise ShardMergeError("worker binding receipt file SHA256 mismatch")
    receipt = _json(receipt_path, "worker binding receipt")
    content_sha = _verified_content(receipt, "worker binding receipt")
    if content_sha != shard_record.get("worker_binding_receipt_content_sha256"):
        raise ShardMergeError("manifest/worker binding receipt content SHA mismatch")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("receipt_type") != BINDING_TYPE
        or receipt.get("status") != BINDING_STATUS
    ):
        raise ShardMergeError("worker binding receipt contract/status mismatch")
    expected_parent = {
        "path": str(plan_path),
        "file_sha256": plan_file_sha256,
        "content_sha256": plan["content_sha256"],
    }
    expected_shard = {
        "path": str(shard_path),
        "file_sha256": shard_file_sha256,
        "content_sha256": shard["content_sha256"],
    }
    if receipt.get("parent_plan") != expected_parent:
        raise ShardMergeError("worker binding parent-plan mismatch")
    if receipt.get("shard") != expected_shard:
        raise ShardMergeError("worker binding shard-plan mismatch")
    for receipt_key, manifest_key in (
        ("auditor", "auditor"),
        ("source_reader", "source_reader"),
        ("worker_wrapper", "worker_wrapper"),
    ):
        expected = {
            "path": str(reviewed_paths[manifest_key]),
            "file_sha256": manifest[manifest_key]["sha256"],
        }
        if receipt.get(receipt_key) != expected:
            raise ShardMergeError(f"worker binding {receipt_key} lineage mismatch")
    expected_schema = {
        "path": str(reviewed_paths["worker_binding_schema"]),
        "file_sha256": manifest["worker_binding_schema"]["sha256"],
        "schema_id": BINDING_SCHEMA_ID,
    }
    if receipt.get("binding_schema") != expected_schema:
        raise ShardMergeError("worker binding schema lineage mismatch")
    audit_output = Path(str(receipt.get("audit_output", "")))
    if not audit_output.is_absolute() or audit_output.is_symlink():
        raise ShardMergeError("worker audit output path is unsafe")
    if str(audit_output) != str(shard_record.get("audit_output", "")):
        raise ShardMergeError("manifest/worker audit output path mismatch")
    arguments = receipt.get("audit_arguments")
    expected_arguments = {
        "remote_policy": "gate",
        "source_timeout_seconds": manifest["source_timeout_seconds"],
        "verify_source_sha": True,
    }
    if arguments != expected_arguments:
        raise ShardMergeError("worker binding audit arguments mismatch")
    return receipt, audit_output.resolve()


def _validate_review_shards(
    *,
    manifest_data: Mapping[str, Any],
    parent_data: Mapping[str, Any],
    plan_path: Path,
) -> list[dict[str, Any]]:
    manifest = manifest_data["manifest"]
    plan = parent_data["plan"]
    raw_units = plan["units"]
    vfms = sorted({str(item["vfm"]) for item in raw_units})
    records = manifest["shards"]
    if len(records) != len(vfms):
        raise ShardMergeError("review shard count does not equal parent-plan VFM count")
    seen_vfms: set[str] = set()
    seen_units: set[tuple[Any, ...]] = set()
    validated: list[dict[str, Any]] = []
    for expected_ordinal, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise ShardMergeError("review manifest contains a non-object shard")
        vfm = str(record.get("vfm", ""))
        if record.get("ordinal") != expected_ordinal or vfm != vfms[expected_ordinal - 1]:
            raise ShardMergeError("review shards are not in contiguous sorted VFM order")
        if vfm in seen_vfms:
            raise ShardMergeError(f"duplicate review VFM shard: {vfm}")
        seen_vfms.add(vfm)
        shard_path = _regular(Path(str(record.get("plan", ""))), f"{vfm} shard plan")
        shard_file_sha = sha256_file(shard_path)
        if shard_file_sha != record.get("plan_file_sha256"):
            raise ShardMergeError(f"{vfm} shard-plan file SHA256 mismatch")
        shard = _json(shard_path, f"{vfm} shard plan")
        content_sha = _verified_content(shard, f"{vfm} shard plan")
        if content_sha != record.get("plan_content_sha256"):
            raise ShardMergeError(f"{vfm} shard-plan content SHA mismatch")
        expected_units = [item for item in raw_units if str(item["vfm"]) == vfm]
        parent_binding = {
            "path": str(plan_path),
            "file_sha256": parent_data["plan_file_sha256"],
            "content_sha256": plan["content_sha256"],
        }
        expected_shard = {
            key: value for key, value in plan.items() if key not in {"content_sha256", "units"}
        }
        expected_shard["parent_plan"] = parent_binding
        expected_shard["units"] = expected_units
        expected_shard["content_sha256"] = canonical_json_sha(expected_shard)
        if shard != expected_shard:
            raise ShardMergeError(f"{vfm} shard is not the exact parent-plan VFM partition")
        unit_keys = [_unit_key(item) for item in expected_units]
        overlap = seen_units.intersection(unit_keys)
        if overlap:
            raise ShardMergeError(f"overlapping target units across shards: {next(iter(overlap))}")
        seen_units.update(unit_keys)
        layers = sorted({int(item["layer"]) for item in expected_units})
        scenes = len({str(item["scene_id"]) for item in expected_units})
        if (
            record.get("units") != len(expected_units)
            or record.get("layers") != layers
            or record.get("scenes") != scenes
        ):
            raise ShardMergeError(f"{vfm} manifest shard counts mismatch")
        receipt, audit_output = _validate_binding_receipt(
            record,
            manifest=manifest,
            plan_path=plan_path,
            plan=plan,
            plan_file_sha256=parent_data["plan_file_sha256"],
            shard_path=shard_path,
            shard=shard,
            shard_file_sha256=shard_file_sha,
            reviewed_paths=manifest_data["paths"],
        )
        validated.append(
            {
                "vfm": vfm,
                "record": record,
                "shard": shard,
                "shard_path": shard_path,
                "shard_file_sha256": shard_file_sha,
                "unit_keys": unit_keys,
                "receipt": receipt,
                "receipt_path": Path(record["worker_binding_receipt"]).resolve(),
                "audit_output": audit_output,
            }
        )
    parent_keys = set(parent_data["unit_map"])
    if seen_units != parent_keys:
        raise ShardMergeError(
            "review shards omit or add parent-plan units: "
            f"missing={len(parent_keys - seen_units)} extra={len(seen_units - parent_keys)}"
        )
    return validated


def _validate_row_common(
    row: Mapping[str, Any],
    *,
    expected_kind: str,
    units: Mapping[tuple[Any, ...], Mapping[str, Any]],
) -> tuple[tuple[Any, ...], Mapping[str, Any]]:
    if (
        row.get("schema_version") != SCHEMA_VERSION
        or row.get("contract") != CONTRACT
        or row.get("unit_kind") != expected_kind
    ):
        raise ShardMergeError(f"{expected_kind} row contract mismatch")
    key = _unit_key(row)
    unit = units.get(key)
    if unit is None:
        raise ShardMergeError(f"{expected_kind} row is outside its reviewed shard: {key}")
    if row.get("destination_path") != unit["destination_path"]:
        raise ShardMergeError(f"{expected_kind} row destination mismatch: {key}")
    return key, unit


def _validate_gap_worklist_outputs(
    *,
    report: Mapping[str, Any],
    missing: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    bindings = report.get("outputs", {}).get("gap_worklists")
    if not isinstance(bindings, list):
        raise ShardMergeError("shard audit gap_worklists binding is malformed")
    expected_values = build_gap_worklists({"report": report, "missing_units": missing})
    expected = {item["content_sha256"]: item for item in expected_values}
    if len(expected) != len(expected_values):
        raise ShardMergeError("derived shard gap worklists have a content collision")
    seen: set[str] = set()
    for binding in bindings:
        if not isinstance(binding, dict):
            raise ShardMergeError("shard gap worklist binding is not an object")
        path = _regular(Path(str(binding.get("path", ""))), "shard gap worklist")
        try:
            path.relative_to((output_dir / "gap_worklists").resolve())
        except ValueError as exc:
            raise ShardMergeError("shard gap worklist escapes its audit output") from exc
        if sha256_file(path) != binding.get("file_sha256"):
            raise ShardMergeError("shard gap worklist file SHA256 mismatch")
        value = _json(path, "shard gap worklist")
        content_sha = _verified_content(value, "shard gap worklist")
        if content_sha != binding.get("content_sha256") or content_sha in seen:
            raise ShardMergeError("shard gap worklist content binding is duplicate/mismatched")
        seen.add(content_sha)
        if value != expected.get(content_sha) or binding.get("counts") != value.get("counts"):
            raise ShardMergeError("shard gap worklist does not exactly derive from missing rows")
    if seen != set(expected):
        raise ShardMergeError("shard gap worklists omit or add definite missing rows")


def _validate_shard_audit(
    shard_data: Mapping[str, Any],
    *,
    report_path: Path,
    parent_data: Mapping[str, Any],
    manifest_data: Mapping[str, Any],
    route_output: BinaryIO,
    route_digest: Any,
) -> dict[str, Any]:
    vfm = shard_data["vfm"]
    report = _json(report_path, f"{vfm} route audit")
    report_content_sha = _verified_content(report, f"{vfm} route audit")
    paths = manifest_data["paths"]
    common_expected = {
        "schema_version": SCHEMA_VERSION,
        "audit_type": AUDIT_TYPE,
        "contract": CONTRACT,
        "canonical_protocol": CANONICAL_PROTOCOL,
        "canonical_protocol_sha256": parent_data["protocol"].get(
            "canonical_protocol_sha256"
        ),
        "protocol_file": str(parent_data["protocol_path"]),
        "protocol_file_sha256": sha256_file(parent_data["protocol_path"]),
        "canonical_source": parent_data["canonical_metadata"],
        "plan": str(shard_data["shard_path"]),
        "plan_sha256": shard_data["shard_file_sha256"],
        "auditor_source": str(paths["auditor"]),
        "auditor_source_sha256": sha256_file(paths["auditor"]),
        "source_reader_source": str(paths["source_reader"]),
        "source_reader_source_sha256": sha256_file(paths["source_reader"]),
        "destination_root": str(parent_data["destination_root"]),
        "verify_source_sha": True,
        "remote_policy": "gate",
        "source_timeout_seconds": shard_data["receipt"]["audit_arguments"][
            "source_timeout_seconds"
        ],
    }
    for field, expected in common_expected.items():
        if report.get(field) != expected:
            raise ShardMergeError(f"{vfm} route audit {field} lineage mismatch")
    if (
        report.get("status") not in SAFE_AUDIT_STATUSES
        or report.get("dry_run") is not True
        or report.get("feature_payload_written") is not False
        or report.get("seeded_namespace_written") is not False
        or report.get("hard_errors") != []
        or report.get("hard_errors_truncated") is not False
    ):
        raise ShardMergeError(f"{vfm} route audit is not safe to merge")
    outputs = report.get("outputs")
    if not isinstance(outputs, dict):
        raise ShardMergeError(f"{vfm} route audit has no output bindings")
    output_dir = shard_data["audit_output"]
    if report_path != (output_dir / "target_row_route_audit.json").resolve():
        raise ShardMergeError(f"{vfm} report path differs from worker binding")
    unit_keys = set(shard_data["unit_keys"])
    units = {key: parent_data["unit_map"][key] for key in unit_keys}
    routed: dict[tuple[Any, ...], set[int]] = defaultdict(set)
    missing_by_unit: dict[tuple[Any, ...], set[int]] = defaultdict(set)
    pending_by_unit: dict[tuple[Any, ...], set[int]] = defaultdict(set)
    routes_by_kind: Counter[str] = Counter()
    route_records = 0
    previous_sort_key: tuple[Any, ...] | None = None
    for row in _iter_bound_jsonl(
        outputs.get("target_row_routes"),
        expected_path=output_dir / "target_row_routes.jsonl",
        label=f"{vfm} target row routes",
    ):
        key, unit = _validate_row_common(
            row, expected_kind="target_row", units=units
        )
        try:
            raw_id = int(row["target_raw_id"])
            canonical_index = int(row["canonical_row_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ShardMergeError(f"{vfm} route raw-ID/index is malformed") from exc
        expected_ids = unit["expected_raw_ids"]
        if (
            canonical_index < 0
            or canonical_index >= len(expected_ids)
            or int(expected_ids[canonical_index]) != raw_id
        ):
            raise ShardMergeError(f"{vfm} route canonical row index mismatch")
        if raw_id in routed[key]:
            raise ShardMergeError(f"duplicate routed raw ID for {key}: {raw_id}")
        routed[key].add(raw_id)
        if row.get("tensor_key") != unit["tensor_key"]:
            raise ShardMergeError(f"{vfm} route tensor key mismatch")
        sort_key = _route_sort_key(row)
        if previous_sort_key is not None and sort_key <= previous_sort_key:
            raise ShardMergeError(f"{vfm} route output is not strictly sorted")
        previous_sort_key = sort_key
        routes_by_kind[str(row.get("source_kind", ""))] += 1
        payload = _jsonl_bytes(row)
        route_output.write(payload)
        route_digest.update(payload)
        route_records += 1

    missing_rows = list(
        _iter_bound_jsonl(
            outputs.get("missing_target_units"),
            expected_path=output_dir / "missing_target_units.jsonl",
            label=f"{vfm} missing target units",
        )
    )
    seen_missing: set[tuple[Any, ...]] = set()
    for row in missing_rows:
        key, unit = _validate_row_common(row, expected_kind="target_gap", units=units)
        if key in seen_missing:
            raise ShardMergeError(f"duplicate missing target unit: {key}")
        seen_missing.add(key)
        try:
            raw_ids = tuple(int(value) for value in row["missing_raw_ids"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ShardMergeError(f"malformed missing raw IDs: {key}") from exc
        if not raw_ids or raw_ids != tuple(sorted(set(raw_ids))):
            raise ShardMergeError(f"missing raw IDs are empty/duplicate/unsorted: {key}")
        if (
            row.get("status") != "REEXTRACT_MISSING"
            or row.get("tensor_key") != unit["tensor_key"]
            or row.get("expected_tail_shape") != unit["expected_tail_shape"]
            or row.get("expected_dtype") != unit["expected_dtype"]
            or row.get("prompt_contract") != unit["prompt_contract"]
            or row.get("preprocess_identity") != unit["preprocess_identity"]
            or row.get("canonical_source_sha256")
            != parent_data["canonical_metadata"]["sha256"]
        ):
            raise ShardMergeError(f"missing target unit semantic mismatch: {key}")
        missing_by_unit[key].update(raw_ids)

    pending_rows = list(
        _iter_bound_jsonl(
            outputs.get("pending_target_units"),
            expected_path=output_dir / "pending_target_units.jsonl",
            label=f"{vfm} pending target units",
        )
    )
    seen_pending: set[tuple[Any, ...]] = set()
    for row in pending_rows:
        key, _unit = _validate_row_common(
            row, expected_kind="target_pending", units=units
        )
        if key in seen_pending:
            raise ShardMergeError(f"duplicate pending target unit: {key}")
        seen_pending.add(key)
        try:
            raw_ids = tuple(int(value) for value in row["pending_raw_ids"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ShardMergeError(f"malformed pending raw IDs: {key}") from exc
        if (
            not raw_ids
            or raw_ids != tuple(sorted(set(raw_ids)))
            or row.get("status") != "PENDING_AOSS_GATE"
        ):
            raise ShardMergeError(f"pending raw IDs are invalid: {key}")
        pending_by_unit[key].update(raw_ids)

    unit_reports_value = report.get("units")
    if not isinstance(unit_reports_value, list):
        raise ShardMergeError(f"{vfm} route audit units list is malformed")
    unit_reports: dict[tuple[Any, ...], dict[str, Any]] = {}
    for value in unit_reports_value:
        if not isinstance(value, dict):
            raise ShardMergeError(f"{vfm} unit report is not an object")
        key = _unit_key(value)
        if key not in units or key in unit_reports:
            raise ShardMergeError(f"{vfm} unit report is duplicate/out of shard: {key}")
        unit_reports[key] = value
    if set(unit_reports) != unit_keys:
        raise ShardMergeError(f"{vfm} unit reports do not cover the shard exactly")

    for key, unit in units.items():
        expected_ids = set(int(value) for value in unit["expected_raw_ids"])
        route_ids = routed.get(key, set())
        missing_ids = missing_by_unit.get(key, set())
        pending_ids = pending_by_unit.get(key, set())
        if (
            route_ids & missing_ids
            or route_ids & pending_ids
            or missing_ids & pending_ids
            or route_ids | missing_ids | pending_ids != expected_ids
        ):
            raise ShardMergeError(f"route/missing/pending do not partition canonical rows: {key}")
        unit_report = unit_reports[key]
        expected_counts = {
            "expected_rows": len(expected_ids),
            "routed_rows": len(route_ids),
            "missing_rows": len(missing_ids),
            "pending_rows": len(pending_ids),
        }
        for field, expected in expected_counts.items():
            if unit_report.get(field) != expected:
                raise ShardMergeError(f"unit report {field} mismatch: {key}")
        if int(unit_report.get("forbidden_duplicate_rows", -1)) != 0:
            raise ShardMergeError(f"unit report contains forbidden duplicates: {key}")

    counts = report.get("counts")
    if not isinstance(counts, dict):
        raise ShardMergeError(f"{vfm} audit counts are malformed")
    derived_counts = {
        "units": len(units),
        "expected_rows": sum(len(unit["expected_raw_ids"]) for unit in units.values()),
        "routed_rows": route_records,
        "missing_rows": sum(len(value) for value in missing_by_unit.values()),
        "pending_rows": sum(len(value) for value in pending_by_unit.values()),
        "duplicate_rows": sum(int(item.get("duplicate_rows", 0)) for item in unit_reports.values()),
        "duplicate_rows_requiring_bitwise_check": sum(
            int(item.get("conditional_duplicate_rows", 0))
            for item in unit_reports.values()
        ),
        "source_payloads_passed": sum(
            int(item.get("sources_passed", 0)) for item in unit_reports.values()
        ),
        "source_payloads_full_sha_verified": sum(
            1
            for item in unit_reports.values()
            for evidence in item.get("source_evidence", [])
            if evidence.get("payload_sha256_verified") is True
        ),
        "source_payloads_header_only_unverified": sum(
            1
            for item in unit_reports.values()
            for evidence in item.get("source_evidence", [])
            if evidence.get("payload_sha256_verified") is not True
        ),
        "source_payloads_failed": sum(
            int(item.get("sources_failed", 0)) for item in unit_reports.values()
        ),
        "source_payloads_pending": sum(
            int(item.get("sources_pending", 0)) for item in unit_reports.values()
        ),
        "hard_errors": 0,
        "routes_by_kind": dict(routes_by_kind),
    }
    for field, expected in derived_counts.items():
        if counts.get(field) != expected:
            raise ShardMergeError(f"{vfm} audit aggregate {field} mismatch")
    derived_status = (
        "PENDING" if pending_rows else "INCOMPLETE" if missing_rows else "PASS"
    )
    if report.get("status") != derived_status or report.get("ready_to_materialize") is not (
        derived_status == "PASS"
    ):
        raise ShardMergeError(f"{vfm} audit status does not match its row partition")
    _validate_gap_worklist_outputs(
        report=report, missing=missing_rows, output_dir=output_dir
    )
    return {
        "report": report,
        "report_path": report_path,
        "report_content_sha256": report_content_sha,
        "route_records": route_records,
        "routes_by_kind": routes_by_kind,
        "missing": missing_rows,
        "pending": pending_rows,
        "unit_reports": unit_reports,
    }


def _write_payload(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def _publish_unified_outputs(
    *,
    staging: Path,
    output_dir: Path,
    report: dict[str, Any],
    route_records: int,
    route_sha256: str,
    missing: Sequence[Mapping[str, Any]],
    pending: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    missing_payload = b"".join(_jsonl_bytes(item) for item in missing)
    pending_payload = b"".join(_jsonl_bytes(item) for item in pending)
    _write_payload(staging / "missing_target_units.jsonl", missing_payload)
    _write_payload(staging / "pending_target_units.jsonl", pending_payload)
    gap_bindings: list[dict[str, Any]] = []
    gap_result = {"report": report, "missing_units": missing}
    seen_names: set[str] = set()
    for worklist in build_gap_worklists(gap_result):
        name = _slug(
            f"{worklist['vfm']}-t{worklist['timestep']}-"
            f"{canonical_json_sha([worklist['model_id'], worklist['source'], worklist['output_layers']])[:12]}"
        ) + ".json"
        if name in seen_names:
            raise ShardMergeError(f"unified gap worklist filename collision: {name}")
        seen_names.add(name)
        payload = (
            json.dumps(worklist, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode()
        file_sha = _write_payload(staging / "gap_worklists" / name, payload)
        gap_bindings.append(
            {
                "path": str(output_dir / "gap_worklists" / name),
                "file_sha256": file_sha,
                "content_sha256": worklist["content_sha256"],
                "counts": worklist["counts"],
            }
        )
    report["outputs"] = {
        "target_row_routes": {
            "path": str(output_dir / "target_row_routes.jsonl"),
            "records": route_records,
            "sha256": route_sha256,
        },
        "missing_target_units": {
            "path": str(output_dir / "missing_target_units.jsonl"),
            "records": len(missing),
            "sha256": hashlib.sha256(missing_payload).hexdigest(),
        },
        "pending_target_units": {
            "path": str(output_dir / "pending_target_units.jsonl"),
            "records": len(pending),
            "sha256": hashlib.sha256(pending_payload).hexdigest(),
        },
        "gap_worklists": gap_bindings,
    }
    report["content_sha256"] = canonical_json_sha(
        {key: value for key, value in report.items() if key != "content_sha256"}
    )
    report_payload = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    _write_payload(staging / "target_row_route_audit.json", report_payload)
    return report


def _parse_audit_overrides(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        vfm, separator, raw_path = value.partition("=")
        if not separator or not vfm or not raw_path or vfm in result:
            raise ShardMergeError(
                "--shard-audit must be unique VFM=/absolute/report.json entries"
            )
        path = Path(raw_path)
        if not path.is_absolute():
            raise ShardMergeError("--shard-audit paths must be absolute")
        result[vfm] = path.resolve()
    return result


def merge_audits(
    *,
    plan_path: Path,
    review_manifest_path: Path,
    output_dir: Path,
    shard_audits: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    plan_path = _regular(plan_path.resolve(), "parent target merge plan")
    review_manifest_path = _regular(
        review_manifest_path.resolve(), "target audit review manifest"
    )
    output_dir = output_dir.resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise ShardMergeError(f"unified audit output already exists: {output_dir}")
    parent_data = _validate_parent_plan(plan_path)
    manifest_data = _validate_review_manifest(
        review_manifest_path,
        plan_path=plan_path,
        plan=parent_data["plan"],
        plan_file_sha256=parent_data["plan_file_sha256"],
    )
    reviewed_shards = _validate_review_shards(
        manifest_data=manifest_data,
        parent_data=parent_data,
        plan_path=plan_path,
    )
    overrides = dict(shard_audits or {})
    expected_vfms = {item["vfm"] for item in reviewed_shards}
    if overrides and set(overrides) != expected_vfms:
        raise ShardMergeError(
            "explicit shard-audit set must exactly equal reviewed VFMs: "
            f"missing={sorted(expected_vfms - set(overrides))} "
            f"extra={sorted(set(overrides) - expected_vfms)}"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        route_path = staging / "target_row_routes.jsonl"
        route_digest = hashlib.sha256()
        route_records = 0
        all_missing: list[dict[str, Any]] = []
        all_pending: list[dict[str, Any]] = []
        unit_reports: dict[tuple[Any, ...], dict[str, Any]] = {}
        shard_results: list[dict[str, Any]] = []
        total_routes_by_kind: Counter[str] = Counter()
        with route_path.open("xb") as route_output:
            for shard in reviewed_shards:
                expected_report = (
                    shard["audit_output"] / "target_row_route_audit.json"
                ).resolve()
                report_path = overrides.get(shard["vfm"], expected_report).resolve()
                if report_path != expected_report:
                    raise ShardMergeError(
                        f"explicit {shard['vfm']} audit path differs from worker binding"
                    )
                report_path = _regular(report_path, f"{shard['vfm']} route audit")
                result = _validate_shard_audit(
                    shard,
                    report_path=report_path,
                    parent_data=parent_data,
                    manifest_data=manifest_data,
                    route_output=route_output,
                    route_digest=route_digest,
                )
                route_records += result["route_records"]
                total_routes_by_kind.update(result["routes_by_kind"])
                all_missing.extend(result["missing"])
                all_pending.extend(result["pending"])
                overlap = set(unit_reports).intersection(result["unit_reports"])
                if overlap:
                    raise ShardMergeError(
                        f"duplicate unit reports across audits: {next(iter(overlap))}"
                    )
                unit_reports.update(result["unit_reports"])
                shard_results.append({**shard, **result})
            route_output.flush()
            os.fsync(route_output.fileno())
        if set(unit_reports) != set(parent_data["unit_map"]):
            raise ShardMergeError("shard audits do not cover every parent-plan unit")
        ordered_unit_reports = [
            unit_reports[_unit_key(unit)] for unit in parent_data["normalized_units"]
        ]
        all_missing.sort(key=_unit_sort_key)
        all_pending.sort(key=_unit_sort_key)
        status = (
            "PENDING" if all_pending else "INCOMPLETE" if all_missing else "PASS"
        )
        counts = {
            "units": len(ordered_unit_reports),
            "expected_rows": sum(
                int(item["expected_rows"]) for item in ordered_unit_reports
            ),
            "routed_rows": route_records,
            "missing_rows": sum(len(item["missing_raw_ids"]) for item in all_missing),
            "pending_rows": sum(len(item["pending_raw_ids"]) for item in all_pending),
            "duplicate_rows": sum(
                int(item.get("duplicate_rows", 0)) for item in ordered_unit_reports
            ),
            "duplicate_rows_requiring_bitwise_check": sum(
                int(item.get("conditional_duplicate_rows", 0))
                for item in ordered_unit_reports
            ),
            "source_payloads_passed": sum(
                int(item.get("sources_passed", 0)) for item in ordered_unit_reports
            ),
            "source_payloads_full_sha_verified": sum(
                1
                for item in ordered_unit_reports
                for evidence in item.get("source_evidence", [])
                if evidence.get("payload_sha256_verified") is True
            ),
            "source_payloads_header_only_unverified": sum(
                1
                for item in ordered_unit_reports
                for evidence in item.get("source_evidence", [])
                if evidence.get("payload_sha256_verified") is not True
            ),
            "source_payloads_failed": sum(
                int(item.get("sources_failed", 0)) for item in ordered_unit_reports
            ),
            "source_payloads_pending": sum(
                int(item.get("sources_pending", 0)) for item in ordered_unit_reports
            ),
            "hard_errors": 0,
            "routes_by_kind": dict(total_routes_by_kind),
        }
        plan = parent_data["plan"]
        manifest = manifest_data["manifest"]
        report: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "audit_type": AUDIT_TYPE,
            "contract": CONTRACT,
            "status": status,
            "ready_to_materialize": status == "PASS",
            "dry_run": True,
            "feature_payload_written": False,
            "seeded_namespace_written": False,
            "canonical_protocol": CANONICAL_PROTOCOL,
            "canonical_protocol_sha256": parent_data["protocol"].get(
                "canonical_protocol_sha256"
            ),
            "protocol_file": str(parent_data["protocol_path"]),
            "protocol_file_sha256": sha256_file(parent_data["protocol_path"]),
            "canonical_source": parent_data["canonical_metadata"],
            "plan": str(plan_path),
            "plan_sha256": parent_data["plan_file_sha256"],
            "plan_content_sha256": plan["content_sha256"],
            "auditor_source": str(manifest_data["paths"]["auditor"]),
            "auditor_source_sha256": manifest["auditor"]["sha256"],
            "source_reader_source": str(manifest_data["paths"]["source_reader"]),
            "source_reader_source_sha256": manifest["source_reader"]["sha256"],
            "merger_source": str(Path(__file__).resolve()),
            "merger_source_sha256": sha256_file(Path(__file__).resolve()),
            "review_manifest": str(review_manifest_path),
            "review_manifest_sha256": sha256_file(review_manifest_path),
            "review_manifest_content_sha256": manifest["content_sha256"],
            "destination_root": str(parent_data["destination_root"]),
            "destination_namespace": "FEAT_TARGET_CANONICAL_V2_MIXED_V1",
            "verify_source_sha": True,
            "remote_policy": "gate",
            "source_timeout_seconds": manifest["source_timeout_seconds"],
            "counts": counts,
            "units": ordered_unit_reports,
            "hard_errors": [],
            "hard_errors_truncated": False,
            "shard_audit_merge": {
                "merge_contract": "CANONICAL_V2_TARGET_ROUTE_AUDIT_SHARD_MERGE_V1",
                "partition_verified_exact": True,
                "shards": [
                    {
                        "ordinal": item["record"]["ordinal"],
                        "vfm": item["vfm"],
                        "units": len(item["unit_keys"]),
                        "shard_plan": str(item["shard_path"]),
                        "shard_plan_sha256": item["shard_file_sha256"],
                        "worker_binding_receipt": str(item["receipt_path"]),
                        "worker_binding_receipt_sha256": sha256_file(
                            item["receipt_path"]
                        ),
                        "route_audit": str(item["report_path"]),
                        "route_audit_sha256": sha256_file(item["report_path"]),
                        "route_audit_content_sha256": item[
                            "report_content_sha256"
                        ],
                    }
                    for item in shard_results
                ],
            },
        }
        report = _publish_unified_outputs(
            staging=staging,
            output_dir=output_dir,
            report=report,
            route_records=route_records,
            route_sha256=route_digest.hexdigest(),
            missing=all_missing,
            pending=all_pending,
        )
        os.replace(staging, output_dir)
        return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--review-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--shard-audit",
        action="append",
        default=[],
        metavar="VFM=/ABSOLUTE/target_row_route_audit.json",
        help=(
            "optional explicit audit paths; when used, every reviewed VFM must be "
            "listed and each path must equal its worker-bound output"
        ),
    )
    args = parser.parse_args(argv)
    try:
        report = merge_audits(
            plan_path=args.plan,
            review_manifest_path=args.review_manifest,
            output_dir=args.output_dir,
            shard_audits=_parse_audit_overrides(args.shard_audit),
        )
    except (OSError, ShardMergeError) as exc:
        print(
            json.dumps(
                {"status": "FAIL", "error": str(exc)},
                sort_keys=True,
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
