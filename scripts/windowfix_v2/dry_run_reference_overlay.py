#!/usr/bin/env python3
"""Validate a LEGACY_EXACT_REFERENCE_V1 routing plan without payload reads.

The program has no execute mode.  It reads JSON worklists and safetensors
headers only; it never hashes tensor data, copies payloads, writes route roots,
submits SCO, or starts probing.  A PASS certifies temporal/physical routing,
not byte integrity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import tempfile
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from urllib.parse import unquote, urlparse

from analyses.windowfix_v2.build_hybrid_reuse_plan import canonical_sha, sha256_file
from analyses.windowfix_v2.build_reference_overlay_plan import (
    REFERENCE_CONTRACT,
    REFERENCE_NAMESPACES,
)


class ReferenceAuditError(ValueError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ReferenceAuditError(f"not a regular plan file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReferenceAuditError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ReferenceAuditError(f"not a regular worklist: {path}")
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ReferenceAuditError(f"{path}:{line_number}: expected object")
            yield value


def _write_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _local_source(uri: str) -> Path | None:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        if parsed.netloc not in {"", "localhost"}:
            raise ReferenceAuditError(f"non-local file URI is forbidden: {uri}")
        return Path(unquote(parsed.path))
    if not parsed.scheme:
        return Path(uri)
    return None


def _tensor_shape(path: Path, tensor_key: str) -> tuple[int, ...]:
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        length = handle.read(8)
        if len(length) != 8:
            raise ReferenceAuditError(f"truncated safetensors header: {path}")
        header_size = struct.unpack("<Q", length)[0]
        if header_size <= 1 or header_size > min(file_size - 8, 128 * 1024 * 1024):
            raise ReferenceAuditError(f"invalid safetensors header size: {path}")
        header = json.loads(handle.read(header_size).decode("utf-8"))
    tensor = header.get(tensor_key)
    if not isinstance(tensor, dict) or not isinstance(tensor.get("shape"), list):
        raise ReferenceAuditError(f"missing tensor {tensor_key!r}: {path}")
    shape = tuple(int(value) for value in tensor["shape"])
    offsets = tensor.get("data_offsets")
    if (
        not shape
        or any(value < 0 for value in shape)
        or not isinstance(offsets, list)
        or len(offsets) != 2
        or int(offsets[0]) < 0
        or int(offsets[1]) < int(offsets[0])
        or int(offsets[1]) > file_size - 8 - header_size
    ):
        raise ReferenceAuditError(f"invalid tensor header/offsets: {path}")
    return shape


def validate(plan_dir: Path) -> dict[str, Any]:
    plan_path = plan_dir / "reference_overlay_plan.json"
    plan = _read_json(plan_path)
    if plan.get("contract") != REFERENCE_CONTRACT:
        raise ReferenceAuditError("wrong reference contract")
    if plan.get("byte_level_certified") is not False:
        raise ReferenceAuditError("reference plan cannot claim byte certification")
    if plan.get("payload_full_read_performed") is not False:
        raise ReferenceAuditError("reference plan cannot claim a payload full read")
    if plan.get("payload_copy_performed") is not False:
        raise ReferenceAuditError("reference plan cannot claim payload copying")
    if plan.get("route_roots_written") is not False:
        raise ReferenceAuditError("reference dry-run cannot write route roots")
    if plan.get("seeded_namespace_written") is not False:
        raise ReferenceAuditError("reference plan violates SEEDED_V1 isolation")
    if plan.get("route_namespaces") != REFERENCE_NAMESPACES:
        raise ReferenceAuditError("unexpected reference namespaces")
    expected_content_sha = canonical_sha(
        {key: value for key, value in plan.items() if key != "content_sha256"}
    )
    if plan.get("content_sha256") != expected_content_sha:
        raise ReferenceAuditError("reference plan content SHA mismatch")

    bindings = plan.get("action_worklists")
    if not isinstance(bindings, dict):
        raise ReferenceAuditError("reference plan has no action worklists")
    rows_by_name: dict[str, list[dict[str, Any]]] = {}
    for name, binding in sorted(bindings.items()):
        if Path(name).name != name or not name.endswith(".jsonl"):
            raise ReferenceAuditError(f"unsafe worklist name: {name!r}")
        path = plan_dir / name
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != binding.get("content_sha256"):
            raise ReferenceAuditError(f"worklist content SHA mismatch: {path}")
        rows = list(_read_jsonl(path))
        if len(rows) != int(binding.get("records", -1)):
            raise ReferenceAuditError(f"worklist record count mismatch: {path}")
        rows_by_name[name] = rows

    routes = rows_by_name.get("reference_routes.jsonl", [])
    seen_routes: set[tuple[str, str]] = set()
    local_sources: set[Path] = set()
    remote_sources: set[str] = set()
    target_rows_checked = 0
    errors: list[str] = []
    for row in routes:
        if row.get("contract") != REFERENCE_CONTRACT:
            errors.append(f"wrong route contract: {row.get('spec_id')}")
            continue
        if row.get("action") != "REFERENCE_SOURCE_READ_ONLY":
            errors.append(f"unsafe route action: {row.get('action')}")
        if row.get("byte_level_certified") is not False:
            errors.append(f"route claims byte certification: {row.get('spec_id')}")
        if row.get("declared_sha256_verified") is not False:
            errors.append(f"route claims SHA verification: {row.get('spec_id')}")
        route_identity = (str(row.get("route_root")), str(row.get("route_key")))
        if route_identity in seen_routes:
            errors.append(f"duplicate route: {route_identity}")
        seen_routes.add(route_identity)
        if "SEEDED_V1" in route_identity[0]:
            errors.append(f"route enters SEEDED_V1: {route_identity[0]}")
        source_uri = str(row.get("source_uri", ""))
        try:
            local = _local_source(source_uri)
        except ReferenceAuditError as exc:
            errors.append(str(exc))
            continue
        if local is None:
            remote_sources.add(source_uri)
            continue
        if not local.is_absolute() or local.is_symlink() or not local.is_file():
            errors.append(f"local source is not a regular file: {local}")
            continue
        local_sources.add(local)
        if local.stat().st_size != int(row.get("source_size_bytes", -1)):
            errors.append(f"source size changed: {local}")
            continue
        try:
            shape = _tensor_shape(local, str(row.get("tensor_key", "feat")))
            if row.get("unit_kind") == "target_row":
                source_row = int(row["source_row_index"])
                if source_row < 0 or source_row >= shape[0]:
                    errors.append(f"target source row exceeds shape {shape}: {local}")
                target_rows_checked += 1
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            errors.append(str(exc))

    status = "FAIL" if errors else "PASS_TEMPORAL_PHYSICAL_REFERENCE_ONLY"
    return {
        "schema_version": 1,
        "audit_type": "legacy_exact_reference_overlay_dry_run",
        "status": status,
        "dry_run": True,
        "byte_level_certified": False,
        "payload_full_read_performed": False,
        "payload_hash_performed": False,
        "payload_copy_performed": False,
        "route_roots_written": False,
        "sco_submission_performed": False,
        "probing_started": False,
        "plan": str(plan_path),
        "plan_file_sha256": sha256_file(plan_path),
        "counts": {
            "reference_routes": len(routes),
            "unique_local_sources_header_checked": len(local_sources),
            "remote_sources_preverified_by_classifier": len(remote_sources),
            "target_rows_checked": target_rows_checked,
            "errors": len(errors),
        },
        "errors": errors[:100],
        "error_list_truncated": len(errors) > 100,
        "warning": (
            "PASS does not certify payload bytes; it certifies canonical raw-ID and "
            "physical-identity routing against currently existing sources."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = validate(args.plan_dir.resolve())
    if args.output is not None:
        _write_atomic(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if report["status"] != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
