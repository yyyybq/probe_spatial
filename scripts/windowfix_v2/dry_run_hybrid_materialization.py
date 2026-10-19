#!/usr/bin/env python3
"""Validate a mixed-cache import plan without copying feature payloads.

There is intentionally no execute mode in this program.  It validates the
plan/worklist bindings, namespace separation, source file identity, target-row
bounds, and destination collisions.  ``--verify-source-sha`` rereads every
unique source payload and is the final pre-copy gate; without it the report is
metadata-only and explicitly not ready to copy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


IMPORT_CONTRACT = "LEGACY_EXACT_IMPORT_V1"
MIXED_NAMESPACES = {
    "streaming": "FEAT_STREAMING_CANONICAL_V2_MIXED_V1",
    "target": "FEAT_TARGET_CANONICAL_V2_MIXED_V1",
}


class MaterializationError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise MaterializationError(f"not a regular plan file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MaterializationError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise MaterializationError(f"not a regular worklist: {path}")
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise MaterializationError(f"{path}:{line_number}: expected object")
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


def _tensor_shape(path: Path, tensor_key: str) -> tuple[int, ...]:
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        size_bytes = handle.read(8)
        if len(size_bytes) != 8:
            raise MaterializationError(f"truncated safetensors file: {path}")
        header_size = struct.unpack("<Q", size_bytes)[0]
        if header_size <= 1 or header_size > min(file_size - 8, 128 * 1024 * 1024):
            raise MaterializationError(f"invalid safetensors header size: {path}")
        header = json.loads(handle.read(header_size).decode("utf-8"))
    tensor = header.get(tensor_key)
    if not isinstance(tensor, dict) or not isinstance(tensor.get("shape"), list):
        raise MaterializationError(f"missing tensor {tensor_key!r}: {path}")
    shape = tuple(int(value) for value in tensor["shape"])
    if not shape or any(value < 0 for value in shape):
        raise MaterializationError(f"invalid tensor shape: {path}")
    return shape


def validate(plan_dir: Path, *, verify_source_sha: bool) -> dict[str, Any]:
    plan_path = plan_dir / "hybrid_plan.json"
    plan = read_json(plan_path)
    if plan.get("contract") != IMPORT_CONTRACT:
        raise MaterializationError("plan is not LEGACY_EXACT_IMPORT_V1")
    recorded_content_sha = plan.get("content_sha256")
    actual_content_sha = canonical_sha(
        {key: value for key, value in plan.items() if key != "content_sha256"}
    )
    if recorded_content_sha != actual_content_sha:
        raise MaterializationError("hybrid plan content SHA mismatch")
    if plan.get("destination_namespaces") != MIXED_NAMESPACES:
        raise MaterializationError("unexpected destination namespace binding")
    if plan.get("seeded_namespace_written") is not False:
        raise MaterializationError("plan does not preserve SEEDED_V1 isolation")

    worklist_specs = plan.get("action_worklists")
    if not isinstance(worklist_specs, dict):
        raise MaterializationError("plan has no bound action worklists")
    rows_by_name: dict[str, list[dict[str, Any]]] = {}
    for name, binding in sorted(worklist_specs.items()):
        if Path(name).name != name or not name.endswith(".jsonl"):
            raise MaterializationError(f"unsafe worklist name: {name!r}")
        path = plan_dir / name
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != binding.get("content_sha256"):
            raise MaterializationError(f"worklist content SHA mismatch: {path}")
        rows = list(read_jsonl(path))
        if len(rows) != int(binding.get("records", -1)):
            raise MaterializationError(f"worklist record count mismatch: {path}")
        rows_by_name[name] = rows

    sources: dict[Path, tuple[int, str]] = {}
    destinations: set[Path] = set()
    target_bounds_checked = 0
    errors: list[str] = []
    actions = rows_by_name.get("import_files.jsonl", []) + rows_by_name.get(
        "import_target_rows.jsonl", []
    )
    for row in actions:
        if row.get("contract") != IMPORT_CONTRACT:
            errors.append(f"wrong contract for {row.get('spec_id')}")
            continue
        source = Path(str(row.get("source_path", "")))
        destination = Path(str(row.get("destination_path", "")))
        if not source.is_absolute() or source.is_symlink() or not source.is_file():
            errors.append(f"source is not a local regular file: {source}")
            continue
        if not destination.is_absolute() or "SEEDED_V1" in destination.parts:
            errors.append(f"unsafe destination: {destination}")
            continue
        expected_namespace = MIXED_NAMESPACES[str(row["destination_mode"])]
        if expected_namespace not in destination.parts:
            errors.append(f"destination is outside mixed namespace: {destination}")
            continue
        if destination.exists() or destination.is_symlink():
            errors.append(f"destination already exists; overwrite forbidden: {destination}")
        if row.get("action") == "COPY_WHOLE_FILE":
            if destination in destinations:
                errors.append(f"duplicate destination: {destination}")
            destinations.add(destination)
        expected_size = int(row.get("source_size_bytes", -1))
        expected_sha = str(row.get("source_payload_sha256", ""))
        previous = sources.get(source)
        identity = (expected_size, expected_sha)
        if previous is not None and previous != identity:
            errors.append(f"conflicting identity for source: {source}")
            continue
        sources[source] = identity
        if source.stat().st_size != expected_size:
            errors.append(f"source size changed: {source}")
        try:
            shape = _tensor_shape(source, str(row.get("tensor_key", "feat")))
            if row.get("action") == "IMPORT_TARGET_ROW":
                index = int(row["source_row_index"])
                if index < 0 or index >= shape[0]:
                    errors.append(f"target row {index} exceeds source shape {shape}: {source}")
                target_bounds_checked += 1
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            errors.append(str(exc))

    hashes_verified = 0
    if verify_source_sha:
        for source, (_size, expected_sha) in sources.items():
            if sha256_file(source) != expected_sha:
                errors.append(f"source SHA changed: {source}")
            hashes_verified += 1

    if errors:
        status = "FAIL"
    elif verify_source_sha:
        status = "PASS_READY_FOR_SEPARATE_COPY_IMPLEMENTATION"
    else:
        status = "PASS_METADATA_ONLY_HASH_GATE_PENDING"
    return {
        "schema_version": 1,
        "audit_type": "hybrid_materialization_dry_run",
        "dry_run": True,
        "copy_performed": False,
        "status": status,
        "ready_to_copy": status == "PASS_READY_FOR_SEPARATE_COPY_IMPLEMENTATION",
        "plan": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "verify_source_sha": verify_source_sha,
        "counts": {
            "import_actions": len(actions),
            "unique_source_files": len(sources),
            "source_hashes_verified": hashes_verified,
            "target_row_bounds_checked": target_bounds_checked,
            "destination_files": len(destinations),
            "errors": len(errors),
            "actions_by_type": dict(Counter(str(row.get("action")) for row in actions)),
        },
        "errors": errors[:100],
        "error_list_truncated": len(errors) > 100,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--verify-source-sha", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = validate(args.plan_dir.resolve(), verify_source_sha=args.verify_source_sha)
    if args.output is not None:
        _write_atomic(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if report["status"] != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
