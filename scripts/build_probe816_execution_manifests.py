#!/usr/bin/env python3
"""Freeze exact owner-backed sample manifests and bind them to queue units."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any


PARENT_SHA = "ebc496eb0587cda372ccf29a7c5244ef454096151f73c8eed9511967ded1902d"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--streaming-db", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--owner-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-queue-rows", type=int, default=760)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError(f"output exists: {args.output_dir}")
    if sha256(args.parent_manifest) != PARENT_SHA:
        raise ValueError("parent manifest SHA mismatch")
    parent_rows = jsonl(args.parent_manifest)
    parent = {}
    parent_order = {}
    for ordinal, row in enumerate(parent_rows):
        key = (str(row["source"]), str(row["scene_id"]), int(row["streaming_window_id"]))
        if key in parent:
            raise ValueError(f"duplicate parent sample key: {key}")
        parent[key] = row
        parent_order[key] = ordinal

    con = sqlite3.connect(f"file:{args.streaming_db.resolve()}?mode=ro&immutable=1", uri=True)
    con.row_factory = sqlite3.Row
    status = con.execute("SELECT value FROM metadata WHERE key='status'").fetchone()
    if status is None or status[0] != "READY":
        raise ValueError("streaming DB is not READY")
    streaming_owner_rows = int(
        con.execute("SELECT COUNT(*) FROM streaming_owners").fetchone()[0]
    )
    specs = list(con.execute(
        "SELECT DISTINCT vfm,layer,timestep,prefix FROM streaming_owners ORDER BY 1,2,3,4"
    ))
    signature_groups: dict[str, list[tuple[str, int, int, int]]] = defaultdict(list)
    rows_by_signature: dict[str, list[dict[str, Any]]] = {}
    spec_signature: dict[tuple[str, int, int, int], str] = {}
    for spec_row in specs:
        spec = (str(spec_row[0]), int(spec_row[1]), int(spec_row[2]), int(spec_row[3]))
        owner_rows = list(con.execute(
            "SELECT source,scene_id,window_id,raw_ids_json,unit_id FROM streaming_owners "
            "WHERE vfm=? AND layer=? AND timestep=? AND prefix=? ORDER BY source,scene_id,window_id",
            spec,
        ))
        identity = [
            [str(row["source"]), str(row["scene_id"]), int(row["window_id"]), json.loads(row["raw_ids_json"])]
            for row in owner_rows
        ]
        signature = canonical_sha(identity)
        signature_groups[signature].append(spec)
        spec_signature[spec] = signature
        if signature in rows_by_signature:
            continue
        manifest_rows = []
        for owner in owner_rows:
            key = (str(owner["source"]), str(owner["scene_id"]), int(owner["window_id"]))
            if key not in parent:
                raise ValueError(f"owner key absent from canonical parent: {key}")
            raw_ids = [int(v) for v in json.loads(owner["raw_ids_json"])]
            base = dict(parent[key])
            if raw_ids != [int(v) for v in base["streaming_window_indices"][: len(raw_ids)]]:
                raise ValueError(f"owner raw IDs differ from canonical parent: {spec}/{key}")
            base["sample_id"] = f"{key[1]}_w{key[2]:03d}_p{len(raw_ids)}"
            base["streaming_tail"] = len(raw_ids) - 1
            base["streaming_tail_raw"] = raw_ids[-1]
            base["streaming_indices"] = raw_ids
            base["owner_unit_id"] = str(owner["unit_id"])
            manifest_rows.append((parent_order[key], base))
        rows_by_signature[signature] = [row for _, row in sorted(manifest_rows)]
    con.close()

    manifests_dir = args.output_dir / "manifests"
    manifests_dir.mkdir(parents=True)
    manifest_bindings = {}
    for signature, rows in sorted(rows_by_signature.items()):
        prefixes = {len(row["streaming_indices"]) for row in rows}
        if len(prefixes) != 1:
            raise ValueError("availability class spans prefixes")
        prefix = next(iter(prefixes))
        path = manifests_dir / f"owner_set_p{prefix}_{signature[:12]}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
        manifest_bindings[signature] = {
            "path": str(path.resolve()), "sha256": sha256(path), "rows": len(rows),
            "prefix": prefix, "specs": [list(spec) for spec in signature_groups[signature]],
        }

    queue_rows = jsonl(args.queue)
    if len(queue_rows) != args.expected_queue_rows:
        raise ValueError(
            f"input queue is not {args.expected_queue_rows} unique units: {len(queue_rows)}"
        )
    for row in queue_rows:
        spec = (str(row["model"]), int(row["disk_layer"]), int(row["timestep"]), int(row["prefix"]))
        signature = spec_signature.get(spec)
        if signature is None:
            raise ValueError(f"queue spec absent from owner DB: {spec}")
        binding = manifest_bindings[signature]
        row["execution_manifest"] = {key: binding[key] for key in ("path", "sha256", "rows")}
        row["protocol"]["canonical_parent_manifest_sha256"] = PARENT_SHA
        row["protocol"]["canonical_manifest_sha256"] = binding["sha256"]
        row["protocol"]["sample_count_all_splits"] = binding["rows"]

    queue_path = args.output_dir / f"queue_{len(queue_rows)}.jsonl"
    with queue_path.open("w", encoding="utf-8") as handle:
        for row in queue_rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    owner_map_rows = jsonl(args.owner_map)
    queue_keys = {str(row["unique_key"]) for row in queue_rows}
    owner_map_keys = {str(row["unique_key"]) for row in owner_map_rows}
    if owner_map_keys != queue_keys:
        raise ValueError(
            f"owner map key closure mismatch: missing={len(queue_keys-owner_map_keys)} "
            f"extra={len(owner_map_keys-queue_keys)}"
        )
    owner_map_out = args.output_dir / f"experiment_owner_map_{len(owner_map_rows)}.jsonl"
    owner_map_out.write_bytes(args.owner_map.read_bytes())
    summary = {
        "schema": "CANONICAL_V2_PROBE816_OWNER_BACKED_QUEUE_V2",
        "status": "FROZEN_NOT_SUBMITTED",
        "parent_manifest": {"path": str(args.parent_manifest.resolve()), "sha256": PARENT_SHA, "rows": len(parent_rows)},
        "streaming_database": {"path": str(args.streaming_db.resolve()), "sha256": sha256(args.streaming_db), "rows": streaming_owner_rows},
        "queue": {"path": str(queue_path.resolve()), "sha256": sha256(queue_path), "rows": len(queue_rows)},
        "experiment_owner_map": {"path": str(owner_map_out.resolve()), "sha256": sha256(owner_map_out), "rows": len(owner_map_rows)},
        "availability_classes": list(manifest_bindings.values()),
        "counts": {
            "source_registry_rows": len(owner_map_rows),
            "unique_execution_units": len(queue_rows),
            "duplicate_alias_rows": len(owner_map_rows) - len(queue_rows),
            "manifest_classes": len(manifest_bindings),
        },
        "sample_policy": "exact streaming owner key intersection in canonical parent order; no runtime dropping",
    }
    summary_path = args.output_dir / "manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
