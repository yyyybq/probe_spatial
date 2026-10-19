#!/usr/bin/env python3
"""Freeze the receipt-union owner table and execution manifests as one overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def db_ready(path: Path, table: str) -> int:
    db = sqlite3.connect(f"file:{path.resolve()}?mode=ro&immutable=1", uri=True)
    status = db.execute("SELECT value FROM metadata WHERE key='status'").fetchone()
    rows = int(db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    db.close()
    if not status or status[0] != "READY":
        raise ValueError(f"database is not READY: {path}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--experiments", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"output exists: {args.output}")
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    execution = json.loads(args.execution_manifest.read_text(encoding="utf-8"))
    if (
        audit.get("status") != "READY"
        or audit.get("zero_holes") is not True
        or audit.get("zero_duplicate_effective_owners") is not True
    ):
        raise ValueError("receipt union audit is not closed")
    if execution.get("status") != "FROZEN_NOT_SUBMITTED":
        raise ValueError("execution manifests are not frozen")
    streaming_db = Path(audit["streaming"]["database"]["path"])
    target_db = Path(audit["target"]["database"]["path"])
    if sha256(streaming_db) != audit["streaming"]["database"]["sha256"]:
        raise ValueError("streaming DB SHA mismatch")
    if sha256(target_db) != audit["target"]["database"]["sha256"]:
        raise ValueError("target DB SHA mismatch")
    stream_rows = db_ready(streaming_db, "streaming_owners")
    target_rows = db_ready(target_db, "target_owners")
    if stream_rows != 327_748 or target_rows != 715_863:
        raise ValueError(f"owner row count mismatch: {stream_rows}/{target_rows}")
    if execution["streaming_database"]["sha256"] != sha256(streaming_db):
        raise ValueError("execution manifest is bound to another streaming DB")
    result = {
        "schema": "CANONICAL_V2_PROBE816_MIXED_OWNER_OVERLAY_V1",
        "status": "READY",
        "fail_closed": True,
        "models": audit["models"],
        "experiments": {
            "path": str(args.experiments.resolve()),
            "sha256": sha256(args.experiments),
            "rows": 816,
        },
        "canonical_manifest": {
            "path": str(args.canonical_manifest.resolve()),
            "sha256": sha256(args.canonical_manifest),
            "rows": 2721,
        },
        "frozen_queue_manifest": {
            "path": str(args.execution_manifest.resolve()),
            "sha256": sha256(args.execution_manifest),
        },
        "streaming_database": {
            "path": str(streaming_db.resolve()),
            "sha256": sha256(streaming_db),
            "rows": stream_rows,
        },
        "target_database": {
            "path": str(target_db.resolve()),
            "sha256": sha256(target_db),
            "rows": target_rows,
        },
        "receipt_union_audit": {
            "path": str(args.audit.resolve()),
            "sha256": sha256(args.audit),
        },
        "minimax_bf16_route": audit["minimax_bf16_route"],
        "counts": {
            "streaming_holes": 0,
            "streaming_duplicate_owners": 0,
            "target_holes": 0,
            "target_duplicate_owners": 0,
        },
        "owner_policy": (
            "exact composite key and raw frame IDs; receipt URI union; "
            "no runtime path fallback or sample dropping"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
