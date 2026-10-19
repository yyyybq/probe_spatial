#!/usr/bin/env python3
"""Bind the seven-model owner databases, user scope, and execution queue."""

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


def db_check(path: Path, table: str, expected: int) -> None:
    db = sqlite3.connect(f"file:{path.resolve()}?mode=ro&immutable=1", uri=True)
    status = db.execute("SELECT value FROM metadata WHERE key='status'").fetchone()
    count = db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    minimax = db.execute(f"SELECT COUNT(*) FROM {table} WHERE vfm='minimax_h3'").fetchone()[0]
    db.close()
    if not status or status[0] != "READY" or count != expected or minimax != 0:
        raise RuntimeError(f"database validation failed: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"output exists: {args.output}")
    audit = json.loads(args.audit.read_text())
    execution = json.loads(args.execution_manifest.read_text())
    if (
        audit.get("status") != "READY" or audit.get("zero_holes") is not True
        or audit.get("zero_duplicate_effective_owners") is not True
        or audit.get("minimax_rows") != {"streaming": 0, "target": 0, "target_expected": 0}
    ):
        raise RuntimeError("seven-model owner audit is not closed")
    if execution.get("status") != "FROZEN_NOT_SUBMITTED":
        raise RuntimeError("execution manifest is not frozen")
    if execution.get("counts", {}).get("unique_execution_units") != 528:
        raise RuntimeError("execution manifest is not the 528-unit user scope")
    stream = Path(audit["streaming"]["database"]["path"])
    target = Path(audit["target"]["database"]["path"])
    if sha256(stream) != audit["streaming"]["database"]["sha256"]:
        raise RuntimeError("streaming DB SHA mismatch")
    if sha256(target) != audit["target"]["database"]["sha256"]:
        raise RuntimeError("target DB SHA mismatch")
    if execution["streaming_database"]["sha256"] != sha256(stream):
        raise RuntimeError("execution queue is bound to another streaming DB")
    db_check(stream, "streaming_owners", 293_868)
    db_check(target, "target_owners", 565_155)
    result = {
        "schema": "CANONICAL_V2_PROBE816_MIXED_OWNER_OVERLAY_V1",
        "status": "READY",
        "fail_closed": True,
        "models": audit["models"],
        "user_scope": audit["user_scope"],
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
            "path": str(stream.resolve()), "sha256": sha256(stream), "rows": 293_868,
        },
        "target_database": {
            "path": str(target.resolve()), "sha256": sha256(target), "rows": 565_155,
        },
        "owner_audit": {
            "path": str(args.audit.resolve()), "sha256": sha256(args.audit),
        },
        "counts": {
            "streaming_holes": 0, "streaming_duplicate_owners": 0,
            "target_holes": 0, "target_duplicate_owners": 0,
            "minimax_owner_rows": 0,
        },
        "owner_policy": audit["owner_policy"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
