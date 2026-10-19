#!/usr/bin/env python3
"""Bind payload-readable streaming/target owner DBs into a fail-closed overlay.

Row coverage is deliberately insufficient: an owner route can have a valid
key, index and header metadata while the actual ``.sft`` object is absent.
This builder therefore requires an independently frozen materialization audit
covering the exact two database digests before it may emit ``READY``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def db_status(path: Path, table: str) -> tuple[str, int]:
    con = sqlite3.connect(f"file:{path.resolve()}?mode=ro&immutable=1", uri=True)
    try:
        status = con.execute("SELECT value FROM metadata WHERE key='status'").fetchone()
        rows = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        con.close()
    return ("" if status is None else str(status[0])), int(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--streaming-audit", type=Path, required=True)
    parser.add_argument("--target-audit", type=Path, required=True)
    parser.add_argument("--experiments", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--frozen-queue-manifest", type=Path, required=True)
    parser.add_argument("--payload-audit", type=Path, required=True,
                        help="immutable full materialization audit for these DB digests")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"output exists: {args.output}")
    stream_audit = json.loads(args.streaming_audit.read_text())
    target_audit = json.loads(args.target_audit.read_text())
    for name, audit in (("streaming", stream_audit), ("target", target_audit)):
        if audit.get("status") != "READY" or audit.get("fail_closed") is not True:
            raise ValueError(f"{name} audit is not fail-closed READY")
        if audit.get("zero_holes") is not True or audit.get("zero_duplicate_effective_owners") is not True:
            raise ValueError(f"{name} audit is not zero-hole/zero-duplicate")
    stream_db = Path(stream_audit["database"]["path"])
    target_db = Path(target_audit["database"]["path"])
    if sha256(stream_db) != stream_audit["database"]["sha256"]:
        raise ValueError("streaming DB SHA mismatch")
    if sha256(target_db) != target_audit["database"]["sha256"]:
        raise ValueError("target DB SHA mismatch")
    stream_status, stream_rows = db_status(stream_db, "streaming_owners")
    target_status, target_rows = db_status(target_db, "target_owners")
    if (stream_status, stream_rows) != ("READY", 207_624):
        raise ValueError(f"bad streaming DB contract: {stream_status}/{stream_rows}")
    if (target_status, target_rows) != ("READY", 715_863):
        raise ValueError(f"bad target DB contract: {target_status}/{target_rows}")
    payload_audit = json.loads(args.payload_audit.read_text())
    if (payload_audit.get("schema") != "CANONICAL_V2_PROBE816_PAYLOAD_MATERIALIZATION_AUDIT_V1"
            or payload_audit.get("status") != "READY"
            or payload_audit.get("fail_closed") is not True):
        raise ValueError("payload audit is not fail-closed READY")
    bindings = payload_audit.get("database_bindings") or {}
    if bindings.get("streaming_sha256") != sha256(stream_db):
        raise ValueError("payload audit streaming DB binding mismatch")
    if bindings.get("target_sha256") != sha256(target_db):
        raise ValueError("payload audit target DB binding mismatch")
    for kind in ("streaming", "target"):
        report = payload_audit.get(kind) or {}
        if (int(report.get("missing_payloads", -1)) != 0
                or int(report.get("unreadable_payloads", -1)) != 0
                or int(report.get("size_mismatches", -1)) != 0
                or int(report.get("header_mismatches", -1)) != 0
                or int(report.get("verified_payloads", -1)) != int(report.get("expected_payloads", -2))):
            raise ValueError(f"payload audit is incomplete for {kind}: {report}")
    frozen_queue = json.loads(args.frozen_queue_manifest.read_text())
    if frozen_queue.get("status") != "FROZEN_NOT_SUBMITTED":
        raise ValueError("queue is not the pre-submission frozen queue")
    result = {
        "schema": "CANONICAL_V2_PROBE816_MIXED_OWNER_OVERLAY_V1",
        "status": "READY",
        "fail_closed": True,
        "experiments": {"path": str(args.experiments.resolve()), "sha256": sha256(args.experiments), "rows": 816},
        "canonical_manifest": {"path": str(args.canonical_manifest.resolve()), "sha256": sha256(args.canonical_manifest)},
        "frozen_queue_manifest": {"path": str(args.frozen_queue_manifest.resolve()), "sha256": sha256(args.frozen_queue_manifest)},
        "streaming_database": {"path": str(stream_db.resolve()), "sha256": sha256(stream_db), "rows": stream_rows},
        "target_database": {"path": str(target_db.resolve()), "sha256": sha256(target_db), "rows": target_rows},
        "payload_materialization_audit": {"path": str(args.payload_audit.resolve()), "sha256": sha256(args.payload_audit)},
        "owner_policy": "exact composite key; no path fallback; verify size/header/full SHA when bound",
        "counts": {"streaming_holes": 0, "streaming_duplicate_owners": 0, "target_holes": 0, "target_duplicate_owners": 0},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
