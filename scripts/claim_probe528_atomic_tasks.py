#!/usr/bin/env python3
"""Acquire immutable per-task ownership for one atomic feature worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--tasks-sha256", required=True)
    parser.add_argument("--queue-sha256", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--claim-root", type=Path, required=True)
    args = parser.parse_args()
    tasks = args.tasks.resolve()
    if sha256(tasks) != args.tasks_sha256:
        raise RuntimeError("worker task-list SHA mismatch")
    rows = [json.loads(line) for line in tasks.open() if line.strip()]
    keys = [str(row["unique_key"]) for row in rows]
    if len(keys) != len(set(keys)) or not keys:
        raise RuntimeError("worker task list is empty or contains duplicate keys")
    root = args.claim_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "PROBE528_ATOMIC_FEATURE_CLAIM_V1",
        "owner": args.owner,
        "queue_sha256": args.queue_sha256,
        "tasks_path": str(tasks),
        "tasks_sha256": args.tasks_sha256,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    created = 0
    for key in sorted(keys):
        path = root / key[:2] / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        claim = {**payload, "unique_key": key}
        encoded = json.dumps(claim, sort_keys=True, separators=(",", ":")) + "\n"
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        except FileExistsError:
            existing = json.loads(path.read_text())
            if (
                existing.get("owner") != args.owner
                or existing.get("queue_sha256") != args.queue_sha256
                or existing.get("tasks_sha256") != args.tasks_sha256
            ):
                raise RuntimeError(f"atomic task already has another owner: {key}")
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        created += 1
    print(json.dumps({"owner": args.owner, "tasks": len(keys), "claims_created": created}, sort_keys=True))


if __name__ == "__main__":
    main()
