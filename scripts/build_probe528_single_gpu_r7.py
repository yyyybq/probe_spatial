#!/usr/bin/env python3
"""Freeze the user-approved 528-unit scope into one-item Zoetrope jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


TARGET_PROBES = {"action_dynamics", "path_integration"}
TARGET_GAPS = {
    "cogvideox": {0, 5, 10, 30, 35, 40},
    "dino": {0},
    "f3r": {24},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--queue-sha256", required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    queue = args.queue.resolve()
    output = args.output.resolve()
    if sha256(queue) != args.queue_sha256:
        raise RuntimeError("queue SHA mismatch")
    if output.exists():
        raise RuntimeError(f"output exists: {output}")
    runtime = json.loads(args.runtime_manifest.read_text())
    if runtime.get("status") != "FROZEN":
        raise RuntimeError("runtime is not frozen")
    rows = [json.loads(line) for line in queue.read_text().splitlines() if line.strip()]
    if len(rows) != 528 or len({row["unique_key"] for row in rows}) != 528:
        raise RuntimeError("expected 528 unique queue rows")
    records = []
    for ordinal, row in enumerate(rows):
        item = output / "items" / f"{ordinal:04d}_{hashlib.sha256(row['unique_key'].encode()).hexdigest()[:12]}.jsonl"
        write_exclusive(item, json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        blocked = (
            row["probe"] in TARGET_PROBES
            and int(row["layer"]) in TARGET_GAPS.get(row["model"], set())
        )
        records.append({
            "ordinal": ordinal,
            "unique_key": row["unique_key"],
            "model": row["model"],
            "layer": int(row["layer"]),
            "prefix": int(row["prefix"]),
            "probe": row["probe"],
            "item_path": str(item),
            "item_sha256": sha256(item),
            "submission_gate": "BLOCKED_TARGET_GAP" if blocked else "ELIGIBLE",
        })
    counts = Counter(record["submission_gate"] for record in records)
    if counts != {"ELIGIBLE": 464, "BLOCKED_TARGET_GAP": 64}:
        raise RuntimeError(f"unexpected target-gap partition: {counts}")
    ledger = {
        "schema": "MAIN_V2_PROBE528_SINGLE_GPU_R7_PLAN_V1",
        "status": "FROZEN_NOT_SUBMITTED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "resource_pool": "zoetrope",
        "worker_spec": "N4lS.Iq.I80.1",
        "priority": "HIGHEST",
        "queue": {"path": str(queue), "sha256": args.queue_sha256},
        "runtime_manifest": {
            "path": str(args.runtime_manifest.resolve()),
            "sha256": sha256(args.runtime_manifest.resolve()),
            "source_sha256_file_sha256": runtime["source_sha256_file_sha256"],
        },
        "counts": dict(counts),
        "target_gap_rule": {
            "source_audit": "valid7_target_coverage_20260922_r1/audit.json",
            "cogvideox_layers": sorted(TARGET_GAPS["cogvideox"]),
            "dino_layers": sorted(TARGET_GAPS["dino"]),
            "f3r_layers": sorted(TARGET_GAPS["f3r"]),
            "affected_probes": sorted(TARGET_PROBES),
        },
        "jobs": records,
    }
    write_exclusive(output / "plan.json", json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": ledger["status"], "counts": dict(counts)}, sort_keys=True))


if __name__ == "__main__":
    main()
