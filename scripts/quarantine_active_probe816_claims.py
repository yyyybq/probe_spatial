#!/usr/bin/env python3
"""Atomically quarantine unclaimed model items in live probe816 shards.

Existing claims are never modified: they may belong to a currently running
worker.  Newly created claim directories contain a durable reason receipt and
cause the immutable queue runner to record SKIPPED_CLAIMED for that item.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def atomic_json(path: Path, value: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    with open(descriptor, "w", encoding="utf-8", closefd=True) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    Path(temporary).replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("queue", type=Path, nargs="+")
    args = parser.parse_args()

    rows: dict[str, dict] = {}
    for queue in args.queue:
        for line in queue.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("model") != args.model:
                continue
            key = str(row["unique_key"])
            if key in rows:
                raise RuntimeError(f"duplicate quarantined key: {key}")
            rows[key] = row

    claim_root = args.campaign / "claims"
    claim_root.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    already_claimed: list[str] = []
    already_quarantined: list[str] = []
    timestamp = datetime.now(timezone.utc).isoformat()
    for key in sorted(rows):
        claim = claim_root / key
        try:
            claim.mkdir()
        except FileExistsError:
            if (claim / "QUARANTINED.json").is_file():
                already_quarantined.append(key)
            else:
                already_claimed.append(key)
            continue
        receipt = {
            "schema": "PROBE816_LIVE_QUARANTINE_CLAIM_V1",
            "created_utc": timestamp,
            "unique_key": key,
            "model": args.model,
            "reason": "source FP16 feature payload contains non-finite values; training loss can be silently zeroed",
            "completion_evidence": False,
            "resume_allowed": False,
        }
        atomic_json(claim / "QUARANTINED.json", receipt)
        created.append(key)

    report = {
        "schema": "PROBE816_LIVE_QUARANTINE_REPORT_V1",
        "created_utc": timestamp,
        "campaign": str(args.campaign.resolve()),
        "model": args.model,
        "source_keys": len(rows),
        "newly_quarantined": len(created),
        "already_quarantined": len(already_quarantined),
        "already_claimed_by_live_or_prior_worker": len(already_claimed),
        "newly_quarantined_keys": created,
        "already_claimed_keys": already_claimed,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.report, report)
    print(json.dumps({k: v for k, v in report.items() if not k.endswith("_keys")}, indent=2))


if __name__ == "__main__":
    main()
