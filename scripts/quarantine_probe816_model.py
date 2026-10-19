#!/usr/bin/env python3
"""Split immutable probe queues by an explicitly quarantined model.

The output is a new namespace.  The source queues are never changed, and the
manifest proves exact, disjoint coverage of every source unique key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quarantine-model", required=True)
    args = parser.parse_args()

    inputs = sorted(args.input_dir.glob("*.jsonl"))
    if not inputs:
        raise SystemExit("no JSONL queues found")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    source_keys: set[str] = set()
    eligible_keys: set[str] = set()
    quarantine_keys: set[str] = set()
    shards: list[dict] = []

    for source in inputs:
        rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        keys = [str(row["unique_key"]) for row in rows]
        if len(keys) != len(set(keys)):
            raise RuntimeError(f"duplicate unique_key inside {source}")
        overlap = source_keys.intersection(keys)
        if overlap:
            raise RuntimeError(f"cross-shard duplicates: {sorted(overlap)[:3]}")
        source_keys.update(keys)

        quarantined = [row for row in rows if row.get("model") == args.quarantine_model]
        eligible = [row for row in rows if row.get("model") != args.quarantine_model]
        eligible_keys.update(str(row["unique_key"]) for row in eligible)
        quarantine_keys.update(str(row["unique_key"]) for row in quarantined)

        eligible_path = args.output_dir / f"{source.stem}.eligible.jsonl"
        quarantine_path = args.output_dir / f"{source.stem}.quarantine.jsonl"
        write_jsonl(eligible_path, eligible)
        write_jsonl(quarantine_path, quarantined)
        shards.append(
            {
                "source": str(source.resolve()),
                "source_rows": len(rows),
                "source_sha256": sha256(source),
                "eligible": {
                    "path": str(eligible_path.resolve()),
                    "rows": len(eligible),
                    "sha256": sha256(eligible_path),
                    "models": dict(sorted(Counter(row["model"] for row in eligible).items())),
                },
                "quarantine": {
                    "path": str(quarantine_path.resolve()),
                    "rows": len(quarantined),
                    "sha256": sha256(quarantine_path),
                },
            }
        )

    if eligible_keys & quarantine_keys:
        raise RuntimeError("eligible/quarantine overlap")
    if eligible_keys | quarantine_keys != source_keys:
        raise RuntimeError("split does not exactly cover source")

    manifest = {
        "schema": "PROBE816_MODEL_QUARANTINE_SPLIT_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "quarantine_model": args.quarantine_model,
        "source_unique_keys": len(source_keys),
        "eligible_unique_keys": len(eligible_keys),
        "quarantine_unique_keys": len(quarantine_keys),
        "exact_cover": True,
        "disjoint": True,
        "shards": shards,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.chmod(manifest_path, 0o444)
    for path in args.output_dir.glob("*.jsonl"):
        os.chmod(path, 0o444)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
