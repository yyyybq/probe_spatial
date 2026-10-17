#!/usr/bin/env python3
"""Combine the three dated experiment inventories without merging their scopes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("canonical_v2", "timestep", "analyses")
REQUIRED = {
    "experiment_id", "entry_level", "model", "probe", "layer", "prefix", "seed",
    "training_status", "correctness_status", "result_status", "ready_for_release",
    "remaining_work", "evidence_path", "observed_at",
}
STATUSES = {
    "entry_level": {"run", "package", "scope"},
    "training_status": {"COMPLETE", "INCOMPLETE", "UNKNOWN", "NOT_APPLICABLE"},
    "correctness_status": {
        "VERIFIED", "PROTOCOL_INVALID", "DEGENERATE", "NOT_VERIFIED", "NOT_APPLICABLE",
    },
    "ready_for_release": {"YES_WITH_SCOPE", "NO", "UNKNOWN"},
}


def read_inventory(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows = list(reader)
    ids: set[str] = set()
    for line, row in enumerate(rows, 2):
        identity = row["experiment_id"]
        if not identity or identity in ids:
            raise ValueError(f"{path}:{line}: empty or duplicate experiment_id {identity!r}")
        ids.add(identity)
        for field, accepted in STATUSES.items():
            if row[field] not in accepted:
                raise ValueError(f"{path}:{line}: unknown {field} {row[field]!r}")
        if row["ready_for_release"] == "YES_WITH_SCOPE" and row["correctness_status"] not in {
            "VERIFIED", "NOT_APPLICABLE",
        }:
            raise ValueError(f"{path}:{line}: release candidate lacks verified result evidence")
    return rows


def summarize(rows: list[dict[str, str]]) -> dict:
    by_level: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_level[row["entry_level"]].append(row)
    return {
        level: {
            "entries": len(items),
            "training_status": dict(sorted(Counter(r["training_status"] for r in items).items())),
            "correctness_status": dict(sorted(Counter(r["correctness_status"] for r in items).items())),
            "result_status": dict(sorted(Counter(r["result_status"] for r in items).items())),
            "ready_for_release": dict(sorted(Counter(r["ready_for_release"] for r in items).items())),
            "training_complete_and_scope_verified": sum(
                r["training_status"] == "COMPLETE" and r["correctness_status"] == "VERIFIED"
                for r in items
            ),
        }
        for level, items in sorted(by_level.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "metadata/release_inventory")
    args = parser.parse_args()
    combined: list[dict[str, str]] = []
    groups = {}
    inputs = {}
    for group in GROUPS:
        path = args.input_dir / f"{group}.csv"
        rows = read_inventory(path)
        groups[group] = summarize(rows)
        inputs[group] = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        combined.extend({"inventory_group": group, **row} for row in rows)

    common = ["inventory_group", "experiment_id", "entry_level", "model", "probe", "layer",
              "prefix", "seed", "training_status", "correctness_status", "result_status",
              "ready_for_release", "remaining_work", "evidence_path", "observed_at"]
    fields = common + sorted({field for row in combined for field in row} - set(common))
    with (args.input_dir / "all_experiments.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(combined)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope_note": "Local evidence inventory, not a live scheduler query. Do not sum run, package, or scope entries.",
        "ready_definition": "YES_WITH_SCOPE is a scientific result candidate within its declared split/protocol, not completed code packaging or publication.",
        "correctness_definition": "VERIFIED applies only to each row's declared scope. Canonical V2 verifies training identity and input protocol gates, not final metrics; consult correctness_scope and evaluation_acceptance_status.",
        "inputs": inputs,
        "groups": groups,
    }
    (args.input_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(groups, indent=2))


if __name__ == "__main__":
    main()
