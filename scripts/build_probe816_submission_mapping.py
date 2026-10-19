#!/usr/bin/env python3
"""Create the exact 816-row old experiment -> unique owner -> new job map."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


EXPECTED_EXPERIMENTS_SHA = "e0c035595fd7c11af7dbb928717de2518ddd2e52d97424658964aeab19ae4470"
EXPECTED_OWNER_MAP_SHA = "998f38078186dfbc15d3260054b0c5902cc8b75e4ccbd3b022e8db9199868812"
JOBS = {
    "pilot": "pt-ddaz35ft",
    "remainder_00": "pt-yeyon3ui",
    "remainder_01": "pt-fk787qrs",
    "remainder_02": "pt-f8qsm780",
    "remainder_03": "pt-uxcegr23",
    "remainder_04": "pt-e7vedn9x",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=Path, required=True)
    parser.add_argument("--owner-map", type=Path, required=True)
    parser.add_argument("--queues-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError(f"output exists: {args.output_dir}")
    if sha256(args.experiments) != EXPECTED_EXPERIMENTS_SHA or sha256(args.owner_map) != EXPECTED_OWNER_MAP_SHA:
        raise ValueError("frozen registry/owner-map SHA mismatch")
    with args.experiments.open(newline="", encoding="utf-8") as handle:
        experiments = {row["experiment_id"]: row for row in csv.DictReader(handle)}
    owner_rows = [json.loads(line) for line in args.owner_map.open(encoding="utf-8") if line.strip()]
    key_to_shard = {}
    queue_bindings = {}
    for shard in JOBS:
        filename = "pilot_8.jsonl" if shard == "pilot" else f"{shard}.jsonl"
        path = args.queues_dir / filename
        rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        queue_bindings[shard] = {"path": str(path.resolve()), "sha256": sha256(path), "rows": len(rows), "job_id": JOBS[shard]}
        for row in rows:
            key = row["unique_key"]
            if key in key_to_shard:
                raise ValueError(f"duplicate queue ownership: {key}")
            key_to_shard[key] = shard
    if len(key_to_shard) != 760:
        raise ValueError(f"expected 760 unique execution owners, got {len(key_to_shard)}")
    output_rows = []
    for owner in owner_rows:
        old = experiments[owner["experiment_id"]]
        shard = key_to_shard[owner["unique_key"]]
        output_rows.append({
            "experiment_id": owner["experiment_id"],
            "alias_role": owner["role"],
            "owner_experiment_id": owner["owner_experiment_id"],
            "unique_key": owner["unique_key"],
            "old_train_job_id": old["train_job_id"],
            "old_training_status": old["training_status"],
            "old_run_artifact_path": old["run_artifact_path"],
            "old_checkpoint_path": old["checkpoint_path"],
            "new_queue_shard": shard,
            "new_job_id": JOBS[shard],
            "new_output_dir": (
                "/mnt/umm/users/yinbaiqiao/probe_spatial/logs/inscene15k_streaming/runs/"
                f"main_v2_probe_v2_r5/main_v2_probe_v2_r5_20260919_r1/{owner['unique_key']}"
            ),
            "submission_status": "SUBMITTED",
        })
    if len(output_rows) != 816 or len({row["experiment_id"] for row in output_rows}) != 816:
        raise ValueError("mapping is not an exact 816-row registry cover")
    args.output_dir.mkdir(parents=True)
    csv_path = args.output_dir / "old_experiment_to_new_job_816.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader(); writer.writerows(sorted(output_rows, key=lambda row: row["experiment_id"]))
    manifest = {
        "schema": "MAIN_V2_PROBE816_SUBMISSION_MAPPING_V1", "status": "SUBMITTED",
        "source_registry": {"path": str(args.experiments.resolve()), "sha256": EXPECTED_EXPERIMENTS_SHA, "rows": 816},
        "owner_map": {"path": str(args.owner_map.resolve()), "sha256": EXPECTED_OWNER_MAP_SHA, "rows": 816},
        "mapping": {"path": str(csv_path.resolve()), "sha256": sha256(csv_path), "rows": 816},
        "counts": {"source_rows": 816, "unique_execution_units": 760, "duplicate_aliases": 56, "new_scheduler_jobs": 6},
        "queues": queue_bindings,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

