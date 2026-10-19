#!/usr/bin/env python3
"""Produce an exhaustive old-experiment -> terminal category/new-job ledger."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import subprocess
from collections import Counter
from datetime import datetime, timezone


SCO = "/mnt/umm/users/yinbaiqiao/.sco/bin/sco"


def scheduler_jobs() -> dict[str, dict]:
    proc = subprocess.run(
        [SCO, "acp", "jobs", "list", "--workspace-name=aigc", "--page-size", "500", "-o", "json"],
        check=True, text=True, stdout=subprocess.PIPE,
    )
    return {row["name"]: row for row in json.loads(proc.stdout)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", required=True, type=pathlib.Path)
    parser.add_argument("--plan", required=True, type=pathlib.Path)
    parser.add_argument("--journal", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--fresh-verification", type=pathlib.Path)
    parser.add_argument("--protected-verification", type=pathlib.Path)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    plan = json.loads(args.plan.read_text())
    events = [json.loads(line) for line in args.journal.read_text().splitlines() if line]
    mapping = {}
    for event in events:
        if event.get("event") == "create_returned" and event.get("new_job_id"):
            key = event["unique_key"]
            if key in mapping and mapping[key] != event["new_job_id"]:
                raise RuntimeError(f"duplicate replacement writer for {key}: {mapping[key]}, {event['new_job_id']}")
            mapping[key] = event["new_job_id"]
    planned = {row["unique_key"] for row in plan["replacements"]}
    planned_rows = {row["unique_key"]: row for row in plan["replacements"]}
    extras = set(mapping) - planned
    if extras:
        raise RuntimeError(f"submitted keys outside plan: {sorted(extras)[:3]}")
    scheduler = scheduler_jobs()
    if args.fresh_verification:
        fresh = json.loads(args.fresh_verification.read_text())
        scheduler.update({row["job_id"]: row["scheduler"] for row in fresh["jobs"] if row["scheduler"]})
    if args.protected_verification:
        protected_fresh = json.loads(args.protected_verification.read_text())
        scheduler.update({row["name"]: row for row in protected_fresh["jobs"]})
    old_jobs = {row["job_id"]: row for row in inventory["jobs"]}
    protected = set(plan["protected_old_jobs"])
    rows = []
    for exp in inventory["experiments"]:
        key = exp["unique_key"]
        replacement = planned_rows.get(key, {})
        complete = pathlib.Path(exp["output_evidence"]["train_complete_path"]).is_file()
        new_job_id = mapping.get(key)
        if complete:
            category, reason = "已完成", "validated TRAIN_COMPLETE marker"
        elif exp["old_job_id"] in protected:
            state = scheduler.get(exp["old_job_id"], old_jobs[exp["old_job_id"]]["scheduler"])["state"]
            category, reason = "保留运行", f"protected whole old job; scheduler={state}"
        elif new_job_id:
            state = scheduler.get(new_job_id, {}).get("state", "NOT_IN_FIRST_500")
            category, reason = "新任务已提交", f"scheduler={state}"
        else:
            category, reason = "具体原因阻塞", "planned replacement has not been submitted"
        rows.append({
            "unique_key": key, "old_job_id": exp["old_job_id"],
            "category": category, "reason": reason, "new_job_id": new_job_id,
            "campaign_id": exp["campaign_id"], "backend": exp["backend"],
            "source": exp["source"], "timestep": exp["timestep"],
            "layer": exp["layer"], "prefix": exp["prefix"],
            "horizon": exp["horizon"], "probe": exp["probe"],
            "seed": exp["seed"],
            "old_output_dir": exp["output_evidence"]["run_dir"],
            "new_output_dir": replacement.get("new_output_dir"),
        })
    if len(rows) != len({row["unique_key"] for row in rows}):
        raise RuntimeError("reconciliation contains duplicate experiment keys")
    counts = Counter(row["category"] for row in rows)
    report = {
        "schema": "tstep-probe-reconciliation-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "counts": dict(counts), "experiments": rows,
        "submitted_mapping_count": len(mapping),
        "planned_replacement_count": len(planned),
        "extra_submissions": [],
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    tsv_path = args.output.with_suffix(".tsv")
    fieldnames = [
        "unique_key", "campaign_id", "backend", "source", "timestep", "layer",
        "prefix", "horizon", "probe", "seed", "old_job_id", "category",
        "reason", "new_job_id", "old_output_dir", "new_output_dir",
    ]
    with tsv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"counts": report["counts"], "mapping": len(mapping)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
