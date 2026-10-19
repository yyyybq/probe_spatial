#!/usr/bin/env python3
"""Freshly describe every submitted replacement and verify exact scheduler identity."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
import subprocess
from collections import Counter
from datetime import datetime, timezone


SCO = "/mnt/umm/users/yinbaiqiao/.sco/bin/sco"


def describe(item):
    key, job_id, record = item
    proc = subprocess.run(
        [SCO, "acp", "jobs", "describe", "--workspace-name=aigc", "-o", "json", job_id],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if proc.returncode:
        return {"unique_key": key, "job_id": job_id, "verified": False,
                "error": proc.stdout, "scheduler": None}
    job = json.loads(proc.stdout)
    role, spec = job["roles"][0], job["roles"][0]["resource_spec"][0]
    checks = {
        "display_name": job["display_name"] == record["job_name"],
        "command": role["startup_script"] == record["command"],
        "one_gpu": int(spec["limits"]["nvidia.com/gpu"]) == 1,
        "worker_spec": spec["name"] == record["worker_spec"],
        "nonfailure_state": job["state"] not in {"FAILED", "SUSPENDED", "STOPPED", "DELETED"},
    }
    return {"unique_key": key, "old_job_id": record["old_job_id"],
            "job_id": job_id, "verified": all(checks.values()), "checks": checks,
            "scheduler": job}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=pathlib.Path)
    parser.add_argument("--journal", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--parallel", type=int, default=12)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    records = {row["unique_key"]: row for row in plan["replacements"]}
    mapping = {}
    for line in args.journal.read_text().splitlines():
        event = json.loads(line)
        if event.get("event") == "create_returned" and event.get("new_job_id"):
            key, job_id = event["unique_key"], event["new_job_id"]
            if key in mapping and mapping[key] != job_id:
                raise RuntimeError(f"duplicate job IDs for {key}")
            mapping[key] = job_id
    if set(mapping) != set(records):
        raise RuntimeError(f"mapping/plan mismatch: mapped={len(mapping)}, planned={len(records)}")
    items = [(key, mapping[key], records[key]) for key in sorted(records)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        rows = list(pool.map(describe, items))
    states = Counter(
        row["scheduler"]["state"] if row["scheduler"] else "DESCRIBE_ERROR" for row in rows
    )
    report = {
        "schema": "tstep-probe-fresh-scheduler-verification-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(), "jobs": rows,
        "counts": {"total": len(rows), "verified": sum(row["verified"] for row in rows),
                   "states": dict(states)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["counts"], indent=2))
    return 0 if all(row["verified"] for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
