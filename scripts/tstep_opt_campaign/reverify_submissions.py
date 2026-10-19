#!/usr/bin/env python3
"""Re-describe created jobs and append durable verification events."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
from datetime import datetime, timezone


SCO = "/mnt/umm/users/yinbaiqiao/.sco/bin/sco"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=pathlib.Path)
    parser.add_argument("--journal", required=True, type=pathlib.Path)
    parser.add_argument("--only-group")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    records = {row["unique_key"]: row for row in plan["replacements"]}
    events = [json.loads(line) for line in args.journal.read_text().splitlines() if line]
    created = {}
    for event in events:
        if event.get("event") == "create_returned" and event.get("new_job_id"):
            created[event["unique_key"]] = event["new_job_id"]
    output = []
    with args.journal.open("a") as handle:
        for key, job_id in sorted(created.items()):
            record = records[key]
            if args.only_group and record["group_id"] != args.only_group:
                continue
            proc = subprocess.run(
                [SCO, "acp", "jobs", "describe", "--workspace-name=aigc", "-o", "json", job_id],
                check=True, text=True, stdout=subprocess.PIPE,
            )
            job = json.loads(proc.stdout)
            role, spec = job["roles"][0], job["roles"][0]["resource_spec"][0]
            checks = {
                "display_name": job["display_name"] == record["job_name"],
                "command": role["startup_script"] == record["command"],
                "one_gpu": int(spec["limits"]["nvidia.com/gpu"]) == 1,
                "worker_spec": spec["name"] == record["worker_spec"],
                "accepted_state": job["state"] in {
                    "CREATING", "QUEUEING", "STARTING", "RUNNING", "SUCCEEDED"
                },
            }
            event = {
                "at": now(), "event": "scheduler_reverified", "unique_key": key,
                "old_job_id": record["old_job_id"], "new_job_id": job_id,
                "verified": all(checks.values()), "checks": checks,
                "state": job["state"], "start_time": job.get("start_time"),
                "complete_time": job.get("complete_time"),
            }
            handle.write(json.dumps(event) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            output.append(event)
    summary = {"jobs": len(output), "verified": sum(row["verified"] for row in output),
               "states": {state: sum(row["state"] == state for row in output)
                          for state in sorted({row["state"] for row in output})}}
    print(json.dumps(summary, indent=2))
    return 0 if all(row["verified"] for row in output) else 2


if __name__ == "__main__":
    raise SystemExit(main())
