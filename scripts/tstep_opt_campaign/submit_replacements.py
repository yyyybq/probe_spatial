#!/usr/bin/env python3
"""Submit an independent-probe plan with a durable exactly-once ledger."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib
import re
import subprocess
import sys
import threading
from datetime import datetime, timezone


SCO = "/mnt/umm/users/yinbaiqiao/.sco/bin/sco"
PILOT_GROUP = "wan_t999_l15_p8"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_locked(path: pathlib.Path, lock: threading.Lock, value: dict) -> None:
    encoded = json.dumps(value, ensure_ascii=False) + "\n"
    with lock:
        with path.open("a") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())


def describe(job_id: str) -> dict:
    proc = subprocess.run(
        [SCO, "acp", "jobs", "describe", "--workspace-name=aigc", "-o", "json", job_id],
        check=True, text=True, stdout=subprocess.PIPE,
    )
    return json.loads(proc.stdout)


def create_one(record: dict, journal: pathlib.Path, lock: threading.Lock) -> dict:
    cmd = [
        SCO, "acp", "jobs", "create", "--workspace-name=aigc", "--aec2-name=zoetrope",
        f"--job-name={record['job_name']}", "--priority=HIGHEST",
        "--container-image-url=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617",
        "--storage-mount=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm",
        "--training-framework=pytorch", "--worker-nodes=1",
        f"--worker-spec={record['worker_spec']}", f"--command={record['command']}",
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ids = re.findall(r"pt-[a-z0-9]+", proc.stdout)
    event = {
        "at": now(), "event": "create_returned", "unique_key": record["unique_key"],
        "old_job_id": record["old_job_id"], "returncode": proc.returncode,
        "output": proc.stdout, "new_job_id": ids[-1] if ids else None,
    }
    append_locked(journal, lock, event)
    if proc.returncode or not ids:
        event["verified"] = False
        event["error"] = "create_failed_or_no_job_id"
        return event
    job_id = ids[-1]
    try:
        job = describe(job_id)
        role = job["roles"][0]
        spec = role["resource_spec"][0]
        checks = {
            "display_name": job["display_name"] == record["job_name"],
            "command": role["startup_script"] == record["command"],
            "one_gpu": int(spec["limits"]["nvidia.com/gpu"]) == 1,
            "worker_spec": spec["name"] == record["worker_spec"],
            "accepted_state": job["state"] in {"CREATING", "QUEUEING", "STARTING", "RUNNING"},
        }
        verified = all(checks.values())
        verified_event = {
            "at": now(), "event": "scheduler_verified", "unique_key": record["unique_key"],
            "old_job_id": record["old_job_id"], "new_job_id": job_id,
            "verified": verified, "checks": checks, "state": job["state"],
            "display_name": job["display_name"],
        }
        append_locked(journal, lock, verified_event)
        return verified_event
    except Exception as exc:
        failed = dict(event, verified=False, error=f"describe_failed: {exc}")
        append_locked(journal, lock, {"at": now(), "event": "verification_error", **failed})
        return failed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=pathlib.Path)
    parser.add_argument("--journal", required=True, type=pathlib.Path)
    parser.add_argument("--mode", required=True, choices=["pilot", "rest", "all"])
    parser.add_argument("--parallel", type=int, default=1)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    args.journal.parent.mkdir(parents=True, exist_ok=True)
    prior = []
    if args.journal.exists():
        prior = [json.loads(line) for line in args.journal.read_text().splitlines() if line]
    mapped = {
        row["unique_key"] for row in prior
        if row.get("new_job_id") and row.get("event") == "create_returned"
    }
    records = []
    for record in plan["replacements"]:
        pilot = record["group_id"] == PILOT_GROUP
        selected = args.mode == "all" or (args.mode == "pilot" and pilot) or (args.mode == "rest" and not pilot)
        if selected and record["unique_key"] not in mapped:
            records.append(record)
    lock = threading.Lock()
    results = []
    if args.parallel == 1:
        for record in records:
            results.append(create_one(record, args.journal, lock))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = [pool.submit(create_one, record, args.journal, lock) for record in records]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
    summary = {
        "at": now(), "event": "submit_summary", "mode": args.mode,
        "selected_unmapped": len(records),
        "verified": sum(bool(r.get("verified")) for r in results),
        "failed": [r for r in results if not r.get("verified")],
    }
    append_locked(args.journal, lock, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if not summary["failed"] else 2


if __name__ == "__main__":
    sys.exit(main())
