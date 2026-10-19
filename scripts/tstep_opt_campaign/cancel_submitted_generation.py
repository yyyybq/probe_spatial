#!/usr/bin/env python3
"""Stop one exact submitted replacement generation and wait for terminal state."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib
import subprocess
import threading
import time
from collections import Counter
from datetime import datetime, timezone


SCO = "/mnt/umm/users/yinbaiqiao/.sco/bin/sco"
TERMINAL = {"SUCCEEDED", "FAILED", "SUSPENDED", "STOPPED", "DELETED"}


def call(*args):
    return subprocess.run([SCO, "acp", "jobs", *args], text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def describe(job_id):
    proc = call("describe", "--workspace-name=aigc", "-o", "json", job_id)
    if proc.returncode:
        raise RuntimeError(proc.stdout)
    return json.loads(proc.stdout)


def now():
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=pathlib.Path)
    parser.add_argument("--submission-journal", required=True, type=pathlib.Path)
    parser.add_argument("--journal", required=True, type=pathlib.Path)
    parser.add_argument("--parallel", type=int, default=12)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    planned = {row["unique_key"]: row for row in plan["replacements"]}
    mapping = {}
    for line in args.submission_journal.read_text().splitlines():
        event = json.loads(line)
        if event.get("event") == "create_returned" and event.get("new_job_id"):
            mapping[event["unique_key"]] = event["new_job_id"]
    if set(mapping) != set(planned):
        raise RuntimeError("refusing generation cancel: mapping does not exactly equal plan")
    lock = threading.Lock()
    args.journal.parent.mkdir(parents=True, exist_ok=True)

    def append(event):
        with lock, args.journal.open("a") as handle:
            handle.write(json.dumps(event) + "\n")
            handle.flush(); os.fsync(handle.fileno())

    def stop_one(item):
        key, job_id = item
        before = describe(job_id)
        if before["display_name"] != planned[key]["job_name"]:
            raise RuntimeError(f"identity mismatch for {job_id}")
        event = {"at": now(), "event": "pre_stop", "unique_key": key,
                 "job_id": job_id, "state": before["state"],
                 "start_time": before.get("start_time")}
        append(event)
        if before["state"] in TERMINAL:
            return before["state"]
        proc = call("stop", "--workspace-name=aigc", job_id)
        append({"at": now(), "event": "stop_returned", "unique_key": key,
                "job_id": job_id, "returncode": proc.returncode, "output": proc.stdout})
        if proc.returncode:
            raise RuntimeError(f"stop failed for {job_id}: {proc.stdout}")
        return "STOP_REQUESTED"

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        initial = list(pool.map(stop_one, sorted(mapping.items())))
    deadline = time.time() + 600
    remaining = dict(mapping)
    final = {}
    while remaining and time.time() < deadline:
        def check(item):
            key, job_id = item
            return key, describe(job_id)["state"]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
            checked = list(pool.map(check, remaining.items()))
        for key, state in checked:
            if state in TERMINAL:
                final[key] = state
                append({"at": now(), "event": "terminal", "unique_key": key,
                        "job_id": mapping[key], "state": state})
                remaining.pop(key)
        if remaining:
            time.sleep(5)
    summary = {"at": now(), "event": "summary", "jobs": len(mapping),
               "initial_states": dict(Counter(initial)), "terminal_states": dict(Counter(final.values())),
               "timeouts": sorted(remaining)}
    append(summary)
    print(json.dumps(summary, indent=2))
    return 0 if not remaining else 2


if __name__ == "__main__":
    raise SystemExit(main())
