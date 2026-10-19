#!/usr/bin/env python3
"""Safely stop only not-yet-running jobs from a frozen campaign inventory."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
from datetime import datetime, timezone


SCO = "/mnt/umm/users/yinbaiqiao/.sco/bin/sco"
CANCELLABLE = {"STARTING", "QUEUEING", "PENDING", "CREATING", "SCHEDULING", "WAITING"}
PROTECT = {"RUNNING"}
TERMINAL = {"SUSPENDED", "CANCELED", "CANCELLED", "STOPPED", "FAILED", "SUCCEEDED", "DELETED"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def describe(job_id: str) -> dict:
    proc = subprocess.run(
        [SCO, "acp", "jobs", "describe", "--workspace-name=aigc", "-o", "json", job_id],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return json.loads(proc.stdout)


def append_event(path: pathlib.Path, event: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        handle.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", required=True, type=pathlib.Path)
    parser.add_argument("--journal", required=True, type=pathlib.Path)
    parser.add_argument("--wait-seconds", type=int, default=900)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    if inventory.get("scope_rule") != "exact membership in the two launcher/sco_job_ids.txt files":
        raise SystemExit("refusing inventory with unexpected scope rule")
    args.journal.parent.mkdir(parents=True, exist_ok=True)
    if args.journal.exists():
        raise SystemExit(f"refusing to append to existing journal: {args.journal}")

    # Only jobs that were cancellable in the immutable pre-cancel snapshot are
    # candidates.  Every one is described again immediately before stop.
    candidates = [
        record for record in inventory["jobs"] if record["scheduler"]["state"] in CANCELLABLE
    ]
    protected = []
    requested = []
    skipped = []
    for record in candidates:
        job_id = record["job_id"]
        fresh = describe(job_id)
        state = fresh["state"]
        checked = {"at": now(), "event": "pre_stop_check", "job_id": job_id, "state": state,
                   "start_time": fresh.get("start_time"), "display_name": fresh.get("display_name")}
        append_event(args.journal, checked)
        # A populated start_time is treated conservatively even if the control
        # plane has not yet converged to RUNNING.
        if state in PROTECT or fresh.get("start_time"):
            protected.append(job_id)
            append_event(args.journal, {"at": now(), "event": "protected", "job_id": job_id,
                                        "reason": "running_or_has_start_time", "state": state})
            continue
        if state in TERMINAL:
            skipped.append(job_id)
            append_event(args.journal, {"at": now(), "event": "already_terminal", "job_id": job_id,
                                        "state": state})
            continue
        if state not in CANCELLABLE:
            protected.append(job_id)
            append_event(args.journal, {"at": now(), "event": "protected", "job_id": job_id,
                                        "reason": "unexpected_state_fail_closed", "state": state})
            continue
        proc = subprocess.run(
            [SCO, "acp", "jobs", "stop", "--workspace-name=aigc", job_id],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        append_event(args.journal, {"at": now(), "event": "stop_requested", "job_id": job_id,
                                    "pre_state": state, "returncode": proc.returncode,
                                    "output": proc.stdout})
        if proc.returncode != 0:
            raise RuntimeError(f"stop failed for {job_id}: {proc.stdout}")
        requested.append(job_id)

    deadline = time.monotonic() + args.wait_seconds
    pending = set(requested)
    final = {}
    while pending and time.monotonic() < deadline:
        for job_id in list(pending):
            fresh = describe(job_id)
            state = fresh["state"]
            if state in TERMINAL:
                final[job_id] = state
                pending.remove(job_id)
                append_event(args.journal, {"at": now(), "event": "terminal_confirmed",
                                            "job_id": job_id, "state": state})
            elif state == "RUNNING" or fresh.get("start_time"):
                # This should be impossible after an accepted stop, but fail closed
                # and make the anomaly explicit rather than issuing another signal.
                protected.append(job_id)
                pending.remove(job_id)
                append_event(args.journal, {"at": now(), "event": "post_stop_running_anomaly",
                                            "job_id": job_id, "state": state,
                                            "start_time": fresh.get("start_time")})
        if pending:
            time.sleep(10)

    summary = {
        "at": now(),
        "event": "summary",
        "candidate_count": len(candidates),
        "stop_requested": requested,
        "already_terminal": skipped,
        "newly_protected": sorted(set(protected)),
        "terminal_states": final,
        "timed_out": sorted(pending),
    }
    append_event(args.journal, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 2 if pending else 0


if __name__ == "__main__":
    sys.exit(main())
