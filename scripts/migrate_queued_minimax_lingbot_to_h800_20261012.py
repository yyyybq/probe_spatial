#!/usr/bin/env python3
"""Move only never-started MiniMax/LingBot queue entries from Zoetrope to H800."""

from __future__ import annotations

import concurrent.futures
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import subprocess
import tempfile
import time
from collections import Counter


ROOT = pathlib.Path("/mnt/umm/users/yinbaiqiao/probe_spatial")
ART = ROOT / "artifacts/main_v2_probe_v2_816_20260917"
OUT = ART / "minimax_lingbot_h800_migration_20261012_r1"
LEDGER = OUT / "migration.json"
SCO = "/mnt/umm/users/yinbaiqiao/.sco/bin/sco"
IMAGE = "registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617"
MOUNT = "019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm"
QUEUED = {"CREATING", "STARTING", "QUEUEING", "PENDING"}
STOPPED = {"SUSPENDED", "STOPPED", "CANCELLED"}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load(path):
    return json.loads(pathlib.Path(path).read_text())


def save(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        pathlib.Path(temporary).unlink(missing_ok=True)


def sco(*args):
    proc = subprocess.run(
        [SCO, "acp", "jobs", *args],
        text=True,
        capture_output=True,
        timeout=180,
    )
    if proc.returncode:
        raise RuntimeError(f"sco {' '.join(args[:2])} failed rc={proc.returncode}\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout + proc.stderr


def describe(job_id):
    return json.loads(sco("describe", "--workspace-name=aigc", "-o", "json", job_id))


def describe_batch(job_ids):
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        jobs = list(executor.map(describe, job_ids))
    return {job["name"]: job for job in jobs}


def wait_stopped(job_ids, timeout_seconds=180):
    deadline = time.monotonic() + timeout_seconds
    while True:
        exact = describe_batch(job_ids)
        started = [job for job in exact.values() if job.get("start_time")]
        if started:
            raise RuntimeError(f"start race while waiting for stop: {[job['name'] for job in started]}")
        pending = [job for job in exact.values() if job.get("state") not in STOPPED]
        if not pending:
            return exact
        if time.monotonic() >= deadline:
            states = {job["name"]: job.get("state") for job in pending}
            raise RuntimeError(f"timed out waiting for stopped state: {states}")
        time.sleep(5)


def replicas(job):
    return [
        spec.get("replicas")
        for role in job.get("roles") or []
        for spec in role.get("resource_spec") or []
    ]


def never_started(job):
    values = replicas(job)
    return (
        job.get("state") in QUEUED
        and not job.get("start_time")
        and bool(values)
        and all(value == 0 for value in values)
    )


def create(name, command):
    output = sco(
        "create",
        "--workspace-name=aigc",
        "--aec2-name=h800",
        f"--job-name={name}",
        "--priority=HIGHEST",
        "--quota-type=reserved",
        f"--container-image-url={IMAGE}",
        f"--storage-mount={MOUNT}",
        "--training-framework=pytorch",
        "--worker-nodes=1",
        "--worker-spec=N4lS.Iq.I80.1",
        f"--command={command}",
    )
    ids = list(dict.fromkeys(re.findall(r"pt-[a-z0-9]+", output)))
    if len(ids) != 1:
        raise RuntimeError(f"ambiguous create output for {name}: {output}")
    return ids[0], output


def source_records():
    specs = [
        (
            "minimax_non_l50",
            ART / "minimax_all_layers_20261011_r1/submissions.json",
            lambda rec: int(rec["layer"]) != 50,
        ),
        (
            "minimax_l50_disk49_fix",
            ART / "minimax_all_layers_l50_disk49_fix_20261011_r2/submissions.json",
            lambda rec: True,
        ),
        (
            "lingbot_c40",
            ART / "lingbot40_c_zoetrope_recovery_20261011_r1/submissions.json",
            lambda rec: True,
        ),
    ]
    result = {}
    for campaign, path, keep in specs:
        for key, rec in load(path)["jobs"].items():
            if keep(rec):
                result[key] = {"campaign": campaign, "unique_key": key, **rec}
    return result


def initialize():
    if LEDGER.exists():
        return load(LEDGER)
    sources = source_records()
    exact = describe_batch([record["job_id"] for record in sources.values()])
    selected = {}
    protected = {}
    for key, record in sources.items():
        job = exact[record["job_id"]]
        snapshot = {
            "state": job.get("state"),
            "start_time": job.get("start_time"),
            "replicas": replicas(job),
            "resource_pool": (job.get("resource_pool") or {}).get("name"),
        }
        if snapshot["resource_pool"] == "zoetrope" and never_started(job):
            digest = hashlib.sha256((key + record["job_id"]).encode()).hexdigest()[:16]
            selected[key] = {
                "campaign": record["campaign"],
                "unique_key": key,
                "old_job_id": record["job_id"],
                "old_job_name": record["job_name"],
                "new_job_name": f"bq-h8-mig-{digest}",
                "command": record["command"],
                "selected_snapshot": snapshot,
                "state": "SELECTED",
            }
        else:
            protected[key] = {
                "campaign": record["campaign"],
                "job_id": record["job_id"],
                "snapshot": snapshot,
                "reason": "already_started_completed_or_not_zoetrope",
            }
    ledger = {
        "schema": "probe_spatial.queued_pool_migration.v1",
        "created_utc": now(),
        "source_pool": "zoetrope",
        "target_pool": "h800",
        "selected_count": len(selected),
        "jobs": selected,
        "protected": protected,
    }
    save(LEDGER, ledger)
    return ledger


def migrate():
    ledger = initialize()
    records = ledger["jobs"]
    items = sorted(records.items())
    for offset in range(0, len(items), 16):
        batch = items[offset : offset + 16]
        active = [(key, rec) for key, rec in batch if rec["state"] not in {"SUBMITTED", "PROTECTED_STARTED"}]
        if not active:
            continue
        fresh = describe_batch([rec["old_job_id"] for _, rec in active])
        to_stop = []
        for key, rec in active:
            old = fresh[rec["old_job_id"]]
            if rec["state"] in {"STOP_INTENT", "STOPPED_CONFIRMED", "SUBMIT_INTENT"} and old.get("state") in STOPPED:
                continue
            if rec["state"] == "STOP_INTENT" and old.get("state") == "SUSPENDING" and not old.get("start_time"):
                continue
            if not never_started(old) or (old.get("resource_pool") or {}).get("name") != "zoetrope":
                rec["state"] = "PROTECTED_STARTED"
                rec["protected_utc"] = now()
                rec["protected_snapshot"] = {
                    "state": old.get("state"),
                    "start_time": old.get("start_time"),
                    "replicas": replicas(old),
                    "resource_pool": (old.get("resource_pool") or {}).get("name"),
                }
                continue
            rec["state"] = "STOP_INTENT"
            rec["stop_intent_utc"] = now()
            to_stop.append(rec["old_job_id"])
        save(LEDGER, ledger)
        if to_stop:
            output = sco("stop", "--workspace-name=aigc", *to_stop)
            for _, rec in active:
                if rec["old_job_id"] in to_stop:
                    rec["stop_output"] = output
            save(LEDGER, ledger)
        stopping = [(key, rec) for key, rec in active if rec["state"] in {"STOP_INTENT", "STOPPED_CONFIRMED", "SUBMIT_INTENT"}]
        after = wait_stopped([rec["old_job_id"] for _, rec in stopping]) if stopping else {}
        for key, rec in stopping:
            old = after[rec["old_job_id"]]
            if old.get("state") not in STOPPED:
                raise RuntimeError(f"old job not confirmed stopped: {rec['old_job_id']} state={old.get('state')}")
            if old.get("start_time"):
                rec["state"] = "STOPPED_AFTER_START_REVIEW"
                rec["start_race_snapshot"] = old
                save(LEDGER, ledger)
                raise RuntimeError(f"start race detected: {rec['old_job_id']}")
            rec["state"] = "STOPPED_CONFIRMED"
            rec["stopped_confirmed_utc"] = now()
            save(LEDGER, ledger)
            rec["state"] = "SUBMIT_INTENT"
            rec["submit_intent_utc"] = now()
            save(LEDGER, ledger)
            new_job_id, output = create(rec["new_job_name"], rec["command"])
            rec["new_job_id"] = new_job_id
            rec["create_output"] = output
            rec["state"] = "SUBMITTED"
            rec["submitted_utc"] = now()
            save(LEDGER, ledger)
        counts = Counter(rec["state"] for rec in records.values())
        print(json.dumps({"processed": min(offset + 16, len(items)), "total": len(items), "states": dict(counts)}, sort_keys=True), flush=True)

    replacements = [rec for rec in records.values() if rec["state"] == "SUBMITTED"]
    exact_new = describe_batch([rec["new_job_id"] for rec in replacements]) if replacements else {}
    post = []
    for rec in replacements:
        job = exact_new[rec["new_job_id"]]
        post.append(
            {
                "unique_key": rec["unique_key"],
                "old_job_id": rec["old_job_id"],
                "new_job_id": rec["new_job_id"],
                "state": job.get("state"),
                "start_time": job.get("start_time"),
                "replicas": replicas(job),
                "resource_pool": (job.get("resource_pool") or {}).get("name"),
            }
        )
    ledger["completed_utc"] = now()
    ledger["state_counts"] = dict(Counter(rec["state"] for rec in records.values()))
    ledger["replacement_scheduler_counts"] = dict(Counter(row["state"] for row in post))
    ledger["replacement_resource_pools"] = dict(Counter(row["resource_pool"] for row in post))
    save(LEDGER, ledger)
    save(OUT / "post_migration_audit.json", {"observed_utc": now(), "jobs": post})
    print(
        json.dumps(
            {
                "ledger": str(LEDGER),
                "state_counts": ledger["state_counts"],
                "replacement_scheduler_counts": ledger["replacement_scheduler_counts"],
                "replacement_resource_pools": ledger["replacement_resource_pools"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    migrate()
