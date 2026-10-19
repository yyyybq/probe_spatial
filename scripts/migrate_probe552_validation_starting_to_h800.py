#!/usr/bin/env python3
"""Move the frozen, never-started probe552 validators from Zoetrope to H800."""
import json
import os
import re
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/mnt/umm/users/yinbaiqiao/probe_spatial")
VALIDATION = ROOT / "artifacts/main_v2_probe_v2_816_20260917/probe552_validation_stream_ab128_20260923_r1"
SNAPSHOT = VALIDATION / "pre_h800_migration_snapshot.json"
SUBMISSIONS = VALIDATION / "submissions.json"
OUT = VALIDATION / "zoetrope_to_h800_migration"
SCO = "/mnt/umm/users/yinbaiqiao/.sco/bin/sco"
CANCELLABLE = {"STARTING", "QUEUEING", "PENDING", "CREATING", "SCHEDULING", "WAITING"}
TERMINAL = {"SUSPENDED", "CANCELED", "CANCELLED", "STOPPED", "FAILED", "SUCCEEDED", "DELETED"}
JOB_RE = re.compile(r"job id\s*:\s*(pt-[a-z0-9]+)", re.I)


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.fchmod(fd, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def run(argv, check=True):
    completed = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
    if check and completed.returncode:
        raise RuntimeError(f"command failed rc={completed.returncode}: {argv}\n{completed.stdout}")
    return completed


def describe(job_id):
    return json.loads(run([SCO, "acp", "jobs", "describe", "--workspace-name=aigc", "-o", "json", job_id]).stdout)


def journal(record):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "journal.jsonl").open("a") as handle:
        handle.write(json.dumps(dict(at=now(), **record), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def submit_h800(old):
    role = old["roles"][0]
    spec = role["resource_spec"][0]
    mounts = ",".join(f"{mount['id']}:{mount['mount_path']}" for mount in old["mount"])
    display_name = old["display_name"] + "-h8"
    argv = [
        SCO, "acp", "jobs", "create", "--workspace-name=aigc", "--aec2-name=h800",
        f"--job-name={display_name}", f"--priority={old['scheduling']['priority']}",
        f"--quota-type={old['scheduling']['quota_type'].lower()}",
        f"--container-image-url={role['image_path']}", f"--storage-mount={mounts}",
        f"--training-framework={old['framework'].lower()}",
        f"--worker-nodes={role['total_replicas']}", f"--worker-spec={spec['name']}",
        f"--command={role['startup_script']}",
    ]
    completed = run(argv)
    match = JOB_RE.search(completed.stdout)
    if not match:
        raise RuntimeError(f"could not parse H800 job ID: {completed.stdout}")
    return dict(job_id=match.group(1), display_name=display_name, argv=argv, response=completed.stdout.strip())


def main():
    if (OUT / "result.json").exists():
        raise RuntimeError("migration already completed")
    snapshot = json.loads(SNAPSHOT.read_text())
    submissions = json.loads(SUBMISSIONS.read_text())
    shard_by_id = {row["job_id"]: shard for shard, row in submissions["jobs"].items()}
    selected = []
    for job in snapshot["jobs"]:
        if job["name"] not in shard_by_id:
            continue
        if job.get("resource_pool", {}).get("name") != "zoetrope":
            raise RuntimeError(f"unexpected pool for {job['name']}")
        if job["state"] != "STARTING" or job.get("start_time"):
            raise RuntimeError(f"snapshot includes a started/non-STARTING job: {job['name']}")
        selected.append(job)
    if len(selected) != 32:
        raise RuntimeError(f"expected 32 frozen jobs, got {len(selected)}")
    OUT.mkdir(parents=True, exist_ok=True)
    atomic(OUT / "inventory.json", {"created_utc": now(), "jobs": selected})

    stopped, protected = [], []
    for old in selected:
        fresh = describe(old["name"])
        journal({"event": "pre_stop", "job_id": old["name"], "state": fresh["state"],
                 "start_time": fresh.get("start_time")})
        if fresh["state"] not in CANCELLABLE or fresh.get("start_time"):
            protected.append(dict(job_id=old["name"], state=fresh["state"], start_time=fresh.get("start_time")))
            journal({"event": "protected", **protected[-1]})
            continue
        response = run([SCO, "acp", "jobs", "stop", "--workspace-name=aigc", old["name"]])
        stopped.append(old)
        journal({"event": "stop_requested", "job_id": old["name"], "output": response.stdout.strip()})

    deadline = time.monotonic() + 900
    pending = {job["name"]: job for job in stopped}
    while pending and time.monotonic() < deadline:
        for job_id in list(pending):
            fresh = describe(job_id)
            if fresh["state"] in TERMINAL:
                journal({"event": "terminal_confirmed", "job_id": job_id, "state": fresh["state"]})
                pending.pop(job_id)
            elif fresh["state"] == "RUNNING" or fresh.get("start_time"):
                raise RuntimeError(f"{job_id} started during cancellation")
        if pending:
            time.sleep(5)
    if pending:
        raise RuntimeError(f"timed out waiting for cancellation: {sorted(pending)}")

    replacements = []
    for old in stopped:
        replacement = submit_h800(old)
        shard = shard_by_id[old["name"]]
        record = submissions["jobs"][shard]
        prior = list(record.get("prior_attempts", []))
        prior.append(dict(job_id=old["name"], display_name=old["display_name"], pool="zoetrope",
                          state="CANCELLED_FOR_H800_MIGRATION", stopped_utc=now()))
        record.update(job_id=replacement["job_id"], name=replacement["display_name"], state="SUBMITTED",
                      submitted_utc=now(), pool="h800", prior_attempts=prior,
                      migration=dict(from_job_id=old["name"], from_pool="zoetrope", to_pool="h800"))
        submissions["updated_utc"] = now()
        atomic(SUBMISSIONS, submissions)
        migrated = dict(shard=int(shard), old_job=old["name"], old_display_name=old["display_name"],
                        new_job=replacement["job_id"], new_display_name=replacement["display_name"],
                        pool="h800", command=old["roles"][0]["startup_script"])
        replacements.append(migrated)
        journal({"event": "replacement_submitted", **migrated})

    result = dict(completed_utc=now(), selected=len(selected), stopped=len(stopped),
                  protected=protected, replacements=replacements)
    atomic(OUT / "result.json", result)
    print(json.dumps(dict(selected=len(selected), stopped=len(stopped), protected=len(protected),
                          replacements=len(replacements)), ensure_ascii=False))


if __name__ == "__main__":
    main()
