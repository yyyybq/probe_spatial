#!/usr/bin/env python3
"""Move never-started Wan probing jobs from H800 to Zoetrope exactly once."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import tempfile
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/mnt/umm/users/yinbaiqiao/probe_spatial")
OUT = ROOT / "artifacts/main_v2_probe_v2_816_20260917/single_gpu_wan_ab112_20260923_r1"
SOURCE_LEDGER = OUT / "submissions.json"
MIGRATION_LEDGER = OUT / "migration_h800_starting_to_zoetrope_20260924.json"
SCO = "/mnt/umm/users/yinbaiqiao/.sco/bin/sco"
WORKSPACE = "aigc"
JOB_RE = re.compile(r"job id\s*:\s*(pt-[a-z0-9]+)", re.I)
LOCK = threading.Lock()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.fchmod(fd, 0o644)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def scheduler_jobs() -> list[dict]:
    completed = subprocess.run(
        [SCO, "acp", "jobs", "list", f"--workspace-name={WORKSPACE}", "--page-size=500", "-o", "json"],
        text=True,
        capture_output=True,
        check=True,
        timeout=180,
    )
    return json.loads(completed.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--parallel", type=int, default=8)
    args = parser.parse_args()

    source = json.loads(SOURCE_LEDGER.read_text())
    source_by_id = {record["job_id"]: (key, record) for key, record in source["jobs"].items()}
    live = scheduler_jobs()
    live_by_id = {job["name"]: job for job in live}

    eligible = []
    for old_id, (key, record) in source_by_id.items():
        job = live_by_id.get(old_id)
        if not job:
            continue
        if (
            job.get("state") == "STARTING"
            and not job.get("start_time")
            and (job.get("resource_pool") or {}).get("name") == "h800"
        ):
            eligible.append((key, record, job))
    eligible.sort(key=lambda row: row[1]["ordinal"])

    states = Counter(
        live_by_id[job_id]["state"]
        for job_id in source_by_id
        if job_id in live_by_id
    )
    print(json.dumps({"source_jobs": len(source_by_id), "states": states, "eligible": len(eligible)}))
    if not args.execute:
        for key, record, job in eligible:
            print(json.dumps({"unique_key": key, "old_job_id": job["name"], "ordinal": record["ordinal"]}))
        return 0

    if MIGRATION_LEDGER.exists():
        migration = json.loads(MIGRATION_LEDGER.read_text())
        frozen_ids = {record["old_job_id"] for record in migration["jobs"].values()}
        current_ids = {job["name"] for _, _, job in eligible}
        if frozen_ids != current_ids:
            raise RuntimeError("existing migration scope differs from current never-started H800 scope")
    else:
        migration = {
            "schema": "WAN_AB112_H800_TO_ZOETROPE_MIGRATION_V1",
            "created_utc": now(),
            "source_ledger": str(SOURCE_LEDGER),
            "selection_rule": "source ledger job is H800 STARTING with no start_time in fresh scheduler snapshot",
            "resource_pool": "zoetrope",
            "jobs": {
                key: {
                    "unique_key": key,
                    "ordinal": record["ordinal"],
                    "item_path": record["item_path"],
                    "item_sha256": record["item_sha256"],
                    "command": record["command"],
                    "old_job_id": job["name"],
                    "old_job_name": job["display_name"],
                    "old_state": job["state"],
                    "old_start_time": job.get("start_time"),
                    "migration_state": "SELECTED",
                }
                for key, record, job in eligible
            },
        }
        atomic_json(MIGRATION_LEDGER, migration)

    pending_stop = [record for record in migration["jobs"].values() if record["migration_state"] == "SELECTED"]
    for offset in range(0, len(pending_stop), 20):
        batch = pending_stop[offset : offset + 20]
        argv = [SCO, "acp", "jobs", "stop", f"--workspace-name={WORKSPACE}"] + [row["old_job_id"] for row in batch]
        completed = subprocess.run(argv, text=True, capture_output=True, timeout=180)
        if completed.returncode:
            raise RuntimeError(f"failed to stop H800 batch: {completed.stdout}\n{completed.stderr}")
        stopped_utc = now()
        for row in batch:
            row["migration_state"] = "H800_STOP_REQUESTED"
            row["stop_requested_utc"] = stopped_utc
            row["stop_stdout"] = completed.stdout
            row["stop_stderr"] = completed.stderr
        atomic_json(MIGRATION_LEDGER, migration)

    def submit(row: dict) -> tuple[str, dict]:
        name = f"bq-wan112-zp{row['ordinal']:03d}-{row['item_sha256'][:6]}"
        argv = [
            SCO, "acp", "jobs", "create", f"--workspace-name={WORKSPACE}",
            "--aec2-name=zoetrope", f"--job-name={name}", "--priority=HIGHEST",
            "--quota-type=reserved",
            "--container-image-url=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617",
            "--storage-mount=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm",
            "--training-framework=pytorch", "--worker-nodes=1",
            "--worker-spec=N4lS.Iq.I80.1", f"--command={row['command']}",
        ]
        completed = subprocess.run(argv, text=True, capture_output=True, timeout=180)
        match = JOB_RE.search(completed.stdout + "\n" + completed.stderr)
        result = dict(row)
        result.update(
            new_job_name=name,
            new_job_id=match.group(1) if match else None,
            submit_returncode=completed.returncode,
            submit_stdout=completed.stdout,
            submit_stderr=completed.stderr,
            submitted_utc=now(),
            migration_state="ZOETROPE_SUBMITTED" if completed.returncode == 0 and match else "ZOETROPE_SUBMISSION_FAILED",
        )
        return row["unique_key"], result

    pending_submit = [
        row for row in migration["jobs"].values()
        if row["migration_state"] in {"H800_STOP_REQUESTED", "ZOETROPE_SUBMISSION_FAILED"}
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        futures = [pool.submit(submit, row) for row in pending_submit]
        for future in concurrent.futures.as_completed(futures):
            key, result = future.result()
            with LOCK:
                migration["jobs"][key] = result
                migration["updated_utc"] = now()
                atomic_json(MIGRATION_LEDGER, migration)
            print(json.dumps({"unique_key": key, "old_job_id": result["old_job_id"], "new_job_id": result["new_job_id"], "state": result["migration_state"]}), flush=True)

    failures = [row for row in migration["jobs"].values() if row["migration_state"] != "ZOETROPE_SUBMITTED"]
    print(json.dumps({"selected": len(migration["jobs"]), "submitted": len(migration["jobs"]) - len(failures), "failures": len(failures)}))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
