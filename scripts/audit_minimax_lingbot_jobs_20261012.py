#!/usr/bin/env python3
"""Describe the exact active MiniMax/LingBot campaign jobs and save a snapshot."""

from __future__ import annotations

import concurrent.futures
import datetime as dt
import json
import pathlib
import subprocess
from collections import Counter


ROOT = pathlib.Path("/mnt/umm/users/yinbaiqiao/probe_spatial")
ART = ROOT / "artifacts/main_v2_probe_v2_816_20260917"
SCO = "/mnt/umm/users/yinbaiqiao/.sco/bin/sco"
OUT = ART / "minimax_lingbot_h800_migration_20261012_r1"


def load_jobs(path: pathlib.Path, campaign: str, keep=None):
    data = json.loads(path.read_text())
    result = []
    for unique_key, rec in data["jobs"].items():
        if keep is not None and not keep(rec):
            continue
        result.append({"campaign": campaign, "unique_key": unique_key, **rec})
    return result


def load_minimax_l30_accepted(path: pathlib.Path):
    data = json.loads(path.read_text())
    result = []
    for unique_key, rec in data["jobs"].items():
        job_ids = rec.get("job_ids") or []
        if len(job_ids) != 1:
            raise RuntimeError(f"expected one accepted L30 job id for {unique_key}, got {job_ids}")
        result.append(
            {
                "campaign": "minimax_l30_accepted",
                "unique_key": unique_key,
                **rec,
                "job_id": job_ids[0],
                "layer": 30,
            }
        )
    return result


def describe(rec):
    cmd = [
        SCO,
        "acp",
        "jobs",
        "describe",
        "--workspace-name=aigc",
        "-o",
        "json",
        rec["job_id"],
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode:
        return {
            **rec,
            "describe_error": proc.stderr.strip() or proc.stdout.strip(),
            "describe_returncode": proc.returncode,
        }
    job = json.loads(proc.stdout)
    replicas = []
    for role in job.get("roles") or []:
        for spec in role.get("resource_spec") or []:
            replicas.append(spec.get("replicas"))
    return {
        **rec,
        "scheduler": job,
        "scheduler_state": job.get("state"),
        "scheduler_start_time": job.get("start_time"),
        "scheduler_complete_time": job.get("complete_time"),
        "scheduler_resource_pool": (job.get("resource_pool") or {}).get("name"),
        "scheduler_replicas": replicas,
    }


def main():
    records = []
    records += load_minimax_l30_accepted(
        ART / "minimax_l30_ab16_20260930_r2/formal_submission_ledger.json"
    )
    records += load_jobs(
        ART / "minimax_all_layers_20261011_r1/submissions.json",
        "minimax_non_l50",
        keep=lambda r: int(r["layer"]) != 50,
    )
    records += load_jobs(
        ART / "minimax_all_layers_l50_disk49_fix_20261011_r2/submissions.json",
        "minimax_l50_disk49_fix",
    )
    records += load_jobs(
        ART / "lingbot40_c_zoetrope_recovery_20261011_r1/submissions.json",
        "lingbot_c40",
    )
    migration_path = ART / "minimax_lingbot_h800_migration_20261012_r1/migration.json"
    migration = json.loads(migration_path.read_text()) if migration_path.exists() else None
    if migration:
        replacements = migration.get("jobs", {})
        for record in records:
            replacement = replacements.get(record["unique_key"])
            if replacement and replacement.get("state") == "SUBMITTED":
                record["superseded_job_id"] = record["job_id"]
                record["job_id"] = replacement["new_job_id"]
                record["job_name"] = replacement["new_job_name"]
                record["migration_ledger"] = str(migration_path)
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        exact = list(executor.map(describe, records))
    observed = dt.datetime.now(dt.timezone.utc).isoformat()
    summary = {}
    for campaign in sorted({r["campaign"] for r in exact}):
        selected = [r for r in exact if r["campaign"] == campaign]
        summary[campaign] = {
            "count": len(selected),
            "states": dict(Counter(r.get("scheduler_state", "DESCRIBE_ERROR") for r in selected)),
            "resource_pools": dict(Counter(r.get("scheduler_resource_pool", "UNKNOWN") for r in selected)),
            "never_started_zero_replica": sum(
                r.get("scheduler_state") in {"CREATING", "STARTING", "QUEUEING", "PENDING"}
                and not r.get("scheduler_start_time")
                and r.get("scheduler_replicas")
                and all(v == 0 for v in r["scheduler_replicas"])
                for r in selected
            ),
        }
    payload = {
        "schema": "probe_spatial.exact_job_audit.v1",
        "observed_utc": observed,
        "count": len(exact),
        "summary": summary,
        "jobs": exact,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    filename = "effective_post_migration_audit.json" if migration else "pre_migration_audit.json"
    (OUT / filename).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"observed_utc": observed, "count": len(exact), "summary": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
