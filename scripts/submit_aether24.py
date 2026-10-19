#!/usr/bin/env python3
"""Submit the 24 validated Aether probes as isolated one-GPU SCO jobs."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/main_v2_probe_v2_816_20260917"
PLAN = BASE / "single_gpu_aether24_20260923_r1/plan.json"
OUT = BASE / "single_gpu_aether24_20260923_r1"
RUNNER = ROOT / "runtime/main_v2_probe_v2_aether24_r1/run_single.py"
RUN_ID = "main_v2_probe_v2_aether24_20260923_r1"
PYTHON = "/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python"
SCO = "/mnt/umm/users/yinbaiqiao/.sco/bin/sco"
JOB_RE = re.compile(r"job id\s*:\s*(pt-[a-z0-9]+)", re.I)
LOCK = threading.Lock()


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.fchmod(fd, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    rows = [
        row for row in plan["jobs"]
        if row["model"] == "aether"
    ]
    if len(rows) != 24:
        raise RuntimeError(f"expected 24 Aether rows, got {len(rows)}")
    runner_sha = sha(RUNNER)
    ledger_path = OUT / "submissions.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {
        "schema": "AETHER24_PRODUCTION_SUBMISSIONS_V1",
        "run_id": RUN_ID,
        "overlay": str(BASE / "probe552_validation_aether_target8_20260923_r4/overlay/manifest.json"),
        "resource_pool": "zoetrope",
        "worker_spec": "N4lS.Iq.I80.1",
        "jobs": {},
    }
    pending = [row for row in rows if row["unique_key"] not in ledger["jobs"]]

    def submit(row):
        command = " ".join([
            PYTHON, str(RUNNER), "--item", row["item_path"],
            "--item-sha256", row["item_sha256"], "--run-id", RUN_ID,
        ])
        name = f"bq-aeth24-prod-{row['ordinal']:03d}-{row['item_sha256'][:6]}"
        if not args.submit:
            return row["unique_key"], {**row, "job_name": name, "startup_command": command, "dry_run": True}
        argv = [
            SCO, "acp", "jobs", "create", "--workspace-name=aigc",
            "--aec2-name=zoetrope", f"--job-name={name}", "--priority=HIGHEST",
            "--quota-type=reserved",
            "--container-image-url=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617",
            "--storage-mount=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm",
            "--training-framework=pytorch", "--worker-nodes=1",
            "--worker-spec=N4lS.Iq.I80.1", f"--command={command}",
        ]
        completed = subprocess.run(argv, text=True, capture_output=True, timeout=120)
        match = JOB_RE.search(completed.stdout + "\n" + completed.stderr)
        return row["unique_key"], {
            **row, "job_name": name, "job_id": match.group(1) if match else None,
            "startup_command": command, "runner_sha256": runner_sha,
            "returncode": completed.returncode, "stdout": completed.stdout,
            "stderr": completed.stderr, "submitted_utc": now(),
        }

    if not args.submit:
        for row in pending:
            print(json.dumps(submit(row)[1], sort_keys=True))
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(submit, row) for row in pending]
        for future in concurrent.futures.as_completed(futures):
            key, record = future.result()
            with LOCK:
                ledger["jobs"][key] = record
                ledger["updated_utc"] = now()
                atomic(ledger_path, ledger)
            print(f"{key}\t{record['job_id'] or 'SUBMIT_ERROR'}", flush=True)
    failures = [row for row in ledger["jobs"].values() if row["returncode"] or not row["job_id"]]
    print(json.dumps({"recorded": len(ledger["jobs"]), "failures": len(failures)}))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
