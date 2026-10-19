#!/usr/bin/env python3
"""Submit validated VideoMAE-72 and VJEPA2-96 production probing jobs."""

from __future__ import annotations

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


ROOT = Path("/mnt/umm/users/yinbaiqiao/probe_spatial")
BASE = ROOT / "artifacts/main_v2_probe_v2_816_20260917"
OUT = BASE / "single_gpu_vm72_vj96_20260924_r1"
PLAN = OUT / "plan.json"
VALIDATION = BASE / "probe_validation_vm72_vj96_20260924_r1"
RUNNER = ROOT / "runtime/main_v2_probe_v2_vm72_vj96_r1/run_single.py"
RUN_ID = "main_v2_probe_v2_vm72_vj96_20260924_r1"
PYTHON = "/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python"
SCO = "/mnt/umm/users/yinbaiqiao/.sco/bin/sco"
JOB_RE = re.compile(r"job id\s*:\s*(pt-[a-z0-9]+)", re.I)
LOCK = threading.Lock()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def validation_gate() -> dict:
    bundle = json.loads((VALIDATION / "bundle.json").read_text())
    if bundle["cases_count"] != 168 or bundle["shard_count"] != 28:
        raise RuntimeError("unexpected validation bundle scope")
    inventories = []
    for shard in range(28):
        path = VALIDATION / "groups" / f"{shard:02d}" / "inventory_result.json"
        if not path.exists():
            raise RuntimeError(f"inventory shard {shard} is not complete")
        inventories.append(json.loads(path.read_text()))
    if any(row.get("status") != "PASS" for row in inventories):
        raise RuntimeError("one or more full-owner inventories failed")
    results = []
    for entry in bundle["cases"]:
        path = Path(entry["case_path"]).parent / "result.json"
        if not path.exists():
            raise RuntimeError(f"validation is incomplete for {entry['unique_key']}")
        results.append(json.loads(path.read_text()))
    failed = [row for row in results if row.get("status") != "PASS"]
    if failed:
        raise RuntimeError(f"{len(failed)} validation cases are not PASS")
    gate = {
        "schema": "VM72_VJ96_PRODUCTION_GATE_V1",
        "created_utc": now(),
        "inventory_groups_pass": len(inventories),
        "validation_cases_pass": len(results),
        "bundle_sha256": sha256(VALIDATION / "bundle.json"),
        "runner_sha256": sha256(RUNNER),
    }
    atomic_json(OUT / "production_gate.json", gate)
    return gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--aec2-name", choices=("h800", "zoetrope"), default="h800")
    parser.add_argument("--parallel", type=int, default=8)
    args = parser.parse_args()
    validation_gate()
    plan = json.loads(PLAN.read_text())
    if plan["status"] != "FROZEN_NOT_SUBMITTED" or len(plan["jobs"]) != 168:
        raise RuntimeError("unexpected production plan")
    ledger_path = OUT / "submissions.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {
        "schema": "VM72_VJ96_PRODUCTION_SUBMISSIONS_V1",
        "run_id": RUN_ID,
        "resource_pool": args.aec2_name,
        "worker_spec": "N4lS.Iq.I80.1",
        "jobs": {},
    }
    if ledger["resource_pool"] != args.aec2_name:
        raise RuntimeError("submission ledger resource pool mismatch")
    pending = [row for row in plan["jobs"] if row["unique_key"] not in ledger["jobs"]]

    def submit(row: dict) -> tuple[str, dict]:
        command = " ".join([
            PYTHON, str(RUNNER), "--item", row["item_path"],
            "--item-sha256", row["item_sha256"], "--run-id", RUN_ID,
        ])
        model = "vm" if row["model"] == "videomae" else "vj"
        name = f"bq-{model}-prod-{row['ordinal']:03d}-{row['item_sha256'][:6]}"
        if not args.submit:
            return row["unique_key"], {**row, "job_name": name, "command": command, "dry_run": True}
        argv = [
            SCO, "acp", "jobs", "create", "--workspace-name=aigc",
            f"--aec2-name={args.aec2_name}", f"--job-name={name}", "--priority=HIGHEST",
            "--quota-type=reserved",
            "--container-image-url=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617",
            "--storage-mount=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm",
            "--training-framework=pytorch", "--worker-nodes=1",
            "--worker-spec=N4lS.Iq.I80.1", f"--command={command}",
        ]
        completed = subprocess.run(argv, text=True, capture_output=True, timeout=180)
        match = JOB_RE.search(completed.stdout + "\n" + completed.stderr)
        return row["unique_key"], {
            **row, "job_name": name, "job_id": match.group(1) if match else None,
            "command": command, "returncode": completed.returncode,
            "stdout": completed.stdout, "stderr": completed.stderr,
            "submitted_utc": now(),
            "state": "SUBMITTED" if completed.returncode == 0 and match else "SUBMISSION_FAILED",
        }

    if not args.submit:
        print(json.dumps({"ready_to_submit": len(pending), "resource_pool": args.aec2_name}))
        return 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        futures = [pool.submit(submit, row) for row in pending]
        for future in concurrent.futures.as_completed(futures):
            key, record = future.result()
            with LOCK:
                ledger["jobs"][key] = record
                ledger["updated_utc"] = now()
                atomic_json(ledger_path, ledger)
            print(f"{key}\t{record['job_id'] or 'SUBMIT_ERROR'}", flush=True)
    failures = [row for row in ledger["jobs"].values() if row.get("state") != "SUBMITTED"]
    print(json.dumps({"recorded": len(ledger["jobs"]), "failures": len(failures)}))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
