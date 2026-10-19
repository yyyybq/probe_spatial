#!/usr/bin/env python3
"""Submit SHA-bound r7 canary or production one-GPU jobs to Zoetrope."""

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
from typing import Any


ROOT = Path("/mnt/umm/users/yinbaiqiao/probe_spatial")
PLAN_ROOT = ROOT / "artifacts/main_v2_probe_v2_816_20260917/single_gpu_r7_20260922_r2"
PLAN = PLAN_ROOT / "plan.json"
WRAPPER = ROOT / "runtime/main_v2_probe_v2_r7_single_v3/run_single.sh"
SCO = "/mnt/umm/users/yinbaiqiao/.sco/bin/sco"
RUN_ID = "main_v2_probe_v2_r7_single_20260922_r2"
PLAN_SHA256 = "b5bb5ac7c03c6567476a2c538a6f5d314c54d089a2eb91bd186cf3e22691959f"
WRAPPER_SHA256 = "8d015a628d3d4494fc39e287f8150fb5159762b6b51af5b18eb320921b57e58d"
JOB_RE = re.compile(r"job id\s*:\s*(pt-[a-z0-9]+)", re.I)
LOCK = threading.Lock()
CANARY_SELECTORS = (
    ("cogvideox", "streaming_depth", 0, 8),
    ("videomae", "streaming_mask", 5, 8),
    ("vjepa2", "view_consistency", 5, 8),
    ("lingbot", "ego_belief_v2", 10, 8),
    ("qwen2_5_vl", "action_dynamics", -1, 8),
    ("lingbot", "path_integration", 10, 12),
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--canary", action="store_true")
    mode.add_argument("--production", action="store_true")
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--only-key", action="append", default=[])
    parser.add_argument("--retry", action="store_true")
    args = parser.parse_args()
    if PLAN_SHA256 and sha256(PLAN) != PLAN_SHA256:
        raise RuntimeError("plan SHA mismatch")
    if sha256(WRAPPER) != WRAPPER_SHA256:
        raise RuntimeError("wrapper SHA mismatch")
    plan = json.loads(PLAN.read_text())
    if plan["run_id"] != RUN_ID or plan["status"] != "FROZEN_NOT_SUBMITTED":
        raise RuntimeError("unexpected frozen plan identity")
    eligible = [row for row in plan["jobs"] if row["submission_gate"] == "ELIGIBLE"]
    if len(eligible) != 464:
        raise RuntimeError("eligible count mismatch")
    if args.preflight:
        by_selector = {
            (row["model"], row["probe"], row["layer"], row["prefix"]): row
            for row in eligible
        }
        pending = [by_selector[CANARY_SELECTORS[0]]]
        ledger_path = PLAN_ROOT / "preflight_submissions.json"
        canary_flag = " --validate-only"
        name_prefix = "bq-v2r7v"
    elif args.canary:
        by_selector = {
            (row["model"], row["probe"], row["layer"], row["prefix"]): row
            for row in eligible
        }
        pending = [by_selector[selector] for selector in CANARY_SELECTORS]
        ledger_path = PLAN_ROOT / "canary_submissions.json"
        canary_flag = " --canary"
        name_prefix = "bq-v2r7c"
    else:
        pending = eligible
        ledger_path = PLAN_ROOT / "production_submissions.json"
        canary_flag = ""
        name_prefix = "bq-v2r7p"
    requested = set(args.only_key)
    known = {row["unique_key"] for row in pending}
    if requested - known:
        raise RuntimeError(f"requested keys are not in this mode: {sorted(requested - known)}")
    if requested:
        pending = [row for row in pending if row["unique_key"] in requested]
    ledger = {
        "schema": "MAIN_V2_PROBE528_R7_SUBMISSIONS_V1",
        "mode": "preflight" if args.preflight else ("canary" if args.canary else "production"),
        "run_id": RUN_ID,
        "resource_pool": "zoetrope",
        "worker_spec": "N4lS.Iq.I80.1",
        "priority": "HIGHEST",
        "created_utc": now(),
        "jobs": {},
    }
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
    if not args.retry:
        pending = [row for row in pending if row["unique_key"] not in ledger["jobs"]]
    if args.limit is not None:
        pending = pending[:args.limit]

    def submit(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        name = f"{name_prefix}-{row['ordinal']:03d}-{row['item_sha256'][:6]}"
        command = (
            f"cd {ROOT} && bash {WRAPPER} {row['item_path']} {row['item_sha256']} "
            f"{RUN_ID}{canary_flag}"
        )
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
        output = completed.stdout + "\n" + completed.stderr
        match = JOB_RE.search(output)
        return row["unique_key"], {
            **row, "job_name": name, "job_id": match.group(1) if match else None,
            "returncode": completed.returncode, "stdout": completed.stdout,
            "stderr": completed.stderr, "startup_command": command,
            "submitted_utc": now(),
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        futures = [pool.submit(submit, row) for row in pending]
        for future in concurrent.futures.as_completed(futures):
            key, value = future.result()
            with LOCK:
                previous = ledger["jobs"].get(key)
                if previous is not None:
                    history = list(previous.get("prior_attempts", []))
                    history.append({k: v for k, v in previous.items() if k != "prior_attempts"})
                    value["prior_attempts"] = history
                ledger["jobs"][key] = value
                ledger["updated_utc"] = now()
                atomic_json(ledger_path, ledger)
            print(f"{key}\t{value['job_id'] or 'SUBMIT_ERROR'}", flush=True)
    failures = [row for row in ledger["jobs"].values() if row["returncode"] or not row["job_id"]]
    print(json.dumps({"recorded": len(ledger["jobs"]), "failures": len(failures)}))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
