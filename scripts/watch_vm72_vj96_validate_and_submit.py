#!/usr/bin/env python3
"""Wait for all VM/VJ validations, then submit the frozen production plan once."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/mnt/umm/users/yinbaiqiao/probe_spatial")
BASE = ROOT / "artifacts/main_v2_probe_v2_816_20260917"
VALIDATION = BASE / "probe_validation_vm72_vj96_20260924_r1"
PRODUCTION = BASE / "single_gpu_vm72_vj96_20260924_r1"
STATUS = PRODUCTION / "validation_and_submission_watcher.json"
PYTHON = "/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python"
SUBMITTER = ROOT / "scripts/submit_vm72_vj96_production.py"


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
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def readable_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except (PermissionError, json.JSONDecodeError):
        return None


def main() -> int:
    bundle = json.loads((VALIDATION / "bundle.json").read_text())
    if bundle["cases_count"] != 168 or bundle["shard_count"] != 28:
        raise RuntimeError("unexpected validation scope")
    while True:
        inventories = [
            readable_json(VALIDATION / "groups" / f"{shard:02d}" / "inventory_result.json")
            for shard in range(28)
        ]
        results = [readable_json(Path(entry["case_path"]).parent / "result.json") for entry in bundle["cases"]]
        inventory_states = Counter(row.get("status") for row in inventories if row)
        case_states = Counter(row.get("status") for row in results if row)
        state = {
            "schema": "VM72_VJ96_VALIDATE_AND_SUBMIT_WATCHER_V1",
            "updated_utc": now(),
            "state": "WAITING_VALIDATION",
            "inventory_receipts": sum(row is not None for row in inventories),
            "inventory_states": dict(inventory_states),
            "case_receipts": sum(row is not None for row in results),
            "case_states": dict(case_states),
            "production_pool": "h800",
        }
        if inventory_states.get("FAIL") or case_states.get("FAIL"):
            state["state"] = "VALIDATION_FAILED_NO_PRODUCTION_SUBMISSION"
            atomic_json(STATUS, state)
            return 2
        if inventory_states.get("PASS") == 28 and case_states.get("PASS") == 168:
            state["state"] = "VALIDATION_PASSED_SUBMITTING"
            atomic_json(STATUS, state)
            completed = subprocess.run(
                [PYTHON, str(SUBMITTER), "--submit", "--aec2-name", "h800", "--parallel", "8"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=7200,
            )
            state.update(
                updated_utc=now(),
                state="PRODUCTION_SUBMITTED" if completed.returncode == 0 else "PRODUCTION_SUBMISSION_FAILED",
                submit_returncode=completed.returncode,
                submit_stdout=completed.stdout,
                submit_stderr=completed.stderr,
            )
            atomic_json(STATUS, state)
            return completed.returncode
        atomic_json(STATUS, state)
        time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
