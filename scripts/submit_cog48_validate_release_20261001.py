#!/usr/bin/env python3
"""Submit the 24 frozen CogVideoX C validate-then-release shards to H800."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from probe_campaign_20260924_followup import create


ROOT = Path("/mnt/umm/users/yinbaiqiao/probe_spatial")
BASE = ROOT / "artifacts/main_v2_probe_v2_816_20260917"
VALIDATION = BASE / "probe_validation_cog48_c_20261001_r1"
PRODUCTION = BASE / "single_gpu_cog48_c_h800_20261001_r1"
PYTHON = "/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python"
VALIDATOR = ROOT / "scripts/run_probe_hybrid_validation.py"
SUBMITTER = ROOT / "scripts/probe_campaign_20260924_followup.py"
WRAPPER = VALIDATION / "validate_then_release_h800.py"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main() -> None:
    bundle = VALIDATION / "bundle.json"
    metadata = json.loads(bundle.read_text())
    if metadata["cases_count"] != 48 or metadata["shard_count"] != 24:
        raise RuntimeError("unexpected frozen CogVideoX C scope")
    production_plan = json.loads((PRODUCTION / "plan.json").read_text())
    if len(production_plan["jobs"]) != 48:
        raise RuntimeError("unexpected production scope")
    bindings = {
        "bundle_sha256": sha256(bundle),
        "validator_sha256": sha256(VALIDATOR),
        "submitter_sha256": sha256(SUBMITTER),
        "wrapper_sha256": sha256(WRAPPER),
    }
    ledger_path = VALIDATION / "validation_release_submissions.json"
    ledger = (
        json.loads(ledger_path.read_text())
        if ledger_path.exists()
        else {
            "schema": "COG48_C_VALIDATE_RELEASE_H800_V1",
            "created_utc": now(),
            "resource_pool": "h800",
            "worker_spec": "N4lS.Iq.I80.1",
            "bindings": bindings,
            "jobs": {},
        }
    )
    if ledger["bindings"] != bindings:
        raise RuntimeError("frozen validation bindings changed")
    for shard in range(metadata["shard_count"]):
        key = str(shard)
        existing = ledger["jobs"].get(key)
        if existing:
            if existing.get("job_id"):
                continue
            raise RuntimeError(f"ambiguous prior submit intent for shard {shard}")
        argv = [
            PYTHON,
            str(WRAPPER),
            str(bundle),
            str(PRODUCTION),
            str(shard),
            bindings["bundle_sha256"],
            bindings["validator_sha256"],
            bindings["submitter_sha256"],
        ]
        command = shlex.join(argv)
        name = f"bq-cogc48-vr-s{shard:02d}-1001"
        record = {
            "shard": shard,
            "job_name": name,
            "command": command,
            "state": "SUBMIT_INTENT",
            "submitted_utc": now(),
        }
        ledger["jobs"][key] = record
        save(ledger_path, ledger)
        record["job_id"] = create(name, command, "h800")
        record["state"] = "SUBMITTED"
        save(ledger_path, ledger)
        print(json.dumps({"shard": shard, "job_id": record["job_id"]}), flush=True)
    print(json.dumps({"submitted": len(ledger["jobs"]), "pool": "h800"}), flush=True)


if __name__ == "__main__":
    main()
