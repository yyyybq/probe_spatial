#!/usr/bin/env python3
"""Recover the five retry jobs whose newest checkpoint was truncated."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/mnt/umm/users/yinbaiqiao/probe_spatial")
BASE = ROOT / "artifacts/main_v2_probe_v2_816_20260917/retry_probe74_zoetrope_20260930_r1"
SCO = "/mnt/umm/users/yinbaiqiao/.sco/bin/sco"
PYTHON = "/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python"
RECOVERIES = (
    ("cog84", 7, "epoch=60-step=74542.ckpt"),
    ("cog84", 16, "epoch=39-step=48880.ckpt"),
    ("ling32", 1, "epoch=52-step=64766.ckpt"),
    ("ling32", 6, "epoch=31-step=39104.ckpt"),
    ("ling32", 15, "epoch=76-step=94094.ckpt"),
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def valid_checkpoint(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            return len(archive.infolist()) > 100
    except (OSError, zipfile.BadZipFile):
        return False


def main() -> None:
    ledger_path = BASE / "corrupt_checkpoint_recovery_r2.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {
        "schema": "PROBE74_CORRUPT_CHECKPOINT_RECOVERY_R2_V1",
        "created_utc": now(),
        "jobs": {},
    }
    for group, index, fallback_name in RECOVERIES:
        item_path = BASE / group / "items" / f"{index:04d}.jsonl"
        item = json.loads(item_path.read_text())
        key = item["unique_key"]
        if key in ledger["jobs"] and ledger["jobs"][key].get("state") == "SUBMITTED":
            continue
        override_index = next(i for i, arg in enumerate(item["scientific_overrides"])
                              if arg.startswith("ckpt_path="))
        current = Path(item["scientific_overrides"][override_index].split("=", 1)[1].strip("'"))
        fallback = current.parent / fallback_name
        if not fallback.is_file() or not valid_checkpoint(fallback):
            raise RuntimeError(f"fallback checkpoint is not structurally valid: {fallback}")
        if current != fallback and valid_checkpoint(current):
            raise RuntimeError(f"refusing to replace a structurally valid current checkpoint: {current}")
        old_path = str(current)
        item["scientific_overrides"][override_index] = f"ckpt_path='{fallback}'"
        item_path.write_text(json.dumps(item, separators=(",", ":"), sort_keys=True) + "\n")
        item_sha = sha(item_path)
        runner = BASE / group / "run_single.py"
        run_id = f"main_v2_retry_probe74_zoetrope_20260930_r1_{group}"
        command = (f"{PYTHON} {runner} --item {item_path} --item-sha256 {item_sha} "
                   f"--run-id {run_id}")
        record = {
            "unique_key": key,
            "group": group,
            "item_path": str(item_path),
            "item_sha256": item_sha,
            "corrupt_checkpoint": old_path,
            "corrupt_size_bytes": Path(old_path).stat().st_size,
            "fallback_checkpoint": str(fallback),
            "fallback_size_bytes": fallback.stat().st_size,
            "fallback_zip_entries": len(zipfile.ZipFile(fallback).infolist()),
            "command": command,
            "state": "INTENT",
            "created_utc": now(),
        }
        ledger["jobs"][key] = record
        write(ledger_path, ledger)
        job_name = f"bq-r74c-z-r2-{group[:2]}-{index:02d}-0930"
        argv = [SCO, "acp", "jobs", "create", "--workspace-name=aigc", "--aec2-name=zoetrope",
                f"--job-name={job_name}", "--priority=HIGHEST", "--quota-type=reserved",
                "--container-image-url=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617",
                "--storage-mount=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm",
                "--training-framework=pytorch", "--worker-nodes=1", "--worker-spec=N4lS.Iq.I80.1",
                f"--command={command}"]
        completed = subprocess.run(argv, text=True, capture_output=True, timeout=120)
        output = completed.stdout + completed.stderr
        ids = list(dict.fromkeys(re.findall(r"pt-[a-z0-9]+", output)))
        record.update(job_name=job_name,
                      state="SUBMITTED" if completed.returncode == 0 and len(ids) == 1 else "FAILED",
                      job_ids=ids, returncode=completed.returncode, scheduler_output=output,
                      submitted_utc=now())
        write(ledger_path, ledger)
        if record["state"] != "SUBMITTED":
            raise RuntimeError(f"submission failed for {key}: {output}")
    print(json.dumps(ledger, indent=2))


if __name__ == "__main__":
    main()
