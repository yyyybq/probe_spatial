#!/usr/bin/env python3
"""Submit the eight frozen probe816 training canaries as one-GPU H800 jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCO = "/mnt/umm/users/yinbaiqiao/.sco/bin/sco"
IMAGE = "registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617"
MOUNT = "019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm"
ROOT = Path("/mnt/umm/users/yinbaiqiao/probe_spatial")
RUNNER = ROOT / "scripts/run_probe816_training_canary.py"
SHORT = {
    "cogvideox": "cv", "videomae": "vm", "dino": "dn", "f3r": "f3",
    "vjepa2": "vj", "qwen2_5_vl": "qw", "lingbot": "lb", "minimax_h3": "mm",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--models", default="", help="Optional comma-separated model allow-list")
    args = parser.parse_args()
    bundle_path = args.bundle.resolve()
    bundle = json.loads(bundle_path.read_text())
    if bundle.get("status") != "FROZEN_NOT_SUBMITTED" or bundle.get("case_count") != 8:
        raise RuntimeError("bundle is not the expected frozen eight-case bundle")
    ledger_path = bundle_path.parent / "submission_ledger.json"
    if ledger_path.exists():
        raise RuntimeError(f"refusing duplicate submission: {ledger_path}")
    runner_sha = digest(RUNNER)
    records: list[dict[str, Any]] = []
    requested = {value for value in args.models.split(",") if value}
    if requested and not requested.issubset(SHORT):
        raise RuntimeError(f"unknown requested models: {sorted(requested - set(SHORT))}")
    for entry in bundle["cases"]:
        model = entry["model"]
        if requested and model not in requested:
            continue
        case_path = Path(entry["case_path"])
        if digest(case_path) != entry["case_sha256"]:
            raise RuntimeError(f"case SHA changed: {model}")
        command = (
            f"cd {ROOT} && /mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python "
            f"{RUNNER} --case {case_path} --expected-case-sha256 {entry['case_sha256']} "
            f"--expected-runner-sha256 {runner_sha}"
        )
        if not re.fullmatch(r"[a-z0-9-]{1,12}", args.tag):
            raise RuntimeError("invalid submission tag")
        job_name = f"bq-p816-{args.tag}-{SHORT[model]}-0922"
        argv = [
            SCO, "acp", "jobs", "create", "--workspace-name=aigc",
            "--aec2-name=h800", f"--job-name={job_name}", "--priority=HIGHEST",
            f"--container-image-url={IMAGE}", f"--storage-mount={MOUNT}",
            "--training-framework=pytorch", "--worker-nodes=1",
            "--worker-spec=N4lS.Iq.I80.1", f"--command={command}",
        ]
        completed = subprocess.run(argv, text=True, capture_output=True, check=False)
        combined = completed.stdout + "\n" + completed.stderr
        ids = list(dict.fromkeys(re.findall(r"pt-[a-z0-9]+", combined)))
        record = {
            "model": model, "job_name": job_name, "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
            "returncode": completed.returncode, "stdout": completed.stdout,
            "stderr": completed.stderr, "job_ids": ids, "command": command,
            "case_path": str(case_path), "case_sha256": entry["case_sha256"],
            "runner_sha256": runner_sha, "resource": entry["resource"],
        }
        records.append(record)
        atomic_json(bundle_path.parent / f"submission_{model}.json", record)
        print(json.dumps({"model": model, "returncode": completed.returncode, "job_ids": ids}))
        if completed.returncode != 0 or len(ids) != 1:
            atomic_json(ledger_path, {"status": "PARTIAL_FAILURE", "records": records})
            raise RuntimeError(f"submission failed for {model}")
    expected_count = len(requested) if requested else int(bundle["case_count"])
    if len(records) != expected_count:
        raise RuntimeError(f"submitted {len(records)} cases, expected {expected_count}")
    atomic_json(ledger_path, {
        "schema": "PROBE816_RECEIPT_UNION_TRAINING_CANARY_SUBMISSION_V1",
        "status": "SUBMITTED", "bundle": str(bundle_path),
        "bundle_sha256": digest(bundle_path), "runner_sha256": runner_sha,
        "records": records,
    })


if __name__ == "__main__":
    main()
