#!/usr/bin/env python3
"""Refresh scheduler and verifier state for MiniMax BF16 V2 production shards."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/mnt/umm/users/yinbaiqiao/probe_spatial")
CAMPAIGN = ROOT / "artifacts/main_v2_probe_v2_816_20260917"
LEDGER = CAMPAIGN / "minimax_bf16_v2_production_v3_submission_ledger.json"
STATE_ROOT = CAMPAIGN / "minimax_bf16_v2/production_extract_v3"
SCO = "/mnt/umm/users/yinbaiqiao/.sco/bin/sco"
TERMINAL = {"SUCCEEDED", "FAILED", "CANCELLED", "DELETED", "STOPPED"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o644)
    Path(temporary).replace(path)


def describe(job_id: str) -> dict:
    result = subprocess.run(
        [SCO, "acp", "jobs", "describe", "--workspace-name=aigc", "-o", "json", job_id],
        text=True, capture_output=True, timeout=60,
    )
    if result.returncode:
        raise RuntimeError(f"describe failed for {job_id}: {result.stderr}")
    return json.loads(result.stdout)


def main() -> None:
    ledger = json.loads(LEDGER.read_text())
    if ledger.get("shard_count") != 8:
        raise RuntimeError("unexpected production ledger")
    states: dict[str, int] = {}
    stream_total = target_total = scene_total = 0
    all_pass = True
    for key, record in sorted(ledger["jobs"].items()):
        state = describe(record["job_id"])
        scheduler_state = str(state["state"]).upper()
        record.update({
            "scheduler_state": scheduler_state,
            "scheduler_start_time": state.get("start_time"),
            "scheduler_complete_time": state.get("complete_time"),
            "scheduler_update_time": state.get("update_time"),
        })
        states[scheduler_state] = states.get(scheduler_state, 0) + 1
        report_path = STATE_ROOT / f"shard_{int(key)}_of_8/summary.json"
        if scheduler_state == "SUCCEEDED":
            if not report_path.is_file():
                raise RuntimeError(f"SUCCEEDED shard lacks verifier report: {key}")
            report = json.loads(report_path.read_text())
            if report.get("status") != "PASS" or report.get("shard_count") != 8:
                raise RuntimeError(f"invalid verifier report: {key}")
            record["verification_report"] = str(report_path.resolve())
            record["verification_report_sha256"] = sha256(report_path)
            scene_total += int(report["scene_count"])
            stream_total += int(report["streaming_payloads_checked"])
            target_total += int(report["target_payloads_checked"])
        else:
            all_pass = False
            if scheduler_state in TERMINAL:
                record["terminal_failure"] = True
    if any(state in TERMINAL - {"SUCCEEDED"} for state in states):
        ledger["status"] = "TERMINAL_FAILURE"
    elif all_pass and len(ledger["jobs"]) == 8:
        if (scene_total, stream_total, target_total) != (681, 33_880, 2_724):
            raise RuntimeError(
                f"aggregate verifier counts differ: {scene_total}/{stream_total}/{target_total}"
            )
        ledger["status"] = "SUCCEEDED"
    else:
        ledger["status"] = "RUNNING"
    ledger["scheduler_state_counts"] = states
    ledger["verified_counts"] = {
        "scenes": scene_total,
        "streaming_payloads": stream_total,
        "target_payloads": target_total,
    }
    ledger["reconciled_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(LEDGER, ledger)
    print(json.dumps({
        "status": ledger["status"], "scheduler_state_counts": states,
        "verified_counts": ledger["verified_counts"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
