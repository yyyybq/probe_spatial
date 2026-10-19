#!/usr/bin/env python3
"""Prepare and submit the remaining MiniMax-H3 and LingBot probes to Zoetrope."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shlex
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/umm/users/yinbaiqiao/probe_spatial")
BASE = ROOT / "artifacts/main_v2_probe_v2_816_20260917"
MM_OUT = BASE / "minimax_all_layers_20261011_r1"
MM_RUNNER = ROOT / "runtime/main_v2_probe_minimax_all_20261011_r1/run_single.py"
MM_SOURCE = ROOT / "runtime/main_v2_probe_minimax_bf16_v2/source"
MM_SOURCE_SHA = MM_SOURCE.parent / "source.sha256"
MM_RUN_ID = "main_v2_probe_minimax_all_20261011_r1"
MM_L50_FIX = BASE / "minimax_all_layers_l50_disk49_fix_20261011_r2"
LB_PLAN = BASE / "single_gpu_lingbot40_c_h800_20261004_r1/plan.json"
LB_BAD_LEDGER = BASE / "single_gpu_lingbot40_c_h800_20261004_r1/submissions.json"
LB_OUT = BASE / "lingbot40_c_zoetrope_recovery_20261011_r1"
SCHEDULER_SNAPSHOT = BASE / "probe_progress_20261001/scheduler_jobs.json"
PYTHON = "/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python"
SCO = "/mnt/umm/users/yinbaiqiao/.sco/bin/sco"
IMAGE = "registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617"
MOUNT = "019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def scheduler_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page in range(1, 6):
        completed = subprocess.run(
            [SCO, "acp", "jobs", "list", "--workspace-name=aigc", "--page-size=500", f"--page-token={page}", "-o", "json"],
            text=True,
            capture_output=True,
            timeout=180,
        )
        if completed.returncode:
            raise RuntimeError(completed.stdout + "\n" + completed.stderr)
        batch = json.loads(completed.stdout)
        rows.extend(batch)
        if len(batch) < 500:
            break
    return rows


def create_job(name: str, command: str) -> tuple[str, str]:
    argv = [
        SCO, "acp", "jobs", "create", "--workspace-name=aigc", "--aec2-name=zoetrope",
        f"--job-name={name}", "--priority=HIGHEST", "--quota-type=reserved",
        f"--container-image-url={IMAGE}", f"--storage-mount={MOUNT}",
        "--training-framework=pytorch", "--worker-nodes=1", "--worker-spec=N4lS.Iq.I80.1",
        f"--command={command}",
    ]
    completed = subprocess.run(argv, text=True, capture_output=True, timeout=180)
    output = completed.stdout + completed.stderr
    ids = list(dict.fromkeys(re.findall(r"pt-[a-z0-9]+", output)))
    if completed.returncode or len(ids) != 1:
        raise RuntimeError(f"create failed for {name}: rc={completed.returncode}\n{output}")
    return ids[0], output


def feature_receipts() -> dict[str, Any]:
    specs = [
        ("l30_streaming", BASE / "minimax_layer30_priority_20260924_r1", "streaming", {30}, 10884),
        ("l30_target", BASE / "minimax_layer30_priority_20260924_r1", "target", {30}, 681),
        ("nonl30_streaming", BASE / "minimax_nonl30_recovery_20260930_r1", "streaming", {10, 20, 40, 49}, 43536),
        ("nonl30_target", BASE / "minimax_nonl30_recovery_20260930_r1", "target", {20, 40, 49}, 2043),
    ]
    evidence: dict[str, Any] = {}
    for label, root, mode, expected_layers, expected_units in specs:
        paths = sorted(root.glob(f"{mode}-*-of-8/COMPLETE.json"))
        if len(paths) != 8:
            raise RuntimeError(f"{label}: expected eight completion receipts, got {len(paths)}")
        records = [load(path) for path in paths]
        if any(record.get("status") != "PASS_REMOTE_VERIFIED" or record.get("mode") != mode for record in records):
            raise RuntimeError(f"{label}: non-PASS completion receipt")
        layers = {int(layer) for record in records for layer in (record.get("layers") or [record.get("layer")])}
        units = sum(int(record["units"]) for record in records)
        if layers != expected_layers or units != expected_units:
            raise RuntimeError(f"{label}: layers/units mismatch: {layers}, {units}")
        evidence[label] = {"receipt_paths": [str(path) for path in paths], "layers": sorted(layers), "units": units}
    return evidence


def accepted_minimax_keys() -> set[str]:
    ledger = load(BASE / "minimax_l30_ab16_20260930_r2/formal_submission_ledger.json")
    snapshot = load(SCHEDULER_SNAPSHOT)["jobs"]
    by_id = {row["name"]: row for row in snapshot}
    accepted: set[str] = set()
    for key, record in ledger["jobs"].items():
        job_id = record["job_ids"][-1]
        if by_id.get(job_id, {}).get("state") != "SUCCEEDED":
            raise RuntimeError(f"accepted MiniMax L30 job is not SUCCEEDED: {key} {job_id}")
        match = re.fullmatch(r"minimax_h3__l30__p(8|12|16|24)__(streaming_depth|streaming_mask|view_consistency|ego_belief_v2)__seed42", key)
        if not match:
            raise RuntimeError(f"unexpected accepted MiniMax key: {key}")
        prefix, probe = match.groups()
        accepted.add(f"minimax_h3__scannetpp__t250__l30__p{prefix}__{probe}__seed42")
    if len(accepted) != 16:
        raise RuntimeError(f"expected 16 accepted MiniMax jobs, got {len(accepted)}")
    return accepted


def prepare_minimax() -> None:
    if (MM_OUT / "plan.json").exists():
        print(json.dumps({"status": "ALREADY_PREPARED", "plan": str(MM_OUT / "plan.json")}, indent=2))
        return
    queue_paths = sorted((BASE / "submission_queues_r2_minimax_quarantine").glob("*.quarantine.jsonl"))
    rows: list[dict[str, Any]] = []
    for path in queue_paths:
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    rows.sort(key=lambda row: (int(row["queue_ordinal"]), row["unique_key"]))
    if len(rows) != 148 or len({row["unique_key"] for row in rows}) != 148:
        raise RuntimeError("MiniMax frozen quarantine is not the expected 148 unique experiments")
    if {row["model"] for row in rows} != {"minimax_h3"}:
        raise RuntimeError("MiniMax frozen quarantine contains another model")
    accepted = accepted_minimax_keys()
    pending = [row for row in rows if row["unique_key"] not in accepted]
    if len(pending) != 132:
        raise RuntimeError(f"expected 132 pending MiniMax experiments, got {len(pending)}")
    evidence = feature_receipts()
    if sha256(MM_SOURCE_SHA) != "ab8b6a0d0e8d68f01d35cc37ca5af52be0be6f4007f9557f25a6e8c872c46933":
        raise RuntimeError("MiniMax immutable source checksum-file identity changed")
    jobs = []
    for ordinal, row in enumerate(pending):
        manifest = Path(row["execution_manifest"]["path"])
        if not manifest.is_file() or sha256(manifest) != row["execution_manifest"]["sha256"]:
            raise RuntimeError(f"execution manifest mismatch: {row['unique_key']}")
        config = MM_SOURCE / "configs/experiment" / f"{row['experiment_config']}.yaml"
        if not config.is_file() or sha256(config) != row["experiment_config_sha256"]:
            raise RuntimeError(f"experiment config mismatch: {row['unique_key']}")
        item = MM_OUT / "items" / f"{ordinal:04d}.jsonl"
        item.parent.mkdir(parents=True, exist_ok=True)
        item.write_text(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        os.chmod(item, 0o644)
        item_sha = sha256(item)
        key = row["unique_key"]
        name = "bq-mm-all-" + hashlib.sha256((MM_RUN_ID + key).encode()).hexdigest()[:16]
        command = shlex.join([PYTHON, str(MM_RUNNER), "--item", str(item), "--item-sha256", item_sha, "--run-id", MM_RUN_ID])
        jobs.append({
            "ordinal": ordinal,
            "unique_key": key,
            "layer": row["layer"],
            "disk_layer": row["disk_layer"],
            "prefix": row["prefix"],
            "probe": row["probe"],
            "requires_target": row["requires_target"],
            "item_path": str(item),
            "item_sha256": item_sha,
            "job_name": name,
            "command": command,
        })
    plan = {
        "schema": "MINIMAX_ALL_LAYERS_PROBE_PLAN_V1",
        "status": "READY_TO_SUBMIT",
        "created_utc": now(),
        "resource_pool": "zoetrope",
        "worker_spec": "N4lS.Iq.I80.1",
        "scope_count": 148,
        "already_completed_count": 16,
        "pending_count": 132,
        "source_queue_paths": [str(path) for path in queue_paths],
        "accepted_keys": sorted(accepted),
        "feature_evidence": evidence,
        "runner": str(MM_RUNNER),
        "runner_sha256": sha256(MM_RUNNER),
        "source_checksum_file": str(MM_SOURCE_SHA),
        "source_checksum_file_sha256": sha256(MM_SOURCE_SHA),
        "jobs": jobs,
    }
    atomic(MM_OUT / "plan.json", plan)
    print(json.dumps({"status": plan["status"], "scope": 148, "completed": 16, "pending": 132, "plan": str(MM_OUT / "plan.json")}, indent=2))


def submit_plan(plan_path: Path, ledger_path: Path, schema: str) -> None:
    plan = load(plan_path)
    if plan.get("runner_sha256") and sha256(Path(plan["runner"])) != plan["runner_sha256"]:
        raise RuntimeError("runner SHA mismatch")
    live = {row.get("display_name"): row for row in scheduler_rows()}
    ledger = load(ledger_path) if ledger_path.exists() else {
        "schema": schema,
        "resource_pool": "zoetrope",
        "created_utc": now(),
        "jobs": {},
    }
    failures = []
    for job in plan["jobs"]:
        key = job["unique_key"]
        record = ledger["jobs"].get(key)
        if record and record.get("state") in {"SUBMITTED", "RECONCILED_EXISTING"}:
            continue
        existing = live.get(job["job_name"])
        if existing:
            ledger["jobs"][key] = {**job, "job_id": existing["name"], "state": "RECONCILED_EXISTING", "scheduler_state": existing.get("state"), "reconciled_utc": now()}
            atomic(ledger_path, ledger)
            continue
        record = {**job, "state": "SUBMIT_INTENT", "submitted_utc": now()}
        ledger["jobs"][key] = record
        atomic(ledger_path, ledger)
        try:
            job_id, output = create_job(job["job_name"], job["command"])
            record.update(job_id=job_id, state="SUBMITTED", scheduler_output=output, submitted_utc=now())
            live[job["job_name"]] = {"name": job_id, "display_name": job["job_name"], "state": "SUBMITTED"}
            print(key, job_id, flush=True)
        except Exception as error:
            record.update(state="FAILED", error=f"{type(error).__name__}: {error}", failed_utc=now())
            failures.append(key)
            print(f"FAILED {key}: {error}", flush=True)
        atomic(ledger_path, ledger)
    counts: dict[str, int] = {}
    for record in ledger["jobs"].values():
        counts[record["state"]] = counts.get(record["state"], 0) + 1
    ledger["updated_utc"] = now()
    ledger["state_counts"] = counts
    atomic(ledger_path, ledger)
    print(json.dumps({"ledger": str(ledger_path), "states": counts, "failures": failures}, indent=2))
    if failures:
        raise RuntimeError(f"{len(failures)} submissions failed")


def submit_minimax() -> None:
    prepare_minimax()
    submit_plan(MM_OUT / "plan.json", MM_OUT / "submissions.json", "MINIMAX_ALL_LAYERS_PROBE_SUBMISSIONS_V1")


def stop_invalid_l50_jobs() -> None:
    original = load(MM_OUT / "submissions.json")
    l50 = {key: record for key, record in original["jobs"].items() if int(record["layer"]) == 50}
    if len(l50) != 32:
        raise RuntimeError(f"expected 32 original L50 jobs, got {len(l50)}")
    live = {row["name"]: row for row in scheduler_rows()}
    active_states = {"CREATING", "STARTING", "RUNNING", "QUEUEING", "PENDING"}
    stopping = [record["job_id"] for record in l50.values() if live.get(record["job_id"], {}).get("state") in active_states]
    before = {record["job_id"]: live.get(record["job_id"], {}).get("state", "UNKNOWN") for record in l50.values()}
    if stopping:
        completed = subprocess.run(
            [SCO, "acp", "jobs", "stop", "--workspace-name=aigc", *stopping],
            text=True,
            capture_output=True,
            timeout=180,
        )
        if completed.returncode:
            raise RuntimeError(completed.stdout + "\n" + completed.stderr)
    after: dict[str, str] = {}
    for _ in range(6):
        refreshed = {row["name"]: row for row in scheduler_rows()}
        after = {record["job_id"]: refreshed.get(record["job_id"], {}).get("state", "UNKNOWN") for record in l50.values()}
        if not any(state in active_states for state in after.values()):
            break
        time.sleep(3)
    if any(state in active_states for state in after.values()):
        raise RuntimeError(f"original L50 jobs remain active after stop: {after}")
    claim_root = MM_OUT / "state/claims"
    l50_claims = [claim_root / key for key in l50]
    for _ in range(6):
        remaining_claims = [str(path) for path in l50_claims if path.exists()]
        if not remaining_claims:
            break
        time.sleep(3)
    if remaining_claims:
        raise RuntimeError(f"original L50 writer claims remain after stop: {remaining_claims}")
    atomic(MM_L50_FIX / "stop_receipt.json", {
        "schema": "MINIMAX_L50_DISK49_STOP_RECEIPT_V1",
        "stopped_utc": now(),
        "reason": "logical layer 50 must read disk layer 49; original jobs requested feature_t250_layer50.sft",
        "before": before,
        "after": after,
        "stop_requested": stopping,
    })


def prepare_l50_fix() -> Path:
    path = MM_L50_FIX / "plan.json"
    if path.exists():
        return path
    original = load(MM_OUT / "plan.json")
    jobs = []
    l50 = [job for job in original["jobs"] if int(job["layer"]) == 50]
    if len(l50) != 32:
        raise RuntimeError(f"expected 32 L50 plan jobs, got {len(l50)}")
    for ordinal, old in enumerate(l50):
        row = json.loads(Path(old["item_path"]).read_text())
        if int(row["layer"]) != 50 or int(row["disk_layer"]) != 49:
            raise RuntimeError(f"unexpected L50 mapping: {old['unique_key']}")
        overrides = list(row["scientific_overrides"])
        matches = [index for index, value in enumerate(overrides) if value == "feature_layer=50"]
        if len(matches) != 1 or "feat_postfix=_t250_layer49" not in overrides:
            raise RuntimeError(f"unexpected L50 scientific overrides: {old['unique_key']}")
        overrides[matches[0]] = "feature_layer=49"
        row["scientific_overrides"] = overrides
        item = MM_OUT / "items" / f"l50fix_{ordinal:04d}.jsonl"
        item.write_text(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        os.chmod(item, 0o644)
        item_sha = sha256(item)
        key = row["unique_key"]
        name = "bq-mm-l50fx-" + hashlib.sha256((MM_RUN_ID + "disk49fix" + key).encode()).hexdigest()[:14]
        command = shlex.join([PYTHON, str(MM_RUNNER), "--item", str(item), "--item-sha256", item_sha, "--run-id", MM_RUN_ID])
        jobs.append({**old, "ordinal": ordinal, "item_path": str(item), "item_sha256": item_sha,
                     "job_name": name, "command": command, "logical_layer": 50, "disk_layer": 49})
    plan = {
        "schema": "MINIMAX_L50_DISK49_FIX_PLAN_V1",
        "status": "READY_TO_SUBMIT_AFTER_ORIGINAL_L50_STOP",
        "created_utc": now(),
        "resource_pool": "zoetrope",
        "worker_spec": "N4lS.Iq.I80.1",
        "reason": "direct canonical roots use feature_layer for the filename; logical L50 maps to disk layer 49",
        "runner": str(MM_RUNNER),
        "runner_sha256": sha256(MM_RUNNER),
        "jobs": jobs,
    }
    atomic(path, plan)
    return path


def submit_l50_fix() -> None:
    stop_invalid_l50_jobs()
    plan = prepare_l50_fix()
    submit_plan(plan, MM_L50_FIX / "submissions.json", "MINIMAX_L50_DISK49_FIX_SUBMISSIONS_V1")


def prepare_lingbot_recovery_plan() -> Path:
    path = LB_OUT / "plan.json"
    if path.exists():
        return path
    original = load(LB_PLAN)
    bad = load(LB_BAD_LEDGER)
    if len(original.get("jobs", [])) != 40:
        raise RuntimeError("LingBot C recovery plan is not the expected 40 experiments")
    if len(bad.get("jobs", {})) != 19 or any(record.get("state") != "SUBMIT_INTENT" for record in bad["jobs"].values()):
        raise RuntimeError("LingBot failed release ledger no longer matches the audited 19 intents")
    jobs = []
    for original_job in original["jobs"]:
        result_path = Path(original_job["validation_result"])
        result = load(result_path)
        result_sha = sha256(result_path)
        frozen_result_sha = original_job.get("validation_result_sha256")
        if (result.get("status") != "PASS"
                or result.get("case_sha256") != original_job["case_sha256"]
                or (frozen_result_sha and result_sha != frozen_result_sha)):
            raise RuntimeError(f"LingBot validation is not PASS and SHA-bound: {original_job['unique_key']}")
        if sha256(Path(original_job["item_path"])) != original_job["item_sha256"]:
            raise RuntimeError(f"LingBot item SHA mismatch: {original_job['unique_key']}")
        key = original_job["unique_key"]
        name = "bq-lbc40-z2-" + hashlib.sha256((original["run_id"] + key).encode()).hexdigest()[:14]
        command = shlex.join([PYTHON, original["runner"], "--item", original_job["item_path"], "--item-sha256", original_job["item_sha256"], "--run-id", original["run_id"]])
        jobs.append({**original_job, "validation_result_sha256": result_sha,
                     "job_name": name, "command": command})
    plan = {
        "schema": "LINGBOT_C40_ZOETROPE_RECOVERY_PLAN_V1",
        "status": "READY_TO_SUBMIT",
        "created_utc": now(),
        "resource_pool": "zoetrope",
        "worker_spec": "N4lS.Iq.I80.1",
        "replaces_failed_validation_release_jobs": 20,
        "reconciled_stale_submit_intents": 19,
        "runner": original["runner"],
        "runner_sha256": original["runner_sha256"],
        "jobs": jobs,
    }
    atomic(path, plan)
    return path


def submit_lingbot() -> None:
    plan = prepare_lingbot_recovery_plan()
    submit_plan(plan, LB_OUT / "submissions.json", "LINGBOT_C40_ZOETROPE_RECOVERY_SUBMISSIONS_V1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare-minimax", "submit-minimax", "submit-lingbot", "submit-l50-fix", "submit-all"])
    args = parser.parse_args()
    if args.action == "prepare-minimax":
        prepare_minimax()
    elif args.action == "submit-minimax":
        submit_minimax()
    elif args.action == "submit-lingbot":
        submit_lingbot()
    elif args.action == "submit-l50-fix":
        submit_l50_fix()
    else:
        submit_lingbot()
        submit_minimax()


if __name__ == "__main__":
    main()
