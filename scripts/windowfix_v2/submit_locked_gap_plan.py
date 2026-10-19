#!/usr/bin/env python3
"""Submit one immutable canonical-v2 gap SCO plan and persist submit receipts.

This controller deliberately accepts only jobs that invoke the canonical gap
runner with a batch owned by the supplied, lock-verified plan.  It leaves the
feature lifecycle to ``run_canonical_gap_shard.py`` and its sync-loop child.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


JOB_ID_RE = re.compile(r"\b(pt-[a-z0-9]+)\b")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    path.chmod(0o644)


def require_hash(path: Path, expected: str, description: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"{description} SHA mismatch: {actual} != {expected}: {path}")


def _as_path(plan_dir: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else plan_dir / candidate


def validate_plan(plan_dir: Path) -> tuple[dict[str, Any], dict[str, Any], list[tuple[Path, dict[str, Any], dict[str, Any]]]]:
    lock_path = plan_dir / "PLAN.lock.json"
    if not lock_path.exists():
        lock_path = plan_dir / "PLAN.lock"
    lock = load_json(lock_path)

    summary_path = plan_dir / "summary.json"
    if "summary_sha256" in lock:
        require_hash(summary_path, lock["summary_sha256"], "summary")
    elif "summary" in lock and isinstance(lock["summary"], dict):
        require_hash(summary_path, lock["summary"].get("sha256", ""), "summary")
    summary = load_json(summary_path)

    jobspecs = lock.get("jobspecs") or lock.get("job_specs")
    if not isinstance(jobspecs, dict) or not jobspecs:
        raise RuntimeError("PLAN.lock has no jobspec hash map")

    validated: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    for jobspec_ref, jobspec_sha in sorted(jobspecs.items()):
        jobspec_path = _as_path(plan_dir, jobspec_ref)
        if not jobspec_path.exists() and not Path(jobspec_ref).is_absolute():
            jobspec_path = plan_dir / "jobspecs" / jobspec_ref
        if not jobspec_path.exists():
            raise RuntimeError(f"missing jobspec: {jobspec_path}")
        if isinstance(jobspec_sha, dict):
            jobspec_sha = jobspec_sha.get("sha256")
        require_hash(jobspec_path, str(jobspec_sha), "jobspec")
        jobspec = load_json(jobspec_path)
        worker_script = jobspec.get("worker_script")
        worker_script_sha = jobspec.get("worker_script_sha256")
        worker_script_text = ""
        if worker_script:
            worker_script_path = Path(str(worker_script))
            require_hash(worker_script_path, str(worker_script_sha), "worker script")
            worker_script_text = worker_script_path.read_text(encoding="utf-8")

        argv = jobspec.get("sco_argv") or jobspec.get("argv")
        if not isinstance(argv, list) or len(argv) < 4:
            raise RuntimeError(f"jobspec missing SCO argv: {jobspec_path}")
        joined = " ".join(map(str, argv))
        command_text = joined + "\n" + worker_script_text
        if "sco acp jobs create" not in joined or "run_canonical_gap_shard.py" not in command_text:
            raise RuntimeError(f"jobspec is not a canonical gap SCO create: {jobspec_path}")
        batch_match = re.search(r"--batch\s+([^\s'\"]+)", command_text)
        if not batch_match:
            raise RuntimeError(f"runner batch missing from jobspec: {jobspec_path}")
        batch_path = Path(batch_match.group(1))
        if not batch_path.is_absolute():
            batch_path = plan_dir / batch_path
        if plan_dir.resolve() not in batch_path.resolve().parents:
            raise RuntimeError(f"job batch does not belong to plan: {batch_path}")
        batch = load_json(batch_path)
        validated.append((jobspec_path, jobspec, batch))
    return lock, summary, validated


def extract_metadata(batch: dict[str, Any]) -> dict[str, Any]:
    shards = batch.get("shards", [])
    worklists: list[dict[str, Any]] = []
    output_dirs: list[str] = []
    backends: set[str] = set()
    for shard in shards:
        shard_payload = shard
        shard_path = shard.get("path")
        if shard_path and Path(str(shard_path)).exists():
            shard_payload = load_json(Path(str(shard_path)))
        worklist = shard_payload.get("worklist") or shard_payload.get("worklist_path")
        if worklist:
            path = Path(str(worklist))
            worklists.append({
                "path": str(path),
                "file_sha256": shard_payload.get("worklist_file_sha256") or (sha256(path) if path.exists() else None),
                "content_sha256": shard_payload.get("worklist_content_sha256"),
            })
        output = shard_payload.get("output_dir") or shard_payload.get("feature_root")
        if output:
            output_dirs.append(str(output))
        backend = shard_payload.get("vfm") or shard_payload.get("backend")
        if backend:
            backends.add(str(backend))
    return {
        "backend": sorted(backends),
        "output_dirs": sorted(set(output_dirs)),
        "worklists": worklists,
        "item_count": sum(int(shard.get("item_count", shard.get("layer_item_count", 0))) for shard in shards),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--sco", default="sco")
    parser.add_argument("--only-batch", action="append", default=[], help="submit only these lock-verified batch IDs")
    parser.add_argument("--execute", action="store_true", help="perform SCO submission; default validates only")
    args = parser.parse_args()

    plan_dir = args.plan_dir.resolve()
    lock, summary, jobs = validate_plan(plan_dir)
    lock_path = plan_dir / ("PLAN.lock.json" if (plan_dir / "PLAN.lock.json").exists() else "PLAN.lock")
    plan_lock_sha = sha256(lock_path)
    plan_id = lock.get("plan_id") or summary.get("plan_id")
    if not plan_id:
        raise RuntimeError("immutable plan ID missing")

    for jobspec_path, jobspec, batch in jobs:
        if args.only_batch and str(batch.get("batch_id")) not in set(args.only_batch):
            continue
        job_name = str(jobspec.get("job_name") or jobspec.get("name") or jobspec_path.stem)
        receipt_path = args.receipt_dir / f"{job_name}.json"
        if receipt_path.exists():
            existing = load_json(receipt_path)
            if existing.get("job_id"):
                print(f"already recorded {job_name}: {existing['job_id']}")
                continue

        argv = [str(x) for x in (jobspec.get("sco_argv") or jobspec["argv"])]
        if argv[0] == "sco":
            argv[0] = args.sco
        metadata = extract_metadata(batch)
        now = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
        receipt: dict[str, Any] = {
            "kind": "canonical_v2_gap_sco_submission_receipt_v1",
            "submitted_at": now,
            "submission_state": "validated_not_submitted",
            "plan_id": plan_id,
            "plan_dir": str(plan_dir),
            "plan_lock_path": str(lock_path),
            "plan_lock_sha256": plan_lock_sha,
            "jobspec_path": str(jobspec_path),
            "jobspec_sha256": sha256(jobspec_path),
            "batch_id": batch.get("batch_id"),
            "batch_sha256": jobspec.get("batch_sha256"),
            "worklist_sha256": metadata["worklists"],
            "backend": metadata["backend"],
            "resource": jobspec.get("resource") or jobspec.get("resources"),
            "output_dirs": metadata["output_dirs"],
            "item_count": metadata["item_count"],
            "sco_argv": argv,
            "runner_receipt_contract": "remote canonical-unit receipt, then local payload deletion only after verified receipt",
        }
        if args.execute:
            completed = subprocess.run(argv, text=True, capture_output=True, check=False)
            receipt.update({
                "submission_state": "submitted" if completed.returncode == 0 else "submission_failed",
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            })
            match = JOB_ID_RE.search(completed.stdout + "\n" + completed.stderr)
            if match:
                receipt["job_id"] = match.group(1)
                receipt["current_status"] = "submitted"
            if completed.returncode != 0:
                atomic_json(receipt_path, receipt)
                raise RuntimeError(f"SCO submit failed for {job_name}; receipt: {receipt_path}")
        atomic_json(receipt_path, receipt)
        print(f"{job_name}: {receipt.get('job_id', 'validated')} -> {receipt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
