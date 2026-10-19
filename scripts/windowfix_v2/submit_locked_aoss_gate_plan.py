#!/usr/bin/env python3
"""Submit an immutable, fully-preflighted canonical-v2 AOSS gate wrapper."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


JOB_ID = re.compile(r"\b(pt-[a-z0-9]+)\b")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)
    path.chmod(0o644)


def validate(root: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    lock_path = root / "PLAN.lock.json"
    lock = read(lock_path)
    if lock.get("status") != "IMMUTABLE_REVIEW_ARTIFACT_NOT_SUBMITTED":
        raise RuntimeError("wrapper lock is not a review-only immutable artifact")
    files = lock.get("artifact_files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("wrapper lock has no artifact file bindings")
    for rel, binding in files.items():
        path = root / rel
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing regular locked artifact: {path}")
        expected = binding.get("file_sha256")
        if sha(path) != expected:
            raise RuntimeError(f"locked artifact SHA mismatch: {path}")
    summary = read(root / "summary.json")
    if summary.get("status") != "READY_FOR_REVIEW_NOT_SUBMITTED":
        raise RuntimeError("wrapper summary is not ready for controlled submit")
    counts = summary.get("counts", {})
    if int(counts.get("jobspecs", -1)) != 128 or int(counts.get("worker_bindings", -1)) != 128:
        raise RuntimeError("expected exactly 128 fully bound gate shards")
    return lock_path, lock, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-dir", required=True, type=Path)
    parser.add_argument("--receipt-dir", required=True, type=Path)
    parser.add_argument("--sco", default="sco")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = args.plan_dir.resolve()
    lock_path, lock, summary = validate(root)
    plan_lock_sha = sha(lock_path)
    source = summary.get("source_gate_plan", {})
    source_plan_path = Path(str(source.get("path", "")))
    if not source_plan_path.is_file():
        raise RuntimeError("bound source gate plan is missing")

    specs = sorted((root / "jobspecs").glob("*.json"))
    for spec_path in specs:
        spec = read(spec_path)
        if spec.get("status") != "READY_FOR_REVIEW_NOT_SUBMITTED":
            raise RuntimeError(f"jobspec is not a review-only ready artifact: {spec_path}")
        binding = spec.get("binding_receipt", {})
        binding_path = Path(str(binding.get("path", "")))
        if not binding_path.is_file() or sha(binding_path) != binding.get("file_sha256"):
            raise RuntimeError(f"jobspec binding SHA mismatch: {spec_path}")
        bound = read(binding_path)
        output_dir = bound.get("gate_execution", {}).get("output_dir")
        argv = [str(item) for item in spec.get("sco_argv", [])]
        if len(argv) < 4 or "sco acp jobs create" not in " ".join(argv):
            raise RuntimeError(f"invalid SCO create argv: {spec_path}")
        if argv[0] == "sco":
            argv[0] = args.sco
        receipt_path = args.receipt_dir / f"{spec_path.stem}.json"
        if receipt_path.exists() and read(receipt_path).get("job_id"):
            print(f"already recorded {spec_path.stem}")
            continue
        receipt: dict[str, Any] = {
            "kind": "canonical_v2_aoss_gate_sco_submission_receipt_v1",
            "submitted_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
            "submission_state": "validated_not_submitted",
            "plan_dir": str(root),
            "plan_lock_path": str(lock_path),
            "plan_lock_sha256": plan_lock_sha,
            "source_gate_plan": str(source_plan_path),
            "source_gate_plan_file_sha256": sha(source_plan_path),
            "source_gate_plan_content_sha256": source.get("content_sha256"),
            "catalogue_manifest": summary.get("source_gate_inputs", {}).get("catalog_manifest"),
            "jobspec_path": str(spec_path),
            "jobspec_sha256": sha(spec_path),
            "binding_path": str(binding_path),
            "binding_sha256": sha(binding_path),
            "shard_id": spec.get("shard_id"),
            "shard_index": spec.get("shard_index"),
            "expected_request_records": spec.get("expected_request_records"),
            "resource": summary.get("sco_environment"),
            "output_dir": output_dir,
            "sco_argv": argv,
            "receipt_contract": "LEGACY_EXACT_REFERENCE_REMOTE_GATE_RECEIPT_V1",
            "read_only": True,
        }
        if args.execute:
            completed = subprocess.run(argv, text=True, capture_output=True, check=False)
            receipt.update({"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr})
            found = JOB_ID.search(completed.stdout + "\n" + completed.stderr)
            if completed.returncode == 0 and found:
                receipt.update({"submission_state": "submitted", "job_id": found.group(1), "current_status": "submitted"})
            else:
                receipt["submission_state"] = "submission_failed"
                write(receipt_path, receipt)
                raise RuntimeError(f"SCO submission failed for {spec_path}; receipt retained")
        write(receipt_path, receipt)
        print(f"{spec.get('shard_id')}: {receipt.get('job_id', 'validated')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
