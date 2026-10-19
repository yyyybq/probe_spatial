#!/usr/bin/env python3
"""Submit a frozen target AOSS-audit manifest and write per-job receipts.

This is deliberately narrow: it accepts only the hardened r5 target route-audit
manifest and validates every SHA-bound binding before invoking SCO.  Receipts
are additive operational evidence; no reviewed plan artifact is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_PLAN_TYPE = "canonical_v2_target_route_audit_sco_review"
JOB_ID = re.compile(r"\bpt-[a-z0-9]+\b")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"{label} must be a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"{label} is not a JSON object: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise SystemExit(f"refusing symlink receipt: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o644)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def require_sha(path: Path, expected: str, label: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64 or sha256_file(path) != expected:
        raise SystemExit(f"{label} SHA-256 mismatch: {path}")


def checked_jobs(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = read_json(manifest_path, "audit manifest")
    if manifest.get("plan_type") != EXPECTED_PLAN_TYPE:
        raise SystemExit("refusing non-hardened target AOSS audit plan")
    content = dict(manifest)
    stored = content.pop("content_sha256", None)
    if stored != canonical_json_sha(content):
        raise SystemExit("audit manifest content SHA mismatch")
    jobs: list[dict[str, Any]] = []
    for item in manifest.get("shards", []):
        if not isinstance(item, dict):
            raise SystemExit("audit manifest contains a non-object shard")
        shard = Path(str(item["plan"])).resolve()
        binding = Path(str(item["worker_binding_receipt"])).resolve()
        require_sha(shard, str(item["plan_file_sha256"]), "audit shard")
        require_sha(binding, str(item["worker_binding_receipt_file_sha256"]), "worker binding")
        argv = item.get("sco_argv")
        if not isinstance(argv, list) or not all(isinstance(value, str) for value in argv):
            raise SystemExit(f"invalid SCO argv for {item.get('vfm')}")
        if argv[:4] != ["sco", "acp", "jobs", "create"]:
            raise SystemExit(f"unexpected SCO command for {item.get('vfm')}")
        names = [value.split("=", 1)[1] for value in argv if value.startswith("--job-name=")]
        if len(names) != 1 or not names[0].startswith("bq-wf2-tgt-audit-"):
            raise SystemExit(f"unexpected audit job name: {names}")
        jobs.append({"item": item, "argv": argv, "job_name": names[0]})
    if len(jobs) != 10:
        raise SystemExit(f"expected 10 audit jobs, got {len(jobs)}")
    return manifest, jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sco", type=Path, required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.sco.is_file() or not os.access(args.sco, os.X_OK):
        raise SystemExit(f"SCO CLI is not executable: {args.sco}")
    manifest, jobs = checked_jobs(args.manifest)
    for job in jobs:
        item = job["item"]
        argv = [str(args.sco), *job["argv"][1:]]
        now = datetime.now(timezone.utc).isoformat()
        receipt = {
            "schema_version": 1,
            "kind": "canonical_v2_target_aoss_audit_sco_submission",
            "submitted_at": now,
            "manifest_path": str(args.manifest.resolve()),
            "manifest_content_sha256": manifest["content_sha256"],
            "job_name": job["job_name"],
            "vfm": item["vfm"],
            "layers": item["layers"],
            "plan_path": item["plan"],
            "plan_file_sha256": item["plan_file_sha256"],
            "plan_content_sha256": item["plan_content_sha256"],
            "worker_binding_receipt": item["worker_binding_receipt"],
            "worker_binding_receipt_file_sha256": item["worker_binding_receipt_file_sha256"],
            "worker_binding_receipt_content_sha256": item["worker_binding_receipt_content_sha256"],
            "output_dir": item["audit_output"],
            "resources": {
                "aec2": "h800", "worker_nodes": 1, "worker_spec": "N4lS.Iq.I80.8"
            },
            "sco_argv": argv,
            "sco_command_rendered": " ".join(argv),
            "worker_command": item["worker_command"],
            "status": "DRY_RUN_NOT_SUBMITTED" if not args.execute else "SUBMISSION_FAILED",
        }
        if args.execute:
            process = subprocess.run(argv, text=True, capture_output=True, check=False)
            receipt["sco_returncode"] = process.returncode
            receipt["sco_stdout"] = process.stdout
            receipt["sco_stderr"] = process.stderr
            match = JOB_ID.search(process.stdout + "\n" + process.stderr)
            if process.returncode == 0 and match:
                receipt["sco_job_id"] = match.group(0)
                receipt["status"] = "SUBMITTED"
            elif process.returncode == 0:
                receipt["status"] = "SUBMITTED_JOB_ID_UNPARSED"
            else:
                receipt["status"] = "SUBMISSION_FAILED"
        receipt["content_sha256"] = canonical_json_sha(receipt)
        atomic_json(args.receipt_dir / f"{item['ordinal']:02d}-{item['vfm']}.json", receipt)
        print(json.dumps({"job_name": job["job_name"], "status": receipt["status"], "job_id": receipt.get("sco_job_id")}, sort_keys=True))
        if args.execute and receipt["status"] not in {"SUBMITTED", "SUBMITTED_JOB_ID_UNPARSED"}:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
