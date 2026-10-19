#!/usr/bin/env python3
"""Write immutable operational receipts for submitted backend canaries.

The renderer deliberately remains review-only.  This companion consumes its
lock-verified specs *after* a human/controller has called SCO, and preserves
the exact SCO argv, immutable plan bindings, receipt/log paths, and observed
SCO state in one additive file per backend.  It never creates jobs itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def regular_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"{label} must be a regular non-symlink file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"{label} must be a JSON object: {path}")
    return value


def atomic_json_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise SystemExit(f"refusing to overwrite execution receipt: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # No prior target is allowed, so replace is an atomic publication.
        os.replace(temporary, path)
        os.chmod(path, 0o644)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def parse_mapping(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"expected backend=value mapping, got {value!r}")
        backend, mapped = value.split("=", 1)
        if not backend or not mapped:
            raise SystemExit(f"invalid backend=value mapping: {value!r}")
        if backend in result:
            raise SystemExit(f"duplicate backend mapping: {backend}")
        result[backend] = mapped
    return result


def sco_option(argv: list[str], prefix: str) -> str:
    """Read one immutable ``--name=value`` option from a rendered SCO argv."""
    matches = [value[len(prefix):] for value in argv if value.startswith(prefix)]
    if len(matches) != 1 or not matches[0]:
        raise SystemExit(f"rendered SCO argv must contain exactly one {prefix} option")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--sco", default="sco")
    parser.add_argument(
        "--zone",
        default="",
        help="Optional SCO API zone override, e.g. cn-fz-01z.",
    )
    parser.add_argument("--subscription", default="")
    parser.add_argument("--resource-group", default="")
    parser.add_argument("--vpc-id", default="")
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Submit every lock-verified canary spec and write its immutable "
            "operational receipt.  Without this flag the command only records "
            "explicitly supplied pre-existing job IDs."
        ),
    )
    parser.add_argument(
        "--job-id", action="append", default=[], metavar="VFM=PT_ID",
        help="one SCO job ID for each backend in the frozen canary plan",
    )
    parser.add_argument(
        "--sco-state", action="append", default=[], metavar="VFM=STATE",
        help="observed SCO state for each backend (default SUBMITTED)",
    )
    parser.add_argument(
        "--submitted-at-utc", action="append", default=[], metavar="VFM=ISO8601",
        help="actual SCO create time for each backend; omit only when unavailable",
    )
    args = parser.parse_args()

    plan_dir = args.plan_dir.resolve()
    summary_path = plan_dir / "summary.json"
    lock_path = plan_dir / "PLAN.lock.json"
    summary = regular_json(summary_path, "canary plan summary")
    lock = regular_json(lock_path, "canary plan lock")
    if summary.get("status") != "READY_FOR_REVIEW_NOT_SUBMITTED":
        raise SystemExit("canary plan is not an immutable review artifact")
    if sha256(summary_path) != lock.get("summary_sha256"):
        raise SystemExit("canary plan summary SHA mismatch")
    specs_lock = lock.get("specs")
    if not isinstance(specs_lock, dict) or not specs_lock:
        raise SystemExit("canary plan lock has no specs")
    worker_scripts_lock = lock.get("worker_scripts", {})
    if worker_scripts_lock is not None and not isinstance(worker_scripts_lock, dict):
        raise SystemExit("canary plan worker_scripts lock must be a map")
    job_ids = parse_mapping(args.job_id)
    states = parse_mapping(args.sco_state)
    submitted_times = parse_mapping(args.submitted_at_utc)
    expected = {Path(name).stem for name in specs_lock}
    if args.execute:
        if job_ids or states or submitted_times:
            raise SystemExit(
                "--execute submits fresh jobs; do not mix it with manually supplied job/state/time mappings"
            )
    elif set(job_ids) != expected:
        raise SystemExit(f"job IDs must cover exactly {sorted(expected)}, got {sorted(job_ids)}")
    if set(states) - expected:
        raise SystemExit("SCO states name an unknown backend")
    if set(submitted_times) - expected:
        raise SystemExit("submission timestamps name an unknown backend")
    observed_at = datetime.now(timezone.utc).isoformat()

    for backend in sorted(expected):
        spec_path = plan_dir / "specs" / f"{backend}.json"
        expected_sha = specs_lock.get(f"{backend}.json")
        if sha256(spec_path) != expected_sha:
            raise SystemExit(f"canary spec SHA mismatch: {backend}")
        spec = regular_json(spec_path, "canary spec")
        worker_script = spec.get("worker_script")
        worker_script_sha = spec.get("worker_script_sha256")
        if worker_script:
            script_path = Path(str(worker_script))
            if sha256(script_path) != worker_script_sha:
                raise SystemExit(f"canary worker script SHA mismatch: {backend}")
            locked_script_sha = worker_scripts_lock.get(script_path.name)
            if locked_script_sha and locked_script_sha != worker_script_sha:
                raise SystemExit(f"canary worker script lock mismatch: {backend}")
        raw = dict(spec)
        content_sha = raw.pop("content_sha256", None)
        if content_sha != canonical_sha(raw):
            raise SystemExit(f"canary spec content SHA mismatch: {backend}")
        argv = spec.get("sco_argv")
        if not isinstance(argv, list) or argv[:4] != ["sco", "acp", "jobs", "create"]:
            raise SystemExit(f"invalid SCO create argv: {backend}")
        actual_argv = [str(item) for item in argv]
        actual_argv[0] = args.sco
        global_options = []
        for option, value in (
            ("--zone", args.zone),
            ("--subscription", args.subscription),
            ("--resource-group", args.resource_group),
        ):
            if value:
                global_options.extend([option, value])
        if global_options:
            actual_argv[1:1] = global_options
        if args.vpc_id:
            actual_argv.insert(4 + len(global_options), f"--vpc-id={args.vpc_id}")
        worker_spec = sco_option(actual_argv, "--worker-spec=")
        aec2 = sco_option(actual_argv, "--aec2-name=")
        worker_nodes = int(sco_option(actual_argv, "--worker-nodes="))
        submitted_at = submitted_times.get(backend)
        observed_state = states.get(backend, "SUBMITTED")
        stdout = None
        stderr = None
        returncode = None
        job_id = job_ids.get(backend)
        if args.execute:
            completed = subprocess.run(
                actual_argv,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            stdout, stderr, returncode = completed.stdout, completed.stderr, completed.returncode
            match = re.search(r"\b(pt-[a-z0-9]+)\b", stdout + "\n" + stderr)
            if completed.returncode != 0 or match is None:
                failed = {
                    "kind": "canonical_v2_backend_canary_sco_submission_failure_v1",
                    "backend": backend,
                    "plan_dir": str(plan_dir),
                    "spec_path": str(spec_path),
                    "spec_file_sha256": expected_sha,
                    "sco_argv": actual_argv,
                    "returncode": returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                }
                failed["content_sha256"] = canonical_sha(failed)
                atomic_json_new(args.receipt_dir / f"{backend}.json", failed)
                raise SystemExit(
                    f"SCO submission failed for {backend}; receipt written to {args.receipt_dir / f'{backend}.json'}"
                )
            job_id = match.group(1)
            submitted_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            observed_state = "SUBMITTED"
        receipt = {
            "schema_version": 1,
            "kind": "canonical_v2_backend_canary_sco_submission_receipt_v1",
            "submitted_at_utc": submitted_at,
            "recorded_at_utc": observed_at,
            "plan_dir": str(plan_dir),
            "plan_summary_sha256": sha256(summary_path),
            "plan_lock_sha256": sha256(lock_path),
            "plan_id": summary.get("source_gap_plan"),
            "backend": backend,
            "sco_job_id": job_id,
            "observed_sco_state": observed_state,
            "spec_path": str(spec_path),
            "spec_file_sha256": expected_sha,
            "spec_content_sha256": content_sha,
            "source_gap_plan": spec.get("source_gap_plan"),
            "source_plan_lock_sha256": spec.get("source_plan_lock_sha256"),
            "blocked_worklist": spec.get("blocked_worklist"),
            "blocked_worklist_file_sha256": spec.get("blocked_worklist_file_sha256"),
            "blocked_worklist_content_sha256": spec.get("blocked_worklist_content_sha256"),
            "backend_provenance": spec.get("backend_provenance"),
            "resource": {
                "aec2": aec2,
                "worker_nodes": worker_nodes,
                "worker_spec": worker_spec,
            },
            "sco_argv": actual_argv,
            "worker_command": spec.get("worker_command"),
            "output_receipt": spec.get("receipt"),
            "log_path": spec.get("log"),
        }
        if args.execute:
            receipt.update({
                "submission_returncode": returncode,
                "submission_stdout": stdout,
                "submission_stderr": stderr,
            })
        receipt["content_sha256"] = canonical_sha(receipt)
        atomic_json_new(args.receipt_dir / f"{backend}.json", receipt)
        print(json.dumps({"backend": backend, "job_id": job_id}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
