#!/usr/bin/env python3
"""Write one immutable local receipt for an SCO submission."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_create(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--science-tag", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--submit-command", required=True)
    parser.add_argument("--request-manifest", type=Path, required=True)
    parser.add_argument("--entry-script", type=Path, required=True)
    parser.add_argument("--extractor", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--aec2", required=True)
    parser.add_argument("--worker-nodes", type=int, required=True)
    parser.add_argument("--nproc-per-node", type=int, required=True)
    args = parser.parse_args()
    for path in (args.request_manifest, args.entry_script, args.extractor):
        if not path.is_file():
            raise FileNotFoundError(path)
    payload = {
        "schema_version": 1,
        "science_tag": args.science_tag,
        "status": "SUBMITTED_UNVERIFIED",
        "submitted_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "sco_job_id": args.job_id,
        "submit_command": args.submit_command,
        "request_manifest": str(args.request_manifest.resolve()),
        "request_manifest_sha256": sha256(args.request_manifest),
        "entry_script": str(args.entry_script.resolve()),
        "entry_script_sha256": sha256(args.entry_script),
        "extractor": str(args.extractor.resolve()),
        "extractor_sha256": sha256(args.extractor),
        "output_root": str(args.output_root.resolve()),
        "resources": {
            "aec2": args.aec2,
            "worker_nodes": args.worker_nodes,
            "nproc_per_node": args.nproc_per_node,
        },
        "worker_receipt_expected": "coverage_receipt.json",
    }
    atomic_create(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
