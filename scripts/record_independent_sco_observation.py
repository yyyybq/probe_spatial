#!/usr/bin/env python3
"""Persist a timestamped, immutable SCO control-plane observation."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


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
    parser.add_argument("--state", required=True)
    parser.add_argument("--detail", required=True)
    parser.add_argument("--log-path")
    args = parser.parse_args()
    atomic_create(args.output.resolve(), {
        "schema_version": 1,
        "science_tag": args.science_tag,
        "observed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "sco_job_id": args.job_id,
        "state": args.state,
        "detail": args.detail,
        "log_path": args.log_path,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
