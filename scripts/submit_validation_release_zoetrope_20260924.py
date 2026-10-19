#!/usr/bin/env python3
"""Submit one validation shard per GPU and release PASS probes to Zoetrope."""

import argparse
import json
import shlex
from pathlib import Path

from probe_campaign_20260924_followup import BASE, PY, ROOT, create, load, now, save, sha


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--production", type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--validation-only", action="store_true",
                        help="Run SHA-bound canaries without releasing individual shards.")
    parser.add_argument("--pool", choices=("zoetrope", "h800"), default="zoetrope")
    args = parser.parse_args()

    bundle_path = args.bundle.resolve()
    if not args.validation_only and args.production is None:
        parser.error("--production is required unless --validation-only is set")
    production = args.production.resolve() if args.production else None
    bundle = load(bundle_path)
    validation_runner = ROOT / "scripts/run_probe_hybrid_validation.py"
    release_runner = ROOT / "scripts/run_validation_release_20260924.py"
    submitter = ROOT / "scripts/probe_campaign_20260924_followup.py"
    ledger_path = bundle_path.parent / ("validation_submissions.json" if args.validation_only else "submissions.json")
    ledger = load(ledger_path) if ledger_path.exists() else {
        "schema": "VALIDATE_THEN_RELEASE_ZOETROPE_V1",
        "resource_pool": args.pool,
        "jobs": {},
    }

    for shard in range(bundle["shard_count"]):
        key = str(shard)
        if key in ledger["jobs"]:
            continue
        if args.validation_only:
            command = shlex.join([
                PY, str(validation_runner), "--bundle", str(bundle_path), "--shard", key,
                "--bundle-sha256", sha(bundle_path), "--runner-sha256", sha(validation_runner),
            ])
        else:
            command = shlex.join([
                PY, str(release_runner), str(bundle_path), str(production), key,
                sha(bundle_path), sha(validation_runner), sha(submitter),
            ])
        record = {
            "shard": shard,
            "state": "SUBMIT_INTENT",
            "command": command,
            "submitted_utc": now(),
        }
        ledger["jobs"][key] = record
        save(ledger_path, ledger)
        record["job_id"] = create(f"bq-{args.tag}-s{shard:02d}", command, args.pool)
        record["state"] = "SUBMITTED"
        save(ledger_path, ledger)
        print(json.dumps({"shard": shard, "job_id": record["job_id"]}), flush=True)


if __name__ == "__main__":
    main()
