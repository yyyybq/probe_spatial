#!/usr/bin/env python3
"""Write fresh exact scheduler descriptions for an explicit job-ID list."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
import subprocess
from datetime import datetime, timezone


SCO = "/mnt/umm/users/yinbaiqiao/.sco/bin/sco"


def describe(job_id):
    proc = subprocess.run(
        [SCO, "acp", "jobs", "describe", "--workspace-name=aigc", "-o", "json", job_id],
        check=True, text=True, stdout=subprocess.PIPE,
    )
    return json.loads(proc.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", action="append", required=True)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(args.job_id))) as pool:
        jobs = list(pool.map(describe, args.job_id))
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "jobs": jobs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({job["name"]: job["state"] for job in jobs}, indent=2))


if __name__ == "__main__":
    main()
