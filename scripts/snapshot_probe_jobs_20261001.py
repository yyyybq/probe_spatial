#!/usr/bin/env python3
"""Take one complete SCO scheduler snapshot for the probing progress audit."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


SCO = "/mnt/umm/users/yinbaiqiao/.sco/bin/sco"
OUT = Path("/mnt/umm/users/yinbaiqiao/probe_spatial/artifacts/main_v2_probe_v2_816_20260917/probe_progress_20261001")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    jobs = []
    for page in range(1, 21):
        completed = subprocess.run(
            [SCO, "acp", "jobs", "list", "--workspace-name=aigc", "--page-size=100",
             f"--page-token={page}", "-o", "json"],
            text=True, capture_output=True, timeout=90,
        )
        if completed.returncode:
            raise RuntimeError(completed.stdout + "\n" + completed.stderr)
        rows = json.loads(completed.stdout)
        jobs.extend(rows)
        if len(rows) < 100:
            break
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "PROBE_SCO_SNAPSHOT_V1", "observed_utc": now(), "jobs": jobs,
               "job_count": len(jobs)}
    fd, temporary = tempfile.mkstemp(prefix=".scheduler_jobs.", dir=OUT)
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o644)
    os.replace(temporary, OUT / "scheduler_jobs.json")
    print(json.dumps({"observed_utc": payload["observed_utc"], "job_count": len(jobs)}))


if __name__ == "__main__":
    main()
