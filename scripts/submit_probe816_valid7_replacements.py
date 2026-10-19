#!/usr/bin/env python3
"""Submit only the reviewed, MiniMax-quarantined replacement shards."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/mnt/umm/users/yinbaiqiao/probe_spatial")
QUEUE_ROOT = ROOT / "artifacts/main_v2_probe_v2_816_20260917/submission_queues_r2_minimax_quarantine"
LEDGER = ROOT / "artifacts/main_v2_probe_v2_816_20260917/submission_mapping_r2_valid7/submission_ledger.json"
SCO = "/mnt/umm/users/yinbaiqiao/.sco/bin/sco"
RUN_ID = "main_v2_probe_v2_r5_valid7_20260919_r2"
WRAPPER = "runtime/main_v2_probe_v2_r5/run_probe816_queue.sh"

SHARDS = {
    "remainder_02": {
        "sha256": "58e3b996e351a3f425f65a3b06f1181fd6c0f86d3fa71cc64b7262814d53c6f1",
        "rows": 118,
        "replaces": "pt-f8qsm780",
    },
    "remainder_03": {
        "sha256": "c4ec1eda97489c4260be36b2565f302e47ad66447ec519b9591209ab7f2fa004",
        "rows": 121,
        "replaces": "pt-uxcegr23",
    },
    "remainder_04": {
        "sha256": "fb5d48656250f115b698be85f754efd42d766204099a7a5497eac2d24726f67b",
        "rows": 122,
        "replaces": "pt-e7vedn9x",
    },
}


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    with open(descriptor, "w", encoding="utf-8", closefd=True) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    Path(temporary).replace(path)


def main() -> None:
    if LEDGER.exists():
        ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    else:
        ledger = {
            "schema": "PROBE816_VALID7_REPLACEMENT_SUBMISSIONS_V1",
            "run_id": RUN_ID,
            "quarantined_model": "minimax_h3",
            "jobs": {},
        }
    for shard, spec in SHARDS.items():
        if shard in ledger["jobs"]:
            print(f"SKIP {shard} already={ledger['jobs'][shard]['job_id']}")
            continue
        queue = QUEUE_ROOT / f"{shard}.eligible.jsonl"
        actual_sha = file_sha(queue)
        rows = sum(1 for line in queue.open() if line.strip())
        if actual_sha != spec["sha256"] or rows != spec["rows"]:
            raise SystemExit(f"binding mismatch for {shard}: sha={actual_sha} rows={rows}")
        worker_shard = f"{shard}_valid7"
        startup = (
            f"cd {ROOT} && bash {WRAPPER} {queue} {actual_sha} "
            f"{RUN_ID} {worker_shard} 0"
        )
        argv = [
            SCO,
            "acp", "jobs", "create",
            "--workspace-name=aigc",
            "--aec2-name=zoetrope",
            f"--job-name=bq-v2p816-{shard[-2:]}-v7-r2",
            "--priority=HIGHEST",
            "--quota-type=reserved",
            "--container-image-url=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617",
            "--storage-mount=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm",
            "--training-framework=pytorch",
            "--worker-nodes=1",
            "--worker-spec=N4lS.Iq.I80.8",
            f"--command={startup}",
        ]
        completed = subprocess.run(argv, text=True, capture_output=True, timeout=120)
        output = (completed.stdout or "") + (completed.stderr or "")
        match = re.search(r"job id\s*:\s*(pt-[a-z0-9]+)", output, re.IGNORECASE)
        if completed.returncode or not match:
            raise SystemExit(f"submit failed {shard} rc={completed.returncode}: {output}")
        job_id = match.group(1)
        ledger["jobs"][shard] = {
            "job_id": job_id,
            "replaces_job_id": spec["replaces"],
            "queue_path": str(queue),
            "queue_sha256": actual_sha,
            "execution_units": rows,
            "startup_command": startup,
            "sco_argv": argv,
            "submitted_utc": datetime.now(timezone.utc).isoformat(),
        }
        ledger["updated_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(LEDGER, ledger)
        print(f"SUBMITTED {shard} {job_id}")


if __name__ == "__main__":
    main()
