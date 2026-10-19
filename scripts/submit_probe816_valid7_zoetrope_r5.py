#!/usr/bin/env python3
"""Submit the three validated non-MiniMax probe816 shards to Zoetrope."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/mnt/umm/users/yinbaiqiao/probe_spatial")
QUEUE_ROOT = ROOT / "artifacts/main_v2_probe_v2_816_20260917/submission_queues_r2_minimax_quarantine"
MAPPING_ROOT = ROOT / "artifacts/main_v2_probe_v2_816_20260917/submission_mapping_r2_valid7"
LEDGER = MAPPING_ROOT / "zoetrope_r6_submission_ledger.json"
SCO = "/mnt/umm/users/yinbaiqiao/.sco/bin/sco"
RUN_ID = "main_v2_probe_v2_r5_valid7_20260920_r6"
WRAPPER = "runtime/main_v2_probe_v2_r5/run_probe816_valid7_queue_v6.sh"
WRAPPER_SHA256 = "23fba7cdc4e6563b0515fba0a4c472796430e9d904bb8eb3ebb54ff3037c8d93"

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
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o644)
    Path(temporary).replace(path)


def validate_queues() -> dict[str, dict]:
    if file_sha(ROOT / WRAPPER) != WRAPPER_SHA256:
        raise SystemExit("refusing: immutable valid7 wrapper SHA mismatch")
    seen: set[str] = set()
    validated = {}
    for shard, spec in SHARDS.items():
        queue = QUEUE_ROOT / f"{shard}.eligible.jsonl"
        actual_sha = file_sha(queue)
        rows = []
        for line in queue.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("model") == "minimax_h3":
                raise SystemExit(f"refusing: MiniMax row present in {queue}")
            key = row.get("unique_key") or row.get("experiment_key")
            if not key:
                raise SystemExit(f"refusing: row lacks unique experiment key in {queue}")
            if key in seen:
                raise SystemExit(f"refusing: duplicate experiment key across shards: {key}")
            seen.add(key)
            rows.append(key)
        if actual_sha != spec["sha256"] or len(rows) != spec["rows"]:
            raise SystemExit(
                f"binding mismatch for {shard}: sha={actual_sha} rows={len(rows)}"
            )
        validated[shard] = {"queue": queue, "keys": rows, "sha256": actual_sha}
    if len(seen) != 361:
        raise SystemExit(f"refusing: expected 361 unique valid7 experiments, found {len(seen)}")
    return validated


def main() -> None:
    validated = validate_queues()
    if LEDGER.exists():
        ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
        if ledger.get("run_id") != RUN_ID:
            raise SystemExit(f"refusing incompatible existing ledger {LEDGER}")
    else:
        ledger = {
            "schema": "PROBE816_VALID7_ZOETROPE_SUBMISSIONS_V1",
            "run_id": RUN_ID,
            "source_pack": str(ROOT / "runtime/main_v2_probe_v2_r5"),
            "source_manifest_sha256": "346d60e0",
            "quarantined_model": "minimax_h3",
            "resource_pool": "zoetrope",
            "worker_spec": "N4lS.Iq.I80.8",
            "workers_per_node": 8,
            "total_unique_experiments": 361,
            "jobs": {},
        }

    for shard, spec in SHARDS.items():
        if shard in ledger["jobs"]:
            print(f"SKIP {shard} already={ledger['jobs'][shard]['job_id']}")
            continue
        binding = validated[shard]
        queue = binding["queue"]
        worker_shard = f"{shard}_valid7_zoe_r5"
        startup = (
            f"cd {ROOT} && bash {WRAPPER} {queue} {binding['sha256']} "
            f"{RUN_ID} {worker_shard} 0"
        )
        argv = [
            SCO,
            "acp", "jobs", "create",
            "--workspace-name=aigc",
            "--aec2-name=zoetrope",
            f"--job-name=bq-v2p816-{shard[-2:]}-v7-zoe-r6",
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
            ledger.setdefault("failed_attempts", []).append(
                {
                    "shard": shard,
                    "attempted_utc": datetime.now(timezone.utc).isoformat(),
                    "returncode": completed.returncode,
                    "output": output,
                    "sco_argv": argv,
                }
            )
            atomic_json(LEDGER, ledger)
            raise SystemExit(f"submit failed {shard} rc={completed.returncode}: {output}")
        job_id = match.group(1)
        ledger["jobs"][shard] = {
            "job_id": job_id,
            "replaces_job_id": spec["replaces"],
            "queue_path": str(queue),
            "queue_sha256": binding["sha256"],
            "execution_units": len(binding["keys"]),
            "experiment_keys": binding["keys"],
            "startup_command": startup,
            "sco_argv": argv,
            "submitted_utc": datetime.now(timezone.utc).isoformat(),
        }
        ledger["updated_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(LEDGER, ledger)
        print(f"SUBMITTED {shard} {job_id}")


if __name__ == "__main__":
    main()
