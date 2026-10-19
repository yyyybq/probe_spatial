#!/usr/bin/env python3
"""Submit idempotent single-GPU MiniMax BF16 production extraction shards."""

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


ROOT = Path("/mnt/umm/users/yinbaiqiao/probe_spatial")
ROUTE = ROOT / "artifacts/main_v2_probe_v2_816_20260917/minimax_bf16_v1/production_route_v1/route_receipt.json"
LAUNCHER = ROOT / "runtime/main_v2_minimax_bf16_v1/run_production_shard_v1.sh"
LEDGER = ROOT / "artifacts/main_v2_probe_v2_816_20260917/minimax_bf16_v1/production_extract_v1/submission_ledger.json"
SCO = "/mnt/umm/users/yinbaiqiao/.sco/bin/sco"
SHARD_COUNT = 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True); handle.write("\n")
        handle.flush(); os.fsync(handle.fileno())
    os.chmod(temporary, 0o644); Path(temporary).replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", default="0", help="comma-separated shard indices")
    args = parser.parse_args()
    requested = sorted({int(value) for value in args.shards.split(",") if value.strip()})
    if any(not 0 <= value < SHARD_COUNT for value in requested):
        raise SystemExit("invalid shard index")
    route = json.loads(ROUTE.read_text())
    if route.get("status") != "PASS" or route.get("scene_count") != 681:
        raise SystemExit("route receipt is not the exact PASS route")
    route_sha = sha256(ROUTE)
    launcher_sha = sha256(LAUNCHER)
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {
        "schema": "MINIMAX_H3_BF16_PRODUCTION_SUBMISSIONS_V1",
        "resource_pool": "zoetrope",
        "worker_spec": "N4lS.Iq.I80.1",
        "shard_count": SHARD_COUNT,
        "route_path": str(ROUTE),
        "route_sha256": route_sha,
        "launcher_path": str(LAUNCHER),
        "launcher_sha256": launcher_sha,
        "jobs": {},
    }
    if ledger.get("route_sha256") != route_sha or ledger.get("launcher_sha256") != launcher_sha:
        raise SystemExit("existing submission ledger binding mismatch")
    for shard in requested:
        key = f"{shard:02d}"
        if key in ledger["jobs"]:
            print(f"SKIP shard={key} job={ledger['jobs'][key]['job_id']}")
            continue
        command = (
            f"cd {ROOT} && env EXPECTED_ROUTE_SHA256={route_sha} "
            f"SHARD_INDEX={shard} SHARD_COUNT={SHARD_COUNT} bash {LAUNCHER}"
        )
        argv = [
            SCO, "acp", "jobs", "create", "--workspace-name=aigc", "--aec2-name=zoetrope",
            f"--job-name=bq-mm-bf16-prod-s{shard:02d}-r1", "--priority=HIGHEST",
            "--quota-type=reserved", "--container-image-url=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617",
            "--storage-mount=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm",
            "--training-framework=pytorch", "--worker-nodes=1", "--worker-spec=N4lS.Iq.I80.1",
            f"--command={command}",
        ]
        result = subprocess.run(argv, text=True, capture_output=True, timeout=120)
        output = (result.stdout or "") + (result.stderr or "")
        match = re.search(r"job id\s*:\s*(pt-[a-z0-9]+)", output, re.I)
        if result.returncode or not match:
            raise SystemExit(f"submission failed shard={key} rc={result.returncode}: {output}")
        ledger["jobs"][key] = {
            "job_id": match.group(1), "submitted_utc": datetime.now(timezone.utc).isoformat(),
            "startup_command": command, "scene_start": shard * 681 // SHARD_COUNT,
            "scene_end": (shard + 1) * 681 // SHARD_COUNT,
        }
        ledger["updated_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(LEDGER, ledger)
        print(f"SUBMITTED shard={key} job={match.group(1)}")


if __name__ == "__main__":
    main()
