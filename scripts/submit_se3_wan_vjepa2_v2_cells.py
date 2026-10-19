#!/usr/bin/env python3
"""Idempotently submit isolated exact-v2 layer cells."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/mnt/umm/users/yinbaiqiao/probe_spatial")
RESULT = ROOT / "results/experimental/se3_wan_vjepa2_exact_v2_layers_20260916_r1"
RECEIPT = RESULT / "launcher/cell_submissions.json"
SCO = "/mnt/umm/users/yinbaiqiao/.sco/bin/sco"
CELLS = (
    "wan_l15", "vjepa2_l11",
    "wan_l0", "wan_l5", "wan_l10", "wan_l20", "wan_l25", "wan_l29",
    "vjepa2_l0", "vjepa2_l5", "vjepa2_l17", "vjepa2_l23",
)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    with open(descriptor, "w", encoding="utf-8", closefd=True) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    Path(temporary).replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", required=True, help="comma-separated registered cells")
    parser.add_argument("--retry", action="store_true", help="submit a replacement and retain the prior attempt")
    args = parser.parse_args()
    selected = [value.strip() for value in args.cells.split(",") if value.strip()]
    unknown = sorted(set(selected) - set(CELLS))
    if unknown:
        raise SystemExit(f"unregistered cells: {unknown}")
    receipt = json.loads(RECEIPT.read_text()) if RECEIPT.exists() else {"jobs": {}}
    for cell in selected:
        if cell in receipt["jobs"] and not args.retry:
            print(f"SKIP {cell} already={receipt['jobs'][cell]['job_id']}")
            continue
        prior = receipt["jobs"].get(cell)
        name = f"bq-se3-v2-{cell.replace('_', '-')}-{datetime.now(timezone.utc):%m%d%H%M%S}"
        command = f"cd {ROOT} && CELL={cell} bash scripts/run_se3_wan_vjepa2_v2_cell.sh"
        argv = [
            SCO, "acp", "jobs", "create", "--workspace-name=aigc", "--aec2-name=zoetrope",
            f"--job-name={name}", "--priority=HIGHEST", "--quota-type=reserved",
            "--container-image-url=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617",
            "--storage-mount=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm",
            "--training-framework=pytorch", "--worker-nodes=1", "--worker-spec=N4lS.Iq.I80.1",
            f"--command={command}",
        ]
        completed = subprocess.run(argv, text=True, capture_output=True, timeout=120)
        output = (completed.stdout or "") + (completed.stderr or "")
        match = re.search(r"job id\s*:\s*(pt-[a-z0-9]+)", output)
        if completed.returncode or not match:
            raise SystemExit(f"submit failed {cell}: rc={completed.returncode}: {output}")
        receipt["jobs"][cell] = {
            "cell": cell, "job_id": match.group(1), "job_name": name,
            "submitted_utc": datetime.now(timezone.utc).isoformat(), "argv": argv,
            "prior_attempts": ([prior] if prior else []) + (prior.get("prior_attempts", []) if prior else []),
        }
        receipt["updated_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(RECEIPT, receipt)
        print(f"SUBMITTED {cell} {match.group(1)}")


if __name__ == "__main__":
    main()
