#!/usr/bin/env python3
"""Preflight report for a real five-checkpoint feature canary."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .io_utils import write_json
from .protocol import validate_state_bank


def _runtime() -> dict[str, Any]:
    packages = {name: importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "PIL")}
    nvidia = shutil.which("nvidia-smi")
    gpu = {"available": False, "detail": "nvidia-smi unavailable"}
    if nvidia:
        completed = subprocess.run([nvidia, "--query-gpu=name,memory.free", "--format=csv,noheader"], text=True, capture_output=True, check=False)
        gpu = {"available": completed.returncode == 0, "detail": completed.stdout.strip() or completed.stderr.strip()}
    return {"packages": packages, "gpu": gpu, "ready": all(packages.values()) and gpu["available"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--state-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    checkpoint_status = {name: row["status"] for name, row in audit["checkpoints"].items()}
    state = validate_state_bank(args.state_bank) if args.state_bank.is_file() else {"valid": False, "errors": ["state bank not built"]}
    runtime = _runtime()
    failures = []
    if not all(value == "VERIFIED" for value in checkpoint_status.values()):
        failures.append("not every requested exact checkpoint is available")
    if not state["valid"]:
        failures.append("state bank is absent or invalid")
    if not runtime["ready"]:
        failures.append("selected host lacks required ML runtime or visible GPU")
    payload = {
        "status": "READY" if not failures else "BLOCKED",
        "requested_state_count": 48,
        "checkpoint_status": checkpoint_status,
        "state_bank": state,
        "runtime": runtime,
        "failures": failures,
        "next_action": "Run extract_features.py --limit 48 once status is READY; do not launch the 2,000-state MVP before the canary artifacts pass validation.",
    }
    write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.strict and failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
