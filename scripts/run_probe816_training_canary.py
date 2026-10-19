#!/usr/bin/env python3
"""Run and verify one frozen probe816 short training canary."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def fail(message: str) -> None:
    raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--expected-case-sha256", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    args = parser.parse_args()
    if digest(Path(__file__).resolve()) != args.expected_runner_sha256:
        fail("runner SHA mismatch")
    case_path = args.case.resolve()
    if digest(case_path) != args.expected_case_sha256:
        fail("case SHA mismatch")
    case = json.loads(case_path.read_text())
    case_dir = case_path.parent
    result_path = case_dir / "result.json"
    if result_path.exists():
        fail(f"refusing to overwrite canary result: {result_path}")
    started = now()
    started_clock = time.monotonic()
    result: dict[str, Any] = {
        "schema": "PROBE816_RECEIPT_UNION_TRAINING_CANARY_RESULT_V1",
        "model": case["model"],
        "unique_key": case["unique_key"],
        "case_sha256": args.expected_case_sha256,
        "started_at_utc": started,
        "status": "FAIL",
    }
    try:
        overlay = Path(case["overlay_manifest"])
        manifest = Path(case["manifest"])
        runtime = Path(case["runtime"])
        if digest(overlay) != case["overlay_manifest_sha256"]:
            fail("overlay SHA mismatch")
        if digest(manifest) != case["manifest_sha256"]:
            fail("canary sample manifest SHA mismatch")
        if not runtime.is_dir() or runtime.is_symlink():
            fail("runtime source is missing or unsafe")
        gpu_probe = subprocess.run(
            [str(case["command"][0]), "-c", "import torch; print(torch.cuda.device_count())"],
            cwd=runtime, text=True, capture_output=True, check=False,
        )
        if gpu_probe.returncode != 0 or gpu_probe.stdout.strip() != "1":
            fail(f"expected exactly one visible CUDA device: {gpu_probe.stdout!r} {gpu_probe.stderr!r}")
        environment = os.environ.copy()
        environment.update({str(k): str(v) for k, v in case["environment"].items()})
        environment["AOSS_CACHE_DIR"] = str((case_dir / "aoss_cache").resolve())
        environment["AOSS_FS_METRICS"] = str((case_dir / "aoss_metrics.json").resolve())
        environment["AOSS_FS_METRICS_MODE"] = "summary"
        log_path = case_dir / "train.log"
        with log_path.open("x") as log:
            process = subprocess.run(
                [str(value) for value in case["command"]], cwd=runtime,
                env=environment, stdout=log, stderr=subprocess.STDOUT,
                text=True, check=False,
            )
        result["returncode"] = process.returncode
        result["train_log"] = str(log_path)
        result["train_log_sha256"] = digest(log_path)
        if process.returncode != 0:
            fail(f"training command returned {process.returncode}")
        logs = log_path.read_text(errors="replace")
        optimizer_marker = "max_steps=1" in logs and "Trainer.fit" in logs
        metric_paths = list((case_dir / "output").glob("**/metrics.csv"))
        if len(metric_paths) != 1:
            fail(f"expected one metrics.csv, found {len(metric_paths)}")
        metric_path = metric_paths[0]
        rows = list(csv.DictReader(metric_path.open()))
        finite_losses: list[float] = []
        total_samples: list[float] = []
        for row in rows:
            value = row.get("train/loss")
            if value not in (None, ""):
                number = float(value)
                if math.isfinite(number):
                    finite_losses.append(number)
            sample_value = row.get("trainer/total_samples")
            if sample_value not in (None, ""):
                number = float(sample_value)
                if math.isfinite(number):
                    total_samples.append(number)
        if not finite_losses:
            fail("no finite train/loss was logged")
        if not total_samples or max(total_samples) < 2:
            fail("training did not consume a full production-size batch")
        if not optimizer_marker:
            fail("Lightning did not report reaching max_steps=1")
        result.update({
            "status": "PASS",
            "forward": "PASS",
            "loss": "PASS",
            "optimizer_step": "PASS",
            "finite_train_losses": finite_losses,
            "max_total_samples": max(total_samples),
            "metrics_path": str(metric_path),
            "metrics_sha256": digest(metric_path),
            "sample_ids": [item["sample_id"] for item in case["samples"]],
            "sample_owner_origins": sorted({item["owner_origin"] for item in case["samples"]}),
        })
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["ended_at_utc"] = now()
    result["elapsed_seconds"] = time.monotonic() - started_clock
    atomic_json(result_path, result)
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
