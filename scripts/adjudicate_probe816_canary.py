#!/usr/bin/env python3
"""Apply explicit BF16 tolerances and summarize probe816 canary resources."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


EXPECTED = {
    "queue_sha256": "ad194b6ea5e53a2dd53ecd5e567b74c324b70d52219ab1a988b8bab69bc91726",
    "overlay_sha256": "3b8bef7a7cbd11dd7a2443be3787f141d49a652525a0955f367762f4b2355404",
    "source_manifest_sha256": "346d60e0f8067efffd2ebdfbf1344ba9984f206a106f89d35f2fb7c350aebcb9",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def equivalent_value(key: str, left: float, right: float) -> tuple[bool, float, float]:
    if math.isnan(left) or math.isnan(right):
        return math.isnan(left) and math.isnan(right), math.nan, math.nan
    if math.isinf(left) or math.isinf(right):
        return left == right, math.inf, math.inf
    absolute = abs(left - right)
    relative = absolute / max(abs(left), abs(right), 1e-12)
    discrete_tokens = ("n_valid", "n_used", "total_samples", "trainer/epoch", "trainer/lr", "top1", "top3", "R@1")
    if any(token in key for token in discrete_tokens):
        return left == right, absolute, relative
    absolute_limit = 0.1 if key.endswith("_deg") else 1e-3
    return absolute <= absolute_limit or relative <= 1e-3, absolute, relative


def gpu_stats(path: Path, intervals: dict[tuple[int, str], tuple[float, float]]) -> dict[str, Any]:
    values: dict[tuple[int, str], list[tuple[float, float]]] = {key: [] for key in intervals}
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            try:
                stamp = datetime.fromisoformat(row["utc"]).timestamp()
                gpu = int(row["index"])
                util = float(row["util_gpu_pct"])
                memory = float(row["memory_used_mib"])
            except (KeyError, TypeError, ValueError):
                continue
            for (wanted_gpu, mode), (start, end) in intervals.items():
                if gpu == wanted_gpu and start <= stamp <= end:
                    values[(wanted_gpu, mode)].append((util, memory))
    result = {}
    for (gpu, mode), rows in values.items():
        result[f"gpu{gpu}:{mode}"] = {
            "samples": len(rows),
            "mean_util_pct": statistics.fmean(value[0] for value in rows) if rows else None,
            "active_sample_fraction": sum(value[0] > 0 for value in rows) / len(rows) if rows else None,
            "max_memory_mib": max((value[1] for value in rows), default=None),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canary-root", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads((args.canary_root / "selection.json").read_text(encoding="utf-8"))
    bindings = {
        "queue_sha256": sha256(args.queue),
        "overlay_sha256": sha256(args.overlay),
        "source_manifest_sha256": sha256(args.source_manifest),
    }
    if bindings != EXPECTED or selection["queue_sha256"] != EXPECTED["queue_sha256"] or selection["overlay_sha256"] != EXPECTED["overlay_sha256"]:
        raise RuntimeError("canary lineage binding mismatch")
    cases = []
    intervals = {}
    all_pass = True
    worker1_eligible = []
    opt_speedups = []
    for gpu, selected in enumerate(selection["cases"]):
        key = selected["unique_key"]
        case_dir = args.canary_root / "cases" / key
        runs = {}
        for mode in ("reference", "optimized", "optimized_worker1"):
            result_path = case_dir / mode / "result.json"
            if not result_path.is_file():
                raise RuntimeError(f"missing result: {result_path}")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            runs[mode] = result
            start = datetime.fromisoformat(result["started_utc"]).timestamp()
            intervals[(gpu, mode)] = (start, start + float(result["elapsed_seconds"]))
        ref, opt, worker1 = (runs[name] for name in ("reference", "optimized", "optimized_worker1"))
        rc_ok = all(run["returncode"] == 0 for run in runs.values())
        if set(ref["metrics"]) != set(opt["metrics"]):
            raise RuntimeError(f"metric key mismatch for {key}")
        comparisons = {}
        numeric_ok = True
        for metric in sorted(ref["metrics"]):
            ok, absolute, relative = equivalent_value(metric, float(ref["metrics"][metric]), float(opt["metrics"][metric]))
            comparisons[metric] = {"pass": ok, "absolute_error": absolute, "relative_error": relative}
            numeric_ok &= ok
        worker_keys_ok = set(opt["metrics"]) == set(worker1["metrics"])
        worker_numeric_ok = worker_keys_ok
        if worker_keys_ok:
            for metric in opt["metrics"]:
                worker_numeric_ok &= equivalent_value(
                    metric, float(opt["metrics"][metric]), float(worker1["metrics"][metric])
                )[0]
        speedup = float(ref["elapsed_seconds"]) / float(opt["elapsed_seconds"])
        worker_speedup = float(opt["elapsed_seconds"]) / float(worker1["elapsed_seconds"])
        opt_speedups.append(speedup)
        worker1_eligible.append(worker_numeric_ok and worker_speedup >= 1.05)
        passed = rc_ok and numeric_ok
        all_pass &= passed
        cases.append({
            "gpu": gpu, "unique_key": key, "pass": passed, "returncodes_ok": rc_ok,
            "reference_vs_optimized": comparisons,
            "elapsed_seconds": {name: value["elapsed_seconds"] for name, value in runs.items()},
            "cold_to_warm_speedup": speedup,
            "worker1_vs_worker0_speedup": worker_speedup,
            "worker1_numeric_equivalent": worker_numeric_ok,
        })
    gpu = gpu_stats(args.canary_root / "gpu_samples.csv", intervals)
    # This sample was taken in-container during the optimized phase with vmstat
    # and free; it is intentionally recorded as a point observation, not an
    # invented phase average.
    host_sample = {
        "utc": "2026-09-19T11:23:11Z", "load_average_1m": 4.64,
        "memory_total_gib": 2015, "memory_used_gib": 34, "memory_available_gib": 1970,
        "vmstat_three_samples": [
            {"cpu_user_pct": 8, "cpu_system_pct": 2, "cpu_idle_pct": 89, "io_wait_pct": 0, "block_in_kib_s": 54, "block_out_kib_s": 62},
            {"cpu_user_pct": 2, "cpu_system_pct": 1, "cpu_idle_pct": 96, "io_wait_pct": 0, "block_in_kib_s": 0, "block_out_kib_s": 52},
            {"cpu_user_pct": 2, "cpu_system_pct": 1, "cpu_idle_pct": 97, "io_wait_pct": 0, "block_in_kib_s": 0, "block_out_kib_s": 28},
        ],
    }
    payload = {
        "schema": "MAIN_V2_PROBE816_CANARY_ADJUDICATION_V1",
        "status": "PASS" if all_pass else "FAIL",
        "bindings": bindings,
        "cases": cases,
        "performance": {
            "median_cold_reference_to_warm_optimized_speedup": statistics.median(opt_speedups),
            "interpretation": "includes first-cache versus warm-cache cost; not attributed solely to code",
            "gpu_samples": gpu,
            "host_point_sample": host_sample,
        },
        "dataloader_decision": {
            "num_workers": 1 if all(worker1_eligible) else 0,
            "enabled": all(worker1_eligible),
            "reason": "enabled only if all eight cases preserve numeric behavior and improve elapsed time by at least 5%",
        },
        "tolerances": {
            "discrete_and_count_metrics": "exact including paired NaN",
            "angular_degrees_absolute": 0.1,
            "other_float_absolute_or_relative": 1e-3,
            "rationale": "two independent bf16-mixed H800 runs; thresholds are below task-semantic resolution and reported per metric",
        },
    }
    if args.output.exists():
        raise RuntimeError(f"output exists: {args.output}")
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True))
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())

