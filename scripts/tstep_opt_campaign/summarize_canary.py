#!/usr/bin/env python3
"""Aggregate paired canary timing, tensor-volume, memory, and logging results."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics


def pct_change(old: float, new: float) -> float:
    return 100.0 * (new - old) / old if old else 0.0


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def read_system_metrics(suite: pathlib.Path) -> dict:
    dmon_rows = []
    dmon = suite / "metrics" / "nvidia_dmon.log"
    if dmon.exists():
        for line in dmon.read_text().splitlines():
            fields = line.split()
            if line.lstrip().startswith("#") or len(fields) < 22:
                continue
            try:
                dmon_rows.append([float(value) for value in fields])
            except ValueError:
                continue
    vmstat_rows = []
    vmstat = suite / "metrics" / "vmstat.log"
    if vmstat.exists():
        for line in vmstat.read_text().splitlines():
            fields = line.split()
            if len(fields) != 17:
                continue
            try:
                vmstat_rows.append([float(value) for value in fields])
            except ValueError:
                continue
    return {
        "suite": str(suite),
        "gpu_samples": len(dmon_rows),
        "gpu_sm_mean_pct": mean([row[4] for row in dmon_rows]),
        "gpu_sm_active_mean_pct": mean([row[4] for row in dmon_rows if row[4] > 0]),
        "gpu_sm_max_pct": max((row[4] for row in dmon_rows), default=0.0),
        "gpu_fb_max_mib": max((row[14] for row in dmon_rows), default=0.0),
        "gpu_pcie_rx_mean_mib_s": mean([row[20] for row in dmon_rows]),
        "gpu_pcie_tx_mean_mib_s": mean([row[21] for row in dmon_rows]),
        "vmstat_samples": len(vmstat_rows),
        "cpu_busy_mean_pct": mean([row[12] + row[13] for row in vmstat_rows]),
        "cpu_busy_max_pct": max((row[12] + row[13] for row in vmstat_rows), default=0.0),
        "io_wait_mean_pct": mean([row[15] for row in vmstat_rows]),
        "io_wait_max_pct": max((row[15] for row in vmstat_rows), default=0.0),
        "free_memory_min_kib": min((row[3] for row in vmstat_rows), default=0.0),
        "block_in_max_kib_s": max((row[8] for row in vmstat_rows), default=0.0),
        "block_out_max_kib_s": max((row[9] for row in vmstat_rows), default=0.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", action="append", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    rows = []
    log_bytes = {"old": 0, "optimized": 0}
    system_metrics = []
    for suite in args.suite:
        system_metrics.append(read_system_metrics(suite))
        cases = suite / "cases"
        for old_path in sorted(cases.glob("old__*.json")):
            optimized_path = cases / ("optimized__" + old_path.name[len("old__"):])
            if not optimized_path.exists():
                continue
            old, new = json.loads(old_path.read_text()), json.loads(optimized_path.read_text())
            old_data = statistics.mean(old["data_seconds_cold_then_steady"][1:])
            new_data = statistics.mean(new["data_seconds_cold_then_steady"][1:])
            old_h2d = statistics.mean(old["h2d_ms"]) if old["h2d_ms"] else 0.0
            new_h2d = statistics.mean(new["h2d_ms"]) if new["h2d_ms"] else 0.0
            old_compute = statistics.mean(old["gpu_compute_ms"]) if old["gpu_compute_ms"] else 0.0
            new_compute = statistics.mean(new["gpu_compute_ms"]) if new["gpu_compute_ms"] else 0.0
            old_e2e = old_data + old_h2d / 1000.0 + old_compute / 1000.0
            new_e2e = new_data + new_h2d / 1000.0 + new_compute / 1000.0
            rows.append({
                "case": old_path.stem[len("old__"):], "task": old["task"],
                "backend": old["backend"], "split": old["split"],
                "old_cold_data_s": old["data_seconds_cold_then_steady"][0],
                "new_cold_data_s": new["data_seconds_cold_then_steady"][0],
                "cold_data_change_pct": pct_change(
                    old["data_seconds_cold_then_steady"][0],
                    new["data_seconds_cold_then_steady"][0],
                ),
                "old_steady_data_s": old_data, "new_steady_data_s": new_data,
                "steady_data_change_pct": pct_change(old_data, new_data),
                "old_h2d_ms": old_h2d, "new_h2d_ms": new_h2d,
                "h2d_change_pct": pct_change(old_h2d, new_h2d),
                "old_compute_ms": old_compute, "new_compute_ms": new_compute,
                "compute_change_pct": pct_change(old_compute, new_compute),
                "old_steady_e2e_step_s": old_e2e,
                "new_steady_e2e_step_s": new_e2e,
                "steady_e2e_step_change_pct": pct_change(old_e2e, new_e2e),
                "old_logical_samples_per_s": 1.0 / old_e2e,
                "new_logical_samples_per_s": 1.0 / new_e2e,
                "logical_samples_per_s_change_pct": pct_change(1.0 / old_e2e, 1.0 / new_e2e),
                "old_tensor_bytes": old["sample_tensor_bytes"],
                "new_tensor_bytes": new["sample_tensor_bytes"],
                "tensor_change_pct": pct_change(old["sample_tensor_bytes"], new["sample_tensor_bytes"]),
                "old_peak_memory": old["max_memory_allocated_bytes"],
                "new_peak_memory": new["max_memory_allocated_bytes"],
                "memory_change_pct": pct_change(old["max_memory_allocated_bytes"], new["max_memory_allocated_bytes"]),
            })
        for path in (suite / "metrics").glob("*__aoss.jsonl"):
            key = "optimized" if path.name.startswith("optimized__") else "old"
            log_bytes[key] += path.stat().st_size
    summary = {
        "schema": "tstep-probe-canary-performance-v1", "paired_cases": len(rows),
        "measurement_note": "data time is synchronous dataset service time (the unhidden lower-level component of DataLoader wait); CUDA events cover H2D/compute only and are outside formal training loops. E2E is the sum of steady dataset service, H2D, and GPU compute for one canary logical sample; it does not assume production worker overlap.",
        "median_changes_pct": {
            key: statistics.median(row[key] for row in rows) for key in (
                "cold_data_change_pct", "steady_data_change_pct", "h2d_change_pct", "compute_change_pct",
                "steady_e2e_step_change_pct", "logical_samples_per_s_change_pct",
                "tensor_change_pct", "memory_change_pct"
            )
        },
        "system_metrics": system_metrics,
        "aoss_log_bytes": log_bytes,
        "aoss_log_change_pct": pct_change(log_bytes["old"], log_bytes["optimized"]),
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
