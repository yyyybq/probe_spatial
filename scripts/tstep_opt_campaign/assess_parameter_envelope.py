#!/usr/bin/env python3
"""Accept optimized CUDA variation only inside an observed old-vs-old envelope."""

from __future__ import annotations

import argparse
import json
import pathlib


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--factor", type=float, default=1.01)
    parser.add_argument("--absolute-slack", type=float, default=1e-7)
    args = parser.parse_args()
    rows, failures = [], []
    grouped = {}
    for baseline_path in sorted(args.directory.glob("old_repeat*__*.json")):
        suffix = baseline_path.name.split("__", 1)[1]
        grouped.setdefault(suffix, []).append(baseline_path)
    for suffix, baseline_paths in sorted(grouped.items()):
        candidate_path = args.directory / f"optimized__{suffix}"
        if not candidate_path.exists():
            failures.append(f"missing optimized report for {suffix}")
            continue
        baselines = [json.loads(path.read_text()) for path in baseline_paths]
        candidate = json.loads(candidate_path.read_text())
        limits = {
            metric: max(baseline[metric] for baseline in baselines) * args.factor + args.absolute_slack
            for metric in ("max_abs", "l2", "mean_abs")
        }
        checks = {metric: candidate[metric] <= limit for metric, limit in limits.items()}
        if not all(checks.values()):
            failures.append(f"{suffix}: optimized variation exceeds old-repeat envelope")
        rows.append({
            "case": suffix[:-5], "checks": checks, "limits": limits,
            "old_repeat_count": len(baselines),
            "old_repeat_envelope": {
                key: max(baseline[key] for baseline in baselines)
                for key in ("max_abs", "l2", "relative_l2", "mean_abs")
            },
            "optimized": {key: candidate[key] for key in ("max_abs", "l2", "relative_l2", "mean_abs")},
        })
    report = {
        "schema": "tstep-probe-parameter-envelope-v1", "factor": args.factor,
        "absolute_slack": args.absolute_slack, "passed": not failures,
        "rationale": "Independent old runs are not bit deterministic on H800; optimized numerical variation must not exceed the measured old-vs-old envelope for max_abs, L2, or mean_abs.",
        "failures": failures, "results": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
