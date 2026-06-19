#!/usr/bin/env python3
"""Aggregate eval summaries across `_seedN` runs into mean/std tables."""

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_root")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="seed_summary.csv")
    args = parser.parse_args()
    groups = defaultdict(list)
    for path in Path(args.runs_root).rglob(f"{args.split}_summary.json"):
        data = json.loads(path.read_text())
        job = data.get("job_name", path.parents[1].name)
        group = re.sub(r"_seed\d+$", "", job)
        groups[group].append(data)
    rows = []
    for group, runs in sorted(groups.items()):
        numeric = sorted(set.intersection(*[
            {k for k, v in run.items() if isinstance(v, (int, float)) and math.isfinite(v)}
            for run in runs
        ]))
        row = {"group": group, "n_seeds": len(runs)}
        for key in numeric:
            values = [float(run[key]) for run in runs]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1)
            row[f"{key}_mean"] = mean
            row[f"{key}_std"] = math.sqrt(variance)
        rows.append(row)
    keys = sorted({key for row in rows for key in row})
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}: groups={len(rows)}")


if __name__ == "__main__":
    main()
