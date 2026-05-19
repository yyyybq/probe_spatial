"""Aggregate eval/{split}_summary.json across multiple runs into a comparison table.

Usage:
    python vidfm3d/eval_diag_compare.py \
        --runs logs/inscene15k_ext_view_consistency_wan_v1 \
               logs/inscene15k_ext_view_consistency_cogvideox_v1 \
               logs/inscene15k_ext_view_consistency_vjepa2_v1 \
        --split val \
        --output comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True,
                   help="List of run directories (each containing eval/{split}_summary.json)")
    p.add_argument("--split", default="val")
    p.add_argument("--output", default="comparison.csv")
    return p.parse_args()


def main():
    args = parse_args()
    rows = []
    keys = set()
    for run in args.runs:
        cands = glob.glob(os.path.join(run, "**", f"{args.split}_summary.json"),
                          recursive=True)
        if not cands:
            sys.stderr.write(f"[WARN] no summary found in {run}\n")
            continue
        # Take the most recent one
        cands.sort(key=os.path.getmtime, reverse=True)
        with open(cands[0]) as f:
            data = json.load(f)
        data["run"] = os.path.basename(run.rstrip("/"))
        rows.append(data)
        keys.update(data.keys())

    if not rows:
        sys.stderr.write("No rows to write.\n")
        sys.exit(1)

    keys = ["run", "probe_type"] + sorted(k for k in keys if k not in {"run", "probe_type"})
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {args.output} with {len(rows)} rows")


if __name__ == "__main__":
    main()
