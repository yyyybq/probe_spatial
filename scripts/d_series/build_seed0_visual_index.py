#!/usr/bin/env python3
"""Select deterministic seed-0 D3 best/median/worst qualitative examples."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.d_series.feature_store import MANIFEST_SHA256


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    grouped = defaultdict(list)
    sources = []
    for metrics_path in sorted(args.run_root.glob("d3_abcd_*/per_sample_metrics.csv")):
        receipt_path = metrics_path.parent / "run_receipt.json"
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("status") != "complete_development_evaluation"
                or receipt.get("experiment") != "D3"
                or receipt.get("model") != "vjepa2_vitl256"
                or receipt.get("seed") != 0
                or receipt.get("manifest_sha256") != MANIFEST_SHA256):
            raise ValueError(f"invalid D3 diagnostic receipt: {receipt_path}")
        with metrics_path.open() as handle:
            for row in csv.DictReader(handle):
                if row["split"] == "dev" and row["condition"] == "frozen_readout_transfer":
                    grouped[(row["branch"], int(row["horizon"]))].append(row)
        sources.append({"metrics": str(metrics_path), "receipt": str(receipt_path)})

    output_rows = []
    for (branch, horizon), rows in sorted(grouped.items()):
        if len(rows) != 276 or len({row["scene_id"] for row in rows}) != 69:
            raise ValueError(f"incomplete qualitative group: {(branch, horizon, len(rows))}")
        ranked = sorted(rows, key=lambda row: (float(row["mae"]), row["sample_id"]))
        for rule, row in (("best", ranked[0]), ("median", ranked[len(ranked) // 2]), ("worst", ranked[-1])):
            output_rows.append({
                "label": "development-only / seed0 / diagnostic",
                "selection_rule": rule,
                "branch": branch,
                "horizon": horizon,
                "sample_id": row["sample_id"],
                "scene_id": row["scene_id"],
                "window_id": row["window_id"],
                "frozen_transfer_mae": row["mae"],
                "manifest_sha256": MANIFEST_SHA256,
            })
    if len(output_rows) != 12:
        raise ValueError(f"expected 12 selections, got {len(output_rows)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    (args.output.with_suffix(".sources.json")).write_text(json.dumps({
        "status": "COMPLETE_DIAGNOSTIC_VISUAL_INDEX",
        "label": "development-only / seed0 / diagnostic",
        "rows": len(output_rows),
        "sources": sources,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "COMPLETE_DIAGNOSTIC_VISUAL_INDEX", "rows": len(output_rows), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
