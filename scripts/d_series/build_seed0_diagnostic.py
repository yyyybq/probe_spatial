#!/usr/bin/env python3
"""Build the development-only seed0 V-JEPA2 D1/D3 diagnostic tables."""

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
from vidfm3d.utils.layout_metrics import scene_cluster_paired_bootstrap, scene_macro


def load(root: Path):
    records, sources = [], []
    candidates = []
    for path in sorted(root.rglob("per_sample_metrics.csv")):
        receipt_path = path.parent / "run_receipt.json"
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("status") != "complete_development_evaluation" or receipt.get("manifest_sha256") != MANIFEST_SHA256:
            raise ValueError(f"invalid diagnostic receipt: {receipt_path}")
        if receipt.get("seed") != 0 or receipt.get("model") != "vjepa2_vitl256":
            continue
        candidates.append((path, receipt_path, receipt))
    has_full_d3 = any(receipt.get("experiment") == "D3" and receipt.get("phase") != "AB"
                      for _, _, receipt in candidates)
    for path, receipt_path, receipt in candidates:
        if has_full_d3 and receipt.get("experiment") == "D3" and receipt.get("phase") == "AB":
            continue
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        records.extend(rows); sources.append({"metrics": str(path), "receipt": str(receipt_path)})
    return records, sources


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    records, sources = load(args.run_root)
    if not records:
        raise ValueError("no completed seed0 diagnostic records")
    rows_out = []
    grouped = defaultdict(list)
    for row in records:
        if row["split"] != "dev" or row["manifest_sha256"] != MANIFEST_SHA256:
            raise ValueError("non-frozen diagnostic record")
        key = (row["experiment"], row.get("reference", ""), row["branch"], int(row["horizon"]))
        grouped[key].append(row)
    for (experiment, reference, branch, horizon), group in sorted(grouped.items()):
        conditions = sorted({row["condition"] for row in group})
        required = ({"true_target", "train_layout_prior", "within_scene_wrong_target", "last_context"}
                    if experiment == "D1" else
                    {"native_target_reference", "frozen_readout_transfer", "train_layout_prior", "within_scene_wrong_target", "last_context"})
        if not required.issubset(conditions):
            raise ValueError(f"missing diagnostic conditions: {(experiment, reference, branch, horizon)}")
        for condition in conditions:
            selected = [row for row in group if row["condition"] == condition]
            if len(selected) != 276 or len({row["scene_id"] for row in selected}) != 69:
                raise ValueError(f"incomplete h{horizon} condition: {(experiment, reference, branch, condition)}")
            comparator = ("train_layout_prior" if experiment == "D1" else "native_target_reference")
            ci = scene_cluster_paired_bootstrap(group, condition, comparator, n_boot=10_000, seed=20270301)
            result = {
                "label": "development-only / seed0 / diagnostic", "experiment": experiment,
                "reference": reference, "branch": branch, "horizon": horizon, "condition": condition,
                "scene_macro_mae": scene_macro(selected, "mae"),
                "scene_macro_absrel": scene_macro(selected, "absrel"),
                "scene_macro_delta1": scene_macro(selected, "delta1"),
                "paired_comparator": comparator, "delta_mae": ci["delta"],
                "delta_ci_low": ci["ci_low"], "delta_ci_high": ci["ci_high"],
                "n": len(selected), "scenes": 69, "seed": 0, "split": "dev",
                "manifest_sha256": MANIFEST_SHA256,
            }
            if experiment == "D1" and condition == "true_target":
                wrong = scene_cluster_paired_bootstrap(group, condition, "within_scene_wrong_target", n_boot=10_000, seed=20270301)
                result |= {"delta_vs_wrong": wrong["delta"], "wrong_ci_low": wrong["ci_low"], "wrong_ci_high": wrong["ci_high"]}
            rows_out.append(result)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    fields = list(dict.fromkeys(key for row in rows_out for key in row))
    with (args.output_dir / "seed0_diagnostic.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows_out)
    summary = {"schema_version": 1, "status": "COMPLETE_DIAGNOSTIC", "label": "development-only / seed0 / diagnostic",
               "manifest_sha256": MANIFEST_SHA256, "rows": rows_out, "sources": sources}
    (args.output_dir / "seed0_diagnostic.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    lines = ["# V-JEPA2 seed0 D1/D3 diagnostic", "", "**development-only / seed0 / diagnostic — not a formal three-seed table.**", "",
             "| experiment | reference | branch | h | condition | scene-macro MAE | AbsRel | Delta1 | paired MAE delta [95% CI] |",
             "|---|---|---:|---:|---|---:|---:|---:|---:|"]
    for row in rows_out:
        lines.append(f"| {row['experiment']} | {row['reference']} | {row['branch']} | {row['horizon']} | {row['condition']} | "
                     f"{row['scene_macro_mae']:.6f} | {row['scene_macro_absrel']:.6f} | {row['scene_macro_delta1']:.6f} | "
                     f"{row['delta_mae']:.6f} [{row['delta_ci_low']:.6f}, {row['delta_ci_high']:.6f}] |")
    (args.output_dir / "seed0_diagnostic.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": summary["status"], "table_rows": len(rows_out), "output": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
