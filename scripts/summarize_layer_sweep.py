#!/usr/bin/env python3
"""Aggregate layer-wise eval summaries and report best/default/last scores."""

from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

_FEATURE_LAYERS_PATH = Path(__file__).resolve().parents[1] / "vidfm3d" / "utils" / "feature_layers.py"
_SPEC = importlib.util.spec_from_file_location("feature_layers", _FEATURE_LAYERS_PATH)
feature_layers = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = feature_layers
_SPEC.loader.exec_module(feature_layers)

get_feature_layer_spec = feature_layers.get_feature_layer_spec


PRIMARY_METRIC = {
    "view_consistency": ("balanced_acc", "max"),
    "abnormal": ("pair_acc", "max"),
    "ego_belief": ("mean_az_err_deg", "min"),
    "ego_belief_v2": ("mean_ang_err_deg", "min"),
    "action_dynamics": ("global_R@1", "max"),
    "path_integration": ("global_R@1", "max"),
    "counterfactual": ("intervention_validity", "max"),
}


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _infer_layer(row: dict[str, Any], path: str) -> int | None:
    if row.get("feature_layer") is not None:
        return int(row["feature_layer"])
    for text in (row.get("feat_postfix"), path):
        match = re.search(r"layer(-?\d+)", str(text))
        if match:
            return int(match.group(1))
    return None


def _better(a: dict[str, Any], b: dict[str, Any], metric: str, direction: str) -> bool:
    av = _as_float(a.get(metric))
    bv = _as_float(b.get(metric))
    if math.isnan(av):
        return False
    if math.isnan(bv):
        return True
    return av > bv if direction == "max" else av < bv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="logs/inscene15k_ext/runs")
    parser.add_argument("--pattern", default="*layer*")
    parser.add_argument("--split", default="val")
    parser.add_argument("--vfm", default=None)
    parser.add_argument("--probe", default=None)
    parser.add_argument("--metric", default=None)
    parser.add_argument("--direction", choices=["max", "min"], default=None)
    parser.add_argument("--output", default="layer_sweep.csv")
    args = parser.parse_args()

    summary_paths = glob.glob(
        os.path.join(args.runs_root, args.pattern, "eval", f"{args.split}_summary.json")
    )
    rows = []
    for path in summary_paths:
        with open(path) as f:
            row = json.load(f)
        row["run"] = Path(path).parents[1].name
        row["summary_path"] = path
        row["feature_layer"] = _infer_layer(row, path)
        if args.probe and row.get("probe_type") != args.probe:
            continue
        if args.vfm and args.vfm not in row["run"] and args.vfm not in str(row.get("job_name", "")):
            continue
        rows.append(row)

    if not rows:
        raise SystemExit("No layer summaries matched.")

    probe_type = args.probe or rows[0].get("probe_type")
    metric, default_direction = PRIMARY_METRIC.get(str(probe_type), ("global_R@1", "max"))
    metric = args.metric or metric
    direction = args.direction or default_direction

    rows.sort(key=lambda r: (r["feature_layer"] is None, r["feature_layer"] or 0, r["run"]))
    best = rows[0]
    for row in rows[1:]:
        if _better(row, best, metric, direction):
            best = row

    spec = get_feature_layer_spec(args.vfm) if args.vfm else None
    default_row = None
    last_row = None
    if spec is not None:
        for row in rows:
            if row.get("feature_layer") == spec.default_layer:
                default_row = row
            if spec.last_layer is not None and row.get("feature_layer") == spec.last_layer:
                last_row = row

    keys = ["run", "probe_type", "feature_layer", "feat_postfix", metric]
    extra = sorted({k for row in rows for k in row.keys()} - set(keys) - {"summary_path"})
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys + extra, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.output} with {len(rows)} rows")
    print(f"metric={metric} direction={direction}")
    print(f"best_layer={best.get('feature_layer')} best_score={best.get(metric)} run={best.get('run')}")
    if default_row is not None:
        print(f"default_layer={default_row.get('feature_layer')} default_score={default_row.get(metric)}")
    if last_row is not None:
        print(f"last_layer={last_row.get('feature_layer')} last_score={last_row.get(metric)}")


if __name__ == "__main__":
    main()
