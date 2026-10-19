#!/usr/bin/env python3
"""Supplement frozen closure with overlap and loop-match diagnostics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def raw_set(row: dict, task: str) -> set[str]:
    if task == "actionability":
        return set(row["context_raw_ids"] + [row["anchor_raw_id"], row["goal_raw_id"],
                    row["candidate_endpoint_raw_id_1"], row["candidate_endpoint_raw_id_2"]])
    return set([row["anchor_raw_id"], *row["waypoint_raw_ids"]])


def overlap_audit(rows: list[dict], task: str) -> dict:
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_scene[row["scene_id"]].append(row)
    pairs = []; exact = near_80 = near_50 = 0; maximum = 0.0
    for scene_rows in by_scene.values():
        values = [(row["sample_id"], raw_set(row, task)) for row in scene_rows]
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                score = len(values[i][1] & values[j][1]) / len(values[i][1] | values[j][1])
                maximum = max(maximum, score)
                exact += int(score == 1); near_80 += int(.8 <= score < 1); near_50 += int(.5 <= score < .8)
                if score >= .8:
                    pairs.append({"sample_id_1": values[i][0], "sample_id_2": values[j][0], "jaccard": score})
    return {"definition": "Jaccard overlap of task-relevant exact raw IDs within the same scene",
            "record_pairs_compared": sum(len(v) * (len(v) - 1) // 2 for v in by_scene.values()),
            "exact_set_pairs": exact, "near_pairs_jaccard_0.8_to_below_1": near_80,
            "overlap_pairs_jaccard_0.5_to_below_0.8": near_50, "maximum_nonidentical_jaccard": maximum,
            "near_pair_details": pairs}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    if args.output.exists(): raise SystemExit(f"refusing to overwrite {args.output}")
    files = {
        "actionability_train": ("actionability_p8_independent_v1_train.jsonl", "actionability"),
        "actionability_validation": ("actionability_p8_independent_v1_val.jsonl", "actionability"),
        "algebra_train": ("algebra_p8_independent_v1_train.jsonl", "algebra"),
        "algebra_validation": ("algebra_p8_independent_v1_val.jsonl", "algebra"),
    }
    result = {name: overlap_audit(read(args.source_dir / filename), task) for name, (filename, task) in files.items()}
    loops = read(args.source_dir / "algebra_p8_independent_v1_real_loops.jsonl")
    indexed = {row["sample_id"]: row for row in loops}; seen = set(); deltas = []; ratios = []
    for row in loops:
        pair = tuple(sorted((row["sample_id"], row["matched_non_loop_id"])))
        if pair in seen: continue
        seen.add(pair); other = indexed[row["matched_non_loop_id"]]
        a, b = float(row["path_length"]), float(other["path_length"])
        deltas.append(abs(a - b)); ratios.append(max(a, b) / min(a, b))
    result["loop_path_matching"] = {"pairs": len(seen), "absolute_delta_median": np.median(deltas),
        "absolute_delta_p90": np.quantile(deltas, .9), "absolute_delta_max": max(deltas),
        "length_ratio_median": np.median(ratios), "length_ratio_p90": np.quantile(ratios, .9),
        "length_ratio_max": max(ratios)}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); print(json.dumps(result["loop_path_matching"], indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
