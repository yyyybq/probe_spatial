#!/usr/bin/env python3
"""Fail-closed role preflight for the shared DINO science-pilot request."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


TAG = "independent_actionability_algebra_p8_20260831T091224Z_28bda217"
ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "artifacts" / TAG / "manifests"


def rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    isolated = {}
    contexts = {}
    for unit in rows(MANIFESTS / "science_pilot_subset_v1_feature_request_v2_dino_l0.jsonl"):
        if unit["feature_role"] == "isolated_frame":
            previous = isolated.get(unit["raw_frame_id"])
            if previous and previous != unit["unit_id"]:
                raise SystemExit(f"ambiguous isolated raw ID {unit['raw_frame_id']}")
            isolated[unit["raw_frame_id"]] = unit["unit_id"]
        elif unit["feature_role"] == "streaming_context_p8":
            key = tuple(unit["context_raw_ids"])
            previous = contexts.get(key)
            if previous and previous != unit["unit_id"]:
                raise SystemExit(f"ambiguous p8 context {key}")
            contexts[key] = unit["unit_id"]
    missing = {}
    names = ("actionability_train", "actionability_val", "algebra_train", "algebra_val", "algebra_real_loops")
    for name in names:
        for row in rows(MANIFESTS / f"science_pilot_subset_v1_{name}.jsonl"):
            raw_ids = [row["anchor_raw_id"]]
            if name.startswith("actionability"):
                raw_ids.extend((row["goal_raw_id"], row["candidate_endpoint_raw_id_1"], row["candidate_endpoint_raw_id_2"]))
                context_required = tuple(row["context_raw_ids"]) not in contexts
            elif name.startswith("algebra_train") or name.startswith("algebra_val"):
                raw_ids.extend(row["horizon_target_raw_ids"].values())
                context_required = False
            else:
                raw_ids.append(row["final_raw_id"])
                context_required = False
            absent = sorted(raw for raw in raw_ids if raw not in isolated)
            if context_required:
                absent.append("ordered_p8_context")
            if absent:
                missing[row["sample_id"]] = absent
    if missing:
        raise SystemExit(json.dumps({"status": "FAILED_VALIDATION", "missing": missing}, indent=2))
    print(json.dumps({"status": "COMPLETE_VALID", "isolated_raw_ids": len(isolated), "ordered_p8_contexts": len(contexts)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
