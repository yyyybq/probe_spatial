#!/usr/bin/env python3
"""Audit exact C-probe target-frame coverage for a frozen probing queue."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


TARGET_PROBES = {"action_dynamics", "path_integration"}
DISK_LAYER = {("lingbot", 48): 47}
CANONICAL_SOURCE_SHA = "ebc496eb0587cda372ccf29a7c5244ef454096151f73c8eed9511967ded1902d"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lines(path: Path):
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=Path, required=True)
    parser.add_argument("--canonical-source", type=Path, required=True)
    parser.add_argument("--legacy-routes", type=Path, required=True)
    parser.add_argument("--gap-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"output exists: {args.output}")
    if sha256(args.canonical_source) != CANONICAL_SOURCE_SHA:
        raise RuntimeError("canonical source SHA mismatch")

    with args.experiments.open(newline="", encoding="utf-8") as handle:
        experiment_rows = list(csv.DictReader(handle))
    specs = set()
    for row in experiment_rows:
        if row["probe_config"] not in TARGET_PROBES:
            continue
        model = row["model"]
        logical_layer = int(row["layer"])
        disk_layer = DISK_LAYER.get((model, logical_layer), logical_layer)
        timestep = 250 if model == "lingbot" else 749
        specs.add((model, disk_layer, timestep))

    expected: dict[str, set[int]] = defaultdict(set)
    for row in lines(args.canonical_source):
        scene = str(row["scene_id"])
        window = tuple(int(value) for value in row["streaming_window_indices"])
        for prefix in (8, 12, 16, 24):
            for horizon in (0, 1, 2, 4):
                expected[scene].add(window[prefix - 1 + horizon])
    if len(expected) != 681 or sum(map(len, expected.values())) != 37_677:
        raise RuntimeError("unexpected canonical target universe")

    owners: dict[tuple[str, int, int, str, int], tuple[str, str]] = {}
    duplicate_identical = 0

    def add_owner(key, owner):
        nonlocal duplicate_identical
        prior = owners.get(key)
        if prior is None:
            owners[key] = owner
        elif prior == owner:
            duplicate_identical += 1
        else:
            raise RuntimeError(f"conflicting target owners for {key}: {prior} != {owner}")

    for row in lines(args.legacy_routes):
        spec = (str(row["vfm"]), int(row["layer"]), int(row["timestep"]))
        if spec not in specs:
            continue
        key = (*spec, str(row["scene_id"]), int(row["target_raw_id"]))
        add_owner(key, ("legacy", str(row["source_feature_path"])))

    for row in lines(args.gap_index):
        identity = row.get("identity") or {}
        if row.get("mode") != "target_isolated" or identity.get("payload_role") != "feature":
            continue
        spec = (
            str(row.get("vfm") or identity.get("vfm") or ""),
            int(identity.get("layer", -999)),
            int(identity.get("timestep", -999)),
        )
        if spec not in specs:
            continue
        scene = str(identity["scene"])
        for raw_id in identity.get("target_indices") or ():
            key = (*spec, scene, int(raw_id))
            add_owner(key, ("receipt", f"{row['root'].rstrip('/')}/{row['relative_path']}"))

    per_spec = []
    total_holes = 0
    for spec in sorted(specs):
        missing = []
        covered = 0
        for scene, raw_ids in sorted(expected.items()):
            scene_missing = [raw for raw in sorted(raw_ids) if (*spec, scene, raw) not in owners]
            covered += len(raw_ids) - len(scene_missing)
            if scene_missing:
                missing.append({"scene_id": scene, "raw_frame_ids": scene_missing})
        holes = sum(len(item["raw_frame_ids"]) for item in missing)
        total_holes += holes
        per_spec.append({
            "model": spec[0], "disk_layer": spec[1], "timestep": spec[2],
            "expected_rows": 37_677, "covered_rows": covered, "holes": holes,
            "missing_scenes": len(missing), "missing": missing,
        })

    result = {
        "schema": "PROBE816_VALID7_TARGET_COVERAGE_AUDIT_V1",
        "status": "READY" if total_holes == 0 else "BLOCKED_TARGET_HOLES",
        "zero_duplicate_effective_owners": True,
        "counts": {
            "experiments": len(experiment_rows), "selected_specs": len(specs),
            "expected_rows": len(specs) * 37_677,
            "covered_rows": len(specs) * 37_677 - total_holes,
            "holes": total_holes,
            "identical_duplicate_receipts_collapsed": duplicate_identical,
            "holes_by_model": dict(Counter(
                row["model"] for row in per_spec for _ in range(row["holes"])
            )),
        },
        "inputs": {
            "experiments": {"path": str(args.experiments.resolve()), "sha256": sha256(args.experiments)},
            "canonical_source": {"path": str(args.canonical_source.resolve()), "sha256": sha256(args.canonical_source)},
            "legacy_routes": {"path": str(args.legacy_routes.resolve()), "sha256": sha256(args.legacy_routes)},
            "gap_index": {"path": str(args.gap_index.resolve()), "sha256": sha256(args.gap_index)},
        },
        "specs": per_spec,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"], "counts": result["counts"],
        "blocked_specs": [
            {k: row[k] for k in ("model", "disk_layer", "timestep", "holes", "missing_scenes")}
            for row in per_spec if row["holes"]
        ],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
