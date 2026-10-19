#!/usr/bin/env python3
"""Build the exact Wan-L29 feature request for continuous-action subset v2."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def identity(row: dict[str, Any]) -> tuple[str, tuple[str, ...] | str]:
    if row["feature_role"] == "streaming_context_p8":
        return row["feature_role"], tuple(row["context_raw_ids"])
    return row["feature_role"], row["raw_frame_id"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-request", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    source_rows = [row for row in read_jsonl(args.source_request) if row["model"] == "wan"]
    source_index = {identity(row): row for row in source_rows}
    if len(source_index) != len(source_rows):
        raise ValueError("ambiguous source Wan request identities")

    wanted: dict[tuple[str, tuple[str, ...] | str], dict[str, Any]] = {}
    samples: dict[tuple[str, tuple[str, ...] | str], set[str]] = defaultdict(set)
    roles: dict[tuple[str, tuple[str, ...] | str], set[str]] = defaultdict(set)
    split_counts: dict[str, int] = {}
    scenes: dict[str, set[str]] = {}
    manifest_hashes: dict[str, str] = {}
    for split, path in (("fit", args.fit), ("dev", args.dev), ("validation", args.validation)):
        rows = read_jsonl(path)
        split_counts[split] = len(rows)
        scenes[split] = {row["scene_id"] for row in rows}
        manifest_hashes[split] = sha256(path)
        for row in rows:
            current_key = ("streaming_context_p8", tuple(row["context_raw_ids"]))
            goal_key = ("isolated_frame", row["horizon_target_raw_ids"]["h4"])
            for key, role in ((current_key, "continuous_current_p8"), (goal_key, "continuous_goal_h4")):
                if key not in source_index:
                    raise KeyError(f"missing exact source identity: {split}/{row['sample_id']}/{key}")
                wanted[key] = source_index[key]
                samples[key].add(row["sample_id"])
                roles[key].add(role)

    if split_counts != {"fit": 803, "dev": 221, "validation": 256}:
        raise ValueError(f"unexpected frozen split counts: {split_counts}")
    if scenes["fit"] & scenes["dev"] or scenes["fit"] & scenes["validation"] or scenes["dev"] & scenes["validation"]:
        raise ValueError("scene split overlap")

    output_rows: list[dict[str, Any]] = []
    for key in sorted(wanted, key=lambda value: canonical(value)):
        source = wanted[key]
        unit_identity = {
            "schema": "continuous-action-wan-l29-unit-v1",
            "model": "wan",
            "layer": 29,
            "timestep": 749,
            "feature_role": source["feature_role"],
            "raw_frame_id": source["raw_frame_id"],
            "context_raw_ids": source["context_raw_ids"],
        }
        unit_id = "cal29_" + hashlib.sha256(canonical(unit_identity).encode()).hexdigest()[:24]
        output_rows.append({
            "context_raw_ids": source["context_raw_ids"],
            "exact_raw_id_required": True,
            "fallback_forbidden": True,
            "feature_role": source["feature_role"],
            "layer": 29,
            "model": "wan",
            "raw_frame_id": source["raw_frame_id"],
            "requested_roles": sorted(roles[key]),
            "scene_id": source["scene_id"],
            "source_manifest_sha256": manifest_hashes,
            "source_sample_ids": sorted(samples[key]),
            "timestep": 749,
            "unit_id": unit_id,
        })
    output_rows.sort(key=lambda row: row["unit_id"])
    if len({row["unit_id"] for row in output_rows}) != len(output_rows):
        raise ValueError("unit ID collision")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() or args.audit.exists():
        raise FileExistsError("refusing to overwrite request or audit")
    args.output.write_text("".join(canonical(row) + "\n" for row in output_rows))
    request_sha = sha256(args.output)
    audit = {
        "status": "COMPLETE_VALID",
        "request_sha256": request_sha,
        "source_request_sha256": sha256(args.source_request),
        "manifest_sha256": manifest_hashes,
        "split_counts": split_counts,
        "split_scenes": {key: len(value) for key, value in scenes.items()},
        "requested_units": len(output_rows),
        "streaming_context_p8": sum(row["feature_role"] == "streaming_context_p8" for row in output_rows),
        "isolated_h4": sum(row["feature_role"] == "isolated_frame" for row in output_rows),
        "layer": 29,
        "timestep": 749,
        "exact_lookup_only": True,
    }
    args.audit.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
