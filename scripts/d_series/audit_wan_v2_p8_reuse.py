#!/usr/bin/env python3
"""Fail-closed Wan V2 -> D-series p8 identity and reuse audit.

Frame identity and numerical feature identity are deliberately separate.  A
canonical row may map to exactly the same RGB frames while still being unsafe
to reuse when layer, prompt, or sample-local noise identity differs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


D_MANIFEST_SHA256 = "781e25848de48fa5219aad876bd10cb8f9445328b145a19193a6f6b2ab5efcc9"
PARENT_MANIFEST_SHA256 = "20976fc86abfdc96d40446a0fe15fc32af699369309c540f9a352e868ce89f91"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    temporary.replace(path)


def frame_raw_id(name: str) -> int:
    stem = Path(name).stem
    if not stem.startswith("frame_"):
        raise ValueError(f"unexpected frame name: {name}")
    return int(stem.removeprefix("frame_"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("experiments/d_series_scannetpp_v1/data/sample_manifest.jsonl"),
    )
    parser.add_argument(
        "--parent-manifest", type=Path,
        default=Path("artifacts/tstep_canonical_v2/manifests/inscene_c_h124_prefix8.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=6)
    args = parser.parse_args()

    if sha256(args.manifest) != D_MANIFEST_SHA256:
        raise SystemExit("D-series manifest hash mismatch")
    if sha256(args.parent_manifest) != PARENT_MANIFEST_SHA256:
        raise SystemExit("canonical p8 parent manifest hash mismatch")
    if args.num_shards < 1:
        raise SystemExit("num-shards must be positive")

    d_rows = read_jsonl(args.manifest)
    parent_rows = read_jsonl(args.parent_manifest)
    parents = {str(row["sample_id"]): row for row in parent_rows}
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in d_rows:
        by_parent[str(row["parent_sample_id"])].append(row)
    if len(d_rows) != 5442 or len(by_parent) != 2721 or len(parents) != 2721:
        raise SystemExit(
            f"unexpected inventory d_rows={len(d_rows)} parents={len(by_parent)} "
            f"canonical={len(parents)}"
        )
    if set(by_parent) != set(parents):
        raise SystemExit("D-series parent sample set differs from canonical p8 manifest")

    # The completed canonical-V2 Wan family is not numerically interchangeable
    # with the frozen D-series Wan contract.  These are hard semantic gates,
    # not availability guesses.
    incompatibilities = [
        "required_layer20_not_present_in_legacy_p8_exact_full_layers_0_5_10_15_25_29",
        "canonical_v2_wan_prompt_literal_dot_but_dseries_requires_empty_prompt",
        "canonical_v2_noise_identity_not_dseries_sha256_42_pipe_role_and_sample_id",
    ]
    mapping: list[dict[str, Any]] = []
    gap_windows: list[dict[str, Any]] = []
    role_counts: Counter[str] = Counter()
    identity_counts: Counter[str] = Counter()

    for parent_id in sorted(by_parent):
        parent = parents[parent_id]
        rows = sorted(by_parent[parent_id], key=lambda item: int(item["horizon"]))
        if [int(row["horizon"]) for row in rows] != [2, 4]:
            raise SystemExit(f"{parent_id}: expected h2/h4 rows")
        valid_frames = [str(name) for name in parent["valid_frames"]]
        prefix_positions = [int(value) for value in parent["streaming_indices"]]
        prefix_names = [valid_frames[position] for position in prefix_positions]
        prefix_raw = [frame_raw_id(name) for name in prefix_names]
        expected_prefix = [int(value) for value in rows[0]["prefix_raw_frame_ids"]]
        if prefix_raw != expected_prefix or prefix_names != rows[0]["prefix_frame_names"]:
            raise SystemExit(f"{parent_id}: prefix identity mismatch")

        roles = []
        prefix_record = {
            "role": "prefix",
            "dseries_id": parent_id,
            "canonical_positions": prefix_positions,
            "frame_names": prefix_names,
            "raw_frame_ids": prefix_raw,
            "frame_identity_exact": True,
            "feature_reusable": False,
            "reuse_rejection_reasons": incompatibilities,
        }
        mapping.append({"parent_sample_id": parent_id, **prefix_record})
        roles.append(prefix_record)
        role_counts["prefix_gap"] += 1
        identity_counts["prefix_exact"] += 1

        window = [int(value) for value in parent["streaming_window_indices"]]
        prefix_len = int(parent["prefix_length"])
        for row in rows:
            horizon = int(row["horizon"])
            position_index = prefix_len + horizon - 1
            canonical_position = int(window[position_index])
            frame_name = valid_frames[canonical_position]
            raw_id = frame_raw_id(frame_name)
            if frame_name != row["target_frame_name"] or raw_id != int(row["target_raw_frame_id"]):
                raise SystemExit(f"{row['sample_id']}: target identity mismatch")
            role = f"h{horizon}"
            record = {
                "role": role,
                "dseries_id": row["sample_id"],
                "canonical_position": canonical_position,
                "canonical_window_offset": position_index,
                "frame_name": frame_name,
                "raw_frame_id": raw_id,
                "frame_identity_exact": True,
                "feature_reusable": False,
                "reuse_rejection_reasons": incompatibilities,
            }
            mapping.append({"parent_sample_id": parent_id, **record})
            roles.append(record)
            role_counts[f"{role}_gap"] += 1
            identity_counts[f"{role}_exact"] += 1

        gap_windows.append({
            "parent_sample_id": parent_id,
            "scene_id": parent["scene_id"],
            "split": rows[0]["split"],
            "roles_to_extract": ["prefix", "h2", "h4"],
            "role_records": roles,
        })

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    write_jsonl(out / "identity_mapping.jsonl", mapping)
    write_jsonl(out / "gap_windows.jsonl", gap_windows)

    shard_records = []
    for shard in range(args.num_shards):
        rows = gap_windows[shard::args.num_shards]
        selection = out / f"shard_{shard:02d}_parent_ids.txt"
        selection.write_text("".join(row["parent_sample_id"] + "\n" for row in rows))
        shard_records.append({
            "shard": shard,
            "selection_file": str(selection.resolve()),
            "selection_sha256": sha256(selection),
            "window_count": len(rows),
            "role_count": 3 * len(rows),
            "scene_count": len({row["scene_id"] for row in rows}),
            "first_parent_sample_id": rows[0]["parent_sample_id"],
            "last_parent_sample_id": rows[-1]["parent_sample_id"],
        })

    summary = {
        "status": "COMPLETE_EXACT_IDENTITY_ZERO_NUMERICAL_REUSE",
        "development_only": True,
        "dseries_manifest": str(args.manifest.resolve()),
        "dseries_manifest_sha256": sha256(args.manifest),
        "canonical_p8_manifest": str(args.parent_manifest.resolve()),
        "canonical_p8_manifest_sha256": sha256(args.parent_manifest),
        "dseries_contract": {
            "model_revision": "0fad780a534b6463e45facd96134c9f345acfa5b",
            "layer_zero_based": 20,
            "timestep_argument": 749,
            "prompt": "",
            "noise_seed_scheme": "sha256(42|role:sample_id)_uint32_little_endian",
        },
        "canonical_v2_candidate_contract": {
            "prompt": ".",
            "legacy_p8_exact_full_layers": [0, 5, 10, 15, 25, 29],
            "layer20_route_catalog_disposition": "REEXTRACT_IDENTITY_MISMATCH",
        },
        "window_count": len(gap_windows),
        "role_count": len(mapping),
        "frame_identity_counts": dict(identity_counts),
        "reusable_role_counts": {"prefix": 0, "h2": 0, "h4": 0},
        "gap_role_counts": dict(role_counts),
        "incompatibilities": incompatibilities,
        "mapping_file": str((out / "identity_mapping.jsonl").resolve()),
        "mapping_sha256": sha256(out / "identity_mapping.jsonl"),
        "gap_file": str((out / "gap_windows.jsonl").resolve()),
        "gap_sha256": sha256(out / "gap_windows.jsonl"),
        "num_shards": args.num_shards,
        "shards": shard_records,
    }
    write_json(out / "audit_summary.json", summary)
    print(json.dumps({
        "status": summary["status"], "windows": len(gap_windows),
        "roles": len(mapping), "reusable": 0,
        "shards": [item["window_count"] for item in shard_records],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
