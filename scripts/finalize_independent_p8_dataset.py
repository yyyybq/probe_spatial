#!/usr/bin/env python3
"""Close and version the independent P8 dataset without changing source manifests.

The script derives one shared, scene-disjoint development split from the frozen
training scenes, copies the frozen held-out manifests verbatim into a paper
package, and emits an audit plus a hash index. It never reads model scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SEED = "independent-p8-final-dev-v1-20260913"
FRAME_RE = re.compile(r"^scannetpp:([^/]+)/frame_(\d+)\.jpg$")


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_path(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def canonical_jsonl(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
        for row in rows
    )


def create(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o644)
        os.link(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def create_json(path: Path, value: Any) -> None:
    create(path, (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode())


def raw_scene(raw_id: str) -> str:
    match = FRAME_RE.fullmatch(raw_id)
    if match is None:
        raise ValueError(f"malformed raw ID: {raw_id}")
    return match.group(1)


def raw_index(raw_id: str) -> int:
    match = FRAME_RE.fullmatch(raw_id)
    if match is None:
        raise ValueError(f"malformed raw ID: {raw_id}")
    return int(match.group(2))


def all_raw_ids(row: dict[str, Any]) -> list[str]:
    values = list(row.get("context_raw_ids", []))
    for key in (
        "anchor_raw_id", "goal_raw_id", "candidate_endpoint_raw_id_1",
        "candidate_endpoint_raw_id_2", "final_raw_id",
    ):
        if row.get(key):
            values.append(row[key])
    values.extend(row.get("waypoint_raw_ids", []))
    values.extend(row.get("horizon_target_raw_ids", {}).values())
    return values


def assert_common(row: dict[str, Any], expected_split: str) -> None:
    assert row["split"] == expected_split
    context = row["context_raw_ids"]
    assert len(context) == 8 and len(set(context)) == 8
    assert context[-1] == row["anchor_raw_id"]
    assert [raw_index(raw) for raw in context] == sorted(raw_index(raw) for raw in context)
    assert all(raw_scene(raw) == row["scene_id"] for raw in all_raw_ids(row))


def action_audit(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    exact_signatures: Counter[tuple[Any, ...]] = Counter()
    anchor_goal_signatures: Counter[tuple[str, str, str]] = Counter()
    endpoint_seen = 0
    goal_seen = 0
    label_errors = 0
    scale_variation = 0
    for row in rows:
        assert_common(row, split)
        assert len(row["action_1_9d"]) == len(row["action_2_9d"]) == 9
        expected = 1 if row["utility_1"] > row["utility_2"] else 2
        label_errors += int(expected != row["best_candidate"])
        assert abs(abs(row["utility_1"] - row["utility_2"]) - row["utility_margin"]) < 1e-9
        assert row["positive_progress"] is True and max(row["utility_1"], row["utility_2"]) > 0
        endpoints = (row["candidate_endpoint_raw_id_1"], row["candidate_endpoint_raw_id_2"])
        endpoint_seen += int(any(value in row["context_raw_ids"] for value in endpoints))
        goal_seen += int(row["goal_raw_id"] in row["context_raw_ids"])
        exact_signatures[(tuple(row["context_raw_ids"]), row["goal_raw_id"], endpoints)] += 1
        anchor_goal_signatures[(row["scene_id"], row["anchor_raw_id"], row["goal_raw_id"])] += 1
        # No scene_scale field means the frozen action is already in its recorded
        # physical convention; a sample-varying hidden scale is therefore absent.
        scale_variation += int("scene_scale" in row)
    return {
        "records": len(rows),
        "scenes": len({row["scene_id"] for row in rows}),
        "duplicate_sample_ids": len(rows) - len({row["sample_id"] for row in rows}),
        "duplicate_exact_raw_combinations": sum(count - 1 for count in exact_signatures.values() if count > 1),
        "repeated_anchor_goal_combinations": sum(count - 1 for count in anchor_goal_signatures.values() if count > 1),
        "label_errors": label_errors,
        "goal_in_context": goal_seen,
        "candidate_endpoint_in_context": endpoint_seen,
        "records_with_scene_scale_field": scale_variation,
        "label_counts": dict(sorted(Counter(row["best_candidate"] for row in rows).items())),
        "goal_direction_counts": dict(sorted(Counter(row["goal_temporal_direction"] for row in rows).items())),
    }


def algebra_audit(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    signatures: Counter[tuple[Any, ...]] = Counter()
    path_lengths: Counter[int] = Counter()
    for row in rows:
        assert_common(row, split)
        assert row["path_length"] == 4
        assert len(row["primitive_actions_9d"]) == 4
        assert all(len(action) == 9 for action in row["primitive_actions_9d"])
        assert len(row["waypoint_raw_ids"]) == len(row["waypoint_poses"]) == 4
        assert row["horizon_target_raw_ids"]["h1"] == row["waypoint_raw_ids"][0]
        assert row["horizon_target_raw_ids"]["h2"] == row["waypoint_raw_ids"][1]
        assert row["horizon_target_raw_ids"]["h4"] == row["waypoint_raw_ids"][3]
        assert row["final_raw_id"] == row["waypoint_raw_ids"][3]
        signatures[(tuple(row["context_raw_ids"]), tuple(row["waypoint_raw_ids"]))] += 1
        path_lengths[row["path_length"]] += 1
    return {
        "records": len(rows),
        "scenes": len({row["scene_id"] for row in rows}),
        "duplicate_sample_ids": len(rows) - len({row["sample_id"] for row in rows}),
        "duplicate_exact_raw_combinations": sum(count - 1 for count in signatures.values() if count > 1),
        "path_lengths": dict(sorted(path_lengths.items())),
    }


def loop_audit(rows: list[dict[str, Any]], heldout_scenes: set[str]) -> dict[str, Any]:
    by_id = {row["sample_id"]: row for row in rows}
    real = [row for row in rows if row["real_loop_flag"]]
    nonloop = [row for row in rows if not row["real_loop_flag"]]
    errors: list[str] = []
    path_delta: list[int] = []
    for row in rows:
        assert_common(row, "val")
        if row["scene_id"] not in heldout_scenes:
            errors.append(f"non-held-out scene: {row['sample_id']}")
    for row in real:
        mate = by_id.get(row["matched_non_loop_id"])
        if mate is None or mate["real_loop_flag"] or mate["scene_id"] != row["scene_id"]:
            errors.append(f"invalid mate: {row['sample_id']}")
            continue
        path_delta.append(abs(int(row["path_length"]) - int(mate["path_length"])))
    reverse_counts = Counter(row.get("matched_real_loop_id") for row in nonloop)
    return {
        "records": len(rows),
        "pairs": len(real),
        "real_loop_records": len(real),
        "matched_non_loop_records": len(nonloop),
        "scenes": len({row["scene_id"] for row in rows}),
        "pairing_errors": errors,
        "max_path_length_delta": max(path_delta) if path_delta else None,
        "nonloop_reverse_link_duplicates": sum(count - 1 for key, count in reverse_counts.items() if key and count > 1),
        "path_length": {
            "real_min": min(int(row["path_length"]) for row in real),
            "real_max": max(int(row["path_length"]) for row in real),
            "nonloop_min": min(int(row["path_length"]) for row in nonloop),
            "nonloop_max": max(int(row["path_length"]) for row in nonloop),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_manifest_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")

    names = {
        "actionability_train": "actionability_p8_independent_v1_train.jsonl",
        "actionability_heldout": "actionability_p8_independent_v1_val.jsonl",
        "algebra_train": "algebra_p8_independent_v1_train.jsonl",
        "algebra_heldout": "algebra_p8_independent_v1_val.jsonl",
        "algebra_real_loops": "algebra_p8_independent_v1_real_loops.jsonl",
    }
    paths = {key: source / value for key, value in names.items()}
    rows = {key: read_jsonl(path) for key, path in paths.items()}
    split = json.loads((source / "scene_split_independent_v1.json").read_text())
    official_train = set(split["splits"]["train"])
    official_heldout = set(split["splits"]["val"])

    observed_train = {row["scene_id"] for key in ("actionability_train", "algebra_train") for row in rows[key]}
    observed_heldout = {
        row["scene_id"]
        for key in ("actionability_heldout", "algebra_heldout", "algebra_real_loops")
        for row in rows[key]
    }
    if observed_train & observed_heldout:
        raise SystemExit("cross-task train/held-out scene contamination")
    if not observed_train <= official_train or not observed_heldout <= official_heldout:
        raise SystemExit("manifest scene not in official global split")

    ranked_train_scenes = sorted(
        observed_train,
        key=lambda scene: digest_bytes(f"{SEED}|{scene}".encode()),
    )
    dev_count = max(1, round(0.20 * len(ranked_train_scenes)))
    dev_scenes = set(ranked_train_scenes[:dev_count])
    fit_scenes = observed_train - dev_scenes

    partitions: dict[str, list[dict[str, Any]]] = {}
    for task in ("actionability", "algebra"):
        source_rows = rows[f"{task}_train"]
        partitions[f"{task}_fit"] = [row for row in source_rows if row["scene_id"] in fit_scenes]
        partitions[f"{task}_dev"] = [row for row in source_rows if row["scene_id"] in dev_scenes]
        assert len(partitions[f"{task}_fit"]) + len(partitions[f"{task}_dev"]) == len(source_rows)

    pilot_names = {
        "actionability": "science_pilot_subset_v1_actionability_val.jsonl",
        "algebra": "science_pilot_subset_v1_algebra_val.jsonl",
        "loops": "science_pilot_subset_v1_algebra_real_loops.jsonl",
    }
    pilot_rows = {key: read_jsonl(source / value) for key, value in pilot_names.items()}
    pilot_heldout_scenes = {row["scene_id"] for values in pilot_rows.values() for row in values}

    audits = {
        "actionability_train": action_audit(rows["actionability_train"], "train"),
        "actionability_heldout": action_audit(rows["actionability_heldout"], "val"),
        "algebra_train": algebra_audit(rows["algebra_train"], "train"),
        "algebra_heldout": algebra_audit(rows["algebra_heldout"], "val"),
        "algebra_real_loops": loop_audit(rows["algebra_real_loops"], official_heldout),
    }
    request = source / "feature_request_manifest_v2.jsonl"
    request_rows = read_jsonl(request)
    request_counts = Counter(row["model"] for row in request_rows)
    unit_ids = [row["unit_id"] for row in request_rows]

    output.mkdir(parents=True)
    manifest_dir = output / "manifests"
    generated: dict[str, dict[str, Any]] = {}
    for key, values in partitions.items():
        path = manifest_dir / f"{key}_scene_disjoint_v1.jsonl"
        payload = canonical_jsonl(values)
        create(path, payload)
        generated[key] = {
            "path": str(path), "sha256": digest_bytes(payload), "records": len(values),
            "scenes": len({row["scene_id"] for row in values}),
        }
    for key in ("actionability_heldout", "algebra_heldout", "algebra_real_loops"):
        destination = manifest_dir / names[key]
        payload = paths[key].read_bytes()
        create(destination, payload)
        generated[key] = {
            "path": str(destination), "sha256": digest_bytes(payload), "records": len(rows[key]),
            "scenes": len({row["scene_id"] for row in rows[key]}), "verbatim_source_copy": True,
        }
    dev_split = {
        "schema_version": 1,
        "seed": SEED,
        "method": "SHA256 rank over the union of observed frozen training scenes; first round(20%) are dev",
        "fit_scenes": sorted(fit_scenes),
        "dev_scenes": sorted(dev_scenes),
        "heldout_scenes": sorted(observed_heldout),
    }
    create_json(manifest_dir / "scene_split_final_v1.json", dev_split)

    audit = {
        "schema_version": 1,
        "status": "COMPLETE_VALID",
        "source_manifest_dir": str(source),
        "source_hashes": {key: digest_path(path) for key, path in paths.items()},
        "source_global_scene_split_sha256": digest_path(source / "scene_split_independent_v1.json"),
        "global_scene_universe": {
            "scanned_directory_candidates": split["scene_directory_candidates"],
            "usable_scene_roots": split["scene_root_count"],
            "rejected_scenes": split["rejected_scenes"],
            "official_train_scenes": len(official_train),
            "official_heldout_scenes": len(official_heldout),
            "observed_task_train_scenes": len(observed_train),
            "observed_task_heldout_scenes": len(observed_heldout),
        },
        "scene_leakage": {
            "cross_task_train_heldout_overlap": sorted(observed_train & observed_heldout),
            "action_train_vs_algebra_heldout": sorted(
                {row["scene_id"] for row in rows["actionability_train"]}
                & {row["scene_id"] for row in rows["algebra_heldout"]}
            ),
            "algebra_train_vs_action_heldout": sorted(
                {row["scene_id"] for row in rows["algebra_train"]}
                & {row["scene_id"] for row in rows["actionability_heldout"]}
            ),
            "fit_vs_dev": sorted(fit_scenes & dev_scenes),
        },
        "heldout_development_history": {
            "pilot_analysis_exists": True,
            "pilot_actionability_heldout_samples": len(pilot_rows["actionability"]),
            "pilot_algebra_heldout_samples": len(pilot_rows["algebra"]),
            "pilot_loop_pairs": len(pilot_rows["loops"]) // 2,
            "pilot_heldout_scenes": len(pilot_heldout_scenes),
            "all_actionability_heldout_scenes_touched_by_pilot": (
                {row["scene_id"] for row in rows["actionability_heldout"]} <= pilot_heldout_scenes
            ),
            "all_algebra_heldout_scenes_touched_by_pilot": (
                {row["scene_id"] for row in rows["algebra_heldout"]} <= pilot_heldout_scenes
            ),
            "designation": "validation_not_blind_test",
            "confirmatory_scene_disjoint_subset_available": False,
        },
        "task_audits": audits,
        "feature_request_v2": {
            "path": str(request),
            "sha256": digest_path(request),
            "units": len(request_rows),
            "unique_unit_ids": len(set(unit_ids)),
            "duplicate_unit_ids": len(unit_ids) - len(set(unit_ids)),
            "per_model": dict(sorted(request_counts.items())),
        },
        "generated_manifests": generated,
    }
    create_json(output / "DATASET_CLOSURE_AUDIT.json", audit)
    index = {
        "schema_version": 1,
        "source": {str(path): digest_path(path) for path in [*paths.values(), source / "scene_split_independent_v1.json", request]},
        "generated": {
            str(path): digest_path(path)
            for path in sorted(manifest_dir.iterdir())
            if path.is_file()
        },
    }
    create_json(output / "MANIFEST_HASH_INDEX.json", index)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
