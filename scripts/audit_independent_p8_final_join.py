#!/usr/bin/env python3
"""Audit the frozen independent-P8 feature/label join without rescanning payloads.

Coverage receipts establish payload integrity.  This audit establishes the
orthogonal contract that every consumer resolves features by the exact actual
raw ID (or ordered p8 tuple), and independently recomputes pose-derived labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


MODELS = ("dino", "f3r", "vjepa2", "wan")
ROTATION_WEIGHT = 0.5
MODEL_PREPROCESS = {
    "dino": (728, 420),
    "f3r": (512, 288),
    "vjepa2": None,
    "wan": (832, 480),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o644)
        os.link(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def raw_scene(raw_id: str) -> str:
    prefix = "scannetpp:"
    if not raw_id.startswith(prefix) or "/" not in raw_id[len(prefix):]:
        raise ValueError(f"malformed raw ID: {raw_id}")
    return raw_id[len(prefix):].split("/", 1)[0]


def image_path(root: Path, raw_id: str) -> Path:
    scene, name = raw_id[len("scannetpp:"):].split("/", 1)
    path = root / scene / "images" / name
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    return path


def rotation_angle(a: np.ndarray, b: np.ndarray) -> float:
    relative = a.T @ b
    cosine = (float(np.trace(relative)) - 1.0) * 0.5
    return float(math.acos(max(-1.0, min(1.0, cosine))))


def motion_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3])) + ROTATION_WEIGHT * rotation_angle(a[:3, :3], b[:3, :3])


def action9(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, bw = np.linalg.inv(a), np.linalg.inv(b)
    rotation = bw[:3, :3] @ aw[:3, :3].T
    translation = bw[:3, 3] - rotation @ aw[:3, 3]
    return np.concatenate((rotation[:2].reshape(-1), translation))


def assert_close(actual: Any, expected: Any, label: str, atol: float = 1e-9) -> None:
    if not np.allclose(np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64), rtol=0.0, atol=atol):
        raise ValueError(f"{label} mismatch")


def unit_indices(request_rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[Any, dict[str, Any]]]]:
    result: dict[str, dict[str, dict[Any, dict[str, Any]]]] = {}
    for model in MODELS:
        result[model] = {"isolated": {}, "context": {}}
    unit_ids: set[str] = set()
    for row in request_rows:
        model = row["model"]
        if row["unit_id"] in unit_ids:
            raise ValueError(f"duplicate unit ID: {row['unit_id']}")
        unit_ids.add(row["unit_id"])
        if row["feature_role"] == "isolated_frame":
            key = row["raw_frame_id"]
            target = result[model]["isolated"]
            if row["context_raw_ids"] is not None:
                raise ValueError(f"isolated unit has context: {row['unit_id']}")
        elif row["feature_role"] == "streaming_context_p8":
            key = tuple(row["context_raw_ids"])
            target = result[model]["context"]
            if len(key) != 8 or key[-1] != row["raw_frame_id"]:
                raise ValueError(f"invalid ordered context: {row['unit_id']}")
        else:
            raise ValueError(f"invalid role: {row['feature_role']}")
        if key in target:
            raise ValueError(f"ambiguous exact lookup for {model}: {key}")
        target[key] = row
    return result


def validate_common(row: dict[str, Any]) -> None:
    context = row["context_raw_ids"]
    if len(context) != 8 or len(set(context)) != 8 or context[-1] != row["anchor_raw_id"]:
        raise ValueError(f"bad p8 context: {row['sample_id']}")
    raw_ids = list(context)
    for key in ("anchor_raw_id", "goal_raw_id", "candidate_endpoint_raw_id_1", "candidate_endpoint_raw_id_2", "final_raw_id"):
        if row.get(key):
            raw_ids.append(row[key])
    raw_ids.extend(row.get("waypoint_raw_ids", []))
    if any(raw_scene(raw) != row["scene_id"] for raw in raw_ids):
        raise ValueError(f"cross-scene raw ID: {row['sample_id']}")
    if "scene_scale" in row:
        raise ValueError(f"unexpected scene_scale: {row['sample_id']}")


def validate_action(row: dict[str, Any]) -> None:
    validate_common(row)
    anchor = np.asarray(row["anchor_pose"], dtype=np.float64)
    goal = np.asarray(row["goal_pose"], dtype=np.float64)
    endpoints = [np.asarray(row["endpoint_pose_1"], dtype=np.float64), np.asarray(row["endpoint_pose_2"], dtype=np.float64)]
    for index, endpoint in enumerate(endpoints, 1):
        assert_close(row[f"action_{index}_9d"], action9(anchor, endpoint), f"{row['sample_id']} action {index}")
        utility = motion_distance(anchor, goal) - motion_distance(endpoint, goal)
        assert_close(row[f"utility_{index}"], utility, f"{row['sample_id']} utility {index}")
    expected = 1 if row["utility_1"] > row["utility_2"] else 2
    if row["best_candidate"] != expected:
        raise ValueError(f"label mismatch: {row['sample_id']}")
    assert_close(row["utility_margin"], abs(row["utility_1"] - row["utility_2"]), f"{row['sample_id']} margin")
    assert_close(row["goal_pose_distance"], motion_distance(anchor, goal), f"{row['sample_id']} goal distance")


def validate_algebra(row: dict[str, Any]) -> None:
    validate_common(row)
    if len(row["waypoint_raw_ids"]) != 4 or len(row["waypoint_poses"]) != 4 or len(row["primitive_actions_9d"]) != 4:
        raise ValueError(f"invalid primitive path: {row['sample_id']}")
    expected_targets = {"h1": row["waypoint_raw_ids"][0], "h2": row["waypoint_raw_ids"][1], "h4": row["waypoint_raw_ids"][3]}
    if row["horizon_target_raw_ids"] != expected_targets or row["final_raw_id"] != row["waypoint_raw_ids"][3]:
        raise ValueError(f"horizon/raw-ID mismatch: {row['sample_id']}")
    poses = [np.asarray(row["anchor_pose"], dtype=np.float64)] + [np.asarray(value, dtype=np.float64) for value in row["waypoint_poses"]]
    for index in range(4):
        assert_close(row["primitive_actions_9d"][index], action9(poses[index], poses[index + 1]), f"{row['sample_id']} primitive {index + 1}")
        assert_close(row["primitive_motion_magnitudes"][index], motion_distance(poses[index], poses[index + 1]), f"{row['sample_id']} motion {index + 1}")


def resolve_row(row: dict[str, Any], indices: dict[str, dict[str, dict[Any, dict[str, Any]]]], counts: Counter[str]) -> None:
    for model in MODELS:
        context_unit = indices[model]["context"].get(tuple(row["context_raw_ids"]))
        if context_unit is None:
            raise ValueError(f"missing exact context for {model}/{row['sample_id']}")
        counts[f"{model}|context"] += 1
        raw_roles = []
        for role, key in (("anchor", "anchor_raw_id"), ("goal", "goal_raw_id"),
                          ("endpoint_1", "candidate_endpoint_raw_id_1"), ("endpoint_2", "candidate_endpoint_raw_id_2"),
                          ("final", "final_raw_id")):
            if row.get(key):
                raw_roles.append((role, row[key]))
        # The immutable request contains the registered h1/h2/h4 targets.  h3
        # remains part of the primitive action chain and is pose-audited above,
        # but it is intentionally not a feature-supervision lookup.
        if row.get("horizon_target_raw_ids"):
            raw_roles.extend((horizon, row["horizon_target_raw_ids"][horizon]) for horizon in ("h1", "h2", "h4"))
        for role, raw in raw_roles:
            unit = indices[model]["isolated"].get(raw)
            if unit is None or unit["raw_frame_id"] != raw or unit["context_raw_ids"] is not None:
                raise ValueError(f"missing exact isolated {role} for {model}/{row['sample_id']}")
            counts[f"{model}|{role}"] += 1


def actual_load_evidence(model: str, role: str, split: str, row: dict[str, Any], raw_ids: list[str],
                         indices: dict[str, dict[str, dict[Any, dict[str, Any]]]], data_root: Path,
                         feature_root: Path) -> dict[str, Any]:
    is_context = role == "context"
    unit = indices[model]["context"][tuple(raw_ids)] if is_context else indices[model]["isolated"][raw_ids[0]]
    feature = feature_root / model / "scannetpp" / unit["scene_id"] / f"{unit['unit_id']}.safetensors"
    sidecar = Path(f"{feature}.manifest.json")
    meta = json.loads(sidecar.read_text())
    logical = raw_ids if is_context else [raw_ids[0]]
    if meta.get("logical_input_raw_ids") != logical or meta.get("raw_frame_id") != unit["raw_frame_id"] or meta.get("context_raw_ids") != unit["context_raw_ids"]:
        raise ValueError(f"sampled sidecar input mismatch: {model}/{row['sample_id']}/{role}")
    loaded = []
    for raw in logical:
        path = image_path(data_root, raw)
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            source_size = list(rgb.size)
            target = MODEL_PREPROCESS[model]
            processed = rgb if target is None else rgb.resize(target, Image.Resampling.BICUBIC)
            pixel_sha = hashlib.sha256(np.asarray(processed).tobytes()).hexdigest()
        loaded.append({"raw_id": raw, "resolved_path": str(path.resolve()), "file_sha256": sha256(path),
                       "source_size_wh": source_size, "preprocessed_size_wh": list(processed.size),
                       "preprocessed_rgb_sha256": pixel_sha})
    expanded = list(logical)
    target_count = {"dino": len(logical), "f3r": 64, "vjepa2": 64, "wan": 81}[model]
    expanded.extend([expanded[-1]] * (target_count - len(expanded)))
    return {"model": model, "component_split": split, "semantic_role": role, "sample_id": row["sample_id"],
            "unit_id": unit["unit_id"], "actual_loaded_inputs": loaded,
            "ordered_expanded_raw_ids_sha256": hashlib.sha256(json.dumps(expanded, separators=(",", ":")).encode()).hexdigest(),
            "expanded_count": len(expanded), "sidecar": str(sidecar), "feature": str(feature)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest-dir", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    request_rows = read_jsonl(args.request_manifest)
    indices = unit_indices(request_rows)
    source = args.source_manifest_dir
    final = args.manifest_dir
    files = {
        "action_fit": final / "actionability_fit_scene_disjoint_v1.jsonl",
        "action_dev": final / "actionability_dev_scene_disjoint_v1.jsonl",
        "action_validation": final / "actionability_p8_independent_v1_val.jsonl",
        "action_train": source / "actionability_p8_independent_v1_train.jsonl",
        "algebra_fit": final / "algebra_fit_scene_disjoint_v1.jsonl",
        "algebra_dev": final / "algebra_dev_scene_disjoint_v1.jsonl",
        "algebra_validation": final / "algebra_p8_independent_v1_val.jsonl",
        "algebra_train": source / "algebra_p8_independent_v1_train.jsonl",
        "loops_validation": final / "algebra_p8_independent_v1_real_loops.jsonl",
    }
    rows = {name: read_jsonl(path) for name, path in files.items()}
    if {row["sample_id"] for row in rows["action_train"]} != {row["sample_id"] for row in rows["action_fit"] + rows["action_dev"]}:
        raise ValueError("action fit/dev do not partition training")
    if {row["sample_id"] for row in rows["algebra_train"]} != {row["sample_id"] for row in rows["algebra_fit"] + rows["algebra_dev"]}:
        raise ValueError("algebra fit/dev do not partition training")
    train_scenes = {row["scene_id"] for key in ("action_train", "algebra_train") for row in rows[key]}
    validation_scenes = {row["scene_id"] for key in ("action_validation", "algebra_validation", "loops_validation") for row in rows[key]}
    if train_scenes & validation_scenes:
        raise ValueError("train/validation scene overlap")
    if ({row["scene_id"] for row in rows["action_fit"]} & {row["scene_id"] for row in rows["action_dev"]}
            or {row["scene_id"] for row in rows["algebra_fit"]} & {row["scene_id"] for row in rows["algebra_dev"]}):
        raise ValueError("fit/dev scene overlap")

    counts: Counter[str] = Counter()
    audited: set[tuple[str, str]] = set()
    for name, group in rows.items():
        for row in group:
            identity = (name, row["sample_id"])
            if identity in audited:
                raise ValueError(f"duplicate sample: {identity}")
            audited.add(identity)
            if name.startswith("action"):
                validate_action(row)
            elif name.startswith("algebra"):
                validate_algebra(row)
            else:
                validate_common(row)
            resolve_row(row, indices, counts)

    evidence = []
    sample_groups = (
        ("action_fit", "fit"), ("action_dev", "dev"), ("action_validation", "validation"),
        ("algebra_fit", "fit"), ("algebra_dev", "dev"), ("algebra_validation", "validation"),
        ("loops_validation", "validation"),
    )
    for model in MODELS:
        for group_name, split in sample_groups:
            row = sorted(rows[group_name], key=lambda value: value["sample_id"])[0]
            roles: list[tuple[str, list[str]]] = [("context", list(row["context_raw_ids"])), ("anchor", [row["anchor_raw_id"]])]
            if group_name.startswith("action"):
                roles.extend((("goal", [row["goal_raw_id"]]), ("endpoint_1", [row["candidate_endpoint_raw_id_1"]]),
                              ("endpoint_2", [row["candidate_endpoint_raw_id_2"]])))
            elif group_name.startswith("algebra"):
                roles.extend((horizon, [row["horizon_target_raw_ids"][horizon]]) for horizon in ("h1", "h2", "h4"))
            else:
                roles.append(("final", [row["final_raw_id"]]))
            for role, raw_ids in roles:
                evidence.append(actual_load_evidence(model, role, f"{group_name}:{split}", row, raw_ids,
                                                     indices, args.data_root, args.feature_root))

    per_model = {
        model: {"request_units": sum(row["model"] == model for row in request_rows),
                "isolated_units": len(indices[model]["isolated"]), "context_units": len(indices[model]["context"]),
                "resolved_references": sum(value for key, value in counts.items() if key.startswith(model + "|")),
                "actual_load_checks": sum(item["model"] == model for item in evidence)}
        for model in MODELS
    }
    output = {
        "status": "COMPLETE_VALID", "schema_version": 1,
        "request_manifest_sha256": sha256(args.request_manifest),
        "manifest_sha256": {name: sha256(path) for name, path in files.items()},
        "records": {name: len(group) for name, group in rows.items()},
        "scene_boundaries": {"train_scenes": len(train_scenes), "validation_scenes": len(validation_scenes), "overlap": 0},
        "pose_convention": "camera-to-world source poses; action maps anchor camera to endpoint camera; scene_scale=1",
        "lookup_policy": "exact actual raw ID / ordered eight-ID tuple only; requested_roles and row order unused",
        "learned_input_policy": "features plus registered 9D actions only; pose/utility/label/raw ID/scene/time-direction metadata excluded",
        "per_model": per_model,
        "actual_load_and_preprocess_evidence": evidence,
    }
    atomic_json(args.output, output)
    print(json.dumps({key: output[key] for key in ("status", "records", "scene_boundaries", "per_model")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
