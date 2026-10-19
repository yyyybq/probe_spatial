#!/usr/bin/env python3
"""Freeze an independent raw-pose p8 Actionability/Algebra study.

This builder deliberately has no dependency on legacy C1/C2 manifests,
features, checkpoints, registries, or result tables.  It reads only the
processed ScanNet++ image/pose metadata, builds a study-local scene split, and
emits positive raw-ID allow-lists for the later exact-feature stage.

The output roots are create-once.  A caller must provide a new science tag for
every attempt; a pre-existing root is an error rather than an overwrite.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from array import array
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Running a script by its path puts ``scripts/`` (not the repository root) on
# sys.path.  The project is intentionally not installed into the shared
# environment, so make the repository-local implementation explicit rather
# than importing a possibly unrelated site-package version.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODEL_SETTINGS = (
    {"model": "dino", "layer": 0, "timestep": None},
    {"model": "f3r", "layer": 24, "timestep": None},
    {"model": "vjepa2", "layer": 23, "timestep": None},
    {"model": "wan", "layer": 20, "timestep": 749},
)

PREFIX = 8
MOTION_STEP = 0.35
ROTATION_WEIGHT = 0.5
GOAL_OFFSET_OBSERVATIONS = 4
LOCAL_LOOP_CLOSE_DISTANCE = MOTION_STEP
MIN_LOOP_OBSERVATIONS = 8


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_rank(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_text(path, "".join(canonical_json(row) + "\n" for row in rows))


def run_text(argv: list[str]) -> tuple[int, str, str]:
    completed = subprocess.run(argv, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    return completed.returncode, completed.stdout, completed.stderr


def git_provenance(generator: Path, submit_script: Path) -> dict[str, Any]:
    status_rc, status_out, status_err = run_text(["git", "status", "--porcelain=v2"])
    diff_rc, diff_out, diff_err = run_text(["git", "diff", "--binary"])
    commit_rc, commit_out, commit_err = run_text(["git", "rev-parse", "HEAD"])
    critical = [
        generator,
        PROJECT_ROOT / "vidfm3d/utils/temporal_sampling.py",
        PROJECT_ROOT / "vidfm3d/utils/spatial_diag.py",
        PROJECT_ROOT / "vidfm3d/models/components/probe_path_integration.py",
        PROJECT_ROOT / "features/run_inscene15k.py",
        PROJECT_ROOT / "features/dino/extract_features.py",
        PROJECT_ROOT / "features/f3r/extract_features.py",
        PROJECT_ROOT / "features/vjepa2/vjepa2_feature.py",
        PROJECT_ROOT / "features/wan/extract_features.py",
        PROJECT_ROOT / "configs/splits/inscene15k_v1.json",
    ]
    source_hashes = {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
        for path in critical
        if path.is_file() and not path.is_symlink()
    }
    submit = {
        "path": str(submit_script),
        "exists": submit_script.exists(),
        "size_bytes": submit_script.stat().st_size if submit_script.exists() else None,
        "sha256": sha256_file(submit_script) if submit_script.is_file() else None,
    }
    if submit["size_bytes"] and int(submit["size_bytes"]) > 1:
        syntax_rc, _, syntax_err = run_text(["bash", "-n", str(submit_script)])
        submit["bash_n_returncode"] = syntax_rc
        submit["bash_n_stderr"] = syntax_err
    else:
        submit["bash_n_returncode"] = None
        submit["recovery_required_before_sco"] = (
            "submit.sh is empty/truncated; only a verified Cursor History copy may restore it"
        )
    return {
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "git": {
            "status_returncode": status_rc,
            "status_porcelain_v2": status_out,
            "status_stderr": status_err,
            "diff_binary_returncode": diff_rc,
            "diff_binary_sha256": sha256_bytes(diff_out.encode("utf-8")),
            "diff_binary": diff_out,
            "diff_binary_stderr": diff_err,
            "head_returncode": commit_rc,
            "head": commit_out.strip(),
            "head_stderr": commit_err,
        },
        "critical_source_sha256": source_hashes,
        "submit_sh": submit,
    }


def rotation_angle_rad(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    relative = np.asarray(rotation_a, dtype=np.float64).T @ np.asarray(rotation_b, dtype=np.float64)
    cosine = (float(np.trace(relative)) - 1.0) * 0.5
    return float(math.acos(max(-1.0, min(1.0, cosine))))


def pose_motion_components(pose_a: np.ndarray, pose_b: np.ndarray) -> tuple[float, float]:
    translation = float(np.linalg.norm(pose_a[:3, 3] - pose_b[:3, 3]))
    rotation = rotation_angle_rad(pose_a[:3, :3], pose_b[:3, :3])
    return translation, rotation


def motion_distance(pose_a: np.ndarray, pose_b: np.ndarray) -> float:
    translation, rotation = pose_motion_components(pose_a, pose_b)
    return translation + ROTATION_WEIGHT * rotation


def action_9d(anchor_c2w: np.ndarray, endpoint_c2w: np.ndarray) -> list[float]:
    """Project-standard 9D action: anchor camera coordinates -> endpoint camera coordinates."""
    anchor_w2c = np.linalg.inv(anchor_c2w)
    endpoint_w2c = np.linalg.inv(endpoint_c2w)
    rotation = endpoint_w2c[:3, :3] @ anchor_w2c[:3, :3].T
    translation = endpoint_w2c[:3, 3] - rotation @ anchor_w2c[:3, 3]
    return [float(value) for value in np.concatenate((rotation[:2].reshape(-1), translation))]


def raw_id(scene_id: str, frame_name: str) -> str:
    return f"scannetpp:{scene_id}/{frame_name}"


def pose_list(pose: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in np.asarray(pose, dtype=np.float64)]


def collect_scene(scene_dir: Path) -> tuple[dict[str, Any] | None, str | None]:
    scene_id = scene_dir.name
    image_dir = scene_dir / "images"
    mask_dir = scene_dir / "refined_ins_ids"
    metadata_path = scene_dir / "scene_iphone_metadata.npz"
    if not image_dir.is_dir() or not mask_dir.is_dir() or not metadata_path.is_file():
        return None, "missing_required_scene_artifact"
    try:
        metadata = np.load(metadata_path, allow_pickle=True)
        frame_names = [str(value) for value in metadata["images"].tolist()]
        trajectories = np.asarray(metadata["trajectories"], dtype=np.float64)
    except Exception as exc:  # pragma: no cover - depends on corrupt external data
        return None, f"unreadable_metadata:{type(exc).__name__}"
    image_names = {path.name for path in image_dir.glob("*.jpg")}
    mask_names = {path.name for path in mask_dir.glob("*.npy")}
    valid_indices = [
        index for index, name in enumerate(frame_names)
        if name in image_names and f"{name}.npy" in mask_names
    ]
    if len(valid_indices) < PREFIX + GOAL_OFFSET_OBSERVATIONS + 3:
        return None, "insufficient_valid_raw_frames"
    poses = trajectories[valid_indices]
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not np.isfinite(poses).all():
        return None, "invalid_pose_matrix"
    valid_frames = [frame_names[index] for index in valid_indices]
    return {
        "source": "scannetpp",
        "scene_id": scene_id,
        "scene_dir": str(scene_dir),
        "metadata_path": str(metadata_path),
        "valid_frames": valid_frames,
        "valid_pose_indices": [int(value) for value in valid_indices],
        "poses_c2w": poses,
    }, None


def motion_uniform_indices(poses_c2w: np.ndarray) -> list[int]:
    """Use the project motion-normalised metric and fixed study configuration."""
    from vidfm3d.utils.temporal_sampling import motion_uniform_observation_indices

    return [
        int(index)
        for index in motion_uniform_observation_indices(
            poses_c2w,
            motion_step=MOTION_STEP,
            rotation_weight=ROTATION_WEIGHT,
        )
    ]


def make_context(scene: dict[str, Any], anchor_index: int) -> list[str] | None:
    if anchor_index < PREFIX - 1:
        return None
    return [raw_id(scene["scene_id"], frame) for frame in scene["valid_frames"][anchor_index - PREFIX + 1 : anchor_index + 1]]


def iter_actionability_raw(scene: dict[str, Any], split: str) -> Iterable[dict[str, Any]]:
    """Yield raw-pose A0 candidates one at a time.

    Some raw scenes are long enough that materialising all pose-bearing
    dictionaries at once is needlessly close to the login-node memory limit.
    The exact candidate definition is unchanged; callers merely consume the
    universe as a stream.
    """
    observed = motion_uniform_indices(scene["poses_c2w"])
    for position in range(1, len(observed) - 1):
        anchor_index = observed[position]
        context = make_context(scene, anchor_index)
        if context is None:
            continue
        previous_index, next_index = observed[position - 1], observed[position + 1]
        if previous_index >= anchor_index - PREFIX + 1:
            continue
        candidate_indices = (previous_index, next_index)
        if any(index in range(anchor_index - PREFIX + 1, anchor_index + 1) for index in candidate_indices):
            continue
        for direction, goal_position in (("past", position - GOAL_OFFSET_OBSERVATIONS), ("future", position + GOAL_OFFSET_OBSERVATIONS)):
            if goal_position < 0 or goal_position >= len(observed):
                continue
            goal_index = observed[goal_position]
            if goal_index in candidate_indices or goal_index in range(anchor_index - PREFIX + 1, anchor_index + 1):
                continue
            anchor_pose = scene["poses_c2w"][anchor_index]
            goal_pose = scene["poses_c2w"][goal_index]
            endpoint_poses = [scene["poses_c2w"][index] for index in candidate_indices]
            utility = [
                motion_distance(anchor_pose, goal_pose) - motion_distance(endpoint, goal_pose)
                for endpoint in endpoint_poses
            ]
            candidate_motions = [motion_distance(anchor_pose, endpoint) for endpoint in endpoint_poses]
            motion_delta = abs(candidate_motions[0] - candidate_motions[1]) / max(max(candidate_motions), 1e-8)
            goal_translation, goal_rotation = pose_motion_components(anchor_pose, goal_pose)
            best = 1 if utility[0] > utility[1] else 2
            sample_id = (
                f"a0_{scene['scene_id']}_a{anchor_index:05d}_g{goal_index:05d}_"
                f"{direction}"
            )
            yield {
                    "sample_id": sample_id,
                    "split": split,
                    "variant": "A0_bidirectional_progress",
                    "scene_id": scene["scene_id"],
                    "context_raw_ids": context,
                    "anchor_raw_id": raw_id(scene["scene_id"], scene["valid_frames"][anchor_index]),
                    "goal_raw_id": raw_id(scene["scene_id"], scene["valid_frames"][goal_index]),
                    "candidate_endpoint_raw_id_1": raw_id(scene["scene_id"], scene["valid_frames"][previous_index]),
                    "candidate_endpoint_raw_id_2": raw_id(scene["scene_id"], scene["valid_frames"][next_index]),
                    "anchor_valid_index": int(anchor_index),
                    "goal_valid_index": int(goal_index),
                    "candidate_endpoint_valid_indices": [int(previous_index), int(next_index)],
                    "anchor_pose": pose_list(anchor_pose),
                    "goal_pose": pose_list(goal_pose),
                    "endpoint_pose_1": pose_list(endpoint_poses[0]),
                    "endpoint_pose_2": pose_list(endpoint_poses[1]),
                    "action_1_9d": action_9d(anchor_pose, endpoint_poses[0]),
                    "action_2_9d": action_9d(anchor_pose, endpoint_poses[1]),
                    "utility_1": float(utility[0]),
                    "utility_2": float(utility[1]),
                    "best_candidate": best,
                    "utility_margin": float(abs(utility[0] - utility[1])),
                    "goal_pose_distance": float(motion_distance(anchor_pose, goal_pose)),
                    "goal_temporal_direction": direction,
                    "translation_dominant": bool(goal_translation >= ROTATION_WEIGHT * goal_rotation),
                    "rotation_dominant": bool(goal_translation < ROTATION_WEIGHT * goal_rotation),
                    "candidate_motion_1": float(candidate_motions[0]),
                    "candidate_motion_2": float(candidate_motions[1]),
                    "candidate_motion_difference": float(motion_delta),
                    "positive_progress": bool(max(utility) > 0.0),
            }


def iter_algebra_raw(scene: dict[str, Any], split: str) -> Iterable[dict[str, Any]]:
    """Yield primitive real-trajectory paths without per-scene buffering."""
    observed = motion_uniform_indices(scene["poses_c2w"])
    for position in range(0, len(observed) - 4):
        anchor_index = observed[position]
        context = make_context(scene, anchor_index)
        if context is None:
            continue
        waypoint_indices = observed[position + 1 : position + 5]
        if len(waypoint_indices) != 4 or any(index in range(anchor_index - PREFIX + 1, anchor_index + 1) for index in waypoint_indices):
            continue
        anchor_pose = scene["poses_c2w"][anchor_index]
        path_indices = [anchor_index, *waypoint_indices]
        path_poses = [scene["poses_c2w"][index] for index in path_indices]
        primitive_actions = [action_9d(path_poses[index], path_poses[index + 1]) for index in range(4)]
        primitive_motion = [motion_distance(path_poses[index], path_poses[index + 1]) for index in range(4)]
        final_translation, final_rotation = pose_motion_components(anchor_pose, path_poses[-1])
        sample_id = f"alg_{scene['scene_id']}_a{anchor_index:05d}_h4_{waypoint_indices[-1]:05d}"
        yield {
                "sample_id": sample_id,
                "split": split,
                "record_role": "primitive_trainable_path",
                "scene_id": scene["scene_id"],
                "context_raw_ids": context,
                "anchor_raw_id": raw_id(scene["scene_id"], scene["valid_frames"][anchor_index]),
                "waypoint_raw_ids": [raw_id(scene["scene_id"], scene["valid_frames"][index]) for index in waypoint_indices],
                "horizon_target_raw_ids": {
                    "h1": raw_id(scene["scene_id"], scene["valid_frames"][waypoint_indices[0]]),
                    "h2": raw_id(scene["scene_id"], scene["valid_frames"][waypoint_indices[1]]),
                    "h4": raw_id(scene["scene_id"], scene["valid_frames"][waypoint_indices[3]]),
                },
                "final_raw_id": raw_id(scene["scene_id"], scene["valid_frames"][waypoint_indices[-1]]),
                "anchor_valid_index": int(anchor_index),
                "waypoint_valid_indices": [int(index) for index in waypoint_indices],
                "primitive_actions_9d": primitive_actions,
                "anchor_pose": pose_list(anchor_pose),
                "waypoint_poses": [pose_list(pose) for pose in path_poses[1:]],
                "final_pose": pose_list(path_poses[-1]),
                "path_length": 4,
                "cumulative_translation": float(final_translation),
                "cumulative_rotation": float(final_rotation),
                "cumulative_motion": float(motion_distance(anchor_pose, path_poses[-1])),
                "primitive_motion_magnitudes": [float(value) for value in primitive_motion],
                "real_loop_flag": False,
                "matched_non_loop_id": None,
        }


def quantile(values: list[float], amount: float, label: str) -> float:
    if not values:
        raise RuntimeError(f"cannot freeze threshold {label}: no train values")
    return float(np.quantile(np.asarray(values, dtype=np.float64), amount))


def freeze_thresholds(action_rows: list[dict[str, Any]], algebra_rows: list[dict[str, Any]]) -> dict[str, Any]:
    action_train = [row for row in action_rows if row["split"] == "train"]
    algebra_train = [row for row in algebra_rows if row["split"] == "train"]
    margins = [float(row["utility_margin"]) for row in action_train if float(row["utility_margin"]) > 1e-10]
    if not margins:
        raise RuntimeError("all train actionability candidates are exact ties")
    return {
        "schema_version": 1,
        "derived_only_from_split": "train",
        "motion_metric": {
            "implementation": "vidfm3d.utils.temporal_sampling cumulative camera motion",
            "translation_term": "Euclidean camera-center distance",
            "rotation_term": "rotation_weight * geodesic_angle_rad",
            "rotation_weight": ROTATION_WEIGHT,
            "motion_step": MOTION_STEP,
        },
        "actionability": {
            "local_motion_max_q95": quantile(
                [max(float(row["candidate_motion_1"]), float(row["candidate_motion_2"])) for row in action_train],
                0.95,
                "actionability.local_motion_max_q95",
            ),
            "utility_margin_min_q25_nonzero": quantile(margins, 0.25, "actionability.utility_margin"),
            "goal_pose_distance_min_q25": quantile(
                [float(row["goal_pose_distance"]) for row in action_train], 0.25, "actionability.goal_pose_distance"
            ),
            "candidate_motion_difference_max_q90": quantile(
                [float(row["candidate_motion_difference"]) for row in action_train], 0.90, "actionability.motion_difference"
            ),
            "requires_positive_progress": True,
        },
        "algebra": {
            "primitive_motion_max_q95": quantile(
                [max(float(value) for value in row["primitive_motion_magnitudes"]) for row in algebra_train],
                0.95,
                "algebra.primitive_motion_max_q95",
            ),
            "real_loop_close_distance": LOCAL_LOOP_CLOSE_DISTANCE,
            "real_loop_min_observations": MIN_LOOP_OBSERVATIONS,
        },
    }


def freeze_thresholds_from_train_metrics(
    action_metrics: tuple[array, array, array, array], algebra_metrics: array
) -> dict[str, Any]:
    """Freeze thresholds without retaining the large pose-bearing universe in RAM."""
    if not action_metrics or not algebra_metrics:
        raise RuntimeError("cannot freeze raw-pose thresholds without train candidates")
    motion_max, margins, goal_distance, motion_difference = action_metrics
    nonzero_margins = array("d", (value for value in margins if value > 1e-10))
    if not nonzero_margins:
        raise RuntimeError("all train actionability candidates are exact ties")
    return {
        "schema_version": 1,
        "derived_only_from_split": "train",
        "motion_metric": {
            "implementation": "vidfm3d.utils.temporal_sampling cumulative camera motion",
            "translation_term": "Euclidean camera-center distance",
            "rotation_term": "rotation_weight * geodesic_angle_rad",
            "rotation_weight": ROTATION_WEIGHT,
            "motion_step": MOTION_STEP,
        },
        "actionability": {
            "local_motion_max_q95": quantile(motion_max, 0.95, "actionability.local_motion_max_q95"),
            "utility_margin_min_q25_nonzero": quantile(nonzero_margins, 0.25, "actionability.utility_margin"),
            "goal_pose_distance_min_q25": quantile(goal_distance, 0.25, "actionability.goal_pose_distance"),
            "candidate_motion_difference_max_q90": quantile(motion_difference, 0.90, "actionability.motion_difference"),
            "requires_positive_progress": True,
        },
        "algebra": {
            "primitive_motion_max_q95": quantile(algebra_metrics, 0.95, "algebra.primitive_motion_max_q95"),
            "real_loop_close_distance": LOCAL_LOOP_CLOSE_DISTANCE,
            "real_loop_min_observations": MIN_LOOP_OBSERVATIONS,
        },
    }


def actionability_rejection(row: dict[str, Any], thresholds: dict[str, Any]) -> str | None:
    config = thresholds["actionability"]
    if not row["positive_progress"]:
        return "no_positive_progress"
    if row["utility_margin"] <= config["utility_margin_min_q25_nonzero"]:
        return "utility_margin_near_tie"
    if row["goal_pose_distance"] < config["goal_pose_distance_min_q25"]:
        return "goal_distance_too_small"
    if max(row["candidate_motion_1"], row["candidate_motion_2"]) > config["local_motion_max_q95"]:
        return "candidate_outside_local_motion_envelope"
    if row["candidate_motion_difference"] > config["candidate_motion_difference_max_q90"]:
        return "candidate_motion_mismatch"
    return None


def algebra_rejection(row: dict[str, Any], thresholds: dict[str, Any]) -> str | None:
    if max(row["primitive_motion_magnitudes"]) > thresholds["algebra"]["primitive_motion_max_q95"]:
        return "primitive_action_outside_local_motion_envelope"
    return None


def retain_lowest(pool: dict[str, list[dict[str, Any]]], scene_id: str, row: dict[str, Any], limit: int) -> None:
    """Retain only deterministic low-rank rows needed under the scene cap."""
    values = pool.setdefault(scene_id, [])
    values.append(row)
    values.sort(key=lambda item: (stable_rank(item["sample_id"]), item["sample_id"]))
    del values[limit:]


def select_action_pilot(
    pools: dict[tuple[str, str, int], dict[str, dict[str, Any]]], split: str, target: int
) -> list[dict[str, Any]]:
    groups = sorted({group for row_split, direction, winner in pools for group in [(direction, winner)] if row_split == split})
    if not groups:
        return []
    quotas = {group: target // len(groups) for group in groups}
    for group in groups[: target % len(groups)]:
        quotas[group] += 1
    selected: list[dict[str, Any]] = []
    for direction, winner in groups:
        values = list(pools.get((split, direction, winner), {}).values())
        values.sort(key=lambda row: (stable_rank(row["sample_id"]), row["sample_id"]))
        selected.extend(values[: quotas[(direction, winner)]])
    return sorted(selected, key=lambda row: row["sample_id"])


def select_algebra_pilot(pools: dict[str, dict[str, list[dict[str, Any]]]], split: str, target: int) -> list[dict[str, Any]]:
    values = [row for scene_rows in pools.get(split, {}).values() for row in scene_rows]
    values.sort(key=lambda row: (stable_rank(row["sample_id"]), row["sample_id"]))
    return sorted(values[:target], key=lambda row: row["sample_id"])


def actionability_eligible(rows: list[dict[str, Any]], thresholds: dict[str, Any]) -> tuple[list[dict[str, Any]], Counter[str]]:
    config = thresholds["actionability"]
    accepted: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    for row in rows:
        if not row["positive_progress"]:
            rejected["no_positive_progress"] += 1
        elif row["utility_margin"] <= config["utility_margin_min_q25_nonzero"]:
            rejected["utility_margin_near_tie"] += 1
        elif row["goal_pose_distance"] < config["goal_pose_distance_min_q25"]:
            rejected["goal_distance_too_small"] += 1
        elif max(row["candidate_motion_1"], row["candidate_motion_2"]) > config["local_motion_max_q95"]:
            rejected["candidate_outside_local_motion_envelope"] += 1
        elif row["candidate_motion_difference"] > config["candidate_motion_difference_max_q90"]:
            rejected["candidate_motion_mismatch"] += 1
        else:
            accepted.append(row)
    return accepted, rejected


def algebra_eligible(rows: list[dict[str, Any]], thresholds: dict[str, Any]) -> tuple[list[dict[str, Any]], Counter[str]]:
    maximum = thresholds["algebra"]["primitive_motion_max_q95"]
    accepted: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    for row in rows:
        if max(row["primitive_motion_magnitudes"]) > maximum:
            rejected["primitive_action_outside_local_motion_envelope"] += 1
        else:
            accepted.append(row)
    return accepted, rejected


def balanced_select(
    rows: list[dict[str, Any]], *, target: int, per_scene_cap: int, group_key
) -> list[dict[str, Any]]:
    """Deterministic, label-balanced selection with a hard scene cap."""
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(group_key(row))].append(row)
    if not grouped:
        return []
    quotas = {key: target // len(grouped) for key in grouped}
    for key in sorted(quotas)[: target % len(grouped)]:
        quotas[key] += 1
    selected: list[dict[str, Any]] = []
    scene_counts: Counter[str] = Counter()
    for key in sorted(grouped):
        candidates = sorted(grouped[key], key=lambda row: (stable_rank(row["sample_id"]), row["sample_id"]))
        want = quotas[key]
        for row in candidates:
            if len([item for item in selected if tuple(group_key(item)) == key]) >= want:
                break
            if scene_counts[row["scene_id"]] >= per_scene_cap:
                continue
            selected.append(row)
            scene_counts[row["scene_id"]] += 1
    # If a group cannot satisfy its quota, fill from remaining eligible rows without changing the scene cap.
    if len(selected) < target:
        used = {row["sample_id"] for row in selected}
        remainder = sorted((row for row in rows if row["sample_id"] not in used), key=lambda row: (stable_rank(row["sample_id"]), row["sample_id"]))
        for row in remainder:
            if len(selected) >= target:
                break
            if scene_counts[row["scene_id"]] >= per_scene_cap:
                continue
            selected.append(row)
            scene_counts[row["scene_id"]] += 1
    return sorted(selected, key=lambda row: row["sample_id"])


def build_real_loop_rows(scenes: list[dict[str, Any]], split_by_scene: dict[str, str]) -> list[dict[str, Any]]:
    """Find only real observed loop closures; never create synthetic inverse paths."""
    rows: list[dict[str, Any]] = []
    for scene in scenes:
        observed = motion_uniform_indices(scene["poses_c2w"])
        poses = scene["poses_c2w"]
        found = 0
        for start_pos in range(0, max(0, len(observed) - MIN_LOOP_OBSERVATIONS)):
            if found >= 4:
                break
            start_index = observed[start_pos]
            context = make_context(scene, start_index)
            if context is None:
                continue
            candidate_positions = np.asarray(range(start_pos + MIN_LOOP_OBSERVATIONS, len(observed)), dtype=np.int64)
            if candidate_positions.size == 0:
                continue
            candidate_indices = np.asarray([observed[int(value)] for value in candidate_positions], dtype=np.int64)
            center_distance = np.linalg.norm(poses[candidate_indices, :3, 3] - poses[start_index, :3, 3], axis=1)
            rotations = np.asarray([
                rotation_angle_rad(poses[start_index, :3, :3], poses[index, :3, :3])
                for index in candidate_indices
            ])
            close = center_distance + ROTATION_WEIGHT * rotations <= LOCAL_LOOP_CLOSE_DISTANCE
            if not bool(close.any()):
                continue
            end_pos = int(candidate_positions[np.flatnonzero(close)[0]])
            end_index = observed[end_pos]
            # A matched non-loop has the same start and nearest available path length,
            # but is visibly outside the fixed loop envelope.
            loop_len = end_pos - start_pos
            nonloop_options = [
                candidate for candidate in range(start_pos + MIN_LOOP_OBSERVATIONS, len(observed))
                if motion_distance(poses[start_index], poses[observed[candidate]]) > 4.0 * LOCAL_LOOP_CLOSE_DISTANCE
            ]
            matched_pos = min(nonloop_options, key=lambda value: (abs((value - start_pos) - loop_len), value)) if nonloop_options else None
            loop_id = f"loop_{scene['scene_id']}_a{start_index:05d}_e{end_index:05d}"
            matched_id = None
            if matched_pos is not None:
                matched_index = observed[matched_pos]
                matched_id = f"nonloop_{scene['scene_id']}_a{start_index:05d}_e{matched_index:05d}"
            rows.append(
                {
                    "sample_id": loop_id,
                    "split": split_by_scene[scene["scene_id"]],
                    "record_role": "real_loop_evaluation_only",
                    "scene_id": scene["scene_id"],
                    "context_raw_ids": context,
                    "anchor_raw_id": raw_id(scene["scene_id"], scene["valid_frames"][start_index]),
                    "final_raw_id": raw_id(scene["scene_id"], scene["valid_frames"][end_index]),
                    "anchor_valid_index": int(start_index),
                    "final_valid_index": int(end_index),
                    "path_length": int(loop_len),
                    "anchor_pose": pose_list(poses[start_index]),
                    "final_pose": pose_list(poses[end_index]),
                    "closure_motion_distance": float(motion_distance(poses[start_index], poses[end_index])),
                    "real_loop_flag": True,
                    "matched_non_loop_id": matched_id,
                }
            )
            if matched_pos is not None:
                matched_index = observed[matched_pos]
                rows.append(
                    {
                        "sample_id": matched_id,
                        "split": split_by_scene[scene["scene_id"]],
                        "record_role": "matched_non_loop_evaluation_only",
                        "scene_id": scene["scene_id"],
                        "context_raw_ids": context,
                        "anchor_raw_id": raw_id(scene["scene_id"], scene["valid_frames"][start_index]),
                        "final_raw_id": raw_id(scene["scene_id"], scene["valid_frames"][matched_index]),
                        "anchor_valid_index": int(start_index),
                        "final_valid_index": int(matched_index),
                        "path_length": int(matched_pos - start_pos),
                        "anchor_pose": pose_list(poses[start_index]),
                        "final_pose": pose_list(poses[matched_index]),
                        "closure_motion_distance": float(motion_distance(poses[start_index], poses[matched_index])),
                        "real_loop_flag": False,
                        "matched_non_loop_id": loop_id,
                    }
                )
            found += 1
    return sorted(rows, key=lambda row: row["sample_id"])


def collect_feature_requests(
    action_rows: list[dict[str, Any]], algebra_rows: list[dict[str, Any]], loop_rows: list[dict[str, Any]], manifest_hashes: dict[str, str]
) -> list[dict[str, Any]]:
    requested: dict[tuple[str, int, int | None, str, str, str], set[str]] = defaultdict(set)

    def add(model: dict[str, Any], scene_id: str, frame_id: str, role: str, sample_id: str) -> None:
        key = (model["model"], int(model["layer"]), model["timestep"], scene_id, frame_id, role)
        requested[key].add(sample_id)

    for row in action_rows:
        for model in MODEL_SETTINGS:
            for context_id in row["context_raw_ids"]:
                add(model, row["scene_id"], context_id, "streaming_context_p8", row["sample_id"])
            add(model, row["scene_id"], row["anchor_raw_id"], "isolated_anchor", row["sample_id"])
            add(model, row["scene_id"], row["goal_raw_id"], "isolated_goal", row["sample_id"])
            add(model, row["scene_id"], row["candidate_endpoint_raw_id_1"], "isolated_candidate_endpoint_1", row["sample_id"])
            add(model, row["scene_id"], row["candidate_endpoint_raw_id_2"], "isolated_candidate_endpoint_2", row["sample_id"])
    for row in algebra_rows:
        for model in MODEL_SETTINGS:
            for context_id in row["context_raw_ids"]:
                add(model, row["scene_id"], context_id, "streaming_context_p8", row["sample_id"])
            add(model, row["scene_id"], row["anchor_raw_id"], "isolated_anchor", row["sample_id"])
            for horizon, role in (("h1", "isolated_h1"), ("h2", "isolated_h2"), ("h4", "isolated_h4")):
                add(model, row["scene_id"], row["horizon_target_raw_ids"][horizon], role, row["sample_id"])
    for row in loop_rows:
        for model in MODEL_SETTINGS:
            for context_id in row["context_raw_ids"]:
                add(model, row["scene_id"], context_id, "streaming_context_p8", row["sample_id"])
            add(model, row["scene_id"], row["anchor_raw_id"], "isolated_anchor", row["sample_id"])
            add(model, row["scene_id"], row["final_raw_id"], "isolated_loop_endpoint", row["sample_id"])
    rows = []
    for key, sample_ids in sorted(requested.items()):
        model, layer, timestep, scene_id, frame_id, role = key
        rows.append(
            {
                "request_id": "fr_" + hashlib.sha256(canonical_json(key).encode("utf-8")).hexdigest()[:20],
                "model": model,
                "layer": layer,
                "timestep": timestep,
                "scene_id": scene_id,
                "raw_frame_id": frame_id,
                "feature_role": role,
                "source_sample_ids": sorted(sample_ids),
                "source_manifest_sha256": manifest_hashes,
                "exact_raw_id_required": True,
                "fallback_forbidden": True,
            }
        )
    return rows


def validate_rows(action_rows: list[dict[str, Any]], algebra_rows: list[dict[str, Any]]) -> dict[str, Any]:
    action_ids = [row["sample_id"] for row in action_rows]
    algebra_ids = [row["sample_id"] for row in algebra_rows]
    action_raw = [
        (row["scene_id"], raw)
        for row in action_rows
        for raw in [*row["context_raw_ids"], row["anchor_raw_id"], row["goal_raw_id"], row["candidate_endpoint_raw_id_1"], row["candidate_endpoint_raw_id_2"]]
    ]
    algebra_raw = [
        (row["scene_id"], raw)
        for row in algebra_rows
        for raw in [*row["context_raw_ids"], row["anchor_raw_id"], *row["waypoint_raw_ids"]]
    ]
    return {
        "action_duplicate_sample_ids": len(action_ids) - len(set(action_ids)),
        "algebra_duplicate_sample_ids": len(algebra_ids) - len(set(algebra_ids)),
        "action_raw_id_records": len(action_raw),
        "action_unique_scene_raw_ids": len(set(action_raw)),
        "algebra_raw_id_records": len(algebra_raw),
        "algebra_unique_scene_raw_ids": len(set(algebra_raw)),
        "invalid_action_context_lengths": sum(len(row["context_raw_ids"]) != PREFIX for row in action_rows),
        "invalid_algebra_context_lengths": sum(len(row["context_raw_ids"]) != PREFIX for row in algebra_rows),
        "scene_overlap_train_val_action": sorted(
            set(row["scene_id"] for row in action_rows if row["split"] == "train")
            & set(row["scene_id"] for row in action_rows if row["split"] == "val")
        ),
        "scene_overlap_train_val_algebra": sorted(
            set(row["scene_id"] for row in algebra_rows if row["split"] == "train")
            & set(row["scene_id"] for row in algebra_rows if row["split"] == "val")
        ),
    }


def markdown_audit(science_tag: str, provenance: dict[str, Any]) -> str:
    submit = provenance["submit_sh"]
    return f"""# Independent p8 Actionability + Algebra audit\n\n+Study: `{science_tag}`\n\n+Status: `COMPLETE_VALID` for scope quarantine and raw-pose setup.\n\n+This study is independent of the historical C-series.  It does not load old C1/C2 checkpoints, old target lineage, canonical result CSVs, or registry entries.  The historical raw-ID mismatch is recorded as the reason for this isolation, not as a reason to manufacture `n=0`.\n\n+## Frozen pre-study identity\n\n+- Git HEAD: `{provenance['git']['head']}`\n+- `git diff --binary` SHA256: `{provenance['git']['diff_binary_sha256']}`\n+- Critical source SHA256: `artifacts/{science_tag}/provenance/start_state.json`\n+- `submit.sh`: size `{submit['size_bytes']}` bytes; SHA256 `{submit['sha256']}`\n\n+## Scope\n\n+- Models: DINO L0; F3R L24; V-JEPA2 L23; Wan L20/t749.\n+- Prefix: 8 only.\n+- Explicitly excluded: B2, old C checkpoint reuse, canonical registry/results edits, p12/p16/p24, sweeps, text goals, online RL, and backbone fine-tuning.\n+"""


def markdown_feasibility(summary: dict[str, Any]) -> str:
    action = summary["actionability"]
    algebra = summary["algebra"]
    return f"""# Raw-pose feasibility\n\n+Status: `COMPLETE_VALID`\n\n+All universes were built directly from observed ScanNet++ image/mask-backed raw frames and camera trajectories.  Distances use the project motion-normalized camera metric: translation plus `0.5 ×` rotation geodesic angle in radians.\n\n+## A0 bidirectional progress\n\n+- Raw candidates: {action['raw_candidates']}\n+- Eligible after train-only thresholds: {action['eligible_candidates']}\n+- Frozen train / val pilot: {action['train_samples']} / {action['val_samples']}\n+- Train / val scenes: {action['train_scenes']} / {action['val_scenes']}\n+- A1 observed-local-branch: `NOT_RUN` — no branch graph is inferred from a single observed trajectory; no synthetic branch is substituted.\n\n+## Clean dynamics / algebra\n\n+- Raw primitive paths: {algebra['raw_candidates']}\n+- Eligible primitive paths: {algebra['eligible_candidates']}\n+- Frozen train / val pilot: {algebra['train_samples']} / {algebra['val_samples']}\n+- Real observed loop records: {algebra['real_loop_records']}\n\n+Thresholds, rejection reasons, split SHA, and uniqueness checks are in `manifests/manifest_summary.json`.\n+"""


def markdown_feature_audit(feature_summary: dict[str, Any], provenance: dict[str, Any]) -> str:
    submit = provenance["submit_sh"]
    submit_status = "BLOCKED_ENVIRONMENT" if submit.get("size_bytes") in {None, 0, 1} else "NOT_RUN"
    return f"""# Feature lineage audit\n\n+Status: `COMPLETE_WITH_CAVEAT`\n\n+An exact raw-ID feature request manifest has been frozen.  No historical feature is accepted by filename/window/tail similarity.  Reuse requires a sidecar or indexed extractor record proving each requested actual raw ID.\n\n+- Requests: {feature_summary['request_rows']}\n+- Unique requested `(model, layer, timestep, scene, raw-ID)` tuples: {feature_summary['unique_feature_units']}\n+- Missing/reuse resolution: `NOT_RUN`\n+- Fallback policy: forbidden\n+- SCO submission prerequisite: `{submit_status}` because `/mnt/umm/users/yinbaiqiao/submit.sh` is {submit.get('size_bytes')} bytes.  Only a verified Cursor History copy may restore it.\n\n+The next valid action is an exact sidecar audit followed by a DINO-only raw-ID extraction/load smoke for missing requests.\n+"""


def write_status_matrix(path: Path, science_tag: str, submit_size: int | None) -> None:
    rows = [
        ("scope_quarantine", "COMPLETE_VALID", "Legacy C1/C2 excluded; no canonical output touched"),
        ("raw_pose_universe", "COMPLETE_VALID", "Independent scene split and observed raw IDs frozen"),
        ("feature_request_manifest", "COMPLETE_VALID", "Positive exact raw-ID allow-list frozen"),
        ("feature_lineage_resolution", "NOT_RUN", "Requires sidecar/index audit"),
        ("dino_feature_load_smoke", "NOT_RUN", "Requires exact feature resolution"),
        ("dino_direct_actionability", "NOT_RUN", "Requires DINO coverage gate"),
        ("dino_clean_dynamics_algebra", "NOT_RUN", "Requires DINO coverage gate"),
        ("f3r_vjepa2_wan_expansion", "NOT_RUN", "Requires DINO validation"),
        ("sco_submission_adapter", "BLOCKED_ENVIRONMENT" if submit_size in {None, 0, 1} else "NOT_RUN", "submit.sh recovery required" if submit_size in {None, 0, 1} else "not yet needed"),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["science_tag", "component", "status", "detail"])
        for component, status, detail in rows:
            writer.writerow([science_tag, component, status, detail])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--science-tag", required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument(
        "--scene-root",
        type=Path,
        default=Path("/mnt/umm/users/yinbaiqiao/InsScene-15K/data/processed_scannetpp_v2/processed_scannetpp_v2"),
    )
    parser.add_argument("--submit-script", type=Path, default=Path("/mnt/umm/users/yinbaiqiao/submit.sh"))
    parser.add_argument("--train-target", type=int, default=1024)
    parser.add_argument("--val-target", type=int, default=256)
    parser.add_argument("--per-scene-cap", type=int, default=4)
    args = parser.parse_args()

    if not args.science_tag.startswith("independent_actionability_algebra_p8_"):
        raise SystemExit("science tag must declare the independent p8 study")
    artifact_root = args.artifacts_root.resolve()
    reports_root = args.reports_root.resolve()
    if artifact_root.exists() or reports_root.exists():
        raise SystemExit("refusing to overwrite an existing independent study root")
    if not args.scene_root.is_dir():
        raise SystemExit(f"scene root is unavailable: {args.scene_root}")
    if args.train_target < 256 or args.val_target < 64 or args.per_scene_cap < 1:
        raise SystemExit("pilot sizes must meet the requested minimums and scene cap must be positive")

    staging_artifacts = artifact_root.with_name(f".{artifact_root.name}.tmp-{os.getpid()}")
    staging_reports = reports_root.with_name(f".{reports_root.name}.tmp-{os.getpid()}")
    staging_artifacts.mkdir(parents=True)
    staging_reports.mkdir(parents=True)
    try:
        provenance = git_provenance(Path(__file__).resolve(), args.submit_script)
        atomic_json(staging_artifacts / "provenance" / "start_state.json", provenance)

        scenes: list[dict[str, Any]] = []
        rejected_scenes: Counter[str] = Counter()
        candidate_dirs = sorted(
            (path for path in args.scene_root.iterdir() if path.is_dir()),
            key=lambda path: (stable_rank(f"independent-p8-v1:{path.name}"), path.name),
        )
        for scene_dir in candidate_dirs:
            scene, reason = collect_scene(scene_dir)
            if scene is None:
                rejected_scenes[str(reason)] += 1
            else:
                scenes.append(scene)
        if len(scenes) < 80:
            raise RuntimeError(f"only {len(scenes)} raw-pose scenes are usable")

        ranked_scenes = sorted(scenes, key=lambda scene: (stable_rank(f"independent-p8-v1:{scene['scene_id']}"), scene["scene_id"]))
        train_count = int(round(0.8 * len(ranked_scenes)))
        split_by_scene = {
            scene["scene_id"]: ("train" if index < train_count else "val")
            for index, scene in enumerate(ranked_scenes)
        }
        split_payload = {
            "schema_version": 1,
            "study": args.science_tag,
            "source": "raw ScanNet++ camera poses and observed images",
            "split_method": "sha256-ranked deterministic scene split",
            "seed_material": "independent-p8-v1",
            "train_ratio": 0.8,
            "splits": {
                "train": sorted(scene_id for scene_id, split in split_by_scene.items() if split == "train"),
                "val": sorted(scene_id for scene_id, split in split_by_scene.items() if split == "val"),
            },
            "scene_root": str(args.scene_root),
            "scene_directory_candidates": len(candidate_dirs),
            "study_scene_count_requested": "all eligible raw scenes",
            "scene_root_count": len(scenes),
            "rejected_scenes": dict(sorted(rejected_scenes.items())),
        }
        atomic_json(staging_artifacts / "manifests" / "scene_split_independent_v1.json", split_payload)

        # First pass: derive every threshold from the complete train universe
        # without retaining pose-bearing candidate records in memory.
        action_motion_max = array("d")
        action_margins = array("d")
        action_goal_distance = array("d")
        action_motion_difference = array("d")
        algebra_metrics = array("d")
        raw_action_count = 0
        raw_algebra_count = 0
        for scene in scenes:
            split = split_by_scene[scene["scene_id"]]
            for row in iter_actionability_raw(scene, split):
                raw_action_count += 1
                if split == "train":
                    action_motion_max.append(max(float(row["candidate_motion_1"]), float(row["candidate_motion_2"])))
                    action_margins.append(float(row["utility_margin"]))
                    action_goal_distance.append(float(row["goal_pose_distance"]))
                    action_motion_difference.append(float(row["candidate_motion_difference"]))
            for row in iter_algebra_raw(scene, split):
                raw_algebra_count += 1
                if split == "train":
                    algebra_metrics.append(max(float(value) for value in row["primitive_motion_magnitudes"]))
        thresholds = freeze_thresholds_from_train_metrics(
            (action_motion_max, action_margins, action_goal_distance, action_motion_difference), algebra_metrics
        )

        # Second pass: retain only rows that can survive the fixed scene cap and
        # balance quotas.  This keeps a full eligible-universe audit while
        # avoiding a large in-memory pose duplication.
        action_pools: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
        algebra_pools: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
        action_eligible_count = 0
        algebra_eligible_count = 0
        action_rejected: Counter[str] = Counter()
        algebra_rejected: Counter[str] = Counter()
        for scene in scenes:
            split = split_by_scene[scene["scene_id"]]
            for row in iter_actionability_raw(scene, split):
                rejection = actionability_rejection(row, thresholds)
                if rejection is not None:
                    action_rejected[rejection] += 1
                    continue
                action_eligible_count += 1
                key = (split, str(row["goal_temporal_direction"]), int(row["best_candidate"]))
                prior = action_pools[key].get(scene["scene_id"])
                if prior is None or (stable_rank(row["sample_id"]), row["sample_id"]) < (stable_rank(prior["sample_id"]), prior["sample_id"]):
                    action_pools[key][scene["scene_id"]] = row
            for row in iter_algebra_raw(scene, split):
                rejection = algebra_rejection(row, thresholds)
                if rejection is not None:
                    algebra_rejected[rejection] += 1
                    continue
                algebra_eligible_count += 1
                retain_lowest(algebra_pools[split], scene["scene_id"], row, args.per_scene_cap)
        action_train = select_action_pilot(action_pools, "train", args.train_target)
        action_val = select_action_pilot(action_pools, "val", args.val_target)
        algebra_train = select_algebra_pilot(algebra_pools, "train", args.train_target)
        algebra_val = select_algebra_pilot(algebra_pools, "val", args.val_target)
        if len(action_train) < 256 or len(action_val) < 64:
            raise RuntimeError(f"actionability pilot below minimum: train={len(action_train)} val={len(action_val)}")
        if len(algebra_train) < 256 or len(algebra_val) < 64:
            raise RuntimeError(f"algebra pilot below minimum: train={len(algebra_train)} val={len(algebra_val)}")
        # Real-loop search is evaluation-only and scoped to validation scenes;
        # no loop or synthetic inverse can enter the training manifest.
        algebra_val_scenes = {row["scene_id"] for row in algebra_val}
        loop_rows = build_real_loop_rows(
            [scene for scene in scenes if scene["scene_id"] in algebra_val_scenes], split_by_scene
        )

        manifests = staging_artifacts / "manifests"
        action_train_path = manifests / "actionability_p8_independent_v1_train.jsonl"
        action_val_path = manifests / "actionability_p8_independent_v1_val.jsonl"
        algebra_train_path = manifests / "algebra_p8_independent_v1_train.jsonl"
        algebra_val_path = manifests / "algebra_p8_independent_v1_val.jsonl"
        loop_path = manifests / "algebra_p8_independent_v1_real_loops.jsonl"
        atomic_jsonl(action_train_path, action_train)
        atomic_jsonl(action_val_path, action_val)
        atomic_jsonl(algebra_train_path, algebra_train)
        atomic_jsonl(algebra_val_path, algebra_val)
        atomic_jsonl(loop_path, loop_rows)
        manifest_hashes = {
            "actionability_train": sha256_file(action_train_path),
            "actionability_val": sha256_file(action_val_path),
            "algebra_train": sha256_file(algebra_train_path),
            "algebra_val": sha256_file(algebra_val_path),
            "algebra_real_loops": sha256_file(loop_path),
            "scene_split": sha256_file(manifests / "scene_split_independent_v1.json"),
        }
        feature_requests = collect_feature_requests(
            [*action_train, *action_val], [*algebra_train, *algebra_val], loop_rows, manifest_hashes
        )
        feature_path = manifests / "feature_request_manifest.jsonl"
        atomic_jsonl(feature_path, feature_requests)
        validation = validate_rows([*action_train, *action_val], [*algebra_train, *algebra_val])
        if any(validation[key] for key in ("action_duplicate_sample_ids", "algebra_duplicate_sample_ids", "invalid_action_context_lengths", "invalid_algebra_context_lengths", "scene_overlap_train_val_action", "scene_overlap_train_val_algebra")):
            raise RuntimeError(f"manifest integrity failure: {validation}")
        summary = {
            "schema_version": 1,
            "science_tag": args.science_tag,
            "status": "COMPLETE_VALID",
            "settings": {
                "prefix": PREFIX,
                "models": MODEL_SETTINGS,
                "goal_offset_observations": GOAL_OFFSET_OBSERVATIONS,
                "per_scene_cap": args.per_scene_cap,
                "train_target": args.train_target,
                "val_target": args.val_target,
            },
            "thresholds": thresholds,
            "manifest_sha256": manifest_hashes,
            "actionability": {
                "raw_candidates": raw_action_count,
                "eligible_candidates": action_eligible_count,
                "rejection_reasons": dict(sorted(action_rejected.items())),
                "train_samples": len(action_train),
                "val_samples": len(action_val),
                "train_scenes": len({row["scene_id"] for row in action_train}),
                "val_scenes": len({row["scene_id"] for row in action_val}),
                "train_strata": Counter(f"{row['goal_temporal_direction']}:candidate_{row['best_candidate']}" for row in action_train),
                "val_strata": Counter(f"{row['goal_temporal_direction']}:candidate_{row['best_candidate']}" for row in action_val),
                "a1_observed_branch_status": "NOT_RUN_NO_SYNTHETIC_BRANCH_SUBSTITUTE",
            },
            "algebra": {
                "raw_candidates": raw_algebra_count,
                "eligible_candidates": algebra_eligible_count,
                "rejection_reasons": dict(sorted(algebra_rejected.items())),
                "train_samples": len(algebra_train),
                "val_samples": len(algebra_val),
                "train_scenes": len({row["scene_id"] for row in algebra_train}),
                "val_scenes": len({row["scene_id"] for row in algebra_val}),
                "real_loop_records": len(loop_rows),
                "real_loop_pairs": sum(row["record_role"] == "real_loop_evaluation_only" for row in loop_rows),
            },
            "integrity": validation,
        }
        atomic_json(manifests / "manifest_summary.json", summary)
        feature_summary = {
            "schema_version": 1,
            "science_tag": args.science_tag,
            "request_rows": len(feature_requests),
            "unique_feature_units": len({(row["model"], row["layer"], row["timestep"], row["scene_id"], row["raw_frame_id"]) for row in feature_requests}),
            "roles": dict(sorted(Counter(row["feature_role"] for row in feature_requests).items())),
            "per_model": dict(sorted(Counter(row["model"] for row in feature_requests).items())),
            "manifest_sha256": sha256_file(feature_path),
            "reuse_policy": "actual raw-ID proof only; filename/window/tail inference forbidden",
            "fallback_policy": "forbidden",
        }
        atomic_json(manifests / "feature_request_summary.json", feature_summary)

        atomic_text(staging_reports / "00_AUDIT.md", markdown_audit(args.science_tag, provenance))
        atomic_text(staging_reports / "RAW_POSE_FEASIBILITY.md", markdown_feasibility(summary))
        atomic_text(staging_reports / "FEATURE_LINEAGE_AUDIT.md", markdown_feature_audit(feature_summary, provenance))
        write_status_matrix(staging_reports / "STATUS_MATRIX.csv", args.science_tag, provenance["submit_sh"].get("size_bytes"))

        os.replace(staging_artifacts, artifact_root)
        os.replace(staging_reports, reports_root)
        print(json.dumps({"science_tag": args.science_tag, "artifact_root": str(artifact_root), "reports_root": str(reports_root), "summary": summary, "feature_summary": feature_summary}, indent=2, default=dict))
        return 0
    except Exception:
        shutil.rmtree(staging_artifacts, ignore_errors=True)
        shutil.rmtree(staging_reports, ignore_errors=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
