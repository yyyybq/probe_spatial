"""Model-independent synthetic-branch residual diagnostic.

The observed-video protocol is the primary experiment.  This module retains
the earlier dual-image/synthetic-pose branch manifest only as an optional
future extension; it must never choose models, block observed-order runs, or
turn nearest observed frames into ground truth.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from analyses.se3_latent_mvp.geometry import camera_to_world_to_extrinsic, relative_pose, se3_log
from analyses.se3_protocol_v2.geometry import (
    branch_poses,
    group_commutator,
    pose_residual,
    rotation_degrees,
    translation_norm,
)


POSE_CONVENTION = {
    "raw_pose": "ScanNet++ scene_iphone_metadata.npz trajectories are camera_to_world",
    "extrinsic": "E = inv(camera_to_world), world_to_camera, column-vector convention",
    "relative": "T_AB = E_B @ inv(E_A), maps camera-A coordinates to camera-B coordinates",
    "branch": "E_12 = T2 @ T1 @ E_A; E_21 = T1 @ T2 @ E_A",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def dump_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=as_json) + "\n")


def dump_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=as_json) + "\n")


@dataclass(frozen=True)
class SceneRecord:
    scene_id: str
    original_split: str
    valid_frames: tuple[str, ...]
    trajectories_c2w: np.ndarray
    extrinsics_w2c: np.ndarray
    intrinsics: np.ndarray
    root: Path

    @property
    def extrinsics(self) -> np.ndarray:
        return self.extrinsics_w2c


def load_manifest_scenes(manifest: Path, raw_root: Path) -> dict[str, str]:
    splits: dict[str, str] = {}
    with manifest.open() as handle:
        for line in handle:
            row = json.loads(line)
            scene = str(row["scene_id"])
            split = str(row["split"])
            old = splits.setdefault(scene, split)
            if old != split:
                raise RuntimeError(f"scene {scene} has inconsistent manifest splits")
    missing = sorted(scene for scene in splits if not (raw_root / scene / "scene_iphone_metadata.npz").is_file())
    if missing:
        raise RuntimeError(f"manifest scenes missing pose metadata: {missing[:5]}")
    return splits


def load_scene(scene_id: str, original_split: str, raw_root: Path) -> SceneRecord:
    root = raw_root / scene_id
    with np.load(root / "scene_iphone_metadata.npz") as data:
        trajectories = np.asarray(data["trajectories"], dtype=np.float64)
        intrinsics = np.asarray(data["intrinsics"], dtype=np.float64)
        frames = tuple(str(value) for value in data["images"].tolist())
    if not (len(trajectories) == len(intrinsics) == len(frames)):
        raise RuntimeError(f"{scene_id}: pose/intrinsic/image length mismatch")
    extrinsics = np.stack([camera_to_world_to_extrinsic(pose) for pose in trajectories])
    return SceneRecord(scene_id, original_split, frames, trajectories, extrinsics, intrinsics, root)


def optional_dual_image_extension() -> dict[str, Any]:
    """Declare the former renderer path out of scope without searching for it."""
    return {
        "track": "optional_dual_image_grounding",
        "status": "N/A_not_a_gate_not_audited",
        "policy": "Do not search for or construct mesh, NeRF, 3DGS, or ScanNet++ renderers in protocol-v2.",
        "nearest_observed_policy": "residual diagnostic only; never a main-experiment target feature.",
    }


def edge_motions(scene: SceneRecord, stride: int = 1, sample_step: int = 1) -> list[dict[str, Any]]:
    extrinsics = scene.extrinsics
    result: list[dict[str, Any]] = []
    for source in range(0, len(extrinsics) - stride, sample_step):
        target = source + stride
        transform = relative_pose(extrinsics[source], extrinsics[target])
        result.append(
            {
                "scene_id": scene.scene_id,
                "source_index": source,
                "target_index": target,
                "transform": transform,
                "rotation_deg": rotation_degrees(transform),
                "translation_m": translation_norm(transform),
            }
        )
    return result


def motion_quantiles(motions: list[dict[str, Any]]) -> dict[str, float]:
    if not motions:
        raise RuntimeError("cannot form quantiles from zero train primitive transitions")
    rotation = np.asarray([row["rotation_deg"] for row in motions])
    translation = np.asarray([row["translation_m"] for row in motions])
    result: dict[str, float] = {}
    for name, values in (("rotation_deg", rotation), ("translation_m", translation)):
        for quantile in (0.1, 0.25, 0.5, 0.75, 0.9):
            result[f"{name}_q{int(quantile * 100):02d}"] = float(np.quantile(values, quantile))
    return result


def _motion_axis(transform: np.ndarray) -> np.ndarray | None:
    omega = se3_log(transform)[:3]
    norm = float(np.linalg.norm(omega))
    return None if norm < 1e-12 else omega / norm


def branch_family(t1: np.ndarray, t2: np.ndarray, quantiles: dict[str, float]) -> str | None:
    r1, r2 = rotation_degrees(t1), rotation_degrees(t2)
    d1, d2 = translation_norm(t1), translation_norm(t2)
    q = quantiles
    comm = group_commutator(t1, t2)
    c = translation_norm(comm) + math.radians(rotation_degrees(comm))
    # Ordered deliberately: a rotation/translation pair is the principal positive.
    r_hi = q["rotation_deg_q75"]
    r_lo = q["rotation_deg_q50"]
    d_hi = q["translation_m_q75"]
    if (r1 >= r_hi and d2 >= d_hi and r2 <= r_lo) or (r2 >= r_hi and d1 >= d_hi and r1 <= r_lo):
        return "positive_rotation_translation"
    axis1, axis2 = _motion_axis(t1), _motion_axis(t2)
    if axis1 is not None and axis2 is not None and r1 >= r_hi and r2 >= r_hi and abs(float(axis1 @ axis2)) <= 0.8:
        return "positive_different_axis_rotation_rotation"
    if r1 <= q["rotation_deg_q25"] and r2 <= q["rotation_deg_q25"] and d1 >= q["translation_m_q50"] and d2 >= q["translation_m_q50"]:
        return "negative_translation_dominant_translation_dominant"
    if c <= math.radians(q["rotation_deg_q10"]) + q["translation_m_q10"]:
        return "negative_near_commuting"
    return None


def split_scene_ids(original_splits: dict[str, str], seed: int) -> dict[str, str]:
    """Keep canonical-val scenes exclusively held out; split canonical train scenes."""
    train_scenes = sorted(scene for scene, split in original_splits.items() if split == "train")
    test_scenes = sorted(scene for scene, split in original_splits.items() if split == "val")
    if not train_scenes or not test_scenes:
        raise RuntimeError("expected non-empty canonical train and val scene sets")
    rng = np.random.default_rng(seed)
    train_scenes = [str(value) for value in rng.permutation(np.asarray(train_scenes, dtype=object))]
    n_train = int(math.floor(0.8 * len(train_scenes)))
    mapping = {scene: "train" for scene in train_scenes[:n_train]}
    mapping.update({scene: "val" for scene in train_scenes[n_train:]})
    mapping.update({scene: "test" for scene in test_scenes})
    groups = {split: {scene for scene, value in mapping.items() if value == split} for split in ("train", "val", "test")}
    if any(not group for group in groups.values()) or any(groups[left] & groups[right] for left in groups for right in groups if left != right):
        raise RuntimeError("scene-disjoint split construction failed")
    return mapping


def nearest_observed(scene: SceneRecord, requested_e: np.ndarray, score_rotation_m: float = 0.03) -> dict[str, Any]:
    """Return an explicitly approximate RGB observation and its residual."""
    poses = scene.extrinsics
    # Vectorized ranking matters: each branch asks for two nearest views over a
    # full trajectory.  Compute only the winning log-map residual precisely.
    deltas = poses @ np.linalg.inv(requested_e)
    translation = np.linalg.norm(deltas[:, :3, 3], axis=1)
    cosine = np.clip((np.trace(deltas[:, :3, :3], axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
    rotation_rad = np.arccos(cosine)
    scores = translation + score_rotation_m * rotation_rad
    index = int(np.argmin(scores))
    image = scene.root / "images" / scene.valid_frames[index]
    depth = scene.root / "depth" / (Path(scene.valid_frames[index]).stem + ".png")
    return {
        "kind": "approx_nearest_observed_rgb",
        "raw_frame_index": index,
        "image_path": str(image),
        "image_sha256": sha256(image),
        "depth_path": str(depth),
        "depth_sha256": sha256(depth) if depth.is_file() else None,
        "pose_residual": pose_residual(requested_e, poses[index]),
        "observed_depth_coverage": depth_coverage(depth),
    }


def exact_observed(scene: SceneRecord, index: int, requested_e: np.ndarray) -> dict[str, Any]:
    image = scene.root / "images" / scene.valid_frames[index]
    depth = scene.root / "depth" / (Path(scene.valid_frames[index]).stem + ".png")
    return {
        "kind": "observed_rgb_not_rendered",
        "raw_frame_index": int(index),
        "image_path": str(image),
        "image_sha256": sha256(image),
        "depth_path": str(depth),
        "depth_sha256": sha256(depth) if depth.is_file() else None,
        "pose_residual": pose_residual(requested_e, scene.extrinsics[index]),
        "observed_depth_coverage": depth_coverage(depth),
    }


def depth_coverage(path: Path) -> float | None:
    if not path.is_file():
        return None
    with Image.open(path) as image:
        depth = np.asarray(image)
    return float(np.mean(depth > 0))


def make_branch(scene: SceneRecord, split: str, a: int, b: int, c: int, quantiles: dict[str, float], family: str | None = None) -> dict[str, Any]:
    extrinsics = scene.extrinsics
    t1 = relative_pose(extrinsics[a], extrinsics[b])
    t2 = relative_pose(extrinsics[b], extrinsics[c])
    if family == "negative_identity_nonidentity":
        t1 = np.eye(4)
        # For an identity control, choose the non-identity transform from A so
        # every endpoint is the same observed pose C.  Reusing B->C would make
        # even the negative-control endpoint synthetic, which is not intended.
        t2 = relative_pose(extrinsics[a], extrinsics[c])
    poses = branch_poses(extrinsics[a], t1, t2)
    commutator = group_commutator(t1, t2)
    inferred = family or branch_family(t1, t2, quantiles) or "unstratified"
    identity_control = family == "negative_identity_nonidentity"
    endpoint_records = {
        "source": exact_observed(scene, a, poses["E_A"]),
        "T1": exact_observed(scene, b if not identity_control else a, poses["E_T1"]),
        "T2": exact_observed(scene, c, poses["E_T2"]) if identity_control else nearest_observed(scene, poses["E_T2"]),
        "12": exact_observed(scene, c, poses["E_12"]),
        "21": exact_observed(scene, c, poses["E_21"]) if identity_control else nearest_observed(scene, poses["E_21"]),
    }
    max_t = max(value["pose_residual"]["translation_m"] for value in endpoint_records.values())
    max_r = max(value["pose_residual"]["rotation_deg"] for value in endpoint_records.values())
    branch_key = f"{scene.scene_id}:{a}:{b}:{c}:{inferred}"
    return {
        "branch_id": hashlib.sha256(branch_key.encode()).hexdigest()[:20],
        "scene_id": scene.scene_id,
        "split": split,
        "grounding_mode": "approx_grounded_nearest_observed_v1",
        "render_validity": {
            "track": "optional_dual_image_grounding",
            "status": "N/A_not_a_main_experiment_target",
            "raw_rgb_dynamic_object_risk": True,
            "max_observed_endpoint_translation_residual_m": max_t,
            "max_observed_endpoint_rotation_residual_deg": max_r,
        },
        "E_A": poses["E_A"],
        "T1": t1,
        "T2": t2,
        "E_T1": poses["E_T1"],
        "E_T2": poses["E_T2"],
        "E_12": poses["E_12"],
        "E_21": poses["E_21"],
        "endpoint_observations": endpoint_records,
        "intrinsics_source": scene.intrinsics[a],
        "intrinsics_provenance": str(scene.root / "scene_iphone_metadata.npz"),
        "motion": {
            "family": inferred,
            "T1_rotation_deg": rotation_degrees(t1),
            "T1_translation_m": translation_norm(t1),
            "T2_rotation_deg": rotation_degrees(t2),
            "T2_translation_m": translation_norm(t2),
            "log_T1": se3_log(t1),
            "log_T2": se3_log(t2),
            "log_T2T1": se3_log(t2 @ t1),
            "log_T1T2": se3_log(t1 @ t2),
            "commutator": commutator,
            "commutator_rotation_deg": rotation_degrees(commutator),
            "commutator_translation_m": translation_norm(commutator),
        },
        "pose_provenance": {
            "metadata_path": str(scene.root / "scene_iphone_metadata.npz"),
            "metadata_sha256": sha256(scene.root / "scene_iphone_metadata.npz"),
            "source_raw_frame_index": a,
            "t1_observed_raw_frame_index": b,
            "t2t1_observed_raw_frame_index": c,
            "pose_convention": POSE_CONVENTION,
        },
    }


def select_branches(test_scenes: list[SceneRecord], quantiles: dict[str, float], target: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    buckets: dict[str, list[tuple[SceneRecord, int, int, int]]] = defaultdict(list)
    for scene in test_scenes:
        # Stride sampling is geometry-only and independent of feature values.
        for horizon in (1, 2, 3):
            step = max(1, (len(scene.valid_frames) - 2 * horizon) // 80)
            for a in range(0, len(scene.valid_frames) - 2 * horizon, step):
                b, c = a + horizon, a + 2 * horizon
                t1 = relative_pose(scene.extrinsics[a], scene.extrinsics[b])
                t2 = relative_pose(scene.extrinsics[b], scene.extrinsics[c])
                family = branch_family(t1, t2, quantiles)
                if family:
                    buckets[family].append((scene, a, b, c))
                buckets["negative_identity_nonidentity"].append((scene, a, b, c))
    quotas = {
        "positive_rotation_translation": int(target * 0.35),
        "positive_different_axis_rotation_rotation": int(target * 0.20),
        "negative_translation_dominant_translation_dominant": int(target * 0.20),
        "negative_near_commuting": int(target * 0.15),
        "negative_identity_nonidentity": target - int(target * 0.90),
    }
    rows: list[dict[str, Any]] = []
    used: set[tuple[str, int, int, int, str]] = set()
    available = {name: len(values) for name, values in buckets.items()}

    def balanced_take(candidates: list[tuple[SceneRecord, int, int, int]], count: int) -> list[tuple[SceneRecord, int, int, int]]:
        """Take candidates round-robin by scene rather than file order."""
        by_scene: dict[str, list[tuple[SceneRecord, int, int, int]]] = defaultdict(list)
        for candidate in candidates:
            by_scene[candidate[0].scene_id].append(candidate)
        selected: list[tuple[SceneRecord, int, int, int]] = []
        while len(selected) < count:
            made_progress = False
            for scene_id in sorted(by_scene):
                if by_scene[scene_id] and len(selected) < count:
                    selected.append(by_scene[scene_id].pop(0))
                    made_progress = True
            if not made_progress:
                break
        return selected

    for family, quota in quotas.items():
        for scene, a, b, c in balanced_take(buckets.get(family, []), quota):
            key = (scene.scene_id, a, b, c, family)
            if key not in used:
                used.add(key)
                rows.append(make_branch(scene, "test", a, b, c, quantiles, family))
    # Do not silently pad a deficient requested family: remaining rows retain a
    # diagnostic-only label and coverage records the exact shortage.
    if len(rows) < target:
        for scene in test_scenes:
            for horizon in (1, 2, 3):
                step = max(1, (len(scene.valid_frames) - 2 * horizon) // 80)
                for a in range(0, len(scene.valid_frames) - 2 * horizon, step):
                    if len(rows) >= target:
                        break
                    b, c = a + horizon, a + 2 * horizon
                    key = (scene.scene_id, a, b, c, "unstratified")
                    if key not in used:
                        used.add(key)
                        rows.append(make_branch(scene, "test", a, b, c, quantiles, "unstratified"))
                if len(rows) >= target:
                    break
            if len(rows) >= target:
                break
    counts = Counter(row["motion"]["family"] for row in rows)
    return rows, {"requested": quotas, "available_candidates": available, "selected": dict(counts), "total": len(rows)}


def select_primitives(train_scenes: list[SceneRecord], target: int) -> list[dict[str, Any]]:
    """Balanced, geometry-only one-step primitive inventory.

    Round-robin selection avoids filling the requested 1k quota from the first
    few long trajectories in lexicographic scene order.
    """
    rows: list[dict[str, Any]] = []
    per_scene = [
        edge_motions(
            scene,
            stride=1,
            sample_step=max(1, (len(scene.valid_frames) - 1) // 8),
        )
        for scene in train_scenes
    ]
    while len(rows) < target and any(per_scene):
        for scene, motions in zip(train_scenes, per_scene):
            if not motions or len(rows) >= target:
                continue
            motion = motions.pop(0)
            source, endpoint = motion["source_index"], motion["target_index"]
            image = scene.root / "images" / scene.valid_frames[source]
            target_image = scene.root / "images" / scene.valid_frames[endpoint]
            rows.append(
                {
                    "primitive_id": hashlib.sha256(f"{scene.scene_id}:{source}:{endpoint}".encode()).hexdigest()[:20],
                    "scene_id": scene.scene_id,
                    "split": "train",
                    "source_raw_frame_index": source,
                    "target_raw_frame_index": endpoint,
                    "source_image_path": str(image),
                    "target_image_path": str(target_image),
                    "source_image_sha256": sha256(image),
                    "target_image_sha256": sha256(target_image),
                    "T_AB": motion["transform"],
                    "log_T_AB": se3_log(motion["transform"]),
                    "rotation_deg": motion["rotation_deg"],
                    "translation_m": motion["translation_m"],
                }
            )
    return rows
