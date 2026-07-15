"""Temporal frame sampling utilities for InsScene-15K streaming probes.

The temporal streaming probes should use real video-like sources only.  ScanNet++
provides iPhone trajectories sampled from a real video; Infinigen frames are
independent rendered views and should not be treated as time steps.
"""

from __future__ import annotations

import math
import re
from typing import Iterable

import numpy as np


def frame_number_from_name(name: str) -> int | None:
    """Return the primary frame number encoded in common InsScene filenames."""
    patterns = (
        r"frame_(\d+)\.jpg$",
        r"Image_(\d+)_",
        r"camview_(\d+)_",
    )
    for pattern in patterns:
        match = re.search(pattern, str(name))
        if match:
            return int(match.group(1))
    return None


def natural_frame_key(name: str):
    """Sort key that respects numeric frame ids when filenames are not zero-padded."""
    frame_id = frame_number_from_name(name)
    return (0, frame_id, str(name)) if frame_id is not None else (1, str(name))


def sort_frame_names(names: Iterable[str]) -> list[str]:
    return sorted(list(names), key=natural_frame_key)


def rotation_angle_rad(r0: np.ndarray, r1: np.ndarray) -> float:
    """Geodesic angle between two rotation matrices."""
    rel = np.asarray(r0, dtype=np.float64).T @ np.asarray(r1, dtype=np.float64)
    cos_theta = (float(np.trace(rel)) - 1.0) * 0.5
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return float(math.acos(cos_theta))


def cumulative_camera_motion(
    poses_c2w: np.ndarray,
    rotation_weight: float = 0.5,
) -> np.ndarray:
    """Cumulative camera motion for a camera-to-world trajectory.

    Translation is measured in the dataset's metric units.  Rotation contributes
    ``rotation_weight * angle_rad`` so pure turning still advances the observation
    sequence.
    """
    poses = np.asarray(poses_c2w, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Expected poses_c2w with shape (N,4,4), got {poses.shape}")
    if len(poses) == 0:
        return np.zeros(0, dtype=np.float64)
    if len(poses) == 1:
        return np.zeros(1, dtype=np.float64)

    centers = poses[:, :3, 3]
    trans = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    rots = np.asarray(
        [rotation_angle_rad(poses[i, :3, :3], poses[i + 1, :3, :3]) for i in range(len(poses) - 1)],
        dtype=np.float64,
    )
    step = trans + float(rotation_weight) * rots
    return np.concatenate([[0.0], np.cumsum(step)])


def motion_uniform_observation_indices(
    poses_c2w: np.ndarray,
    motion_step: float = 0.35,
    rotation_weight: float = 0.5,
) -> list[int]:
    """Select frame indices approximately uniformly in cumulative camera motion."""
    poses = np.asarray(poses_c2w, dtype=np.float64)
    n = len(poses)
    if n == 0:
        return []
    if n == 1:
        return [0]

    motion_step = float(motion_step)
    if motion_step <= 0:
        return list(range(n))

    cumulative = cumulative_camera_motion(poses, rotation_weight=rotation_weight)
    total = float(cumulative[-1])
    if total <= 1e-8:
        return list(range(n))

    targets = np.arange(0.0, total + motion_step * 0.5, motion_step, dtype=np.float64)
    if targets[-1] < total:
        targets = np.concatenate([targets, [total]])

    selected: list[int] = []
    for target in targets:
        j = int(np.searchsorted(cumulative, target, side="left"))
        if j >= n:
            j = n - 1
        if j > 0 and abs(cumulative[j - 1] - target) <= abs(cumulative[j] - target):
            j -= 1
        if not selected or j > selected[-1]:
            selected.append(j)

    if selected[-1] != n - 1:
        selected.append(n - 1)
    return selected


def temporal_windows_from_poses(
    poses_c2w: np.ndarray,
    observations_per_window: int,
    motion_step: float = 0.35,
    rotation_weight: float = 0.5,
    window_stride: int = 8,
    max_windows_per_scene: int = 4,
) -> list[dict]:
    """Build motion-normalized observation windows from a ScanNet++ trajectory."""
    observations_per_window = int(observations_per_window)
    if observations_per_window <= 0:
        return []

    obs = motion_uniform_observation_indices(
        poses_c2w,
        motion_step=motion_step,
        rotation_weight=rotation_weight,
    )
    if len(obs) < observations_per_window:
        return []

    stride = max(int(window_stride), 1)
    starts = list(range(0, len(obs) - observations_per_window + 1, stride))
    if not starts:
        starts = [0]
    last_start = len(obs) - observations_per_window
    if starts[-1] != last_start:
        starts.append(last_start)

    max_windows = int(max_windows_per_scene)
    if max_windows > 0 and len(starts) > max_windows:
        keep = np.linspace(0, len(starts) - 1, max_windows).round().astype(int).tolist()
        starts = [starts[i] for i in sorted(set(keep))]

    windows = []
    for window_id, start in enumerate(starts):
        indices = obs[start : start + observations_per_window]
        windows.append(
            {
                "window_id": int(window_id),
                "obs_start": int(start),
                "indices": [int(i) for i in indices],
                "sampling": "motion_uniform",
                "motion_step": float(motion_step),
                "rotation_weight": float(rotation_weight),
            }
        )
    return windows
