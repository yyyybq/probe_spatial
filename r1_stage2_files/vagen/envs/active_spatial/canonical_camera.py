"""Canonical camera utilities for Active Spatial R1 audits and generation.

This module is intentionally side-effect free. It does not change reward,
termination, sampling, or training behavior; callers must opt in through an
explicit camera_model_version.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np


CANONICAL_CAMERA_H1_RESIZE_V1 = "canonical_camera_h1_resize_v1"
HISTORICAL_V46_UNSCALED_ENV256 = "historical_v46_unscaled_env256"


@dataclass(frozen=True)
class ResolutionSource:
    native_resolution_source: str
    render_resolution_source: str


@dataclass(frozen=True)
class CanonicalCamera:
    K_native: list[list[float]]
    K_effective: list[list[float]]
    native_resolution: tuple[int, int]
    render_resolution: tuple[int, int]
    c2w: list[list[float]]
    w2c: list[list[float]]
    camera_model_version: str
    transform: str
    convention: str
    resolution_source: ResolutionSource

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["resolution_source"] = asdict(self.resolution_source)
        return data


def _as_k3(k: Any) -> np.ndarray:
    arr = np.asarray(k, dtype=np.float64)
    if arr.shape == (4, 4):
        arr = arr[:3, :3]
    if arr.shape != (3, 3):
        raise ValueError(f"camera intrinsics must be 3x3 or 4x4, got {arr.shape}")
    return arr


def _as_size(size: Sequence[int | float], name: str) -> tuple[int, int]:
    if len(size) != 2:
        raise ValueError(f"{name} must be (width, height), got {size!r}")
    width, height = int(round(float(size[0]))), int(round(float(size[1])))
    if width <= 0 or height <= 0:
        raise ValueError(f"{name} must be positive, got {size!r}")
    return width, height


def scale_intrinsics_for_resize(
    K_native: Any,
    native_size: Sequence[int | float],
    render_size: Sequence[int | float],
) -> np.ndarray:
    """Scale native intrinsics for direct rendering after native->render resize.

    Supports non-uniform sx/sy. This is the H1 model validated by the real
    renderer equivalence test.
    """
    K = _as_k3(K_native).copy()
    native_width, native_height = _as_size(native_size, "native_size")
    render_width, render_height = _as_size(render_size, "render_size")
    sx = float(render_width) / float(native_width)
    sy = float(render_height) / float(native_height)
    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy
    return K


def normalize_vector(vec: Any, fallback: Sequence[float] = (1.0, 0.0, 0.0)) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64)[:3]
    norm = float(np.linalg.norm(arr))
    if norm < 1e-10:
        return np.asarray(fallback, dtype=np.float64)
    return arr / norm


def camera_pose_from_forward(camera_position: Any, camera_forward: Any) -> np.ndarray:
    """Build c2w with X-right, Y-down, Z-forward camera coordinates."""
    forward = normalize_vector(camera_forward)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    if float(np.linalg.norm(right)) < 1e-8:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    right = normalize_vector(right)
    down = normalize_vector(np.cross(forward, right), fallback=(0.0, 0.0, -1.0))
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = np.asarray(camera_position, dtype=np.float64)[:3]
    return c2w


def infer_native_resolution(
    K_native: Any,
    item: Optional[Mapping[str, Any]] = None,
    fallback: Optional[Sequence[int | float]] = None,
) -> tuple[tuple[int, int], str]:
    """Infer native resolution with an auditable source label.

    Priority:
    1. Explicit item metadata fields.
    2. Principal point center assumption: width=2*cx, height=2*cy.
    3. Caller-provided fallback.
    """
    item = item or {}
    for key in ("native_resolution", "source_resolution", "camera_native_resolution"):
        value = item.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return _as_size(value, key), key

    for w_key, h_key in (
        ("native_width", "native_height"),
        ("source_width", "source_height"),
        ("camera_native_width", "camera_native_height"),
    ):
        if w_key in item and h_key in item:
            return _as_size((item[w_key], item[h_key]), f"{w_key}/{h_key}"), f"{w_key}/{h_key}"

    K = _as_k3(K_native)
    cx, cy = float(K[0, 2]), float(K[1, 2])
    if cx > 0 and cy > 0 and np.isfinite(cx) and np.isfinite(cy):
        width, height = int(round(2.0 * cx)), int(round(2.0 * cy))
        if width > 0 and height > 0:
            return (width, height), "intrinsics_principal_point_center_2x"

    if fallback is not None:
        return _as_size(fallback, "fallback_native_size"), "caller_fallback"
    raise ValueError("native resolution is missing and cannot be inferred from intrinsics")


def build_canonical_camera(
    *,
    K_native: Any,
    native_size: Optional[Sequence[int | float]] = None,
    render_size: Sequence[int | float],
    transform: str,
    c2w: Optional[Any] = None,
    camera_position: Optional[Any] = None,
    camera_forward: Optional[Any] = None,
    item: Optional[Mapping[str, Any]] = None,
    camera_model_version: str = CANONICAL_CAMERA_H1_RESIZE_V1,
) -> CanonicalCamera:
    if camera_model_version != CANONICAL_CAMERA_H1_RESIZE_V1:
        raise ValueError(f"unsupported canonical camera model: {camera_model_version}")
    if transform != "resize":
        raise ValueError(f"{CANONICAL_CAMERA_H1_RESIZE_V1} only supports transform='resize'")

    K3 = _as_k3(K_native)
    if native_size is None:
        native_size, native_source = infer_native_resolution(K3, item=item)
    else:
        native_size = _as_size(native_size, "native_size")
        native_source = "explicit_argument"
    render_size = _as_size(render_size, "render_size")
    K_eff = scale_intrinsics_for_resize(K3, native_size, render_size)

    if c2w is None:
        if camera_position is None or camera_forward is None:
            raise ValueError("provide c2w or both camera_position and camera_forward")
        c2w_arr = camera_pose_from_forward(camera_position, camera_forward)
    else:
        c2w_arr = np.asarray(c2w, dtype=np.float64)
        if c2w_arr.shape != (4, 4):
            raise ValueError(f"c2w must be 4x4, got {c2w_arr.shape}")
    w2c_arr = np.linalg.inv(c2w_arr)

    return CanonicalCamera(
        K_native=K3.tolist(),
        K_effective=K_eff.tolist(),
        native_resolution=native_size,
        render_resolution=render_size,
        c2w=c2w_arr.tolist(),
        w2c=w2c_arr.tolist(),
        camera_model_version=camera_model_version,
        transform=transform,
        convention="opencv_colmap_x_right_y_down_z_forward_c2w",
        resolution_source=ResolutionSource(
            native_resolution_source=native_source,
            render_resolution_source="explicit_argument",
        ),
    )


def build_historical_v46_camera(
    *,
    K_native: Any,
    render_size: Sequence[int | float],
    c2w: Any,
) -> CanonicalCamera:
    K3 = _as_k3(K_native)
    render_size = _as_size(render_size, "render_size")
    c2w_arr = np.asarray(c2w, dtype=np.float64)
    return CanonicalCamera(
        K_native=K3.tolist(),
        K_effective=K3.tolist(),
        native_resolution=render_size,
        render_resolution=render_size,
        c2w=c2w_arr.tolist(),
        w2c=np.linalg.inv(c2w_arr).tolist(),
        camera_model_version=HISTORICAL_V46_UNSCALED_ENV256,
        transform="none",
        convention="opencv_colmap_x_right_y_down_z_forward_c2w",
        resolution_source=ResolutionSource(
            native_resolution_source="historical_env_render_size",
            render_resolution_source="explicit_argument",
        ),
    )


def camera_params_for_visual_metrics(
    item: Mapping[str, Any],
    camera: CanonicalCamera,
) -> dict[str, Any]:
    params = dict(item.get("task_params") or {})
    params.update(
        {
            "_target_object": item.get("target_object"),
            "_camera_intrinsics": camera.K_effective,
            "_image_width": int(camera.render_resolution[0]),
            "_image_height": int(camera.render_resolution[1]),
            "_camera_pose_c2w": camera.c2w,
            "_camera_model_version": camera.camera_model_version,
            "_camera_convention": camera.convention,
            "_native_resolution": list(camera.native_resolution),
            "_render_resolution": list(camera.render_resolution),
        }
    )
    return params
