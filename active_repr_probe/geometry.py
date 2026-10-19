"""Auditable, dependency-light labels for Active Spatial states.

These helpers intentionally cover only labels that can be derived from an
explicit camera pose and target geometry.  Visibility, occlusion, task
relations, and counterfactual transitions require environment evidence and
are represented as unavailable rather than approximated here.
"""

from __future__ import annotations

import math
from typing import Any, Iterable


LABEL_VERSION = "active_repr_geometry_v1"
UNAVAILABLE = "UNAVAILABLE"


def _as_vector(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        raise ValueError(f"{name} must contain at least three values")
    return float(value[0]), float(value[1]), float(value[2])


def target_center(item: dict[str, Any]) -> tuple[float, float, float] | None:
    """Return the primary target-object center without inventing a fallback."""
    target = item.get("target_object")
    if isinstance(target, dict):
        primary = target.get("primary")
        if isinstance(primary, dict) and primary.get("center") is not None:
            return _as_vector(primary["center"], "target_object.primary.center")
        objects = target.get("objects")
        if isinstance(objects, list) and objects and isinstance(objects[0], dict):
            if objects[0].get("center") is not None:
                return _as_vector(objects[0]["center"], "target_object.objects[0].center")
    region = item.get("target_region")
    if isinstance(region, dict):
        params = region.get("params")
        if isinstance(params, dict) and params.get("object_center") is not None:
            return _as_vector(params["object_center"], "target_region.params.object_center")
    return None


def pose_position_and_axes(c2w: Any) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """Return position, right, up, forward from a camera-to-world 4x4 pose.

    Active Spatial's ViewManipulator uses local +Z as forward.  This explicit
    convention is tested below and must not be replaced by a w2c convention.
    """
    if not isinstance(c2w, (list, tuple)) or len(c2w) < 3:
        raise ValueError("c2w pose must be a 4x4-like nested sequence")
    try:
        right = (float(c2w[0][0]), float(c2w[1][0]), float(c2w[2][0]))
        up = (float(c2w[0][1]), float(c2w[1][1]), float(c2w[2][1]))
        forward = (float(c2w[0][2]), float(c2w[1][2]), float(c2w[2][2]))
        position = (float(c2w[0][3]), float(c2w[1][3]), float(c2w[2][3]))
    except (IndexError, TypeError) as exc:
        raise ValueError("c2w pose must expose rows 0..2 and columns 0..3") from exc
    return position, right, up, forward


def _dot(a: Iterable[float], b: Iterable[float]) -> float:
    left, right = tuple(a), tuple(b)
    if len(left) != len(right):
        raise ValueError("dot product arguments must have the same length")
    return sum(x * y for x, y in zip(left, right))


def relative_camera_vector(c2w: Any, point_world: Any) -> tuple[float, float, float]:
    """Target vector in the camera coordinate system (right, up, forward)."""
    position, right, up, forward = pose_position_and_axes(c2w)
    target = _as_vector(point_world, "point_world")
    delta = tuple(target[i] - position[i] for i in range(3))
    return _dot(delta, right), _dot(delta, up), _dot(delta, forward)


def distance_m(c2w: Any, point_world: Any) -> float:
    x, y, z = relative_camera_vector(c2w, point_world)
    return math.sqrt(x * x + y * y + z * z)


def direction_bin(c2w: Any, point_world: Any, *, bins: int = 8, front_boundary_deg: float = 22.5) -> dict[str, Any]:
    """Azimuth bin centered on camera forward, plus a three-way elevation."""
    if bins <= 0 or 360 % bins:
        raise ValueError("bins must be a positive divisor of 360")
    x, y, z = relative_camera_vector(c2w, point_world)
    horizontal = math.hypot(x, z)
    angle = math.degrees(math.atan2(x, z))  # +x is right, +z is forward
    width = 360.0 / bins
    azimuth = int(math.floor(((angle + width / 2.0) % 360.0) / width))
    elevation = "level"
    if horizontal > 1e-9:
        elev = math.degrees(math.atan2(y, horizontal))
        if elev > front_boundary_deg:
            elevation = "up"
        elif elev < -front_boundary_deg:
            elevation = "down"
    return {"azimuth_deg": angle, "azimuth_bin": azimuth, "elevation": elevation}


def distance_bin(value_m: float, edges_m: tuple[float, ...] = (1.0, 2.0, 4.0, 6.0)) -> int:
    if value_m < 0:
        raise ValueError("distance must be non-negative")
    return sum(value_m >= edge for edge in edges_m)


def rotation_delta_deg(previous_c2w: Any, current_c2w: Any) -> float:
    """Geodesic rotation magnitude for two c2w poses, in degrees."""
    _, pr, pu, pf = pose_position_and_axes(previous_c2w)
    _, cr, cu, cf = pose_position_and_axes(current_c2w)
    # trace(R_prev^T R_current), with axes represented as columns.
    trace = _dot(pr, cr) + _dot(pu, cu) + _dot(pf, cf)
    cosine = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    return math.degrees(math.acos(cosine))


def geometric_labels(item: dict[str, Any], c2w: Any) -> dict[str, Any]:
    center = target_center(item)
    if center is None:
        return {
            "label_version": LABEL_VERSION,
            "target_center": UNAVAILABLE,
            "relative_direction": UNAVAILABLE,
            "distance_m": UNAVAILABLE,
            "distance_bin": UNAVAILABLE,
        }
    dist = distance_m(c2w, center)
    return {
        "label_version": LABEL_VERSION,
        "target_center": list(center),
        "relative_direction": direction_bin(c2w, center),
        "distance_m": dist,
        "distance_bin": distance_bin(dist),
    }
