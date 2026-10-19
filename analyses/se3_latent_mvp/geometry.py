"""Small, dependency-light SE(3) helpers used by the experimental MVP."""

from __future__ import annotations

import math

import numpy as np


def homogeneous(extrinsic: np.ndarray) -> np.ndarray:
    """Promote a 3x4 or validate a 4x4 homogeneous transform."""
    value = np.asarray(extrinsic, dtype=np.float64)
    if value.shape == (4, 4):
        return value.copy()
    if value.shape != (3, 4):
        raise ValueError(f"expected (3,4) or (4,4), got {value.shape}")
    result = np.eye(4, dtype=np.float64)
    result[:3] = value
    return result


def camera_to_world_to_extrinsic(camera_to_world: np.ndarray) -> np.ndarray:
    """Convert a camera-to-world matrix to a world-to-camera 4x4 matrix."""
    return np.linalg.inv(homogeneous(camera_to_world))


def relative_pose(extr_from: np.ndarray, extr_to: np.ndarray) -> np.ndarray:
    """Return the transform mapping cam_from coordinates into cam_to."""
    return homogeneous(extr_to) @ np.linalg.inv(homogeneous(extr_from))


def encode_relative_pose_9d(transform: np.ndarray) -> np.ndarray:
    """Match the repository's first-two-rows rotation + translation encoding."""
    transform = homogeneous(transform)
    return np.concatenate((transform[:2, :3].reshape(-1), transform[:3, 3]))


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def so3_exp(omega: np.ndarray) -> np.ndarray:
    omega = np.asarray(omega, dtype=np.float64)
    theta = float(np.linalg.norm(omega))
    matrix = skew(omega)
    if theta < 1e-8:
        return np.eye(3) + matrix + 0.5 * matrix @ matrix
    a = math.sin(theta) / theta
    b = (1.0 - math.cos(theta)) / (theta * theta)
    return np.eye(3) + a * matrix + b * matrix @ matrix


def so3_log(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    theta = math.acos(cosine)
    vee = np.asarray(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    if theta < 1e-8:
        return 0.5 * vee
    if math.pi - theta < 1e-5:
        # Eigenvector extraction is stable near pi, where sin(theta) vanishes.
        values, vectors = np.linalg.eig(rotation)
        axis = np.real(vectors[:, np.argmin(np.abs(values - 1.0))])
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        dominant = int(np.argmax(np.abs(axis)))
        if axis[dominant] * vee[dominant] < 0:
            axis = -axis
        return theta * axis
    return theta * vee / (2.0 * math.sin(theta))


def se3_exp(twist: np.ndarray) -> np.ndarray:
    """Exponential map for twist order [omega_x,y,z, v_x,y,z]."""
    twist = np.asarray(twist, dtype=np.float64)
    if twist.shape != (6,):
        raise ValueError(f"expected a six-vector, got {twist.shape}")
    omega, velocity = twist[:3], twist[3:]
    theta = float(np.linalg.norm(omega))
    matrix = skew(omega)
    rotation = so3_exp(omega)
    if theta < 1e-8:
        jacobian = np.eye(3) + 0.5 * matrix + (matrix @ matrix) / 6.0
    else:
        jacobian = (
            np.eye(3)
            + (1.0 - math.cos(theta)) / (theta * theta) * matrix
            + (theta - math.sin(theta)) / (theta**3) * matrix @ matrix
        )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = jacobian @ velocity
    return transform


def se3_log(transform: np.ndarray) -> np.ndarray:
    """Logarithm map with twist order [omega_x,y,z, v_x,y,z]."""
    transform = homogeneous(transform)
    omega = so3_log(transform[:3, :3])
    theta = float(np.linalg.norm(omega))
    matrix = skew(omega)
    if theta < 1e-8:
        jacobian_inverse = np.eye(3) - 0.5 * matrix + (matrix @ matrix) / 12.0
    else:
        coefficient = 1.0 / (theta * theta) - (
            1.0 + math.cos(theta)
        ) / (2.0 * theta * math.sin(theta))
        jacobian_inverse = np.eye(3) - 0.5 * matrix + coefficient * matrix @ matrix
    velocity = jacobian_inverse @ transform[:3, 3]
    return np.concatenate((omega, velocity))


def rotation_angle(transform: np.ndarray) -> float:
    return float(np.linalg.norm(so3_log(homogeneous(transform)[:3, :3])))


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    transform = homogeneous(transform)
    points = np.asarray(points, dtype=np.float64)
    return points @ transform[:3, :3].T + transform[:3, 3]

