"""SE(3) helpers and protocol-v2 algebraic invariants."""

from __future__ import annotations

import numpy as np

from analyses.se3_latent_mvp.geometry import homogeneous, se3_exp, se3_log


def branch_poses(e_a: np.ndarray, t1: np.ndarray, t2: np.ndarray) -> dict[str, np.ndarray]:
    """Return the five w2c poses required by an ordered branch.

    The repository convention is column-vector, world-to-camera extrinsics.
    Thus a camera-coordinate change left-multiplies an extrinsic: applying
    ``T1`` then ``T2`` yields ``T2 @ T1 @ E_A``.
    """
    e_a = homogeneous(e_a)
    t1 = homogeneous(t1)
    t2 = homogeneous(t2)
    return {
        "E_A": e_a,
        "E_T1": t1 @ e_a,
        "E_T2": t2 @ e_a,
        "E_12": t2 @ t1 @ e_a,
        "E_21": t1 @ t2 @ e_a,
    }


def se3_bracket(xi_left: np.ndarray, xi_right: np.ndarray) -> np.ndarray:
    """Lie bracket for twists ordered ``[omega, v]``."""
    xi_left = np.asarray(xi_left, dtype=np.float64)
    xi_right = np.asarray(xi_right, dtype=np.float64)
    w1, v1 = xi_left[:3], xi_left[3:]
    w2, v2 = xi_right[:3], xi_right[3:]
    return np.concatenate((np.cross(w1, w2), np.cross(w1, v2) - np.cross(w2, v1)))


def bch_second_log(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """Second-order BCH approximation to ``log(T2 @ T1)``."""
    xi1, xi2 = se3_log(t1), se3_log(t2)
    return xi2 + xi1 + 0.5 * se3_bracket(xi2, xi1)


def group_commutator(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """Identity iff ``t1`` and ``t2`` commute, in the protocol order."""
    t1 = homogeneous(t1)
    t2 = homogeneous(t2)
    return (t2 @ t1) @ np.linalg.inv(t1 @ t2)


def rotation_degrees(transform: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(se3_log(transform)[:3])))


def translation_norm(transform: np.ndarray) -> float:
    return float(np.linalg.norm(homogeneous(transform)[:3, 3]))


def pose_residual(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Pose residual from a requested w2c endpoint to an observed w2c pose."""
    delta = homogeneous(candidate) @ np.linalg.inv(homogeneous(reference))
    return {
        "translation_m": translation_norm(delta),
        "rotation_deg": rotation_degrees(delta),
    }


def is_commuting(t1: np.ndarray, t2: np.ndarray, atol: float = 1e-10) -> bool:
    return bool(np.max(np.abs(homogeneous(t2) @ homogeneous(t1) - homogeneous(t1) @ homogeneous(t2))) <= atol)

