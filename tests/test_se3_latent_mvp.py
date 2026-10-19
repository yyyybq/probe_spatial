from __future__ import annotations

import numpy as np

from analyses.se3_latent_mvp.geometry import (
    relative_pose,
    se3_exp,
    se3_log,
    transform_points,
)


def test_pose_composition_order() -> None:
    t_ab = se3_exp(np.asarray([0.12, -0.08, 0.03, 0.4, -0.2, 0.1]))
    t_bc = se3_exp(np.asarray([-0.04, 0.16, 0.07, -0.1, 0.3, 0.2]))
    e_a = np.eye(4)
    e_b = t_ab @ e_a
    e_c = t_bc @ e_b
    recovered_ab = relative_pose(e_a, e_b)
    recovered_bc = relative_pose(e_b, e_c)
    recovered_ac = relative_pose(e_a, e_c)
    np.testing.assert_allclose(recovered_bc @ recovered_ab, recovered_ac, atol=1e-10)


def test_se3_log_exp_roundtrip() -> None:
    twist = np.asarray([0.18, -0.09, 0.11, 0.35, -0.22, 0.14])
    np.testing.assert_allclose(se3_exp(se3_log(se3_exp(twist))), se3_exp(twist), atol=1e-9)


def test_synthetic_equivariant_predictor_sanity() -> None:
    # A noiseless latent made from 3D points has exact composition/cycle and
    # exposes non-commutativity for a rotation-translation pair.
    points = np.asarray([[0.2, -0.1, 1.0], [1.1, 0.4, -0.3], [-0.7, 0.8, 0.5]])
    t1 = se3_exp(np.asarray([0.0, 0.0, 0.35, 0.5, 0.0, 0.0]))
    t2 = se3_exp(np.asarray([0.0, -0.25, 0.0, 0.0, 0.3, 0.0]))
    sequential = transform_points(t2, transform_points(t1, points))
    direct = transform_points(t2 @ t1, points)
    cycle = transform_points(np.linalg.inv(t1), transform_points(t1, points))
    swapped = transform_points(t1, transform_points(t2, points))
    np.testing.assert_allclose(sequential, direct, atol=1e-10)
    np.testing.assert_allclose(cycle, points, atol=1e-10)
    assert np.linalg.norm(sequential - swapped) > 1e-2

