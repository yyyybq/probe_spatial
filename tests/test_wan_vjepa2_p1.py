from __future__ import annotations

import numpy as np
import torch

from analyses.se3_latent_mvp.geometry import encode_relative_pose_9d, so3_exp, transform_points
from analyses.se3_protocol_v2.wan_vjepa2_p1 import (
    POINTS,
    bin_value,
    camera_center,
    point_rmse,
    ratio_distribution,
)
from vidfm3d.models.components.probe_path_integration import PathIntegrationProbe


class CountingProbe(PathIntegrationProbe):
    def __init__(self) -> None:
        super().__init__(in_channels=6, action_dim=9, hidden_dim=8, num_layers=2, dropout=0.0)
        self.update_calls = 0

    def update_state(self, state, action):
        self.update_calls += 1
        return super().update_state(state, action)


def test_identity_padding_really_changes_update_call_count():
    model = CountingProbe().eval()
    context = torch.randn(3, 1, 1, 1, 6)
    action = torch.randn(3, 9)
    identity = torch.from_numpy(encode_relative_pose_9d(np.eye(4)).astype(np.float32)).expand(3, -1)
    state = model.initialize_state(context)
    direct = model.decode_state(model.update_state(state, action))
    pre = model.decode_state(model.update_state(model.update_state(state, identity), action))
    post_state = model.update_state(state, action)
    post = model.decode_state(model.update_state(post_state, identity))
    assert model.update_calls == 5
    assert direct.shape == pre.shape == post.shape == (3, 6)


def test_order_geometry_keeps_rotation_and_translation_units_separate():
    t1 = np.eye(4); t2 = np.eye(4)
    t1[:3, :3] = so3_exp(np.asarray([0.0, 0.0, 0.5])); t1[:3, 3] = [1.0, 0.0, 0.0]
    t2[:3, :3] = so3_exp(np.asarray([0.0, 0.4, 0.0])); t2[:3, 3] = [0.0, 1.0, 0.0]
    correct, swapped = t2 @ t1, t1 @ t2
    rotation_rad = np.linalg.norm(__import__("analyses.se3_latent_mvp.geometry", fromlist=["so3_log"]).so3_log(swapped[:3, :3] @ correct[:3, :3].T))
    center_m = np.linalg.norm(camera_center(swapped) - camera_center(correct))
    assert rotation_rad > 0
    assert center_m > 0
    assert not np.isclose(rotation_rad, center_m)


def test_analytic_oracle_composes_exactly_and_identity_is_neutral():
    t1 = np.eye(4); t2 = np.eye(4)
    t1[:3, 3] = [.2, -.1, .3]
    t2[:3, :3] = so3_exp(np.asarray([.1, .2, -.1]))
    sequential = transform_points(t2, transform_points(t1, POINTS))
    direct = transform_points(t2 @ t1, POINTS)
    padded = transform_points(np.eye(4), direct)
    assert point_rmse(sequential.reshape(1, -1), direct.reshape(1, -1))[0] < 1e-7
    assert point_rmse(padded.reshape(1, -1), direct.reshape(1, -1))[0] == 0


def test_frozen_bin_boundary_rule_is_deterministic():
    rule = {"q1": 1.0, "q2": 2.0}
    assert bin_value(1.0, rule) == "low"
    assert bin_value(1.5, rule) == "medium"
    assert bin_value(2.0, rule) == "medium"
    assert bin_value(3.0, rule) == "high"


def test_normalization_is_ratio_of_scene_macro_aggregates():
    numerator = np.asarray([[1.0, 9.0]])
    denominator = np.asarray([[1.0, 3.0]])
    draws = np.asarray([[0, 1]])
    value = ratio_distribution(numerator, denominator, draws, 0.0)[0]
    assert np.isclose(value, 10.0 / 4.0)
    assert not np.isclose(value, np.mean([1.0 / 1.0, 9.0 / 3.0]))
