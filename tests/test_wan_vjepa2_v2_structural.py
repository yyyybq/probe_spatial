from __future__ import annotations

import numpy as np
import torch

from analyses.se3_latent_mvp.geometry import encode_relative_pose_9d, so3_exp
from analyses.se3_protocol_v2.wan_vjepa2_v2_train import (
    AdditiveUpdater,
    evaluate_baseline,
)


class CountingAdditive(AdditiveUpdater):
    def __init__(self) -> None:
        super().__init__(channels=4, hidden_dim=3)
        self.calls = 0

    def update(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return super().update(state, action)


def test_additive_sequential_and_swapped_paths_are_executed_independently():
    model = CountingAdditive().eval()
    batch = 5
    z_a = torch.randn(batch, 4)
    zeros = torch.zeros(batch, 4)
    action1, action2, composite = (torch.randn(batch, 9) for _ in range(3))
    arrays, _ = evaluate_baseline(
        "commutative_additive", {"commutative_additive": model},
        zeros, z_a, zeros, zeros, action1, action2, composite,
        torch.zeros(4), torch.ones(4),
    )
    # one-step=1, sequential=2, swapped=2, direct=1 independent updates.
    assert model.calls == 6
    assert np.max(np.abs(arrays["raw_order_margin"])) < 1e-6


def test_direct_action_encodes_matrix_product_not_sum_of_action_vectors():
    t1 = np.eye(4)
    t2 = np.eye(4)
    t1[:3, :3] = so3_exp(np.asarray([0.0, 0.0, 0.3]))
    t1[:3, 3] = [0.2, 0.0, 0.0]
    t2[:3, :3] = so3_exp(np.asarray([0.0, 0.4, 0.0]))
    t2[:3, 3] = [0.0, 0.1, 0.0]
    direct = encode_relative_pose_9d(t2 @ t1)
    illegal_sum = encode_relative_pose_9d(t1) + encode_relative_pose_9d(t2)
    assert not np.allclose(direct, illegal_sum)
    assert np.allclose(direct, encode_relative_pose_9d(np.matmul(t2, t1)))
