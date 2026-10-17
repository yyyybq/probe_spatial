from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from vidfm3d.models.components.probe_action_dynamics import ActionDynamicsProbe
from vidfm3d.models.probe_ext_module import ProbeExtensionLitModule
from vidfm3d.utils.action_validity import (
    aggregate_retrieval,
    make_block_derangement,
    retrieval_per_sample,
    validate_derangement,
)


def test_same_state_different_action_changes_probe_output() -> None:
    torch.manual_seed(7)
    probe = ActionDynamicsProbe(
        in_channels=16,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        max_seq_len=8,
        dropout=0.0,
    ).eval()
    state = torch.randn(1, 3, 2, 2, 16)
    action_a = torch.zeros(1, 9)
    action_b = torch.ones(1, 9)
    with torch.no_grad():
        output_a = probe(state, action_a)
        output_b = probe(state, action_b)
    assert float((output_a - output_b).abs().max()) > 1e-6


class _RecordingProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.seen = None

    def forward(self, input_feat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        self.seen = (input_feat.detach().clone(), action.detach().clone())
        return input_feat.mean(dim=(1, 2, 3)) * self.weight + action[:, :1] * 0.0


def test_training_step_does_not_feed_identifiers_or_indices_to_probe() -> None:
    probe = _RecordingProbe()
    module = ProbeExtensionLitModule(
        probe=probe,
        probe_type="action_dynamics",
        optimizer=lambda params: torch.optim.SGD(params, lr=0.1),
        scheduler=lambda optimizer: None,
    )
    input_feat = torch.randn(3, 2, 1, 1, 4)
    action = torch.randn(3, 9)
    batch = {
        "input_feat": input_feat,
        "action": action,
        "target_feat": torch.randn(3, 1, 1, 4),
        "dyn_valid": torch.ones(3, dtype=torch.bool),
        "sample_id": ["secret-a", "secret-b", "secret-c"],
        "scene_path": ["scene-a", "scene-b", "scene-c"],
        "target_frame_idx": torch.tensor([101, 202, 303]),
        "streaming_window_id": torch.tensor([4, 5, 6]),
    }
    module._step_action_dynamics(batch, train=False)
    seen_state, seen_action = probe.seen
    torch.testing.assert_close(seen_state, input_feat)
    torch.testing.assert_close(seen_action, action)


def test_shuffled_action_is_exact_marginal_scene_independent_derangement() -> None:
    blocks = np.repeat(np.arange(4), 8)
    scenes = np.asarray([f"scene-{i % 8}" for i in range(32)])
    donors = make_block_derangement(blocks, scenes, seed=20260825)
    validate_derangement(donors, blocks, scenes)
    action = np.arange(32 * 9, dtype=np.float32).reshape(32, 9)
    shuffled = action[donors]
    assert not np.any(np.all(shuffled == action, axis=1))
    np.testing.assert_array_equal(np.sort(shuffled, axis=0), np.sort(action, axis=0))


def test_candidate_target_order_is_metric_invariant() -> None:
    generator = torch.Generator().manual_seed(11)
    targets = torch.randn(23, 13, generator=generator)
    predictions = targets + 0.2 * torch.randn(23, 13, generator=generator)
    original = aggregate_retrieval(retrieval_per_sample(predictions, targets))
    permutation = torch.randperm(23, generator=generator)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(23)
    reordered = aggregate_retrieval(
        retrieval_per_sample(predictions, targets[permutation], inverse)
    )
    for key in original:
        assert abs(original[key] - reordered[key]) < 1e-8
