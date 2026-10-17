import pytest
import torch

from vidfm3d.utils import aether_temporal as AETHER_TEMPORAL


def test_aether_interleaved_clip_mapping_and_repeat_tail():
    positions = torch.tensor([0, 1, 2, 3, 4, 5, 6, 80], dtype=torch.float32)
    observed = AETHER_TEMPORAL.aether_streaming_token_indices(
        positions, n_chunks=2, n_feature_frames=11, input_length=81
    )
    assert torch.equal(observed, torch.tensor([0, 1, 12, 1, 12, 1, 12, 21]))


def test_aether_mapping_rejects_unexpected_layout_or_input_length():
    positions = torch.tensor([0, 1], dtype=torch.float32)
    with pytest.raises(ValueError, match="shape"):
        AETHER_TEMPORAL.aether_streaming_token_indices(
            positions, n_chunks=1, n_feature_frames=11, input_length=81
        )
    with pytest.raises(ValueError, match="input_length"):
        AETHER_TEMPORAL.aether_streaming_token_indices(
            positions, n_chunks=2, n_feature_frames=11, input_length=24
        )
