"""Temporal alignment helpers for Aether's two interleaved video clips."""

import torch


def aether_streaming_token_indices(
    positions: torch.Tensor,
    *,
    n_chunks: int,
    n_feature_frames: int,
    input_length: int,
) -> torch.Tensor:
    """Map padded extractor-input positions to flattened Aether token rows.

    Aether splits an 81-frame input into two 41-frame clips. Both clips contain
    frame 0; subsequent odd/even input frames go to clips 0/1 respectively.
    Each clip then has temporal stride four, yielding 11 feature frames.
    """
    if n_chunks != 2 or n_feature_frames != 11:
        raise ValueError(
            "Aether streaming features require shape (2,11,H,W,C); "
            f"got chunks={n_chunks}, feature_frames={n_feature_frames}."
        )
    expected_input_length = 1 + n_chunks * (n_feature_frames - 1) * 4
    if input_length != expected_input_length:
        raise ValueError(
            f"Aether streaming input_length must be {expected_input_length}, "
            f"got {input_length}."
        )
    rounded = positions.round().long()
    if not torch.equal(positions, rounded.to(dtype=positions.dtype)):
        raise ValueError("Aether streaming positions must be integral.")
    if rounded.numel() and (
        int(rounded.min().item()) < 0 or int(rounded.max().item()) >= input_length
    ):
        raise ValueError(
            f"Aether streaming position outside [0,{input_length - 1}]."
        )

    indices = torch.zeros_like(rounded)
    rest = rounded != 0
    if rest.any():
        offsets = rounded[rest] - 1
        clip_indices = offsets.remainder(n_chunks)
        positions_in_clip = offsets.div(n_chunks, rounding_mode="floor") + 1
        token_indices = (positions_in_clip - 1).div(4, rounding_mode="floor") + 1
        indices[rest] = clip_indices * n_feature_frames + token_indices
    return indices
