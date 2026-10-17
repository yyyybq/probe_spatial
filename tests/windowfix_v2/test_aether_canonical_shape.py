from __future__ import annotations

import torch

from features.aether.extract_features import forward_aether


class _FakeAether:
    def __init__(self) -> None:
        self.calls = []

    def forward(self, *, video, t, output_layer_indices, noise_generator):
        self.calls.append(
            {
                "video": list(video),
                "t": t,
                "layers": list(output_layer_indices),
                "generator": noise_generator,
            }
        )
        marker = float(len(self.calls))
        return {
            layer: torch.full((1, 11, 30, 45, 3), marker)
            for layer in output_layer_indices
        }


def test_forward_aether_preserves_two_clip_axis_and_generator():
    model = _FakeAether()
    frames = list(range(81))
    generator = object()

    result = forward_aether(
        model,
        frames,
        t=749,
        layer_ids=[1, 7],
        noise_generator=generator,
    )

    assert len(model.calls) == 2
    assert all(call["generator"] is generator for call in model.calls)
    assert all(len(call["video"]) == 41 for call in model.calls)
    assert model.calls[0]["video"] == [0] + list(range(1, 80, 2))
    assert model.calls[1]["video"] == [0] + list(range(2, 81, 2))
    assert result[1].shape == (2, 11, 30, 45, 3)
    assert torch.all(result[1][0] == 1)
    assert torch.all(result[1][1] == 2)


def test_forward_aether_requires_81_input_frames():
    model = _FakeAether()
    try:
        forward_aether(model, list(range(80)), t=749, layer_ids=[1])
    except ValueError as exc:
        assert "exactly 81" in str(exc)
    else:
        raise AssertionError("80-frame input unexpectedly passed Aether contract")
