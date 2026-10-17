"""Relative-camera-movement head for the A1 probing protocol.

The head mirrors the iterative VGGT camera head, but operates on adjacent
camera-token differences and predicts one motion for every sequence edge.
Its 9-D output is ``[first two rotation rows (6), translation (3)]`` and maps
coordinates from camera ``i`` into camera ``i + 1``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from vidfm3d.vggt.layers import Mlp
from vidfm3d.vggt.layers.block import Block


class RelativeMovementHead(nn.Module):
    """Predict ``S - 1`` adjacent relative movements from ``S`` frames."""

    target_dim = 9

    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
    ) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            *[
                Block(
                    dim=dim_in,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    init_values=init_values,
                )
                for _ in range(trunk_depth)
            ]
        )
        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)
        self.empty_movement_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.embed_movement = nn.Linear(self.target_dim, dim_in)
        self.movement_ln_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True)
        )
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)
        self.movement_branch = Mlp(
            in_features=dim_in,
            hidden_features=dim_in // 2,
            out_features=self.target_dim,
            drop=0,
        )

    def forward(
        self, aggregated_tokens_list: list[torch.Tensor], num_iterations: int = 4
    ) -> list[torch.Tensor]:
        tokens = aggregated_tokens_list[-1]
        if tokens.shape[1] < 2:
            raise ValueError(
                "RelativeMovementHead requires at least two frames, "
                f"but received S={tokens.shape[1]}"
            )

        camera_tokens = self.token_norm(tokens[:, :, 0])
        # The subtraction has no trainable parameters, keeps the head width equal
        # to CameraHead, and gives one explicitly ordered token per adjacent edge.
        edge_tokens = (camera_tokens[:, 1:] - camera_tokens[:, :-1]) / math.sqrt(2.0)
        return self._iterative_refinement(edge_tokens, num_iterations)

    def _iterative_refinement(
        self, edge_tokens: torch.Tensor, num_iterations: int
    ) -> list[torch.Tensor]:
        batch, edges, _ = edge_tokens.shape
        prediction = None
        predictions = []

        for _ in range(num_iterations):
            if prediction is None:
                module_input = self.embed_movement(
                    self.empty_movement_tokens.expand(batch, edges, -1)
                )
            else:
                prediction = prediction.detach()
                module_input = self.embed_movement(prediction)

            shift, scale, gate = self.movement_ln_modulation(module_input).chunk(
                3, dim=-1
            )
            modulated = gate * (self.adaln_norm(edge_tokens) * (1 + scale) + shift)
            refined = self.trunk(modulated + edge_tokens)
            delta = self.movement_branch(self.trunk_norm(refined))
            prediction = delta if prediction is None else prediction + delta
            # Rotation is projected onto SO(3) only for geometric metrics. Raw
            # 6-D rows remain the regression target, as in the C-series action
            # encoding already used by this repository.
            predictions.append(prediction)

        return predictions
