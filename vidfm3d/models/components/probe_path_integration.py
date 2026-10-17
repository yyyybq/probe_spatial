"""C2 - Path Integration probe head.

Given a causal context-segment VFM feature and a sequence of relative camera
actions, recurrently predict the pooled target VFM feature at each waypoint.
The evaluation layer maps those predicted features back to poses through
retrieval, which exposes final pose error, drift rate, and loop closure error
without training a pose regressor.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class PathIntegrationProbe(nn.Module):
    def __init__(
        self,
        in_channels: int = 1536,
        action_dim: int = 9,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        conditioning: str = "full",
    ) -> None:
        super().__init__()
        if conditioning not in {"full", "visual_only", "action_only"}:
            raise ValueError(f"Unknown conditioning={conditioning!r}")
        self.conditioning = conditioning
        self.constant_context = nn.Parameter(torch.zeros(1, in_channels))
        self.constant_action = nn.Parameter(torch.zeros(1, 1, action_dim))
        nn.init.normal_(self.constant_context, std=0.02)
        nn.init.normal_(self.constant_action, std=0.02)
        self.feat_proj = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cells = nn.ModuleList([
            nn.GRUCell(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_channels),
        )

    def initialize_state(self, input_feat: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Map a causal streaming/context feature into recurrent state space.

        Isolated target features are decoder supervision only. They must not be
        passed here merely because their last dimension matches ``in_channels``.
        """
        if input_feat.ndim != 5:
            raise ValueError(
                "input_feat must have shape (B,T,H,W,C), got "
                f"{tuple(input_feat.shape)}"
            )
        batch = input_feat.shape[0]
        pooled = input_feat.mean(dim=(1, 2, 3))
        if self.conditioning == "action_only":
            pooled = self.constant_context.expand(batch, -1)
        initial = self.feat_proj(pooled)
        return tuple(initial for _ in self.cells)

    def update_state(
        self,
        state: Sequence[torch.Tensor],
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Apply the reusable update ``U(state, action)`` exactly once."""
        if len(state) != len(self.cells):
            raise ValueError(
                f"state has {len(state)} levels, expected {len(self.cells)}"
            )
        if action.ndim != 2 or action.shape[-1] != self.constant_action.shape[-1]:
            raise ValueError(
                "action must have shape (B, action_dim), got "
                f"{tuple(action.shape)}"
            )
        if self.conditioning == "visual_only":
            action = self.constant_action[:, 0].expand(action.shape[0], -1)
        value = self.action_proj(action)
        next_state = []
        for cell, previous in zip(self.cells, state):
            value = cell(value, previous)
            next_state.append(value)
        return tuple(next_state)

    def decode_state(self, state: Sequence[torch.Tensor]) -> torch.Tensor:
        """Decode one recurrent state into pooled isolated-target space."""
        if len(state) != len(self.cells):
            raise ValueError(
                f"state has {len(state)} levels, expected {len(self.cells)}"
            )
        return self.out_proj(state[-1])

    def rollout_from_state(
        self,
        state: Sequence[torch.Tensor],
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Apply the same updater repeatedly, without feature reinitialization."""
        if actions.ndim != 3:
            raise ValueError(
                f"actions must have shape (B,K,A), got {tuple(actions.shape)}"
            )
        current = tuple(state)
        predictions = []
        for step in range(actions.shape[1]):
            current = self.update_state(current, actions[:, step])
            predictions.append(self.decode_state(current))
        if predictions:
            stacked = torch.stack(predictions, dim=1)
        else:
            stacked = torch.empty(
                actions.shape[0],
                0,
                self.out_proj[-1].out_features,
                device=actions.device,
                dtype=actions.dtype,
            )
        return stacked, current

    def forward(
        self,
        input_feat: torch.Tensor,   # (B, T0, H_f, W_f, C)
        actions: torch.Tensor,      # (B, K, 9), consecutive waypoint actions
    ) -> torch.Tensor:
        state = self.initialize_state(input_feat)
        predictions, _ = self.rollout_from_state(state, actions)
        return predictions                                # (B, K, C)
