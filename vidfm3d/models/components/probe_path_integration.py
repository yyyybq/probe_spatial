"""C2 - Path Integration probe head.

Given a causal context-segment VFM feature and a sequence of relative camera
actions, recurrently predict the pooled target VFM feature at each waypoint.
The evaluation layer maps those predicted features back to poses through
retrieval, which exposes final pose error, drift rate, and loop closure error
without training a pose regressor.
"""

from __future__ import annotations

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
    ) -> None:
        super().__init__()
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

    def forward(
        self,
        input_feat: torch.Tensor,   # (B, T0, H_f, W_f, C)
        actions: torch.Tensor,      # (B, K, 9), consecutive waypoint actions
    ) -> torch.Tensor:
        B, K = actions.shape[:2]
        pooled = input_feat.mean(dim=(1, 2, 3))          # (B, C)
        h0 = self.feat_proj(pooled)                      # (B, D)
        states = [h0 for _ in self.cells]

        preds = []
        for k in range(K):
            x = self.action_proj(actions[:, k])
            next_states = []
            for i, cell in enumerate(self.cells):
                h = cell(x, states[i])
                next_states.append(h)
                x = h
            states = next_states
            preds.append(self.out_proj(states[-1]))

        if not preds:
            return torch.empty(B, 0, pooled.shape[-1], device=pooled.device, dtype=pooled.dtype)
        return torch.stack(preds, dim=1)                  # (B, K, C)
