"""C3 - Counterfactual action probe head.

Given the same observed context and several alternative camera actions, predict
the pooled target VFM feature for each intervention independently. The head is
small by design: success should come from the frozen representation plus action
conditioning, not from a large learned dynamics model.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CounterfactualProbe(nn.Module):
    def __init__(
        self,
        in_channels: int = 1536,
        action_dim: int = 9,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        max_horizons: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.context_proj = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hidden_dim),
        )
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, max_horizons + 1, hidden_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
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
        actions: torch.Tensor,      # (B, K, 9), anchor -> horizon actions
    ) -> torch.Tensor:
        B, K = actions.shape[:2]
        ctx = input_feat.mean(dim=(1, 2, 3))             # (B, C)
        ctx_tok = self.context_proj(ctx).unsqueeze(1)    # (B, 1, D)
        action_tok = self.action_proj(actions)           # (B, K, D)
        tokens = torch.cat([ctx_tok, action_tok], dim=1)
        tokens = tokens + self.pos_embed[:, : K + 1]
        out = self.encoder(tokens)
        return self.out_proj(out[:, 1:])                 # (B, K, C)
