"""C1 - Action-Conditioned Prediction probe head.

Given input features for the first (S-1) frames and a 9-D action token (6D rot
+ 3 trans = relative camera pose from input-last-frame to target frame),
predict the spatially-pooled VFM feature of the target frame.

Input :
    input_feat  (B, S-1, H_f, W_f, C)
    action      (B, 9)
Output:
    pred_feat   (B, C)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ActionDynamicsProbe(nn.Module):
    def __init__(
        self,
        in_channels: int = 1536,
        action_dim: int = 9,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        max_seq_len: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feat_proj = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hidden_dim),
        )
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.query_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.query_token, std=0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len + 2, hidden_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_channels),
        )

    def forward(
        self,
        input_feat: torch.Tensor,   # (B, S-1, H_f, W_f, C)
        action: torch.Tensor,       # (B, 9)
    ) -> torch.Tensor:
        B, T = input_feat.shape[:2]
        z = input_feat.mean(dim=(2, 3))                # (B, T, C)
        z = self.feat_proj(z)                          # (B, T, D)

        action_tok = self.action_proj(action).unsqueeze(1)        # (B, 1, D)
        query = self.query_token.expand(B, -1, -1)                # (B, 1, D)

        tokens = torch.cat([query, action_tok, z], dim=1)         # (B, T+2, D)
        tokens = tokens + self.pos_embed[:, : T + 2]
        out = self.encoder(tokens)                                # (B, T+2, D)
        pred = self.out_proj(out[:, 0])                           # (B, C)
        return pred
