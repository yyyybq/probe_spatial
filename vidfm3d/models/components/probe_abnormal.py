"""A3 - Abnormal Video probe head.

Given S spatially-pooled frame features, predict a binary "abnormal" logit.
A small Transformer with learned positional encoding consumes the sequence;
position embeddings are essential because we want the head to be sensitive
to temporal order (otherwise it can't tell normal from temporally-shuffled
features extracted under scrambled context).

Input :  vfm_feat (B, S, H_f, W_f, C)
Output:  logit    (B,)  -- pre-sigmoid abnormality score
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AbnormalVideoProbe(nn.Module):
    def __init__(
        self,
        in_channels: int = 1536,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        max_seq_len: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.proj_in = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hidden_dim),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len + 1, hidden_dim))
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
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, vfm_feat: torch.Tensor) -> torch.Tensor:
        # vfm_feat: (B, S, H_f, W_f, C)
        B, S = vfm_feat.shape[:2]
        z = vfm_feat.mean(dim=(2, 3))            # (B, S, C)
        z = self.proj_in(z)                      # (B, S, D)
        cls = self.cls_token.expand(B, -1, -1)   # (B, 1, D)
        tokens = torch.cat([cls, z], dim=1)      # (B, S+1, D)
        tokens = tokens + self.pos_embed[:, : S + 1]
        out = self.encoder(tokens)               # (B, S+1, D)
        logit = self.head(out[:, 0]).squeeze(-1)  # (B,)
        return logit
