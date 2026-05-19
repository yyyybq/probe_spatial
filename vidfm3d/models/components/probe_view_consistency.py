"""A2 - View Consistency probe head.

Takes raw VFM features (already pixel-aligned to S sampled views) and predicts
a scalar overlap ratio for every ordered pair of frames.

Input :  vfm_feat      (B, S, H_f, W_f, C)
Output:  pred_overlap  (B, S, S)            -- in [0, 1] after sigmoid
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ViewConsistencyProbe(nn.Module):
    def __init__(
        self,
        in_channels: int = 1536,
        hidden_dim: int = 512,
        proj_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
        )
        self.pair_mlp = nn.Sequential(
            nn.LayerNorm(proj_dim * 4),
            nn.Linear(proj_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _pool(self, vfm_feat: torch.Tensor) -> torch.Tensor:
        # (B, S, H_f, W_f, C) -> (B, S, C)
        return vfm_feat.mean(dim=(2, 3))

    def forward(self, vfm_feat: torch.Tensor) -> torch.Tensor:
        """vfm_feat: (B, S, H_f, W_f, C). Returns logits (B, S, S)."""
        B, S = vfm_feat.shape[:2]
        z = self._pool(vfm_feat)             # (B, S, C)
        z = self.proj(z)                     # (B, S, P)

        zi = z.unsqueeze(2).expand(-1, -1, S, -1)   # (B, S, S, P)
        zj = z.unsqueeze(1).expand(-1, S, -1, -1)   # (B, S, S, P)
        pair_feat = torch.cat(
            [zi, zj, zi * zj, (zi - zj).abs()], dim=-1
        )                                    # (B, S, S, 4P)
        logits = self.pair_mlp(pair_feat).squeeze(-1)  # (B, S, S)
        return logits
