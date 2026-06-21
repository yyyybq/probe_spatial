"""B1 - Ego-Centric Belief probe head.

Given a sequence of S frame features and a per-frame mask identifying the
conditioned object in past frames, predict its polar location in the last
frame's camera coordinates. The reference frame is a target convention, not
an explicit camera-pose input.

Input :
    vfm_feat       (B, S, H_f, W_f, C)
    obj_mask_feat  (B, S, H_f, W_f)        -- bool, True where object is visible
Output:
    polar_pred     (B, 3)                  -- (azimuth, elevation, log_dist)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EgoBeliefProbe(nn.Module):
    def __init__(
        self,
        in_channels: int = 1536,
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
        self.frame_query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.frame_query, std=0.02)
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
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    @staticmethod
    def _mask_pool(
        vfm_feat: torch.Tensor,    # (B, S, H_f, W_f, C)
        obj_mask: torch.Tensor,    # (B, S, H_f, W_f) bool/float
    ) -> torch.Tensor:
        """Pool features within the object mask per frame.

        Frames where the mask is empty receive a global-pooled feature so the
        downstream model still gets some signal but later attention can ignore
        them via the validity flag.
        """
        m = obj_mask.float()
        counts = m.sum(dim=(2, 3))                               # (B, S)
        denom = counts.clamp(min=1e-3).unsqueeze(-1)             # (B, S, 1)
        masked = vfm_feat * m.unsqueeze(-1)
        return masked.sum(dim=(2, 3)) / denom                    # (B, S, C)

    def forward(
        self,
        vfm_feat: torch.Tensor,         # (B, S, H_f, W_f, C)
        obj_mask_feat: torch.Tensor,    # (B, S, H_f, W_f) bool
    ) -> torch.Tensor:
        B, S = vfm_feat.shape[:2]

        per_frame = self._mask_pool(vfm_feat, obj_mask_feat)   # (B, S, C)
        z = self.feat_proj(per_frame)                          # (B, S, D)

        # The query identifies the prediction task only. Object identity comes
        # from masked pooling; the final-frame reference is implied by order.
        query = self.frame_query.expand(B, -1, -1)
        tokens = torch.cat([query, z], dim=1)                  # (B, S+1, D)
        tokens = tokens + self.pos_embed[:, : S + 1]

        # Invisible-object frames contribute no substitute scene token. The
        # final frame defines the GT coordinate convention, not an input.
        visible = obj_mask_feat.flatten(2).any(dim=-1)             # (B, S)
        valid = torch.cat(
            [torch.ones(B, 1, dtype=torch.bool, device=visible.device), visible],
            dim=1,
        )
        out = self.encoder(tokens, src_key_padding_mask=~valid)    # (B, S+1, D)
        polar = self.head(out[:, 0])                               # (B, 3)
        return polar
