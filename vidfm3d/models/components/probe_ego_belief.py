"""B1 - Ego-Centric Belief probe head.

Given a sequence of S frame features and a per-frame mask telling where the
hidden object was visible in each past frame (zeros for the last frame),
predict the object's polar location (azimuth, elevation, log_distance) in
the last frame's camera coordinates.

Input :
    vfm_feat       (B, S, H_f, W_f, C)
    obj_mask_feat  (B, S, H_f, W_f)        -- bool, True where object is visible
    last_pose_enc  (B, 9)                  -- 6D rot + 3 trans of last extrinsic
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
        pose_dim: int = 9,
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
        self.pose_proj = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
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
        denom = m.sum(dim=(2, 3)).clamp(min=1e-3).unsqueeze(-1)  # (B, S, 1)
        masked = vfm_feat * m.unsqueeze(-1)
        pooled = masked.sum(dim=(2, 3)) / denom                  # (B, S, C)
        return pooled

    def forward(
        self,
        vfm_feat: torch.Tensor,         # (B, S, H_f, W_f, C)
        obj_mask_feat: torch.Tensor,    # (B, S, H_f, W_f) bool
        last_pose_enc: torch.Tensor,    # (B, 9)
    ) -> torch.Tensor:
        B, S = vfm_feat.shape[:2]

        per_frame = self._mask_pool(vfm_feat, obj_mask_feat)   # (B, S, C)
        z = self.feat_proj(per_frame)                          # (B, S, D)

        # Mark last frame's slot with the pose-conditioned query
        query = self.frame_query.expand(B, -1, -1) + self.pose_proj(last_pose_enc).unsqueeze(1)
        tokens = torch.cat([query, z], dim=1)                  # (B, S+1, D)
        tokens = tokens + self.pos_embed[:, : S + 1]

        # Build attention mask: ignore frames whose obj_mask is empty
        with torch.no_grad():
            valid_per_frame = obj_mask_feat.flatten(2).any(dim=-1)  # (B, S)
            # query token always valid
            valid = torch.cat(
                [torch.ones(B, 1, dtype=torch.bool, device=valid_per_frame.device),
                 valid_per_frame], dim=1
            )                                                     # (B, S+1)
        # nn.TransformerEncoder takes src_key_padding_mask: True = ignore
        pad_mask = ~valid

        out = self.encoder(tokens, src_key_padding_mask=pad_mask)  # (B, S+1, D)
        polar = self.head(out[:, 0])                               # (B, 3)
        return polar
