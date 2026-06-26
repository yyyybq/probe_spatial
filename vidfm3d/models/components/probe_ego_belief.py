"""B1 - Ego-Centric Belief probe head.

Given a sequence of S frame features, a per-frame mask identifying the
conditioned object in past frames, and optional final-view global context,
predict the object's polar location in the last frame's camera coordinates.
The final camera pose is not an input.

Input :
    vfm_feat       (B, S, H_f, W_f, C)
    obj_mask_feat  (B, S, H_f, W_f)        -- bool, True where object is visible
    final global context is pooled from vfm_feat[:, -1] when enabled
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
        decoder_type: str = "transformer",
        num_frames: int = 4,
        use_final_global_feature: bool = True,
    ) -> None:
        super().__init__()
        if decoder_type not in {"linear", "mlp", "transformer"}:
            raise ValueError(f"Unknown decoder_type={decoder_type!r}")
        self.decoder_type = decoder_type
        self.num_frames = num_frames
        self.use_final_global_feature = use_final_global_feature
        extra_context = 1 if use_final_global_feature else 0

        if decoder_type == "linear":
            flat_dim = (num_frames + extra_context) * in_channels + num_frames
            self.flat_readout = nn.Linear(flat_dim, 3)
            return
        if decoder_type == "mlp":
            flat_dim = (num_frames + extra_context) * in_channels + num_frames
            self.flat_readout = nn.Sequential(
                nn.LayerNorm(flat_dim),
                nn.Linear(flat_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 3),
            )
            return

        self.feat_proj = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hidden_dim),
        )
        self.frame_query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.frame_query, std=0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len + 1 + extra_context, hidden_dim))
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

        Frames where the mask is empty produce an all-zero token. Transformer
        mode also removes those tokens with its key-padding mask.
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
        if S != self.num_frames:
            raise ValueError(f"Expected {self.num_frames} frames, got {S}")

        per_frame = self._mask_pool(vfm_feat, obj_mask_feat)   # (B, S, C)
        visible = obj_mask_feat.flatten(2).any(dim=-1)         # (B, S)
        final_global = vfm_feat[:, -1].mean(dim=(1, 2))         # (B, C)
        if self.decoder_type != "transformer":
            parts = [per_frame.flatten(1)]
            if self.use_final_global_feature:
                parts.append(final_global)
            parts.append(visible.float())
            flat = torch.cat(parts, dim=-1)
            return self.flat_readout(flat)

        z = self.feat_proj(per_frame)                          # (B, S, D)

        # The query identifies the prediction task only. Object identity comes
        # from masked pooling. Final-view context is global appearance/context,
        # not camera pose.
        query = self.frame_query.expand(B, -1, -1)
        token_parts = [query]
        valid_parts = [torch.ones(B, 1, dtype=torch.bool, device=visible.device)]
        if self.use_final_global_feature:
            final_tok = self.feat_proj(final_global).unsqueeze(1)
            token_parts.append(final_tok)
            valid_parts.append(torch.ones(B, 1, dtype=torch.bool, device=visible.device))
        token_parts.append(z)
        valid_parts.append(visible)
        tokens = torch.cat(token_parts, dim=1)
        tokens = tokens + self.pos_embed[:, : tokens.shape[1]]

        # Invisible-object frames contribute no substitute scene token. The
        # optional final-view token is global context, not an object substitute.
        valid = torch.cat(valid_parts, dim=1)
        out = self.encoder(tokens, src_key_padding_mask=~valid)    # (B, L, D)
        polar = self.head(out[:, 0])                               # (B, 3)
        return polar
