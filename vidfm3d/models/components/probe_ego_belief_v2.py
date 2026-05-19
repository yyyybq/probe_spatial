"""B2 - Ego-Centric Belief probe v2 (query-token formulation).

This probe tests the same scientific question as B1 but removes the privileged
inputs (GT obj mask + ground-truth past camera poses).  The model must do its
own re-identification and viewpoint reasoning, using ONLY:

    vfm_feat:    (B, S, H_f, W_f, C)   -- S frames of patch features (the last
                                           frame is the "current" view that
                                           anchors the egocentric frame)
    query_feat:  (B, C)                -- a 1-D appearance signature of the
                                           hidden object, obtained externally
                                           by masked-pooling VFM features at
                                           one past frame.  Carries NO spatial
                                           location and NO pose information.

Output: a discrete belief distribution over (azimuth, elevation) bins in the
last frame's camera coordinate system, plus a regressed log-distance scalar.

The probe internally:
    1. Embeds all patches with shared frame-role embeddings (past vs current)
       and 2-D spatial position embeddings.
    2. Adds a single learnable "belief" CLS token initialized by the query
       embedding.
    3. Runs Transformer self-attention over (belief + patches).
    4. Reads off the belief token and decodes joint (az, el) logits plus a
       log-distance scalar.

The model is never told which frame is "current"; we adopt the convention that
the LAST frame in the sequence is current and tag it with a learned role
embedding.  This is the only piece of side information the probe receives.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EgoBeliefProbeV2(nn.Module):
    def __init__(
        self,
        in_channels: int = 1536,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        max_seq_len: int = 8,
        max_h: int = 36,
        max_w: int = 64,
        n_az_bins: int = 16,
        n_el_bins: int = 8,
        patch_pool: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_az_bins = n_az_bins
        self.n_el_bins = n_el_bins
        self.patch_pool = patch_pool

        self.feat_proj = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hidden_dim),
        )
        self.query_proj = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hidden_dim),
        )

        # learnable belief token (added on top of the projected query)
        self.belief_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.belief_token, std=0.02)

        # Spatial 2-D pos embed (factorized) — sized for downsampled grid
        max_h_p = max_h // patch_pool + 1
        max_w_p = max_w // patch_pool + 1
        self.row_embed = nn.Parameter(torch.zeros(1, max_h_p, hidden_dim))
        self.col_embed = nn.Parameter(torch.zeros(1, max_w_p, hidden_dim))
        nn.init.normal_(self.row_embed, std=0.02)
        nn.init.normal_(self.col_embed, std=0.02)

        # Frame role embeddings: 0 = past, 1 = current
        self.role_embed = nn.Parameter(torch.zeros(2, hidden_dim))
        nn.init.normal_(self.role_embed, std=0.02)

        # Frame index embedding (treat all past frames as exchangeable but
        # still give them a small distinct id so attention can differentiate).
        self.frame_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        nn.init.normal_(self.frame_embed, std=0.02)

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

        self.norm_out = nn.LayerNorm(hidden_dim)
        self.bin_head = nn.Linear(hidden_dim, n_az_bins * n_el_bins)
        self.dist_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def _patchify(self, vfm_feat: torch.Tensor) -> torch.Tensor:
        """(B, S, H, W, C) -> (B, S, Hp, Wp, C) with avg-pool patchification."""
        if self.patch_pool == 1:
            return vfm_feat
        B, S, H, W, C = vfm_feat.shape
        x = vfm_feat.permute(0, 1, 4, 2, 3).reshape(B * S, C, H, W)
        x = torch.nn.functional.avg_pool2d(x, kernel_size=self.patch_pool)
        Hp, Wp = x.shape[-2:]
        x = x.reshape(B, S, C, Hp, Wp).permute(0, 1, 3, 4, 2)
        return x

    def forward(
        self,
        vfm_feat: torch.Tensor,    # (B, S, H_f, W_f, C)
        query_feat: torch.Tensor,  # (B, C)
    ) -> dict:
        feat = self._patchify(vfm_feat)                       # (B, S, Hp, Wp, C)
        B, S, Hp, Wp, _ = feat.shape

        z = self.feat_proj(feat)                              # (B, S, Hp, Wp, D)

        # 2-D spatial pos embed (factorized broadcast)
        row = self.row_embed[:, :Hp].unsqueeze(2)              # (1, Hp, 1, D)
        col = self.col_embed[:, :Wp].unsqueeze(1)              # (1, 1, Wp, D)
        spatial = row + col                                    # (1, Hp, Wp, D)
        z = z + spatial.unsqueeze(1)                           # broadcast over S

        # Frame role + index embeddings
        role = self.role_embed[0].expand(S, -1).clone()        # all past
        role[-1] = self.role_embed[1]                          # last = current
        z = z + role.view(1, S, 1, 1, -1)
        z = z + self.frame_embed[:, :S].view(1, S, 1, 1, -1)

        # Flatten to (B, S*Hp*Wp, D)
        z = z.reshape(B, S * Hp * Wp, -1)

        # Belief token: learned + projected query
        q = self.query_proj(query_feat).unsqueeze(1) + self.belief_token  # (B, 1, D)

        tokens = torch.cat([q, z], dim=1)                       # (B, 1 + N, D)
        out = self.encoder(tokens)
        belief = self.norm_out(out[:, 0])                       # (B, D)

        logits = self.bin_head(belief).reshape(B, self.n_az_bins, self.n_el_bins)
        log_dist = self.dist_head(belief).squeeze(-1)           # (B,)
        return {"logits": logits, "log_dist": log_dist}
