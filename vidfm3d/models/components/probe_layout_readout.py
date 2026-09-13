"""Capacity-controlled readouts and predictors for the D-series study.

All tensors use channels-last latent layouts.  A spatial latent is ``B,H,W,C``
or ``B,T,H,W,C``; pooled latents are ``B,C``.  The modules deliberately avoid
an unrestricted ``C -> H*W*hidden`` projection.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _last_time(latent: torch.Tensor) -> torch.Tensor:
    if latent.ndim == 5:
        return latent[:, -1]
    if latent.ndim != 4:
        raise ValueError(f"expected B,H,W,C or B,T,H,W,C; got {tuple(latent.shape)}")
    return latent


def pooled_latent(latent: torch.Tensor) -> torch.Tensor:
    """Pool every non-batch/non-channel axis without changing channel order."""
    if latent.ndim < 2:
        raise ValueError(f"latent must have at least two dimensions; got {latent.ndim}")
    if latent.ndim == 2:
        return latent
    return latent.mean(dim=tuple(range(1, latent.ndim - 1)))


class SpatialDepthReadout(nn.Module):
    """Shared token-wise shallow head followed by fixed bilinear restoration."""

    def __init__(self, in_channels: int, hidden_dim: int = 128, output_size=(72, 128)):
        super().__init__()
        self.output_size = tuple(int(v) for v in output_size)
        self.norm = nn.LayerNorm(in_channels)
        self.token_head = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        latent = _last_time(latent)
        depth = self.token_head(self.norm(latent)).squeeze(-1).unsqueeze(1)
        return F.softplus(F.interpolate(depth, self.output_size, mode="bilinear", align_corners=False))


class LinearSpatialDepthReadout(nn.Module):
    """Weak-capacity token-wise linear control."""

    def __init__(self, in_channels: int, output_size=(72, 128)):
        super().__init__()
        self.output_size = tuple(int(v) for v in output_size)
        self.head = nn.Linear(in_channels, 1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        depth = self.head(_last_time(latent)).squeeze(-1).unsqueeze(1)
        return F.softplus(F.interpolate(depth, self.output_size, mode="bilinear", align_corners=False))


class PooledDepthReadout(nn.Module):
    """Pooled readout with a learned low-resolution grid and small CNN decoder.

    The sample-dependent path is only ``C -> hidden``.  The learned grid is
    shared across samples, preventing a giant sample-specific dense image head.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 128,
        grid_size=(9, 16),
        output_size=(72, 128),
    ):
        super().__init__()
        self.grid_size = tuple(int(v) for v in grid_size)
        self.output_size = tuple(int(v) for v in output_size)
        self.code = nn.Sequential(nn.LayerNorm(in_channels), nn.Linear(in_channels, hidden_dim))
        self.grid = nn.Parameter(torch.empty(1, hidden_dim, *self.grid_size))
        nn.init.trunc_normal_(self.grid, std=0.02)
        mid = max(hidden_dim // 2, 16)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, mid, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid, 1, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        code = self.code(pooled_latent(latent))[:, :, None, None]
        depth = self.decoder(code + self.grid)
        return F.softplus(F.interpolate(depth, self.output_size, mode="bilinear", align_corners=False))


class TokenwiseLinearAdapter(nn.Module):
    """One affine map shared by every temporal/spatial position."""

    def __init__(self, channels: int, bias: bool = True):
        super().__init__()
        self.affine = nn.Linear(channels, channels, bias=bias)
        nn.init.eye_(self.affine.weight)
        if self.affine.bias is not None:
            nn.init.zeros_(self.affine.bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.affine(latent)


class ExternalPooledPredictor(nn.Module):
    """Shared C1 architecture for pooled target-latent prediction.

    ``conditioning='motion'`` receives a geometrically encoded relative pose;
    ``conditioning='context_only'`` is an independently trained equal-capacity
    control whose motion token is a learned constant.  It is not a test-time
    zeroing ablation.
    """

    def __init__(
        self,
        in_channels: int,
        motion_dim: int = 12,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        max_history: int = 24,
        conditioning: str = "motion",
        dropout: float = 0.1,
    ):
        super().__init__()
        if conditioning not in {"motion", "context_only"}:
            raise ValueError("conditioning must be 'motion' or 'context_only'")
        self.conditioning = conditioning
        self.context_proj = nn.Sequential(nn.LayerNorm(in_channels), nn.Linear(in_channels, hidden_dim))
        self.motion_proj = nn.Sequential(
            nn.LayerNorm(motion_dim), nn.Linear(motion_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.constant_motion = nn.Parameter(torch.zeros(1, motion_dim))
        self.query = nn.Parameter(torch.empty(1, 1, hidden_dim))
        self.position = nn.Parameter(torch.empty(1, max_history + 2, hidden_dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            hidden_dim, num_heads, hidden_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, in_channels),
        )

    def forward(self, context_latent: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        if context_latent.ndim == 5:
            context_latent = context_latent.mean(dim=(2, 3))
        if context_latent.ndim != 3:
            raise ValueError(f"context must be B,T,C or B,T,H,W,C; got {tuple(context_latent.shape)}")
        batch, history, _ = context_latent.shape
        if history + 2 > self.position.shape[1]:
            raise ValueError(f"history {history} exceeds configured maximum")
        if self.conditioning == "context_only":
            motion = self.constant_motion.expand(batch, -1)
        tokens = torch.cat([
            self.context_proj(context_latent),
            self.motion_proj(motion).unsqueeze(1),
            self.query.expand(batch, -1, -1),
        ], dim=1)
        tokens = tokens + self.position[:, : history + 2]
        return self.output(self.encoder(tokens)[:, -1])


def parameter_count(module: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)
