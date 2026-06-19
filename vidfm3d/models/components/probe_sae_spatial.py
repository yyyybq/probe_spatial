"""Sparse autoencoder probe for VLM/unified-model spatial representations."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSAESpatialProbe(nn.Module):
    """Train a Top-k SAE and lightweight spatial readouts on frozen activations.

    The SAE is unsupervised by default.  Spatial readouts consume sparse codes
    detached from the SAE graph, so labels measure information already exposed
    by the dictionary rather than steering dictionary learning.
    """

    def __init__(
        self,
        in_channels: int,
        dict_size: int = 8192,
        k: int = 64,
        hidden_dim: int = 512,
        max_tokens_per_batch: int = 8192,
        detach_readout: bool = True,
        use_overlap_readout: bool = True,
        use_ego_readout: bool = True,
    ) -> None:
        super().__init__()
        if k <= 0 or k > dict_size:
            raise ValueError(f"k must be in [1, dict_size], got k={k}, dict_size={dict_size}")
        self.in_channels = in_channels
        self.dict_size = dict_size
        self.k = k
        self.max_tokens_per_batch = max_tokens_per_batch
        self.detach_readout = detach_readout
        self.use_overlap_readout = use_overlap_readout
        self.use_ego_readout = use_ego_readout

        self.pre_norm = nn.LayerNorm(in_channels)
        self.encoder = nn.Linear(in_channels, dict_size)
        self.decoder = nn.Linear(dict_size, in_channels, bias=False)
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        nn.init.normal_(self.decoder.weight, std=1.0 / math.sqrt(dict_size))

        pair_dim = dict_size * 4
        self.overlap_head = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.ego_head = nn.Sequential(
            nn.LayerNorm(dict_size * 2),
            nn.Linear(dict_size * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        acts = F.relu(self.encoder(self.pre_norm(x)))
        values, indices = torch.topk(acts, k=self.k, dim=-1)
        sparse = torch.zeros_like(acts)
        sparse.scatter_(-1, indices, values)
        return sparse

    def decode(self, sparse: torch.Tensor) -> torch.Tensor:
        return self.decoder(sparse)

    def _sample_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[0] <= self.max_tokens_per_batch:
            return tokens
        idx = torch.randperm(tokens.shape[0], device=tokens.device)[: self.max_tokens_per_batch]
        return tokens[idx]

    def forward(self, vfm_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        # vfm_feat: (B, S, H, W, C)
        B, S, H, W, C = vfm_feat.shape
        tokens = vfm_feat.reshape(B * S * H * W, C)
        sampled = self._sample_tokens(tokens)
        sparse = self.encode(sampled)
        recon = self.decode(sparse)

        frame_sparse = self.encode(vfm_feat.reshape(B, S, H * W, C)).mean(dim=2)
        if self.detach_readout:
            frame_sparse = frame_sparse.detach()

        return {
            "sampled_tokens": sampled,
            "sparse": sparse,
            "recon": recon,
            "frame_sparse": frame_sparse,
        }

    def predict_overlap(self, frame_sparse: torch.Tensor) -> torch.Tensor:
        # frame_sparse: (B, S, D)
        B, S, D = frame_sparse.shape
        left = frame_sparse.unsqueeze(2).expand(B, S, S, D)
        right = frame_sparse.unsqueeze(1).expand(B, S, S, D)
        pair = torch.cat([left, right, (left - right).abs(), left * right], dim=-1)
        return self.overlap_head(pair).squeeze(-1)

    def predict_ego(self, frame_sparse: torch.Tensor, query_feat: torch.Tensor | None = None) -> torch.Tensor:
        current = frame_sparse[:, -1]
        if query_feat is None:
            query = frame_sparse[:, :-1].mean(dim=1) if frame_sparse.shape[1] > 1 else current
        else:
            query = self.encode(query_feat)
            if self.detach_readout:
                query = query.detach()
        return self.ego_head(torch.cat([query, current], dim=-1))
