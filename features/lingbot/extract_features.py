#!/usr/bin/env python3
"""Helpers for LingBot-Video feature tensors."""

from __future__ import annotations

from typing import Dict

import torch

from .lingbot_feature import LingBotFeaturizer


def reshape_to_t_h_w_c(
    raw: torch.Tensor,
    *,
    t_tokens: int,
    h_tokens: int,
    w_tokens: int,
) -> torch.Tensor:
    """``(1, T*H*W, C)`` → ``(T, H, W, C)``."""
    assert raw.ndim == 3, f"Expected 3D tensor, got {raw.shape}"
    assert raw.shape[0] == 1, f"Expected batch size 1, got {raw.shape[0]}"
    expected = t_tokens * h_tokens * w_tokens
    assert raw.shape[1] == expected, f"Expected {expected} tokens, got {raw.shape[1]}"
    return (
        raw.squeeze(0)
        .reshape(t_tokens, h_tokens, w_tokens, raw.shape[-1])
        .contiguous()
    )


def reshape_featurizer_output(
    feats: Dict[int, torch.Tensor],
    model: LingBotFeaturizer,
) -> Dict[int, torch.Tensor]:
    return {
        layer: reshape_to_t_h_w_c(
            raw, t_tokens=model.feat_t, h_tokens=model.feat_h, w_tokens=model.feat_w
        )
        for layer, raw in feats.items()
    }
