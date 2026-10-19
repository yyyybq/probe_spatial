"""Grounded order metrics and SeeSE3-inspired navigation baselines.

These functions are model agnostic.  They deliberately require both predicted
branches and both physical targets; a low recursive/direct gap alone is never
reported as a success.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from analyses.se3_latent_mvp.geometry import se3_log
from analyses.se3_protocol_v2.geometry import bch_second_log


DEFAULT_TIE_TOLERANCE = 1e-7
DEFAULT_NORMALIZATION_EPSILON = 1e-8


def cosine_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(left.float(), right.float(), dim=-1)).clamp_min(0.0)


def whitened_nrmse(prediction: torch.Tensor, target: torch.Tensor, feature_std: torch.Tensor) -> torch.Tensor:
    error = (prediction.float() - target.float()) / feature_std.float().clamp_min(1e-8)
    return torch.sqrt(torch.mean(error.square(), dim=-1))


def tie_aware_order_summary(
    recursive_error: np.ndarray,
    reversed_error: np.ndarray,
    copy_error: np.ndarray,
    *,
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE,
    epsilon: float = DEFAULT_NORMALIZATION_EPSILON,
) -> dict[str, float | int]:
    """Summarize one-sided observed-order errors without treating ties as losses.

    The normalized margin is intentionally a ratio of aggregate errors, matching
    the frozen paper protocol.  It is not the mean of per-tuple ratios.
    """
    recursive = np.asarray(recursive_error, dtype=np.float64)
    reversed_ = np.asarray(reversed_error, dtype=np.float64)
    copy = np.asarray(copy_error, dtype=np.float64)
    if not (recursive.shape == reversed_.shape == copy.shape) or recursive.ndim != 1:
        raise ValueError("order-error arrays must be aligned one-dimensional arrays")
    if len(recursive) == 0:
        raise ValueError("cannot summarize an empty order-error array")
    margin = reversed_ - recursive
    wins = margin > tie_tolerance
    losses = margin < -tie_tolerance
    ties = ~(wins | losses)
    denominator = float(copy.mean())
    return {
        "raw_margin": float(margin.mean()),
        "normalized_margin": float(margin.mean() / (denominator + epsilon)),
        "wins": int(wins.sum()),
        "losses": int(losses.sum()),
        "ties": int(ties.sum()),
        "tie_aware_accuracy": float((wins.sum() + 0.5 * ties.sum()) / len(margin)),
        "strict_accuracy": float((margin > 0).mean()),
        "copy_denominator": denominator,
        "near_zero_copy_denominator": bool(denominator <= 10 * epsilon),
    }


def grounded_order_metrics(
    recursive_12: torch.Tensor,
    recursive_21: torch.Tensor,
    direct_12: torch.Tensor,
    direct_21: torch.Tensor,
    target_12: torch.Tensor,
    target_21: torch.Tensor,
    feature_std: torch.Tensor,
    physical_commutator: torch.Tensor | None = None,
) -> dict[str, float]:
    """Metrics specified for the paired grounded non-commutativity protocol."""
    cost_correct = cosine_distance(recursive_12, target_12) + cosine_distance(recursive_21, target_21)
    cost_swapped = cosine_distance(recursive_12, target_21) + cosine_distance(recursive_21, target_12)
    margin = cost_swapped - cost_correct
    predicted_sep = cosine_distance(recursive_12, recursive_21)
    target_sep = cosine_distance(target_12, target_21).clamp_min(1e-8)
    ratio = predicted_sep / target_sep
    summary = {
        "recursive_12_endpoint_cosine": float(cosine_distance(recursive_12, target_12).mean()),
        "recursive_21_endpoint_cosine": float(cosine_distance(recursive_21, target_21).mean()),
        "direct_12_endpoint_cosine": float(cosine_distance(direct_12, target_12).mean()),
        "direct_21_endpoint_cosine": float(cosine_distance(direct_21, target_21).mean()),
        "recursive_12_endpoint_whitened_nrmse": float(whitened_nrmse(recursive_12, target_12, feature_std).mean()),
        "recursive_21_endpoint_whitened_nrmse": float(whitened_nrmse(recursive_21, target_21, feature_std).mean()),
        "pair_assignment_accuracy": float((margin > 0).float().mean()),
        "grounded_order_margin_cosine": float(margin.mean()),
        "predicted_to_gt_separation_ratio": float(ratio.mean()),
        "collapse_rate_ratio_lt_0p10": float((ratio < 0.10).float().mean()),
        "collapse_rate_ratio_lt_0p25": float((ratio < 0.25).float().mean()),
        "collapse_rate_ratio_lt_0p50": float((ratio < 0.50).float().mean()),
        "recursive_direct_gap_12_cosine": float(cosine_distance(recursive_12, direct_12).mean()),
        "recursive_direct_gap_21_cosine": float(cosine_distance(recursive_21, direct_21).mean()),
    }
    if physical_commutator is not None:
        x = physical_commutator.detach().float().cpu().numpy()
        y = predicted_sep.detach().float().cpu().numpy()
        summary["predicted_separation_physical_commutator_pearson"] = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0 else float("nan")
    return summary


def sequential_additive_poincare(g_a: torch.Tensor, w_pinv: torch.Tensor, xi1: torch.Tensor, xi2: torch.Tensor) -> torch.Tensor:
    """SeeSE3 inverse-Poincaré additive baseline; intentionally order blind."""
    return g_a + xi1 @ w_pinv.T + xi2 @ w_pinv.T


def composite_aware_poincare(g_a: torch.Tensor, w_pinv: torch.Tensor, transforms: np.ndarray, use_bch: bool = False) -> torch.Tensor:
    """Use either exact ``log(T2@T1)`` or the second-order BCH approximation."""
    vectors = np.stack([bch_second_log(t[0], t[1]) if use_bch else se3_log(t[1] @ t[0]) for t in transforms])
    xi = torch.from_numpy(vectors).to(dtype=g_a.dtype, device=g_a.device)
    return g_a + xi @ w_pinv.T


def retrieval_metrics(prediction: torch.Tensor, gallery_features: torch.Tensor, gallery_translation: torch.Tensor, gallery_rotation_deg: torch.Tensor, correct_index: torch.Tensor, translation_threshold_m: float, rotation_threshold_deg: float) -> dict[str, float]:
    """Physical retrieval against a common endpoint gallery."""
    distance = torch.cdist(prediction.float(), gallery_features.float())
    retrieved = distance.argmin(dim=1)
    correct = retrieved == correct_index
    translation = gallery_translation[retrieved, correct_index]
    rotation = gallery_rotation_deg[retrieved, correct_index]
    return {
        "correct_branch_top1": float(correct.float().mean()),
        "retrieval_translation_error_m": float(translation.mean()),
        "retrieval_rotation_error_deg": float(rotation.mean()),
        "Hit_translation_rotation": float(((translation <= translation_threshold_m) & (rotation <= rotation_threshold_deg)).float().mean()),
    }
