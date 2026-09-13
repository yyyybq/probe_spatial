"""Depth, layout and scene-cluster statistics for D-series evaluation."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F


def normalize_depth_per_sample(depth: torch.Tensor, valid: torch.Tensor, eps: float = 1e-6):
    """Normalize metric depth by each sample's valid-pixel mean.

    Returns ``(normalized_depth, scale)``.  ``scale`` is label metadata and must
    never be supplied to a readout or used to restore metric predictions.
    """
    if depth.shape != valid.shape:
        raise ValueError("depth and valid mask must have the same shape")
    reduce_dims = tuple(range(1, depth.ndim))
    weights = valid.to(depth.dtype)
    scale = (depth * weights).sum(reduce_dims) / weights.sum(reduce_dims).clamp_min(1)
    view = (depth.shape[0],) + (1,) * (depth.ndim - 1)
    normalized = torch.where(valid, depth / scale.clamp_min(eps).view(view), torch.zeros_like(depth))
    return normalized, scale


def metric_depth_to_camera_pointmap(depth: torch.Tensor, intrinsic: torch.Tensor) -> torch.Tensor:
    """Back-project metric depth to target-camera XYZ in metres."""
    squeeze = depth.ndim == 2
    if squeeze:
        depth, intrinsic = depth[None], intrinsic[None]
    if depth.ndim != 3 or intrinsic.shape[-2:] != (3, 3):
        raise ValueError("expected depth B,H,W and intrinsics B,3,3")
    batch, height, width = depth.shape
    v, u = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype), indexing="ij",
    )
    uv1 = torch.stack((u, v, torch.ones_like(u)), -1).expand(batch, -1, -1, -1)
    # Row-vector form of ``K^{-1} @ [u,v,1]^T``.  The output index ``d``
    # selects a row of K^{-1}; transposing K^{-1} here would be incorrect for
    # non-zero principal points even though it passes an identity-K smoke test.
    rays = torch.einsum("bhwc,bdc->bhwd", uv1, torch.linalg.inv(intrinsic))
    points = rays * depth[..., None]
    return points[0] if squeeze else points


def resize_depth_validity_weighted(
    depth: torch.Tensor, valid: torch.Tensor, size=(72, 128), eps: float = 1e-8,
):
    """Resize depth without bleeding invalid zero pixels into valid labels."""
    if depth.shape != valid.shape:
        raise ValueError("depth and valid mask must have the same shape")
    squeeze = depth.ndim == 2
    if squeeze:
        depth, valid = depth[None], valid[None]
    if depth.ndim != 3:
        raise ValueError("expected H,W or B,H,W")
    weight = valid.to(depth.dtype).unsqueeze(1)
    numerator = F.interpolate((depth * valid).unsqueeze(1), size, mode="area")
    denominator = F.interpolate(weight, size, mode="area")
    resized_valid = denominator > eps
    resized = torch.where(resized_valid, numerator / denominator.clamp_min(eps), torch.zeros_like(numerator))
    if squeeze:
        return resized[0, 0], resized_valid[0, 0]
    return resized[:, 0], resized_valid[:, 0]


def _flat_valid(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor):
    pred, target, valid = torch.broadcast_tensors(pred, target, valid)
    return pred[valid], target[valid]


def depth_metrics(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, eps: float = 1e-6):
    """Return per-sample normalized-depth metrics using the GT mask only."""
    if pred.ndim == target.ndim + 1 and pred.shape[1] == 1:
        pred = pred[:, 0]
    if target.ndim == pred.ndim + 1 and target.shape[1] == 1:
        target = target[:, 0]
    if valid.ndim == pred.ndim + 1 and valid.shape[1] == 1:
        valid = valid[:, 0]
    rows = []
    for p, t, m in zip(pred, target, valid.bool()):
        pv, tv = _flat_valid(p, t, m)
        if tv.numel() == 0:
            rows.append({"mae": float("nan"), "absrel": float("nan"), "delta1": float("nan"), "n_valid": 0})
            continue
        ratio = torch.maximum(pv.clamp_min(eps) / tv.clamp_min(eps), tv.clamp_min(eps) / pv.clamp_min(eps))
        rows.append({
            "mae": (pv - tv).abs().mean().item(),
            "absrel": ((pv - tv).abs() / tv.clamp_min(eps)).mean().item(),
            "delta1": (ratio < 1.25).float().mean().item(),
            "n_valid": int(tv.numel()),
        })
    return rows


def coarse_layout_mae(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, grid=(4, 4)):
    """Per-sample error between masked regional mean depths on a fixed grid."""
    if pred.ndim == 4: pred = pred[:, 0]
    if target.ndim == 4: target = target[:, 0]
    if valid.ndim == 4: valid = valid[:, 0]
    weights = valid.float().unsqueeze(1)
    size = tuple(int(v) for v in grid)
    pnum = F.adaptive_avg_pool2d((pred * valid).unsqueeze(1), size)
    tnum = F.adaptive_avg_pool2d((target * valid).unsqueeze(1), size)
    den = F.adaptive_avg_pool2d(weights, size)
    cell_valid = den > 0
    error = ((pnum / den.clamp_min(1e-8)) - (tnum / den.clamp_min(1e-8))).abs()
    return (error * cell_valid).flatten(1).sum(1) / cell_valid.flatten(1).sum(1).clamp_min(1)


def detail_gradient_mae(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor):
    """Per-sample local-detail error over GT-valid horizontal/vertical neighbours."""
    if pred.ndim == 4: pred = pred[:, 0]
    if target.ndim == 4: target = target[:, 0]
    if valid.ndim == 4: valid = valid[:, 0]
    errors, counts = [], []
    for dim in (1, 2):
        dp = torch.diff(pred, dim=dim)
        dt = torch.diff(target, dim=dim)
        if dim == 1: m = valid[:, 1:] & valid[:, :-1]
        else: m = valid[:, :, 1:] & valid[:, :, :-1]
        errors.append(((dp - dt).abs() * m).flatten(1).sum(1))
        counts.append(m.flatten(1).sum(1))
    return sum(errors) / sum(counts).clamp_min(1)


def scene_macro(records: Iterable[Mapping], metric: str = "mae", scene_key: str = "scene_id") -> float:
    per_scene = defaultdict(list)
    for row in records:
        value = float(row[metric])
        if np.isfinite(value): per_scene[str(row[scene_key])].append(value)
    means = [np.mean(v) for v in per_scene.values() if v]
    return float(np.mean(means)) if means else float("nan")


def scene_cluster_paired_bootstrap(
    records: Iterable[Mapping], method_a: str, method_b: str, *, metric: str = "mae",
    method_key: str = "condition", pair_key: str = "sample_id", scene_key: str = "scene_id",
    n_boot: int = 10_000, seed: int = 42,
):
    """Paired A-B scene-macro difference and 95% scene-cluster bootstrap CI."""
    paired = defaultdict(dict)
    for row in records:
        if row.get(method_key) in {method_a, method_b}:
            paired[(str(row[scene_key]), str(row[pair_key]))][str(row[method_key])] = float(row[metric])
    scene_diffs = defaultdict(list)
    for (scene, _), values in paired.items():
        if method_a in values and method_b in values:
            scene_diffs[scene].append(values[method_a] - values[method_b])
    values = np.asarray([np.mean(v) for v in scene_diffs.values()], dtype=np.float64)
    if values.size == 0:
        return {"delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_scenes": 0}
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return {"delta": float(values.mean()), "ci_low": float(lo), "ci_high": float(hi), "n_scenes": int(values.size)}


def latent_diagnostics(predicted: torch.Tensor, target: torch.Tensor):
    p, t = predicted.flatten(1).float(), target.flatten(1).float()
    return {
        "cosine": F.cosine_similarity(p, t, dim=1),
        "pred_variance": p.var(dim=1, unbiased=False),
        "target_variance": t.var(dim=1, unbiased=False),
    }
