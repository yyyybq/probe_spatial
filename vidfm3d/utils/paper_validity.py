"""Small, deterministic utilities for the paper-validity experiments.

The functions in this module do not touch datasets or checkpoints.  Keeping
intersection, matched shuffling, priors, and prefix integration pure makes the
scientific invariants easy to unit test and reuse from evaluation scripts.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np


PREFIXES = (8, 12, 16, 24)


def canonical_payload_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalized_prefix_auc(
    values: Mapping[int, float] | Sequence[float],
    prefixes: Sequence[int] = PREFIXES,
) -> float:
    """Trapezoidal prefix AUC normalized by the actual prefix span."""
    x = np.asarray(prefixes, dtype=np.float64)
    if x.ndim != 1 or x.size < 2 or np.any(np.diff(x) <= 0):
        raise ValueError("prefixes must be a strictly increasing 1-D sequence")
    if isinstance(values, Mapping):
        y = np.asarray([values[int(prefix)] for prefix in prefixes], dtype=np.float64)
    else:
        y = np.asarray(values, dtype=np.float64)
    if y.shape != x.shape:
        raise ValueError(f"expected {x.size} values, got shape={y.shape}")
    if not np.isfinite(y).all():
        return float("nan")
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def prefix_curve_summary(
    values: Mapping[int, float],
    higher_is_better: bool,
    prefixes: Sequence[int] = PREFIXES,
) -> dict[str, float | int]:
    ordered = {int(prefix): float(values[int(prefix)]) for prefix in prefixes}
    finite = [(prefix, value) for prefix, value in ordered.items() if math.isfinite(value)]
    result: dict[str, float | int] = {
        **{f"p{prefix}": value for prefix, value in ordered.items()},
        "prefix_mean": float(np.mean([value for _, value in finite])) if finite else float("nan"),
        "prefix_auc": normalized_prefix_auc(ordered, prefixes),
        "p24_minus_p8": ordered[int(prefixes[-1])] - ordered[int(prefixes[0])],
    }
    if finite:
        best = max(finite, key=lambda item: item[1]) if higher_is_better else min(finite, key=lambda item: item[1])
        worst = min(finite, key=lambda item: item[1]) if higher_is_better else max(finite, key=lambda item: item[1])
        result["best_prefix"] = int(best[0])
        result["worst_prefix"] = int(worst[0])
    else:
        result["best_prefix"] = -1
        result["worst_prefix"] = -1
    return result


def common_sample_intersection(
    ids_by_config: Mapping[str, Iterable[str]],
) -> tuple[list[str], dict[str, int]]:
    """Return a sorted value-independent intersection and native counts."""
    if not ids_by_config:
        return [], {}
    sets = {}
    counts = {}
    for config, ids_iter in ids_by_config.items():
        ids = [str(value) for value in ids_iter]
        if len(ids) != len(set(ids)):
            duplicates = [item for item, count in Counter(ids).items() if count > 1]
            raise ValueError(f"duplicate sample IDs for {config}: {duplicates[:3]}")
        sets[config] = set(ids)
        counts[config] = len(ids)
    common = sorted(set.intersection(*sets.values()))
    return common, counts


def assert_sample_order(records_by_config: Mapping[str, Sequence[Mapping]], key: str = "sample_id") -> None:
    """Assert exact sample-ID order after callers have applied a common manifest."""
    reference = None
    reference_name = None
    for name, records in records_by_config.items():
        current = [str(record[key]) for record in records]
        if reference is None:
            reference = current
            reference_name = name
        elif current != reference:
            raise AssertionError(f"sample order mismatch: {name} vs {reference_name}")


def quantile_edges(train_values: Sequence[float], n_bins: int = 5) -> np.ndarray:
    """Fit deterministic bin edges from train-only scalar statistics."""
    values = np.asarray(train_values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("cannot fit quantile edges without finite train values")
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    inner = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    return np.unique(inner)


def assign_bin(value: float, edges: Sequence[float]) -> int:
    return int(np.searchsorted(np.asarray(edges, dtype=np.float64), float(value), side="right"))


def matched_derangement(
    records: Sequence[Mapping],
    stratum_fields: Sequence[str],
    scene_field: str = "scene_id",
    sample_id_field: str = "sample_id",
) -> tuple[list[int], list[dict[str, object]]]:
    """Deterministically derange records within exact matched strata.

    A minimum-cost assignment maximizes same-scene matches while prohibiting a
    sample from receiving its own action.  Singleton strata are left unmatched
    (index ``-1``); callers must report rather than silently relax their bins.
    """
    from scipy.optimize import linear_sum_assignment

    groups: dict[tuple, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[tuple(record[field] for field in stratum_fields)].append(index)

    permutation = [-1] * len(records)
    audit = []
    for stratum, indices in sorted(groups.items(), key=lambda item: repr(item[0])):
        indices = sorted(indices, key=lambda idx: str(records[idx][sample_id_field]))
        if len(indices) < 2:
            audit.append({
                "stratum": repr(stratum),
                "n": len(indices),
                "matched": 0,
                "same_scene": 0,
                "fallback": len(indices),
                "reason": "singleton_exact_bin",
            })
            continue
        n = len(indices)
        cost = np.zeros((n, n), dtype=np.float64)
        for row, source_idx in enumerate(indices):
            for col, donor_idx in enumerate(indices):
                if source_idx == donor_idx:
                    cost[row, col] = 1e9
                else:
                    same_scene = records[source_idx].get(scene_field) == records[donor_idx].get(scene_field)
                    # Stable epsilon resolves equal-cost assignments identically.
                    cost[row, col] = (0.0 if same_scene else 1.0) + col * 1e-9
        rows, cols = linear_sum_assignment(cost)
        same_scene_count = 0
        for row, col in zip(rows.tolist(), cols.tolist()):
            source_idx, donor_idx = indices[row], indices[col]
            if source_idx == donor_idx:
                raise AssertionError("matched action assignment is not a derangement")
            permutation[source_idx] = donor_idx
            same_scene_count += int(
                records[source_idx].get(scene_field) == records[donor_idx].get(scene_field)
            )
        audit.append({
            "stratum": repr(stratum),
            "n": n,
            "matched": n,
            "same_scene": same_scene_count,
            "fallback": n - same_scene_count,
            "reason": "exact_bin",
        })

    for source_idx, donor_idx in enumerate(permutation):
        if donor_idx >= 0:
            if donor_idx == source_idx:
                raise AssertionError("self assignment detected")
            for field in stratum_fields:
                if records[source_idx][field] != records[donor_idx][field]:
                    raise AssertionError(f"matched field changed: {field}")
    return permutation, audit


def polar_to_joint_bin(
    azimuth: np.ndarray | float,
    elevation: np.ndarray | float,
    n_az_bins: int = 16,
    n_el_bins: int = 8,
) -> np.ndarray:
    az = np.asarray(azimuth, dtype=np.float64)
    el = np.asarray(elevation, dtype=np.float64)
    az_bin = np.clip(((az + np.pi) / (2 * np.pi) * n_az_bins).astype(int), 0, n_az_bins - 1)
    el_bin = np.clip(((el + np.pi / 2) / np.pi * n_el_bins).astype(int), 0, n_el_bins - 1)
    return az_bin * n_el_bins + el_bin


def joint_bin_center(
    joint_bin: np.ndarray | int,
    n_az_bins: int = 16,
    n_el_bins: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    joint = np.asarray(joint_bin, dtype=np.int64)
    az_bin = joint // n_el_bins
    el_bin = joint % n_el_bins
    az = (az_bin + 0.5) * (2 * np.pi / n_az_bins) - np.pi
    el = (el_bin + 0.5) * (np.pi / n_el_bins) - np.pi / 2
    return az, el


def fit_direction_prior(train_polar: np.ndarray) -> dict[str, object]:
    """Fit the B2 direction MAP and log-distance median from train labels only."""
    polar = np.asarray(train_polar, dtype=np.float64)
    if polar.ndim != 2 or polar.shape[1] != 3 or polar.shape[0] == 0:
        raise ValueError("train_polar must have shape (N,3) with N>0")
    bins = polar_to_joint_bin(polar[:, 0], polar[:, 1])
    counts = np.bincount(bins, minlength=16 * 8)
    return {
        "joint_counts": counts.tolist(),
        "map_joint_bin": int(counts.argmax()),
        "median_log_distance": float(np.median(polar[:, 2])),
        "n_train": int(polar.shape[0]),
    }


def b2_metrics_from_polar(
    gt_polar: np.ndarray,
    pred_polar: np.ndarray,
) -> dict[str, float]:
    gt = np.asarray(gt_polar, dtype=np.float64)
    pred = np.asarray(pred_polar, dtype=np.float64)
    if gt.shape != pred.shape or gt.ndim != 2 or gt.shape[1] != 3:
        raise ValueError("gt_polar and pred_polar must both have shape (N,3)")
    delta_az = np.arctan2(np.sin(pred[:, 0] - gt[:, 0]), np.cos(pred[:, 0] - gt[:, 0]))
    cos_angle = (
        np.sin(pred[:, 1]) * np.sin(gt[:, 1])
        + np.cos(pred[:, 1]) * np.cos(gt[:, 1]) * np.cos(delta_az)
    )
    angular = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    pred_bins = polar_to_joint_bin(pred[:, 0], pred[:, 1])
    gt_bins = polar_to_joint_bin(gt[:, 0], gt[:, 1])
    return {
        "n": int(gt.shape[0]),
        "mean_ang_err_deg": float(np.mean(angular)),
        "mean_az_err_deg": float(np.mean(np.degrees(np.abs(delta_az)))),
        "mean_el_err_deg": float(np.mean(np.degrees(np.abs(pred[:, 1] - gt[:, 1])))),
        "top1": float(np.mean(pred_bins == gt_bins)),
        "mean_log_dist_err": float(np.mean(np.abs(pred[:, 2] - gt[:, 2]))),
    }
