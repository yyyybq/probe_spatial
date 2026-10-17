"""Utilities for the Wan p8 action-dynamics construct-validity audit."""

from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


METRIC_NAMES = ("cos", "r1", "r5", "mrr", "rank0", "rank1")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return sha256_bytes(array.tobytes(order="C"))


def retrieval_per_sample(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    positive_indices: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """Return paired retrieval outcomes against one fixed candidate pool.

    ``positive_indices[i]`` identifies the target candidate paired with prediction
    ``i``. Rank0 matches the historical canonical implementation; rank1 is the
    conventional one-based rank used in the validity report.
    """
    if predictions.ndim != 2 or targets.ndim != 2:
        raise ValueError("predictions and targets must both be rank-2")
    if predictions.shape[0] == 0 or targets.shape[0] == 0:
        raise ValueError("retrieval inputs must be non-empty")
    if predictions.shape[1] != targets.shape[1]:
        raise ValueError("prediction and target feature dimensions differ")
    if positive_indices is None:
        if predictions.shape[0] != targets.shape[0]:
            raise ValueError("diagonal positives require equal prediction/target counts")
        positive_indices = torch.arange(predictions.shape[0], device=predictions.device)
    else:
        positive_indices = positive_indices.to(predictions.device, dtype=torch.long)
    if positive_indices.shape != (predictions.shape[0],):
        raise ValueError("positive_indices has the wrong shape")
    if int(positive_indices.min()) < 0 or int(positive_indices.max()) >= targets.shape[0]:
        raise ValueError("positive_indices points outside the candidate pool")

    pred_norm = F.normalize(predictions.float(), dim=-1)
    target_norm = F.normalize(targets.float(), dim=-1)
    similarity = pred_norm @ target_norm.T
    rows = torch.arange(predictions.shape[0], device=predictions.device)
    positive_cos = similarity[rows, positive_indices]
    ordered = similarity.argsort(dim=-1, descending=True)
    rank0 = (ordered == positive_indices[:, None]).to(torch.int64).argmax(dim=-1)
    return {
        "cos": positive_cos,
        "r1": (rank0 == 0).float(),
        "r5": (rank0 < 5).float(),
        "mrr": 1.0 / (rank0.float() + 1.0),
        "rank0": rank0.float(),
        "rank1": rank0.float() + 1.0,
    }


def aggregate_retrieval(per_sample: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {name: float(per_sample[name].float().mean().item()) for name in METRIC_NAMES}


def action_descriptors(actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Convert 6D rotation + translation actions to deterministic motion strata."""
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 9:
        raise ValueError("actions must have shape (N, 9)")
    first = actions[:, :3]
    second = actions[:, 3:6]
    first /= np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-12)
    second -= (first * second).sum(axis=1, keepdims=True) * first
    second /= np.maximum(np.linalg.norm(second, axis=1, keepdims=True), 1e-12)
    third = np.cross(first, second)
    rotation = np.stack([first, second, third], axis=1)

    trace = np.trace(rotation, axis1=1, axis2=2)
    angle = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    skew = np.stack(
        [
            rotation[:, 2, 1] - rotation[:, 1, 2],
            rotation[:, 0, 2] - rotation[:, 2, 0],
            rotation[:, 1, 0] - rotation[:, 0, 1],
        ],
        axis=1,
    )
    denom = 2.0 * np.sin(angle)
    axis = np.zeros_like(skew)
    stable = np.abs(denom) > 1e-7
    axis[stable] = skew[stable] / denom[stable, None]
    rotvec = axis * angle[:, None]
    descriptor = np.concatenate([rotvec, actions[:, 6:9]], axis=1)
    rms = np.sqrt(np.mean(descriptor**2, axis=0))
    rms = np.maximum(rms, 1e-12)
    standardized = descriptor / rms[None, :]
    dominant = np.argmax(np.abs(standardized), axis=1)
    signs = descriptor[np.arange(len(descriptor)), dominant] >= 0
    axes = ("rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z")
    labels = np.asarray([f"{axes[i]}_{'pos' if sign else 'neg'}" for i, sign in zip(dominant, signs)])
    return descriptor, labels, list(axes)


def frequency_blocks(descriptor: np.ndarray, n_blocks: int = 10) -> np.ndarray:
    """Assign equal-frequency blocks by standardized motion magnitude."""
    descriptor = np.asarray(descriptor, dtype=np.float64)
    scale = np.sqrt(np.mean(descriptor**2, axis=0))
    standardized = descriptor / np.maximum(scale, 1e-12)[None, :]
    magnitude = np.linalg.norm(standardized, axis=1)
    order = np.argsort(magnitude, kind="mergesort")
    blocks = np.empty(len(order), dtype=np.int64)
    for block, indices in enumerate(np.array_split(order, n_blocks)):
        blocks[indices] = block
    return blocks


def make_block_derangement(
    blocks: Sequence[int],
    scene_ids: Sequence[str],
    seed: int,
    max_attempts: int = 100_000,
) -> np.ndarray:
    """Permute donors within blocks, forbidding identity and same-scene donors."""
    blocks = np.asarray(blocks, dtype=np.int64)
    scenes = np.asarray(scene_ids, dtype=object)
    if blocks.shape != scenes.shape:
        raise ValueError("blocks and scene_ids must have the same shape")
    rng = np.random.default_rng(seed)
    donors = np.full(len(blocks), -1, dtype=np.int64)
    for block in np.unique(blocks):
        recipients = np.flatnonzero(blocks == block)
        if len(recipients) < 2:
            raise ValueError(f"block {block} is too small for a derangement")
        accepted = None
        for _ in range(max_attempts):
            candidate = rng.permutation(recipients)
            valid = (candidate != recipients) & (scenes[candidate] != scenes[recipients])
            if bool(valid.all()):
                accepted = candidate
                break
        if accepted is None:
            raise RuntimeError(f"could not construct a scene-independent derangement for block {block}")
        donors[recipients] = accepted
    if sorted(donors.tolist()) != list(range(len(donors))):
        raise AssertionError("donor mapping does not preserve the exact marginal")
    return donors


def scene_bootstrap_delta(
    correct: np.ndarray,
    control: np.ndarray,
    scene_ids: Sequence[str],
    reps: int = 10_000,
    seed: int = 0,
) -> Dict[str, float]:
    """Paired scene bootstrap over full-pool per-sample metric contributions."""
    correct = np.asarray(correct, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    scenes = np.asarray(scene_ids, dtype=object)
    if correct.shape != control.shape or correct.shape != scenes.shape:
        raise ValueError("bootstrap arrays must have identical one-dimensional shapes")
    unique = np.asarray(sorted(set(scenes.tolist())), dtype=object)
    by_scene = {scene: np.flatnonzero(scenes == scene) for scene in unique}
    scene_delta = np.asarray([(correct[by_scene[s]] - control[by_scene[s]]).mean() for s in unique])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(reps, len(unique)))
    values = scene_delta[draws].mean(axis=1)
    return {
        "estimate": float((correct - control).mean()),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "bootstrap_p_le_zero": float((np.count_nonzero(values <= 0.0) + 1) / (reps + 1)),
        "reps": int(reps),
        "n_scenes": int(len(unique)),
    }


def validate_derangement(
    donors: Iterable[int], blocks: Sequence[int], scene_ids: Sequence[str]
) -> None:
    donors = np.asarray(list(donors), dtype=np.int64)
    blocks = np.asarray(blocks, dtype=np.int64)
    scenes = np.asarray(scene_ids, dtype=object)
    expected = np.arange(len(donors))
    if sorted(donors.tolist()) != expected.tolist():
        raise AssertionError("mapping is not a permutation")
    if np.any(donors == expected):
        raise AssertionError("mapping has fixed points")
    if np.any(blocks[donors] != blocks):
        raise AssertionError("mapping crosses a frequency block")
    if np.any(scenes[donors] == scenes):
        raise AssertionError("mapping uses same-scene donors")
