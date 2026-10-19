#!/usr/bin/env python3
"""Run a minimal, non-canonical SE(3) latent-structure analysis.

The runner deliberately consumes frozen C-series evaluation dumps instead of
extracting features. It trains small predictors on a scene-disjoint sub-split
using only one-step target loss, then evaluates group structure out of sample.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from analyses.se3_latent_mvp.geometry import (
    camera_to_world_to_extrinsic,
    encode_relative_pose_9d,
    homogeneous,
    relative_pose,
    rotation_angle,
    se3_exp,
    se3_log,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINES = ("copy", "without_condition", "action_only", "wrong_pose", "full")


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def to_float(value: Any) -> float:
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    return float(value)


def percentile_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "rms": float(np.sqrt(np.mean(values * values))),
    }


@dataclass
class TupleRecord:
    sample_id: str
    scene_id: str
    window_id: int
    raw_frame_indices: tuple[int, int, int]
    features: torch.Tensor  # three isolated target features, (3,D)
    extrinsics: np.ndarray  # normalized world->camera matrices, (3,3,4)

    @property
    def transforms(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        t_ab = relative_pose(self.extrinsics[0], self.extrinsics[1])
        t_bc = relative_pose(self.extrinsics[1], self.extrinsics[2])
        t_ac = relative_pose(self.extrinsics[0], self.extrinsics[2])
        return t_ab, t_bc, t_ac


class LatentPredictor(nn.Module):
    """Capacity-controlled residual predictor with matched ablation variants."""

    def __init__(self, feature_dim: int, hidden_dim: int, conditioning: str) -> None:
        super().__init__()
        if conditioning not in {"full", "without_condition", "action_only"}:
            raise ValueError(conditioning)
        self.conditioning = conditioning
        self.constant_state = nn.Parameter(torch.zeros(1, feature_dim))
        self.constant_action = nn.Parameter(torch.zeros(1, 6))
        self.constant_output = nn.Parameter(torch.zeros(1, feature_dim))
        nn.init.normal_(self.constant_state, std=0.01)
        nn.init.normal_(self.constant_action, std=0.01)
        nn.init.normal_(self.constant_output, std=0.01)
        self.state_branch = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
        )
        self.action_branch = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        batch = state.shape[0]
        state_input = (
            self.constant_state.expand(batch, -1)
            if self.conditioning == "action_only"
            else state
        )
        action_input = (
            self.constant_action.expand(batch, -1)
            if self.conditioning == "without_condition"
            else action
        )
        hidden = torch.cat(
            (self.state_branch(state_input), self.action_branch(action_input)), dim=-1
        )
        update = self.update(hidden)
        base = (
            self.constant_output.expand(batch, -1)
            if self.conditioning == "action_only"
            else state
        )
        return base + update


def load_manifest(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validation: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    scene_ids: set[str] = set()
    sample_ids: set[str] = set()
    duplicate_ids: list[str] = []
    total = 0
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            total += 1
            split = str(record["split"])
            split_counts[split] = split_counts.get(split, 0) + 1
            scene_ids.add(str(record["scene_id"]))
            sample_id = str(record["sample_id"])
            if sample_id in sample_ids:
                duplicate_ids.append(sample_id)
            sample_ids.add(sample_id)
            if split == "val":
                validation.append(record)
    return validation, {
        "num_samples": total,
        "num_scenes": len(scene_ids),
        "split_counts": split_counts,
        "num_duplicate_sample_ids": len(duplicate_ids),
    }


def rotation_checks(extrinsic: np.ndarray) -> tuple[float, float]:
    rotation = homogeneous(extrinsic)[:3, :3]
    orthogonal = float(np.max(np.abs(rotation @ rotation.T - np.eye(3))))
    determinant = float(abs(np.linalg.det(rotation) - 1.0))
    return orthogonal, determinant


def load_and_audit_tuples(cfg: dict[str, Any]) -> tuple[list[TupleRecord], dict[str, Any]]:
    manifest_path = resolve(cfg["manifest"])
    c1_path = resolve(cfg["c1_dump"])
    c2_path = resolve(cfg["c2_dump"])
    validation_manifest, coverage = load_manifest(manifest_path)
    c1 = torch.load(c1_path, map_location="cpu", weights_only=False)
    c2 = torch.load(c2_path, map_location="cpu", weights_only=False)
    if not (len(validation_manifest) == len(c1) == len(c2)):
        raise RuntimeError(
            f"manifest/dump count mismatch: {len(validation_manifest)}, {len(c1)}, {len(c2)}"
        )

    target_diffs: list[float] = []
    action_diffs: list[float] = []
    composition_diffs: list[float] = []
    orthogonal_diffs: list[float] = []
    determinant_diffs: list[float] = []
    raw_composition_diffs: list[float] = []
    normalized_raw_rotation_diffs: list[float] = []
    tuple_records: list[TupleRecord] = []
    raw_root = resolve(cfg["raw_scene_root"])

    for index, (c1_record, c2_record, manifest_record) in enumerate(
        zip(c1, c2, validation_manifest)
    ):
        scene_id = str(manifest_record["scene_id"])
        if not (
            str(c1_record["scene_path"]).rstrip("/").endswith(scene_id)
            and str(c2_record["scene_path"]).rstrip("/").endswith(scene_id)
        ):
            raise RuntimeError(f"scene alignment failed at row {index}")
        if not bool(c1_record["valid"]) or not bool(c2_record["valid"].all()):
            raise RuntimeError(f"invalid feature row at {index}")
        window_indices = [int(v) for v in manifest_record["streaming_window_indices"]]
        tail = int(manifest_record["streaming_tail"])
        requested_positions = (tail + 1, tail + 2, tail + 4)
        if requested_positions[-1] >= len(window_indices):
            raise RuntimeError(f"tuple horizon outside window at row {index}")
        raw_indices = tuple(window_indices[position] for position in requested_positions)
        if int(c1_record["target_frame_idx"]) != raw_indices[0]:
            raise RuntimeError(f"target-frame alignment failed at row {index}")

        target_diff = float(
            (c1_record["target"].float() - c2_record["target"][0].float()).abs().max()
        )
        target_diffs.append(target_diff)
        extrinsics = c2_record["target_extrinsics"].double().numpy()
        start = c2_record["start_extrinsic"].double().numpy()
        t_start_a = relative_pose(start, extrinsics[0])
        action_diff = float(
            np.max(
                np.abs(
                    c1_record["action"].double().numpy()
                    - encode_relative_pose_9d(t_start_a)
                )
            )
        )
        action_diffs.append(action_diff)
        t_ab = relative_pose(extrinsics[0], extrinsics[1])
        t_bc = relative_pose(extrinsics[1], extrinsics[2])
        t_ac = relative_pose(extrinsics[0], extrinsics[2])
        composition_diffs.append(float(np.max(np.abs(t_bc @ t_ab - t_ac))))
        for extrinsic in (start, *extrinsics):
            orthogonal, determinant = rotation_checks(extrinsic)
            orthogonal_diffs.append(orthogonal)
            determinant_diffs.append(determinant)

        metadata_path = raw_root / scene_id / "scene_iphone_metadata.npz"
        if not metadata_path.exists():
            raise RuntimeError(f"missing raw pose metadata: {metadata_path}")
        metadata = np.load(metadata_path)
        raw_c2w = metadata["trajectories"][list(raw_indices)]
        raw_extrinsics = np.stack(
            [camera_to_world_to_extrinsic(value) for value in raw_c2w]
        )
        raw_ab = relative_pose(raw_extrinsics[0], raw_extrinsics[1])
        raw_bc = relative_pose(raw_extrinsics[1], raw_extrinsics[2])
        raw_ac = relative_pose(raw_extrinsics[0], raw_extrinsics[2])
        raw_composition_diffs.append(float(np.max(np.abs(raw_bc @ raw_ab - raw_ac))))
        normalized_raw_rotation_diffs.extend(
            [
                float(np.max(np.abs(t_ab[:3, :3] - raw_ab[:3, :3]))),
                float(np.max(np.abs(t_bc[:3, :3] - raw_bc[:3, :3]))),
                float(np.max(np.abs(t_ac[:3, :3] - raw_ac[:3, :3]))),
            ]
        )
        tuple_records.append(
            TupleRecord(
                sample_id=str(manifest_record["sample_id"]),
                scene_id=scene_id,
                window_id=int(manifest_record.get("streaming_window_id", -1)),
                raw_frame_indices=raw_indices,
                features=c2_record["target"].float().clone(),
                extrinsics=extrinsics,
            )
        )

    gates = {
        "target_alignment_max_abs": max(target_diffs),
        "action_alignment_max_abs": max(action_diffs),
        "normalized_pose_composition_max_abs": max(composition_diffs),
        "raw_pose_composition_max_abs": max(raw_composition_diffs),
        "rotation_orthogonality_max_abs": max(orthogonal_diffs),
        "rotation_determinant_max_abs": max(determinant_diffs),
        "normalized_vs_raw_rotation_max_abs": max(normalized_raw_rotation_diffs),
        "all_c1_rows_valid": True,
        "all_c2_horizons_valid": True,
        "tuple_count": len(tuple_records),
        "tuple_scene_count": len({record.scene_id for record in tuple_records}),
    }
    if gates["target_alignment_max_abs"] > float(cfg["alignment_atol"]):
        raise RuntimeError(f"feature alignment gate failed: {gates}")
    pose_atol = float(cfg["pose_atol"])
    pose_keys = (
        "action_alignment_max_abs",
        "normalized_pose_composition_max_abs",
        "raw_pose_composition_max_abs",
        "rotation_orthogonality_max_abs",
        "rotation_determinant_max_abs",
        "normalized_vs_raw_rotation_max_abs",
    )
    if any(gates[key] > pose_atol for key in pose_keys):
        raise RuntimeError(f"pose convention gate failed: {gates}")
    if coverage["num_duplicate_sample_ids"]:
        raise RuntimeError(f"duplicate sample IDs in manifest: {coverage}")
    return tuple_records, {"coverage": coverage, "gates": gates}


def audit_pose_support(records: list[TupleRecord], raw_scene_root: Path) -> dict[str, Any]:
    motions: list[np.ndarray] = []
    scene_cache: dict[str, np.ndarray] = {}
    for record in records:
        if record.scene_id not in scene_cache:
            path = raw_scene_root / record.scene_id / "scene_iphone_metadata.npz"
            scene_cache[record.scene_id] = np.load(path)["trajectories"]
        poses = scene_cache[record.scene_id][list(record.raw_frame_indices)]
        for index_from, index_to in ((0, 1), (1, 2)):
            pose_from, pose_to = poses[index_from], poses[index_to]
            rotation_world = pose_to[:3, :3] @ pose_from[:3, :3].T
            omega_world = se3_log(
                np.block(
                    [
                        [rotation_world, np.zeros((3, 1))],
                        [np.zeros((1, 3)), np.ones((1, 1))],
                    ]
                )
            )[:3]
            translation_world = pose_to[:3, 3] - pose_from[:3, 3]
            motions.append(np.concatenate((omega_world, translation_world)))
    values = np.asarray(motions, dtype=np.float64)
    display = values.copy()
    display[:, :3] = np.degrees(display[:, :3])
    names = (
        "rotation_world_x_deg",
        "rotation_world_y_deg",
        "rotation_world_z_deg",
        "translation_world_x_m",
        "translation_world_y_m",
        "translation_world_z_m",
    )
    summaries = {name: percentile_summary(display[:, i]) for i, name in enumerate(names)}
    rotation_rms = np.sqrt(np.mean(values[:, :3] ** 2, axis=0))
    translation_rms = np.sqrt(np.mean(values[:, 3:] ** 2, axis=0))
    rms_scale = np.maximum(np.sqrt(np.mean(values * values, axis=0)), 1e-12)
    standardized = values / rms_scale
    covariance_eigenvalues = np.linalg.eigvalsh(np.cov(standardized.T))[::-1]
    rotation_nontrivial = np.mean(np.abs(display[:, :3]) > 1.0, axis=0)
    translation_nontrivial = np.mean(np.abs(display[:, 3:]) > 0.01, axis=0)
    rotation_ratio = rotation_rms / max(float(rotation_rms.max()), 1e-12)
    translation_ratio = translation_rms / max(float(translation_rms.max()), 1e-12)
    full_se3 = bool(
        rotation_ratio.min() >= 0.15
        and translation_ratio.min() >= 0.15
        and rotation_nontrivial.min() >= 0.05
        and translation_nontrivial.min() >= 0.05
        and covariance_eigenvalues.min() >= 0.15
    )
    return {
        "classification": "empirical_SE3_6DoF" if full_se3 else "SE2_or_degenerate_subset",
        "interpretation": (
            "All three world-axis rotation and translation components have substantial "
            "coverage; this is not an SE(2)-only trajectory set. Coverage is empirical "
            "and non-uniform, not a claim of uniform sampling over SE(3)."
            if full_se3
            else "At least one rotation/translation direction lacks substantive support; "
            "results must be described as a subgroup or degenerate-motion analysis."
        ),
        "num_relative_motions": int(len(values)),
        "axis_summaries": summaries,
        "rotation_rms_ratio_to_dominant": rotation_ratio.tolist(),
        "translation_rms_ratio_to_dominant": translation_ratio.tolist(),
        "rotation_fraction_abs_gt_1deg": rotation_nontrivial.tolist(),
        "translation_fraction_abs_gt_1cm": translation_nontrivial.tolist(),
        "standardized_covariance_eigenvalues": covariance_eigenvalues.tolist(),
    }


def split_scenes(
    records: list[TupleRecord], seed: int, train_fraction: float, val_fraction: float
) -> tuple[dict[str, str], dict[str, Any]]:
    scenes = np.asarray(sorted({record.scene_id for record in records}), dtype=object)
    rng = np.random.default_rng(seed)
    scenes = scenes[rng.permutation(len(scenes))]
    n_train = int(math.floor(len(scenes) * train_fraction))
    n_val = int(math.floor(len(scenes) * val_fraction))
    if min(n_train, n_val, len(scenes) - n_train - n_val) <= 0:
        raise RuntimeError("scene split would produce an empty partition")
    assignments: dict[str, str] = {}
    for scene in scenes[:n_train]:
        assignments[str(scene)] = "train"
    for scene in scenes[n_train : n_train + n_val]:
        assignments[str(scene)] = "val"
    for scene in scenes[n_train + n_val :]:
        assignments[str(scene)] = "test"
    counts = {
        split: {
            "scenes": sum(value == split for value in assignments.values()),
            "tuples": sum(assignments[record.scene_id] == split for record in records),
        }
        for split in ("train", "val", "test")
    }
    overlap = {
        "train_val": bool(
            {s for s, v in assignments.items() if v == "train"}
            & {s for s, v in assignments.items() if v == "val"}
        ),
        "train_test": bool(
            {s for s, v in assignments.items() if v == "train"}
            & {s for s, v in assignments.items() if v == "test"}
        ),
        "val_test": bool(
            {s for s, v in assignments.items() if v == "val"}
            & {s for s, v in assignments.items() if v == "test"}
        ),
    }
    if any(overlap.values()):
        raise RuntimeError(f"scene leakage detected: {overlap}")
    return assignments, {"counts": counts, "scene_overlap": overlap}


def records_for_split(
    records: Iterable[TupleRecord], assignments: dict[str, str], split: str
) -> list[TupleRecord]:
    return [record for record in records if assignments[record.scene_id] == split]


def pair_tensors(records: list[TupleRecord]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    states: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    actions: list[np.ndarray] = []
    for record in records:
        t_ab, t_bc, _ = record.transforms
        states.extend((record.features[0], record.features[1]))
        targets.extend((record.features[1], record.features[2]))
        actions.extend((se3_log(t_ab), se3_log(t_bc)))
    return torch.stack(states), torch.from_numpy(np.stack(actions)).float(), torch.stack(targets)


def objective(prediction: torch.Tensor, target: torch.Tensor, cosine_weight: float) -> torch.Tensor:
    mse = F.mse_loss(prediction, target)
    cosine = (1.0 - F.cosine_similarity(prediction, target, dim=-1)).mean()
    return mse + cosine_weight * cosine


def train_model(
    conditioning: str,
    train_records: list[TupleRecord],
    val_records: list[TupleRecord],
    cfg: dict[str, Any],
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
) -> tuple[LatentPredictor, list[dict[str, float]]]:
    train_state, train_action, train_target = pair_tensors(train_records)
    val_state, val_action, val_target = pair_tensors(val_records)
    train_action = (train_action - action_mean) / action_std
    val_action = (val_action - action_mean) / action_std
    feature_dim = int(train_state.shape[-1])
    model = LatentPredictor(feature_dim, int(cfg["hidden_dim"]), conditioning)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    generator = torch.Generator().manual_seed(int(cfg["seed"]) + len(conditioning))
    batch_size = int(cfg["batch_size"])
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(int(cfg["max_epochs"])):
        model.train()
        permutation = torch.randperm(len(train_state), generator=generator)
        train_losses: list[float] = []
        for start in range(0, len(permutation), batch_size):
            index = permutation[start : start + batch_size]
            prediction = model(train_state[index], train_action[index])
            loss = objective(prediction, train_target[index], float(cfg["cosine_loss_weight"]))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            val_prediction = model(val_state, val_action)
            val_loss = float(
                objective(val_prediction, val_target, float(cfg["cosine_loss_weight"]))
            )
        history.append(
            {"epoch": float(epoch), "train_loss": float(np.mean(train_losses)), "val_loss": val_loss}
        )
        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= int(cfg["patience"]):
            break
    if best_state is None:
        raise RuntimeError(f"training failed for {conditioning}")
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def cosine_distance(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Numerical round-off can make cosine similarity a few ULPs larger than
    # one for identical vectors. Distances are non-negative by definition.
    return (1.0 - F.cosine_similarity(prediction.float(), target.float(), dim=-1)).clamp_min(0.0)


def normalized_rmse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    error = torch.sqrt(torch.mean((prediction.float() - target.float()) ** 2, dim=-1))
    scale = torch.sqrt(torch.mean(target.float() ** 2, dim=-1)).clamp_min(1e-8)
    return error / scale


def transform_batch(records: list[TupleRecord]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    triples = [record.transforms for record in records]
    return tuple(np.stack([triple[index] for triple in triples]) for index in range(3))  # type: ignore[return-value]


def predict(
    baseline: str,
    learned_models: dict[str, LatentPredictor],
    state: torch.Tensor,
    transforms: np.ndarray,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
) -> torch.Tensor:
    if baseline == "copy":
        return state.clone()
    model_name = "full" if baseline == "wrong_pose" else baseline
    model = learned_models[model_name]
    actions = torch.from_numpy(np.stack([se3_log(value) for value in transforms])).float()
    actions = (actions - action_mean) / action_std
    with torch.no_grad():
        return model(state, actions)


def matched_donor_mapping(records: list[TupleRecord]) -> np.ndarray:
    donors = np.full(len(records), -1, dtype=np.int64)
    by_window: dict[int, list[int]] = {}
    for index, record in enumerate(records):
        by_window.setdefault(record.window_id, []).append(index)
    for indices in by_window.values():
        ordered = sorted(indices, key=lambda i: records[i].scene_id)
        scenes = [records[i].scene_id for i in ordered]
        if len(set(scenes)) < 2:
            raise RuntimeError("matched shuffle block has fewer than two scenes")
        shifted = ordered[1:] + ordered[:1]
        for recipient, donor in zip(ordered, shifted):
            if records[recipient].scene_id == records[donor].scene_id:
                raise RuntimeError("matched shuffle selected a same-scene donor")
            donors[recipient] = donor
    if sorted(donors.tolist()) != list(range(len(records))):
        raise RuntimeError("matched donor mapping is not an exact permutation")
    return donors


def metric_mean(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    return {
        "cosine": float(cosine_distance(prediction, target).mean()),
        "nrmse": float(normalized_rmse(prediction, target).mean()),
    }


def evaluate_baseline(
    baseline: str,
    learned_models: dict[str, LatentPredictor],
    records: list[TupleRecord],
    donor_mapping: np.ndarray,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
) -> tuple[dict[str, float], dict[str, Any]]:
    features = torch.stack([record.features for record in records])
    true_ab, true_bc, true_ac = transform_batch(records)
    if baseline == "wrong_pose":
        action_ab = true_ab[donor_mapping]
        action_bc = true_bc[donor_mapping]
        action_ac = np.stack([b @ a for a, b in zip(action_ab, action_bc)])
    else:
        action_ab, action_bc, action_ac = true_ab, true_bc, true_ac

    # One-step grounding uses only the supervised edge types.
    one_ab = predict(baseline, learned_models, features[:, 0], action_ab, action_mean, action_std)
    one_bc = predict(baseline, learned_models, features[:, 1], action_bc, action_mean, action_std)
    one_prediction = torch.cat((one_ab, one_bc), dim=0)
    one_target = torch.cat((features[:, 1], features[:, 2]), dim=0)
    one = metric_mean(one_prediction, one_target)

    recursive = predict(baseline, learned_models, one_ab, action_bc, action_mean, action_std)
    direct = predict(baseline, learned_models, features[:, 0], action_ac, action_mean, action_std)
    composition_structure = metric_mean(recursive, direct)
    recursive_endpoint = metric_mean(recursive, features[:, 2])
    direct_endpoint = metric_mean(direct, features[:, 2])

    inverse_ab = np.stack([np.linalg.inv(value) for value in action_ab])
    inverse_bc = np.stack([np.linalg.inv(value) for value in action_bc])
    cycle_a = predict(baseline, learned_models, one_ab, inverse_ab, action_mean, action_std)
    cycle_b_forward = one_bc
    cycle_b = predict(
        baseline, learned_models, cycle_b_forward, inverse_bc, action_mean, action_std
    )
    cycle_prediction = torch.cat((cycle_a, cycle_b), dim=0)
    cycle_target = torch.cat((features[:, 0], features[:, 1]), dim=0)
    cycle = metric_mean(cycle_prediction, cycle_target)

    swapped_first = predict(
        baseline, learned_models, features[:, 0], action_bc, action_mean, action_std
    )
    swapped = predict(
        baseline, learned_models, swapped_first, action_ab, action_mean, action_std
    )
    original_net = np.stack([b @ a for a, b in zip(action_ab, action_bc)])
    swapped_net = np.stack([a @ b for a, b in zip(action_ab, action_bc)])
    commutator = np.asarray(
        [np.linalg.norm(original - swap, ord="fro") for original, swap in zip(original_net, swapped_net)]
    )
    rotation_translation = np.asarray(
        [
            rotation_angle(a) > math.radians(1.0)
            and rotation_angle(b) > math.radians(1.0)
            and np.linalg.norm(a[:3, 3]) > 0.01
            and np.linalg.norm(b[:3, 3]) > 0.01
            for a, b in zip(action_ab, action_bc)
        ],
        dtype=bool,
    )
    positive = commutator[rotation_translation]
    threshold = float(np.quantile(positive, 0.25)) if len(positive) else float("inf")
    selected = rotation_translation & (commutator >= threshold)
    if not bool(selected.any()):
        raise RuntimeError("no non-commutative rotation-translation pairs available")
    order_distance = cosine_distance(recursive[selected], swapped[selected])
    true_change = cosine_distance(features[selected, 0], features[selected, 2]).clamp_min(1e-8)
    order_ratio = order_distance / true_change
    order_original_endpoint = metric_mean(recursive[selected], features[selected, 2])

    # Four-step real-endpoint rollout: A -> B -> C -> B -> A.
    rollout_state = features[:, 0]
    rollout_targets = (
        features[:, 1],
        features[:, 2],
        features[:, 1],
        features[:, 0],
    )
    rollout_actions = (action_ab, action_bc, inverse_bc, inverse_ab)
    rollout_cosine: list[float] = []
    rollout_nrmse: list[float] = []
    for action, target in zip(rollout_actions, rollout_targets):
        rollout_state = predict(
            baseline, learned_models, rollout_state, action, action_mean, action_std
        )
        values = metric_mean(rollout_state, target)
        rollout_cosine.append(values["cosine"])
        rollout_nrmse.append(values["nrmse"])

    summary = {
        "baseline": baseline,
        "one_step_cosine": one["cosine"],
        "one_step_nrmse": one["nrmse"],
        "composition_structure_cosine": composition_structure["cosine"],
        "composition_structure_nrmse": composition_structure["nrmse"],
        "composition_recursive_endpoint_cosine": recursive_endpoint["cosine"],
        "composition_recursive_endpoint_nrmse": recursive_endpoint["nrmse"],
        "composition_direct_endpoint_cosine": direct_endpoint["cosine"],
        "composition_direct_endpoint_nrmse": direct_endpoint["nrmse"],
        "cycle_cosine": cycle["cosine"],
        "cycle_nrmse": cycle["nrmse"],
        "order_swap_separation_cosine": float(order_distance.mean()),
        "order_swap_separation_over_true_change": float(order_ratio.mean()),
        "order_swap_collapse_rate_ratio_lt_0p1": float((order_ratio < 0.1).float().mean()),
        "order_original_endpoint_cosine": order_original_endpoint["cosine"],
        "order_selected_pairs": int(selected.sum()),
    }
    for step, (cosine, nrmse) in enumerate(zip(rollout_cosine, rollout_nrmse), 1):
        summary[f"rollout_step{step}_cosine"] = cosine
        summary[f"rollout_step{step}_nrmse"] = nrmse
    details = {
        "order_commutator_frobenius_threshold": threshold,
        "order_commutator_frobenius": percentile_summary(commutator[selected]),
        "rollout_path": "A->B->C->B->A (forward two observed transforms, then exact inverses)",
    }
    return summary, details


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(output: Path, rows: list[dict[str, Any]]) -> None:
    colors = {
        "copy": "#777777",
        "without_condition": "#E69F00",
        "action_only": "#CC79A7",
        "wrong_pose": "#D55E00",
        "full": "#0072B2",
    }
    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    for row in rows:
        name = str(row["baseline"])
        axis.scatter(
            row["one_step_cosine"],
            row["composition_structure_cosine"],
            color=colors[name],
            s=70,
        )
        axis.annotate(name, (row["one_step_cosine"], row["composition_structure_cosine"]), xytext=(5, 4), textcoords="offset points")
    axis.set_xlabel("One-step endpoint cosine distance (lower is correct)")
    axis.set_ylabel("Composition structural cosine distance (lower is consistent)")
    axis.set_title("Correctness vs. consistency — DINO L0, prefix 8")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "correctness_consistency.png", dpi=180)
    figure.savefig(output / "correctness_consistency.pdf")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    steps = np.arange(1, 5)
    for row in rows:
        name = str(row["baseline"])
        values = [row[f"rollout_step{step}_cosine"] for step in steps]
        axis.plot(steps, values, marker="o", label=name, color=colors[name])
    axis.set_xticks(steps)
    axis.set_xlabel("Rollout step")
    axis.set_ylabel("Real endpoint cosine distance")
    axis.set_title("Retrace rollout A→B→C→B→A")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "rollout_drift.png", dpi=180)
    figure.savefig(output / "rollout_drift.pdf")
    plt.close(figure)


def format_table(rows: list[dict[str, Any]]) -> str:
    columns = (
        "baseline",
        "one_step_cosine",
        "composition_structure_cosine",
        "composition_recursive_endpoint_cosine",
        "composition_direct_endpoint_cosine",
        "cycle_cosine",
        "order_swap_separation_cosine",
        "rollout_step4_cosine",
    )
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        values = [str(row["baseline"])] + [f"{float(row[column]):.6f}" for column in columns[1:]]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def render_report(
    output: Path,
    cfg: dict[str, Any],
    audit: dict[str, Any],
    rows: list[dict[str, Any]],
    sanity: dict[str, Any],
    split_audit: dict[str, Any],
) -> None:
    pose = audit["pose_support"]
    gates = audit["tuple_audit"]["gates"]
    report = f"""# SE(3) latent-structure MVP report

Status: **experimental / non-canonical**. This run does not modify the registry,
canonical tables, checkpoints, or feature artifacts.

## Data and coordinate audit

- Selected artifact: `{cfg['model']}`, layer `{cfg['feature_layer']}`, prefix `{cfg['prefix']}`.
- Source: ScanNet++ only; 69 canonical-validation scenes and 276 complete tuples.
- Raw `trajectories` are camera-to-world matrices. The repository converts them
  to world-to-camera extrinsics `E`; `T_AB = E_B E_A^-1` maps camera-A
  coordinates into camera-B coordinates. Therefore `T_AC = T_BC T_AB`.
- Maximum normalized composition residual: `{gates['normalized_pose_composition_max_abs']:.3e}`.
- Maximum raw-pose composition residual: `{gates['raw_pose_composition_max_abs']:.3e}`.
- C1/C2 target alignment maximum absolute difference: `{gates['target_alignment_max_abs']:.3e}`.
- C1 action re-encoding maximum absolute difference: `{gates['action_alignment_max_abs']:.3e}`.
- Pose-support conclusion: **{pose['classification']}**. {pose['interpretation']}
- Static-scene caveat: ScanNet++ capture scenes are treated as static indoor
  reconstructions, but these artifacts have no per-frame dynamic-object/static
  certification. This remains a documented risk rather than a verified label.

## Experimental split and training contract

The 69 canonical-validation scenes were re-split by scene solely for this MVP:
`{split_audit['counts']}`. Scene overlap is `{split_audit['scene_overlap']}`.
Each tuple supplies three isolated target features `(A,B,C)`. Training uses only
the one-step pairs `A->B` and `B->C` with MSE plus cosine target loss. No
composition, cycle, order, or rollout loss is used.

## Results

All distance columns are lower-is-better. Order-swap separation is deliberately
higher-is-more-sensitive and must be read together with its endpoint error.

{format_table(rows)}

The copy predictor illustrates the degeneracy gate: it can have essentially
zero composition/cycle error while its real endpoint grounding remains worse.
See `minimal_results_table.csv` for cosine and normalized-RMSE variants and all
rollout steps.

## Rollout definition

The four-step curve uses real isolated-feature endpoints along
`A -> B -> C -> B -> A`: two observed forward transforms followed by their
exact inverses. This supplies real endpoint error at every step without
inventing an unavailable fourth future feature.

## Sanity checks

```json
{json.dumps(sanity, indent=2, sort_keys=True)}
```

## Reproduction

```bash
PY=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python
MPLCONFIGDIR=/tmp/probe_spatial_mpl "$PY" -m pytest -q tests/test_se3_latent_mvp.py
MPLCONFIGDIR=/tmp/probe_spatial_mpl "$PY" -m analyses.se3_latent_mvp.run_mvp \\
  --config analyses/se3_latent_mvp/config.json \\
  --output-dir results/experimental/se3_latent_mvp_reproduction_seed42
```

## Minimal sweep extension

No sweep is launched here. To extend, provide another frozen C1/C2 dump pair
and matching manifest through the same config fields, then change only
`model`, `feature_layer`, `prefix`, and the input paths. The alignment gates
run before training, so an incomplete layer/timestep/prefix cell stops rather
than silently shrinking the sample set.
"""
    (output / "REPORT.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="analyses/se3_latent_mvp/config.json")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    config_path = resolve(args.config)
    cfg = json.loads(config_path.read_text())
    output = resolve(args.output_dir or cfg["output_dir"])
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite experimental output {output}; pass a new --output-dir"
        )
    output.mkdir(parents=True)
    (output / "checkpoints").mkdir()
    (output / "training_history").mkdir()

    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(cfg["torch_num_threads"]))
    torch.use_deterministic_algorithms(True)

    records, tuple_audit = load_and_audit_tuples(cfg)
    pose_support = audit_pose_support(records, resolve(cfg["raw_scene_root"]))
    assignments, split_audit = split_scenes(
        records,
        seed,
        float(cfg["train_scene_fraction"]),
        float(cfg["val_scene_fraction"]),
    )
    train_records = records_for_split(records, assignments, "train")
    val_records = records_for_split(records, assignments, "val")
    test_records = records_for_split(records, assignments, "test")

    _, train_actions, _ = pair_tensors(train_records)
    action_mean = train_actions.mean(dim=0)
    action_std = train_actions.std(dim=0).clamp_min(1e-5)
    learned_models: dict[str, LatentPredictor] = {}
    training_meta: dict[str, Any] = {}
    for conditioning in ("without_condition", "action_only", "full"):
        torch.manual_seed(seed)
        model, history = train_model(
            conditioning,
            train_records,
            val_records,
            cfg,
            action_mean,
            action_std,
        )
        learned_models[conditioning] = model
        checkpoint = output / "checkpoints" / f"{conditioning}.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "conditioning": conditioning,
                "feature_dim": int(train_records[0].features.shape[-1]),
                "hidden_dim": int(cfg["hidden_dim"]),
                "action_mean": action_mean,
                "action_std": action_std,
                "seed": seed,
                "training_objective": "one_step_mse_plus_cosine_only",
            },
            checkpoint,
        )
        write_csv(output / "training_history" / f"{conditioning}.csv", history)
        training_meta[conditioning] = {
            "epochs_run": len(history),
            "best_val_loss": min(row["val_loss"] for row in history),
            "checkpoint": str(checkpoint.relative_to(REPO_ROOT)),
        }

    donor_mapping = matched_donor_mapping(test_records)
    true_ab, true_bc, _ = transform_batch(test_records)
    shuffled_ab = true_ab[donor_mapping]
    shuffled_bc = true_bc[donor_mapping]
    matched_shuffle_audit = {
        "exact_permutation": sorted(donor_mapping.tolist()) == list(range(len(test_records))),
        "fixed_points": int(np.sum(donor_mapping == np.arange(len(test_records)))),
        "same_scene_donors": int(
            sum(
                test_records[i].scene_id == test_records[j].scene_id
                for i, j in enumerate(donor_mapping)
            )
        ),
        "mean_abs_twist_norm_delta_step1": float(
            np.mean(
                np.abs(
                    np.linalg.norm(np.stack([se3_log(t) for t in true_ab]), axis=1)
                    - np.linalg.norm(np.stack([se3_log(t) for t in shuffled_ab]), axis=1)
                )
            )
        ),
        "mean_abs_twist_norm_delta_step2": float(
            np.mean(
                np.abs(
                    np.linalg.norm(np.stack([se3_log(t) for t in true_bc]), axis=1)
                    - np.linalg.norm(np.stack([se3_log(t) for t in shuffled_bc]), axis=1)
                )
            )
        ),
    }
    if matched_shuffle_audit["fixed_points"] or matched_shuffle_audit["same_scene_donors"]:
        raise RuntimeError(f"wrong-pose mapping gate failed: {matched_shuffle_audit}")

    rows: list[dict[str, Any]] = []
    evaluation_details: dict[str, Any] = {}
    for baseline in BASELINES:
        summary, details = evaluate_baseline(
            baseline,
            learned_models,
            test_records,
            donor_mapping,
            action_mean,
            action_std,
        )
        rows.append(summary)
        evaluation_details[baseline] = details

    copy = next(row for row in rows if row["baseline"] == "copy")
    full = next(row for row in rows if row["baseline"] == "full")
    log_exp_errors = []
    for record in test_records[: min(16, len(test_records))]:
        for transform in record.transforms:
            log_exp_errors.append(float(np.max(np.abs(se3_exp(se3_log(transform)) - transform))))
    sanity = {
        "pose_convention_confirmed": True,
        "feature_manifest_alignment_passed": True,
        "all_tuple_targets_valid": True,
        "scene_disjoint_split_passed": not any(split_audit["scene_overlap"].values()),
        "se3_log_exp_roundtrip_max_abs": max(log_exp_errors),
        "copy_composition_is_near_zero": copy["composition_structure_cosine"] < 1e-6,
        "copy_cycle_is_near_zero": copy["cycle_cosine"] < 1e-6,
        "copy_grounding_is_nontrivial": copy["one_step_cosine"] > 1e-4,
        "grounding_reported_with_structure": True,
        "normal_predictor_finite": all(math.isfinite(float(v)) for k, v in full.items() if k != "baseline"),
        "static_scene_certification": False,
        "static_scene_risk": "ScanNet++ is treated as a static reconstruction source, but no per-frame static/dynamic certification is present.",
    }
    required_sanity = [key for key, value in sanity.items() if isinstance(value, bool) and key != "static_scene_certification" and not value]
    if required_sanity:
        raise RuntimeError(f"sanity checks failed: {required_sanity}")

    sample_rows = [
        {
            "sample_id": record.sample_id,
            "scene_id": record.scene_id,
            "window_id": record.window_id,
            "split": assignments[record.scene_id],
            "frame_A": record.raw_frame_indices[0],
            "frame_B": record.raw_frame_indices[1],
            "frame_C": record.raw_frame_indices[2],
        }
        for record in records
    ]
    write_csv(output / "sample_manifest.csv", sample_rows)
    write_csv(output / "minimal_results_table.csv", rows)
    json_dump(output / "metrics.json", {row["baseline"]: row for row in rows})
    json_dump(output / "evaluation_details.json", evaluation_details)
    json_dump(output / "pose_support.json", pose_support)
    json_dump(output / "tuple_audit.json", tuple_audit)
    json_dump(output / "split_audit.json", split_audit)
    json_dump(output / "matched_shuffle_audit.json", matched_shuffle_audit)
    json_dump(output / "sanity_checks.json", sanity)
    json_dump(output / "training_meta.json", training_meta)
    effective_cfg = dict(cfg)
    effective_cfg["output_dir"] = str(output.relative_to(REPO_ROOT))
    json_dump(output / "config.json", effective_cfg)

    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True, check=False
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--short"], cwd=REPO_ROOT, text=True, capture_output=True, check=False
    ).stdout.splitlines()
    input_paths = {
        "config": config_path,
        "manifest": resolve(cfg["manifest"]),
        "c1_dump": resolve(cfg["c1_dump"]),
        "c2_dump": resolve(cfg["c2_dump"]),
        "runner": Path(__file__).resolve(),
        "geometry": Path(__file__).resolve().with_name("geometry.py"),
        "unit_test": REPO_ROOT / "tests" / "test_se3_latent_mvp.py",
    }
    provenance = {
        "git_commit": git_commit,
        "git_dirty": bool(git_status),
        "git_status_short": git_status,
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "seed": seed,
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for name, path in input_paths.items()
        },
    }
    json_dump(output / "provenance.json", provenance)
    plot_results(output, rows)
    audit = {"tuple_audit": tuple_audit, "pose_support": pose_support}
    render_report(output, cfg, audit, rows, sanity, split_audit)
    print(json.dumps({"output_dir": str(output), "sanity": sanity, "metrics": rows}, indent=2))


if __name__ == "__main__":
    main()
