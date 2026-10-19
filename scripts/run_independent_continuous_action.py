#!/usr/bin/env python3
"""Train and evaluate the frozen Goal-Conditioned Continuous Action probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from run_independent_p8_final_analysis import (
    Store,
    action9_to_matrix,
    pose_action,
    read_jsonl,
    sha256,
    write_csv,
    write_json,
)

MODELS = ("dino", "f3r", "vjepa2", "wan")
SEEDS = (41, 42, 43)
LEARNED_VARIANTS = ("full", "goal_only", "current_only", "shuffled_goal")
ROTATION_WEIGHT = 0.5
EXPECTED_HASHES = {
    "algebra_fit_scene_disjoint_v1.jsonl": "e6190674b145923c512c055f799ffaf4aac52f02351cdd14389f71300d91268e",
    "algebra_dev_scene_disjoint_v1.jsonl": "a67ae524760a25a5c9d5505bf8799c835812683571309a58f35bb86cc6d718d8",
    "algebra_p8_independent_v1_val.jsonl": "947e251d4ff6c75282ba580ec0a00d4a4bbf877077a7386b92b8764a23086688",
}
EXPECTED_REQUEST_SHA = "fb1dbd2d71878edae7eae9034a16ee9809fc0c0cfb67352e0a301ad0edbed369"


class Normalizer:
    def __init__(self, values: torch.Tensor, floor: float = 1e-6) -> None:
        self.mean = values.mean(0)
        self.std = values.std(0).clamp_min(floor)

    def apply(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.std

    def invert(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.std.to(values.device) + self.mean.to(values.device)


class ContinuousActionProbe(torch.nn.Module):
    def __init__(self, feature_width: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(feature_width * 2, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, 9),
        )

    def forward(self, current_goal: torch.Tensor) -> torch.Tensor:
        return self.net(current_goal)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def torch_rotation(action: torch.Tensor) -> torch.Tensor:
    first = torch.nn.functional.normalize(action[..., :3], dim=-1, eps=1e-8)
    second_raw = action[..., 3:6] - first * (first * action[..., 3:6]).sum(-1, keepdim=True)
    second = torch.nn.functional.normalize(second_raw, dim=-1, eps=1e-8)
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-2)


def torch_rotation_error(pred_action: torch.Tensor, target_action: torch.Tensor) -> torch.Tensor:
    pred = torch_rotation(pred_action)
    target = torch_rotation(target_action)
    relative = pred.transpose(-1, -2) @ target
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cosine)


def rotation_angle(a: np.ndarray, b: np.ndarray) -> float:
    relative = a.T @ b
    cosine = (float(np.trace(relative)) - 1.0) * 0.5
    return float(math.acos(max(-1.0, min(1.0, cosine))))


def motion_distance(a: np.ndarray, b: np.ndarray) -> float:
    translation = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    rotation = rotation_angle(a[:3, :3], b[:3, :3])
    return translation + ROTATION_WEIGHT * rotation


def apply_action(anchor_c2w: np.ndarray, action: np.ndarray) -> np.ndarray:
    action_matrix = action9_to_matrix(action.astype(np.float64))
    if not np.isfinite(action_matrix).all() or np.linalg.det(action_matrix[:3, :3]) < 0.99:
        raise ValueError("predicted action did not decode to a proper SE(3) rotation")
    predicted_w2c = action_matrix @ np.linalg.inv(anchor_c2w)
    return np.linalg.inv(predicted_w2c)


def action_components(action: np.ndarray) -> tuple[np.ndarray, float, float]:
    matrix = action9_to_matrix(action.astype(np.float64))
    translation = matrix[:3, 3]
    distance = float(np.linalg.norm(translation))
    direction = translation / max(distance, 1e-12)
    rotation = rotation_angle(np.eye(3), matrix[:3, :3])
    return direction, distance, rotation


def scene_macro(values: torch.Tensor, scene_ids: list[str]) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, scene_id in zip(values.detach().cpu().tolist(), scene_ids):
        grouped[scene_id].append(float(value))
    return float(np.mean([np.mean(items) for items in grouped.values()]))


def derangement(rows: list[dict[str, Any]]) -> list[int]:
    order = sorted(range(len(rows)), key=lambda index: rows[index]["sample_id"])
    for offset in range(1, len(order)):
        mapping = [0] * len(rows)
        for position, source in enumerate(order):
            mapping[source] = order[(position + offset) % len(order)]
        if all(rows[source]["final_raw_id"] != rows[target]["final_raw_id"] for source, target in enumerate(mapping)):
            return mapping
    raise ValueError("could not construct deterministic goal derangement")


def load_tensors(rows: list[dict[str, Any]], store: Store) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    current = torch.stack([store.context(row["context_raw_ids"]) for row in rows])
    goal = torch.stack([store.vector(row["horizon_target_raw_ids"]["h4"]) for row in rows])
    action = torch.tensor([row["primitive_actions_9d"][0] for row in rows], dtype=torch.float32)
    return current, goal, action


def inputs_for_variant(
    current: torch.Tensor,
    goal: torch.Tensor,
    rows: list[dict[str, Any]],
    current_norm: Normalizer,
    goal_norm: Normalizer,
    variant: str,
) -> torch.Tensor:
    c = current_norm.apply(current)
    g = goal_norm.apply(goal)
    if variant == "goal_only":
        c = torch.zeros_like(c)
    elif variant == "current_only":
        g = torch.zeros_like(g)
    elif variant == "shuffled_goal":
        g = g[torch.tensor(derangement(rows), dtype=torch.long)]
    elif variant != "full":
        raise ValueError(variant)
    return torch.cat((c, g), dim=1)


def loss_per_sample(pred_norm: torch.Tensor, target_norm: torch.Tensor, target_normalizer: Normalizer) -> torch.Tensor:
    huber = torch.nn.functional.smooth_l1_loss(pred_norm, target_norm, reduction="none").mean(1)
    pred_action = target_normalizer.invert(pred_norm)
    target_action = target_normalizer.invert(target_norm)
    return huber + 0.1 * torch_rotation_error(pred_action, target_action)


def train_probe(
    train_x: torch.Tensor,
    train_action: torch.Tensor,
    dev_x: torch.Tensor,
    dev_action: torch.Tensor,
    dev_scenes: list[str],
    target_normalizer: Normalizer,
    seed: int,
    device: str,
    epochs: int | None = None,
) -> tuple[ContinuousActionProbe, int, list[float]]:
    seed_all(seed)
    model = ContinuousActionProbe(train_x.shape[1] // 2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(seed + 1700)
    train_target = target_normalizer.apply(train_action)
    dev_target = target_normalizer.apply(dev_action)
    best_loss, best_epoch, patience = math.inf, 1, 0
    curve: list[float] = []
    limit = epochs or 300
    for epoch in range(1, limit + 1):
        order = torch.randperm(len(train_x), generator=generator)
        model.train()
        for start in range(0, len(order), 64):
            index = order[start:start + 64]
            x = train_x[index].to(device)
            y = train_target[index].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_per_sample(model(x), y, target_normalizer).mean()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            dev_pred = model(dev_x.to(device))
            per_sample = loss_per_sample(dev_pred, dev_target.to(device), target_normalizer)
            dev_loss = scene_macro(per_sample, dev_scenes)
        curve.append(dev_loss)
        if dev_loss < best_loss - 1e-8:
            best_loss, best_epoch, patience = dev_loss, epoch, 0
        else:
            patience += 1
        if epochs is None and patience >= 30:
            break
    return model.eval(), best_epoch, curve


def metric_row(
    row: dict[str, Any], model: str, variant: str, seed: int | str,
    predicted_action: np.ndarray,
) -> dict[str, Any]:
    target = np.asarray(row["primitive_actions_9d"][0], dtype=np.float64)
    pred = np.asarray(predicted_action, dtype=np.float64)
    target_matrix = action9_to_matrix(target)
    pred_matrix = action9_to_matrix(pred)
    target_direction, target_distance, target_rotation = action_components(target)
    pred_direction, pred_distance, _ = action_components(pred)
    direction_cosine = float(np.clip(np.dot(target_direction, pred_direction), -1.0, 1.0))
    direction_error = math.degrees(math.acos(direction_cosine))
    distance_mae = abs(pred_distance - target_distance)
    distance_relative_error = distance_mae / max(target_distance, 1e-12)
    rotation_error = math.degrees(rotation_angle(pred_matrix[:3, :3], target_matrix[:3, :3]))

    anchor = np.asarray(row["anchor_pose"], dtype=np.float64)
    goal = np.asarray(row["final_pose"], dtype=np.float64)
    predicted_pose = apply_action(anchor, pred)
    initial_distance = motion_distance(anchor, goal)
    remaining_distance = motion_distance(predicted_pose, goal)
    reduction = initial_distance - remaining_distance
    translation_term = target_distance
    rotation_term = ROTATION_WEIGHT * target_rotation
    first_action_magnitude = translation_term + rotation_term
    return {
        "model": model,
        "variant": variant,
        "seed": seed,
        "sample_id": row["sample_id"],
        "scene_id": row["scene_id"],
        "direction_error_deg": direction_error,
        "distance_mae": distance_mae,
        "distance_relative_error": distance_relative_error,
        "rotation_error_deg": rotation_error,
        "goal_distance_reduction": reduction,
        "positive_progress": int(reduction > 0.0),
        "remaining_goal_distance": remaining_distance,
        "initial_goal_distance": initial_distance,
        "target_translation_distance": target_distance,
        "predicted_translation_distance": pred_distance,
        "target_rotation_angle_deg": math.degrees(target_rotation),
        "first_action_magnitude": first_action_magnitude,
        "translation_dominant": int(translation_term >= rotation_term),
        "predicted_action_9d": json.dumps([float(value) for value in pred], separators=(",", ":")),
        "target_action_9d": json.dumps([float(value) for value in target], separators=(",", ":")),
    }


def verify_rows(fit: list[dict[str, Any]], dev: list[dict[str, Any]], val: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {"fit": fit, "dev": dev, "validation": val}
    scene_sets = {name: {row["scene_id"] for row in rows} for name, rows in groups.items()}
    overlaps = {
        "fit_dev": sorted(scene_sets["fit"] & scene_sets["dev"]),
        "fit_validation": sorted(scene_sets["fit"] & scene_sets["validation"]),
        "dev_validation": sorted(scene_sets["dev"] & scene_sets["validation"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"scene split overlap: {overlaps}")
    action_recompute_atol = 1e-7
    max_action_recompute_abs_error = 0.0
    for split, rows in groups.items():
        for row in rows:
            if len(row["context_raw_ids"]) != 8 or row["context_raw_ids"][-1] != row["anchor_raw_id"]:
                raise ValueError(f"invalid p8 context: {split}/{row['sample_id']}")
            if row["horizon_target_raw_ids"]["h4"] != row["final_raw_id"]:
                raise ValueError(f"invalid h4 goal: {split}/{row['sample_id']}")
            recomputed = pose_action(row["anchor_pose"], row["waypoint_poses"][0])
            stored = np.asarray(row["primitive_actions_9d"][0])
            action_error = float(np.max(np.abs(recomputed - stored)))
            max_action_recompute_abs_error = max(max_action_recompute_abs_error, action_error)
            if not np.allclose(recomputed, stored, rtol=0.0, atol=action_recompute_atol):
                raise ValueError(f"a1 pose/action mismatch: {split}/{row['sample_id']}")
            oracle_pose = apply_action(np.asarray(row["anchor_pose"]), recomputed)
            if motion_distance(oracle_pose, np.asarray(row["waypoint_poses"][0])) > 1e-7:
                raise ValueError(f"SE(3) application mismatch: {split}/{row['sample_id']}")
    return {
        "counts": {name: len(rows) for name, rows in groups.items()},
        "scenes": {name: len(scene_sets[name]) for name in groups},
        "scene_overlap": overlaps,
        "a1_recomputed_from_pose": True,
        "a1_recompute_atol": action_recompute_atol,
        "max_a1_recompute_abs_error": max_action_recompute_abs_error,
        "se3_application_verified": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--request-manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output_target = args.output_dir.resolve()
    if output_target.exists():
        raise SystemExit(f"refusing to overwrite {output_target}")
    output_target.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{output_target.name}.incomplete-", dir=output_target.parent))

    if sha256(args.request_manifest) != EXPECTED_REQUEST_SHA:
        raise ValueError("feature request hash changed")
    for name, expected in EXPECTED_HASHES.items():
        if sha256(args.manifest_dir / name) != expected:
            raise ValueError(f"manifest hash changed: {name}")

    fit = read_jsonl(args.manifest_dir / "algebra_fit_scene_disjoint_v1.jsonl")
    dev = read_jsonl(args.manifest_dir / "algebra_dev_scene_disjoint_v1.jsonl")
    full = read_jsonl(args.source_manifest_dir / "algebra_p8_independent_v1_train.jsonl")
    val = read_jsonl(args.manifest_dir / "algebra_p8_independent_v1_val.jsonl")
    if len(fit) != 803 or len(dev) != 221 or len(full) != 1024 or len(val) != 256:
        raise ValueError("unexpected frozen split sizes")
    row_audit = verify_rows(fit, dev, val)

    store = Store(args.request_manifest, args.feature_root, args.model)
    fit_current, fit_goal, fit_action = load_tensors(fit, store)
    dev_current, dev_goal, dev_action = load_tensors(dev, store)
    full_current, full_goal, full_action = load_tensors(full, store)
    val_current, val_goal, _ = load_tensors(val, store)
    current_norm = Normalizer(fit_current)
    goal_norm = Normalizer(fit_goal)
    target_norm = Normalizer(fit_action)

    results: list[dict[str, Any]] = []
    training: dict[str, Any] = {}
    checkpoints = work / "checkpoints"
    checkpoints.mkdir()
    dev_scenes = [row["scene_id"] for row in dev]
    for variant in LEARNED_VARIANTS:
        fit_x = inputs_for_variant(fit_current, fit_goal, fit, current_norm, goal_norm, variant)
        dev_x = inputs_for_variant(dev_current, dev_goal, dev, current_norm, goal_norm, variant)
        full_x = inputs_for_variant(full_current, full_goal, full, current_norm, goal_norm, variant)
        val_x = inputs_for_variant(val_current, val_goal, val, current_norm, goal_norm, variant)
        for seed in SEEDS:
            _, selected_epoch, curve = train_probe(
                fit_x, fit_action, dev_x, dev_action, dev_scenes, target_norm, seed, args.device
            )
            model, _, _ = train_probe(
                full_x, full_action, dev_x, dev_action, dev_scenes, target_norm, seed,
                args.device, epochs=selected_epoch,
            )
            with torch.no_grad():
                prediction = target_norm.invert(model(val_x.to(args.device))).cpu().numpy()
            for row, pred in zip(val, prediction):
                results.append(metric_row(row, args.model, variant, seed, pred))
            checkpoint = checkpoints / f"{variant}_seed_{seed}.pt"
            torch.save({
                "model": args.model,
                "variant": variant,
                "seed": seed,
                "selected_epoch": selected_epoch,
                "feature_width": int(fit_current.shape[1]),
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "current_mean": current_norm.mean,
                "current_std": current_norm.std,
                "goal_mean": goal_norm.mean,
                "goal_std": goal_norm.std,
                "target_mean": target_norm.mean,
                "target_std": target_norm.std,
            }, checkpoint)
            training[f"{variant}|{seed}"] = {
                "selected_epoch": selected_epoch,
                "dev_loss_curve": curve,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "checkpoint_sha256": sha256(checkpoint),
            }

    mean_action = full_action.mean(0).numpy()
    for row in val:
        results.append(metric_row(row, args.model, "mean_motion_prior", "NA", mean_action))
        target = np.asarray(row["primitive_actions_9d"][0], dtype=np.float64)
        results.append(metric_row(row, args.model, "oracle_a1", "NA", target))

    write_csv(work / "continuous_action_per_sample.csv", results)
    integrity = {
        "status": "COMPLETE_VALID",
        "model": args.model,
        "protocol_sha256": sha256(args.protocol),
        "request_sha256": sha256(args.request_manifest),
        "manifest_sha256": {name: sha256(args.manifest_dir / name) for name in EXPECTED_HASHES},
        "row_audit": row_audit,
        "feature_width": int(fit_current.shape[1]),
        "input_contract": ["current_p8_feature", "goal_h4_isolated_feature"],
        "forbidden_model_inputs": [
            "GT_pose", "goal_pose", "action_label", "future_action_sequence",
            "raw_frame_index", "scene_ID", "trajectory_direction",
        ],
        "learned_variants": list(LEARNED_VARIANTS),
        "seeds": list(SEEDS),
        "validation_records": len(val),
        "per_sample_rows": len(results),
        "training": training,
        "code_sha256": sha256(Path(__file__)),
    }
    write_json(work / "training_and_integrity.json", integrity)
    os.rename(work, output_target)
    print(json.dumps({"status": "COMPLETE_VALID", "model": args.model, "rows": len(results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
