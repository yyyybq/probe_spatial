#!/usr/bin/env python3
"""Freeze and evaluate the SE3-Observed-v1 development analysis.

This runner is deliberately fail-closed about confirmation data.  It consumes
only the four already-exported C1/C2 target-isolated artifacts, preserves the
legacy 276-tuple split, and labels every resulting statistic as development
rather than silently promoting it to an independent test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from analyses.se3_latent_mvp.geometry import se3_exp, se3_log
from analyses.se3_latent_mvp.run_mvp import (
    LatentPredictor,
    cosine_distance,
    load_and_audit_tuples,
    matched_donor_mapping,
    objective,
    pair_tensors,
    records_for_split,
    split_scenes,
    train_model,
    transform_batch,
)
from analyses.se3_protocol_v2.geometry import rotation_degrees, translation_norm
from analyses.se3_protocol_v2.metrics import (
    DEFAULT_NORMALIZATION_EPSILON,
    DEFAULT_TIE_TOLERANCE,
    tie_aware_order_summary,
)
from analyses.se3_protocol_v2.observed_order import action_tensor, derangement, observed_overlap
from analyses.se3_protocol_v2.protocol import POSE_CONVENTION, dump_json, sha256
from analyses.se3_protocol_v2.see_se3_smoke import fit_m3, fit_m4, metrics as readout_r2_metrics, pair_arrays


REPO_ROOT = Path(__file__).resolve().parents[2]
PILOT_ROOT = REPO_ROOT / "results/experimental/se3_protocol_v2_cross_model_pilot_p8_seed42_43_44_r2"
RUN_ROOT = "logs/inscene15k_streaming/runs/c_eval_20260724_1903"
MANIFEST = "manifests/c_retrain_20260722/inscene_c_h124_prefix8.jsonl"
RAW_ROOT = "/mnt/umm/users/yinbaiqiao/InsScene-15K/data/processed_scannetpp_v2/processed_scannetpp_v2"
PRIMARY_BASELINES = (
    "copy",
    "without_condition",
    "action_only",
    "matched_wrong_pose",
    "shuffled_pose",
    "additive",
    "full",
)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    layer: int
    module_name: str
    checkpoint: str
    checkpoint_revision: str
    checkpoint_revision_binding: str
    input_frames: int
    ordered_input_semantics: str
    feature_role: str
    pooling: str
    prefix_semantics: str
    diffusion_condition: str
    c1_dump: str
    c2_dump: str


def _dump(model: str, task: str, layer: int) -> str:
    return f"{RUN_ROOT}/c_eval_20260724_1903_{task}_{model}_p8_layer{layer}/eval/val_predictions.pt"


# Frozen before the paper-v1 scores are produced.  Revisions marked unbound are
# honest provenance gaps: a current cache ref is not retroactive proof of the
# revision that produced a historical remote feature cache.
MODEL_SPECS = (
    ModelSpec(
        "dino", "DINOv2-L", 0, "last_hidden_state (export label 0)",
        "facebook/dinov2-large", "unknown", "historical artifact not revision-bound",
        1, "one RGB frame; no repetition is passed to DINO", "target_isolated anchor token map",
        "spatial mean over 30x52 patch tokens", "p8 names tuple sampling/cache only; not context",
        "N/A deterministic encoder", _dump("dino", "action_dynamics", 0), _dump("dino", "path_integration", 0),
    ),
    ModelSpec(
        "f3r", "Fast3R ViT-L", 24, "returned multiview feature list index 24",
        "jedyang97/Fast3R_ViT_Large_512", "a2c770b768ceb3a53c36c4f7a3619db0413dc3a1",
        "current local cache evidence; historical artifact not revision-bound", 64,
        "the same isolated RGB frame repeated 64 times and jointly encoded by multiview inference",
        "target_isolated middle repeated-view output", "spatial mean over 18x32 tokens",
        "p8 names tuple sampling/cache only; not context", "N/A (deterministic intent; multiview inference API accepts RNG)",
        _dump("f3r", "action_dynamics", 24), _dump("f3r", "path_integration", 24),
    ),
    ModelSpec(
        "vjepa2", "V-JEPA2 ViT-L", 23, "encoder block 23 (0-based)",
        "facebook/vjepa2-vitl-fpc64-256", "unknown", "historical artifact not revision-bound", 64,
        "the same isolated RGB frame repeated 64 times; tubelet encoder only",
        "target_isolated encoder output; skip_predictor=True", "middle temporal token then spatial mean over 16x16",
        "p8 names tuple sampling/cache only; not context", "N/A deterministic encoder",
        _dump("vjepa2", "action_dynamics", 23), _dump("vjepa2", "path_integration", 23),
    ),
    ModelSpec(
        "cogvideox", "CogVideoX-5B-I2V", 20, "transformer block 20",
        "THUDM/CogVideoX-5b-I2V", "a6f0f4858a8395e7429d82493864ce92bf73af11",
        "current local cache evidence; historical artifact not revision-bound", 97,
        "the same isolated RGB frame repeated 97 times; I2V transformer receives no other video content",
        "target_isolated diffusion-transformer hidden state", "middle latent-time token then spatial mean",
        "p8 names tuple sampling/cache only; not context",
        "CogVideoXDDIMScheduler index 749 -> scheduler timestep 250, log-SNR 0.629856 under current cached scheduler; historical noise seed absent",
        _dump("cogvideox", "action_dynamics", 20), _dump("cogvideox", "path_integration", 20),
    ),
)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be a non-empty list of unique integers")
    return seeds


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def file_digest_or_na(path: Path) -> str:
    return sha256(path) if path.is_file() else "N/A"


class AdditivePredictor(nn.Module):
    """Commutative residual baseline P(z,T)=z+g(log(T))."""

    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.increment = nn.Sequential(
            nn.Linear(6, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, feature_dim)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return state + self.increment(action)


def train_additive(
    train_records: list[Any], val_records: list[Any], cfg: dict[str, Any],
    action_mean: torch.Tensor, action_std: torch.Tensor,
) -> tuple[AdditivePredictor, list[dict[str, float]]]:
    train_state, train_action, train_target = pair_tensors(train_records)
    val_state, val_action, val_target = pair_tensors(val_records)
    train_action = (train_action - action_mean) / action_std
    val_action = (val_action - action_mean) / action_std
    model = AdditivePredictor(int(train_state.shape[-1]), int(cfg["hidden_dim"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    generator = torch.Generator().manual_seed(int(cfg["seed"]) + 991)
    best_state, best_val, stale = None, float("inf"), 0
    history: list[dict[str, float]] = []
    for epoch in range(int(cfg["max_epochs"])):
        model.train()
        order = torch.randperm(len(train_state), generator=generator)
        losses = []
        for start in range(0, len(order), int(cfg["batch_size"])):
            index = order[start:start + int(cfg["batch_size"])]
            loss = objective(model(train_state[index], train_action[index]), train_target[index], cfg["cosine_loss_weight"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            val_loss = float(objective(model(val_state, val_action), val_target, cfg["cosine_loss_weight"]))
        history.append({"epoch": float(epoch), "train_loss": float(np.mean(losses)), "val_loss": val_loss})
        if val_loss < best_val - 1e-7:
            best_val, stale = val_loss, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= int(cfg["patience"]):
            break
    if best_state is None:
        raise RuntimeError("additive baseline did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def predict(
    baseline: str, models: dict[str, nn.Module], state: torch.Tensor,
    transforms: np.ndarray, action_mean: torch.Tensor, action_std: torch.Tensor,
) -> torch.Tensor:
    if baseline == "copy":
        return state.clone()
    name = "full" if baseline in {"matched_wrong_pose", "shuffled_pose"} else baseline
    with torch.no_grad():
        return models[name](state, action_tensor(transforms, action_mean, action_std))


def parameter_count(model: nn.Module) -> int:
    return sum(value.numel() for value in model.parameters())


def scene_macro(values: np.ndarray, scenes: np.ndarray) -> float:
    return float(np.mean([np.mean(values[scenes == scene]) for scene in sorted(set(scenes.tolist()))]))


def summarize_prediction_arrays(
    arrays: dict[str, np.ndarray], copy_arrays: dict[str, np.ndarray], scenes: np.ndarray,
    aggregation: str,
) -> dict[str, Any]:
    reducer = (lambda value: float(np.mean(value))) if aggregation == "tuple_micro" else (lambda value: scene_macro(value, scenes))
    rec, rev, direct = arrays["recursive_error"], arrays["reversed_error"], arrays["direct_error"]
    copy_rec = copy_arrays["recursive_error"]
    order = tie_aware_order_summary(
        np.asarray([reducer(rec)]), np.asarray([reducer(rev)]), np.asarray([reducer(copy_rec)]),
        tie_tolerance=DEFAULT_TIE_TOLERANCE, epsilon=DEFAULT_NORMALIZATION_EPSILON,
    )
    # Win/loss/tie counts always remain tuple-level and are therefore auditable.
    tuple_order = tie_aware_order_summary(rec, rev, copy_rec)
    one = np.concatenate((arrays["one_ab_error"], arrays["one_bc_error"]))
    copy_one = np.concatenate((copy_arrays["one_ab_error"], copy_arrays["one_bc_error"]))
    if aggregation == "scene_macro":
        one_scenes = np.concatenate((scenes, scenes))
        one_value, copy_one_value = scene_macro(one, one_scenes), scene_macro(copy_one, one_scenes)
    else:
        one_value, copy_one_value = float(one.mean()), float(copy_one.mean())
    direct_value, rec_value = reducer(direct), reducer(rec)
    result: dict[str, Any] = {
        "aggregation": aggregation,
        "one_step_error": one_value,
        "recursive_two_step_error": rec_value,
        "direct_two_step_error": direct_value,
        "recursive_minus_direct_endpoint_gap": rec_value - direct_value,
        "recursive_direct_prediction_distance": reducer(arrays["recursive_direct_distance"]),
        "one_step_gain_vs_copy": (copy_one_value - one_value) / max(copy_one_value, DEFAULT_NORMALIZATION_EPSILON),
        "two_step_gain_vs_copy": (reducer(copy_rec) - rec_value) / max(reducer(copy_rec), DEFAULT_NORMALIZATION_EPSILON),
        "raw_order_margin": reducer(rev - rec),
        "normalized_order_margin": order["normalized_margin"],
        "tie_aware_accuracy": reducer(arrays["tie_score"]),
        "strict_accuracy": reducer((rev > rec).astype(float)),
        "wins": tuple_order["wins"], "losses": tuple_order["losses"], "ties": tuple_order["ties"],
        "copy_denominator": reducer(copy_rec),
        "near_zero_copy_denominator": reducer(copy_rec) <= 10 * DEFAULT_NORMALIZATION_EPSILON,
    }
    return result


def prediction_arrays(
    baseline: str, models: dict[str, nn.Module], z_a: torch.Tensor, z_b: torch.Tensor, z_c: torch.Tensor,
    t1: np.ndarray, t2: np.ndarray, action_mean: torch.Tensor, action_std: torch.Tensor,
) -> tuple[dict[str, np.ndarray], dict[str, torch.Tensor]]:
    first = predict(baseline, models, z_a, t1, action_mean, action_std)
    one_bc = predict(baseline, models, z_b, t2, action_mean, action_std)
    recursive = predict(baseline, models, first, t2, action_mean, action_std)
    reversed_first = predict(baseline, models, z_a, t2, action_mean, action_std)
    reversed_ = predict(baseline, models, reversed_first, t1, action_mean, action_std)
    # z + g(a) + g(b) is the registered commutative control.  Evaluating the
    # two algebraically identical sums in different float32 association orders
    # creates a tiny numerical pseudo-signal, so enforce the defining equality
    # before any metric or float16 artifact is produced.
    if baseline == "additive":
        reversed_ = recursive.clone()
    composite = np.stack([right @ left for left, right in zip(t1, t2)])
    direct = predict(baseline, models, z_a, composite, action_mean, action_std)
    def np_cos(left: torch.Tensor, right: torch.Tensor) -> np.ndarray:
        return cosine_distance(left, right).cpu().numpy().astype(np.float64)
    rec_error, rev_error = np_cos(recursive, z_c), np_cos(reversed_, z_c)
    margin = rev_error - rec_error
    tie_score = np.where(margin > DEFAULT_TIE_TOLERANCE, 1.0, np.where(margin < -DEFAULT_TIE_TOLERANCE, 0.0, 0.5))
    arrays = {
        "one_ab_error": np_cos(first, z_b), "one_bc_error": np_cos(one_bc, z_c),
        "recursive_error": rec_error, "reversed_error": rev_error, "direct_error": np_cos(direct, z_c),
        "margin": margin, "tie_score": tie_score,
        "recursive_direct_distance": np_cos(recursive, direct),
    }
    predictions = {"one_ab": first, "one_bc": one_bc, "recursive": recursive, "reversed": reversed_, "direct": direct}
    return arrays, predictions


def pose_physical_metrics(target_z: torch.Tensor, prediction_z: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> dict[str, float]:
    target = (target_z * std + mean).cpu().numpy()
    prediction = (prediction_z * std + mean).cpu().numpy()
    residual = [se3_exp(p) @ np.linalg.inv(se3_exp(t)) for p, t in zip(prediction, target)]
    rot = np.asarray([rotation_degrees(value) for value in residual])
    trans = np.asarray([translation_norm(value) for value in residual])
    target_t = np.stack([se3_exp(value)[:3, 3] for value in target])
    pred_t = np.stack([se3_exp(value)[:3, 3] for value in prediction])
    valid = np.linalg.norm(target_t, axis=1) > 0.01
    cos = np.sum(target_t[valid] * pred_t[valid], axis=1) / np.maximum(
        np.linalg.norm(target_t[valid], axis=1) * np.linalg.norm(pred_t[valid], axis=1), 1e-12
    )
    return {
        "rotation_error_deg": float(rot.mean()), "translation_error_m": float(trans.mean()),
        "translation_direction_error_deg": float(np.degrees(np.arccos(np.clip(cos, -1, 1))).mean()) if valid.any() else float("nan"),
        "translation_direction_valid_n": int(valid.sum()), **readout_r2_metrics(target_z, prediction_z, mean, std),
    }


def readout_analysis(partitions: dict[str, list[Any]], seeds: tuple[int, ...], max_epochs: int) -> list[dict[str, Any]]:
    train_s, train_t, train_y = pair_arrays(partitions["train"])
    val_s, val_t, val_y = pair_arrays(partitions["val"])
    test_s, test_t, test_y = pair_arrays(partitions["test"])
    mean, std = train_y.mean(0), train_y.std(0).clamp_min(1e-6)
    train_z, val_z, test_z = (train_y - mean) / std, (val_y - mean) / std, (test_y - mean) / std
    m3_pred, m3_meta = fit_m3(train_s, train_t, train_z, val_s, val_t, val_z, test_s, test_t)
    rows = [{"readout": "M3", "seed": "deterministic", **pose_physical_metrics(test_z, m3_pred, mean, std), **m3_meta}]
    for seed in seeds:
        pred, meta = fit_m4(train_s, train_t, train_z, val_s, val_t, val_z, test_s, test_t, seed, max_epochs)
        rows.append({"readout": "M4", "seed": seed, **pose_physical_metrics(test_z, pred, mean, std), "best_val_mse": meta["best_val_mse"], "epochs": meta["epochs"]})
    return rows


def feature_distance_analysis(partitions: dict[str, list[Any]], seed: int) -> dict[str, Any]:
    train = torch.stack([record.features for record in partitions["train"]]).reshape(-1, partitions["train"][0].features.shape[-1])
    test = torch.stack([record.features for record in partitions["test"]])
    mean = train.mean(0)
    z_a, z_b, z_c = test[:, 0], test[:, 1], test[:, 2]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(z_a))
    test_scenes = np.asarray([record.scene_id for record in partitions["test"]])
    for _ in range(64):
        if np.all(test_scenes[order] != test_scenes):
            break
        order = rng.permutation(len(z_a))
    if not np.all(test_scenes[order] != test_scenes):
        order = np.roll(np.arange(len(z_a)), 4)
    covariance = torch.linalg.eigvalsh(torch.cov(train.T)).flip(0).clamp_min(0)
    total = covariance.sum().clamp_min(1e-12)
    centered_ac = cosine_distance(z_a - mean, z_c - mean)
    return {
        "copy_distance_A_C": float(cosine_distance(z_a, z_c).mean()),
        "same_scene_one_step_distance": float(torch.cat((cosine_distance(z_a, z_b), cosine_distance(z_b, z_c))).mean()),
        "different_scene_distance": float(cosine_distance(z_a, z_c[order]).mean()),
        "centered_copy_distance_A_C_train_mean_only": float(centered_ac.mean()),
        "feature_norm_mean": float(torch.linalg.norm(test.reshape(-1, test.shape[-1]), dim=1).mean()),
        "train_feature_variance_mean": float(train.var(0, unbiased=False).mean()),
        "covariance_top1_fraction": float(covariance[0] / total),
        "covariance_top10_fraction": float(covariance[:10].sum() / total),
        "covariance_effective_rank": float(torch.exp(-(covariance / total * torch.log((covariance / total).clamp_min(1e-12))).sum())),
        "different_scene_pairing_has_zero_same_scene": bool(np.all(test_scenes[order] != test_scenes)),
    }


def load_records(spec: ModelSpec) -> tuple[list[Any], dict[str, Any]]:
    cfg = {
        "manifest": str(resolve(MANIFEST)), "c1_dump": str(resolve(spec.c1_dump)), "c2_dump": str(resolve(spec.c2_dump)),
        "raw_scene_root": str(resolve(RAW_ROOT)), "alignment_atol": 2e-5, "pose_atol": 1e-5,
    }
    records, audit = load_and_audit_tuples(cfg)
    keys = [(r.scene_id, r.sample_id, *r.raw_frame_indices) for r in records]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"{spec.key}: duplicate physical tuple")
    return records, audit


def train_all(
    partitions: dict[str, list[Any]], seed: int, cfg_base: dict[str, Any]
) -> tuple[dict[str, nn.Module], dict[str, Any]]:
    cfg = {**cfg_base, "seed": seed}
    _, train_action, _ = pair_tensors(partitions["train"])
    action_mean, action_std = train_action.mean(0), train_action.std(0).clamp_min(1e-5)
    models: dict[str, nn.Module] = {}
    meta: dict[str, Any] = {}
    for condition in ("without_condition", "action_only", "full"):
        torch.manual_seed(seed)
        model, history = train_model(condition, partitions["train"], partitions["val"], cfg, action_mean, action_std)
        models[condition] = model
        meta[condition] = {"epochs": len(history), "best_val_loss": min(row["val_loss"] for row in history), "parameters": parameter_count(model)}
    torch.manual_seed(seed)
    additive, history = train_additive(partitions["train"], partitions["val"], cfg, action_mean, action_std)
    models["additive"] = additive
    meta["additive"] = {"epochs": len(history), "best_val_loss": min(row["val_loss"] for row in history), "parameters": parameter_count(additive), "order_blind_by_construction": True}
    return models, {"conditions": meta, "action_mean": action_mean, "action_std": action_std}


def geometry_rows(records: list[Any], raw_root: Path) -> list[dict[str, Any]]:
    cache: dict[str, Any] = {}
    rows = []
    for record in records:
        t1, t2, tac = record.transforms
        e_a = np.vstack((record.extrinsics[0], [0, 0, 0, 1]))
        e12, e21 = t2 @ t1 @ e_a, t1 @ t2 @ e_a
        center12 = -e12[:3, :3].T @ e12[:3, 3]
        center21 = -e21[:3, :3].T @ e21[:3, 3]
        rot_sep = rotation_degrees(e21 @ np.linalg.inv(e12))
        trans_sep = float(np.linalg.norm(center12 - center21))
        xi1, xi2, xic = se3_log(t1), se3_log(t2), se3_log(tac)
        rows.append({
            "tuple_id": record.sample_id, "scene_id": record.scene_id,
            "gap_AB": record.raw_frame_indices[1] - record.raw_frame_indices[0],
            "gap_BC": record.raw_frame_indices[2] - record.raw_frame_indices[1],
            "rotation_AB_deg": rotation_degrees(t1), "rotation_BC_deg": rotation_degrees(t2),
            "translation_AB_m": translation_norm(t1), "translation_BC_m": translation_norm(t2),
            "composite_twist_norm": float(np.linalg.norm(xic)),
            "order_rotation_separation_deg": rot_sep,
            "order_camera_center_separation_m": trans_sep,
            "commutator_twist_norm": float(np.linalg.norm(se3_log(t2 @ t1 @ np.linalg.inv(t2) @ np.linalg.inv(t1)))),
            "overlap_AC": observed_overlap(record, raw_root, cache),
        })
    return rows


def evaluate_seed(
    partitions: dict[str, list[Any]], seed: int, cfg_base: dict[str, Any], child: Path,
    artifact_prefix: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    models, training = train_all(partitions, seed, cfg_base)
    action_mean, action_std = training.pop("action_mean"), training.pop("action_std")
    test = partitions["test"]
    features = torch.stack([record.features for record in test])
    z_a, z_b, z_c = features[:, 0], features[:, 1], features[:, 2]
    t1, t2, _ = transform_batch(test)
    donor = matched_donor_mapping(test)
    shuffle1, shuffle2 = derangement(len(test), seed + 101), derangement(len(test), seed + 211)
    actions = {
        "copy": (t1, t2), "without_condition": (t1, t2), "action_only": (t1, t2),
        "additive": (t1, t2), "full": (t1, t2),
        "matched_wrong_pose": (t1[donor], t2[donor]), "shuffled_pose": (t1[shuffle1], t2[shuffle2]),
    }
    arrays_by_baseline: dict[str, dict[str, np.ndarray]] = {}
    predictions_to_save: dict[str, np.ndarray] = {}
    for baseline in PRIMARY_BASELINES:
        arrays, predictions = prediction_arrays(baseline, models, z_a, z_b, z_c, *actions[baseline], action_mean, action_std)
        arrays_by_baseline[baseline] = arrays
        for name, value in predictions.items():
            predictions_to_save[f"{baseline}__{name}"] = value.cpu().numpy().astype(np.float16)
    copy_arrays = arrays_by_baseline["copy"]
    scenes = np.asarray([record.scene_id for record in test])
    seed_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    without = arrays_by_baseline["without_condition"]
    for baseline in PRIMARY_BASELINES:
        arrays = arrays_by_baseline[baseline]
        for aggregation in ("scene_macro", "tuple_micro"):
            row = summarize_prediction_arrays(arrays, copy_arrays, scenes, aggregation)
            one = np.concatenate((arrays["one_ab_error"], arrays["one_bc_error"]))
            without_one = np.concatenate((without["one_ab_error"], without["one_bc_error"]))
            row.update({
                "baseline": baseline, "seed": seed,
                "one_step_gain_vs_without_condition": (float(without_one.mean()) - float(one.mean())) / max(float(without_one.mean()), DEFAULT_NORMALIZATION_EPSILON),
                "two_step_gain_vs_without_condition": (float(without["recursive_error"].mean()) - float(arrays["recursive_error"].mean())) / max(float(without["recursive_error"].mean()), DEFAULT_NORMALIZATION_EPSILON),
            })
            seed_rows.append(row)
        for index, record in enumerate(test):
            sample_rows.append({
                "tuple_id": record.sample_id, "scene_id": record.scene_id, "seed": seed, "baseline": baseline,
                **{key: float(value[index]) for key, value in arrays.items()},
            })
    np.savez_compressed(child / f"raw_predictions_{artifact_prefix}seed{seed}.npz", **predictions_to_save)
    controls = {
        "matched_wrong_pose_fixed_points": int(np.sum(donor == np.arange(len(test)))),
        "matched_wrong_pose_same_scene": int(sum(test[i].scene_id == test[j].scene_id for i, j in enumerate(donor))),
        "shuffle_step1_fixed_points": int(np.sum(shuffle1 == np.arange(len(test)))),
        "shuffle_step2_fixed_points": int(np.sum(shuffle2 == np.arange(len(test)))),
        "pose_marginals_exact_permutations": bool(
            np.allclose(np.sort([np.linalg.norm(se3_log(v)) for v in t1]), np.sort([np.linalg.norm(se3_log(v)) for v in t1[donor]]))
            and np.allclose(np.sort([np.linalg.norm(se3_log(v)) for v in t2]), np.sort([np.linalg.norm(se3_log(v)) for v in t2[shuffle2]]))
        ),
        "additive_max_abs_recursive_reverse_feature_delta": float(np.max(np.abs(
            predictions_to_save["additive__recursive"].astype(np.float32) - predictions_to_save["additive__reversed"].astype(np.float32)
        ))),
    }
    if any(controls[key] for key in controls if key.endswith("fixed_points") or key.endswith("same_scene")) or not controls["pose_marginals_exact_permutations"]:
        raise RuntimeError(f"pose control failed: {controls}")
    return seed_rows, sample_rows, {"training": training, "controls": controls}


def common_dim_partitions(
    partitions: dict[str, list[Any]], dimension: int, seed: int, child: Path,
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    """Fit a train-only centered PCA and freeze it for all three splits."""
    train = torch.stack([record.features for record in partitions["train"]]).reshape(
        -1, partitions["train"][0].features.shape[-1]
    )
    mean = train.mean(0)
    torch.manual_seed(seed)
    _, singular_values, basis = torch.pca_lowrank(train - mean, q=dimension, center=False, niter=4)
    reduced = {
        split: [replace(record, features=(record.features - mean) @ basis) for record in records]
        for split, records in partitions.items()
    }
    torch.save({"train_mean": mean, "basis": basis}, child / f"common{dimension}_train_only_pca.pt")
    return reduced, {
        "dimension": dimension,
        "fit_split": "train_only",
        "centering": "train_mean_only",
        "pca_seed": seed,
        "retained_train_singular_value_fraction": float(
            singular_values.square().sum() / torch.linalg.norm(train - mean).square().clamp_min(1e-12)
        ),
    }


def run_child(output: Path, spec: ModelSpec, seeds: tuple[int, ...], split_seed: int, max_epochs: int, pose_epochs: int) -> None:
    child = output / f"model_{spec.key}"
    if child.exists() and any(child.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty model output: {child}")
    child.mkdir(parents=True, exist_ok=True)
    records, tuple_audit = load_records(spec)
    assignments, split_audit = split_scenes(records, split_seed, 0.7, 0.15)
    partitions = {name: records_for_split(records, assignments, name) for name in ("train", "val", "test")}
    cfg = {"hidden_dim": 64, "learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": 64, "max_epochs": max_epochs, "patience": 30, "cosine_loss_weight": 0.25}
    all_metrics, all_samples, meta = [], [], {}
    for seed in seeds:
        metrics, samples, seed_meta = evaluate_seed(partitions, seed, cfg, child)
        all_metrics.extend(metrics); all_samples.extend(samples); meta[str(seed)] = seed_meta
    # Capacity control is deliberately limited to the widest native feature
    # (CogVideoX) and a 1024-d anchor (DINO).  Both use the identical frozen
    # train-only 128-d PCA and the identical probe/training protocol.
    if spec.key in {"dino", "cogvideox"}:
        reduced, reduction_meta = common_dim_partitions(partitions, 128, split_seed, child)
        capacity_metrics, capacity_samples, capacity_training = [], [], {}
        for seed in seeds:
            metrics, samples, seed_meta = evaluate_seed(
                reduced, seed, cfg, child, artifact_prefix="common128_"
            )
            for row in metrics:
                row["representation"] = "train_only_pca_128"
            for row in samples:
                row["representation"] = "train_only_pca_128"
            capacity_metrics.extend(metrics); capacity_samples.extend(samples)
            capacity_training[str(seed)] = seed_meta
        write_csv(child / "capacity_control_seed_metrics.csv", capacity_metrics)
        write_csv(child / "capacity_control_sample_metrics.csv", capacity_samples)
        dump_json(child / "capacity_control_training.json", capacity_training)
        dump_json(child / "capacity_control_reduction.json", reduction_meta)
    readouts = readout_analysis(partitions, seeds, pose_epochs)
    distances = feature_distance_analysis(partitions, split_seed)
    geometry = geometry_rows(partitions["test"], resolve(RAW_ROOT))
    write_csv(child / "per_seed_metrics.csv", all_metrics)
    write_csv(child / "per_sample_metrics.csv", all_samples)
    write_csv(child / "readout_metrics.csv", readouts)
    write_csv(child / "test_geometry.csv", geometry)
    dump_json(child / "training.json", meta)
    dump_json(child / "feature_distance.json", distances)
    dump_json(child / "tuple_audit.json", tuple_audit)
    dump_json(child / "split.json", {"assignments": assignments, "audit": split_audit})
    dump_json(child / "provenance.json", {
        "model": asdict(spec),
        "inputs": {name: {"path": str(path), "sha256": file_digest_or_na(path)} for name, path in {
            "manifest": resolve(MANIFEST), "c1_dump": resolve(spec.c1_dump), "c2_dump": resolve(spec.c2_dump),
            "extractor": resolve("features/run_inscene15k.py"), "runner": Path(__file__).resolve(),
        }.items()},
    })


def load_manifest_records() -> dict[str, dict[str, Any]]:
    result = {}
    with resolve(MANIFEST).open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] == "val":
                result[str(row["sample_id"])] = row
    return result


def build_frozen_manifest(
    output: Path, records: list[Any], assignments: dict[str, str], split_seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = load_manifest_records()
    rows, csv_rows = [], []
    raw_cache: dict[str, Any] = {}
    overlap_cache: dict[str, Any] = {}
    artifact_hashes = {
        spec.key: {"c1_sha256": file_digest_or_na(resolve(spec.c1_dump)), "c2_sha256": file_digest_or_na(resolve(spec.c2_dump))}
        for spec in MODEL_SPECS
    }
    for record in records:
        manifest_row = source[record.sample_id]
        if record.scene_id not in raw_cache:
            raw_cache[record.scene_id] = np.load(resolve(RAW_ROOT) / record.scene_id / "scene_iphone_metadata.npz")
        raw = raw_cache[record.scene_id]
        indices = list(record.raw_frame_indices)
        c2w = raw["trajectories"][indices]
        intrinsics = raw["intrinsics"][indices]
        names = [str(raw["images"][index]) for index in indices]
        e = np.stack([np.linalg.inv(value) for value in c2w])
        t1, t2, tac = record.transforms
        e12, e21 = t2 @ t1 @ e[0], t1 @ t2 @ e[0]
        center12 = -e12[:3, :3].T @ e12[:3, 3]
        center21 = -e21[:3, :3].T @ e21[:3, 3]
        tuple_id = hashlib.sha256(f"{record.scene_id}|{indices[0]}|{indices[1]}|{indices[2]}".encode()).hexdigest()[:20]
        row = {
            "tuple_id": tuple_id, "legacy_sample_id": record.sample_id, "scene_id": record.scene_id,
            "split": assignments[record.scene_id], "analysis_role": f"legacy_development_{assignments[record.scene_id]}",
            "raw_frame_indices": indices, "raw_frame_ids": names,
            "timestamps_s": [None, None, None], "timestamp_status": "not supplied by source metadata; not inferred from filenames",
            "temporal_gaps_index_units": [indices[1] - indices[0], indices[2] - indices[1]],
            "c2w": c2w.tolist(), "w2c": e.tolist(), "intrinsics": intrinsics.tolist(),
            "T_AB": t1.tolist(), "T_BC": t2.tolist(), "T_AC": tac.tolist(),
            "ordered_target_isolated_input_frame_ids": {spec.key: [[name] * spec.input_frames for name in names] for spec in MODEL_SPECS},
            "physical_sample_key": f"{record.scene_id}:{indices[0]}:{indices[1]}:{indices[2]}",
            "feature_keys": {
                spec.key: {
                    state: f"{spec.key}:target_isolated:{record.scene_id}:{name}:layer{spec.layer}"
                    for state, name in zip(("A", "B", "C"), names)
                }
                for spec in MODEL_SPECS
            },
            "feature_contracts": {
                spec.key: {
                    "feature_role": spec.feature_role, "layer": spec.layer,
                    "module_name": spec.module_name, "pooling": spec.pooling,
                    "checkpoint": spec.checkpoint, "checkpoint_revision": spec.checkpoint_revision,
                    "checkpoint_revision_binding": spec.checkpoint_revision_binding,
                    "prefix_semantics": spec.prefix_semantics,
                    "diffusion_condition": spec.diffusion_condition,
                    "ordered_input_frames_per_state": spec.input_frames,
                    **artifact_hashes[spec.key],
                }
                for spec in MODEL_SPECS
            },
            "motion_geometry": {
                "rotation_AB_deg": rotation_degrees(t1), "rotation_BC_deg": rotation_degrees(t2),
                "rotation_AC_deg": rotation_degrees(tac),
                "translation_AB_m": translation_norm(t1), "translation_BC_m": translation_norm(t2),
                "translation_AC_m": translation_norm(tac),
                "order_rotation_separation_deg": rotation_degrees(e21 @ np.linalg.inv(e12)),
                "order_camera_center_separation_m": float(np.linalg.norm(center12 - center21)),
                "commutator_twist_norm": float(np.linalg.norm(se3_log(t2 @ t1 @ np.linalg.inv(t2) @ np.linalg.inv(t1)))),
                "observed_overlap_AC": observed_overlap(record, resolve(RAW_ROOT), overlap_cache),
            },
            "source_manifest_sample_hash": hashlib.sha256(json.dumps(manifest_row, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        }
        rows.append(row)
        csv_rows.append({"tuple_id": tuple_id, "legacy_sample_id": record.sample_id, "scene_id": record.scene_id, "split": assignments[record.scene_id], "frame_A": indices[0], "frame_B": indices[1], "frame_C": indices[2], "raw_id_A": names[0], "raw_id_B": names[1], "raw_id_C": names[2]})
    jsonl = output / "se3_observed_v1_manifest.jsonl"
    with jsonl.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    write_csv(output / "se3_observed_v1_manifest.csv", csv_rows)
    scenes = {split: sorted({r.scene_id for r in records if assignments[r.scene_id] == split}) for split in ("train", "val", "test")}
    dump_json(output / "split.json", {"seed": split_seed, "assignments": assignments, "scenes": scenes, "status": "legacy_development_not_independent_confirmation"})
    counts = {
        split: {
            "tuples": sum(assignments[r.scene_id] == split for r in records),
            "scenes": len(scenes[split]),
            "unique_frames": len({(r.scene_id, frame) for r in records if assignments[r.scene_id] == split for frame in r.raw_frame_indices}),
        } for split in ("train", "val", "test")
    }
    return rows, {"counts": counts, "manifest_sha256": sha256(jsonl), "physical_tuple_duplicates": len(rows) - len({r["physical_sample_key"] for r in rows})}


def motion_support_audit(records: list[Any], assignments: dict[str, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_rot, train_trans = [], []
    for record in records:
        if assignments[record.scene_id] != "train":
            continue
        for transform in record.transforms[:2]:
            train_rot.append(rotation_degrees(transform)); train_trans.append(translation_norm(transform))
    rot_bounds = (float(np.min(train_rot)), float(np.max(train_rot)))
    trans_bounds = (float(np.min(train_trans)), float(np.max(train_trans)))
    rows = []
    for record in records:
        if assignments[record.scene_id] != "test":
            continue
        composite = record.transforms[2]
        rot, trans = rotation_degrees(composite), translation_norm(composite)
        rows.append({
            "tuple_id": record.sample_id, "scene_id": record.scene_id,
            "composite_rotation_deg": rot, "composite_translation_m": trans,
            "within_train_single_transition_rotation_minmax": rot_bounds[0] <= rot <= rot_bounds[1],
            "within_train_single_transition_translation_minmax": trans_bounds[0] <= trans <= trans_bounds[1],
            "within_joint_train_single_transition_minmax": rot_bounds[0] <= rot <= rot_bounds[1] and trans_bounds[0] <= trans <= trans_bounds[1],
        })
    return rows, {
        "definition": "empirical componentwise min/max support of real train-scene AB and BC single transitions",
        "train_single_transitions": len(train_rot),
        "train_rotation_deg_minmax": rot_bounds, "train_translation_m_minmax": trans_bounds,
        "test_composites": len(rows),
        "test_composites_within_joint_support": sum(bool(r["within_joint_train_single_transition_minmax"]) for r in rows),
        "test_composites_outside_joint_support": sum(not bool(r["within_joint_train_single_transition_minmax"]) for r in rows),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2
        start = stop
    return ranks


def order_geometry_associations(
    sample_rows: list[dict[str, Any]], geometry: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    geometry_index = {(row["scene_id"], row["tuple_id"]): row for row in geometry}
    predictors = (
        "order_rotation_separation_deg", "order_camera_center_separation_m",
        "rotation_AB_deg", "rotation_BC_deg", "translation_AB_m", "translation_BC_m",
        "gap_AB", "gap_BC", "overlap_AC",
    )
    rows = []
    for spec in MODEL_SPECS:
        grouped: dict[tuple[str, str], list[float]] = {}
        for row in sample_rows:
            if row["model"] == spec.key and row["baseline"] == "full":
                grouped.setdefault((row["scene_id"], row["tuple_id"]), []).append(float(row["margin"]))
        keys = sorted(grouped)
        outcome = np.asarray([np.mean(grouped[key]) for key in keys])
        for predictor in predictors:
            value_ = np.asarray([float(geometry_index[key][predictor]) for key in keys])
            pearson = float(np.corrcoef(value_, outcome)[0, 1]) if np.std(value_) > 0 and np.std(outcome) > 0 else float("nan")
            spearman = float(np.corrcoef(_average_ranks(value_), _average_ranks(outcome))[0, 1]) if np.std(value_) > 0 and np.std(outcome) > 0 else float("nan")
            rows.append({
                "model": spec.key, "display_name": spec.display_name, "predictor": predictor,
                "outcome": "full raw observed-order margin averaged over training seeds",
                "n_tuples": len(keys), "pearson_r": pearson, "spearman_rho": spearman,
                "status": "exploratory_descriptive_no_multiplicity_adjustment",
            })
    return rows


def mean_rows(rows: list[dict[str, Any]], group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in group_keys), []).append(row)
    result = []
    for key, values in groups.items():
        out = dict(zip(group_keys, key)); out["n_seeds"] = len(values)
        for column in values[0]:
            if column in group_keys or column in {"seed", "wins", "losses", "ties", "near_zero_copy_denominator"}:
                continue
            try:
                numbers = np.asarray([float(row[column]) for row in values], dtype=float)
            except (TypeError, ValueError):
                continue
            out[f"{column}_mean"] = float(numbers.mean()); out[f"{column}_seed_std"] = float(numbers.std(ddof=0))
        result.append(out)
    return result


def raw_sample_index(rows: list[dict[str, Any]]) -> dict[tuple[str, int, str, str], dict[str, Any]]:
    return {(str(row["scene_id"]), int(row["seed"]), str(row["baseline"]), str(row["tuple_id"])): row for row in rows}


def holm_adjust(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted.tolist()


def paired_bootstrap(
    sample_rows: list[dict[str, Any]], repetitions: int, bootstrap_seed: int,
    seeds: tuple[int, ...], test_scenes: list[str],
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(bootstrap_seed)
    draws = rng.integers(0, len(test_scenes), size=(repetitions, len(test_scenes)))

    comparisons: list[tuple[str, str, str, str, str]] = []
    for spec in MODEL_SPECS:
        comparisons.extend([
            (spec.key, "full", spec.key, "copy", "grounding_gain_vs_copy"),
            (spec.key, "full", spec.key, "copy", "normalized_order_margin"),
            (spec.key, "full", spec.key, "action_only", "normalized_margin_full_minus_action_only"),
            (spec.key, "full", spec.key, "shuffled_pose", "normalized_margin_full_minus_shuffle"),
            (spec.key, "full", spec.key, "matched_wrong_pose", "normalized_margin_full_minus_matched_wrong_pose"),
            (spec.key, "full", spec.key, "additive", "normalized_margin_full_minus_additive"),
        ])
    for left in range(len(MODEL_SPECS)):
        for right in range(left + 1, len(MODEL_SPECS)):
            comparisons.append((MODEL_SPECS[left].key, "full", MODEL_SPECS[right].key, "full", "normalized_margin_model_difference"))

    # The caller passes model-tagged rows, but the compact index above needs a
    # model axis.  Rebuild it explicitly here to retain identical resamples.
    model_index = {(str(r["model"]), str(r["scene_id"]), int(r["seed"]), str(r["baseline"]), str(r["tuple_id"])): r for r in sample_rows}
    tuple_ids = {scene: sorted({key[4] for key in model_index if key[1] == scene}) for scene in test_scenes}
    value_cache: dict[tuple[str, str, str], np.ndarray] = {}
    def values(model: str, baseline: str, field: str) -> np.ndarray:
        key = (model, baseline, field)
        if key not in value_cache:
            scene_means = np.asarray([
                [
                    np.mean([float(model_index[(model, scene, seed, baseline, tid)][field]) for tid in tuple_ids[scene]])
                    for scene in test_scenes
                ]
                for seed in seeds
            ])
            value_cache[key] = scene_means.mean(axis=0)[draws].mean(axis=1)
        return value_cache[key]
    rows = []
    for left_model, left_base, right_model, right_base, metric in comparisons:
        if metric == "grounding_gain_vs_copy":
            full = values(left_model, left_base, "recursive_error")
            copy = values(right_model, right_base, "recursive_error")
            values_np = (copy - full) / np.maximum(copy, DEFAULT_NORMALIZATION_EPSILON)
        elif metric == "normalized_order_margin":
            copy = values(left_model, "copy", "recursive_error")
            values_np = values(left_model, left_base, "margin") / np.maximum(copy, DEFAULT_NORMALIZATION_EPSILON)
        else:
            copy = values(left_model, "copy", "recursive_error")
            left_margin = values(left_model, left_base, "margin") / np.maximum(copy, DEFAULT_NORMALIZATION_EPSILON)
            right_copy = values(right_model, "copy", "recursive_error")
            right_margin = values(right_model, right_base, "margin") / np.maximum(right_copy, DEFAULT_NORMALIZATION_EPSILON)
            values_np = left_margin - right_margin
        p = min(1.0, 2 * min(float(np.mean(values_np <= 0)), float(np.mean(values_np >= 0))))
        rows.append({
            "left_model": left_model, "left_baseline": left_base, "right_model": right_model, "right_baseline": right_base,
            "metric": metric, "estimate": float(values_np.mean()), "ci95_low": float(np.quantile(values_np, .025)),
            "ci95_high": float(np.quantile(values_np, .975)), "bootstrap_p_two_sided": p,
            "bootstrap_unit": "scene", "resamples": repetitions, "training_seeds_averaged": len(seeds),
        })
    adjusted = holm_adjust([float(row["bootstrap_p_two_sided"]) for row in rows])
    for row, value_ in zip(rows, adjusted):
        row["holm_adjusted_p"] = value_
    return rows


def paper_main_rows(
    aggregate: list[dict[str, Any]], bootstrap: list[dict[str, Any]], dataset_audit: dict[str, Any]
) -> list[dict[str, Any]]:
    full = {r["model"]: r for r in aggregate if r["baseline"] == "full" and r["aggregation"] == "scene_macro"}
    ci = {(r["left_model"], r["metric"]): r for r in bootstrap}
    rows = []
    for spec in MODEL_SPECS:
        base = full[spec.key]
        ground = ci[(spec.key, "grounding_gain_vs_copy")]
        order = ci[(spec.key, "normalized_order_margin")]
        row = {
            "model": spec.key, "display_name": spec.display_name,
            "test_tuples": dataset_audit["counts"]["test"]["tuples"],
            "test_scenes": dataset_audit["counts"]["test"]["scenes"],
            "training_seeds": int(base["n_seeds"]),
            "one_step_error": base["one_step_error_mean"],
            "recursive_two_step_error": base["recursive_two_step_error_mean"],
            "direct_two_step_error": base["direct_two_step_error_mean"],
            "recursive_minus_direct_endpoint_gap": base["recursive_minus_direct_endpoint_gap_mean"],
            "grounding_gain_vs_copy": base["two_step_gain_vs_copy_mean"],
            "grounding_gain_ci95_low": ground["ci95_low"], "grounding_gain_ci95_high": ground["ci95_high"],
            "raw_order_margin": base["raw_order_margin_mean"],
            "normalized_order_margin": base["normalized_order_margin_mean"],
            "normalized_order_ci95_low": order["ci95_low"], "normalized_order_ci95_high": order["ci95_high"],
            "tie_aware_accuracy": base["tie_aware_accuracy_mean"],
            "strict_accuracy": base["strict_accuracy_mean"],
        }
        for short, metric in (
            ("action_only", "normalized_margin_full_minus_action_only"),
            ("shuffled_pose", "normalized_margin_full_minus_shuffle"),
            ("matched_wrong_pose", "normalized_margin_full_minus_matched_wrong_pose"),
            ("additive", "normalized_margin_full_minus_additive"),
        ):
            result = ci[(spec.key, metric)]
            row[f"order_effect_vs_{short}"] = result["estimate"]
            row[f"order_effect_vs_{short}_ci95_low"] = result["ci95_low"]
            row[f"order_effect_vs_{short}_ci95_high"] = result["ci95_high"]
            row[f"order_effect_vs_{short}_holm_p"] = result["holm_adjusted_p"]
        rows.append(row)
    return rows


def write_model_card(output: Path, training: dict[str, Any]) -> None:
    rows = []
    for spec in MODEL_SPECS:
        params = training[spec.key][str(42)]["training"]["conditions"]
        rows.append({
            **asdict(spec), "prefix": "p8", "feature_dim": 3072 if spec.key == "cogvideox" else 1024,
            "probe_hidden_width": 64, "full_probe_parameters": params["full"]["parameters"],
            "action_only_probe_parameters": params["action_only"]["parameters"],
            "without_condition_probe_parameters": params["without_condition"]["parameters"],
            "additive_probe_parameters": params["additive"]["parameters"],
            "track": "causal/update target-isolated (no future endpoint in source)",
        })
        capacity_path = output / f"model_{spec.key}" / "capacity_control_training.json"
        if capacity_path.is_file():
            capacity = read_json(capacity_path)[str(42)]["training"]["conditions"]
            rows.append({
                **asdict(spec), "display_name": f"{spec.display_name} (train-only PCA-128)",
                "prefix": "p8", "feature_dim": 128, "probe_hidden_width": 64,
                "full_probe_parameters": capacity["full"]["parameters"],
                "action_only_probe_parameters": capacity["action_only"]["parameters"],
                "without_condition_probe_parameters": capacity["without_condition"]["parameters"],
                "additive_probe_parameters": capacity["additive"]["parameters"],
                "track": "common-dimension capacity control; train-only PCA frozen before probe fitting",
            })
    write_csv(output / "MODEL_CARD_TABLE.csv", rows)


def render_figures(
    output: Path, aggregate: list[dict[str, Any]], paper_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]], geometry: list[dict[str, Any]],
    distances: list[dict[str, Any]], seeds: tuple[int, ...],
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.1))
    for row in paper_rows:
        x, y = row["grounding_gain_vs_copy"], row["normalized_order_margin"]
        ax.errorbar(
            x, y,
            xerr=[[x - row["grounding_gain_ci95_low"]], [row["grounding_gain_ci95_high"] - x]],
            yerr=[[y - row["normalized_order_ci95_low"]], [row["normalized_order_ci95_high"] - y]],
            fmt="o", ms=7, capsize=3,
        )
        ax.annotate(row["display_name"], (x, y), xytext=(5, 4), textcoords="offset points")
    ax.axhline(0, color="black", lw=.8); ax.axvline(0, color="black", lw=.8)
    ax.set(xlabel="two-step grounding gain vs copy", ylabel="normalized observed-order margin", title=f"Legacy development: 44 tuples / 11 scenes / {len(seeds)} seeds\n95% paired scene-bootstrap CI, 2,000 resamples")
    ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(output / "figure_grounding_vs_normalized_order.png", dpi=200); fig.savefig(output / "figure_grounding_vs_normalized_order.pdf"); plt.close(fig)

    geom = {(row["scene_id"], row["tuple_id"]): row for row in geometry}
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    for spec in MODEL_SPECS:
        rows = [r for r in sample_rows if r["model"] == spec.key and r["baseline"] == "full"]
        grouped: dict[tuple[str, str], list[float]] = {}
        for row in rows: grouped.setdefault((row["scene_id"], row["tuple_id"]), []).append(float(row["margin"]))
        xrot, xtrans, y = [], [], []
        for key, values in grouped.items():
            if key not in geom: continue
            xrot.append(float(geom[key]["order_rotation_separation_deg"])); xtrans.append(float(geom[key]["order_camera_center_separation_m"])); y.append(float(np.mean(values)))
        axes[0].scatter(xrot, y, s=20, alpha=.65, label=spec.display_name); axes[1].scatter(xtrans, y, s=20, alpha=.65, label=spec.display_name)
    for ax, label in zip(axes, ("rotation separation (deg)", "camera-center separation (m)")):
        ax.axhline(0, color="black", lw=.7); ax.set_xlabel(label); ax.set_ylabel("raw observed-order margin"); ax.grid(alpha=.2)
    axes[1].legend(frameon=False, fontsize=8); fig.suptitle(f"Order signal vs non-commutativity (44 tuples / 11 scenes / {len(seeds)} seeds; seed means, no CI)"); fig.tight_layout(); fig.savefig(output / "figure_order_vs_noncommutativity.png", dpi=200); fig.savefig(output / "figure_order_vs_noncommutativity.pdf"); plt.close(fig)

    labels = [next(spec.display_name for spec in MODEL_SPECS if spec.key == row["model"]) for row in distances]
    values = np.asarray([[float(row[key]) for row in distances] for key in (
        "copy_distance_A_C", "same_scene_one_step_distance", "different_scene_distance"
    )])
    x = np.arange(len(labels)); width = .24
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for offset, name, value_ in zip((-width, 0, width), ("A-C copy", "same-scene 1-step", "different-scene"), values):
        ax.bar(x + offset, value_, width, label=name)
    ax.set_xticks(x, labels, rotation=15, ha="right"); ax.set_ylabel("cosine distance")
    ax.set_title("Feature-distance sanity (44 tuples / 11 scenes; descriptive, no bootstrap CI)")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.2); fig.tight_layout()
    fig.savefig(output / "figure_feature_distance_sanity.png", dpi=200)
    fig.savefig(output / "figure_feature_distance_sanity.pdf"); plt.close(fig)


def latex_table(rows: list[dict[str, Any]]) -> str:
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Model & One-step $\downarrow$ & Two-step $\downarrow$ & Norm. order $\uparrow$ & Tie-aware acc. $\uparrow$ \\", r"\midrule"]
    for row in rows:
        lines.append(f"{row['display_name']} & {row['one_step_error']:.4f} & {row['recursive_two_step_error']:.4f} & {row['normalized_order_margin']:.4f} & {row['tie_aware_accuracy']:.4f} \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def write_study_documents(output: Path, dataset_audit: dict[str, Any], inventory: dict[str, Any]) -> None:
    (output / "STUDY_SPEC.md").write_text("""# SE3-Observed-v1 study specification

Frozen before the paper-v1 scores in this directory were computed.

Research question: Do frozen visual representations support action-grounded and order-sensitive latent updates under observed camera motion?

The four questions are: (Q1) held-out relative-pose readout from real observed feature pairs; (Q2) whether the correct action improves one- and two-step real-endpoint prediction; (Q3) whether correct action order is closer to the observed endpoint than reversed order and exceeds action-only, shuffled-pose, and commutative-additive controls; and (Q4) how composition error varies with model, available layer/context/diffusion condition, and physical non-commutativity.

The core cells are DINOv2-L export 0, Fast3R export 24, V-JEPA2 encoder block 23, and CogVideoX block 20, all from the existing p8 C artifacts. Seeds are 42--46. Backbones and features are frozen. The residual probe uses hidden width 64, AdamW (lr 1e-3, weight decay 1e-4), MSE plus 0.25 cosine loss, early stopping on development validation, and only real training-scene single-transition targets. No order, composition, cycle, rollout, validation-test, or test loss is used.

Primary aggregation is scene-macro. Tuple-micro is supplementary. Paired scene bootstrap uses 2,000 fixed resamples shared across models and controls. The primary order quantities are raw margin, copy-normalized margin, and tie-aware accuracy with absolute tie tolerance 1e-7 and normalization epsilon 1e-8. Holm correction is applied jointly to the registered control and pairwise model comparisons.

The endpoint convention is `E=inverse(c2w)`, `T_AB=E_B@inverse(E_A)`, and `T_AC=T_BC@T_AB`. Every endpoint is the real observed `z_C`; synthetic-nearest endpoints are excluded.

Conclusion boundary: these are controlled external-probe results, not native model dynamics, full SE(3) equivariance, counterfactual dual-image validation, or evidence that a pretraining objective causes an advantage. This run is a frozen re-analysis of the legacy discovery set, not an independent confirmation.
""")
    (output / "DATASET_CARD.md").write_text(f"""# SE3-Observed-v1 dataset card

Status: **legacy development / not independent confirmation**.

The only fully routed, cross-model, artifact-embedded target-isolated inventory contains 276 unique physical triples from 69 scenes. The frozen split has {dataset_audit['counts']['train']['tuples']}/{dataset_audit['counts']['val']['tuples']}/{dataset_audit['counts']['test']['tuples']} triples, {dataset_audit['counts']['train']['scenes']}/{dataset_audit['counts']['val']['scenes']}/{dataset_audit['counts']['test']['scenes']} scenes, and {dataset_audit['counts']['train']['unique_frames']}/{dataset_audit['counts']['val']['unique_frames']}/{dataset_audit['counts']['test']['unique_frames']} unique scene-frame keys for train/validation/development-test. All 69 scenes were already present in the r2 pilot, so none is called confirmation. Exact scene IDs are in `split.json`.

The manifest is `se3_observed_v1_manifest.jsonl` (SHA-256 `{dataset_audit['manifest_sha256']}`). It separates physical keys from model/context feature keys; records raw indices and names, all c2w/w2c/intrinsics and relative transforms, ordered repeated-frame inputs, temporal gaps, feature/source hashes and contracts, motion/non-commutativity, and observed overlap. The source metadata has no timestamp field, so timestamps are explicitly null rather than inferred from filenames.

Selection is the immutable four-model intersection of the historical C1/C2 p8 validation dumps. There are no duplicate physical triples. Synthetic-nearest endpoints are absent. Splits are scene-disjoint, and the nested development test is held out from this runner's probe training, but it is not scientifically untouched because its scores appeared in the earlier pilot.

The canonical-v2 feature preparation tree was inspected but not consumed: its handoff forbids probing until routing/merge completion. The nominal master test is also not independent of historical C training. Therefore no legal old-69-scenes-excluded confirmation set can be formed from the ready inventory. Requested 4,000/1,000/1,000 counts and 50 new test scenes are unsupported rather than fabricated.
""")
    dump_json(output / "inventory_audit.json", inventory)
    audit_rows = [asdict(spec) for spec in MODEL_SPECS]
    dump_json(output / "observation_contract_audit.json", {
        "gate": "PASS_FOR_TARGET_ISOLATED_CAUSAL_UPDATE_ONLY",
        "cross_model_observation_equivalence": False,
        "reason": "all cells isolate the target image, but repeat count and backbone computation differ",
        "models": audit_rows,
        "historical_unbound_fields": [
            "exact producing checkpoint revision for every historical artifact",
            "CogVideoX historical extraction noise seed",
        ],
    })
    model_lines = "\n".join(
        f"- **{spec.display_name}:** {spec.input_frames} input frame(s); {spec.ordered_input_semantics}. "
        f"Representation: {spec.feature_role}; pooling: {spec.pooling}; condition: {spec.diffusion_condition}."
        for spec in MODEL_SPECS
    )
    (output / "OBSERVATION_CONTRACT_AUDIT.md").write_text(f"""# Observation-contract audit

Gate: **PASS only for the target-isolated causal/update track**. Every state A/B/C is extracted without future endpoint content. The cells are not declared observationally equivalent: repeated-frame count, architecture, pooling, and diffusion computation differ.

{model_lines}

`p8` denotes tuple selection/cache naming, not eight-frame history. DINO consumes one image; the video backbones receive repetitions of that same isolated image. Hence no model sees B while encoding A or C while encoding A/B. Historical checkpoint revisions are not retroactively claimed where the producing artifacts did not bind them. CogVideoX scheduler semantics are recorded, while its historical noise seed remains unbound. Those gaps prohibit a stronger extraction-level causal comparison but do not introduce endpoint leakage into this frozen target-isolated analysis.
""")


def render_findings(output: Path, main: list[dict[str, Any]], bootstrap: list[dict[str, Any]]) -> None:
    full = {r["model"]: r for r in main if r["baseline"] == "full" and r["aggregation"] == "scene_macro"}
    ci = {(r["left_model"], r["metric"], r["right_baseline"]): r for r in bootstrap}
    lines = ["# Findings", "", "All statements below concern the frozen legacy-development split; none is an independent confirmation.", ""]
    for spec in MODEL_SPECS:
        row = full[spec.key]
        ground = ci[(spec.key, "grounding_gain_vs_copy", "copy")]
        state = "supported on this development set" if ground["ci95_low"] > 0 else "inconclusive"
        lines.append(f"- **{spec.display_name} grounding — {state}.** Scene-macro two-step gain versus copy is {row['two_step_gain_vs_copy_mean']:.4f}; paired scene-bootstrap CI [{ground['ci95_low']:.4f}, {ground['ci95_high']:.4f}]. See `paper_main_table.csv` and `paired_bootstrap.csv`.")
        action = ci[(spec.key, "normalized_margin_full_minus_action_only", "action_only")]
        shuffle = ci[(spec.key, "normalized_margin_full_minus_shuffle", "shuffled_pose")]
        matched = ci[(spec.key, "normalized_margin_full_minus_matched_wrong_pose", "matched_wrong_pose")]
        additive = ci[(spec.key, "normalized_margin_full_minus_additive", "additive")]
        controls = (action, shuffle, matched, additive)
        order_state = "supported" if all(float(x["estimate"]) > 0 and float(x["holm_adjusted_p"]) < .05 for x in controls) else "inconclusive"
        lines.append(f"- **{spec.display_name} control-adjusted order — {order_state}.** Full normalized margin is {row['normalized_order_margin_mean']:.4f}, tie-aware accuracy {row['tie_aware_accuracy_mean']:.4f}; paired effects versus action-only/shuffle/matched-wrong/additive have CIs [{action['ci95_low']:.4f},{action['ci95_high']:.4f}], [{shuffle['ci95_low']:.4f},{shuffle['ci95_high']:.4f}], [{matched['ci95_low']:.4f},{matched['ci95_high']:.4f}], and [{additive['ci95_low']:.4f},{additive['ci95_high']:.4f}]. Registered claims require positive direction and Holm-adjusted p<.05 for all controls.")
    lines.extend([
        "", "The feature-distance and M3/M4 tables are descriptive explanations, not alternate endpoint ground truth. M4 is calibrated only on held-out real feature pairs; no decoded pose is treated as independent GT.",
        "", "The M3/M4 labels follow the SeeSE3 readout framing (https://arxiv.org/html/2607.14228v1), but these are protocol-specific readouts and are not claimed as a paper reproduction.",
        "", "Layer, true-context, diffusion-condition, and forward 1/2/4-step curves remain unavailable under a safe matched artifact contract. Prefix-labelled target-isolated caches do not vary genuine history length, and retracing is not relabelled as forward rollout.",
    ])
    capacity_path = output / "capacity_control_summary.csv"
    if capacity_path.is_file():
        capacity = list(csv.DictReader(capacity_path.open()))
        for spec in MODEL_SPECS:
            selected = [r for r in capacity if r["model"] == spec.key and r["baseline"] == "full" and r["aggregation"] == "scene_macro"]
            if selected:
                row = selected[0]
                lines.append(f"- **{spec.display_name} PCA-128 capacity control.** Train-only frozen PCA gives two-step grounding gain {float(row['two_step_gain_vs_copy_mean']):.4f} and normalized order margin {float(row['normalized_order_margin_mean']):.4f}; this is a capacity sensitivity check, not a replacement primary cell.")
    (output / "FINDINGS.md").write_text("\n".join(lines) + "\n")


def write_paper_section(output: Path, main: list[dict[str, Any]]) -> None:
    full = [r for r in main if r["baseline"] == "full" and r["aggregation"] == "scene_macro"]
    result_sentences = " ".join(
        f"{r['display_name']} obtained one-step/two-step cosine error {r['one_step_error_mean']:.4f}/{r['recursive_two_step_error_mean']:.4f}, copy-normalized order margin {r['normalized_order_margin_mean']:.4f}, and tie-aware order accuracy {r['tie_aware_accuracy_mean']:.4f}."
        for r in full
    )
    (output / "paper_section.tex").write_text(r"""\section{Observed-order latent updates under camera motion}
\paragraph{Problem.} We ask whether frozen visual features permit a controlled external probe to predict real observed feature updates under camera motion and whether recursive updates distinguish the observed action order from its reversal. This does not test native model dynamics or full equivariance.

\paragraph{Dataset and protocol.} We use 276 observed triples from 69 ScanNet++ scenes with scene-disjoint 192/40/44 train/validation/development-test triples. Every target is an observed target-isolated feature. Because all scenes appeared in an earlier pilot, this is a frozen development re-analysis rather than independent confirmation. We use world-to-camera extrinsics and compose $T_{AC}=T_{BC}T_{AB}$.

\paragraph{Methods and controls.} Frozen features feed a width-64 residual predictor trained solely on real single-transition pairs. Controls are copy, no-action, action-only, matched wrong pose, shuffled pose, and a commutative $z+g(\log T)$ update. Five training seeds are averaged. Scene-macro estimates use paired 2,000-resample scene bootstrap and Holm correction for registered comparisons.

\paragraph{Results.} """ + result_sentences + r""" Full control-adjusted inferences and confidence intervals are reported in Table~\ref{tab:se3-observed} and the paired-comparison supplement; confidence intervals containing zero are described as inconclusive.

\paragraph{Analysis.} We separately report raw-feature M3/M4 pose readout, feature-distance geometry, recursive versus direct endpoint error, and order margin against rotation and camera-center non-commutativity. Decoded pose is auxiliary and never replaces observed endpoint grounding.

\paragraph{Limitations.} The ready cross-model inventory is small and previously inspected. Historical diffusion artifacts do not bind an extraction noise seed or exact producing checkpoint revision. No safe matched layer sweep, genuine context-length intervention, diffusion-condition sweep, independent confirmation set, or real four-step forward rollout is available. Results support only model-cell-specific external-probe statements and no causal claims about pretraining objectives.
""")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--bootstrap-seed", type=int, default=20260913)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--max-epochs", type=int, default=180)
    parser.add_argument("--pose-max-epochs", type=int, default=240)
    parser.add_argument("--model-child", choices=[spec.key for spec in MODEL_SPECS], help=argparse.SUPPRESS)
    parser.add_argument("--aggregate-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = resolve(args.output_dir)
    seeds = parse_seeds(args.seeds)
    if output.exists() and not args.model_child and not args.aggregate_only:
        raise SystemExit(f"refusing to overwrite output: {output}")
    output.mkdir(parents=True, exist_ok=args.model_child is not None or args.aggregate_only)
    torch.set_num_threads(4)
    if args.model_child:
        run_child(output, next(spec for spec in MODEL_SPECS if spec.key == args.model_child), seeds, args.split_seed, args.max_epochs, args.pose_max_epochs)
        return
    random.seed(args.split_seed); np.random.seed(args.split_seed); torch.manual_seed(args.split_seed)
    if not args.aggregate_only:
        for spec in MODEL_SPECS:
            command = [sys.executable, "-m", "analyses.se3_protocol_v2.paper_v1", "--output-dir", str(output), "--seeds", args.seeds, "--split-seed", str(args.split_seed), "--max-epochs", str(args.max_epochs), "--pose-max-epochs", str(args.pose_max_epochs), "--model-child", spec.key]
            with (output / f"{spec.key}_child.stdout.log").open("w") as stdout, (output / f"{spec.key}_child.stderr.log").open("w") as stderr:
                subprocess.run(command, cwd=REPO_ROOT, stdout=stdout, stderr=stderr, check=True)

    child_metrics, child_samples, capacity_metrics, training, readouts, distances, geometry = [], [], [], {}, [], [], []
    reference_assignments = None
    for spec in MODEL_SPECS:
        child = output / f"model_{spec.key}"
        metrics = list(csv.DictReader((child / "per_seed_metrics.csv").open()))
        samples = list(csv.DictReader((child / "per_sample_metrics.csv").open()))
        for row in metrics: row.update({"model": spec.key, "display_name": spec.display_name})
        for row in samples: row["model"] = spec.key
        child_metrics.extend(metrics); child_samples.extend(samples)
        capacity_path = child / "capacity_control_seed_metrics.csv"
        if capacity_path.is_file():
            for row in csv.DictReader(capacity_path.open()):
                capacity_metrics.append({"model": spec.key, "display_name": spec.display_name, **row})
        training[spec.key] = read_json(child / "training.json")
        for row in csv.DictReader((child / "readout_metrics.csv").open()): readouts.append({"model": spec.key, **row})
        distances.append({"model": spec.key, **read_json(child / "feature_distance.json")})
        if spec.key == "dino": geometry = list(csv.DictReader((child / "test_geometry.csv").open()))
        assignments = read_json(child / "split.json")["assignments"]
        if reference_assignments is None: reference_assignments = assignments
        elif assignments != reference_assignments: raise RuntimeError("model split mismatch")
    assert reference_assignments is not None
    records, load_audit = load_records(MODEL_SPECS[0])
    manifest_rows, dataset_audit = build_frozen_manifest(output, records, reference_assignments, args.split_seed)
    motion_rows, motion_summary = motion_support_audit(records, reference_assignments)
    aggregate = mean_rows(child_metrics, ("model", "display_name", "baseline", "aggregation"))
    test_scenes = sorted({r.scene_id for r in records if reference_assignments[r.scene_id] == "test"})
    bootstrap = paired_bootstrap(child_samples, args.bootstrap_resamples, args.bootstrap_seed, seeds, test_scenes)
    paper_rows = paper_main_rows(aggregate, bootstrap, dataset_audit)
    write_csv(output / "all_baseline_seed_metrics.csv", child_metrics)
    write_csv(output / "raw_per_sample_metrics.csv", child_samples)
    write_csv(output / "aggregate_baselines.csv", aggregate)
    write_csv(output / "paper_main_table.csv", paper_rows)
    if capacity_metrics:
        write_csv(output / "capacity_control_seed_metrics.csv", capacity_metrics)
        write_csv(output / "capacity_control_summary.csv", mean_rows(
            capacity_metrics, ("model", "display_name", "baseline", "aggregation", "representation")
        ))
    write_csv(output / "paired_bootstrap.csv", bootstrap)
    write_csv(output / "readout_m3_m4.csv", readouts)
    write_csv(output / "feature_distance.csv", distances)
    write_csv(output / "test_geometry.csv", geometry)
    write_csv(output / "order_geometry_associations.csv", order_geometry_associations(child_samples, geometry))
    write_csv(output / "composite_motion_support.csv", motion_rows)
    dump_json(output / "composite_motion_support_summary.json", motion_summary)
    (output / "paper_main_table.tex").write_text(latex_table(paper_rows))
    write_model_card(output, training)
    inventory = {
        "ready_cross_model_inventory": dataset_audit, "load_alignment_audit": load_audit,
        "legacy_pilot_scenes": 69, "new_confirmation_scenes": 0,
        "confirmation_status": "UNAVAILABLE_no_ready_old69_excluded_cross_model_artifacts",
        "canonical_v2_status": "not_consumed_incomplete_and_probing_forbidden_by_handoff",
        "videoMAE_extension": "available only on same legacy 276 tuples; omitted from four-model core",
        "qwen_extension": "available only on same legacy 276 tuples; optional and omitted",
        "wan_p8": "excluded_known_unsafe_lineage",
    }
    write_study_documents(output, dataset_audit, inventory)
    render_findings(output, aggregate, bootstrap)
    write_paper_section(output, aggregate)
    render_figures(output, aggregate, paper_rows, child_samples, geometry, distances, seeds)
    canonical_build = (
        f"{sys.executable} -m analyses.se3_protocol_v2.paper_v1 --output-dir <NEW_OUTPUT_DIR> "
        f"--seeds {args.seeds} --split-seed {args.split_seed} --bootstrap-seed {args.bootstrap_seed} "
        f"--bootstrap-resamples {args.bootstrap_resamples} --max-epochs {args.max_epochs} --pose-max-epochs {args.pose_max_epochs}"
    )
    config = {"invocation_argv": sys.argv, "canonical_build_command": canonical_build, "runner_sha256": sha256(Path(__file__).resolve()), "seeds": list(seeds), "split_seed": args.split_seed, "bootstrap_seed": args.bootstrap_seed, "bootstrap_resamples": args.bootstrap_resamples, "tie_tolerance": DEFAULT_TIE_TOLERANCE, "normalization_epsilon": DEFAULT_NORMALIZATION_EPSILON, "models": [asdict(spec) for spec in MODEL_SPECS], "pose_convention": POSE_CONVENTION}
    dump_json(output / "config.json", config)
    dump_json(output / "execution_summary.json", {
        "status": "COMPLETE_LEGACY_DEVELOPMENT_CORE_CONFIRMATION_UNAVAILABLE", "models": [spec.key for spec in MODEL_SPECS],
        "seeds": list(seeds), "tuples": len(manifest_rows), "scenes": len({r["scene_id"] for r in manifest_rows}),
        "test_tuples": dataset_audit["counts"]["test"]["tuples"], "test_scenes": dataset_audit["counts"]["test"]["scenes"],
        "tests_command": f"{sys.executable} -m pytest -q tests/test_se3_protocol_v2.py tests/test_se3_latent_mvp.py",
        "build_command": canonical_build, "last_aggregation_invocation": " ".join(sys.argv), "runner_sha256": sha256(Path(__file__).resolve()), "limitations": ["no independent confirmation inventory", "historical CogVideoX noise seed unbound", "no safe matched layer/context/diffusion sweep", "no real four-step forward rollout"],
    })
    (output / "REPRODUCE.md").write_text(f"""# Reproduce

```bash
export MPLCONFIGDIR=/tmp/probe_spatial_mpl
{sys.executable} -m pytest -q tests/test_se3_protocol_v2.py tests/test_se3_latent_mvp.py
{sys.executable} -m analyses.se3_protocol_v2.paper_v1 --output-dir results/experimental/se3_observed_paper_v1_REPRO --seeds 42,43,44,45,46 --bootstrap-resamples 2000
```

The runner refuses to overwrite an existing top-level output directory. It reads only the paths and hashes listed in each child `provenance.json`.
""")
    (output / "REPORT.md").write_text("""# SE3-Observed-v1 paper-module report

The four-model, five-seed legacy-development module completed. Use `paper_main_table.csv` for the compact primary result, `aggregate_baselines.csv` for every baseline and aggregation, `paired_bootstrap.csv` for paired scene-bootstrap inference, and `FINDINGS.md` for conclusion-bounded interpretation.

The additive control is numerically enforced to be exactly order-insensitive. DINOv2-L and CogVideoX additionally have a train-only frozen PCA-128 capacity control. Observation-contract and dataset audits are explicit. Independent confirmation is unavailable because no ready four-model artifact inventory excludes all 69 historically used scenes; canonical-v2 artifacts were not consumed while their handoff gate remains closed.
""")
    print(json.dumps(read_json(output / "execution_summary.json"), indent=2))


if __name__ == "__main__":
    main()
