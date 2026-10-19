#!/usr/bin/env python3
"""Registered P1 follow-up for the frozen Wan/V-JEPA2 exact-v2 experiment.

This module never trains or modifies the parent visual heads.  It reuses their
predictions/checkpoints, adds identity-padding inference, reconstructs metric
SE(3) geometry, and trains the separately labelled analytic positive control.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors.torch import load_file

from analyses.se3_latent_mvp.geometry import (
    camera_to_world_to_extrinsic,
    encode_relative_pose_9d,
    relative_pose,
    so3_log,
    transform_points,
)
from analyses.se3_protocol_v2.observed_order import observed_overlap
from analyses.se3_protocol_v2.paper_v1 import holm_adjust
from analyses.se3_protocol_v2.wan_vjepa2_v2_data import LOCAL_SCENES, read_jsonl
from analyses.se3_protocol_v2.wan_vjepa2_v2_train import (
    AdditiveUpdater,
    BASELINES,
    _path_model,
    matched_donors,
    train_head,
    write_csv,
)


REPO = Path(__file__).resolve().parents[2]
PARENT = REPO / "results/experimental/se3_wan_vjepa2_exact_v2_layers_20260916_r1"
RESULT = REPO / "results/experimental/se3_wan_vjepa2_exact_v2_p1_20260917_r1"
PROTOCOL_PATH = RESULT / "P1_PROTOCOL.json"
CELLS = (
    "wan_l0", "wan_l5", "wan_l10", "wan_l15", "wan_l20", "wan_l25", "wan_l29",
    "vjepa2_l0", "vjepa2_l5", "vjepa2_l11", "vjepa2_l17", "vjepa2_l23",
)
SEEDS = (42, 43, 44, 45, 46)
REGISTERED_CONTROLS = ("action_only", "matched_shuffle", "commutative_additive")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_frozen_inputs() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    parent_protocol = json.loads((PARENT / "FROZEN_PROTOCOL.json").read_text())
    bindings = {
        "parent_protocol_sha256": sha256(PARENT / "FROZEN_PROTOCOL.json"),
        "common_manifest_sha256": sha256(PARENT / "data/common_manifest.jsonl"),
        "parent_geometry_sha256": sha256(PARENT / "cache/geometry.npz"),
    }
    for key, observed in bindings.items():
        if protocol[key] != observed:
            raise RuntimeError(f"frozen binding changed for {key}: {observed}")
    rows = read_jsonl(PARENT / "data/common_manifest.jsonl")
    counts = {split: sum(row["analysis_split"] == split for row in rows) for split in ("train", "validation", "evaluation")}
    if counts != {"train": 128, "validation": 32, "evaluation": 276}:
        raise RuntimeError(f"frozen split changed: {counts}")
    for cell in CELLS:
        for seed in SEEDS:
            complete = PARENT / "cells" / cell / "seeds" / f"seed{seed}" / "COMPLETE.json"
            if not complete.exists() or json.loads(complete.read_text()).get("status") != "PASS":
                raise RuntimeError(f"missing parent completion: {cell}/seed{seed}")
    (RESULT / "data").mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "PASS_FROZEN_PARENT_REUSED",
        **bindings,
        "split_counts": counts,
        "cells": list(CELLS),
        "seeds": list(SEEDS),
        "parent_visual_training_rerun": False,
        "feature_extraction_rerun": False,
    }
    (RESULT / "SOURCE_BINDINGS.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return protocol, parent_protocol, rows


def cell_parts(cell: str) -> tuple[str, int]:
    model, layer = cell.rsplit("_l", 1)
    return model, int(layer)


def frame_number(value: str) -> int:
    found = re.search(r"(\d+)", Path(value).stem)
    if not found:
        raise ValueError(f"no numeric frame id: {value}")
    return int(found.group(1))


def camera_center(transform: np.ndarray) -> np.ndarray:
    return -transform[:3, :3].T @ transform[:3, 3]


def physical_geometry(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    metadata: dict[str, Any] = {}
    overlap_cache: dict[str, Any] = {}
    table: list[dict[str, Any]] = []
    t1_all, t2_all = [], []
    for row in rows:
        scene = row["scene_id"]
        if scene not in metadata:
            data = np.load(LOCAL_SCENES / scene / "scene_iphone_metadata.npz")
            metadata[scene] = data["trajectories"]
        trajectories = metadata[scene]
        positions = row["abc_frame_positions"]
        pose_indices = row["valid_pose_indices"]
        ia, ib, ic = (int(pose_indices[int(positions[key])]) for key in ("A", "B", "C"))
        ea, eb, ec = (camera_to_world_to_extrinsic(trajectories[index]) for index in (ia, ib, ic))
        t1, t2 = relative_pose(ea, eb), relative_pose(eb, ec)
        correct, swapped = t2 @ t1, t1 @ t2
        rot = float(np.linalg.norm(so3_log(swapped[:3, :3] @ correct[:3, :3].T)))
        center = float(np.linalg.norm(camera_center(swapped) - camera_center(correct)))
        path_rot = float(np.linalg.norm(so3_log(t1[:3, :3])) + np.linalg.norm(so3_log(t2[:3, :3])))
        path_trans = float(np.linalg.norm(camera_center(t1)) + np.linalg.norm(camera_center(t2)))
        record = SimpleNamespace(scene_id=scene, raw_frame_indices=(ia, ib, ic))
        overlap = observed_overlap(record, LOCAL_SCENES, overlap_cache)
        table.append({
            "sample_id": row["sample_id"], "scene_id": scene, "analysis_split": row["analysis_split"],
            "frame_A": row["abc_frame_ids"]["A"], "frame_B": row["abc_frame_ids"]["B"],
            "frame_C": row["abc_frame_ids"]["C"],
            "order_rotation_difference_rad": rot,
            "order_camera_center_difference_m": center,
            "path_rotation_magnitude_rad": path_rot,
            "path_translation_magnitude_m": path_trans,
            "frame_id_interval": frame_number(row["abc_frame_ids"]["C"]) - frame_number(row["abc_frame_ids"]["A"]),
            "view_overlap_AC": overlap,
        })
        t1_all.append(t1); t2_all.append(t2)
    arrays = {"t1": np.stack(t1_all), "t2": np.stack(t2_all)}
    return table, arrays


def freeze_bins(geometry: list[dict[str, Any]], fields: list[str]) -> dict[str, Any]:
    train = [row for row in geometry if row["analysis_split"] == "train"]
    rules: dict[str, Any] = {
        "source": "frozen_train_split_only", "quantiles": [1 / 3, 2 / 3], "labels": ["low", "medium", "high"],
    }
    for field in fields:
        values = np.asarray([float(row[field]) for row in train])
        q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
        rules[field] = {"q1": float(q1), "q2": float(q2), "train_n": len(values)}
    return rules


def bin_value(value: float, rule: dict[str, float]) -> str:
    return "low" if value <= rule["q1"] else ("medium" if value <= rule["q2"] else "high")


def prepare_geometry() -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    protocol, _, rows = verify_frozen_inputs()
    geometry, transforms = physical_geometry(rows)
    fields = protocol["order_geometry"]["stratifiers"]
    rules = freeze_bins(geometry, fields)
    for row in geometry:
        for field in fields:
            row[f"{field}_bin"] = bin_value(float(row[field]), rules[field])
    write_csv(RESULT / "data/physical_geometry.csv", geometry)
    np.savez_compressed(RESULT / "data/physical_transforms.npz", sample_id=np.asarray([r["sample_id"] for r in rows]), **transforms)
    (RESULT / "data/TRAIN_BIN_RULES.json").write_text(json.dumps(rules, indent=2, sort_keys=True) + "\n")
    return geometry, transforms, rules


def cosine_np(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left, right = left.astype(np.float32), right.astype(np.float32)
    numerator = np.sum(left * right, axis=-1)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    return np.maximum(0.0, 1.0 - numerator / np.maximum(denominator, 1e-12)).astype(np.float64)


def load_parent_samples() -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for cell in CELLS:
        model, layer = cell_parts(cell)
        for seed in SEEDS:
            for row in read_csv(PARENT / "cells" / cell / "seeds" / f"seed{seed}" / "per_sample_metrics.csv"):
                converted: dict[str, Any] = dict(row)
                converted.update({"model": model, "layer": layer, "seed": int(row["seed"]), "window_id": int(row["window_id"])})
                for key in (
                    "one_step_error", "sequential_endpoint_error", "direct_endpoint_error",
                    "sequential_direct_prediction_distance", "recursive_minus_direct_endpoint_error",
                    "swapped_endpoint_error", "raw_order_margin", "tie_score", "noncomm_strength",
                ):
                    converted[key] = float(row[key])
                output.append(converted)
    return output


def model_from_checkpoint(checkpoint: dict[str, Any], parent_cfg: dict[str, Any]) -> torch.nn.Module:
    kind, channels = checkpoint["head"], int(checkpoint["channels"])
    model = AdditiveUpdater(channels, int(parent_cfg["hidden_dim"])) if kind == "commutative_additive" else _path_model(
        channels, {"without_condition": "visual_only"}.get(kind, kind), parent_cfg
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def identity_inference(device_name: str = "cpu") -> list[dict[str, Any]]:
    _, parent_protocol, rows = verify_frozen_inputs()
    parent_geometry = dict(np.load(PARENT / "cache/geometry.npz", allow_pickle=True))
    split = np.asarray([row["analysis_split"] for row in rows])
    eval_i = np.flatnonzero(split == "evaluation")
    train_i = np.flatnonzero(split == "train")
    eval_rows = [rows[index] for index in eval_i]
    donors = matched_donors(eval_rows)
    identity_raw = torch.from_numpy(encode_relative_pose_9d(np.eye(4)).astype(np.float32))
    output: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    device = torch.device(device_name)
    for cell in CELLS:
        cache = {key: value.float() for key, value in load_file(str(PARENT / "cache" / f"{cell}.sft"), device="cpu").items()}
        model_name, layer = cell_parts(cell)
        for seed in SEEDS:
            seed_dir = PARENT / "cells" / cell / "seeds" / f"seed{seed}"
            raw_saved = np.load(seed_dir / "raw_predictions.npz")
            full_ckpt = torch.load(seed_dir / "checkpoints/full.pt", map_location="cpu", weights_only=False)
            norm = full_ckpt["normalization"]
            context = ((cache["context"] - norm["context_mean"]) / norm["context_std"])[eval_i].to(device)
            za = ((cache["z_a"] - norm["target_B_mean"]) / norm["target_B_std"])[eval_i].to(device)
            zc = cache["z_c"][eval_i].numpy()
            action_mean, action_std = norm["action_AB_mean"], norm["action_AB_std"]
            identity = ((identity_raw - action_mean) / action_std).expand(len(eval_i), -1).to(device)
            train_actions = torch.from_numpy(np.asarray(parent_geometry["action1"], dtype=np.float32))[train_i]
            standard_train = (train_actions - action_mean) / action_std
            identity_standard = (identity_raw - action_mean) / action_std
            if seed == SEEDS[0]:
                support_rows.append({
                    "cell": cell,
                    "identity_exact_training_match": bool(torch.any(torch.all(torch.isclose(train_actions, identity_raw, atol=1e-8, rtol=0), dim=1))),
                    "identity_coordinatewise_training_range_check": bool(torch.all((identity_raw >= train_actions.min(0).values) & (identity_raw <= train_actions.max(0).values))),
                    "identity_nearest_standardized_training_action_l2": float(torch.linalg.vector_norm(standard_train - identity_standard, dim=1).min()),
                    "range_check_is_not_strict_in_distribution_certification": True,
                })
            a1_raw = torch.from_numpy(np.asarray(parent_geometry["action1"], dtype=np.float32))[eval_i]
            a2_raw = torch.from_numpy(np.asarray(parent_geometry["action2"], dtype=np.float32))[eval_i]
            comp_raw = torch.from_numpy(np.stack([
                encode_relative_pose_9d(right @ left).astype(np.float32)
                for left, right in zip(np.asarray(parent_geometry["t1"])[eval_i], np.asarray(parent_geometry["t2"])[eval_i])
            ]))
            for baseline in BASELINES:
                checkpoint_kind = "full" if baseline == "matched_shuffle" else baseline
                if baseline == "copy":
                    direct_n = pre_n = post_n = za
                else:
                    ckpt = torch.load(seed_dir / "checkpoints" / f"{checkpoint_kind}.pt", map_location="cpu", weights_only=False)
                    model = model_from_checkpoint(ckpt, parent_protocol["training"]).to(device)
                    if baseline == "matched_shuffle":
                        use_comp = comp_raw[donors]
                    else:
                        use_comp = comp_raw
                    composite = ((use_comp - action_mean) / action_std).to(device)
                    with torch.no_grad():
                        if baseline == "commutative_additive":
                            direct_n = model.update(za, composite)
                            pre_n = model.update(model.update(za, identity), composite)
                            post_n = model.update(model.update(za, composite), identity)
                        else:
                            state = model.initialize_state(context[:, None, None, None, :])
                            direct_state = model.update_state(state, composite)
                            direct_n = model.decode_state(direct_state)
                            pre_n = model.decode_state(model.update_state(model.update_state(state, identity), composite))
                            post_n = model.decode_state(model.update_state(direct_state, identity))
                mean = norm["target_B_mean"].to(device); std = norm["target_B_std"].to(device)
                direct = (direct_n * std + mean).detach().cpu().numpy()
                pre = (pre_n * std + mean).detach().cpu().numpy()
                post = (post_n * std + mean).detach().cpu().numpy()
                stored_direct = raw_saved[f"{baseline}__direct"].astype(np.float32)
                audit = float(np.max(np.abs(stored_direct - direct)))
                for index, row in enumerate(eval_rows):
                    output.append({
                        "cell": cell, "model": model_name, "layer": layer, "seed": seed,
                        "baseline": baseline, "sample_id": row["sample_id"], "scene_id": row["scene_id"],
                        "direct_composite_endpoint_error": float(cosine_np(direct[index:index+1], zc[index:index+1])[0]),
                        "identity_then_composite_endpoint_error": float(cosine_np(pre[index:index+1], zc[index:index+1])[0]),
                        "composite_then_identity_endpoint_error": float(cosine_np(post[index:index+1], zc[index:index+1])[0]),
                        "identity_then_composite_vs_direct_distance": float(cosine_np(pre[index:index+1], direct[index:index+1])[0]),
                        "composite_then_identity_vs_direct_distance": float(cosine_np(post[index:index+1], direct[index:index+1])[0]),
                        "recomputed_direct_vs_parent_float16_max_abs_cell_seed_baseline": audit,
                    })
    write_csv(RESULT / "composition/identity_padding_per_sample.csv", output)
    write_csv(RESULT / "composition/identity_action_support.csv", support_rows)
    return output


def scene_seed_array(rows: list[dict[str, Any]], cell: str, baseline: str, field: str, scenes: list[str]) -> np.ndarray:
    values: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["cell"] == cell and row["baseline"] == baseline:
            values[(int(row["seed"]), row["scene_id"])].append(float(row[field]))
    result = np.empty((len(SEEDS), len(scenes)), dtype=np.float64)
    for si, seed in enumerate(SEEDS):
        for ci, scene in enumerate(scenes):
            found = values[(seed, scene)]
            if not found:
                raise RuntimeError(f"missing {cell}/{seed}/{baseline}/{scene}/{field}")
            result[si, ci] = np.mean(found)
    return result


def bootstrap_distribution(per_seed_scene: np.ndarray, draws: np.ndarray) -> np.ndarray:
    return per_seed_scene.mean(0)[draws].mean(1)


def ratio_distribution(
    numerator: np.ndarray, denominator: np.ndarray, draws: np.ndarray, epsilon: float,
) -> np.ndarray:
    """Ratio of scene-macro aggregates, never a mean of per-scene ratios."""
    return bootstrap_distribution(numerator, draws) / (
        bootstrap_distribution(denominator, draws) + epsilon
    )


def interval(distribution: np.ndarray) -> tuple[float, float, float, float]:
    p = min(1.0, 2 * min(float(np.mean(distribution <= 0)), float(np.mean(distribution >= 0))))
    return float(distribution.mean()), float(np.quantile(distribution, .025)), float(np.quantile(distribution, .975)), p


def composition_analysis(parent_rows: list[dict[str, Any]], identity_rows: list[dict[str, Any]], protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    epsilon = float(protocol["statistics"]["normalization_epsilon"])
    repetitions = int(protocol["statistics"]["bootstrap_replicates"])
    rng = np.random.default_rng(int(protocol["statistics"]["bootstrap_seed"]))
    all_scenes = sorted({row["scene_id"] for row in parent_rows})
    draws = rng.integers(0, len(all_scenes), size=(repetitions, len(all_scenes)))
    summary, ci_rows, seed_summary = [], [], []
    for cell in CELLS:
        model, layer = cell_parts(cell)
        copy_seq = scene_seed_array(parent_rows, cell, "copy", "sequential_endpoint_error", all_scenes)
        no_seq = scene_seed_array(parent_rows, cell, "without_condition", "sequential_endpoint_error", all_scenes)
        copy_dir = scene_seed_array(parent_rows, cell, "copy", "direct_endpoint_error", all_scenes)
        no_dir = scene_seed_array(parent_rows, cell, "without_condition", "direct_endpoint_error", all_scenes)
        for baseline in BASELINES:
            seq = scene_seed_array(parent_rows, cell, baseline, "sequential_endpoint_error", all_scenes)
            direct = scene_seed_array(parent_rows, cell, baseline, "direct_endpoint_error", all_scenes)
            pred_dist = scene_seed_array(parent_rows, cell, baseline, "sequential_direct_prediction_distance", all_scenes)
            metrics: dict[str, tuple[float, np.ndarray]] = {
                "sequential_endpoint_error": (float(seq.mean()), bootstrap_distribution(seq, draws)),
                "direct_endpoint_error": (float(direct.mean()), bootstrap_distribution(direct, draws)),
                "sequential_gain_vs_copy": (
                    float((copy_seq.mean() - seq.mean()) / (copy_seq.mean() + epsilon)),
                    ratio_distribution(copy_seq - seq, copy_seq, draws, epsilon),
                ),
                "direct_gain_vs_copy": (
                    float((copy_dir.mean() - direct.mean()) / (copy_dir.mean() + epsilon)),
                    ratio_distribution(copy_dir - direct, copy_dir, draws, epsilon),
                ),
                "sequential_gain_vs_without_condition": (
                    float((no_seq.mean() - seq.mean()) / (no_seq.mean() + epsilon)),
                    ratio_distribution(no_seq - seq, no_seq, draws, epsilon),
                ),
                "direct_gain_vs_without_condition": (
                    float((no_dir.mean() - direct.mean()) / (no_dir.mean() + epsilon)),
                    ratio_distribution(no_dir - direct, no_dir, draws, epsilon),
                ),
                "sequential_direct_prediction_distance": (
                    float(pred_dist.mean()), bootstrap_distribution(pred_dist, draws),
                ),
                "normalized_sequential_direct_distance": (
                    float(pred_dist.mean() / (copy_seq.mean() + epsilon)),
                    ratio_distribution(pred_dist, copy_seq, draws, epsilon),
                ),
                "recursive_minus_direct_endpoint_gap": (
                    float((seq - direct).mean()), bootstrap_distribution(seq - direct, draws),
                ),
            }
            row: dict[str, Any] = {"cell": cell, "model": model, "layer": layer, "baseline": baseline, "seeds": len(SEEDS), "evaluation_scenes": len(all_scenes)}
            for name, (estimate_exact, dist) in metrics.items():
                row[name] = estimate_exact
                estimate, low, high, p = interval(dist)
                ci_rows.append({"cell": cell, "model": model, "layer": layer, "baseline": baseline, "metric": name, "estimate": estimate_exact, "bootstrap_distribution_mean": estimate, "ci95_low": low, "ci95_high": high, "bootstrap_p_two_sided": p, "resamples": repetitions})
            summary.append(row)
            for seed_index, seed in enumerate(SEEDS):
                seed_summary.append({
                    "cell": cell, "model": model, "layer": layer, "baseline": baseline, "seed": seed,
                    "sequential_endpoint_error": float(seq[seed_index].mean()),
                    "direct_endpoint_error": float(direct[seed_index].mean()),
                    "sequential_gain_vs_copy": float((copy_seq[seed_index].mean() - seq[seed_index].mean()) / (copy_seq[seed_index].mean() + epsilon)),
                    "direct_gain_vs_copy": float((copy_dir[seed_index].mean() - direct[seed_index].mean()) / (copy_dir[seed_index].mean() + epsilon)),
                    "sequential_gain_vs_without_condition": float((no_seq[seed_index].mean() - seq[seed_index].mean()) / (no_seq[seed_index].mean() + epsilon)),
                    "direct_gain_vs_without_condition": float((no_dir[seed_index].mean() - direct[seed_index].mean()) / (no_dir[seed_index].mean() + epsilon)),
                    "sequential_direct_prediction_distance": float(pred_dist[seed_index].mean()),
                    "normalized_sequential_direct_distance": float(pred_dist[seed_index].mean() / (copy_seq[seed_index].mean() + epsilon)),
                    "recursive_minus_direct_endpoint_gap": float((seq[seed_index] - direct[seed_index]).mean()),
                })
    write_csv(RESULT / "composition/layer_baseline_summary.csv", summary)
    write_csv(RESULT / "composition/paired_scene_bootstrap.csv", ci_rows)
    write_csv(RESULT / "composition/per_seed_metrics.csv", seed_summary)

    identity_summary: list[dict[str, Any]] = []
    identity_ci: list[dict[str, Any]] = []
    identity_seed: list[dict[str, Any]] = []
    for cell in CELLS:
        model, layer = cell_parts(cell)
        for baseline in BASELINES:
            direct = scene_seed_array(identity_rows, cell, baseline, "direct_composite_endpoint_error", all_scenes)
            pre = scene_seed_array(identity_rows, cell, baseline, "identity_then_composite_endpoint_error", all_scenes)
            post = scene_seed_array(identity_rows, cell, baseline, "composite_then_identity_endpoint_error", all_scenes)
            pre_dist = scene_seed_array(identity_rows, cell, baseline, "identity_then_composite_vs_direct_distance", all_scenes)
            post_dist = scene_seed_array(identity_rows, cell, baseline, "composite_then_identity_vs_direct_distance", all_scenes)
            seq = scene_seed_array(parent_rows, cell, baseline, "sequential_endpoint_error", all_scenes)
            item = {"cell": cell, "model": model, "layer": layer, "baseline": baseline}
            metrics = {
                "identity_pre_minus_direct_endpoint_error": pre - direct,
                "identity_post_minus_direct_endpoint_error": post - direct,
                "sequential_minus_direct_endpoint_error": seq - direct,
                "sequential_minus_identity_pre_endpoint_error": seq - pre,
                "sequential_minus_identity_post_endpoint_error": seq - post,
                "identity_pre_vs_direct_prediction_distance": pre_dist,
                "identity_post_vs_direct_prediction_distance": post_dist,
            }
            for name, matrix in metrics.items():
                item[name] = float(matrix.mean())
                estimate, low, high, p = interval(bootstrap_distribution(matrix, draws))
                identity_ci.append({"cell": cell, "model": model, "layer": layer, "baseline": baseline, "metric": name, "estimate": estimate, "ci95_low": low, "ci95_high": high, "bootstrap_p_two_sided": p, "resamples": repetitions})
            identity_summary.append(item)
            for seed_index, seed in enumerate(SEEDS):
                identity_seed.append({
                    "cell": cell, "model": model, "layer": layer, "baseline": baseline, "seed": seed,
                    **{name: float(matrix[seed_index].mean()) for name, matrix in metrics.items()},
                })
    write_csv(RESULT / "composition/identity_padding_summary.csv", identity_summary)
    write_csv(RESULT / "composition/identity_padding_bootstrap.csv", identity_ci)
    write_csv(RESULT / "composition/identity_padding_per_seed.csv", identity_seed)
    return summary, ci_rows


def subset_scene_seed_array(rows: list[dict[str, Any]], cell: str, baseline: str, field: str, scenes: list[str], allowed: set[str]) -> np.ndarray:
    return scene_seed_array([row for row in rows if row["sample_id"] in allowed], cell, baseline, field, scenes)


def order_analysis(parent_rows: list[dict[str, Any]], geometry: list[dict[str, Any]], rules: dict[str, Any], protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    geometry_by_id = {row["sample_id"]: row for row in geometry if row["analysis_split"] == "evaluation"}
    epsilon = float(protocol["statistics"]["normalization_epsilon"])
    reps = int(protocol["statistics"]["bootstrap_replicates"])
    base_seed = int(protocol["statistics"]["bootstrap_seed"])
    stratified: list[dict[str, Any]] = []
    per_seed: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    fields = protocol["order_geometry"]["stratifiers"]
    margin_index: dict[tuple[str, str, str], dict[int, float]] = defaultdict(dict)
    scene_index: dict[str, str] = {}
    for parent_row in parent_rows:
        margin_index[(parent_row["cell"], parent_row["baseline"], parent_row["sample_id"])][int(parent_row["seed"])] = float(parent_row["raw_order_margin"])
        scene_index[parent_row["sample_id"]] = parent_row["scene_id"]
    for field_index, field in enumerate(fields):
        for bin_index, label in enumerate(("low", "medium", "high")):
            allowed = {sid for sid, row in geometry_by_id.items() if row[f"{field}_bin"] == label}
            scenes = sorted({row["scene_id"] for row in parent_rows if row["sample_id"] in allowed})
            if not scenes:
                continue
            rng = np.random.default_rng(base_seed + 100 * field_index + bin_index)
            draws = rng.integers(0, len(scenes), size=(reps, len(scenes)))
            for cell in CELLS:
                model, layer = cell_parts(cell)
                copy = subset_scene_seed_array(parent_rows, cell, "copy", "sequential_endpoint_error", scenes, allowed)
                baseline_cache: dict[str, np.ndarray] = {}
                for baseline in BASELINES:
                    margin = subset_scene_seed_array(parent_rows, cell, baseline, "raw_order_margin", scenes, allowed)
                    normalized_estimate = float(margin.mean() / (copy.mean() + epsilon))
                    margin_dist = ratio_distribution(margin, copy, draws, epsilon)
                    # Registered win/loss/tie is evaluated once per tuple after
                    # averaging aligned seeds; layers/seeds are not samples.
                    relevant = {
                        sample: seed_values for (indexed_cell, indexed_baseline, sample), seed_values in margin_index.items()
                        if indexed_cell == cell and indexed_baseline == baseline and sample in allowed
                    }
                    averaged = {sample: float(np.mean(list(seed_values.values()))) for sample, seed_values in relevant.items()}
                    tolerance = float(protocol["order_geometry"]["tie_tolerance"])
                    wins = sum(value > tolerance for value in averaged.values())
                    losses = sum(value < -tolerance for value in averaged.values())
                    ties = len(averaged) - wins - losses
                    scene_scores = np.asarray([
                        np.mean([
                            1.0 if averaged[sample] > tolerance else (0.0 if averaged[sample] < -tolerance else 0.5)
                            for sample in averaged if scene_index[sample] == scene
                        ]) for scene in scenes
                    ])[None, :]
                    tie_dist = bootstrap_distribution(scene_scores, draws)
                    baseline_cache[baseline] = margin
                    stratified.append({
                        "cell": cell, "model": model, "layer": layer, "stratifier": field, "bin": label,
                        "bin_q1": rules[field]["q1"], "bin_q2": rules[field]["q2"],
                        "baseline": baseline, "samples": len(allowed), "scenes": len(scenes),
                        "normalized_order_margin": normalized_estimate,
                        "normalized_order_margin_ci95_low": float(np.quantile(margin_dist, .025)),
                        "normalized_order_margin_ci95_high": float(np.quantile(margin_dist, .975)),
                        "wins": wins, "losses": losses, "ties": ties,
                        "tie_aware_accuracy": float((wins + .5 * ties) / len(averaged)),
                        "tie_aware_accuracy_ci95_low": float(np.quantile(tie_dist, .025)),
                        "tie_aware_accuracy_ci95_high": float(np.quantile(tie_dist, .975)),
                    })
                    for seed_index, seed in enumerate(SEEDS):
                        seed_values = [seed_map[seed] for seed_map in relevant.values()]
                        seed_wins = sum(value > tolerance for value in seed_values)
                        seed_losses = sum(value < -tolerance for value in seed_values)
                        seed_ties = len(seed_values) - seed_wins - seed_losses
                        per_seed.append({
                            "cell": cell, "model": model, "layer": layer, "seed": seed,
                            "stratifier": field, "bin": label, "baseline": baseline,
                            "samples": len(seed_values), "scenes": len(scenes),
                            "normalized_order_margin": float(margin[seed_index].mean() / (copy[seed_index].mean() + epsilon)),
                            "wins": seed_wins, "losses": seed_losses, "ties": seed_ties,
                            "tie_aware_accuracy": float((seed_wins + .5 * seed_ties) / len(seed_values)),
                        })
                family_rows = []
                for control in REGISTERED_CONTROLS:
                    difference = baseline_cache["full"] - baseline_cache[control]
                    dist = ratio_distribution(difference, copy, draws, epsilon)
                    estimate, low, high, p = interval(dist)
                    item = {
                        "cell": cell, "model": model, "layer": layer, "stratifier": field, "bin": label,
                        "comparison": f"full_minus_{control}",
                        "estimate": float(difference.mean() / (copy.mean() + epsilon)),
                        "bootstrap_distribution_mean": estimate, "ci95_low": low,
                        "ci95_high": high, "bootstrap_p_two_sided": p, "resamples": reps,
                        "samples": len(allowed), "scenes": len(scenes),
                    }
                    family_rows.append(item); paired.append(item)
                adjusted = holm_adjust([row["bootstrap_p_two_sided"] for row in family_rows])
                for row, value in zip(family_rows, adjusted): row["holm_adjusted_p_within_cell_stratum"] = value
    write_csv(RESULT / "order/stratified_metrics.csv", stratified)
    write_csv(RESULT / "order/paired_control_differences.csv", paired)
    write_csv(RESULT / "order/per_seed_stratified_metrics.csv", per_seed)

    # Geometry/confound association is descriptive and never multiplicity-adjusted.
    correlation_rows = []
    eval_geo = list(geometry_by_id.values())
    for left in fields:
        for right in fields:
            x = np.asarray([float(row[left]) for row in eval_geo]); y = np.asarray([float(row[right]) for row in eval_geo])
            correlation_rows.append({"left": left, "right": right, "pearson_r": float(np.corrcoef(x, y)[0, 1]), "status": "descriptive_no_multiplicity_adjustment"})
    write_csv(RESULT / "order/geometry_confound_correlations.csv", correlation_rows)
    outcome_rows = []
    for cell in CELLS:
        margins: dict[str, list[float]] = defaultdict(list)
        for row in parent_rows:
            if row["cell"] == cell and row["baseline"] == "full":
                margins[row["sample_id"]].append(float(row["raw_order_margin"]))
        sample_ids = sorted(margins)
        outcome = np.asarray([np.mean(margins[sample]) for sample in sample_ids])
        for field in fields:
            predictor = np.asarray([float(geometry_by_id[sample][field]) for sample in sample_ids])
            outcome_rows.append({
                "cell": cell, "predictor": field,
                "outcome": "full_raw_order_margin_averaged_over_aligned_seeds",
                "samples": len(sample_ids),
                "pearson_r": float(np.corrcoef(predictor, outcome)[0, 1]) if predictor.std() > 0 and outcome.std() > 0 else float("nan"),
                "status": "descriptive_confound_check_no_multiplicity_adjustment",
            })
    write_csv(RESULT / "order/outcome_confound_correlations.csv", outcome_rows)
    return stratified, paired


def representative_comparison(parent_rows: list[dict[str, Any]], protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cells = ("wan_l15", "vjepa2_l11")
    scenes = sorted({row["scene_id"] for row in parent_rows})
    reps = int(protocol["statistics"]["bootstrap_replicates"]); epsilon = float(protocol["statistics"]["normalization_epsilon"])
    draws = np.random.default_rng(int(protocol["statistics"]["bootstrap_seed"])).integers(0, len(scenes), size=(reps, len(scenes)))
    seed_rows, specifications = [], {}
    for cell in cells:
        full_seq = scene_seed_array(parent_rows, cell, "full", "sequential_endpoint_error", scenes)
        copy_seq = scene_seed_array(parent_rows, cell, "copy", "sequential_endpoint_error", scenes)
        no_seq = scene_seed_array(parent_rows, cell, "without_condition", "sequential_endpoint_error", scenes)
        comp = scene_seed_array(parent_rows, cell, "full", "sequential_direct_prediction_distance", scenes)
        margin = scene_seed_array(parent_rows, cell, "full", "raw_order_margin", scenes)
        metrics = {
            "copy_relative_gain": (copy_seq - full_seq, copy_seq),
            "no_action_relative_gain": (no_seq - full_seq, no_seq),
            "normalized_composition_distance": (comp, copy_seq),
            "normalized_order_effect": (margin, copy_seq),
        }
        specifications[cell] = metrics
        for seed_index, seed in enumerate(SEEDS):
            for name, (numerator, denominator) in metrics.items():
                seed_rows.append({
                    "cell": cell, "seed": seed, "metric": name,
                    "scene_macro": float(numerator[seed_index].mean() / (denominator[seed_index].mean() + epsilon)),
                })
    comparisons = []
    for name in specifications[cells[0]]:
        left_num, left_den = specifications[cells[0]][name]
        right_num, right_den = specifications[cells[1]][name]
        left_dist = ratio_distribution(left_num, left_den, draws, epsilon)
        right_dist = ratio_distribution(right_num, right_den, draws, epsilon)
        dist = left_dist - right_dist
        bootstrap_mean, low, high, p = interval(dist)
        point = float(left_num.mean() / (left_den.mean() + epsilon) - right_num.mean() / (right_den.mean() + epsilon))
        comparisons.append({"left": cells[0], "right": cells[1], "direction": "Wan_minus_V-JEPA2", "metric": name, "estimate": point, "bootstrap_distribution_mean": bootstrap_mean, "ci95_low": low, "ci95_high": high, "bootstrap_p_two_sided": p, "resamples": reps})
    for row, adjusted in zip(comparisons, holm_adjust([row["bootstrap_p_two_sided"] for row in comparisons])):
        row["holm_adjusted_p"] = adjusted
    write_csv(RESULT / "representative/per_seed_metrics.csv", seed_rows)
    write_csv(RESULT / "representative/paired_model_comparison.csv", comparisons)
    return seed_rows, comparisons


def layer_joint_table(composition: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parent = [row for row in read_csv(PARENT / "layer_main_table.csv") if row["baseline"] == "full"]
    composition_index = {row["cell"]: row for row in composition if row["baseline"] == "full"}
    output = []
    for row in parent:
        new = composition_index[row["cell"]]
        output.append({
            "cell": row["cell"], "model": row["model"], "layer": int(row["layer"]),
            "c1_one_step_error": float(row["one_step_error_mean"]),
            "two_step_grounding_gain_vs_copy": float(new["sequential_gain_vs_copy"]),
            "two_step_gain_vs_without_condition": float(new["sequential_gain_vs_without_condition"]),
            "normalized_sequential_direct_distance": float(new["normalized_sequential_direct_distance"]),
            "recursive_minus_direct_endpoint_gap": float(new["recursive_minus_direct_endpoint_gap"]),
            "normalized_order_margin": float(row["normalized_order_margin_mean"]),
            "tie_aware_accuracy": float(row["tie_aware_accuracy_mean"]),
            "scope": "descriptive_all_frozen_layers",
        })
    write_csv(RESULT / "layer_joint_main_table.csv", output)
    return output


POINTS = np.asarray([
    [-.8, -.3, 1.2], [-.2, .7, 1.6], [.4, -.6, 2.1], [1.0, .2, 2.7],
    [-.7, .9, 3.1], [.3, .4, 3.8], [1.2, -.8, 4.3], [-1.1, -.5, 4.9],
    [.6, 1.1, 5.4], [-.4, -.9, 6.0], [1.3, .7, 6.7], [-1.2, 1.0, 7.3],
], dtype=np.float32)


def point_rmse(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((left.reshape(-1, len(POINTS), 3) - right.reshape(-1, len(POINTS), 3)) ** 2, axis=(1, 2)))


def positive_control(device_name: str = "cpu") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _, parent_protocol, rows = verify_frozen_inputs()
    transforms = dict(np.load(RESULT / "data/physical_transforms.npz", allow_pickle=True))
    t1 = np.asarray(transforms["t1"]); t2 = np.asarray(transforms["t2"])
    split = np.asarray([row["analysis_split"] for row in rows])
    train_i, val_i, eval_i = (np.flatnonzero(split == value) for value in ("train", "validation", "evaluation"))
    canonical = np.tile(POINTS.reshape(1, -1), (len(rows), 1)).astype(np.float32)
    target_b = np.stack([transform_points(t, POINTS).reshape(-1) for t in t1]).astype(np.float32)
    target_c = np.stack([transform_points(b @ a, POINTS).reshape(-1) for a, b in zip(t1, t2)]).astype(np.float32)
    swapped_true = np.stack([transform_points(a @ b, POINTS).reshape(-1) for a, b in zip(t1, t2)]).astype(np.float32)
    action1_raw = torch.from_numpy(np.stack([encode_relative_pose_9d(t).astype(np.float32) for t in t1]))
    action2_raw = torch.from_numpy(np.stack([encode_relative_pose_9d(t).astype(np.float32) for t in t2]))
    comp_raw = torch.from_numpy(np.stack([encode_relative_pose_9d(b @ a).astype(np.float32) for a, b in zip(t1, t2)]))
    identity_raw = torch.from_numpy(encode_relative_pose_9d(np.eye(4)).astype(np.float32))
    context_raw, target_raw = torch.from_numpy(canonical), torch.from_numpy(target_b)
    context_mean, context_std = context_raw[train_i].mean(0), context_raw[train_i].std(0, unbiased=False).clamp_min(1e-5)
    target_mean, target_std = target_raw[train_i].mean(0), target_raw[train_i].std(0, unbiased=False).clamp_min(1e-5)
    action_mean, action_std = action1_raw[train_i].mean(0), action1_raw[train_i].std(0, unbiased=False).clamp_min(1e-5)
    context = (context_raw - context_mean) / context_std
    za = (torch.from_numpy(canonical) - target_mean) / target_std
    target = (target_raw - target_mean) / target_std
    a1, a2, comp = tuple((value - action_mean) / action_std for value in (action1_raw, action2_raw, comp_raw))
    identity = ((identity_raw - action_mean) / action_std).expand(len(eval_i), -1)
    device = torch.device(device_name)
    sample_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    eval_rows = [rows[index] for index in eval_i]
    for seed in SEEDS:
        model, record = train_head(
            kind="full", channels=canonical.shape[1], seed=seed, cfg=parent_protocol["training"], device=device,
            train_context=context[train_i].to(device), train_za=za[train_i].to(device), train_action=a1[train_i].to(device), train_target=target[train_i].to(device),
            val_context=context[val_i].to(device), val_za=za[val_i].to(device), val_action=a1[val_i].to(device), val_target=target[val_i].to(device),
        )
        checkpoint_dir = RESULT / "positive_control/checkpoints"; checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "seed": seed, "head": "full", "channels": canonical.shape[1], "training_record": record, "normalization": {"context_mean": context_mean, "context_std": context_std, "target_B_mean": target_mean, "target_B_std": target_std, "action_AB_mean": action_mean, "action_AB_std": action_std}, "supervision": "single_step_B_only"}, checkpoint_dir / f"seed{seed}.pt")
        with torch.no_grad():
            state = model.initialize_state(context[eval_i].to(device)[:, None, None, None, :])
            one_state = model.update_state(state, a1[eval_i].to(device))
            one = model.decode_state(one_state)
            seq = model.decode_state(model.update_state(one_state, a2[eval_i].to(device)))
            direct_state = model.update_state(state, comp[eval_i].to(device))
            direct = model.decode_state(direct_state)
            swapped_state = model.update_state(model.update_state(state, a2[eval_i].to(device)), a1[eval_i].to(device))
            swapped = model.decode_state(swapped_state)
            pre = model.decode_state(model.update_state(model.update_state(state, identity.to(device)), comp[eval_i].to(device)))
            post = model.decode_state(model.update_state(direct_state, identity.to(device)))
        denorm = lambda value: (value.cpu() * target_std + target_mean).numpy()
        predictions = {"one_step": denorm(one), "sequential": denorm(seq), "direct": denorm(direct), "swapped": denorm(swapped), "identity_pre": denorm(pre), "identity_post": denorm(post)}
        targets = {"one_step": target_b[eval_i], "sequential": target_c[eval_i], "direct": target_c[eval_i], "swapped": target_c[eval_i], "identity_pre": target_c[eval_i], "identity_post": target_c[eval_i]}
        errors = {name: point_rmse(value, targets[name]) for name, value in predictions.items()}
        seq_direct = point_rmse(predictions["sequential"], predictions["direct"])
        for index, row in enumerate(eval_rows):
            item = {"system": "learned_single_step_GRU", "seed": seed, "sample_id": row["sample_id"], "scene_id": row["scene_id"]}
            item.update({f"{name}_point_rmse_m": float(value[index]) for name, value in errors.items()})
            item.update({"sequential_direct_point_rmse_m": float(seq_direct[index]), "raw_order_margin_m": float(errors["swapped"][index] - errors["sequential"][index])})
            sample_rows.append(item)
        seed_rows.append({"system": "learned_single_step_GRU", "seed": seed, "best_epoch": record["best_epoch"], **{f"{name}_point_rmse_m": float(value.mean()) for name, value in errors.items()}, "sequential_direct_point_rmse_m": float(seq_direct.mean()), "raw_order_margin_m": float((errors["swapped"] - errors["sequential"]).mean())})

    oracle_seq = target_c[eval_i]
    oracle_swap = swapped_true[eval_i]
    zero = np.zeros(len(eval_i), dtype=np.float64)
    swap_error = point_rmse(oracle_swap, oracle_seq)
    for index, row in enumerate(eval_rows):
        sample_rows.append({"system": "analytic_oracle", "seed": -1, "sample_id": row["sample_id"], "scene_id": row["scene_id"], "one_step_point_rmse_m": 0.0, "sequential_point_rmse_m": 0.0, "direct_point_rmse_m": 0.0, "swapped_point_rmse_m": float(swap_error[index]), "identity_pre_point_rmse_m": 0.0, "identity_post_point_rmse_m": 0.0, "sequential_direct_point_rmse_m": 0.0, "raw_order_margin_m": float(swap_error[index])})
    seed_rows.append({"system": "analytic_oracle", "seed": -1, "best_epoch": -1, "one_step_point_rmse_m": 0.0, "sequential_point_rmse_m": 0.0, "direct_point_rmse_m": 0.0, "swapped_point_rmse_m": float(swap_error.mean()), "identity_pre_point_rmse_m": 0.0, "identity_post_point_rmse_m": 0.0, "sequential_direct_point_rmse_m": 0.0, "raw_order_margin_m": float(swap_error.mean())})
    write_csv(RESULT / "positive_control/per_sample_metrics.csv", sample_rows)
    write_csv(RESULT / "positive_control/per_seed_metrics.csv", seed_rows)
    support = {"identity_exact_training_match": bool(torch.any(torch.all(torch.isclose(action1_raw[train_i], identity_raw, atol=1e-8, rtol=0), dim=1))), "identity_coordinatewise_training_range_check": bool(torch.all((identity_raw >= action1_raw[train_i].min(0).values) & (identity_raw <= action1_raw[train_i].max(0).values))), "identity_nearest_standardized_training_action_l2": float(torch.linalg.vector_norm((action1_raw[train_i] - action_mean) / action_std - (identity_raw - action_mean) / action_std, dim=1).min()), "range_check_is_not_strict_in_distribution_certification": True}
    (RESULT / "positive_control/identity_support.json").write_text(json.dumps(support, indent=2, sort_keys=True) + "\n")
    return sample_rows, seed_rows


def positive_control_bootstrap(sample_rows: list[dict[str, Any]], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    scenes = sorted({row["scene_id"] for row in sample_rows})
    reps = int(protocol["statistics"]["bootstrap_replicates"])
    draws = np.random.default_rng(int(protocol["statistics"]["bootstrap_seed"])).integers(0, len(scenes), size=(reps, len(scenes)))
    fields = (
        "one_step_point_rmse_m", "sequential_point_rmse_m", "direct_point_rmse_m",
        "swapped_point_rmse_m", "sequential_direct_point_rmse_m", "raw_order_margin_m",
        "identity_pre_point_rmse_m", "identity_post_point_rmse_m",
    )
    output = []
    for system, seeds in (("learned_single_step_GRU", SEEDS), ("analytic_oracle", (-1,))):
        for field in fields:
            index: dict[tuple[int, str], list[float]] = defaultdict(list)
            for row in sample_rows:
                if row["system"] == system:
                    index[(int(row["seed"]), row["scene_id"])].append(float(row[field]))
            matrix = np.asarray([[np.mean(index[(seed, scene)]) for scene in scenes] for seed in seeds])
            dist = bootstrap_distribution(matrix, draws)
            bootstrap_mean, low, high, p = interval(dist)
            output.append({
                "system": system, "metric": field, "estimate": float(matrix.mean()),
                "bootstrap_distribution_mean": bootstrap_mean, "ci95_low": low, "ci95_high": high,
                "bootstrap_p_two_sided": p, "resamples": reps, "scenes": len(scenes),
                "training_seeds_averaged": len(seeds) if system != "analytic_oracle" else 0,
            })
    write_csv(RESULT / "positive_control/paired_scene_bootstrap.csv", output)
    return output


def plots(composition: list[dict[str, Any]], order: list[dict[str, Any]], positive: list[dict[str, Any]]) -> None:
    figure_dir = RESULT / "figures"; figure_dir.mkdir(parents=True, exist_ok=True)
    identity = read_csv(RESULT / "composition/identity_padding_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for cell, color in (("wan_l15", "tab:blue"), ("vjepa2_l11", "tab:orange")):
        row = next(r for r in identity if r["cell"] == cell and r["baseline"] == "full")
        axes[0].bar([f"{cell}\nid→comp", f"{cell}\ncomp→id", f"{cell}\nT1→T2"], [float(row["identity_pre_minus_direct_endpoint_error"]), float(row["identity_post_minus_direct_endpoint_error"]), float(row["sequential_minus_direct_endpoint_error"])], color=color, alpha=.75)
    axes[0].axhline(0, color="black", lw=.7); axes[0].set_ylabel("endpoint error minus direct")
    for model, color in (("wan", "tab:blue"), ("vjepa2", "tab:orange")):
        values = sorted([r for r in composition if r["model"] == model and r["baseline"] == "full"], key=lambda r: int(r["layer"]))
        axes[1].plot([int(r["layer"]) for r in values], [float(r["normalized_sequential_direct_distance"]) for r in values], marker="o", label=model, color=color)
    axes[1].set_xlabel("exported layer"); axes[1].set_ylabel("normalized seq–direct distance"); axes[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(figure_dir / "p1_composition_identity.png", dpi=200); fig.savefig(figure_dir / "p1_composition_identity.pdf"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for axis, field in zip(axes[:2], ("order_rotation_difference_rad", "order_camera_center_difference_m")):
        for cell, color in (("wan_l15", "tab:blue"), ("vjepa2_l11", "tab:orange")):
            values = [r for r in order if r["cell"] == cell and r["stratifier"] == field and r["baseline"] == "full"]
            values.sort(key=lambda r: ("low", "medium", "high").index(r["bin"]))
            axis.plot([r["bin"] for r in values], [float(r["normalized_order_margin"]) for r in values], marker="o", label=cell, color=color)
        axis.axhline(0, color="black", lw=.7); axis.set_xlabel(field.replace("order_", "")); axis.grid(alpha=.2)
    axes[0].set_ylabel("normalized order margin"); axes[0].legend(frameon=False)
    parent_layers = [row for row in read_csv(PARENT / "layer_main_table.csv") if row["baseline"] == "full"]
    for model, color in (("wan", "tab:blue"), ("vjepa2", "tab:orange")):
        values = sorted([row for row in parent_layers if row["model"] == model], key=lambda row: int(row["layer"]))
        axes[2].plot([int(row["layer"]) for row in values], [float(row["normalized_order_margin_mean"]) for row in values], marker="o", color=color, label=model)
    axes[2].axhline(0, color="black", lw=.7); axes[2].set_xlabel("exported layer"); axes[2].set_ylabel("overall normalized order margin"); axes[2].grid(alpha=.2)
    fig.tight_layout(); fig.savefig(figure_dir / "p1_order_physical_bins.png", dpi=200); fig.savefig(figure_dir / "p1_order_physical_bins.pdf"); plt.close(fig)

    learned = [r for r in positive if r["system"] == "learned_single_step_GRU"]
    oracle = next(r for r in positive if r["system"] == "analytic_oracle")
    fields = ("one_step_point_rmse_m", "sequential_point_rmse_m", "direct_point_rmse_m", "sequential_direct_point_rmse_m")
    names = ("one-step", "sequential", "direct", "seq–direct")
    learned_mean = [np.mean([float(r[field]) for r in learned]) for field in fields]
    fig, axis = plt.subplots(figsize=(7, 3.8)); x = np.arange(len(fields)); width = .35
    axis.bar(x-width/2, [float(oracle[field]) for field in fields], width, label="analytic oracle")
    axis.bar(x+width/2, learned_mean, width, label="learned one-step GRU")
    axis.set_xticks(x, names); axis.set_ylabel("3D point RMSE (m)"); axis.legend(frameon=False); axis.set_yscale("symlog", linthresh=1e-8)
    fig.tight_layout(); fig.savefig(figure_dir / "p1_positive_control.png", dpi=200); fig.savefig(figure_dir / "p1_positive_control.pdf"); plt.close(fig)


def findings(composition: list[dict[str, Any]], order: list[dict[str, Any]], comparisons: list[dict[str, Any]], positive: list[dict[str, Any]]) -> str:
    significant = [row for row in comparisons if float(row["holm_adjusted_p"]) < .05 and not (float(row["ci95_low"]) <= 0 <= float(row["ci95_high"]))]
    identity = read_csv(RESULT / "composition/identity_padding_bootstrap.csv")
    lines = [
        "# P1 Findings", "",
        "This follow-up reuses the frozen 436-sample p12 parent experiment and every validation-selected visual checkpoint; no visual head was retrained and no feature was re-extracted.", "",
        "## Predeclared paired model comparison", "",
    ]
    if significant:
        lines.append("Holm-adjusted paired evidence supports: " + "; ".join(f"{r['metric']} (Wan−V-JEPA2 {float(r['estimate']):.4g}, 95% CI [{float(r['ci95_low']):.4g}, {float(r['ci95_high']):.4g}], Holm p={float(r['holm_adjusted_p']):.4g})" for r in significant) + ".")
    else:
        lines.append("None of the four registered Wan L15 versus V-JEPA2 L11 differences passed the joint Holm rule. This is an absence of paired evidence, not evidence that the representations are identical.")
    lines.extend(["", "## Composition and call-count alternative", ""])
    for cell in ("wan_l15", "vjepa2_l11"):
        rows = [r for r in identity if r["cell"] == cell and r["baseline"] == "full"]
        by = {r["metric"]: r for r in rows}
        seq = by["sequential_minus_direct_endpoint_error"]
        pre = by["identity_pre_minus_direct_endpoint_error"]
        post = by["identity_post_minus_direct_endpoint_error"]
        pre_contrast = by["sequential_minus_identity_pre_endpoint_error"]
        post_contrast = by["sequential_minus_identity_post_endpoint_error"]
        call_count_not_excluded = (
            float(pre_contrast["ci95_low"]) <= 0 <= float(pre_contrast["ci95_high"])
            or float(post_contrast["ci95_low"]) <= 0 <= float(post_contrast["ci95_high"])
        )
        conclusion = "the update-count explanation is not excluded" if call_count_not_excluded else "the original two-action gap differs from both identity-padded gaps, so update count alone is insufficient"
        lines.append(f"- {cell}: original two-update minus direct endpoint gap {float(seq['estimate']):.4g} (CI [{float(seq['ci95_low']):.4g}, {float(seq['ci95_high']):.4g}]); identity-before {float(pre['estimate']):.4g}; identity-after {float(post['estimate']):.4g}. Paired sequential-minus-identity contrasts give pre CI [{float(pre_contrast['ci95_low']):.4g}, {float(pre_contrast['ci95_high']):.4g}] and post CI [{float(post_contrast['ci95_low']):.4g}, {float(post_contrast['ci95_high']):.4g}]; {conclusion}. Low endpoint-error similarity by itself is not treated as composition consistency.")
    support = read_csv(RESULT / "composition/identity_action_support.csv")
    exact = all(row["identity_exact_training_match"] == "True" for row in support)
    lines.append(f"Exact identity was {'present' if exact else 'absent'} among frozen training actions. Coordinatewise min–max results are reported only as range checks, never as strict distribution certification.")
    lines.extend(["", "## Order signal and limitations", ""])
    for field, unit in (("order_rotation_difference_rad", "rad"), ("order_camera_center_difference_m", "m")):
        values = [r for r in order if r["cell"] in {"wan_l15", "vjepa2_l11"} and r["stratifier"] == field and r["bin"] == "high" and r["baseline"] == "full"]
        lines.append("- High physical-difference bin (" + unit + "): " + "; ".join(f"{r['cell']} margin {float(r['normalized_order_margin']):.4g}, CI [{float(r['normalized_order_margin_ci95_low']):.4g}, {float(r['normalized_order_margin_ci95_high']):.4g}], tie-aware {float(r['tie_aware_accuracy']):.3f}" for r in values) + ".")
    learned = [r for r in positive if r["system"] == "learned_single_step_GRU"]
    oracle = next(r for r in positive if r["system"] == "analytic_oracle")
    parent_relationships = read_csv(PARENT / "layer_metric_relationships.csv")
    relation = {(row["model"], row["y"]): float(row["pearson_across_exported_layers"]) for row in parent_relationships}
    joint = layer_joint_table(composition)
    normalized_corr = {}
    for model in ("wan", "vjepa2"):
        values = sorted([row for row in joint if row["model"] == model], key=lambda row: row["layer"])
        normalized_corr[model] = float(np.corrcoef([row["c1_one_step_error"] for row in values], [row["normalized_sequential_direct_distance"] for row in values])[0, 1])
    lines.extend(["", "## Descriptive layer relationship", "", f"Across the seven Wan layers, C1 error correlates with raw sequential–direct distance at r={relation[('wan', 'sequential_direct_prediction_distance')]:.3f}, but with the copy-normalized composition distance at r={normalized_corr['wan']:.3f} and normalized order margin at r={relation[('wan', 'normalized_order_margin')]:.3f}. Across the five V-JEPA2 layers the corresponding values are r={relation[('vjepa2', 'sequential_direct_prediction_distance')]:.3f}, r={normalized_corr['vjepa2']:.3f}, and r={relation[('vjepa2', 'normalized_order_margin')]:.3f}. These are descriptive correlations over only 7/5 exported layers: prediction error and structural scores do not move together uniformly, and no best layer is selected post hoc.", "", "## Analytic positive control", "", f"The analytic oracle has sequential/direct gap {float(oracle['sequential_direct_point_rmse_m']):.3g} m and positive correct-versus-swapped margin {float(oracle['raw_order_margin_m']):.4g} m. The single-step learned GRU has mean one-step RMSE {np.mean([float(r['one_step_point_rmse_m']) for r in learned]):.4g} m, sequential RMSE {np.mean([float(r['sequential_point_rmse_m']) for r in learned]):.4g} m, and order margin {np.mean([float(r['raw_order_margin_m']) for r in learned]):.4g} m. The oracle validates the metric implementation, while the learned head's one-step and recursive errors make diagnostic-head limitation an observed alternative explanation rather than a hypothetical one.", "", "Consequently, weak visual order evidence can reflect the physical sample distribution, the single-step-trained recurrent diagnostic, the frozen representation, or an interaction among them. The geometry strata and positive control narrow those alternatives but do not identify representation incapacity, and observed two-step behavior is not a proof of a complete SE(3) group action.", ""])
    return "\n".join(lines)


def run_all(device: str) -> None:
    protocol, _, _ = verify_frozen_inputs()
    geometry, _, rules = prepare_geometry()
    parent_rows = load_parent_samples()
    identity_rows = identity_inference(device)
    composition, _ = composition_analysis(parent_rows, identity_rows, protocol)
    order, _ = order_analysis(parent_rows, geometry, rules, protocol)
    _, comparisons = representative_comparison(parent_rows, protocol)
    positive_samples, positive = positive_control(device)
    positive_control_bootstrap(positive_samples, protocol)
    plots(composition, order, positive)
    (RESULT / "P1_FINDINGS.md").write_text(findings(composition, order, comparisons, positive))
    (RESULT / "REPRODUCE.md").write_text("# Reproduce\n\n```bash\n/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python -m analyses.se3_protocol_v2.wan_vjepa2_p1 all --device cuda\n/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python -m pytest -q tests/test_wan_vjepa2_p1.py\n```\n\nThe command verifies all frozen parent hashes and PASS receipts before reading predictions/checkpoints. It does not rerun parent training or extraction.\n")
    (RESULT / "COMPLETE.json").write_text(json.dumps({"status": "PASS", "parent_training_rerun": False, "features_reextracted": False, "cells": len(CELLS), "seeds": len(SEEDS), "evaluation_samples": 276, "bootstrap_replicates": 2000}, indent=2, sort_keys=True) + "\n")


def run_aggregate_only() -> None:
    """Repair/rebuild tables from frozen raw rows without any inference/training."""
    protocol, _, _ = verify_frozen_inputs()
    parent_rows = load_parent_samples()
    identity_rows = read_csv(RESULT / "composition/identity_padding_per_sample.csv")
    geometry = read_csv(RESULT / "data/physical_geometry.csv")
    rules = json.loads((RESULT / "data/TRAIN_BIN_RULES.json").read_text())
    composition, _ = composition_analysis(parent_rows, identity_rows, protocol)
    order, _ = order_analysis(parent_rows, geometry, rules, protocol)
    _, comparisons = representative_comparison(parent_rows, protocol)
    positive = read_csv(RESULT / "positive_control/per_seed_metrics.csv")
    positive_control_bootstrap(read_csv(RESULT / "positive_control/per_sample_metrics.csv"), protocol)
    plots(composition, order, positive)
    (RESULT / "P1_FINDINGS.md").write_text(findings(composition, order, comparisons, positive))
    (RESULT / "COMPLETE.json").write_text(json.dumps({
        "status": "PASS", "parent_training_rerun": False, "features_reextracted": False,
        "cells": len(CELLS), "seeds": len(SEEDS), "evaluation_samples": 276,
        "bootstrap_replicates": 2000,
        "normalization": "ratio_of_scene_macro_aggregates_verified",
        "order_win_loss_tie_unit": "tuple_after_aligned_seed_average",
    }, indent=2, sort_keys=True) + "\n")


def finalize_existing() -> None:
    """Regenerate joined table, plots and prose from already frozen outputs."""
    composition = read_csv(RESULT / "composition/layer_baseline_summary.csv")
    order = read_csv(RESULT / "order/stratified_metrics.csv")
    comparisons = read_csv(RESULT / "representative/paired_model_comparison.csv")
    positive = read_csv(RESULT / "positive_control/per_seed_metrics.csv")
    plots(composition, order, positive)
    (RESULT / "P1_FINDINGS.md").write_text(findings(composition, order, comparisons, positive))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "identity", "positive-control", "aggregate", "finalize", "all"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "prepare": prepare_geometry()
    elif args.command == "identity": identity_inference(args.device)
    elif args.command == "positive-control": positive_control(args.device)
    elif args.command == "aggregate": run_aggregate_only()
    elif args.command == "finalize": finalize_existing()
    else: run_all(args.device)
