#!/usr/bin/env python3
"""Observed-video grounded-order protocol for frozen C-series features.

The primary target is a real C2 isolated feature ``z_C`` from each observed
``A -> B -> C`` tuple.  Predictors train solely on observed one-step pairs;
order, composition, direct, and pose-grounding quantities are held-out analyses.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from analyses.se3_latent_mvp.geometry import (
    camera_to_world_to_extrinsic,
    relative_pose,
    se3_exp,
    se3_log,
)
from analyses.se3_latent_mvp.run_mvp import (
    LatentPredictor,
    cosine_distance,
    load_and_audit_tuples,
    matched_donor_mapping,
    normalized_rmse,
    pair_tensors,
    records_for_split,
    split_scenes,
    train_model,
    transform_batch,
)
from analyses.se3_protocol_v2.geometry import group_commutator, rotation_degrees, translation_norm
from analyses.se3_protocol_v2.metrics import whitened_nrmse
from analyses.se3_protocol_v2.protocol import POSE_CONVENTION, dump_json, sha256
from analyses.se3_protocol_v2.see_se3_smoke import PoincareAdapter, metrics as pose_r2_metrics


REPO_ROOT = Path(__file__).resolve().parents[2]
LEARNED_CONDITIONS = ("without_condition", "action_only", "full")
BASELINES = (
    "copy",
    "without_condition",
    "action_only",
    "matched_wrong_pose",
    "shuffled_pose",
    "reversed_order",
    "full",
)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def action_tensor(transforms: np.ndarray, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    values = torch.from_numpy(np.stack([se3_log(value) for value in transforms])).float()
    return (values - mean) / std


def learned_predict(
    name: str,
    models: dict[str, LatentPredictor],
    state: torch.Tensor,
    transforms: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    if name == "copy":
        return state.clone()
    model = models["full" if name in {"matched_wrong_pose", "shuffled_pose", "reversed_order"} else name]
    with torch.no_grad():
        return model(state, action_tensor(transforms, mean, std))


def derangement(length: int, seed: int) -> np.ndarray:
    if length < 2:
        raise RuntimeError("need at least two held-out tuples for shuffled-pose control")
    rng = np.random.default_rng(seed)
    for _ in range(64):
        order = rng.permutation(length)
        if not np.any(order == np.arange(length)):
            return order
    return np.roll(np.arange(length), 1)


def kappa(t1: np.ndarray, t2: np.ndarray) -> float:
    return float(np.linalg.norm(se3_log(group_commutator(t1, t2))))


def one_step_metrics(
    ab_prediction: torch.Tensor,
    bc_prediction: torch.Tensor,
    z_b: torch.Tensor,
    z_c: torch.Tensor,
) -> dict[str, float]:
    """Ground every model/control on real observed one-step endpoints."""
    ab = cosine_distance(ab_prediction, z_b)
    bc = cosine_distance(bc_prediction, z_c)
    return {
        "one_step_ab_endpoint_cosine": float(ab.mean()),
        "one_step_bc_endpoint_cosine": float(bc.mean()),
        "one_step_grounding_cosine": float(torch.cat((ab, bc)).mean()),
    }


def mean_metrics(correct: torch.Tensor, reversed_: torch.Tensor, direct: torch.Tensor, target: torch.Tensor, feature_std: torch.Tensor, copy_error: float) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    correct_cos = cosine_distance(correct, target)
    reversed_cos = cosine_distance(reversed_, target)
    direct_cos = cosine_distance(direct, target)
    correct_wr = whitened_nrmse(correct, target, feature_std)
    reversed_wr = whitened_nrmse(reversed_, target, feature_std)
    direct_wr = whitened_nrmse(direct, target, feature_std)
    gap = cosine_distance(correct, direct)
    margin = reversed_cos - correct_cos
    values = {
        "correct_endpoint_cosine": float(correct_cos.mean()),
        "reversed_order_endpoint_cosine": float(reversed_cos.mean()),
        "direct_endpoint_cosine": float(direct_cos.mean()),
        "correct_endpoint_whitened_nrmse": float(correct_wr.mean()),
        "reversed_order_endpoint_whitened_nrmse": float(reversed_wr.mean()),
        "direct_endpoint_whitened_nrmse": float(direct_wr.mean()),
        "grounded_order_margin_cosine": float(margin.mean()),
        "order_ranking_accuracy": float((correct_cos < reversed_cos).float().mean()),
        "recursive_direct_gap_cosine": float(gap.mean()),
        "copy_normalized_improvement": float((copy_error - correct_cos.mean()) / max(copy_error, 1e-8)),
    }
    arrays = {
        "correct_cosine": correct_cos.detach().cpu().numpy(),
        "reversed_cosine": reversed_cos.detach().cpu().numpy(),
        "direct_cosine": direct_cos.detach().cpu().numpy(),
        "margin_cosine": margin.detach().cpu().numpy(),
        "ranking": (correct_cos < reversed_cos).detach().cpu().numpy().astype(float),
        "gap_cosine": gap.detach().cpu().numpy(),
    }
    return values, arrays


def train_pose_adapter(
    train_source: torch.Tensor,
    train_target: torch.Tensor,
    train_y: torch.Tensor,
    val_source: torch.Tensor,
    val_target: torch.Tensor,
    val_y: torch.Tensor,
    seed: int,
    max_epochs: int,
) -> tuple[PoincareAdapter, dict[str, Any]]:
    torch.manual_seed(seed)
    model = PoincareAdapter(int(train_source.shape[-1]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss, stale = float("inf"), 0
    history: list[dict[str, float]] = []
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(max_epochs):
        model.train()
        order = torch.randperm(len(train_source), generator=generator)
        losses: list[float] = []
        for start in range(0, len(order), 64):
            index = order[start : start + 64]
            loss = torch.mean((model(train_source[index], train_target[index]) - train_y[index]) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            val_loss = float(torch.mean((model(val_source, val_target) - val_y) ** 2))
        history.append({"epoch": float(epoch), "train_mse": float(np.mean(losses)), "val_mse": val_loss})
        if val_loss < best_loss - 1e-7:
            best_loss, stale = val_loss, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 30:
            break
    if best_state is None:
        raise RuntimeError("two-sided pose adapter did not fit")
    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_val_mse": best_loss, "epochs": len(history), "history": history}


def pose_error(prediction: np.ndarray, target: np.ndarray, pose_std: np.ndarray) -> dict[str, Any]:
    # Report physical SE(3) residuals, not merely a Euclidean difference of
    # Lie coordinates (whose translational part is coupled to rotation).
    residuals = [se3_exp(estimate) @ np.linalg.inv(se3_exp(reference)) for estimate, reference in zip(prediction, target)]
    rotation = np.asarray([rotation_degrees(value) for value in residuals])
    translation = np.asarray([translation_norm(value) for value in residuals])
    normalized = np.linalg.norm((prediction - target) / pose_std, axis=1)
    return {"rotation_deg": rotation, "translation_m": translation, "normalized": normalized}


def two_sided_pose_metrics(
    adapter: PoincareAdapter,
    source: torch.Tensor,
    correct: torch.Tensor,
    reversed_: torch.Tensor,
    t1: np.ndarray,
    t2: np.ndarray,
    pose_mean: torch.Tensor,
    pose_std: torch.Tensor,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    with torch.no_grad():
        decoded_12 = (adapter(source, correct) * pose_std + pose_mean).cpu().numpy()
        decoded_21 = (adapter(source, reversed_) * pose_std + pose_mean).cpu().numpy()
    target_12 = np.stack([se3_log(right @ left) for left, right in zip(t1, t2)])
    target_21 = np.stack([se3_log(left @ right) for left, right in zip(t1, t2)])
    std = pose_std.cpu().numpy()
    error_12, error_21 = pose_error(decoded_12, target_12, std), pose_error(decoded_21, target_21, std)
    correct_assignment = error_12["normalized"] + error_21["normalized"]
    swapped_assignment = pose_error(decoded_12, target_21, std)["normalized"] + pose_error(decoded_21, target_12, std)["normalized"]
    true_comm = np.stack([group_commutator(left, right) for left, right in zip(t1, t2)])
    pred_comm = np.stack([se3_exp(a) @ np.linalg.inv(se3_exp(b)) for a, b in zip(decoded_12, decoded_21)])
    comm_rot = np.asarray([rotation_degrees(np.linalg.inv(truth) @ pred) for truth, pred in zip(true_comm, pred_comm)])
    comm_trans = np.asarray([translation_norm(np.linalg.inv(truth) @ pred) for truth, pred in zip(true_comm, pred_comm)])
    summary = {
        "decoded_12_rotation_error_deg": float(error_12["rotation_deg"].mean()),
        "decoded_12_translation_error_m": float(error_12["translation_m"].mean()),
        "decoded_21_rotation_error_deg": float(error_21["rotation_deg"].mean()),
        "decoded_21_translation_error_m": float(error_21["translation_m"].mean()),
        "two_branch_pose_assignment_accuracy": float(np.mean(correct_assignment < swapped_assignment)),
        "two_branch_pose_margin": float(np.mean(swapped_assignment - correct_assignment)),
        "decoded_commutator_rotation_error_deg": float(comm_rot.mean()),
        "decoded_commutator_translation_error_m": float(comm_trans.mean()),
    }
    arrays = {
        "decoded_12_rotation_error_deg": error_12["rotation_deg"],
        "decoded_12_translation_error_m": error_12["translation_m"],
        "decoded_21_rotation_error_deg": error_21["rotation_deg"],
        "decoded_21_translation_error_m": error_21["translation_m"],
        "two_branch_assignment": (correct_assignment < swapped_assignment).astype(float),
        "decoded_commutator_rotation_error_deg": comm_rot,
        "decoded_commutator_translation_error_m": comm_trans,
    }
    return summary, arrays


def classify_family(t1: np.ndarray, t2: np.ndarray, thresholds: dict[str, float]) -> str:
    r1, r2 = rotation_degrees(t1), rotation_degrees(t2)
    d1, d2 = translation_norm(t1), translation_norm(t2)
    a1, a2 = se3_log(t1)[:3], se3_log(t2)[:3]
    n1, n2 = np.linalg.norm(a1), np.linalg.norm(a2)
    if n1 >= math.radians(thresholds["rot_high_deg"]) and n2 >= math.radians(thresholds["rot_high_deg"]):
        if abs(float(a1 @ a2) / max(n1 * n2, 1e-12)) <= 0.8:
            return "different_axis_rotation_rotation"
    # Natural hand-held trajectories rarely contain mathematically pure
    # rotations/translations.  This stratum therefore means a pair containing
    # a rotation-rich primitive and a translation-rich primitive, allowing the
    # real camera increments to carry an incidental component of the other
    # degree of freedom.  Different-axis R+R is classified first above.
    if max(r1, r2) >= thresholds["rot_high_deg"] and max(d1, d2) >= thresholds["trans_high_m"]:
        return "rotation_translation"
    if r1 <= thresholds["rot_low_deg"] and r2 <= thresholds["rot_low_deg"] and d1 >= thresholds["trans_mid_m"] and d2 >= thresholds["trans_mid_m"]:
        return "translation_translation_commuting_control"
    return "other"


def observed_overlap(record: Any, raw_root: Path, metadata_cache: dict[str, Any]) -> float:
    """Symmetric real-RGB-D frustum overlap for A/C, independent of features."""
    scene = record.scene_id
    if scene not in metadata_cache:
        root = raw_root / scene
        data = np.load(root / "scene_iphone_metadata.npz")
        metadata_cache[scene] = (root, data["trajectories"], data["intrinsics"], [str(v) for v in data["images"].tolist()])
    root, trajectories, intrinsics, names = metadata_cache[scene]

    def directed(source_idx: int, target_idx: int) -> float:
        with Image.open(root / "depth" / (Path(names[source_idx]).stem + ".png")) as image:
            depth = np.asarray(image).astype(np.float64)
        step = max(1, min(depth.shape[:2]) // 80)
        depth = depth[::step, ::step]
        if np.nanmedian(depth[depth > 0]) > 100:
            depth /= 1000.0
        valid = np.isfinite(depth) & (depth > 1e-4)
        if not valid.any():
            return 0.0
        y, x = np.indices(depth.shape)
        k_source, k_target = intrinsics[source_idx], intrinsics[target_idx]
        z = depth[valid]
        xyz = np.stack(((x[valid] * step - k_source[0, 2]) * z / k_source[0, 0], (y[valid] * step - k_source[1, 2]) * z / k_source[1, 1], z), axis=1)
        e_source = camera_to_world_to_extrinsic(trajectories[source_idx])
        e_target = camera_to_world_to_extrinsic(trajectories[target_idx])
        transform = relative_pose(e_source, e_target)
        target_xyz = xyz @ transform[:3, :3].T + transform[:3, 3]
        tz = target_xyz[:, 2]
        u = k_target[0, 0] * target_xyz[:, 0] / np.maximum(tz, 1e-8) + k_target[0, 2]
        v = k_target[1, 1] * target_xyz[:, 1] / np.maximum(tz, 1e-8) + k_target[1, 2]
        with Image.open(root / "depth" / (Path(names[target_idx]).stem + ".png")) as image:
            h, w = np.asarray(image).shape[:2]
        return float(np.mean((tz > 1e-4) & (u >= 0) & (u < w) & (v >= 0) & (v < h)))

    a, _, c = record.raw_frame_indices
    return min(directed(a, c), directed(c, a))


def stratum_rows(
    baseline: str,
    arrays: dict[str, np.ndarray],
    records: list[Any],
    t1: np.ndarray,
    t2: np.ndarray,
    train_thresholds: dict[str, float],
    overlaps: np.ndarray,
) -> list[dict[str, Any]]:
    kappas = np.asarray([kappa(left, right) for left, right in zip(t1, t2)])
    magnitude = np.asarray([np.linalg.norm(se3_log(right @ left)) for left, right in zip(t1, t2)])
    categories: dict[str, list[str]] = {
        "commutator_bin": ["low" if value <= train_thresholds["kappa_low"] else "medium" if value <= train_thresholds["kappa_high"] else "high" for value in kappas],
        "transform_family": [classify_family(left, right, train_thresholds) for left, right in zip(t1, t2)],
        "motion_magnitude": ["low" if value <= train_thresholds["mag_low"] else "medium" if value <= train_thresholds["mag_high"] else "high" for value in magnitude],
        "view_overlap": ["low" if value < 0.25 else "medium" if value < 0.50 else "high" for value in overlaps],
        "scene": [record.scene_id for record in records],
    }
    result: list[dict[str, Any]] = []
    for name, labels in categories.items():
        for label in sorted(set(labels)):
            mask = np.asarray([value == label for value in labels])
            result.append({
                "baseline": baseline,
                "stratum": name,
                "label": label,
                "n": int(mask.sum()),
                "correct_endpoint_cosine": float(arrays["correct_cosine"][mask].mean()),
                "reversed_endpoint_cosine": float(arrays["reversed_cosine"][mask].mean()),
                "grounded_order_margin_cosine": float(arrays["margin_cosine"][mask].mean()),
                "order_ranking_accuracy": float(arrays["ranking"][mask].mean()),
                "recursive_direct_gap_cosine": float(arrays["gap_cosine"][mask].mean()),
                "kappa_mean": float(kappas[mask].mean()),
                "view_overlap_mean": float(overlaps[mask].mean()),
            })
    return result


def plot_results(output: Path, rows: list[dict[str, Any]], test_arrays: dict[str, dict[str, np.ndarray]], t1: np.ndarray, t2: np.ndarray) -> None:
    figure, axis = plt.subplots(figsize=(6.8, 4.7))
    for row in rows:
        axis.scatter(row["correct_endpoint_cosine"], row["grounded_order_margin_cosine"], s=55)
        axis.annotate(row["baseline"], (row["correct_endpoint_cosine"], row["grounded_order_margin_cosine"]), xytext=(4, 3), textcoords="offset points", fontsize=8)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel("correct endpoint cosine distance (lower is better)")
    axis.set_ylabel("grounded order margin (higher is better)")
    axis.set_title("Observed-video correctness vs order grounding")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "correctness_vs_observed_order.png", dpi=180)
    figure.savefig(output / "correctness_vs_observed_order.pdf")
    plt.close(figure)

    kappas = np.asarray([kappa(left, right) for left, right in zip(t1, t2)])
    figure, axis = plt.subplots(figsize=(6.8, 4.7))
    for baseline in ("full", "action_only", "copy"):
        if baseline not in test_arrays:
            continue
        axis.scatter(kappas, test_arrays[baseline]["margin_cosine"], s=22, alpha=0.6, label=baseline)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel(r"κ = ||log(T2 T1 T2⁻¹ T1⁻¹)||")
    axis.set_ylabel("grounded order margin")
    axis.set_title("Order margin vs physical non-commutativity")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "order_margin_vs_kappa.png", dpi=180)
    figure.savefig(output / "order_margin_vs_kappa.pdf")
    plt.close(figure)


def render_report(output: Path, summary_rows: list[dict[str, Any]], pose_rows: list[dict[str, Any]], audit: dict[str, Any]) -> None:
    columns = ("baseline", "one_step_grounding_cosine", "correct_endpoint_cosine", "reversed_order_endpoint_cosine", "grounded_order_margin_cosine", "order_ranking_accuracy", "direct_endpoint_cosine", "recursive_direct_gap_cosine", "copy_normalized_improvement")
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in summary_rows:
        lines.append("| " + " | ".join([str(row["baseline"])] + [f"{row[key]:.4f}" for key in columns[1:]]) + " |")
    pose_columns = ("baseline", "decoded_12_rotation_error_deg", "decoded_12_translation_error_m", "decoded_21_rotation_error_deg", "decoded_21_translation_error_m", "two_branch_pose_assignment_accuracy", "decoded_commutator_rotation_error_deg", "decoded_commutator_translation_error_m")
    pose_lines = ["| " + " | ".join(pose_columns) + " |", "|" + "---|" * len(pose_columns)]
    for row in pose_rows:
        pose_lines.append("| " + " | ".join([str(row["baseline"])] + [f"{row[key]:.4f}" for key in pose_columns[1:]]) + " |")
    report = f"""# Observed-video Grounded Order Protocol — DINO

Status: **experimental / non-canonical**.  This is the primary protocol-v2
track: all endpoint supervision uses the real observed C2 feature `z_C`; the
optional synthetic dual-image manifest is not consulted.

## Contract

- Pose convention: `E` is w2c, `T_AB=E_B@inv(E_A)`, and `T_AC=T_BC@T_AB`.
- Test tuple: real observed `(A,B,C)`; `T1=T_AB`, `T2=T_BC`.
- Training: only observed `A→B` and `B→C` one-step target loss.  No order,
  composition, cycle, path, or rollout loss is included.
- Main analysis: `z_correct=P(P(z_A,T1),T2)`,
  `z_reversed=P(P(z_A,T2),T1)`, `z_direct=P(z_A,T2@T1)`, all compared with
  the real `z_C`.
- Static/dynamic caveat: RGB capture has no per-frame static certificate; this
  is reported risk, not a renderer-derived claim.

## A. Observed-order feature grounding

{chr(10).join(lines)}

`reversed_order` is an intentional evaluation control: its "correct" column is
the full predictor evaluated with `T2` then `T1`, so a negative margin is
expected.  `matched_wrong_pose` uses another scene's same-window action pair;
`shuffled_pose` uses independently seeded derangements.  See
`stratified_observed_order.csv` for κ/family/motion/overlap/scene rows.

`one_step_grounding_cosine` is the mean of real `A→B` and real `B→C`
endpoint distances.  It is reported beside composition so an algebraically
stable copy-like model cannot be mistaken for a grounded transition model.

## B. Two-sided pose grounding

The frozen M4 adapter is trained only on real observed one-step frame pairs and
is then used to decode each predicted two-step feature relative to real `z_A`.

{chr(10).join(pose_lines)}

The observed-pair M4 upper bound and its held-out R² are in
`two_sided_pose_grounding.json`.  Interpret action-only vs full jointly with A:
pose decodability without real-`z_C` endpoint grounding is not sufficient.

## C. Optional dual-image grounding

N/A by design.  `branches.jsonl` nearest-observed synthetic endpoints remain a
residual diagnostic only and were not used here.

## Gates

```json
{json.dumps(audit, indent=2, sort_keys=True)}
```
"""
    (output / "REPORT.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=180)
    parser.add_argument("--pose-max-epochs", type=int, default=240)
    parser.add_argument("--manifest", default="manifests/c_retrain_20260722/inscene_c_h124_prefix8.jsonl")
    parser.add_argument("--c1-dump", default="logs/inscene15k_streaming/runs/c_eval_20260724_1903/c_eval_20260724_1903_action_dynamics_dino_p8_layer0/eval/val_predictions.pt")
    parser.add_argument("--c2-dump", default="logs/inscene15k_streaming/runs/c_eval_20260724_1903/c_eval_20260724_1903_path_integration_dino_p8_layer0/eval/val_predictions.pt")
    parser.add_argument("--raw-root", default="/mnt/umm/users/yinbaiqiao/InsScene-15K/data/processed_scannetpp_v2/processed_scannetpp_v2")
    args = parser.parse_args()
    output = resolve(args.output_dir)
    if output.exists():
        raise SystemExit(f"refusing to overwrite experimental output: {output}")
    output.mkdir(parents=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    cfg = {
        "manifest": str(resolve(args.manifest)), "c1_dump": str(resolve(args.c1_dump)), "c2_dump": str(resolve(args.c2_dump)),
        "raw_scene_root": str(resolve(args.raw_root)), "alignment_atol": 2e-5, "pose_atol": 1e-5,
        "hidden_dim": 64, "learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": 64,
        "max_epochs": args.max_epochs, "patience": 30, "cosine_loss_weight": 0.25, "seed": args.seed,
    }
    records, tuple_audit = load_and_audit_tuples(cfg)
    assignments, split_audit = split_scenes(records, args.seed, 0.7, 0.15)
    partitions = {name: records_for_split(records, assignments, name) for name in ("train", "val", "test")}
    train_state, train_action, _ = pair_tensors(partitions["train"])
    action_mean, action_std = train_action.mean(0), train_action.std(0).clamp_min(1e-5)
    models: dict[str, LatentPredictor] = {}
    train_meta: dict[str, Any] = {}
    for condition in LEARNED_CONDITIONS:
        torch.manual_seed(args.seed)
        model, history = train_model(condition, partitions["train"], partitions["val"], cfg, action_mean, action_std)
        models[condition] = model
        train_meta[condition] = {"epochs": len(history), "best_val_loss": min(row["val_loss"] for row in history), "objective": "one_step_mse_plus_cosine_only"}

    features = torch.stack([record.features for record in partitions["test"]])
    z_a, z_b, z_c = features[:, 0], features[:, 1], features[:, 2]
    feature_std = torch.stack([record.features for record in partitions["train"]]).reshape(-1, features.shape[-1]).std(0).clamp_min(1e-5)
    true_t1, true_t2, _ = transform_batch(partitions["test"])
    donor = matched_donor_mapping(partitions["test"])
    shuffled_1, shuffled_2 = derangement(len(true_t1), args.seed + 101), derangement(len(true_t2), args.seed + 211)
    action_sets = {
        "copy": (true_t1, true_t2), "without_condition": (true_t1, true_t2), "action_only": (true_t1, true_t2), "full": (true_t1, true_t2),
        "matched_wrong_pose": (true_t1[donor], true_t2[donor]),
        "shuffled_pose": (true_t1[shuffled_1], true_t2[shuffled_2]),
    }
    copy_first = learned_predict("copy", models, z_a, true_t1, action_mean, action_std)
    copy_correct = learned_predict("copy", models, copy_first, true_t2, action_mean, action_std)
    copy_error = float(cosine_distance(copy_correct, z_c).mean())
    summary_rows: list[dict[str, Any]] = []
    test_arrays: dict[str, dict[str, np.ndarray]] = {}
    predicted_pairs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for baseline, (t1, t2) in action_sets.items():
        first = learned_predict(baseline, models, z_a, t1, action_mean, action_std)
        one_step_bc = learned_predict(baseline, models, z_b, t2, action_mean, action_std)
        correct = learned_predict(baseline, models, first, t2, action_mean, action_std)
        reverse_first = learned_predict(baseline, models, z_a, t2, action_mean, action_std)
        reversed_ = learned_predict(baseline, models, reverse_first, t1, action_mean, action_std)
        direct = learned_predict(baseline, models, z_a, np.stack([right @ left for left, right in zip(t1, t2)]), action_mean, action_std)
        summary, arrays = mean_metrics(correct, reversed_, direct, z_c, feature_std, copy_error)
        summary.update(one_step_metrics(first, one_step_bc, z_b, z_c))
        summary["baseline"] = baseline
        summary_rows.append(summary)
        test_arrays[baseline] = arrays
        predicted_pairs[baseline] = (correct, reversed_)
    # Explicit action-order control: intentionally evaluate full with the swapped trajectory.
    full_correct, full_reverse = predicted_pairs["full"]
    reverse_summary, reverse_arrays = mean_metrics(full_reverse, full_correct, learned_predict("full", models, z_a, np.stack([left @ right for left, right in zip(true_t1, true_t2)]), action_mean, action_std), z_c, feature_std, copy_error)
    reverse_summary.update(one_step_metrics(
        learned_predict("full", models, z_a, true_t2, action_mean, action_std),
        learned_predict("full", models, z_b, true_t1, action_mean, action_std), z_b, z_c,
    ))
    reverse_summary["baseline"] = "reversed_order"
    summary_rows.insert(-1, reverse_summary)
    test_arrays["reversed_order"] = reverse_arrays

    # Train Q on observed one-step pairs only.
    def pose_pairs(name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        states, actions, targets = pair_tensors(partitions[name])
        return states, targets, actions
    train_s, train_t, train_xi = pose_pairs("train")
    val_s, val_t, val_xi = pose_pairs("val")
    test_s, test_t, test_xi = pose_pairs("test")
    pose_mean, pose_std = train_xi.mean(0), train_xi.std(0).clamp_min(1e-6)
    adapter, adapter_meta = train_pose_adapter(train_s, train_t, (train_xi - pose_mean) / pose_std, val_s, val_t, (val_xi - pose_mean) / pose_std, args.seed, args.pose_max_epochs)
    with torch.no_grad():
        upper_prediction = adapter(test_s, test_t)
    upper = pose_r2_metrics((test_xi - pose_mean) / pose_std, upper_prediction, pose_mean, pose_std)
    pose_rows: list[dict[str, Any]] = []
    pose_arrays: dict[str, dict[str, np.ndarray]] = {}
    for baseline in ("action_only", "full", "copy", "matched_wrong_pose", "shuffled_pose"):
        p_summary, p_arrays = two_sided_pose_metrics(adapter, z_a, *predicted_pairs[baseline], true_t1, true_t2, pose_mean, pose_std)
        p_summary["baseline"] = baseline
        pose_rows.append(p_summary)
        pose_arrays[baseline] = p_arrays

    train_t1, train_t2, _ = transform_batch(partitions["train"])
    train_kappa = np.asarray([kappa(left, right) for left, right in zip(train_t1, train_t2)])
    train_magnitude = np.asarray([np.linalg.norm(se3_log(right @ left)) for left, right in zip(train_t1, train_t2)])
    train_rot = np.asarray([rotation_degrees(value) for value in np.concatenate((train_t1, train_t2))])
    train_trans = np.asarray([translation_norm(value) for value in np.concatenate((train_t1, train_t2))])
    thresholds = {
        "kappa_low": float(np.quantile(train_kappa, 1 / 3)), "kappa_high": float(np.quantile(train_kappa, 2 / 3)),
        "mag_low": float(np.quantile(train_magnitude, 1 / 3)), "mag_high": float(np.quantile(train_magnitude, 2 / 3)),
        "rot_low_deg": float(np.quantile(train_rot, .25)), "rot_mid_deg": float(np.quantile(train_rot, .5)), "rot_high_deg": float(np.quantile(train_rot, .75)),
        "trans_mid_m": float(np.quantile(train_trans, .5)), "trans_high_m": float(np.quantile(train_trans, .75)),
    }
    cache: dict[str, Any] = {}
    overlaps = np.asarray([observed_overlap(record, resolve(args.raw_root), cache) for record in partitions["test"]])
    stratified: list[dict[str, Any]] = []
    for baseline in BASELINES:
        stratified.extend(stratum_rows(baseline, test_arrays[baseline], partitions["test"], true_t1, true_t2, thresholds, overlaps))
    full_family_counts = {
        str(row["label"]): int(row["n"])
        for row in stratified
        if row["baseline"] == "full" and row["stratum"] == "transform_family"
    }
    requested_families = (
        "rotation_translation",
        "different_axis_rotation_rotation",
        "translation_translation_commuting_control",
    )
    min_family_support = 5
    family_support = {name: int(full_family_counts.get(name, 0)) for name in requested_families}
    support_gate = all(value >= min_family_support for value in family_support.values())
    audit = {
        "pose_convention": POSE_CONVENTION,
        "tuple_alignment": tuple_audit["gates"],
        "scene_split": split_audit,
        "training_one_step_only": True,
        "optional_dual_image_grounding": "N/A_not_used",
        "observed_tuple_feature_coverage": {"tuples": len(records), "test_tuples": len(partitions["test"]), "complete": True},
        "observed_rollout_coverage": {
            "available_real_feature_steps": 2,
            "required_for_1_to_4_step_curve": 4,
            "status": "insufficient_C2_tuple_artifact",
        },
        "stratification_support": {
            "minimum_per_requested_family": min_family_support,
            "counts": family_support,
            "gate": support_gate,
            "reason_if_failed": "Natural C2 held-out tuples do not supply enough requested rotation-translation, different-axis rotation-rotation, and translation-translation controls." if not support_gate else None,
        },
        "representative_cross_model_pilot": {
            "authorized": False,
            "status": "blocked_pending_more_observed_tuple_coverage",
            "blockers": [
                *([] if support_gate else ["requested_motion_family_support_below_minimum"]),
                "observed_C2_artifact_has_only_two_real_transition_steps",
            ],
        },
        "matched_wrong_pose": {"fixed_points": int(np.sum(donor == np.arange(len(donor)))), "same_scene_donors": int(sum(partitions["test"][i].scene_id == partitions["test"][j].scene_id for i, j in enumerate(donor)))},
        "shuffled_pose": {"step1_fixed_points": int(np.sum(shuffled_1 == np.arange(len(shuffled_1)))), "step2_fixed_points": int(np.sum(shuffled_2 == np.arange(len(shuffled_2))))},
        "static_scene_risk": "Observed ScanNet++ RGB has no per-frame dynamic/static certificate.",
    }
    if audit["matched_wrong_pose"]["fixed_points"] or audit["matched_wrong_pose"]["same_scene_donors"] or any(audit["shuffled_pose"].values()):
        raise RuntimeError(f"pose-control derangement gate failed: {audit}")
    config = {**cfg, "output_dir": str(output), "pose_max_epochs": args.pose_max_epochs, "seed": args.seed}
    dump_json(output / "config.json", config)
    dump_json(output / "audit.json", audit)
    dump_json(output / "training.json", train_meta)
    torch.save({name: model.state_dict() for name, model in models.items()}, output / "one_step_predictors.pt")
    torch.save(adapter.state_dict(), output / "two_sided_pose_adapter.pt")
    dump_json(output / "two_sided_pose_grounding.json", {"observed_pair_m4_upper_bound": upper, "adapter_training": {key: value for key, value in adapter_meta.items() if key != "history"}, "predicted_feature_pose_grounding": pose_rows, "full_minus_action_only": {key: next(r for r in pose_rows if r["baseline"] == "full")[key] - next(r for r in pose_rows if r["baseline"] == "action_only")[key] for key in pose_rows[0] if key != "baseline"}})
    dump_json(output / "strata_thresholds.json", thresholds)
    write_csv(output / "observed_order_metrics.csv", summary_rows)
    write_csv(output / "stratified_observed_order.csv", stratified)
    write_csv(output / "two_sided_pose_metrics.csv", pose_rows)
    sample_rows = [{"sample_id": r.sample_id, "scene_id": r.scene_id, "split": assignments[r.scene_id], "frame_A": r.raw_frame_indices[0], "frame_B": r.raw_frame_indices[1], "frame_C": r.raw_frame_indices[2], "view_overlap_AC": float(overlap)} for r, overlap in zip(partitions["test"], overlaps)]
    write_csv(output / "test_sample_manifest.csv", sample_rows)
    provenance = {"python": sys.version, "seed": args.seed, "inputs": {key: {"path": value, "sha256": sha256(Path(value))} for key, value in {"manifest": cfg["manifest"], "c1_dump": cfg["c1_dump"], "c2_dump": cfg["c2_dump"], "runner": str(Path(__file__).resolve())}.items()}}
    dump_json(output / "provenance.json", provenance)
    plot_results(output, summary_rows, test_arrays, true_t1, true_t2)
    render_report(output, summary_rows, pose_rows, audit)
    print(json.dumps({"output": str(output), "observed_order": summary_rows, "pose": pose_rows, "upper_bound": upper}, indent=2))


if __name__ == "__main__":
    main()
