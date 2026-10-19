#!/usr/bin/env python3
"""Run the legacy M3/M4 primitive-frame DINO smoke diagnostic.

It is intentionally neither a grounded-order result nor a SeeSE3 replication:
it uses DINO's only exported endpoint representation (layer 0), a small
scene-disjoint C2 subset, and one seed.  The observed-video protocol has no
renderer prerequisite; the separate replication preflight records whether the
required temporal split, representative middle layer, pair count, and seeds
are available.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from analyses.se3_latent_mvp.run_mvp import (
    load_and_audit_tuples,
    records_for_split,
    split_scenes,
)
from analyses.se3_latent_mvp.geometry import se3_log
from analyses.se3_protocol_v2.protocol import dump_json, sha256


REPO_ROOT = Path(__file__).resolve().parents[2]


class PoincareAdapter(nn.Module):
    """SeeSE3's requested Siamese adapter: d→64→20 then bias-free 20→6."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 20),
        )
        self.readout = nn.Linear(20, 6, bias=False)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.readout(self.phi(target) - self.phi(source))


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def pair_arrays(records: list[Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    states, targets, twists = [], [], []
    for record in records:
        t_ab, t_bc, _ = record.transforms
        states.extend((record.features[0], record.features[1]))
        targets.extend((record.features[1], record.features[2]))
        twists.extend((se3_log(t_ab), se3_log(t_bc)))
    return torch.stack(states).float(), torch.stack(targets).float(), torch.from_numpy(np.stack(twists)).float()


def r2(target: torch.Tensor, prediction: torch.Tensor) -> float:
    target, prediction = target.float(), prediction.float()
    denom = torch.sum((target - target.mean(dim=0, keepdim=True)) ** 2)
    return float(1.0 - torch.sum((target - prediction) ** 2) / denom.clamp_min(1e-12))


def metrics(target_z: torch.Tensor, pred_z: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> dict[str, float]:
    target = target_z * std + mean
    prediction = pred_z * std + mean
    target_t, prediction_t = target[:, 3:], prediction[:, 3:]
    valid = torch.linalg.norm(target_t, dim=-1) > 1e-6
    target_direction = target_t[valid] / torch.linalg.norm(target_t[valid], dim=-1, keepdim=True)
    prediction_direction = prediction_t[valid] / torch.linalg.norm(prediction_t[valid], dim=-1, keepdim=True).clamp_min(1e-8)
    return {
        "overall_zscore_r2": r2(target_z, pred_z),
        "rotation_zscore_r2": r2(target_z[:, :3], pred_z[:, :3]),
        "translation_zscore_r2": r2(target_z[:, 3:], pred_z[:, 3:]),
        "translation_direction_r2": r2(target_direction, prediction_direction),
        "translation_magnitude_r2": r2(torch.linalg.norm(target_t, dim=-1, keepdim=True), torch.linalg.norm(prediction_t, dim=-1, keepdim=True)),
    }


def fit_m3(
    train_source: torch.Tensor,
    train_target: torch.Tensor,
    train_y: torch.Tensor,
    val_source: torch.Tensor,
    val_target: torch.Tensor,
    val_y: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Val-selected ridge is still the raw bias-free linear M3 readout.

    With 192 train primitives and 1,024 latent dimensions, unregularized least
    squares is underdetermined; the ridge strength is selected on held-out
    scenes, never on the test scenes.
    """
    delta = (train_target - train_source).double()
    val_delta = (val_target - val_source).double()
    kernel = delta @ delta.T
    scale = float(torch.trace(kernel) / len(delta))
    identity = torch.eye(len(delta), dtype=torch.float64)
    best: tuple[float, torch.Tensor, float] | None = None
    for multiplier in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0):
        alpha = multiplier * max(scale, 1e-12)
        weight = delta.T @ torch.linalg.solve(kernel + alpha * identity, train_y.double())
        val_loss = float(torch.mean((val_delta @ weight - val_y.double()) ** 2))
        if best is None or val_loss < best[0]:
            best = (val_loss, weight.float(), alpha)
    if best is None:
        raise RuntimeError("M3 ridge search failed")
    return (target - source) @ best[1], {"selected_ridge_alpha": best[2], "best_val_mse": best[0]}


def fit_m4(
    train_source: torch.Tensor,
    train_target: torch.Tensor,
    train_y: torch.Tensor,
    val_source: torch.Tensor,
    val_target: torch.Tensor,
    val_y: torch.Tensor,
    test_source: torch.Tensor,
    test_target: torch.Tensor,
    seed: int,
    max_epochs: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    torch.manual_seed(seed)
    model = PoincareAdapter(int(train_source.shape[-1]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state: dict[str, torch.Tensor] | None = None
    best_val, stale = float("inf"), 0
    history: list[dict[str, float]] = []
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(max_epochs):
        model.train()
        order = torch.randperm(len(train_source), generator=generator)
        losses = []
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
        if val_loss < best_val - 1e-7:
            best_val, stale = val_loss, 0
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 30:
            break
    if best_state is None:
        raise RuntimeError("M4 did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = model(test_source, test_target)
    return prediction, {"best_val_mse": best_val, "epochs": len(history), "history": history}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=240)
    parser.add_argument("--manifest", default="manifests/c_retrain_20260722/inscene_c_h124_prefix8.jsonl")
    parser.add_argument("--c1-dump", default="logs/inscene15k_streaming/runs/c_eval_20260724_1903/c_eval_20260724_1903_action_dynamics_dino_p8_layer0/eval/val_predictions.pt")
    parser.add_argument("--c2-dump", default="logs/inscene15k_streaming/runs/c_eval_20260724_1903/c_eval_20260724_1903_path_integration_dino_p8_layer0/eval/val_predictions.pt")
    parser.add_argument("--raw-root", default="/mnt/umm/users/yinbaiqiao/InsScene-15K/data/processed_scannetpp_v2/processed_scannetpp_v2")
    args = parser.parse_args()
    output = resolve(args.output_dir)
    destination = output / "seese3_m3_m4_primitive_smoke.json"
    if not output.is_dir():
        raise SystemExit(f"protocol output must already exist: {output}")
    if destination.exists():
        raise SystemExit(f"refusing to overwrite smoke output: {destination}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    cfg = {
        "manifest": str(resolve(args.manifest)), "c1_dump": str(resolve(args.c1_dump)), "c2_dump": str(resolve(args.c2_dump)),
        "raw_scene_root": str(resolve(args.raw_root)), "alignment_atol": 2e-5, "pose_atol": 1e-5,
    }
    records, tuple_audit = load_and_audit_tuples(cfg)
    assignment, split_audit = split_scenes(records, args.seed, 0.7, 0.15)
    partitions = {name: records_for_split(records, assignment, name) for name in ("train", "val", "test")}
    train_s, train_t, train_xi = pair_arrays(partitions["train"])
    val_s, val_t, val_xi = pair_arrays(partitions["val"])
    test_s, test_t, test_xi = pair_arrays(partitions["test"])
    mean, std = train_xi.mean(0), train_xi.std(0).clamp_min(1e-6)
    train_y, val_y, test_y = (train_xi - mean) / std, (val_xi - mean) / std, (test_xi - mean) / std
    m3_prediction, m3_meta = fit_m3(train_s, train_t, train_y, val_s, val_t, val_y, test_s, test_t)
    m4_prediction, m4_meta = fit_m4(train_s, train_t, train_y, val_s, val_t, val_y, test_s, test_t, args.seed, args.max_epochs)
    output_data = {
        "status": "primitive_observed_frame_smoke_only",
        "not_a_grounded_order_or_navigation_result": True,
        "feature": {"model": "dino", "layer": 0, "prefix": 8, "role": "C2 isolated target"},
        "method": {
            "M3": "xi_zscore = W @ (z_B - z_A), bias-free least-squares readout",
            "M4": "xi_zscore = W @ (phi(z_B) - phi(z_A)); phi=Linear(d,64),LayerNorm,GELU,Linear(64,20); W=bias-free Linear(20,6)",
            "target_normalization": "mean/std from train primitives only",
        },
        "counts": {name: {"scenes": len({r.scene_id for r in value}), "tuples": len(value), "one_step_primitives": 2 * len(value)} for name, value in partitions.items()},
        "split_audit": split_audit,
        "tuple_audit_gates": tuple_audit["gates"],
        "M3": metrics(test_y, m3_prediction, mean, std),
        "M3_training": m3_meta,
        "M4": metrics(test_y, m4_prediction, mean, std),
        "M4_training": {key: value for key, value in m4_meta.items() if key != "history"},
        "provenance": {key: {"path": cfg[key], "sha256": sha256(Path(cfg[key]))} for key in ("manifest", "c1_dump", "c2_dump")},
        "composite_target_used_for_training": False,
        "grounded_order_metrics": "N/A: this legacy primitive smoke is not the observed-video order runner",
    }
    dump_json(destination, output_data)
    with (output / "REPORT.md").open("a") as handle:
        handle.write("\n## SeeSE3 primitive-frame DINO smoke\n\n")
        handle.write("This is a scene-disjoint observed one-step probe only, not an approximate endpoint order score or a SeeSE3 replication.\n\n")
        handle.write("| probe | overall R² | rotation R² | translation R² | direction R² | magnitude R² | grounded assignment |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---|\n")
        for name in ("M3", "M4"):
            values = output_data[name]
            handle.write(
                f"| {name} | {values['overall_zscore_r2']:.4f} | {values['rotation_zscore_r2']:.4f} | "
                f"{values['translation_zscore_r2']:.4f} | {values['translation_direction_r2']:.4f} | "
                f"{values['translation_magnitude_r2']:.4f} | N/A |\n"
            )
        handle.write("Grounded order metrics are reported by `observed_order.py`; they are not inferred from this legacy smoke.  `seese3_replication.py` owns replication eligibility.\n")
    print(json.dumps({"output": str(destination), "M3": output_data["M3"], "M4": output_data["M4"]}, indent=2))


if __name__ == "__main__":
    main()
