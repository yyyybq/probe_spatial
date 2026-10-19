#!/usr/bin/env python3
"""One-step-only recurrent diagnostics for Wan/V-JEPA2 exact canonical-v2 cells.

The recurrent hidden tuple is the only state accepted by ``update_state``.
Isolated A/B/C features are decoder-space supervision/evaluation and are never
fed to the context initializer during a recursive rollout.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from analyses.se3_latent_mvp.geometry import encode_relative_pose_9d
from analyses.se3_protocol_v2.metrics import cosine_distance
from analyses.se3_protocol_v2.wan_vjepa2_v2_data import RESULT, load_protocol, read_jsonl
from vidfm3d.models.components.probe_path_integration import PathIntegrationProbe


BASELINES = (
    "copy", "without_condition", "action_only", "matched_shuffle",
    "commutative_additive", "full",
)


class AdditiveUpdater(nn.Module):
    """Order-commutative diagnostic ``z <- z + g(a)``."""

    def __init__(self, channels: int, hidden_dim: int) -> None:
        super().__init__()
        self.increment = nn.Sequential(
            nn.Linear(9, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, channels)
        )

    def update(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return state + self.increment(action)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.update(state, action)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    Path(temporary).replace(path)
    path.chmod(0o644)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    Path(temporary).replace(path)
    path.chmod(0o644)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def objective(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(prediction, target) + (1.0 - F.cosine_similarity(prediction, target, dim=-1)).mean()


def cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _mean_std(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return value.mean(0), value.std(0, unbiased=False).clamp_min(1e-5)


def scene_macro(value: np.ndarray, scenes: np.ndarray) -> float:
    return float(np.mean([value[scenes == scene].mean() for scene in sorted(set(scenes.tolist()))]))


def matched_donors(rows: list[dict[str, Any]]) -> np.ndarray:
    """Exact within-window permutation with no fixed or same-scene donor."""
    donors = np.full(len(rows), -1, dtype=np.int64)
    blocks: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        blocks[int(row["streaming_window_id"])].append(index)
    for indices in blocks.values():
        ordered = sorted(indices, key=lambda index: str(rows[index]["scene_id"]))
        if len({rows[index]["scene_id"] for index in ordered}) < 2:
            raise RuntimeError("matched-shuffle block contains fewer than two scenes")
        shifted = ordered[1:] + ordered[:1]
        for recipient, donor in zip(ordered, shifted):
            if rows[recipient]["scene_id"] == rows[donor]["scene_id"]:
                raise RuntimeError("matched shuffle selected a same-scene donor")
            donors[recipient] = donor
    if sorted(donors.tolist()) != list(range(len(rows))) or np.any(donors == np.arange(len(rows))):
        raise RuntimeError("matched shuffle is not a fixed-point-free exact permutation")
    return donors


def _path_model(channels: int, condition: str, cfg: dict[str, Any]) -> PathIntegrationProbe:
    return PathIntegrationProbe(
        in_channels=channels, action_dim=9, hidden_dim=int(cfg["hidden_dim"]),
        num_layers=int(cfg["num_layers"]), dropout=float(cfg["dropout"]),
        conditioning=condition,
    )


def _validation_loss(
    model: nn.Module, context: torch.Tensor, z_a: torch.Tensor, action: torch.Tensor,
    target: torch.Tensor, additive: bool,
) -> float:
    model.eval()
    with torch.no_grad():
        if additive:
            prediction = model(z_a, action)
        else:
            prediction = model(context[:, None, None, None, :], action[:, None, :])[:, 0]
        return float(objective(prediction, target))


def train_head(
    *, kind: str, channels: int, seed: int, cfg: dict[str, Any], device: torch.device,
    train_context: torch.Tensor, train_za: torch.Tensor, train_action: torch.Tensor,
    train_target: torch.Tensor, val_context: torch.Tensor, val_za: torch.Tensor,
    val_action: torch.Tensor, val_target: torch.Tensor,
) -> tuple[nn.Module, dict[str, Any]]:
    seed_everything(seed + {"full": 0, "without_condition": 1000, "action_only": 2000, "commutative_additive": 3000}[kind])
    additive = kind == "commutative_additive"
    condition = {"without_condition": "visual_only"}.get(kind, kind)
    model: nn.Module = AdditiveUpdater(channels, int(cfg["hidden_dim"])) if additive else _path_model(channels, condition, cfg)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"])
    )
    generator = torch.Generator().manual_seed(seed + 8000)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch, best_value, stale = -1, float("inf"), 0
    history: list[dict[str, float | int]] = []
    batch_size = int(cfg["batch_size"])
    for epoch in range(int(cfg["max_epochs"])):
        model.train()
        order = torch.randperm(len(train_target), generator=generator)
        losses = []
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size].to(device)
            if additive:
                prediction = model(train_za[index], train_action[index])
            else:
                prediction = model(
                    train_context[index, None, None, None, :], train_action[index, None, :]
                )[:, 0]
            # Registered training signal: only the real single-step B target.
            loss = objective(prediction, train_target[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["gradient_clip_norm"]))
            optimizer.step()
            losses.append(float(loss.detach()))
        value = _validation_loss(
            model, val_context, val_za, val_action, val_target, additive
        )
        history.append({"epoch": epoch, "train_one_step_B_loss": float(np.mean(losses)), "validation_one_step_B_loss": value})
        if value < best_value - 1e-7:
            best_value, best_epoch, stale = value, epoch, 0
            best_state = cpu_state(model)
        else:
            stale += 1
        if stale >= int(cfg["patience"]):
            break
    if best_state is None:
        raise RuntimeError(f"{kind}: no validation-selected checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "head": kind, "seed": seed, "best_epoch": best_epoch,
        "best_validation_one_step_B_loss": best_value, "epochs_run": len(history),
        "supervision": "A_context_plus_T_AB_to_isolated_B_only",
        "no_composition_order_or_cycle_loss": True, "history": history,
    }


def _denormalize(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return value * std + mean


def _cos_np(left: torch.Tensor, right: torch.Tensor) -> np.ndarray:
    return cosine_distance(left.float(), right.float()).detach().cpu().numpy().astype(np.float64)


def evaluate_baseline(
    baseline: str, models: dict[str, nn.Module], context: torch.Tensor, z_a: torch.Tensor,
    z_b_raw: torch.Tensor, z_c_raw: torch.Tensor, action1: torch.Tensor, action2: torch.Tensor,
    action_composite: torch.Tensor, target_mean: torch.Tensor, target_std: torch.Tensor,
) -> tuple[dict[str, np.ndarray], dict[str, torch.Tensor]]:
    if baseline == "copy":
        one_n = seq_n = direct_n = swap_n = z_a
    elif baseline == "commutative_additive":
        model = models[baseline]
        with torch.no_grad():
            one_n = model.update(z_a, action1)
            # Independently execute both paths. Never force one tensor equal to the other.
            seq_n = model.update(model.update(z_a, action1), action2)
            swap_n = model.update(model.update(z_a, action2), action1)
            direct_n = model.update(z_a, action_composite)
    else:
        model_name = "full" if baseline == "matched_shuffle" else baseline
        model = models[model_name]
        with torch.no_grad():
            state = model.initialize_state(context[:, None, None, None, :])
            first_state = model.update_state(state, action1)
            one_n = model.decode_state(first_state)
            seq_n = model.decode_state(model.update_state(first_state, action2))
            direct_n = model.decode_state(model.update_state(state, action_composite))
            swapped_first = model.update_state(state, action2)
            swap_n = model.decode_state(model.update_state(swapped_first, action1))
    predictions_n = {"one_ab": one_n, "sequential": seq_n, "direct": direct_n, "swapped": swap_n}
    predictions = {key: _denormalize(value, target_mean, target_std) for key, value in predictions_n.items()}
    seq_error = _cos_np(predictions["sequential"], z_c_raw)
    swap_error = _cos_np(predictions["swapped"], z_c_raw)
    arrays = {
        "one_step_error": _cos_np(predictions["one_ab"], z_b_raw),
        "sequential_endpoint_error": seq_error,
        "direct_endpoint_error": _cos_np(predictions["direct"], z_c_raw),
        "sequential_direct_prediction_distance": _cos_np(predictions["sequential"], predictions["direct"]),
        "recursive_minus_direct_endpoint_error": seq_error - _cos_np(predictions["direct"], z_c_raw),
        "swapped_endpoint_error": swap_error,
        "raw_order_margin": swap_error - seq_error,
    }
    return arrays, predictions


def summarize(
    arrays: dict[str, np.ndarray], copy_arrays: dict[str, np.ndarray], scenes: np.ndarray,
    tie_tolerance: float, epsilon: float,
) -> dict[str, Any]:
    margin = arrays["raw_order_margin"]
    wins = margin > tie_tolerance
    losses = margin < -tie_tolerance
    ties = ~(wins | losses)
    copy_endpoint = scene_macro(copy_arrays["sequential_endpoint_error"], scenes)
    endpoint = scene_macro(arrays["sequential_endpoint_error"], scenes)
    return {
        "aggregation": "scene_macro",
        "one_step_error": scene_macro(arrays["one_step_error"], scenes),
        "sequential_endpoint_error": endpoint,
        "direct_endpoint_error": scene_macro(arrays["direct_endpoint_error"], scenes),
        "sequential_direct_prediction_distance": scene_macro(arrays["sequential_direct_prediction_distance"], scenes),
        "recursive_minus_direct_endpoint_error": scene_macro(arrays["recursive_minus_direct_endpoint_error"], scenes),
        "swapped_endpoint_error": scene_macro(arrays["swapped_endpoint_error"], scenes),
        "raw_order_margin": scene_macro(margin, scenes),
        "normalized_order_margin": scene_macro(margin, scenes) / (copy_endpoint + epsilon),
        "two_step_grounding_gain_vs_copy": (copy_endpoint - endpoint) / (copy_endpoint + epsilon),
        "wins": int(wins.sum()), "losses": int(losses.sum()), "ties": int(ties.sum()),
        "tie_aware_accuracy": float((wins.sum() + 0.5 * ties.sum()) / len(margin)),
        "strict_accuracy": float(wins.mean()), "copy_endpoint_denominator": copy_endpoint,
    }


def run_seed(
    cell: str, seed: int, tensors: dict[str, torch.Tensor], geometry: dict[str, np.ndarray],
    rows: list[dict[str, Any]], protocol: dict[str, Any], device: torch.device,
) -> None:
    output = RESULT / "cells" / cell / "seeds" / f"seed{seed}"
    done = output / "COMPLETE.json"
    if done.exists() and json.loads(done.read_text()).get("status") == "PASS":
        return
    output.mkdir(parents=True, exist_ok=True)
    cfg = protocol["training"]
    split = np.asarray([row["analysis_split"] for row in rows])
    train_i = np.flatnonzero(split == "train")
    val_i = np.flatnonzero(split == "validation")
    eval_i = np.flatnonzero(split == "evaluation")
    if not (len(train_i) == 128 and len(val_i) == 32 and len(eval_i) == 276):
        raise RuntimeError("frozen split cardinality changed")

    raw = {key: value.float() for key, value in tensors.items()}
    context_mean, context_std = _mean_std(raw["context"][train_i])
    target_mean, target_std = _mean_std(raw["z_b"][train_i])
    action1_raw = torch.from_numpy(np.asarray(geometry["action1"], dtype=np.float32))
    action2_raw = torch.from_numpy(np.asarray(geometry["action2"], dtype=np.float32))
    action_mean, action_std = _mean_std(action1_raw[train_i])
    composite_raw = torch.from_numpy(np.stack([
        encode_relative_pose_9d(right @ left).astype(np.float32)
        for left, right in zip(geometry["t1"], geometry["t2"])
    ]))

    context = ((raw["context"] - context_mean) / context_std).to(device)
    z_a = ((raw["z_a"] - target_mean) / target_std).to(device)
    z_b = ((raw["z_b"] - target_mean) / target_std).to(device)
    action1 = ((action1_raw - action_mean) / action_std).to(device)
    action2 = ((action2_raw - action_mean) / action_std).to(device)
    composite = ((composite_raw - action_mean) / action_std).to(device)
    channels = int(raw["context"].shape[-1])
    train_t = torch.as_tensor(train_i, dtype=torch.long, device=device)
    val_t = torch.as_tensor(val_i, dtype=torch.long, device=device)
    eval_t = torch.as_tensor(eval_i, dtype=torch.long, device=device)
    models: dict[str, nn.Module] = {}
    training_records = []
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    for kind in ("full", "without_condition", "action_only", "commutative_additive"):
        model, record = train_head(
            kind=kind, channels=channels, seed=seed, cfg=cfg, device=device,
            train_context=context[train_t], train_za=z_a[train_t], train_action=action1[train_t],
            train_target=z_b[train_t], val_context=context[val_t], val_za=z_a[val_t],
            val_action=action1[val_t], val_target=z_b[val_t],
        )
        models[kind] = model
        training_records.append(record)
        torch.save({
            "state_dict": cpu_state(model), "cell": cell, "seed": seed, "head": kind,
            "channels": channels, "training_record": record,
            "normalization": {
                "context_mean": context_mean, "context_std": context_std,
                "target_B_mean": target_mean, "target_B_std": target_std,
                "action_AB_mean": action_mean, "action_AB_std": action_std,
            },
            "supervision": "single_step_B_only",
        }, checkpoint_dir / f"{kind}.pt")

    eval_rows = [rows[index] for index in eval_i]
    donors = matched_donors(eval_rows)
    arrays_by_baseline: dict[str, dict[str, np.ndarray]] = {}
    saved_predictions: dict[str, np.ndarray] = {}
    additive_fp32_delta = float("nan")
    additive_fp32_roundoff_bound = float("nan")
    eval_context, eval_za = context[eval_t], z_a[eval_t]
    eval_a1, eval_a2, eval_comp = action1[eval_t], action2[eval_t], composite[eval_t]
    eval_zb_raw, eval_zc_raw = raw["z_b"][eval_i].to(device), raw["z_c"][eval_i].to(device)
    for baseline in BASELINES:
        if baseline == "matched_shuffle":
            donor_t = torch.as_tensor(donors, device=device)
            a1, a2 = eval_a1[donor_t], eval_a2[donor_t]
            comp = torch.from_numpy(np.stack([
                encode_relative_pose_9d(right @ left).astype(np.float32)
                for left, right in zip(
                    np.asarray(geometry["t1"])[eval_i][donors],
                    np.asarray(geometry["t2"])[eval_i][donors],
                )
            ])).to(device)
            comp = (comp - action_mean.to(device)) / action_std.to(device)
        else:
            a1, a2, comp = eval_a1, eval_a2, eval_comp
        arrays, predictions = evaluate_baseline(
            baseline, models, eval_context, eval_za, eval_zb_raw, eval_zc_raw,
            a1, a2, comp, target_mean.to(device), target_std.to(device),
        )
        arrays_by_baseline[baseline] = arrays
        if baseline == "commutative_additive":
            seq32 = predictions["sequential"].detach().float()
            swap32 = predictions["swapped"].detach().float()
            additive_fp32_delta = float((seq32 - swap32).abs().max())
            scale = max(1.0, float(torch.maximum(seq32.abs().max(), swap32.abs().max())))
            additive_fp32_roundoff_bound = 32.0 * torch.finfo(torch.float32).eps * scale
            if additive_fp32_delta > additive_fp32_roundoff_bound:
                raise RuntimeError(
                    "independently computed additive paths exceed the float32 "
                    f"roundoff bound: delta={additive_fp32_delta}, "
                    f"bound={additive_fp32_roundoff_bound}"
                )
        for name, value in predictions.items():
            saved_predictions[f"{baseline}__{name}"] = value.detach().cpu().numpy().astype(np.float16)

    additive_delta = float(np.max(np.abs(
        saved_predictions["commutative_additive__sequential"].astype(np.float32)
        - saved_predictions["commutative_additive__swapped"].astype(np.float32)
    )))
    scenes = np.asarray([row["scene_id"] for row in eval_rows])
    stats = protocol["statistics"]
    metric_rows, sample_rows = [], []
    copy_arrays = arrays_by_baseline["copy"]
    without_arrays = arrays_by_baseline["without_condition"]
    without_one = scene_macro(without_arrays["one_step_error"], scenes)
    without_two = scene_macro(without_arrays["sequential_endpoint_error"], scenes)
    for baseline in BASELINES:
        arrays = arrays_by_baseline[baseline]
        summary = summarize(
            arrays, copy_arrays, scenes, float(stats["tie_tolerance"]),
            float(stats["normalization_epsilon"]),
        )
        summary.update({
            "cell": cell, "seed": seed, "baseline": baseline,
            "one_step_gain_vs_without_condition": (
                without_one - summary["one_step_error"]
            ) / (without_one + float(stats["normalization_epsilon"])),
            "two_step_gain_vs_without_condition": (
                without_two - summary["sequential_endpoint_error"]
            ) / (without_two + float(stats["normalization_epsilon"])),
        })
        metric_rows.append(summary)
        for index, row in enumerate(eval_rows):
            item = {
                "cell": cell, "seed": seed, "baseline": baseline,
                "sample_id": row["sample_id"], "scene_id": row["scene_id"],
                "window_id": int(row["streaming_window_id"]),
                "noncomm_strength": float(np.asarray(geometry["noncomm_strength"])[eval_i[index]]),
            }
            item.update({key: float(value[index]) for key, value in arrays.items()})
            item["tie_score"] = 1.0 if item["raw_order_margin"] > float(stats["tie_tolerance"]) else (0.0 if item["raw_order_margin"] < -float(stats["tie_tolerance"]) else 0.5)
            sample_rows.append(item)
    np.savez_compressed(
        output / "raw_predictions.npz", sample_id=np.asarray([row["sample_id"] for row in eval_rows]),
        **saved_predictions,
    )
    write_csv(output / "per_sample_metrics.csv", sample_rows)
    write_csv(output / "seed_metrics.csv", metric_rows)
    atomic_json(output / "training_history.json", training_records)
    atomic_json(output / "controls.json", {
        "matched_shuffle_exact_permutation": True,
        "matched_shuffle_fixed_points": int(np.sum(donors == np.arange(len(donors)))),
        "matched_shuffle_same_scene": int(sum(eval_rows[i]["scene_id"] == eval_rows[j]["scene_id"] for i, j in enumerate(donors))),
        "composite_action_constructed_as": "encode_9d(T2 @ T1)",
        "additive_paths_computed_independently": True,
        "additive_max_abs_sequential_swapped_delta_float32": additive_fp32_delta,
        "additive_float32_roundoff_bound": additive_fp32_roundoff_bound,
        "additive_max_abs_sequential_swapped_delta_after_float16_storage": additive_delta,
        "no_true_B_injected_in_recursion": True,
    })
    atomic_json(done, {
        "status": "PASS", "cell": cell, "seed": seed,
        "checkpoint_count": 4, "evaluation_samples": len(eval_i),
        "evaluated_immediately_after_checkpoint_freeze": True,
    })


def run_cell(cell: str, device_name: str) -> None:
    protocol = load_protocol()
    receipt_path = RESULT / "inventory" / f"{cell}.json"
    if not receipt_path.exists() or json.loads(receipt_path.read_text()).get("status") != "PASS_MATERIALIZED_EXACT_V2":
        raise RuntimeError(f"{cell}: exact-v2 materialization did not pass")
    rows = read_jsonl(RESULT / "data/common_manifest.jsonl")
    geometry = dict(np.load(RESULT / "cache/geometry.npz", allow_pickle=True))
    if list(geometry["sample_id"].astype(str)) != [row["sample_id"] for row in rows]:
        raise RuntimeError("geometry order differs from frozen common manifest")
    tensors = load_file(str(RESULT / "cache" / f"{cell}.sft"), device="cpu")
    channels = int(tensors["context"].shape[-1])
    model_name, _ = cell.rsplit("_l", 1)
    registered_channels = int(protocol["models"][model_name]["channels"])
    if channels != registered_channels:
        raise RuntimeError(f"{cell}: observed channels={channels}, registered={registered_channels}")
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    status_path = RESULT / "cells" / cell / "EXECUTION_STATUS.json"
    atomic_json(status_path, {"status": "RUNNING", "cell": cell, "device": str(device)})
    try:
        for seed in protocol["training"]["seeds"]:
            run_seed(cell, int(seed), tensors, geometry, rows, protocol, device)
            atomic_json(status_path, {
                "status": "RUNNING", "cell": cell, "device": str(device),
                "completed_seeds": [value for value in protocol["training"]["seeds"] if (RESULT / "cells" / cell / "seeds" / f"seed{value}/COMPLETE.json").exists()],
            })
        atomic_json(status_path, {
            "status": "PASS", "cell": cell, "device": str(device),
            "completed_seeds": protocol["training"]["seeds"],
        })
    except Exception as error:
        atomic_json(status_path, {
            "status": "FAILED_ISOLATED_CELL", "cell": cell, "device": str(device),
            "error_type": type(error).__name__, "error": str(error),
        })
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_cell(args.cell, args.device)
