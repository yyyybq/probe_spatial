#!/usr/bin/env python3
"""Train the preregistered relational subset-v2 continuous-action probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

import run_independent_continuous_action as v1
from run_independent_p8_final_analysis import Store, read_jsonl, sha256, write_csv, write_json

MODELS = ("dino", "f3r", "vjepa2", "wan")
SEEDS = (41, 42, 43, 44, 45)
LEARNED_VARIANTS = (
    "full_relational",
    "full_concat",
    "current_only",
    "goal_only",
    "shuffled_goal_train",
)
EXPECTED_HASHES = v1.EXPECTED_HASHES
EXPECTED_BASE_REQUEST_SHA = v1.EXPECTED_REQUEST_SHA
EXPECTED_WAN_REQUEST_SHA = "fdd80a9b84f9050c2a4c10e27d94dece74f1f6fde24471231e243f0082713818"
EXPECTED_PROTOCOL_SHA = "005c9adfe2686b29deec8c21ef555a6c22108441fc3e33779e94cd3b583ea0d2"


class RelationalProbe(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.current_projection = torch.nn.Linear(width, 128, bias=False)
        self.goal_projection = torch.nn.Linear(width, 128, bias=False)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(128 * 4, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, 9),
        )

    def forward(self, current: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        current = self.current_projection(current)
        goal = self.goal_projection(goal)
        return self.net(torch.cat((current, goal, goal - current, current * goal), dim=1))


class ConcatProbe(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(width * 2, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, 9),
        )

    def forward(self, current: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((current, goal), dim=1))


def goal_distance(row: dict[str, Any]) -> float:
    return v1.motion_distance(np.asarray(row["anchor_pose"]), np.asarray(row["final_pose"]))


def action_magnitude(row: dict[str, Any]) -> float:
    _, translation, rotation = v1.action_components(np.asarray(row["primitive_actions_9d"][0]))
    return translation + v1.ROTATION_WEIGHT * rotation


def thresholds(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    return {
        "goal_distance": [float(value) for value in np.quantile([goal_distance(row) for row in rows], [1 / 3, 2 / 3])],
        "action_magnitude": [float(value) for value in np.quantile([action_magnitude(row) for row in rows], [1 / 3, 2 / 3])],
    }


def bin_index(value: float, cuts: list[float]) -> int:
    return int(value > cuts[0]) + int(value > cuts[1])


def matched_derangement(rows: list[dict[str, Any]], cuts: dict[str, list[float]]) -> list[int]:
    strata: dict[tuple[int, int], list[int]] = {}
    for index, row in enumerate(rows):
        key = (
            bin_index(goal_distance(row), cuts["goal_distance"]),
            bin_index(action_magnitude(row), cuts["action_magnitude"]),
        )
        strata.setdefault(key, []).append(index)
    mapping = [-1] * len(rows)
    unresolved: list[int] = []
    for members in strata.values():
        members.sort(key=lambda index: rows[index]["sample_id"])
        selected = None
        for offset in range(1, len(members)):
            candidate = [members[(position + offset) % len(members)] for position in range(len(members))]
            if all(
                rows[source]["scene_id"] != rows[target]["scene_id"]
                and rows[source]["final_raw_id"] != rows[target]["final_raw_id"]
                for source, target in zip(members, candidate)
            ):
                selected = candidate
                break
        if selected is None:
            unresolved.extend(members)
        else:
            for source, target in zip(members, selected):
                mapping[source] = target
    if unresolved:
        pool = sorted(range(len(rows)), key=lambda index: rows[index]["sample_id"])
        for source in unresolved:
            source_key = (
                bin_index(goal_distance(rows[source]), cuts["goal_distance"]),
                bin_index(action_magnitude(rows[source]), cuts["action_magnitude"]),
            )
            candidates = [
                target for target in pool
                if rows[source]["scene_id"] != rows[target]["scene_id"]
                and rows[source]["final_raw_id"] != rows[target]["final_raw_id"]
                and target not in mapping
            ]
            if not candidates:
                raise ValueError("could not construct different-scene matched goal intervention")
            def cost(target: int) -> tuple[float, str]:
                target_key = (
                    bin_index(goal_distance(rows[target]), cuts["goal_distance"]),
                    bin_index(action_magnitude(rows[target]), cuts["action_magnitude"]),
                )
                return (abs(source_key[0] - target_key[0]) + abs(source_key[1] - target_key[1]), rows[target]["sample_id"])
            mapping[source] = min(candidates, key=cost)
    if sorted(mapping) != list(range(len(rows))):
        raise ValueError("goal intervention is not a permutation")
    return mapping


def prepare_inputs(
    current: torch.Tensor,
    goal: torch.Tensor,
    rows: list[dict[str, Any]],
    current_norm: v1.Normalizer,
    goal_norm: v1.Normalizer,
    variant: str,
    cuts: dict[str, list[float]],
) -> tuple[torch.Tensor, torch.Tensor, list[int] | None]:
    current = current_norm.apply(current)
    goal = goal_norm.apply(goal)
    mapping = None
    if variant == "current_only":
        goal = torch.zeros_like(goal)
    elif variant == "goal_only":
        current = torch.zeros_like(current)
    elif variant == "shuffled_goal_train":
        mapping = matched_derangement(rows, cuts)
        goal = goal[torch.tensor(mapping, dtype=torch.long)]
    elif variant not in {"full_relational", "full_concat"}:
        raise ValueError(variant)
    return current, goal, mapping


def build_model(width: int, head: str) -> torch.nn.Module:
    if head == "relational":
        return RelationalProbe(width)
    if head == "concat":
        return ConcatProbe(width)
    raise ValueError(head)


def train_probe(
    train_inputs: tuple[torch.Tensor, torch.Tensor],
    train_action: torch.Tensor,
    dev_inputs: tuple[torch.Tensor, torch.Tensor],
    dev_action: torch.Tensor,
    dev_scenes: list[str],
    target_normalizer: v1.Normalizer,
    seed: int,
    device: str,
    head: str,
    epochs: int | None = None,
) -> tuple[torch.nn.Module, int, list[float]]:
    v1.seed_all(seed)
    model = build_model(train_inputs[0].shape[1], head).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(seed + 2700)
    train_target = target_normalizer.apply(train_action)
    dev_target = target_normalizer.apply(dev_action)
    best_loss, best_epoch, patience = math.inf, 1, 0
    curve: list[float] = []
    limit = epochs or 300
    for epoch in range(1, limit + 1):
        order = torch.randperm(len(train_action), generator=generator)
        model.train()
        for start in range(0, len(order), 64):
            index = order[start:start + 64]
            current = train_inputs[0][index].to(device)
            goal = train_inputs[1][index].to(device)
            target = train_target[index].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = v1.loss_per_sample(model(current, goal), target, target_normalizer).mean()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            dev_pred = model(dev_inputs[0].to(device), dev_inputs[1].to(device))
            per_sample = v1.loss_per_sample(dev_pred, dev_target.to(device), target_normalizer)
            dev_loss = v1.scene_macro(per_sample, dev_scenes)
        curve.append(dev_loss)
        if dev_loss < best_loss - 1e-8:
            best_loss, best_epoch, patience = dev_loss, epoch, 0
        else:
            patience += 1
        if epochs is None and patience >= 30:
            break
    return model.eval(), best_epoch, curve


def mapping_sha(rows: list[dict[str, Any]], mapping: list[int]) -> str:
    pairs = [(rows[index]["sample_id"], rows[target]["sample_id"]) for index, target in enumerate(mapping)]
    return hashlib.sha256(json.dumps(pairs, separators=(",", ":")).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--base-request", type=Path, required=True)
    parser.add_argument("--base-feature-root", type=Path, required=True)
    parser.add_argument("--wan-request", type=Path, required=True)
    parser.add_argument("--wan-feature-root", type=Path, required=True)
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

    if sha256(args.protocol) != EXPECTED_PROTOCOL_SHA:
        raise ValueError("protocol hash changed")
    if sha256(args.base_request) != EXPECTED_BASE_REQUEST_SHA:
        raise ValueError("base request hash changed")
    if sha256(args.wan_request) != EXPECTED_WAN_REQUEST_SHA:
        raise ValueError("Wan L29 request hash changed")
    for name, expected in EXPECTED_HASHES.items():
        if sha256(args.manifest_dir / name) != expected:
            raise ValueError(f"manifest hash changed: {name}")

    fit = read_jsonl(args.manifest_dir / "algebra_fit_scene_disjoint_v1.jsonl")
    dev = read_jsonl(args.manifest_dir / "algebra_dev_scene_disjoint_v1.jsonl")
    full = read_jsonl(args.source_manifest_dir / "algebra_p8_independent_v1_train.jsonl")
    val = read_jsonl(args.manifest_dir / "algebra_p8_independent_v1_val.jsonl")
    if (len(fit), len(dev), len(full), len(val)) != (803, 221, 1024, 256):
        raise ValueError("unexpected frozen split sizes")
    row_audit = v1.verify_rows(fit, dev, val)
    cuts = thresholds(fit)

    request = args.wan_request if args.model == "wan" else args.base_request
    feature_root = args.wan_feature_root if args.model == "wan" else args.base_feature_root
    store = Store(request, feature_root, args.model)
    fit_current, fit_goal, fit_action = v1.load_tensors(fit, store)
    dev_current, dev_goal, dev_action = v1.load_tensors(dev, store)
    full_current, full_goal, full_action = v1.load_tensors(full, store)
    val_current, val_goal, _ = v1.load_tensors(val, store)
    current_norm = v1.Normalizer(fit_current)
    goal_norm = v1.Normalizer(fit_goal)
    target_norm = v1.Normalizer(fit_action)
    dev_scenes = [row["scene_id"] for row in dev]

    results: list[dict[str, Any]] = []
    training: dict[str, Any] = {}
    checkpoints = work / "checkpoints"
    checkpoints.mkdir()
    intervention_mapping = matched_derangement(val, cuts)
    intervention_goal = goal_norm.apply(val_goal)[torch.tensor(intervention_mapping, dtype=torch.long)]
    intervention_sha = mapping_sha(val, intervention_mapping)

    for variant in LEARNED_VARIANTS:
        head = "concat" if variant == "full_concat" else "relational"
        fit_c, fit_g, _ = prepare_inputs(fit_current, fit_goal, fit, current_norm, goal_norm, variant, cuts)
        dev_c, dev_g, _ = prepare_inputs(dev_current, dev_goal, dev, current_norm, goal_norm, variant, cuts)
        full_c, full_g, _ = prepare_inputs(full_current, full_goal, full, current_norm, goal_norm, variant, cuts)
        val_c, val_g, _ = prepare_inputs(val_current, val_goal, val, current_norm, goal_norm, variant, cuts)
        for seed in SEEDS:
            _, selected_epoch, curve = train_probe(
                (fit_c, fit_g), fit_action, (dev_c, dev_g), dev_action, dev_scenes,
                target_norm, seed, args.device, head,
            )
            model, _, _ = train_probe(
                (full_c, full_g), full_action, (dev_c, dev_g), dev_action, dev_scenes,
                target_norm, seed, args.device, head, epochs=selected_epoch,
            )
            with torch.no_grad():
                prediction = target_norm.invert(model(val_c.to(args.device), val_g.to(args.device))).cpu().numpy()
            for row, pred in zip(val, prediction):
                results.append(v1.metric_row(row, args.model, variant, seed, pred))
            if variant == "full_relational":
                with torch.no_grad():
                    shuffled_prediction = target_norm.invert(
                        model(val_c.to(args.device), intervention_goal.to(args.device))
                    ).cpu().numpy()
                for row, pred in zip(val, shuffled_prediction):
                    results.append(v1.metric_row(row, args.model, "full_relational_inference_shuffled_goal", seed, pred))

            checkpoint = checkpoints / f"{variant}_seed_{seed}.pt"
            torch.save({
                "model": args.model,
                "variant": variant,
                "head": head,
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
                "head": head,
                "selected_epoch": selected_epoch,
                "dev_loss_curve": curve,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "checkpoint_sha256": sha256(checkpoint),
            }

    mean_action = full_action.mean(0).numpy()
    for row in val:
        results.append(v1.metric_row(row, args.model, "mean_motion_prior", "NA", mean_action))
        results.append(v1.metric_row(row, args.model, "oracle_a1", "NA", np.asarray(row["primitive_actions_9d"][0])))

    write_csv(work / "continuous_action_per_sample.csv", results)
    integrity = {
        "status": "COMPLETE_VALID",
        "model": args.model,
        "protocol_sha256": sha256(args.protocol),
        "request_sha256": sha256(request),
        "base_request_sha256": sha256(args.base_request),
        "wan_request_sha256": sha256(args.wan_request),
        "manifest_sha256": {name: sha256(args.manifest_dir / name) for name in EXPECTED_HASHES},
        "row_audit": row_audit,
        "feature_width": int(fit_current.shape[1]),
        "feature_stage": {"dino": "L0-project-final", "f3r": "L24", "vjepa2": "L23", "wan": "L29/t749"}[args.model],
        "input_contract": ["current_p8_feature", "goal_h4_isolated_feature"],
        "forbidden_model_inputs": [
            "GT_pose", "goal_pose", "action_label", "future_action_sequence",
            "raw_frame_index", "scene_ID", "trajectory_direction",
        ],
        "learned_variants": list(LEARNED_VARIANTS),
        "inference_intervention": {
            "variant": "full_relational_inference_shuffled_goal",
            "different_scene": True,
            "difficulty_matching": ["fit-derived goal-distance tertile", "fit-derived first-action-magnitude tertile"],
            "mapping_sha256": intervention_sha,
        },
        "difficulty_thresholds_fit_only": cuts,
        "seeds": list(SEEDS),
        "validation_records": len(val),
        "per_sample_rows": len(results),
        "expected_per_sample_rows": 8192,
        "training": training,
        "code_sha256": sha256(Path(__file__)),
    }
    if len(results) != 8192:
        raise ValueError(f"unexpected per-sample row count: {len(results)}")
    write_json(work / "training_and_integrity.json", integrity)
    os.rename(work, output_target)
    print(json.dumps({"status": "COMPLETE_VALID", "model": args.model, "rows": len(results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
