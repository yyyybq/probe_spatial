#!/usr/bin/env python3
"""Run the frozen final Actionability, clean dynamics and algebra evaluation.

All trainable modules start from the requested seed in this process. The only
backbone inputs are exact pooled vectors verified against feature-request v2.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
from safetensors.torch import load_file

IDENTITY9 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
SEEDS = (41, 42, 43)
TRANSFORMS = ("raw", "centered", "standardized")
SUPERVISED_HORIZONS = (1, 2, 4)
SUPERVISED_STEPS = (0, 1, 3)


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    fields = sorted({key for row in rows for key in row})
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o644)
        os.link(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def write_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o644)
        os.link(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


class Store:
    def __init__(self, request: Path, root: Path, model: str) -> None:
        self.root, self.model = root, model
        self.isolated: dict[str, dict[str, Any]] = {}
        self.contexts: dict[tuple[str, ...], dict[str, Any]] = {}
        for unit in read_jsonl(request):
            if unit["model"] != model:
                continue
            if unit["feature_role"] == "isolated_frame":
                if unit["raw_frame_id"] in self.isolated:
                    raise ValueError(f"ambiguous isolated raw-ID lookup: {unit['raw_frame_id']}")
                self.isolated[unit["raw_frame_id"]] = unit
            else:
                key = tuple(unit["context_raw_ids"])
                if key in self.contexts:
                    raise ValueError(f"ambiguous ordered-context lookup: {key}")
                self.contexts[key] = unit
        self.cache: dict[str, torch.Tensor] = {}

    def _get(self, unit: dict[str, Any]) -> torch.Tensor:
        if unit["unit_id"] not in self.cache:
            path = self.root / self.model / "scannetpp" / unit["scene_id"] / f"{unit['unit_id']}.safetensors"
            sidecar = Path(f"{path}.manifest.json")
            meta = json.loads(sidecar.read_text())
            logical = unit["context_raw_ids"] if unit["feature_role"] == "streaming_context_p8" else [unit["raw_frame_id"]]
            expected = {
                "unit_id": unit["unit_id"], "model": self.model,
                "feature_role": unit["feature_role"], "raw_frame_id": unit["raw_frame_id"],
                "context_raw_ids": unit["context_raw_ids"], "logical_input_raw_ids": logical,
                "fallback": False,
            }
            differing = [key for key, value in expected.items() if meta.get(key) != value]
            if differing:
                raise ValueError(f"exact feature identity mismatch {path}: {differing}")
            value = load_file(str(path))["feat"].float()
            if value.ndim != 1 or not torch.isfinite(value).all():
                raise ValueError(f"invalid exact feature: {path}")
            self.cache[unit["unit_id"]] = value
        return self.cache[unit["unit_id"]]

    def vector(self, raw_id: str) -> torch.Tensor:
        return self._get(self.isolated[raw_id])

    def context(self, raw_ids: list[str]) -> torch.Tensor:
        return self._get(self.contexts[tuple(raw_ids)])


class FeatureTransform:
    def __init__(self, values: torch.Tensor) -> None:
        self.mean = values.mean(0)
        self.std = values.std(0).clamp_min(1e-5)

    def apply(self, value: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "raw":
            return value
        if mode == "centered":
            return value - self.mean.to(value.device)
        if mode == "standardized":
            return (value - self.mean.to(value.device)) / self.std.to(value.device)
        raise ValueError(mode)

    def standardize(self, value: torch.Tensor) -> torch.Tensor:
        return self.apply(value, "standardized")

    def unstandardize(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.std.to(value.device) + self.mean.to(value.device)


class SharedScorer(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(width, 256), torch.nn.GELU(), torch.nn.Linear(256, 1)
        )

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        # pair: B,2,D; the same network is used for both candidates.
        return self.net(pair).squeeze(-1)


class RecurrentDynamics(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.initial = torch.nn.Linear(width, 256)
        self.cell = torch.nn.GRUCell(9, 256)
        self.decode = torch.nn.Sequential(torch.nn.LayerNorm(256), torch.nn.Linear(256, width))

    def rollout(self, current: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.initial(current))
        outputs = []
        for step in range(actions.shape[1]):
            hidden = self.cell(actions[:, step], hidden)
            outputs.append(current + self.decode(hidden))
        return torch.stack(outputs, dim=1)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1.0 - torch.nn.functional.cosine_similarity(a, b, dim=-1)


def scene_macro(values: torch.Tensor, scene_ids: list[str]) -> float:
    """Mean within scene, then an unweighted mean over scenes."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, scene_id in zip(values.detach().cpu().tolist(), scene_ids):
        grouped[scene_id].append(float(value))
    return float(np.mean([np.mean(items) for items in grouped.values()]))


def standardized_mse(a: torch.Tensor, b: torch.Tensor, transform: FeatureTransform) -> torch.Tensor:
    return torch.nn.functional.mse_loss(transform.standardize(a), transform.standardize(b))


def scorer_features(
    rows: list[dict[str, Any]], store: Store, transform: FeatureTransform, mode: str, variant: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    pairs, labels = [], []
    for row in rows:
        context = transform.apply(store.context(row["context_raw_ids"]), mode)
        anchor = transform.apply(store.vector(row["anchor_raw_id"]), mode)
        goal = transform.apply(store.vector(row["goal_raw_id"]), mode)
        actions = [torch.tensor(row["action_1_9d"]), torch.tensor(row["action_2_9d"])]
        candidate = []
        for action in actions:
            if variant == "direct":
                value = torch.cat((context, anchor, goal, action))
            elif variant == "action_only":
                value = action
            elif variant == "current_action_no_goal":
                value = torch.cat((context, anchor, action))
            elif variant == "goal_action_no_current":
                value = torch.cat((goal, action))
            else:
                raise ValueError(variant)
            candidate.append(value)
        pairs.append(torch.stack(candidate))
        labels.append(int(row["best_candidate"]) - 1)
    return torch.stack(pairs), torch.tensor(labels, dtype=torch.long)


def train_scorer(
    fit_x: torch.Tensor, fit_y: torch.Tensor, dev_x: torch.Tensor, dev_y: torch.Tensor,
    dev_scene_ids: list[str], seed: int, device: str, epochs: int | None = None,
) -> tuple[SharedScorer, int, list[float]]:
    seed_all(seed)
    model = SharedScorer(fit_x.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(seed + 1000)
    best_loss, best_epoch, patience = math.inf, 1, 0
    curve = []
    limit = epochs or 200
    for epoch in range(1, limit + 1):
        order = torch.randperm(len(fit_x), generator=generator)
        model.train()
        for start in range(0, len(order), 64):
            idx = order[start:start + 64]
            swap = torch.rand(len(idx), generator=generator) < 0.5
            x, y = fit_x[idx].clone(), fit_y[idx].clone()
            x[swap] = x[swap].flip(1)
            y[swap] = 1 - y[swap]
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            dev_losses = torch.nn.functional.cross_entropy(
                model(dev_x.to(device)), dev_y.to(device), reduction="none"
            )
            loss = scene_macro(dev_losses, dev_scene_ids)
        curve.append(loss)
        if loss < best_loss - 1e-8:
            best_loss, best_epoch, patience = loss, epoch, 0
        else:
            patience += 1
        if epochs is None and patience >= 20:
            break
    return model.eval(), best_epoch, curve


def action_rows_from_selection(
    source: list[dict[str, Any]], model_name: str, variant: str, transform: str,
    seed: int | None, selected: Iterable[int], shuffled_goal: bool = False,
) -> list[dict[str, Any]]:
    result = []
    for row, choice in zip(source, selected):
        choice = int(choice)
        best = int(row["best_candidate"]) - 1
        utilities = [float(row["utility_1"]), float(row["utility_2"])]
        result.append({
            "model": model_name, "variant": variant, "transform": transform,
            "seed": "NA" if seed is None else seed, "sample_id": row["sample_id"], "scene_id": row["scene_id"],
            "selected_candidate": choice + 1, "best_candidate": best + 1,
            "correct": int(choice == best), "regret": utilities[best] - utilities[choice],
            "selected_utility": utilities[choice], "positive_progress_selected": int(utilities[choice] > 0),
            "goal_distance": row["goal_pose_distance"], "utility_margin": row["utility_margin"],
            "goal_temporal_direction": row["goal_temporal_direction"],
            "motion_type": "rotation" if row["rotation_dominant"] else "translation",
            "endpoint_seen_in_context": int(
                row["candidate_endpoint_raw_id_1"] in row["context_raw_ids"]
                or row["candidate_endpoint_raw_id_2"] in row["context_raw_ids"]
            ),
            "shuffled_goal_intervention": int(shuffled_goal),
        })
    return result


def dynamics_tensors(rows: list[dict[str, Any]], store: Store) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    current, actions, targets = [], [], []
    for row in rows:
        current.append(store.vector(row["anchor_raw_id"]))
        actions.append(torch.tensor(row["primitive_actions_9d"], dtype=torch.float32))
        targets.append(torch.stack([
            store.vector(row["horizon_target_raw_ids"][f"h{horizon}"])
            for horizon in SUPERVISED_HORIZONS
        ]))
    return torch.stack(current), torch.stack(actions), torch.stack(targets)


def train_dynamics(
    fit: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    dev: tuple[torch.Tensor, torch.Tensor, torch.Tensor], transform: FeatureTransform,
    dev_scene_ids: list[str], variant: str, seed: int, device: str, epochs: int | None = None,
) -> tuple[RecurrentDynamics, int, list[float]]:
    seed_all(seed)
    fit_current, fit_action, fit_target = fit
    dev_current, dev_action, dev_target = dev
    fit_current = transform.standardize(fit_current)
    fit_target = transform.standardize(fit_target)
    dev_current = transform.standardize(dev_current)
    dev_target = transform.standardize(dev_target)
    identity_fit = torch.tensor(IDENTITY9).reshape(1, 1, 9).repeat(len(fit_action), 4, 1)
    identity_dev = torch.tensor(IDENTITY9).reshape(1, 1, 9).repeat(len(dev_action), 4, 1)
    if variant == "visual_only":
        fit_action, dev_action = identity_fit, identity_dev
    elif variant == "action_only":
        fit_current, dev_current = torch.zeros_like(fit_current), torch.zeros_like(dev_current)
    model = RecurrentDynamics(fit_current.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(seed + 2000)
    best_loss, best_epoch, patience = math.inf, 1, 0
    curve = []
    limit = epochs or 200
    for epoch in range(1, limit + 1):
        order = torch.randperm(len(fit_current), generator=generator)
        model.train()
        for start in range(0, len(order), 64):
            idx = order[start:start + 64]
            current = fit_current[idx].to(device)
            action = fit_action[idx].to(device)
            target = fit_target[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model.rollout(current, action)[:, SUPERVISED_STEPS]
            loss = torch.nn.functional.mse_loss(pred, target) + 0.1 * cosine_distance(pred, target).mean()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            pred = model.rollout(dev_current.to(device), dev_action.to(device))[:, SUPERVISED_STEPS]
            target = dev_target.to(device)
            per_sample = (pred - target).square().mean(dim=(1, 2)) + 0.1 * cosine_distance(pred, target).mean(dim=1)
            loss = scene_macro(per_sample, dev_scene_ids)
        curve.append(loss)
        if loss < best_loss - 1e-8:
            best_loss, best_epoch, patience = loss, epoch, 0
        else:
            patience += 1
        if epochs is None and patience >= 20:
            break
    return model.eval(), best_epoch, curve


def action9_to_matrix(action: np.ndarray) -> np.ndarray:
    first = action[:3] / max(np.linalg.norm(action[:3]), 1e-12)
    second = action[3:6] - first * np.dot(first, action[3:6])
    second = second / max(np.linalg.norm(second), 1e-12)
    third = np.cross(first, second)
    value = np.eye(4)
    value[:3, :3] = np.stack((first, second, third))
    value[:3, 3] = action[6:9]
    return value


def matrix_to_action9(value: np.ndarray) -> np.ndarray:
    return np.concatenate((value[:2, :3].reshape(-1), value[:3, 3])).astype(np.float32)


def compose9(actions: np.ndarray) -> np.ndarray:
    total = np.eye(4)
    for action in actions:
        total = action9_to_matrix(action) @ total
    return matrix_to_action9(total)


def inverse9(action: np.ndarray) -> np.ndarray:
    return matrix_to_action9(np.linalg.inv(action9_to_matrix(action)))


def pose_action(anchor_c2w: list[list[float]], endpoint_c2w: list[list[float]]) -> np.ndarray:
    a, b = np.asarray(anchor_c2w), np.asarray(endpoint_c2w)
    aw, bw = np.linalg.inv(a), np.linalg.inv(b)
    value = np.eye(4)
    value[:3, :3] = bw[:3, :3] @ aw[:3, :3].T
    value[:3, 3] = bw[:3, 3] - value[:3, :3] @ aw[:3, 3]
    return matrix_to_action9(value)


def fixed_shuffle(rows: list[dict[str, Any]]) -> tuple[list[int | None], int]:
    motions = np.asarray([row["cumulative_motion"] for row in rows])
    edges = np.quantile(motions, np.linspace(0, 1, 11))
    bins = np.clip(np.searchsorted(edges[1:-1], motions, side="right"), 0, 9)
    mapping: list[int | None] = [None] * len(rows)
    unmatched = 0
    for value in range(10):
        indices = sorted(np.flatnonzero(bins == value).tolist(), key=lambda idx: rows[idx]["sample_id"])
        if len(indices) < 2:
            unmatched += len(indices)
            continue
        for source, target in zip(indices, indices[1:] + indices[:1]):
            mapping[source] = target
    return mapping, unmatched


def goal_derangement(rows: list[dict[str, Any]]) -> list[int]:
    """Deterministic sample-ID-ordered cyclic derangement with unequal raw goals."""
    order = sorted(range(len(rows)), key=lambda index: rows[index]["sample_id"])
    for offset in range(1, len(order)):
        mapping = [0] * len(order)
        for position, source in enumerate(order):
            mapping[source] = order[(position + offset) % len(order)]
        if all(rows[source]["goal_raw_id"] != rows[target]["goal_raw_id"] for source, target in enumerate(mapping)):
            return mapping
    raise ValueError("could not construct a raw-goal derangement")


def unstandard_rollout(model: RecurrentDynamics, transform: FeatureTransform, current: torch.Tensor, actions: torch.Tensor, device: str) -> torch.Tensor:
    with torch.no_grad():
        pred = model.rollout(transform.standardize(current).to(device), actions.to(device)).cpu()
    return transform.unstandardize(pred)


def retrieval_ranks(pred: torch.Tensor, rows: list[dict[str, Any]], targets: torch.Tensor, target_index: int, horizon: int) -> list[int]:
    raw_ids = [row["horizon_target_raw_ids"][f"h{horizon}"] for row in rows]
    keys, vectors = [], []
    seen = set()
    for row, raw, vector in zip(rows, raw_ids, targets[:, target_index]):
        key = (row["scene_id"], raw)
        if key not in seen:
            seen.add(key); keys.append(key); vectors.append(vector)
    bank = torch.stack(vectors)
    bank = torch.nn.functional.normalize(bank, dim=1)
    scores = torch.nn.functional.normalize(pred, dim=1) @ bank.T
    result = []
    for index, row in enumerate(rows):
        target_index = keys.index((row["scene_id"], raw_ids[index]))
        order = torch.argsort(scores[index], descending=True)
        result.append(int(torch.nonzero(order == target_index, as_tuple=False)[0, 0].item()) + 1)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("dino", "f3r", "vjepa2", "wan"), required=True)
    parser.add_argument("--request-manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output_target = args.output_dir.resolve()
    if output_target.exists():
        raise SystemExit(f"refusing to overwrite {output_target}")
    output_target.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir = Path(tempfile.mkdtemp(prefix=f".{output_target.name}.incomplete-", dir=output_target.parent))

    store = Store(args.request_manifest.resolve(), args.feature_root.resolve(), args.model)
    source_dir = args.request_manifest.resolve().parent
    final_dir = args.manifest_dir.resolve()
    action_fit = read_jsonl(final_dir / "actionability_fit_scene_disjoint_v1.jsonl")
    action_dev = read_jsonl(final_dir / "actionability_dev_scene_disjoint_v1.jsonl")
    action_train = read_jsonl(source_dir / "actionability_p8_independent_v1_train.jsonl")
    action_val = read_jsonl(final_dir / "actionability_p8_independent_v1_val.jsonl")
    algebra_fit = read_jsonl(final_dir / "algebra_fit_scene_disjoint_v1.jsonl")
    algebra_dev = read_jsonl(final_dir / "algebra_dev_scene_disjoint_v1.jsonl")
    algebra_train = read_jsonl(source_dir / "algebra_p8_independent_v1_train.jsonl")
    algebra_val = read_jsonl(final_dir / "algebra_p8_independent_v1_val.jsonl")
    loops = read_jsonl(final_dir / "algebra_p8_independent_v1_real_loops.jsonl")

    referenced = []
    for row in action_train:
        referenced.extend([store.context(row["context_raw_ids"]), store.vector(row["anchor_raw_id"]), store.vector(row["goal_raw_id"]),
                           store.vector(row["candidate_endpoint_raw_id_1"]), store.vector(row["candidate_endpoint_raw_id_2"])])
    for row in algebra_train:
        referenced.append(store.vector(row["anchor_raw_id"]))
        referenced.extend(
            store.vector(row["horizon_target_raw_ids"][f"h{horizon}"])
            for horizon in SUPERVISED_HORIZONS
        )
    transform = FeatureTransform(torch.stack(referenced))

    action_results: list[dict[str, Any]] = []
    training_log: dict[str, Any] = {"scorers": {}, "dynamics": {}}
    scorer_variants = ("direct", "action_only", "current_action_no_goal", "goal_action_no_current")
    for mode in TRANSFORMS:
        for variant in scorer_variants:
            fit_x, fit_y = scorer_features(action_fit, store, transform, mode, variant)
            dev_x, dev_y = scorer_features(action_dev, store, transform, mode, variant)
            full_x, full_y = scorer_features(action_train, store, transform, mode, variant)
            val_x, _ = scorer_features(action_val, store, transform, mode, variant)
            for seed in SEEDS:
                dev_scenes = [row["scene_id"] for row in action_dev]
                _, selected_epoch, curve = train_scorer(fit_x, fit_y, dev_x, dev_y, dev_scenes, seed, args.device)
                final_model, _, _ = train_scorer(full_x, full_y, dev_x, dev_y, dev_scenes, seed, args.device, epochs=selected_epoch)
                with torch.no_grad():
                    pred = final_model(val_x.to(args.device)).argmax(1).cpu().tolist()
                    swapped = final_model(val_x.flip(1).to(args.device)).argmax(1).cpu()
                equivariance_failures = int(sum(int((1 - p) != int(s)) for p, s in zip(pred, swapped)))
                action_results.extend(action_rows_from_selection(action_val, args.model, variant, mode, seed, pred))
                training_log["scorers"][f"{variant}|{mode}|{seed}"] = {
                    "selected_epoch": selected_epoch, "dev_loss_curve": curve,
                    "parameters": sum(p.numel() for p in final_model.parameters()),
                    "candidate_swap_equivariance_failures": equivariance_failures,
                }
                if variant == "direct":
                    shuffled_goals = goal_derangement(action_val)
                    modified = [dict(row, goal_raw_id=action_val[shuffled_goals[i]]["goal_raw_id"]) for i, row in enumerate(action_val)]
                    shuffled_x, _ = scorer_features(modified, store, transform, mode, variant)
                    with torch.no_grad():
                        shuffled_pred = final_model(shuffled_x.to(args.device)).argmax(1).cpu().tolist()
                    action_results.extend(action_rows_from_selection(action_val, args.model, "direct_shuffled_goal", mode, seed, shuffled_pred, True))

    # Fixed non-learned references.
    temporal_majority = {
        direction: Counter(int(row["best_candidate"]) - 1 for row in action_train if row["goal_temporal_direction"] == direction).most_common(1)[0][0]
        for direction in ("past", "future")
    }
    candidate_majority = Counter(int(row["best_candidate"]) - 1 for row in action_train).most_common(1)[0][0]
    action_results.extend(action_rows_from_selection(action_val, args.model, "random_expected", "NA", None, [0] * len(action_val)))
    # Replace deterministic placeholder regret/correct with analytic expectations.
    for item, row in zip(action_results[-len(action_val):], action_val):
        item["correct"] = 0.5
        item["regret"] = 0.5 * abs(float(row["utility_1"]) - float(row["utility_2"]))
        item["positive_progress_selected"] = 0.5 * (int(row["utility_1"] > 0) + int(row["utility_2"] > 0))
        item["selected_candidate"] = "expected"
        item["selected_utility"] = 0.5 * (float(row["utility_1"]) + float(row["utility_2"]))
    action_results.extend(action_rows_from_selection(action_val, args.model, "candidate_prior", "NA", None, [candidate_majority] * len(action_val)))
    action_results.extend(action_rows_from_selection(action_val, args.model, "temporal_direction_prior", "NA", None,
                                                     [temporal_majority[row["goal_temporal_direction"]] for row in action_val]))
    for mode in TRANSFORMS:
        choices = []
        for row in action_val:
            goal = transform.apply(store.vector(row["goal_raw_id"]), mode)
            endpoints = torch.stack([transform.apply(store.vector(row["candidate_endpoint_raw_id_1"]), mode),
                                     transform.apply(store.vector(row["candidate_endpoint_raw_id_2"]), mode)])
            choices.append(int(torch.nn.functional.cosine_similarity(endpoints, goal.unsqueeze(0), dim=1).argmax().item()))
        action_results.extend(action_rows_from_selection(action_val, args.model, "observed_endpoint_feature_reference", mode, None, choices))

    fit_dyn = dynamics_tensors(algebra_fit, store)
    dev_dyn = dynamics_tensors(algebra_dev, store)
    full_dyn = dynamics_tensors(algebra_train, store)
    val_current, val_actions, val_targets = dynamics_tensors(algebra_val, store)
    shuffle_map, unmatched = fixed_shuffle(algebra_val)
    shuffled_actions = val_actions.clone()
    for index, target_index in enumerate(shuffle_map):
        if target_index is not None:
            shuffled_actions[index] = val_actions[target_index]
    dynamics_results: list[dict[str, Any]] = []
    algebra_results: list[dict[str, Any]] = []
    full_models: dict[int, RecurrentDynamics] = {}
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir()
    for variant in ("full", "visual_only", "action_only"):
        for seed in SEEDS:
            dev_scenes = [row["scene_id"] for row in algebra_dev]
            _, selected_epoch, curve = train_dynamics(fit_dyn, dev_dyn, transform, dev_scenes, variant, seed, args.device)
            model, _, _ = train_dynamics(full_dyn, dev_dyn, transform, dev_scenes, variant, seed, args.device, epochs=selected_epoch)
            if variant == "full":
                full_models[seed] = model
                checkpoint = checkpoint_dir / f"clean_dynamics_seed_{seed}.pt"
                torch.save({
                    "model": args.model,
                    "seed": seed,
                    "variant": variant,
                    "selected_epoch": selected_epoch,
                    "width": val_current.shape[-1],
                    "request_manifest_sha256": sha256(args.request_manifest),
                    "algebra_train_manifest_sha256": sha256(source_dir / "algebra_p8_independent_v1_train.jsonl"),
                    "feature_transform_mean": transform.mean.cpu(),
                    "feature_transform_std": transform.std.cpu(),
                    "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                }, checkpoint)
            # train_dynamics zeroes the standardized current for action-only.
            # A raw feature equal to the train mean maps to that exact zero.
            current = val_current if variant != "action_only" else transform.mean.unsqueeze(0).repeat(len(val_current), 1)
            actions = val_actions
            if variant == "visual_only":
                actions = torch.tensor(IDENTITY9).reshape(1, 1, 9).repeat(len(val_actions), 4, 1)
            pred = unstandard_rollout(model, transform, current, actions, args.device)[:, SUPERVISED_STEPS]
            training_log["dynamics"][f"{variant}|{seed}"] = {
                "selected_epoch": selected_epoch, "dev_loss_curve": curve,
                "parameters": sum(p.numel() for p in model.parameters()),
            }
            for target_index, horizon in enumerate(SUPERVISED_HORIZONS):
                ranks = retrieval_ranks(pred[:, target_index], algebra_val, val_targets, target_index, horizon)
                for row, prediction, target, rank in zip(algebra_val, pred[:, target_index], val_targets[:, target_index], ranks):
                    dynamics_results.append({
                        "model": args.model, "variant": variant, "seed": seed, "sample_id": row["sample_id"],
                        "scene_id": row["scene_id"], "horizon": horizon,
                        "cosine_distance": float(cosine_distance(prediction, target).item()),
                        "mse": float(standardized_mse(prediction, target, transform).item()),
                        "raw_feature_mse": float(torch.nn.functional.mse_loss(prediction, target).item()),
                        "retrieval_rank": rank, "retrieval_r1": int(rank <= 1), "retrieval_r5": int(rank <= 5),
                    })

    identity = torch.tensor(IDENTITY9).reshape(1, 1, 9)
    # Exact anchor-copy prediction baseline, reported at every horizon. It is
    # non-trainable and intentionally has no retrieval score.
    for target_index, horizon in enumerate(SUPERVISED_HORIZONS):
        for row, current, target in zip(algebra_val, val_current, val_targets[:, target_index]):
            dynamics_results.append({
                "model": args.model, "variant": "copy_current", "seed": "NA",
                "sample_id": row["sample_id"], "scene_id": row["scene_id"], "horizon": horizon,
                "cosine_distance": float(cosine_distance(current, target).item()),
                "mse": float(standardized_mse(current, target, transform).item()),
                "raw_feature_mse": float(torch.nn.functional.mse_loss(current, target).item()),
                "retrieval_rank": "NA", "retrieval_r1": "NA", "retrieval_r5": "NA",
            })
    for seed, model in full_models.items():
        correct = unstandard_rollout(model, transform, val_current, val_actions, args.device)
        shuffled = unstandard_rollout(model, transform, val_current, shuffled_actions, args.device)
        for target_index, horizon in enumerate(SUPERVISED_HORIZONS):
            for index, row in enumerate(algebra_val):
                dynamics_results.append({
                    "model": args.model, "variant": "full_matched_action_shuffle", "seed": seed,
                    "sample_id": row["sample_id"], "scene_id": row["scene_id"], "horizon": horizon,
                    "cosine_distance": float(cosine_distance(shuffled[index, SUPERVISED_STEPS[target_index]], val_targets[index, target_index]).item()),
                    "mse": float(standardized_mse(shuffled[index, SUPERVISED_STEPS[target_index]], val_targets[index, target_index], transform).item()),
                    "raw_feature_mse": float(torch.nn.functional.mse_loss(shuffled[index, SUPERVISED_STEPS[target_index]], val_targets[index, target_index]).item()),
                    "retrieval_rank": "NA", "retrieval_r1": "NA", "retrieval_r5": "NA",
                    "matched_shuffle_unavailable": int(shuffle_map[index] is None),
                    "action_sensitivity_cosine": float(cosine_distance(correct[index, SUPERVISED_STEPS[target_index]], shuffled[index, SUPERVISED_STEPS[target_index]]).item()),
                })
        # Dynamics-based Actionability with raw/centered/standardized cosine and shuffled goals.
        action_current = torch.stack([store.vector(row["anchor_raw_id"]) for row in action_val])
        candidate_actions = torch.tensor([[row["action_1_9d"], row["action_2_9d"]] for row in action_val])
        flat_current = action_current[:, None, :].repeat(1, 2, 1).reshape(-1, action_current.shape[-1])
        flat_action = candidate_actions.reshape(-1, 1, 9)
        candidate_pred = unstandard_rollout(model, transform, flat_current, flat_action, args.device)[:, 0].reshape(len(action_val), 2, -1)
        goals = torch.stack([store.vector(row["goal_raw_id"]) for row in action_val])
        shuffled_idx = goal_derangement(action_val)
        for mode in TRANSFORMS:
            p = transform.apply(candidate_pred, mode)
            g = transform.apply(goals, mode)
            choices = torch.nn.functional.cosine_similarity(p, g[:, None, :], dim=2).argmax(1).tolist()
            action_results.extend(action_rows_from_selection(action_val, args.model, "clean_dynamics_selection", mode, seed, choices))
            sg = transform.apply(goals[shuffled_idx], mode)
            shuffled_choice = torch.nn.functional.cosine_similarity(p, sg[:, None, :], dim=2).argmax(1).tolist()
            action_results.extend(action_rows_from_selection(action_val, args.model, "clean_dynamics_selection_shuffled_goal", mode, seed, shuffled_choice, True))

        # Algebra consistency and correctness controls.
        for index, row in enumerate(algebra_val):
            primitive = val_actions[index].numpy()
            composed = compose9(primitive)
            two = np.stack((compose9(primitive[:2]), compose9(primitive[2:])))
            inputs = {
                "four_primitives": primitive,
                "two_composed": two,
                "one_composed": np.stack((composed,)),
                "composed_then_identity": np.stack((composed, *([np.asarray(IDENTITY9)] * 3))),
                "identity_then_composed": np.stack((*([np.asarray(IDENTITY9)] * 3), composed)),
                "identity_middle_composed": np.stack((np.asarray(IDENTITY9), composed, np.asarray(IDENTITY9), np.asarray(IDENTITY9))),
            }
            predictions = {}
            shuffled_primitive = shuffled_actions[index].numpy()
            shuffled_composed = compose9(shuffled_primitive)
            shuffled_inputs = {
                "four_primitives": shuffled_primitive,
                "two_composed": np.stack((compose9(shuffled_primitive[:2]), compose9(shuffled_primitive[2:]))),
                "one_composed": np.stack((shuffled_composed,)),
                "composed_then_identity": np.stack((shuffled_composed, *([np.asarray(IDENTITY9)] * 3))),
                "identity_then_composed": np.stack((*([np.asarray(IDENTITY9)] * 3), shuffled_composed)),
                "identity_middle_composed": np.stack((np.asarray(IDENTITY9), shuffled_composed, np.asarray(IDENTITY9), np.asarray(IDENTITY9))),
            }
            wrong_predictions = {}
            base = val_current[index:index + 1]
            for name, action_array in inputs.items():
                action_tensor = torch.tensor(action_array, dtype=torch.float32).unsqueeze(0)
                predictions[name] = unstandard_rollout(model, transform, base, action_tensor, args.device)[0, -1]
                wrong_tensor = torch.tensor(shuffled_inputs[name], dtype=torch.float32).unsqueeze(0)
                wrong_predictions[name] = unstandard_rollout(model, transform, base, wrong_tensor, args.device)[0, -1]
            target = val_targets[index, 2]
            for name, prediction in predictions.items():
                algebra_results.append({
                    "model": args.model, "seed": seed, "family": "composition", "variant": name,
                    "sample_id": row["sample_id"], "scene_id": row["scene_id"],
                    "prediction_to_endpoint_cosine": float(cosine_distance(prediction, target).item()),
                    "prediction_to_endpoint_mse": float(standardized_mse(prediction, target, transform).item()),
                    "prediction_to_endpoint_raw_feature_mse": float(torch.nn.functional.mse_loss(prediction, target).item()),
                    "consistency_to_four_primitive_cosine": float(cosine_distance(prediction, predictions["four_primitives"]).item()),
                    "action_dependence_wrong_action_separation": float(cosine_distance(prediction, wrong_predictions[name]).item()),
                })
            identity_pred = unstandard_rollout(model, transform, base, identity, args.device)[0, 0]
            forward_then_inverse = torch.tensor(np.stack((composed, inverse9(composed))), dtype=torch.float32).unsqueeze(0)
            inverse_pred = unstandard_rollout(model, transform, base, forward_then_inverse, args.device)[0, -1]
            wrong = shuffled_composed
            wrong_cycle = torch.tensor(np.stack((composed, inverse9(wrong))), dtype=torch.float32).unsqueeze(0)
            wrong_pred = unstandard_rollout(model, transform, base, wrong_cycle, args.device)[0, -1]
            repeat = torch.tensor(np.stack((composed, composed)), dtype=torch.float32).unsqueeze(0)
            repeat_pred = unstandard_rollout(model, transform, base, repeat, args.device)[0, -1]
            for name, prediction in (("identity", identity_pred), ("inverse_cycle", inverse_pred),
                                     ("wrong_inverse_cycle", wrong_pred), ("repeat_composed", repeat_pred),
                                     ("copy_constant", base[0])):
                algebra_results.append({
                    "model": args.model, "seed": seed, "family": "identity_inverse", "variant": name,
                    "sample_id": row["sample_id"], "scene_id": row["scene_id"],
                    "decoded_to_anchor_cosine": float(cosine_distance(prediction, base[0]).item()),
                    "decoded_to_anchor_mse": float(standardized_mse(prediction, base[0], transform).item()),
                    "decoded_to_anchor_raw_feature_mse": float(torch.nn.functional.mse_loss(prediction, base[0]).item()),
                    "forward_prediction_to_endpoint_cosine": float(cosine_distance(correct[index, -1], target).item()),
                    "wrong_action_separation": float(cosine_distance(inverse_pred, wrong_pred).item()),
                })

        # Real observed loops and their matched non-loop controls use the recorded net action.
        for row in loops:
            anchor = store.vector(row["anchor_raw_id"]).unsqueeze(0)
            endpoint = store.vector(row["final_raw_id"])
            net = torch.tensor(pose_action(row["anchor_pose"], row["final_pose"])).reshape(1, 1, 9)
            pred = unstandard_rollout(model, transform, anchor, net, args.device)[0, 0]
            algebra_results.append({
                "model": args.model, "seed": seed, "family": "observed_loop_net_action",
                "variant": "real_loop" if row["real_loop_flag"] else "matched_non_loop",
                "sample_id": row["sample_id"], "pair_id": "|".join(sorted((row["sample_id"], row["matched_non_loop_id"]))), "scene_id": row["scene_id"],
                "path_length": row["path_length"], "closure_motion_distance": row["closure_motion_distance"],
                "prediction_to_actual_endpoint_cosine": float(cosine_distance(pred, endpoint).item()),
                "actual_endpoint_to_anchor_cosine": float(cosine_distance(endpoint, anchor[0]).item()),
                "prediction_to_anchor_cosine": float(cosine_distance(pred, anchor[0]).item()),
            })

    write_csv(args.output_dir / "actionability_per_sample.csv", action_results)
    write_csv(args.output_dir / "dynamics_per_sample.csv", dynamics_results)
    write_csv(args.output_dir / "algebra_per_sample.csv", algebra_results)
    write_json(args.output_dir / "training_and_integrity.json", {
        "status": "COMPLETE_VALID", "model": args.model, "seeds": list(SEEDS),
        "request_manifest_sha256": sha256(args.request_manifest),
        "feature_root": str(args.feature_root.resolve()),
        "fit_dev_policy": "shared scene-disjoint train-derived split; validation unused for selection",
        "matched_action_shuffle_unavailable": unmatched,
        "checkpoints": {path.name: sha256(path) for path in sorted(checkpoint_dir.glob("*.pt"))},
        "actionability_rows": len(action_results), "dynamics_rows": len(dynamics_results),
        "algebra_rows": len(algebra_results), "training": training_log,
    })
    os.rename(args.output_dir, output_target)
    print(json.dumps({"status": "COMPLETE_VALID", "model": args.model,
                      "actionability_rows": len(action_results), "dynamics_rows": len(dynamics_results),
                      "algebra_rows": len(algebra_results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
