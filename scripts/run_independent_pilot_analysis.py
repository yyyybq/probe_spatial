#!/usr/bin/env python3
"""Clean, from-scratch Actionability and Action Algebra pilot analysis.

No checkpoint path is accepted: every trainable probe starts in this process from
a fixed random seed.  Inputs are resolved only through exact-v2 unit receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o644)
        os.link(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def feature_path(root: Path, unit: dict[str, Any]) -> Path:
    return root / "dino" / "scannetpp" / unit["scene_id"] / f"{unit['unit_id']}.safetensors"


class FeatureStore:
    def __init__(self, request: Path, root: Path, tag: str) -> None:
        self.root, self.tag = root, tag
        self.isolated: dict[str, dict[str, Any]] = {}
        self.contexts: dict[tuple[str, ...], dict[str, Any]] = {}
        for unit in read_jsonl(request):
            if unit["model"] != "dino":
                continue
            if unit["feature_role"] == "isolated_frame":
                previous = self.isolated.get(unit["raw_frame_id"])
                if previous is not None and previous["unit_id"] != unit["unit_id"]:
                    raise ValueError(f"ambiguous isolated raw ID {unit['raw_frame_id']}")
                self.isolated[unit["raw_frame_id"]] = unit
            elif unit["feature_role"] == "streaming_context_p8":
                key = tuple(unit["context_raw_ids"])
                previous = self.contexts.get(key)
                if previous is not None and previous["unit_id"] != unit["unit_id"]:
                    raise ValueError(f"ambiguous ordered context {key}")
                self.contexts[key] = unit
            else:
                raise ValueError(f"unsupported feature role {unit['feature_role']}")
        self.cache: dict[str, torch.Tensor] = {}

    def vector(self, raw_id: str) -> torch.Tensor:
        unit = self.isolated.get(raw_id)
        if unit is None:
            raise KeyError(f"missing exact isolated unit for {raw_id}")
        return self._load(unit)

    def context(self, raw_ids: list[str]) -> torch.Tensor:
        unit = self.contexts.get(tuple(raw_ids))
        if unit is None:
            raise KeyError(f"missing exact ordered p8 context {raw_ids}")
        return self._load(unit)

    def _load(self, unit: dict[str, Any]) -> torch.Tensor:
        cached = self.cache.get(unit["unit_id"])
        if cached is not None:
            return cached
        path = feature_path(self.root, unit)
        sidecar = Path(f"{path}.manifest.json")
        meta = json.loads(sidecar.read_text())
        required = {
            "science_tag": self.tag,
            "unit_id": unit["unit_id"],
            "raw_frame_id": unit["raw_frame_id"],
            "context_raw_ids": unit["context_raw_ids"],
            "fallback": False,
        }
        if any(meta.get(key) != value for key, value in required.items()):
            raise ValueError(f"sidecar mismatch for {path}")
        tensor = load_file(str(path))["feat"].float()
        # For ordered p8 units use the eighth/current frame; isolated units have
        # a leading singleton dimension.  Spatial tokens are mean pooled only.
        if unit["feature_role"] == "streaming_context_p8":
            tensor = tensor[-1]
        else:
            tensor = tensor[0]
        vector = tensor.reshape(-1, tensor.shape[-1]).mean(dim=0).contiguous()
        self.cache[unit["unit_id"]] = vector
        return vector


def stack(rows: list[torch.Tensor], device: str) -> torch.Tensor:
    return torch.stack(rows).float().to(device)


class Classifier(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        hidden = min(512, max(64, width // 4))
        self.net = torch.nn.Sequential(torch.nn.Linear(width, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, 2))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class Dynamics(torch.nn.Module):
    def __init__(self, width: int, action_width: int = 9) -> None:
        super().__init__()
        hidden = min(768, max(128, width // 2))
        self.net = torch.nn.Sequential(
            torch.nn.Linear(width + action_width, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, width)
        )

    def forward(self, current: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return current + self.net(torch.cat((current, action), dim=1))


def normalize(train: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = train.mean(0, keepdim=True)
    scale = train.std(0, keepdim=True).clamp_min(1e-6)
    return (train - mean) / scale, mean, scale


def fit_classifier(train_x: torch.Tensor, train_y: torch.Tensor, val_x: torch.Tensor, val_y: torch.Tensor) -> float:
    normalized, mean, scale = normalize(train_x, train_x)
    model = Classifier(train_x.shape[1]).to(train_x.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(160):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(normalized), train_y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return float((model((val_x - mean) / scale).argmax(1) == val_y).float().mean().item())


def fit_dynamics(train_current: torch.Tensor, train_action: torch.Tensor, train_target: torch.Tensor, device: str) -> Dynamics:
    model = Dynamics(train_current.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(220):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(train_current, train_action), train_target)
        loss.backward()
        optimizer.step()
    return model.eval()


def mse_cosine(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    return {
        "mse": float(torch.nn.functional.mse_loss(pred, target).item()),
        "mean_cosine": float(torch.nn.functional.cosine_similarity(pred, target, dim=1).mean().item()),
    }


def pose_action(anchor: list[list[float]], endpoint: list[list[float]]) -> list[float]:
    a, b = np.asarray(anchor, dtype=np.float64), np.asarray(endpoint, dtype=np.float64)
    aw, bw = np.linalg.inv(a), np.linalg.inv(b)
    rotation = bw[:3, :3] @ aw[:3, :3].T
    translation = bw[:3, 3] - rotation @ aw[:3, 3]
    return [float(value) for value in np.concatenate((rotation[:2].reshape(-1), translation))]


def inverse_action(action: torch.Tensor) -> torch.Tensor:
    values = action.detach().cpu().numpy()
    inverse: list[list[float]] = []
    for value in values:
        first = value[:3] / np.linalg.norm(value[:3])
        second = value[3:6] - first * np.dot(first, value[3:6])
        second = second / np.linalg.norm(second)
        third = np.cross(first, second)
        rotation = np.stack((first, second, third))
        translation = value[6:9]
        ri, ti = rotation.T, -rotation.T @ translation
        inverse.append(np.concatenate((ri[:2].reshape(-1), ti)).tolist())
    return torch.tensor(inverse, dtype=action.dtype, device=action.device)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--science-tag", required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--request-manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite analysis namespace: {args.output_dir}")
    torch.manual_seed(20260831)
    np.random.seed(20260831)
    random.seed(20260831)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    store = FeatureStore(args.request_manifest.resolve(), args.feature_root.resolve(), args.science_tag)
    directory = args.manifest_dir.resolve()
    action_train = read_jsonl(directory / "science_pilot_subset_v1_actionability_train.jsonl")
    action_val = read_jsonl(directory / "science_pilot_subset_v1_actionability_val.jsonl")
    algebra_train = read_jsonl(directory / "science_pilot_subset_v1_algebra_train.jsonl")
    algebra_val = read_jsonl(directory / "science_pilot_subset_v1_algebra_val.jsonl")
    loops = read_jsonl(directory / "science_pilot_subset_v1_algebra_real_loops.jsonl")
    device = args.device

    def action_inputs(rows: list[dict[str, Any]], mode: str) -> tuple[torch.Tensor, torch.Tensor]:
        values, labels = [], []
        for row in rows:
            context, anchor = store.context(row["context_raw_ids"]), store.vector(row["anchor_raw_id"])
            goal = store.vector(row["goal_raw_id"])
            endpoints = torch.cat((store.vector(row["candidate_endpoint_raw_id_1"]), store.vector(row["candidate_endpoint_raw_id_2"])))
            actions = torch.tensor(row["action_1_9d"] + row["action_2_9d"], dtype=torch.float32)
            if mode == "endpoint_oracle": value = torch.cat((goal, endpoints))
            elif mode == "direct": value = torch.cat((context, anchor, goal, actions))
            elif mode == "action": value = actions
            elif mode == "current": value = torch.cat((context, anchor))
            elif mode == "goal": value = goal
            else: raise ValueError(mode)
            values.append(value)
            labels.append(int(row["best_candidate"]) - 1)
        return stack(values, device), torch.tensor(labels, device=device)

    oracle_train, labels_train = action_inputs(action_train, "endpoint_oracle")
    oracle_val, labels_val = action_inputs(action_val, "endpoint_oracle")
    direct_train, _ = action_inputs(action_train, "direct")
    direct_val, _ = action_inputs(action_val, "direct")
    action_train_x, _ = action_inputs(action_train, "action")
    action_val_x, _ = action_inputs(action_val, "action")
    current_train, _ = action_inputs(action_train, "current")
    current_val, _ = action_inputs(action_val, "current")
    goal_train, _ = action_inputs(action_train, "goal")
    goal_val, _ = action_inputs(action_val, "goal")
    temporal_majority = {direction: Counter(int(row["best_candidate"]) - 1 for row in action_train if row["goal_temporal_direction"] == direction).most_common(1)[0][0] for direction in ("past", "future")}
    temporal_accuracy = float(np.mean([temporal_majority[row["goal_temporal_direction"]] == int(row["best_candidate"]) - 1 for row in action_val]))
    action_table = {
        "endpoint_feature_oracle": fit_classifier(oracle_train, labels_train, oracle_val, labels_val),
        "direct_goal_conditioned_action_output": fit_classifier(direct_train, labels_train, direct_val, labels_val),
        "random": 0.5,
        "action_only": fit_classifier(action_train_x, labels_train, action_val_x, labels_val),
        "current_only": fit_classifier(current_train, labels_train, current_val, labels_val),
        "goal_only": fit_classifier(goal_train, labels_train, goal_val, labels_val),
        "temporal_direction_majority": temporal_accuracy,
    }

    def dynamics_rows(rows: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        current, h1, h2, h4, actions = [], [], [], [], []
        for row in rows:
            current.append(store.vector(row["anchor_raw_id"]))
            h1.append(store.vector(row["horizon_target_raw_ids"]["h1"]))
            h2.append(store.vector(row["horizon_target_raw_ids"]["h2"]))
            h4.append(store.vector(row["horizon_target_raw_ids"]["h4"]))
            actions.append(torch.tensor(row["primitive_actions_9d"], dtype=torch.float32))
        return stack(current, device), stack(h1, device), stack(h2, device), stack(h4, device), torch.stack(actions).to(device)

    train_current, train_h1, _, _, train_actions = dynamics_rows(algebra_train)
    val_current, val_h1, val_h2, val_h4, val_actions = dynamics_rows(algebra_val)
    clean = fit_dynamics(train_current, train_actions[:, 0], train_h1, device)
    visual = fit_dynamics(train_current, torch.zeros_like(train_actions[:, 0]), train_h1, device)
    action_only = fit_dynamics(torch.zeros_like(train_current), train_actions[:, 0], train_h1, device)
    shuffled = fit_dynamics(train_current, train_actions[torch.randperm(len(train_actions)), 0], train_h1, device)
    with torch.no_grad():
        clean_h1 = clean(val_current, val_actions[:, 0])
        zero_action = torch.tensor([[1, 0, 0, 0, 1, 0, 0, 0, 0]], dtype=torch.float32, device=device).repeat(len(val_current), 1)
        step2 = clean(clean_h1, val_actions[:, 1])
        step3 = clean(step2, val_actions[:, 2])
        step4 = clean(step3, val_actions[:, 3])
        full_actions = torch.tensor([pose_action(row["anchor_pose"], row["final_pose"]) for row in algebra_val], device=device, dtype=torch.float32)
        inverse = clean(val_h4, inverse_action(full_actions))
        dynamics_table = {
            "clean_local_dynamics_h1": mse_cosine(clean_h1, val_h1),
            "sanity_copy": mse_cosine(val_current, val_h1),
            "sanity_visual_only": mse_cosine(visual(val_current, torch.zeros_like(val_actions[:, 0])), val_h1),
            "sanity_action_only": mse_cosine(action_only(torch.zeros_like(val_current), val_actions[:, 0]), val_h1),
            "sanity_matched_shuffle": mse_cosine(shuffled(val_current, val_actions[:, 0]), val_h1),
            "composition_two_step": mse_cosine(step2, val_h2),
            "composition_four_step": mse_cosine(step4, val_h4),
            "identity": mse_cosine(clean(val_current, zero_action), val_current),
            "inverse_cycle": mse_cosine(inverse, val_current),
        }
        loop_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in loops:
            loop_groups["real_loop" if row["real_loop_flag"] else "matched_non_loop"].append(row)
        real_loop: dict[str, dict[str, float]] = {}
        for name, rows in loop_groups.items():
            starts, ends, actions = [], [], []
            for row in rows:
                starts.append(store.vector(row["anchor_raw_id"]))
                ends.append(store.vector(row["final_raw_id"]))
                actions.append(torch.tensor(pose_action(row["anchor_pose"], row["final_pose"]), dtype=torch.float32))
            real_loop[name] = mse_cosine(clean(stack(starts, device), stack(actions, device)), stack(ends, device))

    args.output_dir.mkdir(parents=True)
    torch.save({"state_dict": clean.state_dict(), "seed": 20260831, "kind": "clean_state_action_h1"}, args.output_dir / "frozen_dynamics_clean_state_action.pt")
    result = {
        "schema_version": 1,
        "science_tag": args.science_tag,
        "status": "COMPLETE_VALID",
        "training_policy": "all probes initialized from scratch in this run; no checkpoint was loaded",
        "request_manifest": str(args.request_manifest.resolve()),
        "request_manifest_sha256": sha256(args.request_manifest),
        "feature_root": str(args.feature_root.resolve()),
        "actionability": {"train": len(action_train), "val": len(action_val), "accuracy": action_table},
        "algebra": {"train": len(algebra_train), "val": len(algebra_val), "dynamics": dynamics_table, "real_loop": real_loop},
        "frozen_dynamics_checkpoint": str((args.output_dir / "frozen_dynamics_clean_state_action.pt").resolve()),
    }
    atomic_json(args.output_dir / "pilot_table.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
