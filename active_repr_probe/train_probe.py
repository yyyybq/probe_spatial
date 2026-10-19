#!/usr/bin/env python3
"""Train small frozen-feature probes with scene-disjoint splits and controls.

This is intentionally a compact MVP trainer.  It accepts only feature files
emitted by extract_features.py and checks sample-id equality before training.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

from .io_utils import read_jsonl, write_json


def _torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("BLOCKED: probe training requires torch in the selected environment") from exc
    return torch


def _label(row: dict[str, Any], task: str) -> int | None:
    labels = row.get("labels", {})
    if task == "relative_direction":
        value = labels.get("relative_direction")
        return int(value["azimuth_bin"]) if isinstance(value, dict) and "azimuth_bin" in value else None
    if task == "distance_bin":
        value = labels.get("distance_bin")
        return int(value) if isinstance(value, int) else None
    if task == "visibility":
        value = labels.get("visibility")
        return int(bool(value["is_visible"])) if isinstance(value, dict) and value.get("status") == "VERIFIED" else None
    if task == "occlusion":
        value = labels.get("visibility")
        mapping = {"none": 0, "wall": 1, "object": 2}
        return mapping.get(str(value.get("occlusion_type"))) if isinstance(value, dict) and value.get("status") == "VERIFIED" else None
    if task == "camera_rotation":
        value = labels.get("camera_rotation_from_previous_deg")
        return int(float(value) > 1e-3) if isinstance(value, (int, float)) else None
    if task == "target_relation":
        value = labels.get("target_relation")
        return int(bool(value["current_relation_satisfied"])) if isinstance(value, dict) and value.get("status") == "VERIFIED" else None
    raise ValueError(f"action_gain is emitted as a separate counterfactual table and is not supported by this first trainer: {task}")


def _metrics(y_true: list[int], y_pred: list[int], n_classes: int) -> dict[str, float]:
    per_recall, f1s = [], []
    for cls in range(n_classes):
        tp = sum(a == cls and b == cls for a, b in zip(y_true, y_pred))
        fp = sum(a != cls and b == cls for a, b in zip(y_true, y_pred))
        fn = sum(a == cls and b != cls for a, b in zip(y_true, y_pred))
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        per_recall.append(recall)
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {"balanced_accuracy": sum(per_recall) / n_classes, "macro_f1": sum(f1s) / n_classes, "accuracy": sum(a == b for a, b in zip(y_true, y_pred)) / max(1, len(y_true))}


def _binary_ranking(y_true: list[int], scores: list[float]) -> dict[str, float]:
    positive = sum(y_true)
    negative = len(y_true) - positive
    if not positive or not negative:
        return {"auroc": float("nan"), "pr_auc": float("nan"), "positive_rate": positive / max(1, len(y_true))}
    ordered = sorted(zip(scores, y_true), reverse=True)
    tp = 0; ap = 0.0
    for rank, (_, label) in enumerate(ordered, 1):
        if label:
            tp += 1; ap += tp / rank
    # Pairwise rank equivalent.  Equal-score ties are immaterial for this MVP
    # diagnostic and are reported together with raw scores in the result file.
    wins = sum((a > b) + 0.5 * (a == b) for a, ya in zip(scores, y_true) if ya for b, yb in zip(scores, y_true) if not yb)
    return {"auroc": wins / (positive * negative), "pr_auc": ap / positive, "positive_rate": positive / len(y_true)}


def _fit_eval(torch: Any, x, y, split, kind: str, seed: int, epochs: int, groups=None, gains=None, shuffle_train_labels: bool = False) -> dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    train = [i for i, value in enumerate(split) if value == "train"]
    val = [i for i, value in enumerate(split) if value == "val"]
    test = [i for i, value in enumerate(split) if value == "test"]
    if not train or not val or not test:
        raise RuntimeError("BLOCKED: probe requires nonempty scene-disjoint train/val/test partitions")
    classes = int(y.max().item()) + 1
    mean, std = x[train].mean(0), x[train].std(0).clamp_min(1e-6)
    z = (x - mean) / std
    model = torch.nn.Linear(z.shape[1], classes) if kind == "linear" else torch.nn.Sequential(torch.nn.Linear(z.shape[1], 256), torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(256, classes))
    counts = torch.bincount(y[train], minlength=classes).float().clamp_min(1)
    train_y = y[train].clone()
    if shuffle_train_labels:
        permutation = torch.randperm(len(train_y))
        train_y = train_y[permutation]
    criterion = torch.nn.CrossEntropyLoss(weight=(counts.sum() / counts))
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state = -1.0, None
    for _ in range(epochs):
        model.train(); optim.zero_grad(); loss = criterion(model(z[train]), train_y); loss.backward(); optim.step()
        model.eval()
        with torch.no_grad():
            pred = model(z[val]).argmax(-1).tolist()
        score = _metrics(y[val].tolist(), pred, classes)["balanced_accuracy"]
        if score > best:
            best = score; best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        logits = model(z[test]); pred = logits.argmax(-1).tolist()
    out = _metrics(y[test].tolist(), pred, classes)
    if classes == 2:
        scores = torch.softmax(logits, dim=-1)[:, 1].tolist()
        out.update(_binary_ranking(y[test].tolist(), scores))
        if groups is not None and gains is not None:
            by_group: dict[str, list[tuple[float, float]]] = {}
            for local, global_index in enumerate(test):
                by_group.setdefault(groups[global_index], []).append((scores[local], gains[global_index]))
            regrets = []
            for entries in by_group.values():
                chosen = max(entries, key=lambda value: value[0])[1]
                oracle = max(value[1] for value in entries)
                regrets.append(oracle - chosen)
            out["action_selection_regret"] = sum(regrets) / max(1, len(regrets))
    out.update({"val_balanced_accuracy": best, "n_train": len(train), "n_val": len(val), "n_test": len(test), "n_classes": classes})
    return out


def _action_examples(torch: Any, values, ids: list[str], rows: dict[str, dict[str, Any]], task: str):
    key = "distance_decrease" if task == "action_distance_decrease" else "task_score_improve"
    gain_key = "distance_after_m" if task == "action_distance_decrease" else "task_score_after"
    names = sorted({str(cf.get("requested_action")) for sample_id in ids for cf in rows[sample_id].get("counterfactual_actions", []) if cf.get("requested_action")})
    index_of = {name: idx for idx, name in enumerate(names)}
    vectors = []; labels = []; splits = []; groups = []; gains = []
    for sample_index, sample_id in enumerate(ids):
        for cf in rows[sample_id].get("counterfactual_actions", []):
            label = cf.get(key)
            if not isinstance(label, bool) or str(cf.get("requested_action")) not in index_of:
                continue
            expanded = torch.zeros(values.shape[1] * len(names), dtype=torch.float32)
            offset = index_of[str(cf["requested_action"])] * values.shape[1]
            expanded[offset:offset + values.shape[1]] = values[sample_index].float()
            raw_gain = cf.get(gain_key)
            if not isinstance(raw_gain, (float, int)):
                continue
            vectors.append(expanded); labels.append(int(label)); splits.append(rows[sample_id]["split"]); groups.append(sample_id)
            gains.append(-float(raw_gain) if task == "action_distance_decrease" else float(raw_gain))
    if not vectors:
        return None
    return torch.stack(vectors), torch.tensor(labels, dtype=torch.long), splits, groups, gains, names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--state-bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=["relative_direction", "distance_bin", "visibility", "occlusion", "camera_rotation", "target_relation", "action_distance_decrease", "action_task_score_improve"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--control", choices=("none", "text_only"), default="none")
    parser.add_argument("--run-label-shuffle", action="store_true")
    args = parser.parse_args()
    torch = _torch()
    payload = torch.load(args.features, map_location="cpu", weights_only=False)
    rows = {str(row["sample_id"]): row for row in read_jsonl(args.state_bank)}
    ids = [str(value) for value in payload["sample_ids"]]
    if len(ids) != len(set(ids)) or any(value not in rows for value in ids):
        raise SystemExit("BLOCKED: features and state bank have missing/duplicate sample identities")
    results: list[dict[str, Any]] = []
    for tap, values in payload["taps"].items():
        for task in args.tasks:
            if task.startswith("action_"):
                action_data = _action_examples(torch, values, ids, rows, task)
                if action_data is None:
                    results.append({"task": task, "tap": tap, "status": "BLOCKED", "reason": "no verified counterfactual labels"}); continue
                x, y, split, groups, gains, action_names = action_data
                for kind in ("linear", "mlp_d256"):
                    for seed in (41, 42, 43):
                        for control in (["none", "label_shuffle"] if args.run_label_shuffle else ["none"]):
                            try:
                                result = _fit_eval(torch, x, y, split, "linear" if kind == "linear" else "mlp", seed, args.epochs, groups, gains, shuffle_train_labels=control == "label_shuffle")
                                results.append({"task": task, "tap": tap, "probe": kind, "seed": seed, "status": "COMPLETE", "control": args.control if control == "none" else control, "candidate_actions": "|".join(action_names), **result})
                            except RuntimeError as exc:
                                results.append({"task": task, "tap": tap, "probe": kind, "seed": seed, "control": args.control if control == "none" else control, "status": "BLOCKED", "reason": str(exc)})
                continue
            selected = [(index, _label(rows[sample_id], task)) for index, sample_id in enumerate(ids)]
            selected = [(index, label) for index, label in selected if label is not None]
            if not selected:
                results.append({"task": task, "tap": tap, "status": "BLOCKED", "reason": "no verified labels"}); continue
            index, labels = zip(*selected)
            x = values[list(index)].float(); y = torch.tensor(labels, dtype=torch.long)
            split = [rows[ids[i]]["split"] for i in index]
            for kind in ("linear", "mlp_d256"):
                for seed in (41, 42, 43):
                    for control in (["none", "label_shuffle"] if args.run_label_shuffle else ["none"]):
                        try:
                            result = _fit_eval(torch, x, y, split, "linear" if kind == "linear" else "mlp", seed, args.epochs, shuffle_train_labels=control == "label_shuffle")
                            results.append({"task": task, "tap": tap, "probe": kind, "seed": seed, "control": args.control if control == "none" else control, "status": "COMPLETE", **result})
                        except RuntimeError as exc:
                            results.append({"task": task, "tap": tap, "probe": kind, "seed": seed, "control": args.control if control == "none" else control, "status": "BLOCKED", "reason": str(exc)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in results for key in row})
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(results)
    write_json(args.output_dir / "results.json", {"status": "COMPLETE", "control": args.control, "label_shuffle": args.run_label_shuffle, "results": results})


if __name__ == "__main__":
    main()
