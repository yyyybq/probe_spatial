#!/usr/bin/env python3
"""Exploratory cross-model observed-order pilot on a frozen tuple intersection.

This runner deliberately reuses the r6 observed-video contract.  It does not
construct data, render endpoints, tune layers against order scores, or use any
structure loss.  Its sole purpose is a small, non-canonical group-meeting
comparison: fixed C1/C2 artifacts, one common tuple intersection, matched
scene split, predictor capacity, objective, baselines, and seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from types import SimpleNamespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from analyses.se3_latent_mvp.run_mvp import (
    LatentPredictor,
    cosine_distance,
    load_and_audit_tuples,
    matched_donor_mapping,
    pair_tensors,
    records_for_split,
    split_scenes,
    train_model,
    transform_batch,
)
from analyses.se3_protocol_v2.observed_order import (
    BASELINES,
    LEARNED_CONDITIONS,
    action_tensor,
    derangement,
    learned_predict,
    mean_metrics,
    one_step_metrics,
)
from analyses.se3_protocol_v2.protocol import POSE_CONVENTION, dump_json, sha256


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = "logs/inscene15k_streaming/runs/c_eval_20260724_1903"
MANIFEST = "manifests/c_retrain_20260722/inscene_c_h124_prefix8.jsonl"
RAW_ROOT = "/mnt/umm/users/yinbaiqiao/InsScene-15K/data/processed_scannetpp_v2/processed_scannetpp_v2"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    layer: int
    prefix: str
    context_role: str
    c1_dump: str
    c2_dump: str
    lineage_status: str


def _dump(model: str, task: str, layer: int) -> str:
    return f"{RUN_ROOT}/c_eval_20260724_1903_{task}_{model}_p8_layer{layer}/eval/val_predictions.pt"


# Pre-registered before running this pilot.  The layer/prefix were read from
# the existing C-eval configs; no value here was selected from order results.
MODEL_SPECS = (
    ModelSpec("dino", "DINO", 0, "p8", "single-frame isolated endpoint; p8 fixes observed tuple sampling", _dump("dino", "action_dynamics", 0), _dump("dino", "path_integration", 0), "audited C1/C2 lineage"),
    ModelSpec("f3r", "F3R/Fast3R", 24, "p8", "single-frame geometric endpoint; p8 fixes observed tuple sampling", _dump("f3r", "action_dynamics", 24), _dump("f3r", "path_integration", 24), "audited C1/C2 lineage"),
    ModelSpec("vjepa2", "V-JEPA2", 23, "p8", "temporal-model p8 endpoint role", _dump("vjepa2", "action_dynamics", 23), _dump("vjepa2", "path_integration", 23), "audited C1/C2 lineage"),
    ModelSpec("cogvideox", "CogVideoX", 20, "p8", "video-generation-model p8 endpoint role", _dump("cogvideox", "action_dynamics", 20), _dump("cogvideox", "path_integration", 20), "audited C1/C2 lineage"),
)

UNAVAILABLE_OR_EXCLUDED = (
    {"model": "VideoMAE", "status": "N/A", "reason": "No C-eval C1/C2 val_predictions artifact found; no extraction started."},
    {"model": "V-JEPA (legacy)", "status": "N/A", "reason": "No C-eval C1/C2 val_predictions artifact found; V-JEPA2 is used when explicitly named."},
    {"model": "Qwen2.5-VL", "status": "N/A", "reason": "No direct C-series feature artifact found."},
    {"model": "Wan p8", "status": "excluded", "reason": "Known lineage/coverage risk from the frozen project protocol."},
)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must contain unique integers")
    return seeds


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def tuple_key(record: Any) -> tuple[str, str, tuple[int, int, int]]:
    return (str(record.scene_id), str(record.sample_id), tuple(int(value) for value in record.raw_frame_indices))


def load_single_records(spec: ModelSpec) -> tuple[list[Any], dict[str, Any]]:
    """Load one C artifact only, keeping peak memory independent of model count."""
    cfg = {
        "manifest": str(resolve(MANIFEST)),
        "c1_dump": str(resolve(spec.c1_dump)),
        "c2_dump": str(resolve(spec.c2_dump)),
        "raw_scene_root": str(resolve(RAW_ROOT)),
        "alignment_atol": 2e-5,
        "pose_atol": 1e-5,
    }
    records, result = load_and_audit_tuples(cfg)
    keys = [tuple_key(record) for record in records]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"{spec.key}: duplicate tuple key prevents a safe intersection")
    return records, {
        "raw_tuple_count": len(records),
        "raw_scene_count": len({record.scene_id for record in records}),
        "feature_dim": int(records[0].features.shape[-1]),
        "tuple_gates": result["gates"],
        "manifest": cfg["manifest"],
        "c1_dump": cfg["c1_dump"],
        "c2_dump": cfg["c2_dump"],
        "lineage_status": spec.lineage_status,
    }


def evaluate_seed(records: list[Any], seed: int, cfg_base: dict[str, Any], assignments: dict[str, str]) -> tuple[list[dict[str, Any]], dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    cfg = {**cfg_base, "seed": seed}
    partitions = {name: records_for_split(records, assignments, name) for name in ("train", "val", "test")}
    _, train_action, _ = pair_tensors(partitions["train"])
    action_mean, action_std = train_action.mean(0), train_action.std(0).clamp_min(1e-5)
    models: dict[str, LatentPredictor] = {}
    training: dict[str, Any] = {}
    for condition in LEARNED_CONDITIONS:
        torch.manual_seed(seed)
        model, history = train_model(condition, partitions["train"], partitions["val"], cfg, action_mean, action_std)
        models[condition] = model
        training[condition] = {"epochs": len(history), "best_val_loss": min(row["val_loss"] for row in history)}
    features = torch.stack([record.features for record in partitions["test"]])
    z_a, z_b, z_c = features[:, 0], features[:, 1], features[:, 2]
    feature_std = torch.stack([record.features for record in partitions["train"]]).reshape(-1, features.shape[-1]).std(0).clamp_min(1e-5)
    true_t1, true_t2, _ = transform_batch(partitions["test"])
    donor = matched_donor_mapping(partitions["test"])
    shuffled_1, shuffled_2 = derangement(len(true_t1), seed + 101), derangement(len(true_t2), seed + 211)
    action_sets = {
        "copy": (true_t1, true_t2),
        "without_condition": (true_t1, true_t2),
        "action_only": (true_t1, true_t2),
        "full": (true_t1, true_t2),
        "matched_wrong_pose": (true_t1[donor], true_t2[donor]),
        "shuffled_pose": (true_t1[shuffled_1], true_t2[shuffled_2]),
    }
    copy_first = learned_predict("copy", models, z_a, true_t1, action_mean, action_std)
    copy_correct = learned_predict("copy", models, copy_first, true_t2, action_mean, action_std)
    copy_error = float(cosine_distance(copy_correct, z_c).mean())
    rows: list[dict[str, Any]] = []
    arrays_by_baseline: dict[str, dict[str, np.ndarray]] = {}
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
        summary.update({"baseline": baseline, "seed": seed})
        rows.append(summary)
        arrays_by_baseline[baseline] = arrays
        predicted_pairs[baseline] = (correct, reversed_)
    full_correct, full_reverse = predicted_pairs["full"]
    reverse_summary, reverse_arrays = mean_metrics(full_reverse, full_correct, learned_predict("full", models, z_a, np.stack([left @ right for left, right in zip(true_t1, true_t2)]), action_mean, action_std), z_c, feature_std, copy_error)
    reverse_summary.update(one_step_metrics(
        learned_predict("full", models, z_a, true_t2, action_mean, action_std),
        learned_predict("full", models, z_b, true_t1, action_mean, action_std), z_b, z_c,
    ))
    reverse_summary.update({"baseline": "reversed_order", "seed": seed})
    rows.insert(-1, reverse_summary)
    arrays_by_baseline["reversed_order"] = reverse_arrays
    checks = {
        "matched_donor_fixed_points": int(np.sum(donor == np.arange(len(donor)))),
        "matched_donor_same_scene": int(sum(partitions["test"][i].scene_id == partitions["test"][j].scene_id for i, j in enumerate(donor))),
        "shuffled_step1_fixed_points": int(np.sum(shuffled_1 == np.arange(len(shuffled_1)))),
        "shuffled_step2_fixed_points": int(np.sum(shuffled_2 == np.arange(len(shuffled_2)))),
        "training_one_step_only": True,
    }
    if any(value for key, value in checks.items() if key != "training_one_step_only"):
        raise RuntimeError(f"seed {seed}: pose-control derangement failed: {checks}")
    return rows, arrays_by_baseline, {"training": training, "checks": checks}


MAIN_COLUMNS = (
    "model", "layer", "prefix", "one_step_grounding_cosine", "correct_endpoint_cosine",
    "reversed_order_endpoint_cosine", "grounded_order_margin_cosine", "order_ranking_accuracy", "copy_normalized_improvement",
)


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        for baseline in BASELINES:
            selected = [row for row in rows if row["model"] == spec.key and row["baseline"] == baseline]
            if not selected:
                continue
            summary: dict[str, Any] = {"model": spec.key, "display_name": spec.display_name, "layer": spec.layer, "prefix": spec.prefix, "context_role": spec.context_role, "baseline": baseline, "n_seeds": len(selected)}
            for key in selected[0]:
                if key in {"model", "baseline", "seed"}:
                    continue
                if isinstance(selected[0][key], (float, int, np.floating, np.integer)):
                    values = np.asarray([row[key] for row in selected], dtype=float)
                    summary[f"{key}_mean"] = float(values.mean())
                    summary[f"{key}_std"] = float(values.std(ddof=0))
            result.append(summary)
    return result


def scene_bootstrap(
    model: str,
    per_seed_arrays: list[dict[str, dict[str, np.ndarray]]],
    test_records: list[Any],
    repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    scenes = sorted({record.scene_id for record in test_records})
    indices_by_scene = {scene: np.asarray([index for index, record in enumerate(test_records) if record.scene_id == scene]) for scene in scenes}
    rng = np.random.default_rng(seed)
    samples = {"grounding_gain_vs_copy": [], "grounded_order_margin_cosine": [], "order_ranking_accuracy": []}
    for _ in range(repetitions):
        chosen = rng.choice(scenes, size=len(scenes), replace=True)
        index = np.concatenate([indices_by_scene[str(scene)] for scene in chosen])
        per_seed = {key: [] for key in samples}
        for arrays in per_seed_arrays:
            copy = arrays["copy"]["correct_cosine"][index].mean()
            full = arrays["full"]
            per_seed["grounding_gain_vs_copy"].append(float((copy - full["correct_cosine"][index].mean()) / max(copy, 1e-8)))
            per_seed["grounded_order_margin_cosine"].append(float(full["margin_cosine"][index].mean()))
            per_seed["order_ranking_accuracy"].append(float(full["ranking"][index].mean()))
        for key in samples:
            samples[key].append(float(np.mean(per_seed[key])))
    return [
        {
            "model": model,
            "metric": key,
            "bootstrap_unit": "scene",
            "resamples": repetitions,
            "estimate": float(np.mean(values)),
            "ci95_low": float(np.quantile(values, .025)),
            "ci95_high": float(np.quantile(values, .975)),
        }
        for key, values in samples.items()
    ]


def render_figures(output: Path, main: list[dict[str, Any]], action_only: list[dict[str, Any]]) -> None:
    labels = [row["display_name"] for row in main]
    figure, axis = plt.subplots(figsize=(7.2, 5.1))
    for row in main:
        axis.scatter(row["copy_normalized_improvement_mean"], row["grounded_order_margin_cosine_mean"], s=70)
        axis.annotate(row["display_name"], (row["copy_normalized_improvement_mean"], row["grounded_order_margin_cosine_mean"]), xytext=(5, 4), textcoords="offset points", fontsize=9)
    axis.axhline(0, color="black", linewidth=.8)
    axis.axvline(0, color="black", linewidth=.8)
    axis.set_xlabel("grounding gain vs copy (higher is better)")
    axis.set_ylabel("grounded order margin (higher is better)")
    axis.set_title("Groundability vs order structure")
    axis.grid(alpha=.2)
    figure.tight_layout()
    figure.savefig(output / "figure1_groundability_vs_order.png", dpi=200)
    figure.savefig(output / "figure1_groundability_vs_order.pdf")
    plt.close(figure)

    x = np.arange(len(main))
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    axis.bar(x - .18, [row["correct_endpoint_cosine_mean"] for row in main], .36, label="correct order")
    axis.bar(x + .18, [row["reversed_order_endpoint_cosine_mean"] for row in main], .36, label="reversed order")
    axis.set_xticks(x, labels, rotation=12)
    axis.set_ylabel("2-step endpoint cosine distance (lower is better)")
    axis.set_title("Correct vs reversed observed endpoint")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=.2)
    figure.tight_layout()
    figure.savefig(output / "figure2_correct_vs_reversed.png", dpi=200)
    figure.savefig(output / "figure2_correct_vs_reversed.pdf")
    plt.close(figure)

    by_model_action = {row["model"]: row for row in action_only}
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    axis.bar(x - .18, [row["order_ranking_accuracy_mean"] for row in main], .36, label="full")
    axis.bar(x + .18, [by_model_action[row["model"]]["order_ranking_accuracy_mean"] for row in main], .36, label="action-only")
    axis.axhline(.5, color="black", linewidth=.9, linestyle="--", label="chance")
    axis.set_xticks(x, labels, rotation=12)
    axis.set_ylim(0, 1)
    axis.set_ylabel("order ranking accuracy")
    axis.set_title("Order accuracy across models")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=.2)
    figure.tight_layout()
    figure.savefig(output / "figure3_order_accuracy.png", dpi=200)
    figure.savefig(output / "figure3_order_accuracy.pdf")
    plt.close(figure)


def read_r6_dino() -> dict[str, dict[str, float]]:
    path = REPO_ROOT / "results/experimental/se3_protocol_v2_observed_dino_l0_seed42_r6/observed_order_metrics.csv"
    if not path.is_file():
        raise RuntimeError(f"DINO regression reference absent: {path}")
    with path.open() as handle:
        return {row["baseline"]: {key: float(value) for key, value in row.items() if key != "baseline"} for row in csv.DictReader(handle)}


def dino_regression(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reference = read_r6_dino()
    actual = {row["baseline"]: row for row in rows if row["model"] == "dino" and row["seed"] == 42}
    keys = ("one_step_grounding_cosine", "correct_endpoint_cosine", "reversed_order_endpoint_cosine", "grounded_order_margin_cosine", "order_ranking_accuracy", "copy_normalized_improvement")
    delta = {baseline: {key: abs(actual[baseline][key] - reference[baseline][key]) for key in keys} for baseline in reference}
    maximum = max(value for values in delta.values() for value in values.values())
    if maximum > 1e-7:
        raise RuntimeError(f"DINO r6 regression failed: max delta {maximum}")
    return {"reference": str(REPO_ROOT / "results/experimental/se3_protocol_v2_observed_dino_l0_seed42_r6"), "max_abs_delta": maximum, "passed": True}


def render_report(output: Path, main: list[dict[str, Any]], bootstrap: list[dict[str, Any]], coverage: list[dict[str, Any]], audit: dict[str, Any]) -> None:
    columns = ("model", "layer", "prefix", "one-step", "correct 2-step", "reversed 2-step", "order margin", "order accuracy", "grounding gain vs copy")
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in main:
        lines.append("| " + " | ".join((row["display_name"], str(row["layer"]), row["prefix"], *(f"{row[key]:.4f}" for key in ("one_step_grounding_cosine_mean", "correct_endpoint_cosine_mean", "reversed_order_endpoint_cosine_mean", "grounded_order_margin_cosine_mean", "order_ranking_accuracy_mean", "copy_normalized_improvement_mean")))) + " |")
    ci = {(row["model"], row["metric"]): row for row in bootstrap}
    ci_lines = ["| model | grounding gain 95% CI | order margin 95% CI | order accuracy 95% CI |", "|---|---:|---:|---:|"]
    for row in main:
        def interval(name: str) -> str:
            value = ci[(row["model"], name)]
            return f"[{value['ci95_low']:.4f}, {value['ci95_high']:.4f}]"
        ci_lines.append(f"| {row['display_name']} | {interval('grounding_gain_vs_copy')} | {interval('grounded_order_margin_cosine')} | {interval('order_ranking_accuracy')} |")
    cov_lines = ["| model | layer | prefix/context | raw tuples | common tuples | test tuples/scenes | seeds | lineage |", "|---|---:|---|---:|---:|---:|---:|---|"]
    for row in coverage:
        cov_lines.append(f"| {row['display_name']} | {row['layer']} | {row['prefix']} / {row['context_role']} | {row['raw_tuples']} | {row['common_tuples']} | {row['test_tuples']}/{row['test_scenes']} | {row['seeds']} | {row['lineage_status']} |")
    report = f"""# Preliminary cross-model observed-order pilot

Status: **exploratory / non-canonical**.  The only endpoint ground truth is
the real observed C2 `z_C`; no synthetic endpoint, renderer, order loss,
composition loss, cycle loss, rollout loss, or test supervision was used.

## Main comparison

{chr(10).join(lines)}

All entries are the full predictor's mean over the fixed seeds.  Lower cosine
distance is better; `margin = reversed - correct`; order accuracy is strict
correct-better-than-reversed ranking.  The exact per-seed/all-baseline output
is `all_baseline_seed_metrics.csv`.

## Scene bootstrap (full predictor)

{chr(10).join(ci_lines)}

Scene is the bootstrap unit; each resample averages fixed-seed runs.

## Coverage

{chr(10).join(cov_lines)}

Unavailable/excluded candidates are in `availability.json`.  The identical
common tuple list is `common_tuple_manifest.csv`.

## Hard-gate audit

```json
{json.dumps(audit, indent=2, sort_keys=True)}
```

## Scope limitations

- The common held-out set has only 44 tuples from 11 scenes; this is a group-meeting pilot, not a final statistical benchmark.
- Rare motion-family coverage and 3–4 step rollout are recorded limitations, not blockers for this frozen-intersection comparison.
- `p8` is fixed for all C artifacts; it is not a prefix sweep.  For DINO/F3R it only fixes tuple sampling, not a claimed temporal-context effect.
- No per-frame static-scene certificate is available for raw observed RGB.
"""
    (output / "REPORT.md").write_text(report)


def run_model_child(output: Path, spec: ModelSpec, seeds: tuple[int, ...], split_seed: int, max_epochs: int) -> None:
    """Train/evaluate one model in an isolated process and persist light data.

    C dumps can be large enough that retaining several torch-load allocations
    in a parent process is needlessly fragile.  The parent compares only these
    immutable tuple manifests and small metric arrays after every child exits.
    """
    records, inventory = load_single_records(spec)
    assignments, split_audit = split_scenes(records, split_seed, .7, .15)
    cfg = {"hidden_dim": 64, "learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": 64, "max_epochs": max_epochs, "patience": 30, "cosine_loss_weight": .25}
    all_rows: list[dict[str, Any]] = []
    training: dict[str, Any] = {}
    for seed in seeds:
        rows, arrays, meta = evaluate_seed(records, seed, cfg, assignments)
        all_rows.extend([{**row, "model": spec.key, "display_name": spec.display_name, "layer": spec.layer, "prefix": spec.prefix} for row in rows])
        training[str(seed)] = meta
        np.savez_compressed(
            output / f"bootstrap_arrays_seed{seed}.npz",
            copy_correct=arrays["copy"]["correct_cosine"],
            full_correct=arrays["full"]["correct_cosine"],
            full_margin=arrays["full"]["margin_cosine"],
            full_ranking=arrays["full"]["ranking"],
        )
    tuple_rows = [{"scene_id": record.scene_id, "sample_id": record.sample_id, "frame_A": record.raw_frame_indices[0], "frame_B": record.raw_frame_indices[1], "frame_C": record.raw_frame_indices[2], "split": assignments[record.scene_id]} for record in records]
    dump_json(output / "metrics.json", all_rows)
    dump_json(output / "inventory.json", inventory)
    dump_json(output / "scene_split.json", {"assignments": assignments, "audit": split_audit})
    dump_json(output / "training.json", training)
    write_csv(output / "tuple_manifest.csv", tuple_rows)
    dump_json(output / "provenance.json", {
        "manifest": {"path": inventory["manifest"], "sha256": sha256(Path(inventory["manifest"]))},
        "c1_dump": {"path": inventory["c1_dump"], "sha256": sha256(Path(inventory["c1_dump"]))},
        "c2_dump": {"path": inventory["c2_dump"], "sha256": sha256(Path(inventory["c2_dump"]))},
    })


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def manifest_keys(rows: list[dict[str, Any]]) -> list[tuple[str, str, int, int, int]]:
    return [(str(row["scene_id"]), str(row["sample_id"]), int(row["frame_A"]), int(row["frame_B"]), int(row["frame_C"])) for row in rows]


def run_children(output: Path, seeds: tuple[int, ...], split_seed: int, max_epochs: int) -> None:
    for spec in MODEL_SPECS:
        child = output / f"model_{spec.key}"
        command = [
            sys.executable, "-m", "analyses.se3_protocol_v2.cross_model_pilot",
            "--output-dir", str(child), "--seeds", ",".join(str(seed) for seed in seeds),
            "--seed", str(split_seed), "--max-epochs", str(max_epochs), "--model-child", spec.key,
        ]
        completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
        (output / f"{spec.key}_child.stdout.log").write_text(completed.stdout)
        (output / f"{spec.key}_child.stderr.log").write_text(completed.stderr)
        if completed.returncode:
            raise RuntimeError(f"{spec.key} child failed (exit {completed.returncode}); see {spec.key}_child.stderr.log")


def load_child_bootstrap_arrays(child: Path, seeds: tuple[int, ...]) -> list[dict[str, dict[str, np.ndarray]]]:
    loaded: list[dict[str, dict[str, np.ndarray]]] = []
    for seed in seeds:
        with np.load(child / f"bootstrap_arrays_seed{seed}.npz") as values:
            loaded.append({
                "copy": {"correct_cosine": values["copy_correct"].copy()},
                "full": {
                    "correct_cosine": values["full_correct"].copy(),
                    "margin_cosine": values["full_margin"].copy(),
                    "ranking": values["full_ranking"].copy(),
                },
            })
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--max-epochs", type=int, default=180)
    parser.add_argument("--seed", type=int, default=42, help="fixed scene-split/bootstrap seed; not the model-training seeds")
    parser.add_argument("--model-child", choices=[spec.key for spec in MODEL_SPECS], help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = resolve(args.output_dir)
    if output.exists():
        raise SystemExit(f"refusing to overwrite experimental output: {output}")
    output.mkdir(parents=True)
    seeds = parse_seeds(args.seeds)
    if args.model_child:
        torch.set_num_threads(4)
        run_model_child(output, next(spec for spec in MODEL_SPECS if spec.key == args.model_child), seeds, args.seed, args.max_epochs)
        print(json.dumps({"output": str(output), "model": args.model_child, "seeds": list(seeds)}, indent=2))
        return
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    cfg = {"hidden_dim": 64, "learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": 64, "max_epochs": args.max_epochs, "patience": 30, "cosine_loss_weight": .25}
    run_children(output, seeds, args.seed, args.max_epochs)
    child_rows = {spec.key: read_json(output / f"model_{spec.key}/metrics.json") for spec in MODEL_SPECS}
    child_manifests = {spec.key: list(csv.DictReader((output / f"model_{spec.key}/tuple_manifest.csv").open())) for spec in MODEL_SPECS}
    anchor_keys = manifest_keys(child_manifests["dino"])
    common = set(anchor_keys)
    for spec in MODEL_SPECS:
        keys = manifest_keys(child_manifests[spec.key])
        if keys != anchor_keys:
            raise RuntimeError(f"{spec.key}: model tuple ordering differs from DINO; common-intersection hard gate failed")
        common &= set(keys)
    if len(common) != len(anchor_keys):
        raise RuntimeError("model tuple intersection is not identical to the DINO anchor")
    split_payloads = {spec.key: read_json(output / f"model_{spec.key}/scene_split.json") for spec in MODEL_SPECS}
    assignments = split_payloads["dino"]["assignments"]
    split_audit = split_payloads["dino"]["audit"]
    if any(payload["assignments"] != assignments for payload in split_payloads.values()):
        raise RuntimeError("scene assignment differs across model children")
    all_rows = [row for spec in MODEL_SPECS for row in child_rows[spec.key]]
    inventory = {spec.key: read_json(output / f"model_{spec.key}/inventory.json") for spec in MODEL_SPECS}
    for spec in MODEL_SPECS:
        inventory[spec.key]["common_tuple_count"] = len(common)
        inventory[spec.key]["dropped_for_intersection"] = inventory[spec.key]["raw_tuple_count"] - len(common)
    inventory["intersection"] = {"tuple_count": len(common), "scene_count": len({key[0] for key in common}), "keys_identical_for_all_models": True, "anchor": "dino"}
    training = {spec.key: read_json(output / f"model_{spec.key}/training.json") for spec in MODEL_SPECS}
    regression = dino_regression(all_rows)
    aggregate_rows = aggregate(all_rows)
    main_rows = [row for row in aggregate_rows if row["baseline"] == "full"]
    action_rows = [row for row in aggregate_rows if row["baseline"] == "action_only"]
    bootstrap_rows = [
        row
        for spec in MODEL_SPECS
        for row in scene_bootstrap(
            spec.key,
            load_child_bootstrap_arrays(output / f"model_{spec.key}", seeds),
            [SimpleNamespace(scene_id=row["scene_id"]) for row in child_manifests[spec.key] if row["split"] == "test"],
            args.bootstrap_resamples,
            args.seed + 1000 + spec.layer,
        )
    ]
    coverage_rows = [{
        "model": spec.key,
        "display_name": spec.display_name,
        "layer": spec.layer,
        "prefix": spec.prefix,
        "context_role": spec.context_role,
        "raw_tuples": inventory[spec.key]["raw_tuple_count"],
        "common_tuples": inventory[spec.key]["common_tuple_count"],
        "test_tuples": split_audit["counts"]["test"]["tuples"],
        "test_scenes": split_audit["counts"]["test"]["scenes"],
        "seeds": len(seeds),
        "lineage_status": spec.lineage_status,
    } for spec in MODEL_SPECS]
    audit = {
        "pose_convention": POSE_CONVENTION,
        "common_intersection": inventory["intersection"],
        "scene_split": split_audit,
        "all_models_identical_test_tuple_order": True,
        "all_models_identical_test_tuple_count": split_audit["counts"]["test"]["tuples"],
        "scene_leakage": split_audit["scene_overlap"],
        "dino_r6_regression": regression,
        "training_contract": {"one_step_target_objective_only": True, **cfg, "training_seeds": list(seeds)},
        "limitations_not_blocking_exploratory_pilot": ["small_test_tuple_count", "rare_motion_family_coverage", "no_3_to_4_step_C2_rollout", "no_DINO_middle_layer", "SeeSE3_replication_incomplete", "no_renderer"],
        "static_scene_risk": "Observed ScanNet++ RGB has no per-frame dynamic/static certificate.",
    }
    dump_json(output / "config.json", {"argv": sys.argv, "model_specs": [spec.__dict__ for spec in MODEL_SPECS], "seeds": list(seeds), "bootstrap_resamples": args.bootstrap_resamples, **cfg})
    dump_json(output / "inventory_audit.json", inventory)
    dump_json(output / "audit.json", audit)
    dump_json(output / "availability.json", {"included": [spec.__dict__ for spec in MODEL_SPECS], "unavailable_or_excluded": list(UNAVAILABLE_OR_EXCLUDED)})
    dump_json(output / "training.json", training)
    dump_json(output / "dino_r6_regression.json", regression)
    write_csv(output / "all_baseline_seed_metrics.csv", all_rows)
    write_csv(output / "cross_model_main_table.csv", main_rows)
    write_csv(output / "cross_model_aggregate_baselines.csv", aggregate_rows)
    write_csv(output / "scene_bootstrap_ci.csv", bootstrap_rows)
    write_csv(output / "coverage.csv", coverage_rows)
    write_csv(output / "common_tuple_manifest.csv", child_manifests["dino"])
    provenance = {spec.key: read_json(output / f"model_{spec.key}/provenance.json") for spec in MODEL_SPECS}
    dump_json(output / "provenance.json", provenance)
    render_figures(output, main_rows, action_rows)
    render_report(output, main_rows, bootstrap_rows, coverage_rows, audit)
    print(json.dumps({"output": str(output), "models": [spec.key for spec in MODEL_SPECS], "common_tuples": len(common), "test_tuples": split_audit["counts"]["test"]["tuples"], "dino_regression": regression}, indent=2))


if __name__ == "__main__":
    main()
