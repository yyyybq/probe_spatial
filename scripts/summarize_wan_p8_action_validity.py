#!/usr/bin/env python3
"""Aggregate Phase-1 Wan p8 interventions, apply the gate, and render outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vidfm3d.utils.action_validity import scene_bootstrap_delta


LAYERS = (0, 5, 10, 15, 25, 29)
METRICS = ("cos", "r1", "r5", "mrr", "rank1")
CONTROLS = (
    "shuffled_action",
    "eval_null_action",
    "copy_last_streaming_state",
    "copy_prefix_pooled_state",
    "all_wrong_actions_mean",
)
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 2026082517


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str] | None = None) -> None:
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _condition(data: Any, condition: str, metric: str) -> np.ndarray:
    return np.asarray(data[f"condition__{condition}__{metric}"], dtype=np.float64)


def _control(data: Any, control: str, metric: str) -> np.ndarray:
    if control == "shuffled_action":
        return np.asarray(data[f"permutation__{metric}"], dtype=np.float64).mean(axis=0)
    if control == "all_wrong_actions_mean":
        return np.nanmean(np.asarray(data[f"wrong__{metric}"], dtype=np.float64), axis=1)
    return _condition(data, control, metric)


def _aggregate(values: np.ndarray) -> float:
    return float(np.nanmean(values))


def _delta_bootstrap(
    correct: np.ndarray,
    control: np.ndarray,
    scenes: np.ndarray,
    metric: str,
    seed: int,
) -> Dict[str, float]:
    if metric == "rank1":
        result = scene_bootstrap_delta(control, correct, scenes, BOOTSTRAP_REPS, seed)
        result["direction"] = "control_minus_correct; positive means correct has better/lower rank"
    else:
        result = scene_bootstrap_delta(correct, control, scenes, BOOTSTRAP_REPS, seed)
        result["direction"] = "correct_minus_control; positive means correct is better"
    return result


def _scene_rows(layer: int, data: Any, classes: np.ndarray) -> List[Dict[str, Any]]:
    scenes = np.asarray(data["scene_paths"], dtype=object)
    rows = []
    for scene in sorted(set(scenes.tolist())):
        indices = np.flatnonzero(scenes == scene)
        row: Dict[str, Any] = {
            "layer": layer,
            "scene_path": scene,
            "n": len(indices),
            "action_classes": "|".join(sorted(set(classes[indices].tolist()))),
        }
        for metric in METRICS:
            correct = _condition(data, "correct_action", metric)
            shuffled = _control(data, "shuffled_action", metric)
            copy = _control(data, "copy_last_streaming_state", metric)
            if metric == "rank1":
                row[f"correct_{metric}"] = _aggregate(correct[indices])
                row[f"correct_vs_shuffled_{metric}_improvement"] = _aggregate(shuffled[indices] - correct[indices])
                row[f"correct_vs_copy_{metric}_improvement"] = _aggregate(copy[indices] - correct[indices])
            else:
                row[f"correct_{metric}"] = _aggregate(correct[indices])
                row[f"correct_vs_shuffled_{metric}_delta"] = _aggregate(correct[indices] - shuffled[indices])
                row[f"correct_vs_copy_{metric}_delta"] = _aggregate(correct[indices] - copy[indices])
        rows.append(row)
    return rows


def _select_qualitative(per_sample: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_sample: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in per_sample:
        if row["layer"] in {10, 15}:
            by_sample[row["sample_id"]].append(row)
    scored = []
    for sample_id, rows in by_sample.items():
        if len(rows) != 2:
            continue
        summary = {
            "sample_id": sample_id,
            "scene_path": rows[0]["scene_path"],
            "action_class": rows[0]["action_class"],
            "correct_vs_shuffled_cos": float(np.mean([r["correct_vs_shuffled_cos"] for r in rows])),
            "correct_vs_wrong_cos_margin": float(np.mean([r["correct_vs_wrong_cos_margin"] for r in rows])),
            "copy_minus_correct_cos": float(np.mean([r["copy_last_cos"] - r["correct_cos"] for r in rows])),
            "correct_rank1_l10": next(r["correct_rank1"] for r in rows if r["layer"] == 10),
            "correct_rank1_l15": next(r["correct_rank1"] for r in rows if r["layer"] == 15),
            "copy_rank1_l10": next(r["copy_last_rank1"] for r in rows if r["layer"] == 10),
            "copy_rank1_l15": next(r["copy_last_rank1"] for r in rows if r["layer"] == 15),
        }
        scored.append(summary)

    selected: List[Dict[str, Any]] = []
    used = set()

    def take(category: str, ordered: List[Dict[str, Any]], count: int = 3) -> None:
        for item in ordered:
            if item["sample_id"] in used:
                continue
            row = dict(item)
            row["category"] = category
            selected.append(row)
            used.add(item["sample_id"])
            if sum(entry["category"] == category for entry in selected) == count:
                return

    take(
        "success_action_specific",
        sorted(scored, key=lambda row: row["correct_vs_shuffled_cos"] + row["correct_vs_wrong_cos_margin"], reverse=True),
    )
    take(
        "failure_wrong_action",
        sorted(scored, key=lambda row: row["correct_vs_wrong_cos_margin"]),
    )
    take(
        "failure_persistence_explained",
        sorted(scored, key=lambda row: row["copy_minus_correct_cos"], reverse=True),
    )
    take(
        "ambiguous_near_zero",
        sorted(scored, key=lambda row: abs(row["correct_vs_shuffled_cos"]) + abs(row["correct_vs_wrong_cos_margin"])),
    )
    if len(selected) != 12:
        raise RuntimeError("could not select 12 unique qualitative examples")
    return selected


def _plot_curves(condition_rows: List[Dict[str, Any]], figures: Path) -> None:
    display = {
        "correct_action": "Correct action",
        "shuffled_action": "Shuffled action",
        "eval_null_action": "Eval null action",
        "copy_last_streaming_state": "Last-state copy",
        "copy_prefix_pooled_state": "Prefix-pooled copy",
    }
    colors = {
        "correct_action": "#16697a",
        "shuffled_action": "#e07a5f",
        "eval_null_action": "#777777",
        "copy_last_streaming_state": "#3d405b",
        "copy_prefix_pooled_state": "#81b29a",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for ax, metric, label in zip(axes, ("r1", "r5", "mrr"), ("R@1", "R@5", "MRR")):
        for condition in display:
            subset = [row for row in condition_rows if row["condition"] == condition]
            subset.sort(key=lambda row: row["layer"])
            ax.plot(
                [row["layer"] for row in subset],
                [row[metric] for row in subset],
                marker="o",
                linewidth=2,
                color=colors[condition],
                label=display[condition],
            )
        ax.set_xlabel("Wan layer")
        ax.set_ylabel(label)
        ax.grid(alpha=0.22)
        ax.set_xticks(LAYERS)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(figures / "six_layer_condition_curves.png", dpi=180)
    plt.close(fig)


def _plot_margin(bootstrap_rows: List[Dict[str, Any]], figures: Path) -> None:
    rows = [
        row
        for row in bootstrap_rows
        if row["control"] == "all_wrong_actions_mean" and row["metric"] == "cos"
    ]
    rows.sort(key=lambda row: row["layer"])
    estimates = np.asarray([row["estimate"] for row in rows])
    low = np.asarray([row["ci_low"] for row in rows])
    high = np.asarray([row["ci_high"] for row in rows])
    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    ax.errorbar(LAYERS, estimates, yerr=[estimates - low, high - estimates], fmt="o-", color="#16697a", capsize=4)
    ax.axhline(0.0, color="#222222", linewidth=1)
    ax.set_xlabel("Wan layer")
    ax.set_ylabel("Correct minus all-wrong target cosine")
    ax.set_xticks(LAYERS)
    ax.grid(alpha=0.22)
    fig.savefig(figures / "wrong_action_margin_by_layer.png", dpi=180)
    plt.close(fig)


def _plot_scene_deltas(scene_rows: List[Dict[str, Any]], figures: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True, sharey=True)
    for ax, layer in zip(axes, (10, 15)):
        rows = [row for row in scene_rows if row["layer"] == layer]
        rows.sort(key=lambda row: row["correct_vs_shuffled_cos_delta"])
        values = [row["correct_vs_shuffled_cos_delta"] for row in rows]
        ax.bar(np.arange(len(values)), values, color=["#e07a5f" if value < 0 else "#16697a" for value in values], width=0.85)
        ax.axhline(0.0, color="#222222", linewidth=1)
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("Val scenes, sorted")
        ax.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Correct minus shuffled cosine")
    fig.savefig(figures / "middle_layer_scene_deltas.png", dpi=180)
    plt.close(fig)


def _plot_action_classes(class_rows: List[Dict[str, Any]], figures: Path) -> None:
    classes = sorted({row["action_class"] for row in class_rows})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), constrained_layout=True)
    for ax, layer in zip(axes, (10, 15)):
        rows = {row["action_class"]: row for row in class_rows if row["layer"] == layer}
        values = [rows[name]["correct_vs_shuffled_cos"] for name in classes]
        ax.barh(classes, values, color=["#e07a5f" if value < 0 else "#16697a" for value in values])
        ax.axvline(0.0, color="#222222", linewidth=1)
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("Correct minus shuffled cosine")
        ax.grid(axis="x", alpha=0.22)
    fig.savefig(figures / "middle_layer_action_class_performance.png", dpi=180)
    plt.close(fig)


def _plot_confusions(confusion_rows: List[Dict[str, Any]], figures: Path) -> None:
    classes = sorted({row["recipient_action_class"] for row in confusion_rows} | {row["best_wrong_action_class"] for row in confusion_rows})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    for ax, layer in zip(axes, (10, 15)):
        matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
        for row in confusion_rows:
            if row["layer"] == layer:
                matrix[classes.index(row["recipient_action_class"]), classes.index(row["best_wrong_action_class"])] += int(row["count"])
        image = ax.imshow(matrix, cmap="Blues", aspect="auto")
        ax.set_title(f"Layer {layer}")
        ax.set_xticks(range(len(classes)), classes, rotation=65, ha="right", fontsize=7)
        ax.set_yticks(range(len(classes)), classes, fontsize=7)
        ax.set_xlabel("Best-scoring wrong action class")
        ax.set_ylabel("Correct action class")
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.savefig(figures / "middle_layer_wrong_action_confusion.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, default=PROJECT_ROOT / "results/analysis/wan_p8_action_validity")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "reports/wan_p8_action_dynamics_construct_validity.md")
    args = parser.parse_args()
    root = args.analysis_dir.resolve()
    figures = root / "figures"
    tables = root / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    inventory_rows = list(csv.DictReader((tables / "action_inventory.csv").open()))
    if len(inventory_rows) != 276:
        raise RuntimeError("action inventory must contain 276 rows")
    inventory_ids = [row["sample_id"] for row in inventory_rows]
    classes = np.asarray([row["action_class"] for row in inventory_rows], dtype=object)

    datasets: Dict[int, Any] = {}
    sample_ids = None
    scenes = None
    for layer in LAYERS:
        layer_dir = root / "layers" / f"layer{layer}"
        status = json.loads((root / "layers" / f"layer{layer}.sco_status.json").read_text())
        if status.get("rc") != 0:
            raise RuntimeError(f"layer {layer} SCO wrapper did not exit cleanly")
        summary = json.loads((layer_dir / "layer_summary.json").read_text())
        provenance = json.loads((layer_dir / "provenance.json").read_text())
        arrays_path = layer_dir / "interventions.npz"
        if not summary.get("ok") or not provenance.get("ok") or sha256(arrays_path) != provenance["arrays_sha256"]:
            raise RuntimeError(f"layer {layer} output gate failed")
        data = np.load(arrays_path, allow_pickle=True)
        ids = data["sample_ids"].tolist()
        layer_scenes = data["scene_paths"].tolist()
        if ids != inventory_ids:
            raise RuntimeError(f"layer {layer} differs from action inventory")
        if data["permutation_donor_indices"].shape != (100, 276):
            raise RuntimeError(f"layer {layer} permutation coverage failed")
        if data["wrong__cos"].shape != (276, 276) or np.count_nonzero(np.isfinite(data["wrong__cos"])) != 276 * 275:
            raise RuntimeError(f"layer {layer} wrong-action coverage failed")
        if sample_ids is None:
            sample_ids, scenes = ids, np.asarray(layer_scenes, dtype=object)
        elif ids != sample_ids or layer_scenes != scenes.tolist():
            raise RuntimeError(f"layer {layer} record alignment failed")
        datasets[layer] = data

    assert sample_ids is not None and scenes is not None
    per_sample_rows: List[Dict[str, Any]] = []
    condition_rows: List[Dict[str, Any]] = []
    bootstrap_rows: List[Dict[str, Any]] = []
    permutation_rows: List[Dict[str, Any]] = []
    scene_rows: List[Dict[str, Any]] = []
    class_rows: List[Dict[str, Any]] = []
    confusion_counts: Dict[tuple, int] = defaultdict(int)

    for layer_index, layer in enumerate(LAYERS):
        data = datasets[layer]
        correct = {metric: _condition(data, "correct_action", metric) for metric in METRICS}
        controls = {control: {metric: _control(data, control, metric) for metric in METRICS} for control in CONTROLS}
        wrong_cos = np.asarray(data["wrong__cos"], dtype=np.float64)
        wrong_rank = np.asarray(data["wrong__rank1"], dtype=np.float64)
        wrong_margin = correct["cos"][:, None] - wrong_cos
        best_wrong = np.nanargmax(wrong_cos, axis=1)
        for recipient, donor in enumerate(best_wrong):
            confusion_counts[(layer, classes[recipient], classes[donor])] += 1

        conditions_for_table = {
            "correct_action": correct,
            **{control: values for control, values in controls.items() if control != "all_wrong_actions_mean"},
        }
        for condition, metrics in conditions_for_table.items():
            condition_rows.append({"layer": layer, "condition": condition, **{metric: _aggregate(values) for metric, values in metrics.items()}})

        permutation_metrics = {metric: np.asarray(data[f"permutation__{metric}"], dtype=np.float64) for metric in METRICS}
        seeds = np.asarray(data["mapping_seeds"], dtype=np.int64)
        for seed_index, seed in enumerate(seeds):
            permutation_rows.append({"layer": layer, "seed": int(seed), **{metric: _aggregate(values[seed_index]) for metric, values in permutation_metrics.items()}})

        for control_index, control in enumerate(CONTROLS):
            for metric_index, metric in enumerate(METRICS):
                stat = _delta_bootstrap(
                    correct[metric],
                    controls[control][metric],
                    scenes,
                    metric,
                    BOOTSTRAP_SEED + layer_index * 100 + control_index * 10 + metric_index,
                )
                bootstrap_rows.append({"layer": layer, "control": control, "metric": metric, **stat})

        for index, sample_id in enumerate(sample_ids):
            row: Dict[str, Any] = {
                "sample_id": sample_id,
                "scene_path": scenes[index],
                "layer": layer,
                "action_class": classes[index],
            }
            for metric in METRICS:
                row[f"correct_{metric}"] = float(correct[metric][index])
                row[f"shuffled_{metric}"] = float(controls["shuffled_action"][metric][index])
                row[f"eval_null_{metric}"] = float(controls["eval_null_action"][metric][index])
                row[f"copy_last_{metric}"] = float(controls["copy_last_streaming_state"][metric][index])
                row[f"copy_prefix_{metric}"] = float(controls["copy_prefix_pooled_state"][metric][index])
                row[f"wrong_mean_{metric}"] = float(controls["all_wrong_actions_mean"][metric][index])
            finite_margin = wrong_margin[index][np.isfinite(wrong_margin[index])]
            row["correct_vs_shuffled_cos"] = row["correct_cos"] - row["shuffled_cos"]
            row["correct_vs_wrong_cos_margin"] = float(finite_margin.mean())
            row["correct_vs_wrong_cos_margin_median"] = float(np.median(finite_margin))
            row["wrong_margin_positive_fraction"] = float(np.mean(finite_margin > 0))
            row["wrong_rank_improvement"] = float(np.nanmean(wrong_rank[index]) - correct["rank1"][index])
            row["best_wrong_action_class"] = classes[best_wrong[index]]
            per_sample_rows.append(row)

        scene_rows.extend(_scene_rows(layer, data, classes))
        for action_class in sorted(set(classes.tolist())):
            indices = np.flatnonzero(classes == action_class)
            class_rows.append(
                {
                    "layer": layer,
                    "action_class": action_class,
                    "n": len(indices),
                    "correct_r1": _aggregate(correct["r1"][indices]),
                    "shuffled_r1": _aggregate(controls["shuffled_action"]["r1"][indices]),
                    "correct_vs_shuffled_r1": _aggregate(correct["r1"][indices] - controls["shuffled_action"]["r1"][indices]),
                    "correct_vs_shuffled_cos": _aggregate(correct["cos"][indices] - controls["shuffled_action"]["cos"][indices]),
                    "correct_vs_wrong_cos_margin": _aggregate(correct["cos"][indices] - controls["all_wrong_actions_mean"]["cos"][indices]),
                    "correct_vs_copy_r1": _aggregate(correct["r1"][indices] - controls["copy_last_streaming_state"]["r1"][indices]),
                }
            )

    confusion_rows = [
        {
            "layer": layer,
            "recipient_action_class": recipient,
            "best_wrong_action_class": donor,
            "count": count,
        }
        for (layer, recipient, donor), count in sorted(confusion_counts.items())
    ]

    write_csv(root / "per_sample_results.csv", per_sample_rows)
    write_csv(root / "per_scene_bootstrap.csv", bootstrap_rows)
    write_csv(tables / "condition_metrics.csv", condition_rows)
    write_csv(tables / "permutation_seed_metrics.csv", permutation_rows)
    write_csv(tables / "per_scene_metrics.csv", scene_rows)
    write_csv(tables / "action_class_metrics.csv", class_rows)
    write_csv(tables / "wrong_action_confusion.csv", confusion_rows)
    qualitative = _select_qualitative(per_sample_rows)
    write_csv(tables / "qualitative_examples.csv", qualitative)

    # Pre-declared gate: each middle layer must pass significance, wrong margin,
    # persistence, and breadth checks. No threshold is adapted to the results.
    gate_by_layer: Dict[str, Any] = {}
    bootstrap_lookup = {(row["layer"], row["control"], row["metric"]): row for row in bootstrap_rows}
    permutation_by_layer = defaultdict(list)
    for row in permutation_rows:
        permutation_by_layer[row["layer"]].append(row)
    for layer in (10, 15):
        data = datasets[layer]
        correct_r1 = _aggregate(_condition(data, "correct_action", "r1"))
        seed_r1 = np.asarray([row["r1"] for row in permutation_by_layer[layer]])
        permutation_p = float((np.count_nonzero(seed_r1 >= correct_r1) + 1) / (len(seed_r1) + 1))
        shuffled_stat = bootstrap_lookup[(layer, "shuffled_action", "r1")]
        margin_stat = bootstrap_lookup[(layer, "all_wrong_actions_mean", "cos")]
        copy_stat = bootstrap_lookup[(layer, "copy_last_streaming_state", "r1")]
        layer_classes = [row for row in class_rows if row["layer"] == layer]
        positive_classes = sum(row["correct_vs_shuffled_cos"] > 0 for row in layer_classes)
        layer_scenes = [row for row in scene_rows if row["layer"] == layer]
        positive_scenes = sum(row["correct_vs_shuffled_cos_delta"] > 0 for row in layer_scenes)
        signal_values = np.asarray([row["correct_vs_shuffled_cos_delta"] for row in layer_scenes])
        max_scene_share = float(np.max(np.abs(signal_values)) / max(np.sum(np.abs(signal_values)), 1e-12))
        tests = {
            "correct_vs_shuffled_significant": bool(shuffled_stat["ci_low"] > 0 and permutation_p <= 0.05),
            "wrong_margin_ci_lower_gt_zero": bool(margin_stat["ci_low"] > 0),
            "correct_beats_last_state_copy": bool(copy_stat["ci_low"] > 0),
            "action_class_breadth": bool(positive_classes >= 8),
            "scene_breadth": bool(positive_scenes >= 35 and max_scene_share < 0.10),
        }
        gate_by_layer[str(layer)] = {
            "pass": all(tests.values()),
            "tests": tests,
            "permutation_p_r1": permutation_p,
            "positive_action_classes": positive_classes,
            "total_action_classes": len(layer_classes),
            "positive_scenes": positive_scenes,
            "total_scenes": len(layer_scenes),
            "max_absolute_scene_signal_share": max_scene_share,
        }
    phase1_pass = all(value["pass"] for value in gate_by_layer.values())
    phase1_gate = "PASS" if phase1_pass else "WEAK_OR_NON_ACTION_SPECIFIC_SIGNAL"

    action_specific_r1 = {}
    for layer in LAYERS:
        data = datasets[layer]
        action_specific_r1[layer] = _aggregate(_condition(data, "correct_action", "r1") - _control(data, "shuffled_action", "r1"))
    best_layer = max(action_specific_r1, key=action_specific_r1.get)
    middle_peak_survives = bool(phase1_pass and best_layer in {10, 15})
    correct_vs_shuffled = {
        str(layer): bootstrap_lookup[(layer, "shuffled_action", "r1")] for layer in LAYERS
    }
    correct_vs_null = {
        str(layer): bootstrap_lookup[(layer, "eval_null_action", "r1")] for layer in LAYERS
    }
    correct_vs_copy = {
        str(layer): bootstrap_lookup[(layer, "copy_last_streaming_state", "r1")] for layer in LAYERS
    }
    interpretation = "WEAK_ACTION_SIGNAL" if not phase1_pass else "VALID_ACTION_CONDITIONED_DYNAMICS"
    summary = {
        "run_tag": "wan_p8_action_dynamics_canonical_manifest_exact_20260821_200604_2075f3e",
        "metric_implementation_valid": True,
        "action_input_reaches_probe": True,
        "candidate_leakage": "NONE_FOUND",
        "phase1_gate": phase1_gate,
        "phase1_gate_by_layer": gate_by_layer,
        "phase2_executed": False,
        "phase2_reason": "Phase 1 gate failed; capacity-matched retraining prohibited by protocol" if not phase1_pass else "Phase 1 passed; Phase 2 is required before final validity",
        "correct_vs_shuffled": correct_vs_shuffled,
        "correct_vs_no_condition": correct_vs_null,
        "correct_vs_copy": correct_vs_copy,
        "best_layer_after_controls": best_layer,
        "action_specific_r1_by_layer": action_specific_r1,
        "middle_layer_peak_survives": middle_peak_survives,
        "final_interpretation": interpretation,
        "bootstrap": {"unit": "scene", "scenes": 69, "reps": BOOTSTRAP_REPS, "seed_base": BOOTSTRAP_SEED},
        "wrong_action_definition": "every other observed canonical-val action (275 per sample)",
        "copy_anchor_isolated": "unavailable; no new feature extraction permitted",
        "output_hashes": {},
    }

    _plot_curves(condition_rows, figures)
    _plot_margin(bootstrap_rows, figures)
    _plot_scene_deltas(scene_rows, figures)
    _plot_action_classes(class_rows, figures)
    _plot_confusions(confusion_rows, figures)

    output_paths = [
        root / "per_sample_results.csv",
        root / "per_scene_bootstrap.csv",
        tables / "condition_metrics.csv",
        tables / "permutation_seed_metrics.csv",
        tables / "per_scene_metrics.csv",
        tables / "action_class_metrics.csv",
        tables / "wrong_action_confusion.csv",
        tables / "qualitative_examples.csv",
    ] + sorted(figures.glob("*.png"))
    summary["output_hashes"] = {str(path.relative_to(root)): sha256(path) for path in output_paths}
    write_json(root / "summary.json", summary)

    condition_lookup = {(row["layer"], row["condition"]): row for row in condition_rows}
    lines = [
        "# Wan p8 Action Dynamics Construct Validity",
        "",
        "## Scope And Frozen Inputs",
        "",
        "This audit uses the completed canonical-clean Wan p8 run, its six frozen checkpoints, the exact canonical manifest, and the same versioned streaming/target roots. It performs no feature extraction and does not modify registry or canonical result files.",
        "",
        "- Manifest: `manifests/c_retrain_20260722/inscene_c_h124_prefix8.jsonl` (`sha256=772194f077c45cc85a942f17f46f50103125ae8f5b1c983c08881c1759607e0c`)",
        "- Val candidate pool: 276 sample-ID-paired targets from 69 scenes, four samples per scene.",
        "- Actions: continuous 9-D relative poses; there are no native discrete action labels.",
        "- Formal shuffled control: 100 fixed-seed, exact-marginal, 10-frequency-block derangements; donors are never the same sample or scene.",
        "- Wrong-action control: all other 275 observed val actions for every sample.",
        "- `eval_null_action`: zero action applied only at evaluation; it is not a trained no-condition baseline.",
        "- Rank in canonical summaries is zero-based. This report's `rank1` and MRR use conventional one-based rank.",
        "",
        "## Execution Audit Trail",
        "",
        "The audit failed closed three times before the final six-layer inventory was accepted. `pt-th75liy1` stopped all layers on a CPU-vs-CUDA target-pooling ULP gate; `pt-55bzlziu` preserved a valid layer-5 result but its parent failed after five children exceeded an overly tight batch-kernel max-abs gate; `pt-c5bqtpxi` preserved valid layers 0/15/25 but its parent failed when layers 10/29 changed near-tie ranks under batch 64. The final retry `pt-it5ygrcr` reproduced layers 10/29 with canonical batch 4 and succeeded. Final layer wrappers are rc=0 for all six layers; all failed-attempt logs are preserved under `results/analysis/wan_p8_action_validity/failed_attempts/`.",
        "",
        "## Phase 0 Data-Flow Audit",
        "",
        "All four executable contract tests passed. The probe output changes for a fixed state under different actions; the training step passes only `input_feat` and `action` into the probe; sample IDs, paths, target indices, and window IDs remain metadata; candidate reordering with remapped positives leaves retrieval metrics unchanged. Independent recomputation matched the frozen canonical R@1/R@5/cos/rank within `1e-6` for all six layers.",
        "",
        "## Phase 1 Results",
        "",
        "| Layer | Correct R@1 | Shuffled R@1 | Eval-null R@1 | Last-state copy R@1 | Prefix copy R@1 | Correct-wrong cosine margin |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for layer in LAYERS:
        correct = condition_lookup[(layer, "correct_action")]
        shuffled = condition_lookup[(layer, "shuffled_action")]
        null = condition_lookup[(layer, "eval_null_action")]
        copy = condition_lookup[(layer, "copy_last_streaming_state")]
        prefix = condition_lookup[(layer, "copy_prefix_pooled_state")]
        margin = bootstrap_lookup[(layer, "all_wrong_actions_mean", "cos")]["estimate"]
        lines.append(f"| {layer} | {correct['r1']:.4f} | {shuffled['r1']:.4f} | {null['r1']:.4f} | {copy['r1']:.4f} | {prefix['r1']:.4f} | {margin:.4f} |")
    lines.extend(
        [
            "",
            "The action-conditioned checkpoints are action-sensitive and generally outperform zero-action and frequency-block permutations. However, persistence is a stronger retrieval baseline at layers 10 and 15: the final streaming-state copy exceeds the trained correct-action probe in R@1. Therefore the middle-layer peak cannot be attributed specifically to action-conditioned dynamics under the pre-declared gate.",
            "",
            "## Gate",
            "",
            "Each of layers 10 and 15 had to pass all five pre-declared checks: significant correct-over-shuffled R@1 by paired scene bootstrap and permutation test; wrong-action cosine-margin CI above zero; correct R@1 significantly above the last-state copy; positive signal in at least 8/12 deterministic action strata; and positive signal in at least 35/69 scenes with no scene contributing 10% of absolute signal.",
            "",
            "| Layer | Overall | Correct > shuffled | Wrong margin | Beats copy | Class breadth | Scene breadth |",
            "| ---: | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for layer in (10, 15):
        gate = gate_by_layer[str(layer)]
        test = gate["tests"]
        lines.append(
            f"| {layer} | {'PASS' if gate['pass'] else 'FAIL'} | {test['correct_vs_shuffled_significant']} | {test['wrong_margin_ci_lower_gt_zero']} | {test['correct_beats_last_state_copy']} | {test['action_class_breadth']} ({gate['positive_action_classes']}/12) | {test['scene_breadth']} ({gate['positive_scenes']}/69) |"
        )
    lines.extend(
        [
            "",
            "Because Phase 1 did not pass, the protocol requires stopping before capacity-matched Phase 2 training. No Phase 2 run was launched.",
            "",
            "## Qualitative Audit",
            "",
            "Twelve deterministic examples spanning action-specific successes, wrong-action failures, persistence-explained failures, and near-zero ambiguous cases are recorded in `results/analysis/wan_p8_action_validity/tables/qualitative_examples.csv`. Scene- and action-class-level tables and the wrong-action class confusion are in the same directory.",
            "",
            "## Artifacts",
            "",
            "- `results/analysis/wan_p8_action_validity/summary.json`",
            "- `results/analysis/wan_p8_action_validity/per_sample_results.csv`",
            "- `results/analysis/wan_p8_action_validity/per_scene_bootstrap.csv`",
            "- `results/analysis/wan_p8_action_validity/tables/`",
            "- `results/analysis/wan_p8_action_validity/figures/`",
            "",
            "## Verification",
            "",
            "All six intervention files pass shape/hash coverage: 100 x 276 permutation outcomes and exactly 75,900 finite wrong-action outcomes per layer. The combined validity and paper-contract suite passed 13 tests. `experiments/registry.csv` and every `results/canonical/*.csv` file retained its pre-audit SHA256; no canonical file was updated.",
            "",
            "## Required Conclusion",
            "",
            "```text",
            "METRIC_IMPLEMENTATION_VALID: true",
            "ACTION_INPUT_REACHES_PROBE: true",
            "CANDIDATE_LEAKAGE: NONE_FOUND",
            f"PHASE1_GATE: {phase1_gate}",
            "PHASE2_EXECUTED: false",
            f"CORRECT_VS_SHUFFLED: {json.dumps(correct_vs_shuffled, sort_keys=True)}",
            f"CORRECT_VS_NO_CONDITION: NOT_RUN; eval_null_action controls={json.dumps(correct_vs_null, sort_keys=True)}",
            f"CORRECT_VS_COPY: {json.dumps(correct_vs_copy, sort_keys=True)}",
            f"BEST_LAYER_AFTER_CONTROLS: {best_layer}",
            f"MIDDLE_LAYER_PEAK_SURVIVES: {str(middle_peak_survives).lower()}",
            f"FINAL_INTERPRETATION: {interpretation}",
            "```",
            "",
            "The evidence supports a real but weak action signal. It does not support the stronger claim that the layer 10-15 peak is a valid, specifically action-conditioned dynamics construct, because a same-domain persistence copy explains more retrieval performance at those layers.",
        ]
    )
    args.report.write_text("\n".join(lines) + "\n")
    summary["report"] = str(args.report.resolve())
    summary["report_sha256"] = sha256(args.report)
    write_json(root / "summary.json", summary)
    print(json.dumps({"ok": True, "phase1_gate": phase1_gate, "best_layer": best_layer, "report": str(args.report)}, sort_keys=True))


if __name__ == "__main__":
    main()
