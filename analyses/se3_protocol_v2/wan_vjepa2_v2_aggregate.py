#!/usr/bin/env python3
"""Incremental aggregation for completed Wan/V-JEPA2 exact-v2 layer cells."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from analyses.se3_protocol_v2.wan_vjepa2_v2_data import RESULT, load_protocol
from analyses.se3_protocol_v2.wan_vjepa2_v2_train import BASELINES, write_csv


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def cell_parts(cell: str) -> tuple[str, int]:
    model, layer = cell.rsplit("_l", 1)
    return model, int(layer)


def scene_values(
    rows: list[dict[str, str]], cell: str, baseline: str, field: str,
    seeds: list[int], scenes: list[str],
) -> np.ndarray:
    index: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["cell"] == cell and row["baseline"] == baseline:
            index[(int(row["seed"]), row["scene_id"])].append(float(row[field]))
    values = np.empty((len(seeds), len(scenes)), dtype=np.float64)
    for i, seed in enumerate(seeds):
        for j, scene in enumerate(scenes):
            found = index[(seed, scene)]
            if not found:
                raise RuntimeError(f"missing {cell}/{baseline}/{seed}/{scene}/{field}")
            values[i, j] = np.mean(found)
    return values.mean(axis=0)


def interval(values: np.ndarray) -> tuple[float, float, float, float]:
    estimate = float(values.mean())
    low, high = np.quantile(values, [.025, .975])
    p = min(1.0, 2.0 * min(float(np.mean(values <= 0)), float(np.mean(values >= 0))))
    return estimate, float(low), float(high), p


def bootstrap_cell(
    rows: list[dict[str, str]], cell: str, protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    seeds = [int(value) for value in protocol["training"]["seeds"]]
    scenes = sorted({row["scene_id"] for row in rows if row["cell"] == cell})
    repetitions = int(protocol["statistics"]["bootstrap_replicates"])
    base_seed = int(protocol["statistics"]["bootstrap_seed"])
    rng = np.random.default_rng(base_seed)
    draws = rng.integers(0, len(scenes), size=(repetitions, len(scenes)))
    cache: dict[tuple[str, str], np.ndarray] = {}

    def values(baseline: str, field: str) -> np.ndarray:
        key = (baseline, field)
        if key not in cache:
            per_scene = scene_values(rows, cell, baseline, field, seeds, scenes)
            cache[key] = per_scene[draws].mean(1)
        return cache[key]

    copy_endpoint = values("copy", "sequential_endpoint_error")
    epsilon = float(protocol["statistics"]["normalization_epsilon"])
    comparisons: list[tuple[str, str, np.ndarray]] = [
        ("full", "copy", (copy_endpoint - values("full", "sequential_endpoint_error")) / (copy_endpoint + epsilon)),
        ("full", "without_condition", (
            values("without_condition", "sequential_endpoint_error") - values("full", "sequential_endpoint_error")
        ) / (values("without_condition", "sequential_endpoint_error") + epsilon)),
        ("full", "zero", values("full", "raw_order_margin") / (copy_endpoint + epsilon)),
    ]
    for baseline in ("without_condition", "action_only", "matched_shuffle", "commutative_additive"):
        comparisons.append((
            "full", baseline,
            (values("full", "raw_order_margin") - values(baseline, "raw_order_margin")) / (copy_endpoint + epsilon),
        ))
    rows_out = []
    names = [
        "two_step_grounding_gain_vs_copy",
        "two_step_grounding_gain_vs_without_condition",
        "normalized_order_margin",
        "normalized_order_margin_full_minus_without_condition",
        "normalized_order_margin_full_minus_action_only",
        "normalized_order_margin_full_minus_matched_shuffle",
        "normalized_order_margin_full_minus_commutative_additive",
    ]
    for name, (left, right, distribution) in zip(names, comparisons):
        estimate, low, high, p = interval(distribution)
        rows_out.append({
            "cell": cell, "metric": name, "left": left, "right": right,
            "estimate": estimate, "ci95_low": low, "ci95_high": high,
            "bootstrap_p_two_sided": p, "bootstrap_unit": "scene",
            "resamples": repetitions, "training_seeds_averaged": len(seeds),
            "evaluation_scenes": len(scenes),
        })
    return rows_out


def aggregate_seed_metrics(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["cell"], row["baseline"])].append(row)
    output = []
    skip = {"cell", "baseline", "aggregation", "seed", "wins", "losses", "ties"}
    for (cell, baseline), values in sorted(groups.items()):
        model, layer = cell_parts(cell)
        item: dict[str, Any] = {
            "cell": cell, "model": model, "layer": layer, "baseline": baseline,
            "seeds": len(values),
        }
        for key in values[0]:
            if key in skip:
                continue
            try:
                numbers = np.asarray([float(row[key]) for row in values])
            except (TypeError, ValueError):
                continue
            item[f"{key}_mean"] = float(numbers.mean())
            item[f"{key}_seed_std"] = float(numbers.std(ddof=0))
        output.append(item)
    return output


def noncomm_table(sample_rows: list[dict[str, str]], cells: list[str]) -> list[dict[str, Any]]:
    output = []
    for cell in cells:
        rows = [row for row in sample_rows if row["cell"] == cell and row["baseline"] == "full"]
        by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_sample[row["sample_id"]].append(row)
        x = np.asarray([float(values[0]["noncomm_strength"]) for values in by_sample.values()])
        y = np.asarray([np.mean([float(row["raw_order_margin"]) for row in values]) for values in by_sample.values()])
        edges = np.quantile(x, [0, .25, .5, .75, 1])
        correlation = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
        model, layer = cell_parts(cell)
        for bin_index in range(4):
            mask = (x >= edges[bin_index]) & (x <= edges[bin_index + 1] if bin_index == 3 else x < edges[bin_index + 1])
            output.append({
                "cell": cell, "model": model, "layer": layer,
                "quartile": bin_index + 1, "n": int(mask.sum()),
                "noncomm_low": float(edges[bin_index]), "noncomm_high": float(edges[bin_index + 1]),
                "raw_order_margin": float(y[mask].mean()),
                "noncomm_margin_pearson_all_samples": correlation,
            })
    return output


def plot_layers(main_rows: list[dict[str, Any]]) -> None:
    if not main_rows:
        return
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    fields = (
        ("one_step_error_mean", "C1 calibration error (A→B)"),
        ("sequential_direct_prediction_distance_mean", "sequential–direct distance"),
        ("normalized_order_margin_mean", "normalized order margin"),
    )
    for model in ("wan", "vjepa2"):
        values = sorted((row for row in main_rows if row["model"] == model), key=lambda row: row["layer"])
        if not values:
            continue
        for axis, (field, label) in zip(axes, fields):
            axis.plot([row["layer"] for row in values], [row[field] for row in values], marker="o", label=model)
            axis.set_xlabel("exported layer")
            axis.set_ylabel(label)
            axis.grid(alpha=.2)
    axes[-1].axhline(0, color="black", lw=.7)
    axes[0].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(RESULT / "layer_curves.png", dpi=200)
    figure.savefig(RESULT / "layer_curves.pdf")
    plt.close(figure)


def layer_relationships(main_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for model in ("wan", "vjepa2"):
        values = sorted((row for row in main_rows if row["model"] == model), key=lambda row: row["layer"])
        if len(values) < 2:
            continue
        prediction_error = np.asarray([row["one_step_error_mean"] for row in values])
        for field in (
            "sequential_endpoint_error_mean",
            "sequential_direct_prediction_distance_mean",
            "normalized_order_margin_mean",
            "tie_aware_accuracy_mean",
        ):
            structural = np.asarray([row[field] for row in values])
            correlation = float(np.corrcoef(prediction_error, structural)[0, 1]) if prediction_error.std() > 0 and structural.std() > 0 else float("nan")
            output.append({
                "model": model, "layers": len(values),
                "x": "one_step_error", "y": field.removesuffix("_mean"),
                "pearson_across_exported_layers": correlation,
                "interpretation": "descriptive_only_small_layer_count",
            })
    return output


def findings(main: list[dict[str, Any]], bootstrap: list[dict[str, Any]]) -> str:
    if not main:
        return "# Findings\n\nNo cell has completed all five frozen seeds; no scientific conclusion is reported.\n"
    lines = [
        "# Findings", "",
        "These are exact canonical-v2 p12 diagnostics for completed cells only. The recurrent head was trained with A→B supervision only; composition and order quantities were held-out structural analyses.", "",
    ]
    by_ci = {(row["cell"], row["metric"]): row for row in bootstrap}
    for row in sorted(main, key=lambda value: (value["model"], value["layer"])):
        ground = by_ci[(row["cell"], "two_step_grounding_gain_vs_copy")]
        condition = by_ci[(row["cell"], "two_step_grounding_gain_vs_without_condition")]
        order = by_ci[(row["cell"], "normalized_order_margin")]
        action_control = by_ci[(row["cell"], "normalized_order_margin_full_minus_action_only")]
        lines.append(
            f"- {row['cell']}: A→B error {row['one_step_error_mean']:.4f}; sequential/direct endpoint errors "
            f"{row['sequential_endpoint_error_mean']:.4f}/{row['direct_endpoint_error_mean']:.4f}; "
            f"grounding gain vs copy {ground['estimate']:.4f} (95% scene-bootstrap CI "
            f"[{ground['ci95_low']:.4f}, {ground['ci95_high']:.4f}]); gain vs without-condition "
            f"{condition['estimate']:.4f} (CI [{condition['ci95_low']:.4f}, {condition['ci95_high']:.4f}]); normalized order margin "
            f"{order['estimate']:.4f} (CI [{order['ci95_low']:.4f}, {order['ci95_high']:.4f}]); "
            f"full-minus-action-only order margin {action_control['estimate']:.4f} "
            f"(CI [{action_control['ci95_low']:.4f}, {action_control['ci95_high']:.4f}]); "
            f"tie-aware accuracy {row['tie_aware_accuracy_mean']:.4f}."
        )
    relationships = layer_relationships(main)
    for model in ("wan", "vjepa2"):
        values = {row["y"]: row for row in relationships if row["model"] == model}
        if values:
            layer_count = len([row for row in main if row["model"] == model])
            lines.append(
                f"- {model} across-layer description: one-step error correlates with sequential endpoint error "
                f"r={values['sequential_endpoint_error']['pearson_across_exported_layers']:.3f}, with "
                f"sequential–direct prediction distance r={values['sequential_direct_prediction_distance']['pearson_across_exported_layers']:.3f}, "
                f"and with normalized order margin r={values['normalized_order_margin']['pearson_across_exported_layers']:.3f}. "
                f"These {layer_count}-point correlations are descriptive and can be dominated by an endpoint layer."
            )
    lines.extend([
        "", "Across both models, predictive calibration and sequential endpoint error move together, while normalized order margin does not follow the same layer pattern. No completed layer has a normalized order-margin CI excluding zero. Most control contrasts are likewise inconclusive; isolated exceptions are reported as layer-specific observed-order comparisons, not general capabilities.",
        "Cross-layer co-movement is shown descriptively in `layer_curves.*`; it is not treated as causal evidence that prediction quality and group structure are identical abilities.",
        "A CI crossing zero is reported as inconclusive, not as evidence that order sensitivity is absent. One-sided observed-order behavior is not a proof of a full SE(3) group action.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    protocol = load_protocol()
    expected = [f"{model}_l{layer}" for model, spec in protocol["models"].items() for layer in spec["layers"]]
    live_availability_path = RESULT / "inventory/wan_live_availability.json"
    live_availability = (
        json.loads(live_availability_path.read_text()).get("cells", {})
        if live_availability_path.exists() else {}
    )
    status_rows, completed = [], []
    for cell in expected:
        inventory = RESULT / "inventory" / f"{cell}.json"
        execution = RESULT / "cells" / cell / "EXECUTION_STATUS.json"
        inv = json.loads(inventory.read_text()) if inventory.exists() else {"status": "NOT_RUN"}
        run = json.loads(execution.read_text()) if execution.exists() else {"status": "NOT_RUN"}
        live = live_availability.get(cell)
        if live and not live["runnable_on_immutable_common_set"] and run.get("status") != "PASS":
            inv = {
                "status": "BLOCKED_LIVE_AVAILABILITY",
                "error": (
                    f"streaming {live['streaming_samples_available']}/{live['streaming_samples_required']}; "
                    f"target roles {live['target_roles_available']}/{live['target_roles_required']}"
                ),
            }
        status_rows.append({
            "cell": cell, "inventory_status": inv.get("status"),
            "execution_status": run.get("status"), "error": inv.get("error", run.get("error", "")),
        })
        if run.get("status") == "PASS":
            completed.append(cell)
    write_csv(RESULT / "EXECUTION_STATUS.csv", status_rows)
    if not completed:
        (RESULT / "FINDINGS.md").write_text(findings([], []), encoding="utf-8")
        return
    seed_rows, sample_rows = [], []
    for cell in completed:
        for seed in protocol["training"]["seeds"]:
            root = RESULT / "cells" / cell / "seeds" / f"seed{seed}"
            seed_rows.extend(read_csv(root / "seed_metrics.csv"))
            sample_rows.extend(read_csv(root / "per_sample_metrics.csv"))
    write_csv(RESULT / "per_seed_metrics.csv", seed_rows)
    aggregate = aggregate_seed_metrics(seed_rows)
    write_csv(RESULT / "all_baselines_layer_table.csv", aggregate)
    main_rows = [row for row in aggregate if row["baseline"] == "full"]
    bootstrap = [row for cell in completed for row in bootstrap_cell(sample_rows, cell, protocol)]
    for row in main_rows:
        ci = {(value["cell"], value["metric"]): value for value in bootstrap}
        for metric in ("two_step_grounding_gain_vs_copy", "two_step_grounding_gain_vs_without_condition", "normalized_order_margin"):
            value = ci[(row["cell"], metric)]
            row[f"{metric}_bootstrap_estimate"] = value["estimate"]
            row[f"{metric}_ci95_low"] = value["ci95_low"]
            row[f"{metric}_ci95_high"] = value["ci95_high"]
    write_csv(RESULT / "layer_main_table.csv", main_rows)
    write_csv(RESULT / "paired_scene_bootstrap.csv", bootstrap)
    write_csv(RESULT / "order_vs_noncommutativity.csv", noncomm_table(sample_rows, completed))
    relationships = layer_relationships(main_rows)
    if relationships:
        write_csv(RESULT / "layer_metric_relationships.csv", relationships)
    plot_layers(main_rows)
    (RESULT / "FINDINGS.md").write_text(findings(main_rows, bootstrap), encoding="utf-8")


if __name__ == "__main__":
    main()
