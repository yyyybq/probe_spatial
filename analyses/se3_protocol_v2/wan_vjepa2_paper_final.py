#!/usr/bin/env python3
"""Build the frozen paper package from existing Wan/V-JEPA2 results only.

The program is intentionally read-only with respect to the two source result
directories.  It performs no optimization, inference, or feature extraction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from safetensors.torch import load_file


REPO = Path(__file__).resolve().parents[2]
LAYERS = REPO / "results/experimental/se3_wan_vjepa2_exact_v2_layers_20260916_r1"
P1 = REPO / "results/experimental/se3_wan_vjepa2_exact_v2_p1_20260917_r1"
LEGACY = REPO / "results/experimental/se3_observed_paper_v1_20260913_r2"
DEFAULT_OUTPUT = REPO / "results/experimental/se3_wan_vjepa2_paper_final_20260917_r1"
CELLS = (
    "wan_l0", "wan_l5", "wan_l10", "wan_l15", "wan_l20", "wan_l25", "wan_l29",
    "vjepa2_l0", "vjepa2_l5", "vjepa2_l11", "vjepa2_l17", "vjepa2_l23",
)
SEEDS = (42, 43, 44, 45, 46)
BASELINES = ("copy", "without_condition", "action_only", "matched_shuffle", "commutative_additive", "full")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def f(value: Any) -> float:
    return float(value)


def model_label(model: str) -> str:
    return "Wan" if model == "wan" else "V-JEPA2"


def cosine_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left.astype(np.float32); right = right.astype(np.float32)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.maximum(0.0, 1.0 - np.sum(left * right, axis=1) / np.maximum(denominator, 1e-12))


def scene_seed_matrix(
    rows: list[dict[str, Any]], cell: str, baseline: str, field: str, scenes: list[str],
) -> np.ndarray:
    index: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["cell"] == cell and row["baseline"] == baseline:
            index[(int(row["seed"]), row["scene_id"])].append(f(row[field]))
    return np.asarray([
        [np.mean(index[(seed, scene)]) for scene in scenes]
        for seed in SEEDS
    ])


def data_overlap_audit() -> dict[str, Any]:
    current = read_jsonl(LAYERS / "data/common_manifest.jsonl")
    legacy = read_jsonl(LEGACY / "se3_observed_v1_manifest.jsonl")
    legacy_scenes = {row["scene_id"] for row in legacy}
    split_rows = {split: [row for row in current if row["analysis_split"] == split] for split in ("train", "validation", "evaluation")}
    overlap = {}
    for split, rows in split_rows.items():
        scenes = {row["scene_id"] for row in rows}
        overlap[split] = {
            "samples": len(rows), "scenes": len(scenes),
            "scenes_overlapping_legacy_pilot": len(scenes & legacy_scenes),
            "scenes_not_in_legacy_pilot": len(scenes - legacy_scenes),
        }
    legacy_tuples = {(row["scene_id"], tuple(row["raw_frame_ids"])) for row in legacy}
    current_eval_tuples = {
        (row["scene_id"], tuple(row["abc_frame_ids"][key] for key in ("A", "B", "C")))
        for row in split_rows["evaluation"]
    }
    return {
        "legacy_pilot": {"samples": len(legacy), "scenes": len(legacy_scenes)},
        "exact_v2": overlap,
        "evaluation_scene_set_equals_legacy_scene_set": {row["scene_id"] for row in split_rows["evaluation"]} == legacy_scenes,
        "exact_physical_tuple_overlap_evaluation_vs_legacy": len(current_eval_tuples & legacy_tuples),
        "interpretation": "exact-v2 evaluation uses new frame triples in all 69 legacy-pilot scenes; it is not an independent-scene confirmation set",
    }


def load_parent_sample_rows() -> list[dict[str, Any]]:
    output = []
    for cell in CELLS:
        for seed in SEEDS:
            for row in read_csv(LAYERS / "cells" / cell / "seeds" / f"seed{seed}" / "per_sample_metrics.csv"):
                converted: dict[str, Any] = dict(row)
                converted["seed"] = int(row["seed"])
                for field in (
                    "one_step_error", "sequential_endpoint_error", "direct_endpoint_error",
                    "sequential_direct_prediction_distance", "recursive_minus_direct_endpoint_error",
                    "swapped_endpoint_error", "raw_order_margin", "tie_score",
                ):
                    converted[field] = f(row[field])
                output.append(converted)
    return output


def raw_prediction_audit(manifest: list[dict[str, Any]]) -> dict[str, Any]:
    split = np.asarray([row["analysis_split"] for row in manifest])
    evaluation = np.flatnonzero(split == "evaluation")
    maximum = 0.0; worst: dict[str, Any] = {}
    checked = 0
    for cell in CELLS:
        targets = {key: value.numpy() for key, value in load_file(str(LAYERS / "cache" / f"{cell}.sft"), device="cpu").items()}
        target_b, target_c = targets["z_b"][evaluation], targets["z_c"][evaluation]
        for seed in SEEDS:
            seed_dir = LAYERS / "cells" / cell / "seeds" / f"seed{seed}"
            archive = np.load(seed_dir / "raw_predictions.npz")
            metric_rows = read_csv(seed_dir / "per_sample_metrics.csv")
            by_baseline = {baseline: [row for row in metric_rows if row["baseline"] == baseline] for baseline in BASELINES}
            expected_ids = [row["sample_id"] for row in by_baseline["full"]]
            if list(archive["sample_id"].astype(str)) != expected_ids:
                raise RuntimeError(f"raw prediction ordering mismatch: {cell}/seed{seed}")
            for baseline in BASELINES:
                stored = by_baseline[baseline]
                predictions = {name: archive[f"{baseline}__{name}"] for name in ("one_ab", "sequential", "direct", "swapped")}
                recomputed = {
                    "one_step_error": cosine_distance(predictions["one_ab"], target_b),
                    "sequential_endpoint_error": cosine_distance(predictions["sequential"], target_c),
                    "direct_endpoint_error": cosine_distance(predictions["direct"], target_c),
                    "sequential_direct_prediction_distance": cosine_distance(predictions["sequential"], predictions["direct"]),
                    "swapped_endpoint_error": cosine_distance(predictions["swapped"], target_c),
                }
                recomputed["recursive_minus_direct_endpoint_error"] = recomputed["sequential_endpoint_error"] - recomputed["direct_endpoint_error"]
                recomputed["raw_order_margin"] = recomputed["swapped_endpoint_error"] - recomputed["sequential_endpoint_error"]
                for field, values in recomputed.items():
                    source = np.asarray([f(row[field]) for row in stored])
                    deviation = float(np.max(np.abs(values - source)))
                    checked += len(values)
                    if deviation > maximum:
                        maximum = deviation
                        worst = {"cell": cell, "seed": seed, "baseline": baseline, "metric": field, "max_abs_deviation": deviation}
    tolerance = 5e-4
    if maximum > tolerance:
        raise RuntimeError(f"raw-prediction audit failed: {worst}, tolerance={tolerance}")
    return {
        "status": "PASS_FLOAT16_ARCHIVE_TOLERANCE",
        "comparisons": checked,
        "tolerance": tolerance,
        "global_max_abs_deviation": maximum,
        "worst_case": worst,
        "note": "metrics were computed before float16 archival; the audit recomputes from archived float16 predictions",
    }


def aggregate_audit(parent_rows: list[dict[str, Any]]) -> dict[str, Any]:
    scenes = sorted({row["scene_id"] for row in parent_rows})
    joint = {row["cell"]: row for row in read_csv(P1 / "layer_joint_main_table.csv")}
    composition = {(row["cell"], row["baseline"]): row for row in read_csv(P1 / "composition/layer_baseline_summary.csv")}
    maximum = 0.0; worst = {}
    for cell in CELLS:
        copy = scene_seed_matrix(parent_rows, cell, "copy", "sequential_endpoint_error", scenes)
        without = scene_seed_matrix(parent_rows, cell, "without_condition", "sequential_endpoint_error", scenes)
        sequential = scene_seed_matrix(parent_rows, cell, "full", "sequential_endpoint_error", scenes)
        direct = scene_seed_matrix(parent_rows, cell, "full", "direct_endpoint_error", scenes)
        distance = scene_seed_matrix(parent_rows, cell, "full", "sequential_direct_prediction_distance", scenes)
        epsilon = 1e-8
        calculated = {
            "sequential_endpoint_error": float(sequential.mean()),
            "direct_endpoint_error": float(direct.mean()),
            "sequential_gain_vs_copy": float((copy.mean() - sequential.mean()) / (copy.mean() + epsilon)),
            "sequential_gain_vs_without_condition": float((without.mean() - sequential.mean()) / (without.mean() + epsilon)),
            "normalized_sequential_direct_distance": float(distance.mean() / (copy.mean() + epsilon)),
            "recursive_minus_direct_endpoint_gap": float((sequential - direct).mean()),
        }
        for field, value in calculated.items():
            observed = f(composition[(cell, "full")][field])
            deviation = abs(value - observed)
            if deviation > maximum:
                maximum = deviation; worst = {"cell": cell, "metric": field, "calculated": value, "table": observed}
        one = scene_seed_matrix(parent_rows, cell, "full", "one_step_error", scenes)
        deviation = abs(float(one.mean()) - f(joint[cell]["c1_one_step_error"]))
        if deviation > maximum:
            maximum = deviation; worst = {"cell": cell, "metric": "diagnostic_head_one_step_error", "calculated": float(one.mean()), "table": f(joint[cell]["c1_one_step_error"])}
    if maximum > 1e-12:
        raise RuntimeError(f"aggregate audit failed: {worst}")
    return {"status": "PASS", "global_max_abs_deviation": maximum, "worst_case": worst, "scene_macro_scenes": len(scenes), "seeds_averaged": len(SEEDS)}


def display_ci(low: Any, high: Any, digits: int = 4) -> str:
    return f"[{f(low):.{digits}f}, {f(high):.{digits}f}]"


def build_main_table() -> list[dict[str, Any]]:
    joint = read_csv(P1 / "layer_joint_main_table.csv")
    composition = {(row["cell"], row["baseline"]): row for row in read_csv(P1 / "composition/layer_baseline_summary.csv")}
    composition_ci = {(row["cell"], row["baseline"], row["metric"]): row for row in read_csv(P1 / "composition/paired_scene_bootstrap.csv")}
    parent_ci = {(row["cell"], row["metric"]): row for row in read_csv(LAYERS / "paired_scene_bootstrap.csv")}
    output = []
    model_order = {"wan": 0, "vjepa2": 1}
    for row in sorted(joint, key=lambda value: (model_order[value["model"]], int(value["layer"]))):
        cell = row["cell"]; full = composition[(cell, "full")]
        copy_ci = composition_ci[(cell, "full", "sequential_gain_vs_copy")]
        no_ci = composition_ci[(cell, "full", "sequential_gain_vs_without_condition")]
        order_ci = parent_ci[(cell, "normalized_order_margin")]
        output.append({
            "model": model_label(row["model"]), "layer": int(row["layer"]), "cell": cell,
            "diagnostic_head_one_step_error": f(row["c1_one_step_error"]),
            "sequential_endpoint_error": f(full["sequential_endpoint_error"]),
            "direct_endpoint_error": f(full["direct_endpoint_error"]),
            "gain_vs_copy": f(full["sequential_gain_vs_copy"]),
            "gain_vs_copy_ci95_low": f(copy_ci["ci95_low"]), "gain_vs_copy_ci95_high": f(copy_ci["ci95_high"]),
            "gain_vs_without_action": f(full["sequential_gain_vs_without_condition"]),
            "gain_vs_without_action_ci95_low": f(no_ci["ci95_low"]), "gain_vs_without_action_ci95_high": f(no_ci["ci95_high"]),
            "normalized_sequential_direct_distance": f(full["normalized_sequential_direct_distance"]),
            "normalized_order_margin": f(row["normalized_order_margin"]),
            "normalized_order_margin_ci95_low": f(order_ci["ci95_low"]), "normalized_order_margin_ci95_high": f(order_ci["ci95_high"]),
            "tie_aware_accuracy": f(row["tie_aware_accuracy"]),
        })
    return output


def build_representative_table() -> list[dict[str, Any]]:
    rows = read_csv(P1 / "representative/paired_model_comparison.csv")
    output = []
    for row in rows:
        raw_holm = f(row["holm_adjusted_p"])
        # The implementation uses 2*min(tail fractions), B=2000, then Holm
        # over four registered tests. Zero means 0/2000 opposing-tail draws.
        display = "<0.004" if raw_holm == 0 else f"{raw_holm:.3f}"
        output.append({
            "metric": row["metric"], "direction": "Wan L15 minus V-JEPA2 L11",
            "estimate": f(row["estimate"]), "ci95_low": f(row["ci95_low"]), "ci95_high": f(row["ci95_high"]),
            "bootstrap_p_two_sided_raw": f(row["bootstrap_p_two_sided"]),
            "holm_adjusted_p_raw": raw_holm, "holm_adjusted_p_paper_display": display,
            "opposing_tail_resamples_if_raw_zero": 0 if raw_holm == 0 else "not_applicable",
        })
    return output


def build_noncomm_table() -> list[dict[str, Any]]:
    metrics = read_csv(P1 / "order/stratified_metrics.csv")
    differences = read_csv(P1 / "order/paired_control_differences.csv")
    metric_index = {(row["cell"], row["stratifier"], row["bin"], row["baseline"]): row for row in metrics}
    diff_index = {(row["cell"], row["stratifier"], row["bin"], row["comparison"]): row for row in differences}
    output = []
    for cell in ("wan_l15", "vjepa2_l11"):
        for stratifier, unit in (("order_rotation_difference_rad", "rad"), ("order_camera_center_difference_m", "m")):
            for label in ("low", "medium", "high"):
                row = metric_index[(cell, stratifier, label, "full")]
                item: dict[str, Any] = {
                    "cell": cell, "model": model_label(cell.rsplit("_l", 1)[0]), "stratifier": stratifier,
                    "unit": unit, "bin": label, "training_q1": f(row["bin_q1"]), "training_q2": f(row["bin_q2"]),
                    "samples": int(row["samples"]), "scenes": int(row["scenes"]),
                    "normalized_order_margin": f(row["normalized_order_margin"]),
                    "normalized_order_margin_ci95_low": f(row["normalized_order_margin_ci95_low"]),
                    "normalized_order_margin_ci95_high": f(row["normalized_order_margin_ci95_high"]),
                    "wins": int(row["wins"]), "losses": int(row["losses"]), "ties": int(row["ties"]),
                    "tie_aware_accuracy": f(row["tie_aware_accuracy"]),
                }
                for control in ("action_only", "matched_shuffle", "commutative_additive"):
                    contrast = diff_index[(cell, stratifier, label, f"full_minus_{control}")]
                    item[f"full_minus_{control}"] = f(contrast["estimate"])
                    item[f"full_minus_{control}_ci95_low"] = f(contrast["ci95_low"])
                    item[f"full_minus_{control}_ci95_high"] = f(contrast["ci95_high"])
                    item[f"full_minus_{control}_holm_p"] = f(contrast["holm_adjusted_p_within_cell_stratum"])
                output.append(item)
    return output


def build_positive_table() -> list[dict[str, Any]]:
    rows = read_csv(P1 / "positive_control/paired_scene_bootstrap.csv")
    index = {(row["system"], row["metric"]): row for row in rows}
    output = []
    for system, label in (("analytic_oracle", "Analytic oracle"), ("learned_single_step_GRU", "Single-step learned GRU")):
        item: dict[str, Any] = {"system": label}
        for metric in (
            "one_step_point_rmse_m", "sequential_point_rmse_m", "direct_point_rmse_m",
            "sequential_direct_point_rmse_m", "raw_order_margin_m",
            "identity_pre_point_rmse_m", "identity_post_point_rmse_m",
        ):
            row = index[(system, metric)]
            item[metric] = f(row["estimate"]); item[f"{metric}_ci95_low"] = f(row["ci95_low"]); item[f"{metric}_ci95_high"] = f(row["ci95_high"])
        output.append(item)
    return output


def build_identity_table() -> list[dict[str, Any]]:
    bootstrap = read_csv(P1 / "composition/identity_padding_bootstrap.csv")
    support = {row["cell"]: row for row in read_csv(P1 / "composition/identity_action_support.csv")}
    index = {(row["cell"], row["baseline"], row["metric"]): row for row in bootstrap}
    output = []
    for cell in ("wan_l15", "vjepa2_l11"):
        item: dict[str, Any] = {
            "cell": cell,
            "identity_exact_training_match": support[cell]["identity_exact_training_match"],
            "identity_coordinatewise_training_range_check": support[cell]["identity_coordinatewise_training_range_check"],
            "identity_nearest_standardized_training_action_l2": f(support[cell]["identity_nearest_standardized_training_action_l2"]),
        }
        for metric in (
            "sequential_minus_direct_endpoint_error", "identity_pre_minus_direct_endpoint_error",
            "identity_post_minus_direct_endpoint_error", "sequential_minus_identity_pre_endpoint_error",
            "sequential_minus_identity_post_endpoint_error",
        ):
            row = index[(cell, "full", metric)]
            item[metric] = f(row["estimate"]); item[f"{metric}_ci95_low"] = f(row["ci95_low"]); item[f"{metric}_ci95_high"] = f(row["ci95_high"])
        output.append(item)
    return output


def latex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%")


def write_latex_tables(output: Path, main: list[dict[str, Any]], representative: list[dict[str, Any]], noncomm: list[dict[str, Any]], positive: list[dict[str, Any]]) -> None:
    row_end = r"\\"
    lines = [r"\begin{table*}[t]", r"\centering", r"\small", r"\caption{Frozen multilayer diagnostics. Errors are cosine distances to real isolated targets within each feature space and are not used for cross-model ranking. Gains and normalized quantities are scene-macro. $e_1$ is the diagnostic-head one-step prediction error.}", r"\label{tab:se3_multilayer}", r"\begin{tabular}{llrrrrrrr}", r"\toprule", r"Model & Layer & $e_1$ & $e_{\rm seq}$ & $e_{\rm dir}$ & Gain/copy & Gain/no-act. & Norm. comp. & Norm. order \\", r"\midrule"]
    for row in main:
        lines.append(f"{row['model']} & {row['layer']} & {row['diagnostic_head_one_step_error']:.4f} & {row['sequential_endpoint_error']:.4f} & {row['direct_endpoint_error']:.4f} & {row['gain_vs_copy']:.3f} & {row['gain_vs_without_action']:.3f} & {row['normalized_sequential_direct_distance']:.4f} & {row['normalized_order_margin']:.4f} {row_end}")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    (output / "table_main_layers.tex").write_text("\n".join(lines), encoding="utf-8")

    names = {"copy_relative_gain": r"Gain vs. copy $\uparrow$", "no_action_relative_gain": r"Gain vs. no-action $\uparrow$", "normalized_composition_distance": r"Normalized composition distance $\downarrow$", "normalized_order_effect": r"Normalized order margin $\uparrow$"}
    value_fields = {"copy_relative_gain": "gain_vs_copy", "no_action_relative_gain": "gain_vs_without_action", "normalized_composition_distance": "normalized_sequential_direct_distance", "normalized_order_effect": "normalized_order_margin"}
    cell_rows = {row["cell"]: row for row in main}
    lines = [r"\begin{table*}[t]", r"\centering", r"\small", r"\caption{One-step-to-two-step structural generalization at the predeclared representative layers. The paired difference is Wan L15 minus V-JEPA2 L11; the four comparisons form one Holm family.}", r"\label{tab:se3_representative}", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lrrrr}", r"\toprule", r"Metric & Wan L15 & V-JEPA2 L11 & Paired difference [95\% CI] & Holm $p$ \\", r"\midrule"]
    for row in representative:
        p_display = f"${row['holm_adjusted_p_paper_display']}$" if str(row["holm_adjusted_p_paper_display"]).startswith("<") else row["holm_adjusted_p_paper_display"]
        field = value_fields[row["metric"]]
        lines.append(f"{names[row['metric']]} & ${cell_rows['wan_l15'][field]:.4f}$ & ${cell_rows['vjepa2_l11'][field]:.4f}$ & ${row['estimate']:.4f}\;[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}]$ & {p_display} {row_end}")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    (output / "table_representative.tex").write_text("\n".join(lines), encoding="utf-8")

    high = [row for row in noncomm if row["bin"] == "high"]
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\caption{Observed-order sensitivity in the high physical non-commutativity strata. Rotation and camera-center displacement are never combined into one unit.}", r"\label{tab:se3_noncomm}", r"\begin{tabular}{llrrrr}", r"\toprule", r"Model & Stratum & Samples/scenes & Norm. margin & 95\% CI & Tie-aware acc. \\", r"\midrule"]
    for row in high:
        label = "rotation (rad)" if row["unit"] == "rad" else "camera center (m)"
        lines.append(f"{row['model']} & {label} & {row['samples']}/{row['scenes']} & {row['normalized_order_margin']:.4f} & [{row['normalized_order_margin_ci95_low']:.4f}, {row['normalized_order_margin_ci95_high']:.4f}] & {row['tie_aware_accuracy']:.3f} {row_end}")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (output / "table_noncommutativity.tex").write_text("\n".join(lines), encoding="utf-8")

    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\caption{Analytic positive control. Point errors are in metres. The learned updater uses only A$\rightarrow$B supervision and the same recurrent-head budget as the visual diagnostic.}", r"\label{tab:se3_positive}", r"\begin{tabular}{lrrrrr}", r"\toprule", r"System & One-step & Sequential & Direct & Seq.--direct & Order margin \\", r"\midrule"]
    for row in positive:
        lines.append(f"{row['system']} & {row['one_step_point_rmse_m']:.4f} & {row['sequential_point_rmse_m']:.4f} & {row['direct_point_rmse_m']:.4f} & {row['sequential_direct_point_rmse_m']:.4f} & {row['raw_order_margin_m']:.4f} {row_end}")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (output / "table_positive_control.tex").write_text("\n".join(lines), encoding="utf-8")


def generate_positive_control_figure(figures: Path, positive: list[dict[str, Any]]) -> None:
    fields = ("one_step_point_rmse_m", "sequential_point_rmse_m", "direct_point_rmse_m", "sequential_direct_point_rmse_m")
    labels = ("one-step", "sequential", "direct", "seq.–direct")
    fig, axis = plt.subplots(figsize=(7, 3.8)); x = np.arange(len(fields)); width = .35
    for index, row in enumerate(positive):
        positions = x + (index-.5)*width
        values = [row[field] for field in fields]
        axis.bar(positions, values, width, label=row["system"])
        if index == 0:
            # Zero-height bars are invisible; mark the oracle's exact zeros.
            axis.scatter(positions, values, marker="D", s=24, color="tab:blue", zorder=3)
            for position in positions:
                axis.annotate("0", (position, 0), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=7)
    axis.set_xticks(x, labels); axis.set_ylabel("3D point RMSE (m)"); axis.set_ylim(bottom=-0.035); axis.axhline(0, color="black", linewidth=.7); axis.legend(frameon=False); fig.tight_layout()
    fig.savefig(figures / "figure_positive_control.png", dpi=220); fig.savefig(figures / "figure_positive_control.pdf"); plt.close(fig)


def generate_figures(output: Path, main: list[dict[str, Any]], representative: list[dict[str, Any]], noncomm: list[dict[str, Any]], positive: list[dict[str, Any]]) -> None:
    figures = output / "figures"; figures.mkdir(parents=True, exist_ok=True)
    colors = {"Wan": "tab:blue", "V-JEPA2": "tab:orange"}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for model in ("Wan", "V-JEPA2"):
        rows = sorted([row for row in main if row["model"] == model], key=lambda row: row["layer"])
        x = [row["layer"] for row in rows]
        axes[0, 0].plot(x, [row["diagnostic_head_one_step_error"] for row in rows], marker="o", label=model, color=colors[model])
        axes[0, 1].plot(x, [row["gain_vs_copy"] for row in rows], marker="o", label=f"{model}: copy", color=colors[model])
        axes[0, 1].plot(x, [row["gain_vs_without_action"] for row in rows], marker="s", ls="--", label=f"{model}: no-action", color=colors[model])
        axes[1, 0].plot(x, [row["normalized_sequential_direct_distance"] for row in rows], marker="o", label=model, color=colors[model])
        axes[1, 1].errorbar(x, [row["normalized_order_margin"] for row in rows], yerr=[[row["normalized_order_margin"]-row["normalized_order_margin_ci95_low"] for row in rows], [row["normalized_order_margin_ci95_high"]-row["normalized_order_margin"] for row in rows]], marker="o", capsize=2, label=model, color=colors[model])
    axes[0, 0].set_ylabel("Diagnostic-head one-step error"); axes[0, 1].set_ylabel("Two-step grounding gain")
    axes[1, 0].set_ylabel("Normalized sequential–direct distance"); axes[1, 1].set_ylabel("Normalized order margin")
    for axis in axes.flat: axis.set_xlabel("Exported layer"); axis.grid(alpha=.2)
    axes[1, 1].axhline(0, color="black", lw=.7); axes[0, 0].legend(frameon=False); axes[0, 1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(figures / "figure_multilayer.png", dpi=220); fig.savefig(figures / "figure_multilayer.pdf"); plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 3.8)); x = np.arange(len(representative)); y = np.asarray([row["estimate"] for row in representative])
    low = np.asarray([row["ci95_low"] for row in representative]); high = np.asarray([row["ci95_high"] for row in representative])
    axis.errorbar(x, y, yerr=[y-low, high-y], fmt="o", capsize=4, color="black")
    axis.axhline(0, color="tab:red", lw=.8); axis.set_xticks(x, ["copy gain", "no-action gain", "norm. composition", "order effect"], rotation=15)
    axis.set_ylabel("Wan L15 minus V-JEPA2 L11"); axis.grid(axis="y", alpha=.2); fig.tight_layout()
    fig.savefig(figures / "figure_representative_comparison.png", dpi=220); fig.savefig(figures / "figure_representative_comparison.pdf"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for axis, stratifier in zip(axes, ("order_rotation_difference_rad", "order_camera_center_difference_m")):
        for model in ("Wan", "V-JEPA2"):
            rows = [row for row in noncomm if row["model"] == model and row["stratifier"] == stratifier]
            rows.sort(key=lambda row: ("low", "medium", "high").index(row["bin"]))
            axis.errorbar([row["bin"] for row in rows], [row["normalized_order_margin"] for row in rows], yerr=[[row["normalized_order_margin"]-row["normalized_order_margin_ci95_low"] for row in rows], [row["normalized_order_margin_ci95_high"]-row["normalized_order_margin"] for row in rows]], marker="o", capsize=3, label=model, color=colors[model])
        axis.axhline(0, color="black", lw=.7); axis.grid(alpha=.2)
    axes[0].set_xlabel("Order rotation difference (train-tertile bins)"); axes[1].set_xlabel("Order camera-center difference (train-tertile bins)")
    axes[0].set_ylabel("Normalized order margin"); axes[0].legend(frameon=False); fig.tight_layout()
    fig.savefig(figures / "figure_noncommutativity_strata.png", dpi=220); fig.savefig(figures / "figure_noncommutativity_strata.pdf"); plt.close(fig)

    generate_positive_control_figure(figures, positive)


def paper_section_audit(main: list[dict[str, Any]], representative: list[dict[str, Any]], noncomm: list[dict[str, Any]], positive: list[dict[str, Any]], overlap: dict[str, Any]) -> str:
    rep = {row["metric"]: row for row in representative}
    high = {(row["cell"], row["unit"]): row for row in noncomm if row["bin"] == "high"}
    oracle, learned = positive[0], positive[1]
    parent_relationships = read_csv(LAYERS / "layer_metric_relationships.csv")
    raw_comp_correlation = {
        row["model"]: f(row["pearson_across_exported_layers"])
        for row in parent_relationships if row["y"] == "sequential_direct_prediction_distance"
    }
    order_correlation = {
        row["model"]: f(row["pearson_across_exported_layers"])
        for row in parent_relationships if row["y"] == "normalized_order_margin"
    }
    normalized_comp_correlation = {}
    for label in ("Wan", "V-JEPA2"):
        values = [row for row in main if row["model"] == label]
        normalized_comp_correlation[label] = float(np.corrcoef(
            [row["diagnostic_head_one_step_error"] for row in values],
            [row["normalized_sequential_direct_distance"] for row in values],
        )[0, 1])
    return rf"""\section{{Compositional and observed-order generalization after one-step learning}}
\label{{sec:se3_one_step_generalization}}

\paragraph{{Question.}}
We use a deliberately restricted diagnostic to ask whether an update learned only from real one-step transitions generalizes to two structural tests.  Given a streaming state $s_A$, an external decoder $D$, and camera actions $T_1=T_{{AB}}$ and $T_2=T_{{BC}}$, a shared updater $U_\theta$ produces
\begin{{align}}
\hat z_{{\rm seq}} &= D(U_\theta(U_\theta(s_A,T_1),T_2)), &
\hat z_{{\rm dir}} &= D(U_\theta(s_A,T_2T_1)),\\
\hat z_{{\rm swap}} &= D(U_\theta(U_\theta(s_A,T_2),T_1)).
\end{{align}}
The two core diagnostics are sequential--direct composition consistency and observed-order sensitivity.  The A$\rightarrow$B error is reported only as the diagnostic head's one-step calibration, not as a new C1 experiment.

\paragraph{{Data and frozen interface.}}
The exact-v2 p12 set contains 436 tuples: 128 one-per-scene training tuples, 32 one-per-scene validation tuples, and 276 evaluation tuples from 69 scenes.  The evaluation scenes are exactly the 69 scenes used by the earlier development pilot, although none of the 276 exact A/B/C frame triples is identical to a pilot triple.  Thus exact feature matching does not make this an independent-scene confirmation set.  Each source is a causal streaming prefix of 12 real ordered frames ending at A; $z_A,z_B,z_C$ are isolated-target features used only as supervision or evaluation targets.  Wan uses blocks 0/5/10/15/20/25/29 at fixed scheduler argument 749, literal ``.'' null-prompt surrogate, deterministic sample-local noise, and repeat-tail padding from 12 to 81 input frames.  V-JEPA2 uses encoder blocks 0/5/11/17/23 with \texttt{{skip\_predictor=True}} and the fixed processor/repeat-tail path to 64 frames.

The updater is a two-layer width-128 C2-shaped GRU trained for seeds 42--46 only on A$\rightarrow$B targets, with the backbone frozen and validation selection fixed.  It is an external probe, not either foundation model's native predictor.  Its hidden tuple is the only recurrent state: isolated target features are never fed back as streaming context.  No composition, order, cycle, or multi-step loss is used.  Matrix actions are composed in SE(3) before 9-D encoding.  Controls are copy, without-condition, action-only, matched shuffle, and an independently evaluated commutative additive updater.

\paragraph{{Statistics.}}
We average the five aligned training seeds and aggregate tuples within scene.  Uncertainty uses 2,000 shared paired scene-bootstrap resamples.  The predeclared Wan-L15/V-JEPA2-L11 family contains four comparisons and is Holm corrected.  For the significant comparison the implementation observed zero opposing-tail resamples, yielding a stored Holm value of zero; at finite bootstrap resolution we report $p_{{\rm Holm}}<0.004$, not an exact zero probability.  Layer trends are descriptive only.

\input{{table_main_layers.tex}}

\paragraph{{Endpoint grounding.}}
All twelve layers improve two-step endpoint prediction relative to copy, with 95\% scene-bootstrap intervals above zero.  Relative to the without-action predictor, all five V-JEPA2 layers and six of seven Wan layers have positive intervals; Wan L0 is inconclusive.  These two comparisons answer different questions: copy gain measures endpoint grounding, whereas the without-action contrast isolates the incremental contribution of the correct action.  Raw cosine errors are defined in model-specific feature spaces and are not used to rank models.

Across the seven Wan layers, diagnostic-head one-step error correlates with raw sequential--direct distance at $r={raw_comp_correlation['wan']:.3f}$, with copy-normalized composition distance at $r={normalized_comp_correlation['Wan']:.3f}$, and with normalized order margin at $r={order_correlation['wan']:.3f}$.  Across the five V-JEPA2 layers the corresponding values are $r={raw_comp_correlation['vjepa2']:.3f}$, $r={normalized_comp_correlation['V-JEPA2']:.3f}$, and $r={order_correlation['vjepa2']:.3f}$.  These 7/5-point layer correlations are descriptive and can be dominated by endpoint layers; they are not used to select a best layer or infer a causal relation.

\paragraph{{Composition consistency.}}
At the predeclared representative layers, the normalized sequential--direct distance is {next(row['normalized_sequential_direct_distance'] for row in main if row['cell']=='wan_l15'):.4f} for Wan L15 and {next(row['normalized_sequential_direct_distance'] for row in main if row['cell']=='vjepa2_l11'):.4f} for V-JEPA2 L11.  The paired Wan-minus-V-JEPA2 difference is {rep['normalized_composition_distance']['estimate']:.4f} (95\% CI [{rep['normalized_composition_distance']['ci95_low']:.4f}, {rep['normalized_composition_distance']['ci95_high']:.4f}], $p_{{\rm Holm}}<0.004$; Table~\ref{{tab:se3_representative}}).  Both paths remain grounded by their errors to the real $z_C$ (Table~\ref{{tab:se3_multilayer}}); closeness of endpoint errors alone is not called composition consistency.  The identity-padding comparison is a training-external stress test: exact identity is absent and also fails the coordinatewise training-range check.  It rules out update count alone for Wan L15 but does not rule it out for V-JEPA2 L11.  The representative difference is restricted to this layer pair and metric and is not an overall model ranking or evidence of stronger SE(3) understanding.

\input{{table_representative.tex}}

\paragraph{{Observed-order sensitivity.}}
We separately measure the rotation-angle difference (radians) and camera-center displacement (metres) between $T_2T_1$ and $T_1T_2$.  Tertile thresholds are fixed using training geometry and shared across models and layers.  In the high rotation-difference stratum, Wan L15 has normalized margin {high[('wan_l15','rad')]['normalized_order_margin']:.4f} (CI [{high[('wan_l15','rad')]['normalized_order_margin_ci95_low']:.4f}, {high[('wan_l15','rad')]['normalized_order_margin_ci95_high']:.4f}]) and V-JEPA2 L11 {high[('vjepa2_l11','rad')]['normalized_order_margin']:.4f} (CI [{high[('vjepa2_l11','rad')]['normalized_order_margin_ci95_low']:.4f}, {high[('vjepa2_l11','rad')]['normalized_order_margin_ci95_high']:.4f}]).  The corresponding high camera-center results also cross zero (Table~\ref{{tab:se3_noncomm}}).  No robust control-adjusted order advantage is detected.  Because the high strata contain physically non-commuting trajectories, approximate commutativity alone is insufficient to explain the weak overall effect; motion magnitude, frame interval, and A--C overlap checks remain descriptive potential confounds.

\input{{table_noncommutativity.tex}}

\paragraph{{Analytic positive control and limitations.}}
The exact point-transform oracle has zero one-step, recursive, direct, identity, and sequential--direct error and a positive correct-versus-swapped margin of {oracle['raw_order_margin_m']:.4f}\,m.  With the same recurrent structure and budget but only A$\rightarrow$B supervision, the learned point updater has one-step RMSE {learned['one_step_point_rmse_m']:.4f}\,m, sequential RMSE {learned['sequential_point_rmse_m']:.4f}\,m, and order margin {learned['raw_order_margin_m']:.4f}\,m (its interval includes zero).  This limitation appears in the main text because it prevents attributing weak visual order evidence uniquely to the frozen representation.  It does not invalidate the intended one-step-to-two-step generalization test and is not a reason to change its training objective after observing the result.

\input{{table_positive_control.tex}}

In summary, the frozen features support grounded endpoint prediction, and V-JEPA2 L11 has a smaller normalized composition distance than Wan L15 under the predeclared paired comparison.  We do not detect a robust control-adjusted observed-order advantage.  These results diagnose one-sided generalization of an external recurrent probe; they neither establish absence of SE(3) information nor prove a complete group action.
"""


def paper_section(main: list[dict[str, Any]], representative: list[dict[str, Any]], noncomm: list[dict[str, Any]], positive: list[dict[str, Any]], overlap: dict[str, Any]) -> str:
    """Return the compact, main-text narrative; detailed tables remain supplemental."""
    rep = {row["metric"]: row for row in representative}
    oracle, learned = positive[0], positive[1]
    wan = next(row for row in main if row["cell"] == "wan_l15")
    vjepa2 = next(row for row in main if row["cell"] == "vjepa2_l11")
    return rf"""\section{{Can latent representations be updated by ordered camera motion?}}
\label{{sec:se3_one_step_generalization}}

When a camera moves from view $A$ to $B$ and then to $C$, we ask whether a
frozen visual representation can be updated according to the measured 3D
camera motion, and whether that update preserves motion composition and order.
We train an external recurrent diagnostic head---not the foundation model's
native predictor---only on observed one-step transitions: from a causal p12
streaming state ending at $A$ and the rigid transform $T_1=T_{{AB}}$, it
predicts the isolated target representation at $B$. At test time, the same
updater and checkpoint must reach view $C$ through two alternative executions,
\begin{{align}}
\hat z_{{\rm seq}} &= D\!\left(U(U(s_A,T_1),T_2)\right), &
\hat z_{{\rm dir}} &= D\!\left(U(s_A,T_2T_1)\right),
\end{{align}}
where $T_2=T_{{BC}}$ and the direct path first composes the motions as SE(3)
matrices. Composition consistency asks whether these two routes produce similar
predictions while remaining close to the real endpoint $z_C$. To test whether
the update also preserves order, we reverse the inputs,
$\hat z_{{\rm swap}}=D(U(U(s_A,T_2),T_1))$, and ask whether the correct-order
prediction is closer to $z_C$.

On the frozen exact-v2 split (436 tuples, seven Wan layers, five V-JEPA2 layers,
and seeds 42--46), every layer improves two-step endpoint prediction over
copying $A$; all but Wan L0 also improve over a predictor that does not receive
the camera motion. Thus the diagnostic head uses motion information rather
than only static scene persistence. At the predeclared representative layers,
the normalized sequential--direct distance is
{wan['normalized_sequential_direct_distance']:.4f} for Wan L15 and
{vjepa2['normalized_sequential_direct_distance']:.4f} for V-JEPA2 L11. Their paired difference is
{rep['normalized_composition_distance']['estimate']:.4f} (95\% CI
[{rep['normalized_composition_distance']['ci95_low']:.4f},
{rep['normalized_composition_distance']['ci95_high']:.4f}],
$p_{{\rm Holm}}<0.004$), providing evidence only for greater composition
consistency at this specified layer pair and metric, not an overall model
ranking. In contrast, neither model shows a robust control-adjusted preference
for the correct motion order, including among trajectories with large physical
non-commutativity.

Action-only, matched-shuffle, additive, and identity-padding checks do not
reveal a hidden stable order effect; identity is outside the training-action
support and is treated only as a stress test. The analytic SE(3) oracle passes,
whereas the learned GRU positive control trained with the same one-step
objective has sequential RMSE {learned['sequential_point_rmse_m']:.4f}\,m and no reliable order margin,
despite the oracle's positive margin of {oracle['raw_order_margin_m']:.4f}\,m. Therefore, useful
motion-conditioned endpoint prediction does not by itself establish an
order-preserving latent group action, and the weak order signal cannot be
attributed uniquely to the frozen representation rather than to one-step probe
generalization. Finally, the 276 evaluation tuples are new frame triples from
the same 69 scenes as the legacy pilot, so this is development evidence rather
than an independent-scene confirmation.

\input{{table_representative.tex}}
"""


def final_findings(main: list[dict[str, Any]], representative: list[dict[str, Any]], noncomm: list[dict[str, Any]], positive: list[dict[str, Any]]) -> str:
    rep = {row["metric"]: row for row in representative}
    high = [row for row in noncomm if row["bin"] == "high"]
    oracle, learned = positive
    return f"""# Final Findings

1. **Grounded two-step prediction is supported across the frozen multilayer set, but action-specific gain is distinct.** All 12 layers have positive scene-bootstrap gain over copy; all five V-JEPA2 layers and six of seven Wan layers also have positive gain over the without-action predictor (Wan L0 is inconclusive). See `table_main_layers.csv` and `figures/figure_multilayer.*`.

2. **The only supported predeclared cross-model difference is the representative-layer composition metric.** Wan L15 minus V-JEPA2 L11 normalized sequential--direct distance is {rep['normalized_composition_distance']['estimate']:.5f}, 95% CI [{rep['normalized_composition_distance']['ci95_low']:.5f}, {rep['normalized_composition_distance']['ci95_high']:.5f}]. The stored empirical/Holm value is zero because 0/2000 opposing-tail resamples occurred; with four registered Holm tests the paper reports $p_{{Holm}}<0.004$. Copy gain, no-action gain, and order effect do not differ reliably. This is not an overall model ranking. See `table_representative_comparison.csv` and `figures/figure_representative_comparison.*`.

3. **No robust control-adjusted order advantage is detected, and the cause is not identifiable as representation alone.** Even the high physical non-commutativity strata have CIs crossing zero (`table_noncommutativity_strata.csv`). The analytic oracle succeeds (order margin {oracle['raw_order_margin_m']:.4f} m), while the single-step learned GRU has one-step RMSE {learned['one_step_point_rmse_m']:.4f} m, sequential RMSE {learned['sequential_point_rmse_m']:.4f} m, and order margin {learned['raw_order_margin_m']:.4f} m with an interval crossing zero. Thus data geometry, diagnostic-head generalization, and representation remain partly confounded; this is neither evidence of no SE(3) capability nor proof of a full group action. See `table_positive_control.csv` and `figures/figure_positive_control.*`.
"""


def build(output: Path, skip_raw_audit: bool = False) -> None:
    if output in (LAYERS, P1):
        raise RuntimeError("paper-final output must be independent of source results")
    output.mkdir(parents=True, exist_ok=True)
    manifest = read_jsonl(LAYERS / "data/common_manifest.jsonl")
    overlap = data_overlap_audit()
    parent_rows = load_parent_sample_rows()
    audit = {
        "data_overlap": overlap,
        "aggregate_tables_vs_per_sample_metrics": aggregate_audit(parent_rows),
        "raw_predictions_vs_per_sample_metrics": {"status": "SKIPPED_BY_EXPLICIT_FLAG"} if skip_raw_audit else raw_prediction_audit(manifest),
    }
    write_json(output / "STATISTICAL_AUDIT.json", audit)

    main = build_main_table(); representative = build_representative_table()
    noncomm = build_noncomm_table(); positive = build_positive_table(); identity = build_identity_table()
    write_csv(output / "table_main_layers.csv", main)
    write_csv(output / "table_representative_comparison.csv", representative)
    write_csv(output / "table_noncommutativity_strata.csv", noncomm)
    write_csv(output / "table_positive_control.csv", positive)
    write_csv(output / "table_identity_stress_supplement.csv", identity)
    write_latex_tables(output, main, representative, noncomm, positive)
    generate_figures(output, main, representative, noncomm, positive)
    (output / "paper_section.tex").write_text(paper_section(main, representative, noncomm, positive, overlap), encoding="utf-8")
    (output / "FINAL_FINDINGS.md").write_text(final_findings(main, representative, noncomm, positive), encoding="utf-8")
    (output / "REPRODUCE.md").write_text(
        "# Reproduce the frozen paper package\n\n"
        "```bash\n"
        "MPLCONFIGDIR=/tmp/mpl-se3-paper-final \\\n"
        "/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python \\\n"
        "  -m analyses.se3_protocol_v2.wan_vjepa2_paper_final\n"
        "```\n\nThis command only reads frozen artifacts and regenerates audits, tables, figures, text, and hashes. It performs no training, model inference, or feature extraction.\n",
        encoding="utf-8",
    )

    sources = [
        LAYERS / "FROZEN_PROTOCOL.json", LAYERS / "DATASET_CARD.md", LAYERS / "REPORT.md",
        LAYERS / "FINDINGS.md", LAYERS / "data/common_manifest.jsonl", LAYERS / "layer_main_table.csv",
        LAYERS / "layer_metric_relationships.csv", LAYERS / "paired_scene_bootstrap.csv",
        P1 / "P1_PROTOCOL.json", P1 / "P1_FINDINGS.md",
        P1 / "layer_joint_main_table.csv", P1 / "composition/layer_baseline_summary.csv",
        P1 / "composition/paired_scene_bootstrap.csv", P1 / "representative/paired_model_comparison.csv",
        P1 / "order/stratified_metrics.csv", P1 / "order/paired_control_differences.csv",
        P1 / "positive_control/paired_scene_bootstrap.csv", LEGACY / "se3_observed_v1_manifest.jsonl",
        Path(__file__), REPO / "analyses/se3_protocol_v2/wan_vjepa2_v2_train.py",
        REPO / "vidfm3d/models/components/probe_path_integration.py",
    ]
    generated = sorted(path for path in output.rglob("*") if path.is_file() and path.name != "FREEZE_MANIFEST.json")
    freeze = {
        "schema_version": 1,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper_positioning": "compositional and observed-order generalization after one-step learning",
        "source_experiments": [str(LAYERS.relative_to(REPO)), str(P1.relative_to(REPO))],
        "frozen_configuration": {"samples": 436, "prefix": 12, "wan_layers": [0,5,10,15,20,25,29], "vjepa2_layers": [0,5,11,17,23], "seeds": list(SEEDS), "supervision": "A_to_B_only", "representative_cells": ["wan_l15", "vjepa2_l11"]},
        "prohibited_actions_performed": {"training": False, "feature_extraction": False, "model_inference": False, "scope_expansion": False, "source_overwrite": False},
        "data_overlap": overlap,
        "statistical_display": {"bootstrap_resamples": 2000, "registered_representative_tests": 4, "stored_zero_holm_p": 0.0, "paper_display": "p_Holm < 0.004", "reason": "0/2000 opposing-tail resamples under a two-sided empirical p followed by four-test Holm; zero is not printed as an exact probability"},
        "paper_only_corrections": [
            {"change": "rename C1 error to diagnostic-head one-step prediction error", "numeric_change": False},
            {"change": "display stored Holm p=0 using finite-resampling bound", "source_numeric_value_preserved": True},
            {"change": "state exact-v2 evaluation is same-scene, new-tuple development evidence rather than independent confirmation", "numeric_change": False},
            {"change": "use P1 ratio-of-scene-macro normalized composition summaries", "source_experiments_preserved": True},
            {"change": "emit valid double-backslash row terminators in every generated LaTeX table", "numeric_change": False},
            {"change": "restore the predeclared representative-layer comparison as the main-text table", "numeric_change": False},
            {"change": "rewrite the main-text narrative around camera motion, two execution paths, and action reversal", "numeric_change": False},
        ],
        "source_files": {str(path.relative_to(REPO)): sha256(path) for path in sources},
        "generated_files": {str(path.relative_to(output)): sha256(path) for path in generated},
        "audit_status": "PASS" if all(value.get("status", "PASS") .startswith("PASS") for key, value in audit.items() if key != "data_overlap") else "FAIL",
    }
    write_json(output / "FREEZE_MANIFEST.json", freeze)


def refresh_presentation(output: Path) -> None:
    """Refresh paper presentation and hashes without recomputing statistics."""
    manifest_path = output / "FREEZE_MANIFEST.json"
    if not manifest_path.is_file() or not (output / "STATISTICAL_AUDIT.json").is_file():
        raise RuntimeError("presentation refresh requires an existing, audited paper-final package")
    main = build_main_table()
    representative = build_representative_table()
    noncomm = build_noncomm_table()
    positive = build_positive_table()
    overlap = data_overlap_audit()
    write_latex_tables(output, main, representative, noncomm, positive)
    (output / "paper_section.tex").write_text(
        paper_section(main, representative, noncomm, positive, overlap),
        encoding="utf-8",
    )
    generate_positive_control_figure(output / "figures", positive)
    (output / "REPRODUCE.md").write_text(
        "# Reproduce the frozen paper package\n\n"
        "```bash\n"
        "MPLCONFIGDIR=/tmp/mpl-se3-paper-final \\\n"
        "/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python \\\n"
        f"  -m analyses.se3_protocol_v2.wan_vjepa2_paper_final --output {output}\n"
        "```\n\nThis command only reads frozen artifacts and regenerates audits, tables, figures, text, and hashes. It performs no training, model inference, or feature extraction.\n",
        encoding="utf-8",
    )
    freeze = json.loads(manifest_path.read_text(encoding="utf-8"))
    corrections = [
        {"change": "show analytic-oracle exact zeros explicitly in the positive-control figure", "numeric_change": False},
        {"change": "emit valid double-backslash row terminators in every generated LaTeX table", "numeric_change": False},
        {"change": "restore the predeclared representative-layer comparison as the main-text table", "numeric_change": False},
        {"change": "rewrite the main-text narrative around camera motion, two execution paths, and action reversal", "numeric_change": False},
    ]
    for correction in corrections:
        if correction not in freeze["paper_only_corrections"]:
            freeze["paper_only_corrections"].append(correction)
    freeze["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    generator_key = str(Path(__file__).relative_to(REPO))
    freeze["source_files"][generator_key] = sha256(Path(__file__))
    generated = sorted(path for path in output.rglob("*") if path.is_file() and path.name != "FREEZE_MANIFEST.json")
    freeze["generated_files"] = {str(path.relative_to(output)): sha256(path) for path in generated}
    write_json(manifest_path, freeze)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-raw-audit", action="store_true", help="development only; final package must not use this")
    parser.add_argument("--refresh-presentation-only", action="store_true", help="reuse an existing audited package and refresh figure presentation plus hashes")
    args = parser.parse_args()
    if args.refresh_presentation_only:
        refresh_presentation(args.output)
    else:
        build(args.output, args.skip_raw_audit)
