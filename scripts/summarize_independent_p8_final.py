#!/usr/bin/env python3
"""Create paper tables, figures and a machine-readable final result summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/independent_p8_matplotlib_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BOOTSTRAP_SEED = 20260913
N_BOOTSTRAP = 2000
MODELS = ("dino", "f3r", "vjepa2", "wan")


def fmt(mean: float, sd: float | None, lo: float, hi: float) -> str:
    spread = "NA" if sd is None or np.isnan(sd) else f"{sd:.4f}"
    return f"{mean:.4f} ± {spread} [{lo:.4f}, {hi:.4f}]"


def seed_stats(frame: pd.DataFrame, metric: str) -> tuple[float, float | None]:
    values = pd.to_numeric(frame[metric], errors="coerce")
    seeded = frame.assign(_value=values).dropna(subset=["_value"])
    per_seed = seeded.groupby("seed", dropna=False)["_value"].mean()
    mean = float(per_seed.mean())
    sd = float(per_seed.std(ddof=1)) if len(per_seed) > 1 and "NA" not in per_seed.index.astype(str) else None
    return mean, sd


def cluster_ci(frame: pd.DataFrame, metric: str, seed_offset: int = 0) -> tuple[float, float]:
    work = frame.copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna(subset=[metric])
    # Average repeated seeds for each physical sample before scene resampling.
    per_sample = work.groupby(["scene_id", "sample_id"], as_index=False)[metric].mean()
    scenes = sorted(per_sample.scene_id.unique())
    by_scene = {scene: per_sample.loc[per_sample.scene_id == scene, metric].to_numpy() for scene in scenes}
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    draws = np.empty(N_BOOTSTRAP)
    for index in range(N_BOOTSTRAP):
        chosen = rng.choice(scenes, size=len(scenes), replace=True)
        draws[index] = np.concatenate([by_scene[scene] for scene in chosen]).mean()
    return tuple(np.quantile(draws, [0.025, 0.975]).tolist())


def paired_effect(
    left: pd.DataFrame, right: pd.DataFrame, metric: str, effect_name: str,
    model: str, higher_is_better: bool, seed_offset: int,
) -> dict:
    keys = ["scene_id", "sample_id"]
    l = left.assign(_left=pd.to_numeric(left[metric], errors="coerce")).groupby(keys, as_index=False)._left.mean()
    r = right.assign(_right=pd.to_numeric(right[metric], errors="coerce")).groupby(keys, as_index=False)._right.mean()
    joined = l.merge(r, on=keys, how="inner", validate="one_to_one").dropna()
    joined["effect"] = joined._left - joined._right if higher_is_better else joined._right - joined._left
    scenes = sorted(joined.scene_id.unique())
    by_scene = {scene: joined.loc[joined.scene_id == scene, "effect"].to_numpy() for scene in scenes}
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    draws = np.empty(N_BOOTSTRAP)
    for index in range(N_BOOTSTRAP):
        chosen = rng.choice(scenes, size=len(scenes), replace=True)
        draws[index] = np.concatenate([by_scene[scene] for scene in chosen]).mean()
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"model": model, "effect": effect_name, "metric": metric, "n": len(joined),
            "estimate": joined.effect.mean(), "ci_low": lo, "ci_high": hi,
            "direction": "positive favors first named method"}


def write_table(frame: pd.DataFrame, csv_path: Path, tex_path: Path) -> None:
    if frame.empty:
        raise ValueError(f"empty table: {csv_path}")
    frame.to_csv(csv_path, index=False)
    tex_path.write_text(frame.to_latex(index=False, escape=True, float_format=lambda x: f"{x:.4f}"))


def actionability(all_action: pd.DataFrame, tables: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    groups = ["model", "variant", "transform"]
    for key, frame in all_action.groupby(groups, dropna=False):
        row = dict(zip(groups, key))
        for offset, metric in enumerate(("correct", "regret", "positive_progress_selected")):
            mean, sd = seed_stats(frame, metric)
            lo, hi = cluster_ci(frame, metric, offset)
            row.update({metric: mean, f"{metric}_seed_sd": sd, f"{metric}_ci_low": lo, f"{metric}_ci_high": hi})
        row["scene_macro_accuracy"] = frame.assign(correct=pd.to_numeric(frame.correct)).groupby(["seed", "scene_id"]).correct.mean().groupby("seed").mean().mean()
        regrets = pd.to_numeric(frame.regret)
        row["median_regret"] = regrets.median(); row["p90_regret"] = regrets.quantile(.9)
        row["n_samples"] = frame.sample_id.nunique(); row["n_scenes"] = frame.scene_id.nunique()
        rows.append(row)
    summary = pd.DataFrame(rows)
    write_table(summary, tables / "actionability_all_conditions.csv", tables / "actionability_all_conditions.tex")
    main_order = [
        ("random_expected", "NA"), ("candidate_prior", "NA"), ("temporal_direction_prior", "NA"),
        ("action_only", "raw"), ("current_action_no_goal", "raw"), ("goal_action_no_current", "raw"),
        ("direct", "raw"), ("clean_dynamics_selection", "raw"),
        ("observed_endpoint_feature_reference", "raw"),
    ]
    main_parts = []
    for order, (variant, transform) in enumerate(main_order):
        part = summary[(summary.variant == variant) & (summary["transform"].astype(str) == transform)].copy()
        part["order"] = order; main_parts.append(part)
    main = pd.concat(main_parts).sort_values(["model", "order"]).drop(columns="order")
    write_table(main, tables / "TABLE_1_ACTIONABILITY.csv", tables / "TABLE_1_ACTIONABILITY.tex")

    effects = []
    for mi, model in enumerate(MODELS):
        source = all_action[all_action.model == model]
        direct = source[(source.variant == "direct") & (source["transform"] == "raw")]
        random = source[source.variant == "random_expected"]
        dynamics = source[(source.variant == "clean_dynamics_selection") & (source["transform"] == "raw")]
        endpoint = source[(source.variant == "observed_endpoint_feature_reference") & (source["transform"] == "raw")]
        shuffled = source[(source.variant == "direct_shuffled_goal") & (source["transform"] == "raw")]
        action_only = source[(source.variant == "action_only") & (source["transform"] == "raw")]
        no_goal = source[(source.variant == "current_action_no_goal") & (source["transform"] == "raw")]
        no_current = source[(source.variant == "goal_action_no_current") & (source["transform"] == "raw")]
        for ei, (left, right, name) in enumerate(((direct, random, "direct_minus_random"),
                                                  (direct, action_only, "direct_minus_action_only"),
                                                  (direct, no_goal, "direct_minus_current_action_no_goal"),
                                                  (dynamics, direct, "dynamics_minus_direct"),
                                                  (endpoint, direct, "endpoint_reference_minus_direct"),
                                                  (direct, shuffled, "direct_minus_shuffled_goal"),
                                                  (direct, no_current, "direct_minus_goal_action_no_current"))):
            effects.append(paired_effect(left, right, "correct", name, model, True, 100 + mi * 10 + ei))
            effects.append(paired_effect(left, right, "regret", name, model, False, 200 + mi * 10 + ei))
    effects_frame = pd.DataFrame(effects)
    write_table(effects_frame, tables / "actionability_paired_effects.csv", tables / "actionability_paired_effects.tex")

    strata_cols = ("goal_temporal_direction", "motion_type", "endpoint_seen_in_context")
    strata = []
    primary = all_action[
        all_action.variant.isin(("direct", "clean_dynamics_selection", "observed_endpoint_feature_reference"))
        & (all_action["transform"] == "raw")
    ]
    physical = primary.drop_duplicates(["model", "sample_id"])[["model", "sample_id", "goal_distance", "utility_margin"]].copy()
    for column, output in (("goal_distance", "goal_distance_tertile"), ("utility_margin", "utility_margin_tertile")):
        physical[output] = physical.groupby("model")[column].transform(
            lambda values: pd.qcut(values.rank(method="first"), 3, labels=("low", "mid", "high"))
        )
    primary = primary.merge(physical[["model", "sample_id", "goal_distance_tertile", "utility_margin_tertile"]],
                            on=["model", "sample_id"], how="left", validate="many_to_one")
    strata_cols = (*strata_cols, "goal_distance_tertile", "utility_margin_tertile")
    for column in strata_cols:
        for key, frame in primary.groupby(["model", "variant", column], observed=True):
            strata.append({"model": key[0], "variant": key[1], "stratum": column, "level": key[2],
                           "n": frame.sample_id.nunique(), "accuracy": pd.to_numeric(frame.correct).mean(),
                           "mean_regret": pd.to_numeric(frame.regret).mean(),
                           "positive_progress_rate": pd.to_numeric(frame.positive_progress_selected).mean()})
    strata_frame = pd.DataFrame(strata)
    write_table(strata_frame, tables / "actionability_strata.csv", tables / "actionability_strata.tex")
    return main, effects_frame


def dynamics(all_dynamics: pd.DataFrame, tables: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for key, frame in all_dynamics.groupby(["model", "variant", "horizon"], dropna=False):
        row = {"model": key[0], "variant": key[1], "horizon": int(key[2]),
               "n_samples": frame.sample_id.nunique(), "n_scenes": frame.scene_id.nunique()}
        for offset, metric in enumerate(("cosine_distance", "mse", "retrieval_r1", "retrieval_r5")):
            numeric = pd.to_numeric(frame[metric], errors="coerce")
            if numeric.notna().any():
                mean, sd = seed_stats(frame, metric); lo, hi = cluster_ci(frame, metric, 300 + offset)
                row.update({metric: mean, f"{metric}_seed_sd": sd, f"{metric}_ci_low": lo, f"{metric}_ci_high": hi})
        if "action_sensitivity_cosine" in frame:
            numeric = pd.to_numeric(frame.action_sensitivity_cosine, errors="coerce")
            if numeric.notna().any():
                mean, sd = seed_stats(frame, "action_sensitivity_cosine")
                lo, hi = cluster_ci(frame, "action_sensitivity_cosine", 350)
                row.update({"action_sensitivity_cosine": mean, "action_sensitivity_cosine_seed_sd": sd,
                            "action_sensitivity_cosine_ci_low": lo, "action_sensitivity_cosine_ci_high": hi})
        rows.append(row)
    summary = pd.DataFrame(rows)
    write_table(summary, tables / "dynamics_all_horizons.csv", tables / "dynamics_all_horizons.tex")
    main = summary[summary.horizon.isin([1, 2, 4])].copy()
    write_table(main, tables / "dynamics_main.csv", tables / "dynamics_main.tex")
    effects = []
    for mi, model in enumerate(MODELS):
        source = all_dynamics[all_dynamics.model == model]
        for horizon in (1, 2, 4):
            at_h = source[pd.to_numeric(source.horizon) == horizon]
            full = at_h[at_h.variant == "full"]
            copy = at_h[at_h.variant == "copy_current"]
            visual = at_h[at_h.variant == "visual_only"]
            shuffled = at_h[at_h.variant == "full_matched_action_shuffle"]
            for ei, (left, right, name) in enumerate(((full, copy, "full_gain_over_copy"),
                                                      (full, visual, "full_gain_over_visual_only"),
                                                      (full, shuffled, "full_gain_over_shuffled_action"))):
                for metric_index, metric in enumerate(("cosine_distance", "mse")):
                    value = paired_effect(left, right, metric, name, model, False,
                                          400 + mi * 100 + horizon * 10 + ei * 2 + metric_index)
                    value["horizon"] = horizon
                    effects.append(value)
            for metric_index, metric in enumerate(("retrieval_r1", "retrieval_r5")):
                value = paired_effect(full, visual, metric, "full_gain_over_visual_only", model, True,
                                      1400 + mi * 20 + horizon * 2 + metric_index)
                value["horizon"] = horizon
                effects.append(value)
    effects_frame = pd.DataFrame(effects)
    write_table(effects_frame, tables / "dynamics_paired_effects.csv", tables / "dynamics_paired_effects.tex")
    return main, effects_frame


def algebra(all_algebra: pd.DataFrame, tables: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    composition = all_algebra[all_algebra.family == "composition"]
    comp_rows = []
    for key, frame in composition.groupby(["model", "variant"]):
        row = {"model": key[0], "variant": key[1], "n_samples": frame.sample_id.nunique()}
        for offset, metric in enumerate(("prediction_to_endpoint_cosine", "prediction_to_endpoint_mse",
                                         "consistency_to_four_primitive_cosine", "action_dependence_wrong_action_separation")):
            mean, sd = seed_stats(frame, metric); lo, hi = cluster_ci(frame, metric, 500 + offset)
            row.update({metric: mean, f"{metric}_seed_sd": sd, f"{metric}_ci_low": lo, f"{metric}_ci_high": hi})
        comp_rows.append(row)
    comp = pd.DataFrame(comp_rows)
    write_table(comp, tables / "algebra_composition.csv", tables / "algebra_composition.tex")

    cycles = all_algebra[all_algebra.family == "identity_inverse"]
    cycle_rows = []
    for key, frame in cycles.groupby(["model", "variant"]):
        row = {"model": key[0], "variant": key[1], "n_samples": frame.sample_id.nunique()}
        for offset, metric in enumerate(("decoded_to_anchor_cosine", "decoded_to_anchor_mse",
                                         "forward_prediction_to_endpoint_cosine", "wrong_action_separation")):
            mean, sd = seed_stats(frame, metric); lo, hi = cluster_ci(frame, metric, 600 + offset)
            row.update({metric: mean, f"{metric}_seed_sd": sd, f"{metric}_ci_low": lo, f"{metric}_ci_high": hi})
        cycle_rows.append(row)
    cycle = pd.DataFrame(cycle_rows)
    write_table(cycle, tables / "algebra_identity_inverse.csv", tables / "algebra_identity_inverse.tex")

    loops = all_algebra[all_algebra.family == "observed_loop_net_action"]
    loop_rows = []
    for key, frame in loops.groupby(["model", "variant"]):
        row = {"model": key[0], "variant": key[1], "n_records": frame.sample_id.nunique(),
               "n_pairs": frame.pair_id.nunique(), "n_scenes": frame.scene_id.nunique()}
        for offset, metric in enumerate(("prediction_to_actual_endpoint_cosine", "actual_endpoint_to_anchor_cosine",
                                         "prediction_to_anchor_cosine")):
            mean, sd = seed_stats(frame, metric); lo, hi = cluster_ci(frame, metric, 700 + offset)
            row.update({metric: mean, f"{metric}_seed_sd": sd, f"{metric}_ci_low": lo, f"{metric}_ci_high": hi})
        loop_rows.append(row)
    loop = pd.DataFrame(loop_rows)
    write_table(loop, tables / "algebra_observed_loops.csv", tables / "algebra_observed_loops.tex")

    effects = []
    for mi, model in enumerate(MODELS):
        source_comp = composition[composition.model == model]
        source_cycle = cycles[cycles.model == model]
        comparisons = (
            (source_comp[source_comp.variant == "one_composed"], source_comp[source_comp.variant == "four_primitives"],
             "one_composed_gain_over_four_primitives", "prediction_to_endpoint_cosine"),
            (source_cycle[source_cycle.variant == "inverse_cycle"], source_cycle[source_cycle.variant == "wrong_inverse_cycle"],
             "inverse_gain_over_wrong_inverse", "decoded_to_anchor_cosine"),
            (source_cycle[source_cycle.variant == "inverse_cycle"], source_cycle[source_cycle.variant == "repeat_composed"],
             "inverse_gain_over_repeated_action", "decoded_to_anchor_cosine"),
            (source_cycle[source_cycle.variant == "inverse_cycle"], source_cycle[source_cycle.variant == "copy_constant"],
             "inverse_gain_over_copy", "decoded_to_anchor_cosine"),
        )
        for ci, (left, right, name, metric) in enumerate(comparisons):
            effects.append(paired_effect(left, right, metric, name, model, False, 800 + mi * 10 + ci))

        model_loops = loops[loops.model == model].copy()
        for li, metric in enumerate(("prediction_to_actual_endpoint_cosine", "actual_endpoint_to_anchor_cosine",
                                     "prediction_to_anchor_cosine")):
            values = model_loops.assign(_value=pd.to_numeric(model_loops[metric], errors="coerce")).groupby(
                ["scene_id", "pair_id", "variant"], as_index=False)._value.mean()
            pivot = values.pivot(index=["scene_id", "pair_id"], columns="variant", values="_value").dropna().reset_index()
            pivot["effect"] = pivot["matched_non_loop"] - pivot["real_loop"]
            scenes = sorted(pivot.scene_id.unique()); by_scene = {s: pivot.loc[pivot.scene_id == s, "effect"].to_numpy() for s in scenes}
            rng = np.random.default_rng(BOOTSTRAP_SEED + 900 + mi * 10 + li); draws = []
            for _ in range(N_BOOTSTRAP):
                chosen = rng.choice(scenes, size=len(scenes), replace=True)
                draws.append(np.concatenate([by_scene[s] for s in chosen]).mean())
            lo, hi = np.quantile(draws, [.025, .975])
            effects.append({"model": model, "effect": "matched_non_loop_minus_real_loop", "metric": metric,
                            "n": len(pivot), "estimate": pivot.effect.mean(), "ci_low": lo, "ci_high": hi,
                            "direction": "positive means larger value for matched non-loop"})
    effects_frame = pd.DataFrame(effects)
    write_table(effects_frame, tables / "algebra_paired_effects.csv", tables / "algebra_paired_effects.tex")
    return comp, cycle, loop


def figures(action_raw: pd.DataFrame, action: pd.DataFrame, dynamics_frame: pd.DataFrame,
            comp: pd.DataFrame, cycles: pd.DataFrame, loops: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    labels = {"dino": "DINO", "f3r": "F3R", "vjepa2": "V-JEPA2", "wan": "Wan"}
    selected = action[action.variant.isin(["direct", "clean_dynamics_selection", "observed_endpoint_feature_reference"])]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2)); width = .24; x = np.arange(4)
    variants = ("direct", "clean_dynamics_selection", "observed_endpoint_feature_reference")
    for i, variant in enumerate(variants):
        part = selected[selected.variant == variant].set_index("model").reindex(MODELS)
        for ax, metric, low, high in ((axes[0], "correct", "correct_ci_low", "correct_ci_high"),
                                      (axes[1], "regret", "regret_ci_low", "regret_ci_high")):
            values = part[metric]; ax.bar(x + (i - 1) * width, values, width, label=variant.replace("_", " "))
            lower_error = np.maximum(0.0, values.to_numpy() - part[low].to_numpy())
            upper_error = np.maximum(0.0, part[high].to_numpy() - values.to_numpy())
            ax.errorbar(x + (i - 1) * width, values, [lower_error, upper_error], fmt="none", c="black", capsize=2)
    axes[0].axhline(.5, color="gray", ls="--", lw=1); axes[0].set_ylim(0, 1); axes[0].set_ylabel("Validation accuracy ↑")
    axes[1].set_ylabel("Pose-metric regret ↓")
    for ax in axes: ax.set_xticks(x, [labels[m] for m in MODELS])
    axes[1].legend(fontsize=7); fig.tight_layout()
    fig.savefig(figures_dir / "FIGURE_1_ACTIONABILITY.pdf"); fig.savefig(figures_dir / "FIGURE_1_ACTIONABILITY.png", dpi=220); plt.close(fig)

    main = dynamics_frame[(dynamics_frame.variant.isin(["full", "visual_only", "copy_current"])) & dynamics_frame.horizon.isin([1, 2, 4])]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
    for model in MODELS:
        for variant in ("full", "visual_only", "copy_current"):
            part = main[(main.model == model) & (main.variant == variant)].sort_values("horizon")
            axes[0].plot(part.horizon, part.cosine_distance, marker="o", label=f"{labels[model]} / {variant}")
    axes[0].set_xticks([1, 2, 4]); axes[0].set_xlabel("Horizon"); axes[0].set_ylabel("Forward cosine distance ↓")
    show = comp[comp.variant.isin(["two_composed", "one_composed", "identity_middle_composed"])]
    for variant, marker in (("two_composed", "o"), ("one_composed", "s"), ("identity_middle_composed", "^")):
        part = show[show.variant == variant].set_index("model").reindex(MODELS)
        axes[1].scatter(part.prediction_to_endpoint_cosine, part.consistency_to_four_primitive_cosine,
                        marker=marker, s=55, label=variant.replace("_", " "))
        for model, xv, yv in zip(MODELS, part.prediction_to_endpoint_cosine, part.consistency_to_four_primitive_cosine):
            axes[1].annotate(labels[model], (xv, yv), fontsize=6, xytext=(2, 2), textcoords="offset points")
    axes[1].set_xlabel("Endpoint error ↓"); axes[1].set_ylabel("Consistency to 4 primitives ↓")
    cycle_show = cycles[cycles.variant.isin(["identity", "inverse_cycle", "wrong_inverse_cycle", "repeat_composed", "copy_constant"])]
    pivot = cycle_show.pivot(index="model", columns="variant", values="decoded_to_anchor_cosine").reindex(MODELS)
    pivot.plot.bar(ax=axes[2]); axes[2].set_xlabel(""); axes[2].set_ylabel("Decoded-to-anchor distance ↓"); axes[2].tick_params(axis="x", rotation=0)
    axes[0].legend(fontsize=5, ncol=2); axes[1].legend(fontsize=6); axes[2].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(figures_dir / "FIGURE_2_DYNAMICS_ALGEBRA.pdf")
    fig.savefig(figures_dir / "FIGURE_2_DYNAMICS_ALGEBRA.png", dpi=220); plt.close(fig)

    primary = action_raw[(action_raw.variant.isin(["direct", "clean_dynamics_selection"])) & (action_raw["transform"] == "raw")].copy()
    physical = primary.drop_duplicates(["model", "sample_id"])[["model", "sample_id", "goal_distance"]]
    physical["difficulty"] = physical.groupby("model").goal_distance.transform(
        lambda values: pd.qcut(values.rank(method="first"), 4, labels=("Q1 near", "Q2", "Q3", "Q4 far")))
    primary = primary.merge(physical, on=["model", "sample_id", "goal_distance"], how="left", validate="many_to_one")
    curve = primary.groupby(["model", "variant", "difficulty"], observed=True).correct.apply(lambda x: pd.to_numeric(x).mean()).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for model in MODELS:
        for variant, ls in (("direct", "-"), ("clean_dynamics_selection", "--")):
            part = curve[(curve.model == model) & (curve.variant == variant)]
            axes[0].plot(range(len(part)), part.correct, marker="o", ls=ls, label=f"{labels[model]} / {variant}")
    axes[0].set_xticks(range(4), ["Q1 near", "Q2", "Q3", "Q4 far"]); axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Accuracy ↑"); axes[0].set_xlabel("Goal-distance quartile")
    loop_show = loops.set_index(["model", "variant"])
    width = .35
    for i, variant in enumerate(("real_loop", "matched_non_loop")):
        values = [loop_show.loc[(model, variant), "prediction_to_actual_endpoint_cosine"] for model in MODELS]
        axes[1].bar(np.arange(4) + (i - .5) * width, values, width, label=variant.replace("_", " "))
    axes[1].set_xticks(np.arange(4), [labels[m] for m in MODELS]); axes[1].set_ylabel("Prediction-to-observed endpoint ↓")
    axes[0].legend(fontsize=5, ncol=2); axes[1].legend(fontsize=7); fig.tight_layout()
    fig.savefig(figures_dir / "FIGURE_3_DIFFICULTY_LOOPS.pdf"); fig.savefig(figures_dir / "FIGURE_3_DIFFICULTY_LOOPS.png", dpi=220); plt.close(fig)


def integrity_controls(run_root: Path, tables: Path) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        value = json.loads((run_root / model / "training_and_integrity.json").read_text())
        scorer = value["training"]["scorers"]
        failures = [int(item["candidate_swap_equivariance_failures"]) for item in scorer.values()]
        direct_failures = [int(item["candidate_swap_equivariance_failures"])
                           for key, item in scorer.items() if key.startswith("direct|")]
        rows.append({"model": model, "status": value["status"], "seeds": ",".join(map(str, value["seeds"])),
                     "scorer_configurations": len(scorer),
                     "candidate_swap_failures_all_scorers": sum(failures),
                     "candidate_swap_failures_direct": sum(direct_failures),
                     "matched_action_shuffle_unavailable": value["matched_action_shuffle_unavailable"],
                     "full_dynamics_checkpoints": len(value["checkpoints"]),
                     "supervised_rollout_horizons": "h1,h2,h4"})
    frame = pd.DataFrame(rows)
    write_table(frame, tables / "integrity_controls.csv", tables / "integrity_controls.tex")
    return frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True); tables = args.output_dir / "tables"; tables.mkdir(); figures_dir = args.output_dir / "figures"
    action_parts = []; dynamics_parts = []; algebra_parts = []
    for model in MODELS:
        root = args.run_root / model
        action_parts.append(pd.read_csv(root / "actionability_per_sample.csv", dtype={"seed": str}, keep_default_na=False))
        dynamics_parts.append(pd.read_csv(root / "dynamics_per_sample.csv", dtype={"seed": str}, keep_default_na=False))
        algebra_parts.append(pd.read_csv(root / "algebra_per_sample.csv", dtype={"seed": str}, keep_default_na=False))
    action = pd.concat(action_parts, ignore_index=True); dyn = pd.concat(dynamics_parts, ignore_index=True); alg = pd.concat(algebra_parts, ignore_index=True)
    action.to_csv(args.output_dir / "actionability_per_sample.csv", index=False)
    dyn.to_csv(args.output_dir / "dynamics_per_sample.csv", index=False)
    alg.to_csv(args.output_dir / "algebra_per_sample.csv", index=False)
    action_main, action_effects = actionability(action, tables)
    dyn_main, dyn_effects = dynamics(dyn, tables)
    comp, cycles, loops = algebra(alg, tables)
    controls = integrity_controls(args.run_root, tables)
    table2_rows = []
    for variant in ("full", "visual_only", "action_only", "copy_current", "full_matched_action_shuffle"):
        part = dyn_main[(dyn_main.variant == variant) & (dyn_main.horizon == 4)].copy()
        for _, row in part.iterrows():
            sensitivity = dyn_main[(dyn_main.model == row.model) & (dyn_main.variant == "full_matched_action_shuffle") & (dyn_main.horizon == 4)]
            sensitivity = sensitivity.iloc[0] if variant in ("full", "full_matched_action_shuffle") and len(sensitivity) == 1 else None
            table2_rows.append({"section": "forward_h4", "model": row.model, "variant": row.variant,
                "correctness_cosine": row.cosine_distance, "correctness_cosine_seed_sd": row.get("cosine_distance_seed_sd"),
                "correctness_cosine_ci_low": row.get("cosine_distance_ci_low"), "correctness_cosine_ci_high": row.get("cosine_distance_ci_high"),
                "correctness_mse": row.get("mse"), "correctness_mse_seed_sd": row.get("mse_seed_sd"),
                "correctness_mse_ci_low": row.get("mse_ci_low"), "correctness_mse_ci_high": row.get("mse_ci_high"),
                "structure_consistency": np.nan, "structure_consistency_seed_sd": np.nan,
                "structure_consistency_ci_low": np.nan, "structure_consistency_ci_high": np.nan,
                "action_dependence": sensitivity.get("action_sensitivity_cosine") if sensitivity is not None else np.nan,
                "action_dependence_seed_sd": sensitivity.get("action_sensitivity_cosine_seed_sd") if sensitivity is not None else np.nan,
                "action_dependence_ci_low": sensitivity.get("action_sensitivity_cosine_ci_low") if sensitivity is not None else np.nan,
                "action_dependence_ci_high": sensitivity.get("action_sensitivity_cosine_ci_high") if sensitivity is not None else np.nan})
    for variant in ("four_primitives", "two_composed", "one_composed", "identity_middle_composed"):
        part = comp[comp.variant == variant]
        for _, row in part.iterrows():
            table2_rows.append({"section": "composition", "model": row.model, "variant": row.variant,
                "correctness_cosine": row.prediction_to_endpoint_cosine,
                "correctness_cosine_seed_sd": row.prediction_to_endpoint_cosine_seed_sd,
                "correctness_cosine_ci_low": row.prediction_to_endpoint_cosine_ci_low,
                "correctness_cosine_ci_high": row.prediction_to_endpoint_cosine_ci_high,
                "correctness_mse": row.prediction_to_endpoint_mse,
                "correctness_mse_seed_sd": row.prediction_to_endpoint_mse_seed_sd,
                "correctness_mse_ci_low": row.prediction_to_endpoint_mse_ci_low,
                "correctness_mse_ci_high": row.prediction_to_endpoint_mse_ci_high,
                "structure_consistency": row.consistency_to_four_primitive_cosine,
                "structure_consistency_seed_sd": row.consistency_to_four_primitive_cosine_seed_sd,
                "structure_consistency_ci_low": row.consistency_to_four_primitive_cosine_ci_low,
                "structure_consistency_ci_high": row.consistency_to_four_primitive_cosine_ci_high,
                "action_dependence": row.action_dependence_wrong_action_separation,
                "action_dependence_seed_sd": row.action_dependence_wrong_action_separation_seed_sd,
                "action_dependence_ci_low": row.action_dependence_wrong_action_separation_ci_low,
                "action_dependence_ci_high": row.action_dependence_wrong_action_separation_ci_high})
    for variant in ("identity", "inverse_cycle", "wrong_inverse_cycle", "repeat_composed", "copy_constant"):
        part = cycles[cycles.variant == variant]
        for _, row in part.iterrows():
            table2_rows.append({"section": "identity_inverse", "model": row.model, "variant": row.variant,
                "correctness_cosine": row.forward_prediction_to_endpoint_cosine,
                "correctness_cosine_seed_sd": row.forward_prediction_to_endpoint_cosine_seed_sd,
                "correctness_cosine_ci_low": row.forward_prediction_to_endpoint_cosine_ci_low,
                "correctness_cosine_ci_high": row.forward_prediction_to_endpoint_cosine_ci_high,
                "correctness_mse": np.nan, "correctness_mse_seed_sd": np.nan,
                "correctness_mse_ci_low": np.nan, "correctness_mse_ci_high": np.nan,
                "structure_consistency": row.decoded_to_anchor_cosine,
                "structure_consistency_seed_sd": row.decoded_to_anchor_cosine_seed_sd,
                "structure_consistency_ci_low": row.decoded_to_anchor_cosine_ci_low,
                "structure_consistency_ci_high": row.decoded_to_anchor_cosine_ci_high,
                "action_dependence": row.wrong_action_separation,
                "action_dependence_seed_sd": row.wrong_action_separation_seed_sd,
                "action_dependence_ci_low": row.wrong_action_separation_ci_low,
                "action_dependence_ci_high": row.wrong_action_separation_ci_high})
    table2 = pd.DataFrame(table2_rows)
    write_table(table2, tables / "TABLE_2_DYNAMICS_ALGEBRA.csv", tables / "TABLE_2_DYNAMICS_ALGEBRA.tex")
    figures(action, action_main, dyn_main, comp, cycles, loops, figures_dir)
    hashes = {}
    for path in sorted(args.output_dir.rglob("*")):
        if path.is_file(): hashes[str(path.relative_to(args.output_dir))] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {"status": "COMPLETE_VALID", "models": list(MODELS), "bootstrap_replicates": N_BOOTSTRAP,
               "bootstrap_seed": BOOTSTRAP_SEED, "artifact_sha256": hashes,
               "row_counts": {"actionability": len(action), "dynamics": len(dyn), "algebra": len(alg)},
               "candidate_swap_failures_direct": int(controls.candidate_swap_failures_direct.sum())}
    (args.output_dir / "SUMMARY_INTEGRITY.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
