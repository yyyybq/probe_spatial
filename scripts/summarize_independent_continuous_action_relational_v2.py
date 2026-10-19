#!/usr/bin/env python3
"""Aggregate relational-v2 continuous-action outputs with paired scene bootstrap."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from run_independent_p8_final_analysis import action9_to_matrix, read_jsonl
from run_independent_continuous_action import action_components, motion_distance, rotation_angle

MODELS = ("dino", "f3r", "vjepa2", "wan")
DISPLAY = {"dino": "DINO", "f3r": "Fast3R", "vjepa2": "V-JEPA2", "wan": "Wan"}
METRICS = (
    "direction_error_deg", "distance_mae", "distance_relative_error", "rotation_error_deg",
    "positive_progress", "goal_distance_reduction", "remaining_goal_distance",
)
HIGHER_IS_BETTER = {"positive_progress", "goal_distance_reduction"}
N_BOOTSTRAP = 5000
BOOTSTRAP_SEED = 20260923


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(value.rstrip() + "\n")


def seed_stats(frame: pd.DataFrame, metric: str) -> tuple[float, float | None]:
    work = frame.assign(_value=pd.to_numeric(frame[metric], errors="coerce")).dropna(subset=["_value"])
    values = work.groupby("seed", dropna=False)._value.mean()
    mean = float(values.mean())
    seeded = [float(value) for seed, value in values.items() if str(seed) != "NA"]
    return mean, float(np.std(seeded, ddof=1)) if len(seeded) > 1 else None


def scene_bootstrap(values: pd.DataFrame, column: str, seed: int) -> tuple[float, float]:
    scenes = sorted(values.scene_id.unique())
    grouped = {scene: values.loc[values.scene_id == scene, column].to_numpy() for scene in scenes}
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed)
    draws = np.empty(N_BOOTSTRAP)
    for index in range(N_BOOTSTRAP):
        chosen = rng.choice(scenes, len(scenes), replace=True)
        draws[index] = np.concatenate([grouped[scene] for scene in chosen]).mean()
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def cluster_ci(frame: pd.DataFrame, metric: str, seed: int) -> tuple[float, float]:
    work = frame.assign(value=pd.to_numeric(frame[metric], errors="coerce")).dropna(subset=["value"])
    work = work.groupby(["scene_id", "sample_id"], as_index=False).value.mean()
    return scene_bootstrap(work, "value", seed)


def paired_effect(left: pd.DataFrame, right: pd.DataFrame, metric: str, model: str, comparison: str, seed: int) -> dict[str, object]:
    keys = ["scene_id", "sample_id"]
    l = left.assign(left=pd.to_numeric(left[metric])).groupby(keys, as_index=False).left.mean()
    r = right.assign(right=pd.to_numeric(right[metric])).groupby(keys, as_index=False).right.mean()
    joined = l.merge(r, on=keys, validate="one_to_one")
    joined["effect"] = joined.left - joined.right if metric in HIGHER_IS_BETTER else joined.right - joined.left
    lo, hi = scene_bootstrap(joined, "effect", seed)
    return {
        "model": model, "comparison": comparison, "metric": metric,
        "n_samples": len(joined), "n_scenes": joined.scene_id.nunique(),
        "estimate": float(joined.effect.mean()), "ci_low": lo, "ci_high": hi,
        "direction": "positive_favors_full_or_correct_goal",
    }


def find(frame: pd.DataFrame, **values: object) -> pd.Series:
    result = frame
    for key, value in values.items():
        result = result[result[key] == value]
    if len(result) != 1:
        raise ValueError(f"expected one row for {values}, got {len(result)}")
    return result.iloc[0]


def add_strata(frame: pd.DataFrame, train_rows: list[dict]) -> pd.DataFrame:
    train_goal, train_first = [], []
    for row in train_rows:
        _, distance, rotation = action_components(np.asarray(row["primitive_actions_9d"][0], dtype=np.float64))
        train_goal.append(motion_distance(np.asarray(row["anchor_pose"]), np.asarray(row["final_pose"])))
        train_first.append(distance + 0.5 * rotation)
    goal_edges = np.quantile(train_goal, [1 / 3, 2 / 3])
    first_edges = np.quantile(train_first, [1 / 3, 2 / 3])
    output = frame.copy()
    output["goal_distance_stratum"] = pd.cut(
        pd.to_numeric(output.initial_goal_distance), [-np.inf, *goal_edges, np.inf], labels=("near", "mid", "far")
    )
    output["first_action_magnitude_stratum"] = pd.cut(
        pd.to_numeric(output.first_action_magnitude), [-np.inf, *first_edges, np.inf], labels=("small", "mid", "large")
    )
    output["motion_dominance_stratum"] = output.translation_dominant.map({1: "translation", 0: "rotation"})
    return output


def intervention_rows(frame: pd.DataFrame) -> pd.DataFrame:
    correct = frame[frame.variant == "full_relational"]
    wrong = frame[frame.variant == "full_relational_inference_shuffled_goal"]
    keys = ["model", "seed", "scene_id", "sample_id"]
    joined = correct[keys + ["predicted_action_9d"]].merge(
        wrong[keys + ["predicted_action_9d"]], on=keys, suffixes=("_correct", "_shuffled"), validate="one_to_one"
    )
    rows = []
    for item in joined.itertuples(index=False):
        correct_action = np.asarray(json.loads(item.predicted_action_9d_correct), dtype=np.float64)
        shuffled_action = np.asarray(json.loads(item.predicted_action_9d_shuffled), dtype=np.float64)
        c_direction, c_distance, _ = action_components(correct_action)
        s_direction, s_distance, _ = action_components(shuffled_action)
        cosine = float(np.clip(np.dot(c_direction, s_direction), -1.0, 1.0))
        c_rotation = action9_to_matrix(correct_action)[:3, :3]
        s_rotation = action9_to_matrix(shuffled_action)[:3, :3]
        rows.append({
            "model": item.model, "seed": item.seed, "scene_id": item.scene_id, "sample_id": item.sample_id,
            "prediction_direction_change_deg": math.degrees(math.acos(cosine)),
            "prediction_distance_change": abs(c_distance - s_distance),
            "prediction_rotation_change_deg": math.degrees(rotation_angle(c_rotation, s_rotation)),
        })
    return pd.DataFrame(rows)


def fmt(value: float, sd: float | None, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if sd is None or pd.isna(sd) else f"{value:.{digits}f} ± {sd:.{digits}f}"


def verdict(row: pd.Series) -> str:
    if row.ci_low > 0:
        return "favors full/correct goal"
    if row.ci_high < 0:
        return "favors control/shuffled goal"
    return "interval includes zero"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    parts = []
    for model in MODELS:
        root = args.run_root / model
        integrity = json.loads((root / "training_and_integrity.json").read_text())
        if integrity["status"] != "COMPLETE_VALID" or integrity["model"] != model or integrity["per_sample_rows"] != 8192:
            raise ValueError(f"invalid model integrity: {model}")
        part = pd.read_csv(root / "continuous_action_per_sample.csv", dtype={"seed": str}, keep_default_na=False)
        if len(part) != 8192 or part.sample_id.nunique() != 256:
            raise ValueError(f"unexpected rows for {model}: {len(part)}")
        parts.append(part)
    all_results = pd.concat(parts, ignore_index=True)
    if not np.isfinite(all_results[list(METRICS)].apply(pd.to_numeric).to_numpy()).all():
        raise ValueError("non-finite metric")
    all_results = add_strata(all_results, read_jsonl(args.train_manifest))
    all_results.to_csv(args.output_dir / "continuous_action_per_sample.csv", index=False)

    summaries = []
    for group_index, ((model, variant), frame) in enumerate(all_results.groupby(["model", "variant"], sort=False)):
        row: dict[str, object] = {"model": model, "variant": variant, "n_samples": frame.sample_id.nunique(), "n_scenes": frame.scene_id.nunique()}
        for metric_index, metric in enumerate(METRICS):
            mean, sd = seed_stats(frame, metric)
            lo, hi = cluster_ci(frame, metric, 1000 + group_index * 20 + metric_index)
            row.update({metric: mean, f"{metric}_seed_sd": sd, f"{metric}_ci_low": lo, f"{metric}_ci_high": hi})
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output_dir / "continuous_action_summary.csv", index=False)

    effects = []
    comparisons = (
        ("current_only", "full_relational_minus_current_only"),
        ("goal_only", "full_relational_minus_goal_only"),
        ("full_concat", "full_relational_minus_full_concat"),
        ("mean_motion_prior", "full_relational_minus_motion_prior"),
        ("full_relational_inference_shuffled_goal", "correct_goal_minus_inference_shuffled_goal"),
    )
    for model_index, model in enumerate(MODELS):
        source = all_results[all_results.model == model]
        full = source[source.variant == "full_relational"]
        for comparison_index, (control, name) in enumerate(comparisons):
            for metric_index, metric in enumerate(METRICS):
                effects.append(paired_effect(
                    full, source[source.variant == control], metric, model, name,
                    3000 + model_index * 200 + comparison_index * 20 + metric_index,
                ))
    effect_frame = pd.DataFrame(effects)
    effect_frame.to_csv(args.output_dir / "continuous_action_paired_effects.csv", index=False)

    intervention = intervention_rows(all_results)
    intervention.to_csv(args.output_dir / "continuous_action_goal_intervention.csv", index=False)
    intervention_summary = intervention.groupby("model").agg({
        "prediction_direction_change_deg": "mean", "prediction_distance_change": "mean",
        "prediction_rotation_change_deg": "mean",
    }).reset_index()
    intervention_summary.to_csv(args.output_dir / "continuous_action_goal_intervention_summary.csv", index=False)

    strata_rows = []
    selected = all_results[all_results.variant.isin(("full_relational", "current_only", "goal_only", "full_relational_inference_shuffled_goal"))]
    for column in ("motion_dominance_stratum", "goal_distance_stratum", "first_action_magnitude_stratum"):
        for (model, variant, level), frame in selected.groupby(["model", "variant", column], observed=True):
            row = {"model": model, "variant": variant, "stratum": column, "level": level, "n_samples": frame.sample_id.nunique(), "n_scenes": frame.scene_id.nunique()}
            row.update({metric: pd.to_numeric(frame[metric]).mean() for metric in METRICS})
            strata_rows.append(row)
    pd.DataFrame(strata_rows).to_csv(args.output_dir / "continuous_action_strata.csv", index=False)

    main_rows, control_rows = [], []
    for model in MODELS:
        row = find(summary, model=model, variant="full_relational")
        main_rows.append({
            "model": DISPLAY[model], "direction_error_deg": row.direction_error_deg,
            "direction_error_seed_sd": row.direction_error_deg_seed_sd, "distance_mae": row.distance_mae,
            "distance_mae_seed_sd": row.distance_mae_seed_sd, "rotation_error_deg": row.rotation_error_deg,
            "rotation_error_seed_sd": row.rotation_error_deg_seed_sd, "positive_progress": row.positive_progress,
            "positive_progress_seed_sd": row.positive_progress_seed_sd, "goal_distance_reduction": row.goal_distance_reduction,
            "goal_distance_reduction_seed_sd": row.goal_distance_reduction_seed_sd,
        })
        progress = {variant: find(summary, model=model, variant=variant).positive_progress for variant in (
            "full_relational", "full_concat", "current_only", "goal_only", "mean_motion_prior", "full_relational_inference_shuffled_goal"
        )}
        current_effect = find(effect_frame, model=model, comparison="full_relational_minus_current_only", metric="positive_progress")
        goal_effect = find(effect_frame, model=model, comparison="full_relational_minus_goal_only", metric="positive_progress")
        prior_effect = find(effect_frame, model=model, comparison="full_relational_minus_motion_prior", metric="positive_progress")
        shuffle_effect = find(effect_frame, model=model, comparison="correct_goal_minus_inference_shuffled_goal", metric="positive_progress")
        control_rows.append({"model": DISPLAY[model], **progress,
                             "full_minus_current": current_effect.estimate, "full_minus_current_ci_low": current_effect.ci_low,
                             "full_minus_current_ci_high": current_effect.ci_high,
                             "full_minus_goal": goal_effect.estimate, "full_minus_goal_ci_low": goal_effect.ci_low,
                             "full_minus_goal_ci_high": goal_effect.ci_high,
                             "full_minus_prior": prior_effect.estimate, "full_minus_prior_ci_low": prior_effect.ci_low,
                             "full_minus_prior_ci_high": prior_effect.ci_high,
                             "correct_minus_shuffled": shuffle_effect.estimate,
                             "correct_minus_shuffled_ci_low": shuffle_effect.ci_low, "correct_minus_shuffled_ci_high": shuffle_effect.ci_high})
    main_table = pd.DataFrame(main_rows)
    control_table = pd.DataFrame(control_rows)
    main_table.to_csv(args.output_dir / "TABLE_CONTINUOUS_ACTION_MAIN.csv", index=False)
    control_table.to_csv(args.output_dir / "TABLE_CONTINUOUS_ACTION_CONTROLS.csv", index=False)
    write_new(args.output_dir / "TABLE_CONTINUOUS_ACTION_MAIN.tex", main_table.to_latex(index=False, escape=True, float_format=lambda x: f"{x:.4f}"))
    write_new(args.output_dir / "TABLE_CONTINUOUS_ACTION_CONTROLS.tex", control_table.to_latex(index=False, escape=True, float_format=lambda x: f"{x:.4f}"))

    lines = [
        "# Goal-Conditioned Continuous Action Output — Relational Subset V2",
        "",
        "The frozen evaluation contains 256 trajectories from 108 validation scenes. Results are mean ± SD across five seeds (41–45); intervals use 5,000 scene-cluster paired bootstrap replicates. Validation is not described as a blind test.",
        "",
        "## Main results",
        "",
        "| Model | Direction error ↓ | Distance MAE ↓ | Rotation error ↓ | Positive progress ↑ | Goal reduction ↑ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        row = find(summary, model=model, variant="full_relational")
        lines.append(f"| {DISPLAY[model]} | {fmt(row.direction_error_deg, row.direction_error_deg_seed_sd, 1)}° | {fmt(row.distance_mae, row.distance_mae_seed_sd)} | {fmt(row.rotation_error_deg, row.rotation_error_deg_seed_sd, 1)}° | {fmt(row.positive_progress, row.positive_progress_seed_sd)} | {fmt(row.goal_distance_reduction, row.goal_distance_reduction_seed_sd)} |")
    lines += ["", "## Confirmatory controls", ""]
    for model in MODELS:
        current = find(effect_frame, model=model, comparison="full_relational_minus_current_only", metric="positive_progress")
        goal = find(effect_frame, model=model, comparison="full_relational_minus_goal_only", metric="positive_progress")
        prior = find(effect_frame, model=model, comparison="full_relational_minus_motion_prior", metric="positive_progress")
        shuffled = find(effect_frame, model=model, comparison="correct_goal_minus_inference_shuffled_goal", metric="positive_progress")
        concat = find(effect_frame, model=model, comparison="full_relational_minus_full_concat", metric="positive_progress")
        sensitivity = find(intervention_summary, model=model)
        lines.append(
            f"- **{DISPLAY[model]}:** FULL−CURRENT positive progress {current.estimate:+.3f} "
            f"(95% CI {current.ci_low:+.3f}, {current.ci_high:+.3f}; {verdict(current)}); "
            f"FULL−GOAL {goal.estimate:+.3f}; FULL−PRIOR {prior.estimate:+.3f}; "
            f"correct−shuffled goal {shuffled.estimate:+.3f} (95% CI {shuffled.ci_low:+.3f}, {shuffled.ci_high:+.3f}; {verdict(shuffled)}); "
            f"relational−concat {concat.estimate:+.3f}. Goal shuffling changes predicted direction by "
            f"{sensitivity.prediction_direction_change_deg:.1f}° and rotation by {sensitivity.prediction_rotation_change_deg:.1f}° on average."
        )
    best = summary[summary.variant == "full_relational"].sort_values("positive_progress", ascending=False).iloc[0]
    lines += [
        "", "## Results and interpretation", "",
        f"With no candidate actions provided, the strongest positive-progress rate is {best.positive_progress:.3f} for {DISPLAY[best.model]}. FULL beats GOAL_ONLY in positive progress for all four representations (effects +0.060 to +0.138, with every 95% interval above zero), so the current P8 representation contains information about the next local action beyond the endpoint image alone.",
        "",
        "That result does not establish useful goal conditioning. FULL fails to beat CURRENT_ONLY in positive progress for every model (effects -0.036 to -0.020), and its goal-distance-reduction effect is negative for all four (-0.011 to -0.005). V-JEPA2 is significantly worse than CURRENT_ONLY in positive progress, while DINO, V-JEPA2, and Wan are significantly worse in goal reduction. More decisively, changing only the goal at inference changes the predicted direction by 29.8–45.6 degrees, yet correct-goal versus shuffled-goal effects on positive progress (-0.023 to +0.012) and goal reduction (-0.006 to +0.001) all include zero. The probe reacts to the goal representation, but the reaction is not reliably aligned with reaching the supplied goal.",
        "",
        "The relational head is not a universal remedy. It improves positive progress over concatenation for DINO (+0.026) and Fast3R (+0.032), with intervals above zero, but only Fast3R obtains a clear goal-reduction gain (+0.008). V-JEPA2 and Wan show no reliable utility improvement. Thus the new inductive bias strengthens action readout in selected representations without changing the main negative conclusion about goal-specific control.",
        "",
        "These probes perform supervised readout of the first action on observed four-step trajectories. They do not demonstrate online planning, a native policy, or a backbone world model; differences between backbones cannot be attributed solely to pretraining objectives.",
    ]
    write_new(args.output_dir / "CONTINUOUS_ACTION_RESULTS.md", "\n".join(lines))
    write_new(args.output_dir / "CONTINUOUS_ACTION_INTEGRITY.json", json.dumps({
        "status": "COMPLETE_VALID", "models": list(MODELS), "seeds": [41, 42, 43, 44, 45],
        "bootstrap_replicates": N_BOOTSTRAP, "bootstrap_seed": BOOTSTRAP_SEED,
        "row_counts": {"per_sample": len(all_results), "summary": len(summary), "paired_effects": len(effect_frame), "intervention": len(intervention)},
        "source_integrity_sha256": {model: sha256(args.run_root / model / "training_and_integrity.json") for model in MODELS},
    }, indent=2, sort_keys=True))
    print(json.dumps({"status": "COMPLETE_VALID", "per_sample": len(all_results), "effects": len(effect_frame)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
