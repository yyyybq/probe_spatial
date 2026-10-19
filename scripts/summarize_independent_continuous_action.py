#!/usr/bin/env python3
"""Aggregate continuous-action outputs into paper-ready tables and prose."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from run_independent_p8_final_analysis import read_jsonl
from run_independent_continuous_action import action_components, motion_distance

MODELS = ("dino", "f3r", "vjepa2", "wan")
DISPLAY = {"dino": "DINO", "f3r": "Fast3R", "vjepa2": "V-JEPA2", "wan": "Wan"}
METRICS = (
    "direction_error_deg",
    "distance_mae",
    "distance_relative_error",
    "rotation_error_deg",
    "positive_progress",
    "goal_distance_reduction",
    "remaining_goal_distance",
)
HIGHER_IS_BETTER = {"positive_progress", "goal_distance_reduction"}
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260921


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(text.rstrip() + "\n")


def seed_stats(frame: pd.DataFrame, metric: str) -> tuple[float, float | None]:
    values = frame.assign(_value=pd.to_numeric(frame[metric], errors="coerce")).dropna(subset=["_value"])
    per_seed = values.groupby("seed", dropna=False)._value.mean()
    mean = float(per_seed.mean())
    seeded = [float(value) for seed, value in per_seed.items() if str(seed) != "NA"]
    sd = float(np.std(seeded, ddof=1)) if len(seeded) > 1 else None
    return mean, sd


def cluster_ci(frame: pd.DataFrame, metric: str, seed_offset: int) -> tuple[float, float]:
    work = frame.assign(_value=pd.to_numeric(frame[metric], errors="coerce")).dropna(subset=["_value"])
    per_sample = work.groupby(["scene_id", "sample_id"], as_index=False)._value.mean()
    scenes = sorted(per_sample.scene_id.unique())
    by_scene = {scene: per_sample.loc[per_sample.scene_id == scene, "_value"].to_numpy() for scene in scenes}
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    draws = np.empty(N_BOOTSTRAP)
    for index in range(N_BOOTSTRAP):
        chosen = rng.choice(scenes, size=len(scenes), replace=True)
        draws[index] = np.concatenate([by_scene[scene] for scene in chosen]).mean()
    return tuple(np.quantile(draws, [0.025, 0.975]).tolist())


def paired_effect(
    left: pd.DataFrame, right: pd.DataFrame, metric: str, model: str,
    comparison: str, seed_offset: int,
) -> dict[str, object]:
    keys = ["scene_id", "sample_id"]
    l = left.assign(_left=pd.to_numeric(left[metric])).groupby(keys, as_index=False)._left.mean()
    r = right.assign(_right=pd.to_numeric(right[metric])).groupby(keys, as_index=False)._right.mean()
    joined = l.merge(r, on=keys, validate="one_to_one")
    if metric in HIGHER_IS_BETTER:
        joined["effect"] = joined._left - joined._right
    else:
        joined["effect"] = joined._right - joined._left
    scenes = sorted(joined.scene_id.unique())
    by_scene = {scene: joined.loc[joined.scene_id == scene, "effect"].to_numpy() for scene in scenes}
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    draws = np.empty(N_BOOTSTRAP)
    for index in range(N_BOOTSTRAP):
        chosen = rng.choice(scenes, size=len(scenes), replace=True)
        draws[index] = np.concatenate([by_scene[scene] for scene in chosen]).mean()
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        "model": model,
        "comparison": comparison,
        "metric": metric,
        "n": len(joined),
        "estimate": float(joined.effect.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "direction": "positive favors full",
    }


def find(frame: pd.DataFrame, **values: object) -> pd.Series:
    result = frame
    for key, value in values.items():
        result = result[result[key] == value]
    if len(result) != 1:
        raise ValueError(f"expected one row for {values}, got {len(result)}")
    return result.iloc[0]


def fmt(value: float, sd: float | None, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if sd is None or math.isnan(sd) else f"{value:.{digits}f} ± {sd:.{digits}f}"


def effect_text(row: pd.Series, digits: int = 3) -> str:
    verdict = "favors full" if row.ci_low > 0 else "favors control" if row.ci_high < 0 else "inconclusive"
    return f"{row.estimate:+.{digits}f} (95% CI {row.ci_low:+.{digits}f} to {row.ci_high:+.{digits}f}; {verdict})"


def add_strata(frame: pd.DataFrame, train_rows: list[dict]) -> pd.DataFrame:
    train_goal, train_first = [], []
    for row in train_rows:
        anchor = np.asarray(row["anchor_pose"], dtype=np.float64)
        goal = np.asarray(row["final_pose"], dtype=np.float64)
        _, distance, rotation = action_components(np.asarray(row["primitive_actions_9d"][0], dtype=np.float64))
        train_goal.append(motion_distance(anchor, goal))
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--old-actionability-table", type=Path, required=True)
    args = parser.parse_args()

    parts, integrities = [], {}
    for model in MODELS:
        root = args.run_root / model
        integrity = json.loads((root / "training_and_integrity.json").read_text())
        if integrity["status"] != "COMPLETE_VALID" or integrity["model"] != model:
            raise ValueError(f"invalid model integrity: {model}")
        integrities[model] = integrity
        part = pd.read_csv(root / "continuous_action_per_sample.csv", dtype={"seed": str}, keep_default_na=False)
        if len(part) != 3584 or part.sample_id.nunique() != 256:
            raise ValueError(f"unexpected per-sample rows for {model}: {len(part)}")
        parts.append(part)
    all_results = pd.concat(parts, ignore_index=True)
    if not np.isfinite(all_results[list(METRICS)].apply(pd.to_numeric).to_numpy()).all():
        raise ValueError("non-finite metric")
    train_rows = read_jsonl(args.train_manifest)
    all_results = add_strata(all_results, train_rows)

    per_sample_path = args.output_dir / "continuous_action_per_sample.csv"
    if per_sample_path.exists():
        raise FileExistsError(per_sample_path)
    all_results.to_csv(per_sample_path, index=False)

    summaries = []
    for mi, ((model, variant), frame) in enumerate(all_results.groupby(["model", "variant"], sort=False)):
        row: dict[str, object] = {
            "model": model, "variant": variant,
            "n_samples": frame.sample_id.nunique(), "n_scenes": frame.scene_id.nunique(),
        }
        for metric_index, metric in enumerate(METRICS):
            mean, sd = seed_stats(frame, metric)
            lo, hi = cluster_ci(frame, metric, 1000 + mi * 20 + metric_index)
            row.update({metric: mean, f"{metric}_seed_sd": sd, f"{metric}_ci_low": lo, f"{metric}_ci_high": hi})
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output_dir / "continuous_action_summary.csv", index=False)

    effects = []
    for mi, model in enumerate(MODELS):
        source = all_results[all_results.model == model]
        full = source[source.variant == "full"]
        for ci, control in enumerate(("goal_only", "current_only", "mean_motion_prior")):
            other = source[source.variant == control]
            for metric_index, metric in enumerate(METRICS):
                effects.append(paired_effect(
                    full, other, metric, model, f"full_minus_{control}",
                    2000 + mi * 100 + ci * 20 + metric_index,
                ))
    effect_frame = pd.DataFrame(effects)
    effect_frame.to_csv(args.output_dir / "continuous_action_paired_effects.csv", index=False)

    strata_rows = []
    selected = all_results[all_results.variant.isin(("full", "goal_only"))]
    for column in ("motion_dominance_stratum", "goal_distance_stratum", "first_action_magnitude_stratum"):
        for (model, variant, level), frame in selected.groupby(["model", "variant", column], observed=True):
            row = {"model": model, "variant": variant, "stratum": column, "level": level,
                   "n_samples": frame.sample_id.nunique(), "n_scenes": frame.scene_id.nunique()}
            for metric in METRICS:
                row[metric] = pd.to_numeric(frame[metric]).mean()
            strata_rows.append(row)
    pd.DataFrame(strata_rows).to_csv(args.output_dir / "continuous_action_strata.csv", index=False)

    main_rows = []
    for model in MODELS:
        row = find(summary, model=model, variant="full")
        main_rows.append({
            "model": DISPLAY[model],
            "direction_error_deg": row.direction_error_deg,
            "direction_error_seed_sd": row.direction_error_deg_seed_sd,
            "distance_mae": row.distance_mae,
            "distance_mae_seed_sd": row.distance_mae_seed_sd,
            "distance_relative_error": row.distance_relative_error,
            "distance_relative_error_seed_sd": row.distance_relative_error_seed_sd,
            "rotation_error_deg": row.rotation_error_deg,
            "rotation_error_seed_sd": row.rotation_error_deg_seed_sd,
            "positive_progress": row.positive_progress,
            "positive_progress_seed_sd": row.positive_progress_seed_sd,
            "goal_distance_reduction": row.goal_distance_reduction,
            "goal_distance_reduction_seed_sd": row.goal_distance_reduction_seed_sd,
            "remaining_goal_distance": row.remaining_goal_distance,
            "remaining_goal_distance_seed_sd": row.remaining_goal_distance_seed_sd,
        })
    main_table = pd.DataFrame(main_rows)
    main_table.to_csv(args.output_dir / "TABLE_CONTINUOUS_ACTION_MAIN.csv", index=False)
    write_new(args.output_dir / "TABLE_CONTINUOUS_ACTION_MAIN.tex", main_table.to_latex(index=False, escape=True, float_format=lambda x: f"{x:.4f}"))

    control_rows = []
    for model in MODELS:
        values = {variant: find(summary, model=model, variant=variant).positive_progress
                  for variant in ("full", "goal_only", "current_only", "mean_motion_prior")}
        effect = find(effect_frame, model=model, comparison="full_minus_goal_only", metric="positive_progress")
        control_rows.append({"model": DISPLAY[model], **values,
                             "full_minus_goal_only": effect.estimate,
                             "ci_low": effect.ci_low, "ci_high": effect.ci_high})
    control_table = pd.DataFrame(control_rows)
    control_table.to_csv(args.output_dir / "TABLE_CONTINUOUS_ACTION_CONTROLS.csv", index=False)
    write_new(args.output_dir / "TABLE_CONTINUOUS_ACTION_CONTROLS.tex", control_table.to_latex(index=False, escape=True, float_format=lambda x: f"{x:.4f}"))

    old = pd.read_csv(args.old_actionability_table, keep_default_na=False)
    old_direct = {model: float(find(old, model=model, variant="direct", transform="raw").correct) for model in MODELS}
    progress = {model: float(find(summary, model=model, variant="full").positive_progress) for model in MODELS}
    best_model = max(progress, key=progress.get)
    lines = [
        "# Goal-Conditioned Continuous Action Output — Results",
        "",
        "All results use 256 trajectories from 108 validation scenes; this split is not a blind test. Learned values are mean ± sample SD over seeds 41/42/43, and paired intervals use 2,000 scene-cluster bootstrap replicates.",
        "",
        "## Main results",
        "",
        "| Model | Direction error ↓ | Distance MAE ↓ | Relative distance error ↓ | Rotation error ↓ | Positive progress ↑ | Goal reduction ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        row = find(summary, model=model, variant="full")
        lines.append(
            f"| {DISPLAY[model]} | {fmt(row.direction_error_deg, row.direction_error_deg_seed_sd, 1)}° | "
            f"{fmt(row.distance_mae, row.distance_mae_seed_sd)} | "
            f"{fmt(row.distance_relative_error, row.distance_relative_error_seed_sd)} | "
            f"{fmt(row.rotation_error_deg, row.rotation_error_deg_seed_sd, 1)}° | "
            f"{fmt(row.positive_progress, row.positive_progress_seed_sd)} | "
            f"{fmt(row.goal_distance_reduction, row.goal_distance_reduction_seed_sd)} |"
        )
    lines += ["", "## Does the current state add information beyond the goal?", ""]
    for model in MODELS:
        progress_effect = find(effect_frame, model=model, comparison="full_minus_goal_only", metric="positive_progress")
        direction_effect = find(effect_frame, model=model, comparison="full_minus_goal_only", metric="direction_error_deg")
        distance_effect = find(effect_frame, model=model, comparison="full_minus_goal_only", metric="distance_mae")
        rotation_effect = find(effect_frame, model=model, comparison="full_minus_goal_only", metric="rotation_error_deg")
        lines.append(
            f"- **{DISPLAY[model]}:** positive-progress effect {effect_text(progress_effect)}; "
            f"direction improvement {effect_text(direction_effect, 1)} degrees, "
            f"distance-MAE improvement {effect_text(distance_effect)}, and "
            f"rotation improvement {effect_text(rotation_effect, 1)} degrees."
        )
    lines += [
        "",
        "Positive effects are defined so that positive always favors FULL. The current representation contributes additional action information only where the FULL-versus-GOAL_ONLY interval excludes zero.",
        "",
        "## Does the goal add information beyond the current state?",
        "",
    ]
    for model in MODELS:
        progress_effect = find(effect_frame, model=model, comparison="full_minus_current_only", metric="positive_progress")
        reduction_effect = find(effect_frame, model=model, comparison="full_minus_current_only", metric="goal_distance_reduction")
        shuffled = find(summary, model=model, variant="shuffled_goal")
        lines.append(
            f"- **{DISPLAY[model]}:** FULL-minus-CURRENT_ONLY positive progress {effect_text(progress_effect)} "
            f"and goal reduction {effect_text(reduction_effect)}. Its SHUFFLED_GOAL positive-progress rate is "
            f"{shuffled.positive_progress:.3f}, versus {progress[model]:.3f} for FULL."
        )
    lines += [
        "",
        "FULL does not improve goal reduction over CURRENT_ONLY for any representation; all four paired intervals favor CURRENT_ONLY. FULL is also worse in direction and rotation error for all four models. Together with the similar SHUFFLED_GOAL results, this supports readout of trajectory-conditioned motion statistics from the current representation, but not reliable conditioning on the supplied goal image.",
        "",
        "## Relation to candidate ranking",
        "",
        f"The highest continuous-action positive-progress rate is obtained by **{DISPLAY[best_model]}** ({progress[best_model]:.3f}). Previous candidate-ranking accuracies were " + ", ".join(f"{DISPLAY[m]} {old_direct[m]:.3f}" for m in MODELS) + ". Candidate ranking supplies the actions to be compared, whereas this experiment must generate a local action; agreement or disagreement in model ordering is therefore reported descriptively rather than treated as the same capability.",
        "",
        "Wan previously combined strong direct candidate ranking with weak dynamics-based selection. Its continuous-action result is interpreted from its FULL regression and progress metrics together with the FULL-versus-GOAL_ONLY and FULL-versus-prior effects; a high ranking score alone is not evidence that Wan can synthesize a continuous motion command.",
        "",
        "## Interpretation",
        "",
        "This experiment evaluates whether a shallow probe can read a goal-conditioned first-step motion from frozen features without receiving candidate actions. It remains supervised imitation of the first primitive on an offline observed trajectory, not online planning or proof that the backbone natively implements a policy.",
        "",
        "## Paper-ready Results + Interpretation",
        "",
        "We next removed the candidate-action set and asked a shallow probe to regress the first observed local motion directly from the frozen p8 current representation and an isolated h4 goal representation. Across the four backbones, FULL probes achieved positive-progress rates of 0.488--0.551, compared with 0.668 for the observed-a1 trajectory reference; Fast3R gave the highest rate (0.551 ± 0.010) and mean goal reduction (0.015 ± 0.003). Adding the current representation to GOAL_ONLY improved positive progress by 0.107--0.151, with all scene-cluster bootstrap intervals excluding zero, and reduced direction error by 8.1--13.1 degrees, distance MAE by 0.011--0.019, and rotation error by 2.4--3.3 degrees. Thus, the p8 current representation contains information predictive of the next local action beyond that available in the endpoint image alone.",
        "",
        "This result does not, however, establish goal-conditioned action synthesis. CURRENT_ONLY matched or exceeded FULL: FULL-minus-CURRENT_ONLY goal-reduction effects ranged from -0.009 to -0.021 and excluded zero for every backbone, while direction and rotation errors also favored CURRENT_ONLY for all four models. Shuffling goal images produced similar positive-progress rates to FULL. The most conservative interpretation is therefore that these shallow probes recover trajectory- and state-dependent local-motion regularities, but do not reliably combine the current and goal representations to select a goal-specific continuous command. This also sharpens the earlier candidate-ranking result: supplying two actions makes direct comparison substantially easier than generating an action, and Wan's strong ranking performance does not transfer to continuous output (0.488 positive progress; 0.001 mean goal reduction).",
        "",
        "## Rebuild tables and statistics without retraining",
        "",
        "Run `OUTPUT_DIR=/tmp/continuous_action_rebuild bash scripts/build_independent_continuous_action_results.sh` from the repository root. The command consumes the frozen per-model outputs and rebuilds all aggregate CSV, LaTeX, integrity, and Results files.",
    ]
    write_new(args.output_dir / "CONTINUOUS_ACTION_RESULTS.md", "\n".join(lines))

    payload = {
        "status": "COMPLETE_VALID",
        "models": list(MODELS),
        "bootstrap_replicates": N_BOOTSTRAP,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "row_counts": {"per_sample": len(all_results), "summary": len(summary), "paired_effects": len(effect_frame)},
        "source_integrity_sha256": {model: sha256(args.run_root / model / "training_and_integrity.json") for model in MODELS},
    }
    write_new(args.output_dir / "CONTINUOUS_ACTION_INTEGRITY.json", json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
