#!/usr/bin/env python3
"""Build the read-only global-restart analysis artifacts.

The script reads the frozen registry/canonical tables and supplemental audit
reports.  It writes only the five requested global-restart deliverables and
asserts that every protected source-of-truth file is byte-identical afterward.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "experiments/registry.csv"
CANONICAL_DIR = ROOT / "results/canonical"
OUT_DIR = ROOT / "results/analysis/global_restart"
REPORT = ROOT / "reports/probing_spatial_global_restart_20260827.md"

CANONICAL_FILES = [
    "all_results.csv",
    "a1_results.csv",
    "a2_results.csv",
    "b2_results.csv",
    "c1_results.csv",
    "c2_results.csv",
    "c3_results.csv",
]

MODEL_META = {
    "dino": ("image_self_supervised", "self_distillation", "DINO"),
    "f3r": ("geometry_specialist", "multiview_3d", "F3R"),
    "vjepa2": ("video_representation", "predictive", "V-JEPA2"),
    "videomae": ("video_representation", "masked_reconstruction", "VideoMAE"),
    "qwen2_5_vl": ("mllm", "vision_language", "Qwen2.5-VL"),
    "wan": ("video_generation", "diffusion_or_flow_generation", "Wan"),
    "cogvideox": ("video_generation", "diffusion_or_flow_generation", "CogVideoX"),
    "aether": ("video_generation", "goal_or_video_generation", "Aether"),
    "minimax_h3": ("video_generation", "diffusion_or_flow_generation", "MiniMax-H3"),
    "lingbot": ("video_generation", "diffusion_or_flow_generation", "LingBot"),
}

MODEL_LAYER_GRIDS = {
    "dino": [0],
    "f3r": [24],
    "vjepa2": [0, 5, 11, 17, 23],
    "videomae": [0, 5, 11, 17, 23],
    "qwen2_5_vl": [-1],
    "wan": [0, 5, 10, 15, 20, 25, 29],
    "cogvideox": [0, 5, 10, 15, 20, 25, 30, 35, 40],
    "aether": [1],
    "minimax_h3": [10, 20, 30, 40, 50],
    "lingbot": [10, 20, 30, 40, 48],
}

TASK_META = {
    "streaming_depth/depth": (
        "streaming_depth",
        "direct metric-depth readout",
        "streaming",
        "ground_truth_depth",
        "unconditioned_scene_readout",
    ),
    "streaming_depth/camera": (
        "streaming_depth",
        "camera-pose/geometric readout",
        "streaming",
        "ground_truth_camera_pose",
        "unconditioned_scene_readout",
    ),
    "streaming_depth/mask": (
        "streaming_depth",
        "pixel/instance identity readout from the joint A1 head",
        "streaming",
        "ground_truth_identity_mask",
        "unconditioned_scene_readout",
    ),
    "streaming_mask/mask": (
        "streaming_mask",
        "pixel/instance identity readout from the mask-only head",
        "streaming",
        "ground_truth_identity_mask",
        "unconditioned_scene_readout",
    ),
    "view_consistency": (
        "view_consistency",
        "whether two views overlap",
        "streaming",
        "ground_truth_overlap",
        "view_pair",
    ),
    "ego_belief": (
        "ego_belief",
        "legacy masked hidden-object bearing",
        "streaming",
        "ground_truth_object_bearing",
        "object_mask_or_query_conditioned",
    ),
    "ego_belief_v2": (
        "ego_belief_v2",
        "object-query hidden-location recovery",
        "streaming",
        "ground_truth_object_bearing_distance",
        "object_query_conditioned",
    ),
    "action_dynamics": (
        "action_dynamics",
        "one-step action-conditioned isolated-target feature retrieval",
        "streaming",
        "target",
        "relative_camera_action",
    ),
    "path_integration": (
        "path_integration",
        "multi-step action-sequence target-feature rollout",
        "streaming",
        "target",
        "relative_camera_action_sequence",
    ),
    "counterfactual": (
        "counterfactual",
        "alternative-action intervention consistency",
        "streaming",
        "target",
        "counterfactual_camera_action",
    ),
}

METRIC_DIRECTION = {
    "mean_loss_depth": "lower_is_better",
    "Auc_30": "higher_is_better",
    "mean_loss_identity": "lower_is_better",
    "balanced_acc": "higher_is_better",
    "mean_az_err_deg": "lower_is_better",
    "mean_ang_err_deg": "lower_is_better",
    "global_R@1": "higher_is_better",
    "intervention_validity": "higher_is_better",
}

CORE_MODELS = {
    "dino",
    "f3r",
    "vjepa2",
    "videomae",
    "qwen2_5_vl",
    "wan",
    "cogvideox",
    "aether",
}
CORE_VARIANTS = {"default_streaming", "c_retrain_with_extra"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row_count(path: Path) -> int:
    return max(0, sum(1 for _ in path.open(encoding="utf-8")) - 1)


def f(value: str) -> float:
    return float(value)


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def config_timestep(config: str, model: str) -> tuple[str, str]:
    if MODEL_META[model][0] != "video_generation":
        return "NA_non_diffusion", "model_family"
    path = ROOT / config
    if not path.is_file():
        return "UNKNOWN", "missing_config"
    match = re.search(r"(?m)^feature_timestep:\s*([^\s#]+)", path.read_text(errors="replace"))
    if not match:
        return "UNKNOWN", str(path.relative_to(ROOT))
    return match.group(1), str(path.relative_to(ROOT))


def layer_position(model: str, layer_text: str) -> tuple[str, str]:
    grid = MODEL_LAYER_GRIDS[model]
    layer = int(layer_text)
    if len(grid) == 1:
        return "single_registered_layer", "NA"
    if layer == min(grid):
        label = "first_registered"
    elif layer == max(grid):
        label = "last_registered"
    else:
        label = "intermediate_registered"
    fraction = (layer - min(grid)) / (max(grid) - min(grid))
    return label, f"{fraction:.6f}"


def campaign_cohort(variant: str) -> str:
    return {
        "default_streaming": "default_core",
        "c_retrain_with_extra": "fixed_c_default_plus_extra",
        "eval_20260817": "multilayer_gapfill_eval_20260817",
        "wan_cog_layer_sweep": "historical_wan_cog_formal_only",
    }.get(variant, variant)


def matrix_eligibility(row: dict[str, str], collision_size: int) -> str:
    if row["model"] in CORE_MODELS and row["model_variant"] in CORE_VARIANTS:
        return "core_default_comparable"
    if collision_size > 1:
        return "campaign_separated_collision_do_not_pool"
    if row["family"] == "B1":
        return "supplemental_legacy_b1"
    return "campaign_separated_layer_analysis"


def build_matrix(complete: list[dict[str, str]]) -> tuple[list[dict[str, object]], int]:
    collision_groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in complete:
        key = (
            row["family"],
            row["probe"],
            row["model"],
            row["layer"],
            row["prefix"],
            row["metric_name"],
        )
        collision_groups[key].append(row)

    matrix: list[dict[str, object]] = []
    for row in complete:
        task, question, input_type, target_type, condition = TASK_META[row["probe"]]
        model_family, objective, display = MODEL_META[row["model"]]
        timestep, timestep_source = config_timestep(row["config"], row["model"])
        position, fraction = layer_position(row["model"], row["layer"])
        collision_key = "|".join(
            [
                row["family"],
                row["probe"],
                row["model"],
                row["layer"],
                row["prefix"],
                row["metric_name"],
            ]
        )
        collision_size = len(
            collision_groups[
                (
                    row["family"],
                    row["probe"],
                    row["model"],
                    row["layer"],
                    row["prefix"],
                    row["metric_name"],
                )
            ]
        )
        feature_type = "streaming+target" if target_type == "target" else "streaming"
        matrix.append(
            {
                "experiment_id": row["experiment_id"],
                "canonical_granularity": "atomic_primary_metric",
                "family": row["family"],
                "task": task,
                "probe": row["probe"],
                "capability_question": question,
                "model": row["model"],
                "model_display": display,
                "model_family": model_family,
                "objective_family": objective,
                "backend": row["backend"],
                "model_variant": row["model_variant"],
                "campaign_cohort": campaign_cohort(row["model_variant"]),
                "layer": row["layer"],
                "layer_position": position,
                "layer_fraction_registered_grid": fraction,
                "streaming_prefix": row["prefix"],
                "horizon": row["horizon"],
                "diffusion_step_noise_level": timestep,
                "diffusion_step_source": timestep_source,
                "feature_type": feature_type,
                "input_feature_type": input_type,
                "target_feature_type": target_type,
                "probe_output_type": "prediction" if target_type == "target" else "task_readout",
                "condition": condition,
                "metric_name": row["metric_name"],
                "metric_direction": METRIC_DIRECTION[row["metric_name"]],
                "metric_value": row["metric_value"],
                "seed": row["seed"],
                "train_split": row["train_split"],
                "eval_split": row["eval_split"],
                "dataset_manifest": row["dataset_manifest"],
                "config": row["config"],
                "checkpoint": row["checkpoint"],
                "launcher_job_id": row["launcher_job_id"],
                "raw_artifact_path": row["raw_artifact_path"],
                "canonical_metric_path": row["canonical_metric_path"],
                "status": row["status"],
                "scientific_config_collision_key": collision_key,
                "scientific_config_collision_group_size": collision_size,
                "scientific_config_collision": "YES" if collision_size > 1 else "NO",
                "analysis_eligibility": matrix_eligibility(row, collision_size),
            }
        )
    collision_count = sum(len(rows) > 1 for rows in collision_groups.values())
    return matrix, collision_count


def core_task_rows(
    complete: list[dict[str, str]], family: str, probe: str, metric: str
) -> dict[str, dict[int, float]]:
    result: dict[str, dict[int, float]] = {}
    for model in CORE_MODELS:
        rows = [
            row
            for row in complete
            if row["model"] == model
            and row["model_variant"] in CORE_VARIANTS
            and row["family"] == family
            and row["probe"] == probe
            and row["metric_name"] == metric
        ]
        if rows:
            result[model] = {int(row["prefix"]): f(row["metric_value"]) for row in rows}
    return result


def layer_peak_counts(complete: list[dict[str, str]]) -> Counter[str]:
    rows = [row for row in complete if row["model_variant"] == "eval_20260817"]
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                row["family"],
                row["probe"],
                row["metric_name"],
                row["model"],
                row["prefix"],
            )
        ].append(row)
    counts: Counter[str] = Counter()
    for key, group in groups.items():
        layers = sorted({int(row["layer"]) for row in group})
        if len(layers) < 3 or len(group) != len(layers):
            continue
        values = {int(row["layer"]): f(row["metric_value"]) for row in group}
        higher = METRIC_DIRECTION[key[2]] == "higher_is_better"
        best = (max if higher else min)(layers, key=lambda layer: values[layer])
        if best == layers[0]:
            counts["first"] += 1
        elif best == layers[-1]:
            counts["last"] += 1
        else:
            counts["intermediate"] += 1
    return counts


def tstep_inventory() -> dict[str, object]:
    """Recompute exact feature readiness from the local completion indexes."""
    model_steps = {
        "wan": (499, 999),
        "cogvideox": (499, 999),
        "minimax_h3": (500, 1000),
    }
    layers = [15, 20, 25]
    prefixes = [8, 12, 16, 24]
    exclusions_path = ROOT / "configs/experiment/inscene15k_streaming/tstep_probe_exclusions.txt"
    exclusions = {
        line.strip()
        for line in exclusions_path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    expected: dict[int, tuple[set[str], set[str]]] = {}
    for prefix in prefixes:
        units: set[str] = set()
        scenes: set[str] = set()
        manifest = ROOT / f"manifests/c_retrain_20260722/inscene_c_h124_prefix{prefix}.jsonl"
        for line in manifest.open(encoding="utf-8"):
            row = json.loads(line)
            scene = str(row.get("scene_id") or Path(str(row["scene_dir"])).name)
            units.add(
                f"{scene}/window_{int(row['streaming_window_id']):04d}/"
                f"prefix_{int(row['streaming_tail']):06d}"
            )
            scenes.add(scene)
        expected[prefix] = (units - exclusions, scenes)

    done = ROOT / "logs/tstep_resume4/aoss_done"
    target_index_root = ROOT.parent / "InsScene-15K/FEAT_TARGET"
    inventory: dict[str, object] = {}
    for model, steps in model_steps.items():
        all_scenes = set().union(*(scenes for _, scenes in expected.values()))
        target_indices = {
            scene
            for scene in all_scenes
            if (target_index_root / model / "scannetpp" / scene / "target_indices.npy").is_file()
        }
        for timestep in steps:
            stream_by_layer: dict[int, set[str]] = defaultdict(set)
            target_by_layer: dict[int, set[str]] = defaultdict(set)
            pattern = re.compile(rf"feature_t{timestep}_layer(\d+)\.sft$")
            for path, destination in [
                (done / f"FEAT_STREAMING_{model}_t{timestep}.txt", stream_by_layer),
                (done / f"FEAT_TARGET_{model}_t{timestep}.txt", target_by_layer),
            ]:
                if not path.is_file():
                    continue
                for line in path.open(errors="replace"):
                    rel = line.strip()
                    match = pattern.search(rel.rsplit("/", 1)[-1])
                    if match and int(match.group(1)) in layers and "/" in rel:
                        destination[int(match.group(1))].add(rel.rsplit("/", 1)[0])
            statuses: Counter[str] = Counter()
            found_ranges: list[int] = []
            target_ranges: list[int] = []
            for layer in layers:
                for prefix in prefixes:
                    units, scenes = expected[prefix]
                    stream_found = len(units & stream_by_layer[layer])
                    target_found = len(scenes & target_by_layer[layer])
                    found_ranges.append(stream_found)
                    target_ranges.append(target_found)
                    stream_ready = stream_found == len(units)
                    target_ready = target_found == len(scenes) and len(target_indices) == len(scenes)
                    if stream_ready and target_ready:
                        statuses["ready_full"] += 1
                    elif stream_ready:
                        statuses["ready_streaming_only"] += 1
                    else:
                        statuses["wait_streaming"] += 1
            inventory[f"{model}_t{timestep}"] = {
                "groups": 12,
                "statuses": dict(statuses),
                "stream_found_min": min(found_ranges),
                "stream_found_max": max(found_ranges),
                "target_found_min": min(target_ranges),
                "target_found_max": max(target_ranges),
                "target_indices": len(target_indices),
                "expected_scenes": len(all_scenes),
            }
    return inventory


def build_planned_rows() -> list[dict[str, object]]:
    rows = [
        ("P01", "canonical target-eval gap fill", "completed", "existing_results", "428/428 unique eval job keys; Wan p8 C1 six missing layers filled", "reports/wan_p8_canonical_clean_rerun_wan_p8_action_dynamics_canonical_manifest_exact_20260821_200604_2075f3e.md", "job-level completion; not the same as 761 atomic canonical rows"),
        ("P02", "canonical integrity and global matrix", "completed_now", "existing_results", "955 registry rows; 761 COMPLETE primary rows mapped into experiment_matrix.csv", "experiments/registry.csv;results/canonical/all_results.csv", "21 provenance-distinct scientific-config collisions are flagged, not pooled"),
        ("P03", "core streaming-prefix sweep", "completed_core_partial_extended", "existing_results", "8 core models x 4 prefixes complete for A1/A2/C1/C2/C3; extended models are asymmetric", "results/canonical/all_results.csv", "B2 is not complete on the same core grid"),
        ("P04", "layer sweep", "partially_completed", "existing_results", "multilayer canonical rows exist for Wan/CogVideoX/V-JEPA2/VideoMAE/MiniMax-H3/LingBot", "experiments/registry.csv", "campaign/protocol collisions and holes prevent one monolithic layer leaderboard"),
        ("P05", "diffusion-step/noise analysis", "features_partial_no_probe", "existing_features_need_probe_eval", "Wan t499/t999 full 3-layer x 4-prefix streaming+target feature groups; MiniMax streaming ready; no tstep probe results", "logs/tstep_resume4/aoss_done/;scripts/build_tstep_probe_matrix.py", "CogVideoX remains feature-incomplete"),
        ("P06", "shuffled-frame-order random baseline", "partial_supplemental", "construct_audit_then_eval_or_extract", "legacy A3 exists; selected B2 p12 sample audit exists; no streaming canonical all-task baseline", "README.md;analyses/probe_validity/PROBE_VALIDITY_REPORT.md", "post-feature permutation and pre-backbone frame shuffle answer different constructs"),
        ("P07", "without-condition baseline", "partial_noncanonical", "existing_features_need_probe_eval", "legacy C eval interventions and a DINO SE(3) MVP trained no-condition control exist; no fair global canonical baseline", "reports/paper_validity_20260821T200224Z_2075f3e/FINAL_REPORT.md;results/experimental/se3_latent_mvp_dino_l0_p8_seed42_v3/REPORT.md", "eval-null action is not a trained no-condition head"),
        ("P08", "A-series from prediction features", "not_started_definition_missing", "requires_feature_extraction", "no canonical prediction-feature row and no prediction-feature extraction mode found", "features/run_inscene15k.py;results/canonical/all_results.csv", "target_isolated feature is not a model prediction feature"),
        ("P09", "layer x diffusion-step x prefix analysis cube", "not_started_features_partial", "mixed_feature_and_probe_work", "cube tooling is absent; only parts of the feature grid are ready", "scripts/build_tstep_probe_matrix.py;logs/tstep_resume4/aoss_done/", "run a small gated slice before any full cube"),
        ("P10", "planning / target-conditioned action output (INTACT reference)", "not_started_construct_undefined", "construct_validity_first", "no INTACT literal, planning probe, manifest, or metric found; only Aether pipeline internal planning mode exists", "features/aether/aether_feature.py;features/aether/aether_feature_denoise.py", "must prevent target/candidate leakage and define geometric action-output metrics"),
        ("P11", "demo", "not_started_project_level", "existing_results_then_small_engineering", "generic viser/radar utilities exist but no global probing demo artifact", "scripts/viser_view.py;scripts/plot_radar.py", "defer interactive demo until analysis cohort is frozen"),
        ("P12", "line plots and cross-model visualization", "partially_completed_stale_scope", "existing_results", "C1/C2 prefix curves, difficulty curves, and task-correlation figures exist for the earlier core snapshot", "figures/paper_validity_20260821T200224Z_2075f3e/", "needs regeneration from current 761-row canonical with campaign separation"),
        ("P13", "C shortcut and construct-validity controls", "partial_selected_complete", "construct_validity_first", "paper-validity controls incomplete globally; cross-objective p12 audit and Wan p8 audit complete", "analyses/probe_validity/PROBE_VALIDITY_REPORT.md;reports/wan_p8_action_dynamics_construct_validity.md", "do not expand Wan action-validity"),
        ("P14", "B2 strict hidden-object construct", "partial_selected_audit", "construct_validity_first", "selected p12 content/query audit complete; strict-zero-hidden train/eval remains absent", "analyses/probe_validity/PROBE_VALIDITY_REPORT.md;reports/paper_validity_20260821T200224Z_2075f3e/FINAL_REPORT.md", "current standard B2 can reflect scene/query/visibility priors"),
        ("P15", "held-out test and multi-seed confirmation", "not_started", "existing_features_need_probe_eval", "all 761 canonical primary rows are seed 42 on val", "experiments/registry.csv;docs/selection_leakage_audit.md", "freeze selection before test; do not test every swept cell"),
        ("P16", "global correlation/difficulty/prefix-AUC analysis", "completed_with_caveats", "existing_results", "core C prefix AUC, task correlations, and difficulty curves exist", "reports/paper_validity_20260821T200224Z_2075f3e/FINAL_REPORT.md", "small n=8 model-level correlations; fair gain correlations absent"),
    ]
    fields = [
        "plan_id",
        "planned_item",
        "status",
        "requirement_class",
        "current_evidence",
        "evidence_paths",
        "gap_or_caveat",
    ]
    return [dict(zip(fields, row)) for row in rows]


def build_queue_rows() -> list[dict[str, object]]:
    rows = [
        ("Q01", "P0", "analysis_only", "regenerate current global visualization pack", "761 canonical rows; no new training", "core-default prefix lines; campaign-separated layer lines; model x task heatmap; collision annotations", "none", "none", 0, "plots must reproduce report aggregates and never pool 21 collision groups", "first executable task"),
        ("Q02", "P0", "canonical_packaging_then_eval_only", "complete comparable default-layer B2 angle table", "31/32 default B2 val_summary files already contain mean_ang_err_deg", "audit and stage the 31 existing summaries; eval only F3R p24 if checkpoint/manifest pass", "existing", "none", 0, "32/32 angle rows on one verified protocol before any B2 leaderboard", "do not retrain the full B2 grid"),
        ("Q03", "P0", "probe_train_eval_pilot", "diffusion-step pilot on ready Wan features, excluding action_dynamics", "Wan t499/t999 has 24/24 full feature groups; t749 is canonical", "model=wan; t={499,999}; layer=20; prefix={8,24}; probes={streaming_depth,view_consistency}; compare with t749", "existing_complete", "none", 8, "consistent direction at both prefixes and no metric/schema anomalies", "this is timestep analysis, not Wan action-validity"),
        ("Q04", "P0", "construct_audit_design", "formalize planning / target-conditioned action-output task", "no INTACT artifact or repository task definition", "write task contract, scene-disjoint split, target perturbations, no-op/behavior-cloning/geodesic priors, geometric action metric, leakage tests", "not_applicable", "required_before_any_run", 0, "audit must show target cannot identify the answer/candidate by construction", "no sweep until this passes"),
        ("Q05", "P1", "probe_train_eval_pilot", "matched no-condition C control on one non-Wan representative", "DINO p8 streaming+target features and C dumps exist; raw C1 loses to copy", "DINO l0 p8 only; capacity-matched full vs visual-only/no-condition vs action-only vs matched wrong-action", "existing_complete", "reuse paper-validity contract", 4, "full must beat copy and matched controls on scene-disjoint samples", "stop if gate fails; do not expand to Wan"),
        ("Q06", "P1", "probe_train_eval_replication", "replicate timestep pilot on MiniMax-H3 streaming tasks", "MiniMax t500/t1000 has 24/24 streaming-ready feature groups; targets incomplete", "model=minimax_h3; t={500,1000}; layer=20; prefix={8,24}; probes={streaming_depth,view_consistency}; compare t250", "existing_streaming_only", "none", 8, "replicate or falsify Wan timestep trend", "do not run C tasks until target coverage is complete"),
        ("Q07", "P1", "definition_and_feature_pilot", "define prediction feature and redo a minimal A slice", "no prediction-feature mode or canonical data exists", "one generation model; one layer; t fixed; p8; A1 depth/camera plus A2; compare context hidden vs explicitly defined predicted hidden/output", "requires_new_extraction", "required_before_extraction", 2, "feature semantics fixed and prediction beats/changes context on more than one A metric", "do not launch full A series first"),
        ("Q08", "P1", "evaluation_intervention", "streaming frame-order control on existing selected checkpoints", "B2 p12 post-feature shuffle/reverse already shows weak order sensitivity", "V-JEPA2/VideoMAE/Wan/CogVideoX at p12; clearly separate token-order intervention from pre-backbone frame shuffle", "existing", "required_to_name_construct", 0, "report paired scene bootstrap and effect sizes; no claim of encoder order sensitivity from post-feature shuffle", "new feature extraction only if encoder-sensitivity claim remains important"),
        ("Q09", "P1", "frozen_test_eval", "held-out confirmation of only frozen headline settings", "all canonical rows are val seed42", "one predeclared setting per model/task role; seeds 41/42/43 where training variance matters; one held-out test pass", "existing_or_checkpoint", "selection_policy_freeze", 0, "rank-role conclusions stable with uncertainty", "do not test all layer/prefix cells"),
        ("Q10", "P2", "feature_completion", "finish CogVideoX timestep features only if Q03/Q06 show a useful timestep effect", "CogVideoX t499/t999 streaming partial and targets absent", "complete only the feature cells required by the confirmed pilot design", "requires_new_extraction", "pilot_gate", 0, "exact manifest coverage and finite-feature audit", "conditional; not current P0"),
        ("Q11", "P2", "analysis_cube", "build reduced layer x timestep x prefix cube", "no current tstep probe metrics", "only models/tasks that pass timestep and construct gates; use layer={early,middle,late}, prefix={8,24}", "mixed", "Q03/Q05/Q06", 0, "interaction effects with uncertainty, not best-cell reporting", "never start with the full Cartesian product"),
        ("Q12", "DROP", "do_not_run", "full Wan p8 action-validity expansion", "Phase-1 gate failed because l10/l15 did not beat last-state copy", "none", "none", "gate_failed", 0, "not applicable", "completed case study; no further sweep"),
        ("Q13", "DROP", "do_not_run", "blind recovery of all 162 historical provisional Wan/Cog rows", "registry excludes 162 provisional rows and robust conclusions do not require every historical cell", "none", "unknown_or_missing", "low_value", 0, "not applicable", "recover only a predeclared headline cell if indispensable"),
        ("Q14", "DROP", "do_not_run", "legacy non-streaming A3 full sweep", "legacy A3 has known dtype/seed issues and does not match current streaming protocol", "none", "would_require_new_extraction", "construct_mismatch", 0, "not applicable", "replace with a small streaming control if needed"),
    ]
    fields = [
        "queue_id",
        "priority",
        "action_type",
        "task",
        "current_evidence",
        "minimal_scope",
        "feature_state",
        "construct_validity_dependency",
        "estimated_new_probe_training_jobs",
        "success_or_stop_gate",
        "scope_guard",
    ]
    return [dict(zip(fields, row)) for row in rows]


def robust_findings_md(
    complete: list[dict[str, str]], layer_counts: Counter[str]
) -> str:
    task_specs = [
        ("A1 depth", "A1", "streaming_depth/depth", "mean_loss_depth", False),
        ("A1 camera", "A1", "streaming_depth/camera", "Auc_30", True),
        ("A1 mask", "A1", "streaming_depth/mask", "mean_loss_identity", False),
        ("A2", "A2", "view_consistency", "balanced_acc", True),
        ("C1", "C1", "action_dynamics", "global_R@1", True),
        ("C2", "C2", "path_integration", "global_R@1", True),
        ("C3", "C3", "counterfactual", "intervention_validity", True),
    ]
    improvement_count = 0
    total_pairs = 0
    model_task_means: dict[str, dict[str, float]] = defaultdict(dict)
    for label, family, probe, metric, higher in task_specs:
        data = core_task_rows(complete, family, probe, metric)
        for model, values in data.items():
            delta = values[24] - values[8]
            improvement_count += int(delta > 0 if higher else delta < 0)
            total_pairs += 1
            model_task_means[model][label] = statistics.mean(values.values())

    avg_ranks: dict[str, list[int]] = {model: [] for model in CORE_MODELS}
    for label, _, _, _, higher in task_specs:
        ranked = sorted(
            ((model_task_means[model][label], model) for model in CORE_MODELS),
            reverse=higher,
        )
        for rank, (_, model) in enumerate(ranked, 1):
            avg_ranks[model].append(rank)
    average_rank = {
        model: statistics.mean(ranks) for model, ranks in avg_ranks.items()
    }

    c1 = read_csv(CANONICAL_DIR / "c1_results.csv")
    default_layer = {
        "dino": "0",
        "f3r": "24",
        "vjepa2": "23",
        "videomae": "23",
        "qwen2_5_vl": "-1",
        "wan": "20",
        "cogvideox": "20",
        "aether": "1",
    }
    copy_stats: dict[str, tuple[float, int, int]] = {}
    for model, layer in default_layer.items():
        rows = [row for row in c1 if row["model"] == model and row["layer"] == layer]
        gains = [f(row["gain_over_copy"]) for row in rows if row["gain_over_copy"]]
        copy_stats[model] = (statistics.mean(gains), sum(gain > 0 for gain in gains), len(gains))

    text = f"""# Probing Spatial Global Robust Findings

本文件只把跨 prefix、跨模型或经独立 construct audit 支持的模式列为 finding；不把单个 best score 当作全局结论。主数值来自 `experiments/registry.csv` 与 `results/canonical/*.csv`，selected-control 证据来自明确标注的 supplemental audits。

## 1. 能力是分工的，不存在一个对所有空间任务都占优的表征

- **证据：**8-model default comparable grid 上，F3R 的四-prefix 平均 A1 depth loss={fmt(model_task_means['f3r']['A1 depth'])}（最低）、camera AUC={fmt(model_task_means['f3r']['A1 camera'])}（最高）、A2={fmt(model_task_means['f3r']['A2'])}（最高）；DINO 则在 C1={fmt(model_task_means['dino']['C1'])}、C2={fmt(model_task_means['dino']['C2'])}、C3={fmt(model_task_means['dino']['C3'])} 取得最高 raw primary metric。paper-validity 的 model-level prefix-AUC 相关性进一步显示 A1 camera 与 C1 的 Spearman 约 -0.024，而 C1 与 C2 为 0.952。
- **反例：**DINO 的 A1 camera 均值只有 {fmt(model_task_means['dino']['A1 camera'])}；F3R 的 C1/C2 只有 {fmt(model_task_means['f3r']['C1'])}/{fmt(model_task_means['f3r']['C2'])}。
- **潜在 confound：**任务头、pixel alignment、feature dimensionality、target-feature geometry 和统一超参并不等价；相关性只有 8 个 model-level 点。
- **置信度：**中高，适合写成“representation specialization”，不适合写成通用 intelligence 排名。

## 2. 长 streaming prefix 通常不会稳定提升，退化比提升更常见

- **证据：**在 8 models × 7 comparable primary tasks 的 56 个 p24-vs-p8 配对中，仅 {improvement_count}/56 改善，{total_pairs-improvement_count}/56 退化。C2 为 0/8 改善；A1 camera 为 1/8 改善。DINO C2 从 {fmt(core_task_rows(complete,'C2','path_integration','global_R@1')['dino'][8])} 降到 {fmt(core_task_rows(complete,'C2','path_integration','global_R@1')['dino'][24])}，Qwen 从 {fmt(core_task_rows(complete,'C2','path_integration','global_R@1')['qwen2_5_vl'][8])} 降到 {fmt(core_task_rows(complete,'C2','path_integration','global_R@1')['qwen2_5_vl'][24])}。
- **反例：**F3R camera p24 高于 p8；V-JEPA2 C3 从 {fmt(core_task_rows(complete,'C3','counterfactual','intervention_validity')['vjepa2'][8])} 升到 {fmt(core_task_rows(complete,'C3','counterfactual','intervention_validity')['vjepa2'][24])}。
- **潜在 confound：**prefix-specific 训练、窗口难度和有效样本构成变化会与“记忆长度”混在一起；不能直接解释成模型忘记。
- **置信度：**高（对当前 protocol 的经验模式），中（对模型 memory mechanism 的解释）。

## 3. “中间层普遍最好”不成立；层峰值强烈依赖模型与任务

- **证据：**只在 `eval_20260817` 内、每组至少 3 个且完整 layer cells 的 99 个 model×task×prefix 组中，best layer 为 intermediate {layer_counts['intermediate']}/99、deepest available {layer_counts['last']}/99、earliest available {layer_counts['first']}/99。VideoMAE C3 在三个完整 prefix 组均偏 deepest available；MiniMax-H3 C1/C2/C3 多数偏 earliest；V-JEPA2/B2 多个 prefix 偏中间层。
- **反例：**Wan p8 C1 raw peak 在 layer 10，但 construct audit 说明该峰不击败 persistence；“peak”本身不是构念证据。
- **潜在 confound：**这里的 first/last 是 campaign 中最早/最深可用层，不总是架构真正首尾；layer grid 不对称，21 个 scientific-config collisions 不能跨 campaign 合并。
- **置信度：**高（否定 universal middle-layer claim），中（具体任务的层定位）。

## 4. 当前 ViGen cohort 没有形成通用空间表征优势

- **证据：**在 8-model default comparable grid 的七项平均 rank 中，Wan={average_rank['wan']:.2f}、CogVideoX={average_rank['cogvideox']:.2f}、Aether={average_rank['aether']:.2f}，落后于 DINO={average_rank['dino']:.2f}、F3R={average_rank['f3r']:.2f}、V-JEPA2={average_rank['vjepa2']:.2f}、Qwen={average_rank['qwen2_5_vl']:.2f}、VideoMAE={average_rank['videomae']:.2f}。Aether 在七项均排第 8；Wan 最好的相对位置是 A1 camera/C1 的第 4。
- **反例：**Wan 在 selected C1 p12 audit 中有正 same-scene action-specific gain；LingBot/MiniMax 的 layer campaigns 也存在局部可用 cells。
- **潜在 confound：**ViGen 对 timestep/noise 和 layer 更敏感；Wan/Cog/Aether default 为 t749，MiniMax/LingBot 为 t250；新的 tstep grid 尚无 probe 结果。
- **置信度：**中高，结论应限定为“当前被 probe 的层/step 与 protocol”，而不是模型能力上限。

## 5. C1 raw retrieval 高分经常是 persistence/readability，而不是 action-conditioned dynamics

- **证据：**default C1 中 DINO 和 Qwen raw mean R@1 分别为 {fmt(model_task_means['dino']['C1'])}/{fmt(model_task_means['qwen2_5_vl']['C1'])}，但相对 last-observation copy 的四-prefix平均 gain 分别为 {copy_stats['dino'][0]:+.4f}/{copy_stats['qwen2_5_vl'][0]:+.4f}，且二者 0/4 prefixes 击败 copy。F3R 4/4 击败 copy，但 raw R@1 只有 {fmt(model_task_means['f3r']['C1'])}。Wan p8 audit 中 correct action 显著优于 shuffled/wrong action，但 layer 10/15 仍输给 last-state copy。
- **反例：**CogVideoX 3/4、F3R 4/4 prefixes 的 full probe 高于 copy；Wan 某些层也有 correct-over-shuffled signal。
- **潜在 confound：**copy baseline 本身受 feature temporal smoothness 和 candidate pool 影响；legacy no-action/shuffle 并非全部为 capacity-matched retraining。
- **置信度：**高，足以要求所有 C1 结论先报告 baseline-normalized gain。

## 6. B2 更像 query/history-content retrieval，而不是有序 temporal memory

- **证据：**当前 canonical B2 的 model-wide mean angular error 从 VideoMAE 53.52° 到 MiniMax-H3 77.62°，整体仍粗糙。selected p12 construct audit 中，V-JEPA2/VideoMAE query-mask degradation 为 +15.79°/+11.18°，remove-visible-history 为 +9.22°/+5.83°；但所有 core models 的 shuffle/reverse 平均影响绝对值均小于 2.5°。audit verdict 为 `REINTERPRET_AS_HISTORY_CONTENT_RETRIEVAL`。
- **反例：**Wan 的 order perturbation 数值上最大（约 +1.40°），V-JEPA2/VideoMAE 确有明确 query dependence。
- **潜在 confound：**standard B2 不是 strict-zero-hidden；scene layout、category、visible frame、query construction 与 position prior 都可能贡献。default 8-model angle grid 目前也未完整进入 canonical。
- **置信度：**中高（selected p12 models），不应外推到所有 layers/prefixes。

## 7. C1 与 C2 的模型排序高度相关，可能共享“target-feature retrievability”因子

- **证据：**paper-validity 的 8-model prefix-AUC 上 C1-C2 Spearman=0.952、Kendall=0.857；DINO/Qwen 同时居前，Aether/CogVideoX 同时靠后。
- **反例：**V-JEPA2 C2 相对 C1 更强；horizon-specific C2 指标并不等于 C1 one-step。
- **潜在 confound：**两者共享 streaming input、target-isolated feature space、candidate pool 和 probe family，相关性可能由 feature readability 而非 path integration 产生。
- **置信度：**中；适合推动增量/条件控制，不足以删掉任一任务。

## 8. 当前数据不能回答 diffusion-step 或 prediction-feature 优势

- **证据：**canonical 的生成模型每个 model 只有一个配置 step（Wan/Cog/Aether t749；MiniMax/LingBot t250），没有同模型跨 step canonical metrics；A/B 全部使用 streaming context features，C 使用 streaming input + target-isolated supervision。仓库中没有 canonical `prediction feature` 行，也没有名为 prediction-feature 的 extraction mode。
- **反例：**feature inventory 已经具备 Wan t499/t999 的完整 3-layer×4-prefix streaming+target cells，以及 MiniMax t500/t1000 的 streaming cells。
- **潜在 confound：**“prediction feature”尚未被操作化；Aether pipeline 的 `task=prediction` 代码路径不等于已缓存、可比较的 prediction representation。
- **置信度：**高（coverage fact）；因此任何“prediction 比 context 更可读”的说法目前都应标为 NOT TESTED。

## 全局适用的限制

全部 761 个 canonical primary rows 都是 `seed=42`、`eval_split=val`。因此上述 finding 是当前数据内部的稳健模式，不是已完成的 held-out、多 seed 外部泛化结论。
"""
    return text


def report_md(
    registry: list[dict[str, str]],
    complete: list[dict[str, str]],
    matrix: list[dict[str, object]],
    collision_count: int,
    layer_counts: Counter[str],
    tsteps: dict[str, object],
    protected_hashes: dict[str, str],
) -> str:
    status_counts = Counter(row["status"] for row in registry)
    family_counts = Counter(row["family"] for row in complete)
    family_models: dict[str, set[str]] = defaultdict(set)
    family_layers: dict[str, set[str]] = defaultdict(set)
    family_prefixes: dict[str, set[str]] = defaultdict(set)
    family_metrics: dict[str, set[str]] = defaultdict(set)
    family_variants: dict[str, set[str]] = defaultdict(set)
    for row in complete:
        family_models[row["family"]].add(row["model"])
        family_layers[row["family"]].add(row["layer"])
        family_prefixes[row["family"]].add(row["prefix"])
        family_metrics[row["family"]].add(row["metric_name"])
        family_variants[row["family"]].add(row["model_variant"])

    models = sorted(MODEL_META)
    families = ["A1", "A2", "B1", "B2", "C1", "C2", "C3"]
    counts_by_model_family = Counter((row["model"], row["family"]) for row in complete)
    model_table = [
        "| model | " + " | ".join(families) + " | total |",
        "| --- | " + " | ".join(["---:"] * (len(families) + 1)) + " |",
    ]
    for model in models:
        values = [counts_by_model_family[(model, family)] for family in families]
        model_table.append(
            f"| {model} | " + " | ".join(map(str, values)) + f" | {sum(values)} |"
        )

    coverage_table = [
        "| family | primary rows | models | layers | prefixes | primary metric(s) | variants |",
        "| --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for family in families:
        coverage_table.append(
            f"| {family} | {family_counts[family]} | {len(family_models[family])} | "
            f"{len(family_layers[family])} | {','.join(sorted(family_prefixes[family], key=int))} | "
            f"{', '.join(sorted(family_metrics[family]))} | {', '.join(sorted(family_variants[family]))} |"
        )

    hashes_table = [
        "| source | rows | unique experiment IDs | sha256 |",
        "| --- | ---: | ---: | --- |",
    ]
    for rel in ["experiments/registry.csv"] + [f"results/canonical/{name}" for name in CANONICAL_FILES]:
        path = ROOT / rel
        rows = read_csv(path)
        unique_ids = len({row["experiment_id"] for row in rows}) if rows and "experiment_id" in rows[0] else "—"
        hashes_table.append(f"| `{rel}` | {len(rows)} | {unique_ids} | `{protected_hashes[rel]}` |")

    plan_rows = build_planned_rows()
    plan_summary = Counter(row["status"] for row in plan_rows)
    requirement_summary = Counter(row["requirement_class"] for row in plan_rows)

    tstep_lines = []
    for key, value in tsteps.items():
        statuses = value["statuses"]
        tstep_lines.append(
            f"- `{key}`: {statuses}; streaming found/group range "
            f"{value['stream_found_min']}–{value['stream_found_max']}; target "
            f"{value['target_found_min']}–{value['target_found_max']}/681; target_indices "
            f"{value['target_indices']}/681."
        )

    task_definition_table = """| family | 实际测量对象 | canonical primary metric | 当前覆盖边界 |
| --- | --- | --- | --- |
| A1 | streaming features 的 depth、camera pose 与 identity/mask 可读性 | depth loss ↓；AUC-30 ↑；identity loss ↓ | 10 models，但扩展 layer grid 不对称 |
| A2 | 两段 view 是否共享视域 | balanced accuracy ↑ | 9 models；LingBot 缺失，MiniMax 缺 p24 |
| B1 | legacy masked/query hidden-object bearing | mean azimuth error ↓ | 仅 Wan/CogVideoX 11 rows；supplemental，不作为主线 |
| B2 | object query + prefix patch tokens 对 hidden-object 方位/距离的读出 | mean angular error ↓ | canonical 仅 6 models；standard construct 非 strict-zero-hidden |
| C1 | context + 单步相机 action 预测 isolated target feature 的检索 | global R@1 ↑ | 10 models；raw metric 必须和 no-action/shuffle/copy 一起解释 |
| C2 | action sequence 的 waypoint target-feature rollout/path integration | global R@1 ↑；expanded file 另有 h1/h2/h4 pose error | 10 models；primary table 每 experiment 1 行，expanded file 每 experiment 4 行 |
| C3 | alternative action 对 counterfactual target prediction 的 intervention consistency | intervention validity ↑ | 10 models；expanded file 每 experiment 7 行 |
"""

    text = f"""# probing_spatial 全局主线重启报告

生成日期：2026-08-28 UTC。报告文件名按请求保留 `20260827`。本轮是严格的 metadata/result analysis：没有启动训练、没有抽 feature、没有提交作业、没有修改 `experiments/registry.csv` 或 `results/canonical/*.csv`。

## 执行结论

全局主线可以恢复，但必须先固定两个统计粒度：

1. **428/428 是 eval job-level campaign 完整性。**最终 promotion audit 记录 428 个 valid summaries、428 个 unique job keys、0 duplicates；Wan/p8/action_dynamics 的 6 个缺失 layer jobs 正是最后补齐的部分。
2. **当前 canonical universe 是 761 个 atomic primary-metric experiments。**`registry.csv` 共 955 个唯一 experiment IDs，其中 {status_counts['COMPLETE']} COMPLETE、{status_counts['INCOMPLETE']} INCOMPLETE；`all_results.csv` 恰好包含全部 761 个 COMPLETE IDs，字段逐项一致。

因此不能把“428 个 eval jobs”误写成“canonical 只有 428 行”。一个 eval job 可产生多个 atomic probe outputs；C2/C3 family tables 又展开多个 horizon/secondary metrics。

## 1. Source of truth 与精确完整性

{chr(10).join(hashes_table)}

完整性核验：

- registry experiment IDs：955/955 unique。
- canonical `all_results.csv`：761/761 unique；与 registry COMPLETE ID set 完全相同；family/probe/model/layer/prefix/metric/value/raw path mismatch=0。
- exact duplicate canonical rows=0；duplicate experiment IDs=0。
- 若把 `model_variant` 纳入配置 identity，761 rows 全唯一。
- 若忽略 `model_variant`，存在 **{collision_count} 个 scientific-config collision groups**（42 rows、21 excess）。主要是 VideoMAE/V-JEPA2 A1/A2 的 default vs 20260817 eval，以及 Wan B1/B2 的 historical vs gap-fill。它们数值可显著不同，必须按 campaign 分开，不能平均或任选较优值。
- `scripts/check_project_consistency.py` PASS；保留两个既有 VideoMAE p12 missing-manifest path warnings。

family 文件的展开口径：A1=265 rows、A2=82、B2=113、C1=96；C2=368 rows/92 IDs（每 ID: global + h1/h2/h4）、C3=714 rows/102 IDs（每 ID: 4 global metrics + h1/h2/h4）。B1 的 11 个 primary rows 只存在于 `all_results.csv`，没有单独 `b1_results.csv`。

## 2. 完整分析矩阵与覆盖

`results/analysis/global_restart/experiment_matrix.csv` 有 {len(matrix)} 行，每行对应一个 canonical atomic primary experiment，补全了以下分析维度：task/family、model、model/objective family、layer 及 registered-grid position、streaming prefix、config-declared diffusion step、streaming/target/prediction output role、condition、metric direction、campaign cohort、collision flag 与 provenance。

需要注意：

- canonical registry 本身没有独立 timestep/feature-type/condition 列；矩阵中的 timestep 来自对应 config，task contract 来自 probe/data flow。
- 生成模型 canonical step 为 Wan/CogVideoX/Aether=`749`、MiniMax-H3/LingBot=`250`；非生成 backbone 记为 `NA_non_diffusion`。
- A/B input 是 `streaming`；C input 是 `streaming`、监督 target 是 `target_isolated`，probe output 才是 feature prediction。**target feature 不是 prediction feature。**
- 所有 canonical rows 均为 seed 42、val split。

### family-level 覆盖

{chr(10).join(coverage_table)}

### model × family primary-row 矩阵

{chr(10).join(model_table)}

### comparable core 与扩展 coverage

- 可严格横向比较的 default core：8 models × 4 prefixes × 7 primary task/metrics（A1 depth/camera/mask、A2、C1、C2、C3）= **224/224 rows**。
- B2 不在这个完整 grid 内。registry 中 32 个 default B2 rows 仍为 INCOMPLETE，因为旧总表只机器可读地封装了 top1；独立核验显示其中 31/32 个现存 `val_summary.json` 已包含 `mean_ang_err_deg`，F3R p24 summary 缺失。应先 staging/audit，再补 1 个 eval-only，而不是重训 32 个。
- Layer expansion 覆盖不对称：LingBot 主要只有 p16/p24；MiniMax A2 无 p24；Wan/Cog historical layer sweep 有大量 provisional rows 被正确排除；不能用“每模型 best cell”替代公平 matrix。

## 3. A1、A2、B2、C1、C2、C3 到底测什么

{task_definition_table}

更严格的解释边界：A1/A2 是 frozen-feature readout，不等于显式 3D world model；B2 可能受 query/category/scene/visibility prior；C1/C2/C3 是 target-feature prediction/retrieval，不自动等于物理 dynamics。C3 的 intervention-validity 和 C1/C2 R@1 也不在同一量纲。

## 4. 历史计划搜索结果与完成度

已对 `planning`、`INTACT`、`diffusion step`、`shuffle`、`without condition`、`prediction feature`、`streaming prefix`、`layer sweep`、`analysis cube`、`probing_spatial`、`paper`、`canonical` 做 repository-wide 搜索，并查阅 canonical build/update scripts、历史 reports、paper-validity、probe-validity、SE(3) MVP 与 tstep tooling。

- `INTACT`：仓库 0 个 literal hit。
- `prediction feature` / `analysis cube`：0 个明确计划或产物 hit。
- `planning`：只有 Aether pipeline 内部 task mode 和无关的 capacity-planning 文本；没有 probing task/config/metric/manifest。
- diffusion/tstep：有完整 extraction/probe orchestration code 与当前 feature indexes，但还没有 tstep probe run directory 或 canonical metrics。
- shuffle：有 legacy A3、paper-validity legacy controls、selected p12 B2/C1 sample audit。
- without-condition：有 legacy eval intervention、Wan eval-null-action（明确不是 trained no-condition）和 non-canonical DINO SE(3) MVP。
- streaming prefix/layer/canonical/paper：历史与当前产物充分，但部分文档（例如早期 new-backend status）已被 20260817 canonical gap-fill 超越，不能再作为结果 source of truth。

`planned_vs_completed.csv` 共 {len(plan_rows)} 项；状态分布 `{dict(plan_summary)}`。按需求分类分布 `{dict(requirement_summary)}`。

### 四类缺口

1. **已有数据即可完成：**当前 global plots、prefix robustness、campaign-separated layer curves、task correlation、default B2 的 31 个 raw angle rows staging audit、Wan case-study integration。
2. **已有 feature 但需 probe/eval：**Wan t499/t999 全 feature grid；MiniMax t500/t1000 streaming-only grid；DINO matched no-condition controls；F3R p24 B2 eval-only；frozen headline test/multi-seed。
3. **必须重新抽 feature：**prediction-feature A pilot；CogVideoX incomplete tstep cells；如果问题是“backbone 是否对 frame order 敏感”，则需 pre-backbone shuffled-frame features。
4. **先做 construct-validity audit：**INTACT-style target-conditioned planning/action output；prediction feature 的操作化；strict B2；任何把 C raw retrieval 解释成 dynamics 的扩展。

## 5. 当前最可信的全局发现

完整证据、反例与 confound 写在 `robust_findings.md`。摘要如下：

1. 表征呈能力分工：F3R 是 direct geometry specialist，DINO/Qwen 的 C raw retrieval 强，但两者不是同一能力轴。
2. 长 prefix 经常退化：56 个 core p24-vs-p8 配对只有 13 个改善，C2 是 0/8。
3. universal middle-layer hypothesis 不成立：99 个 campaign-internal complete layer groups 中 intermediate/last/first = {layer_counts['intermediate']}/{layer_counts['last']}/{layer_counts['first']}。
4. 当前 ViGen layers/steps 没有通用空间优势；结论受 timestep coverage 限制。
5. C1 raw R@1 常被 persistence 解释；DINO/Qwen 尽管 raw 高，却在四个 prefix 都输给 copy。
6. B2 selected audit 更支持 query/history-content retrieval，不支持强 ordered-memory claim。
7. C1/C2 model prefix-AUC 排序高度相关（Spearman 0.952），可能共享 target-feature readability 因子。
8. diffusion-step 与 prediction-vs-context 目前均未被 canonical 数据回答。

所有这些结论都受 seed42/val-only 限制；当前不能声称 held-out、多 seed generalization。

## 6. Diffusion-step 与 analysis cube 的真实状态

当前 completion indexes 的 exact-manifest 重算结果：

{chr(10).join(tstep_lines)}

这说明最小可执行路径不是继续抽 Wan feature，而是直接用已有 Wan t499/t999 做 A1/A2 小 pilot，并与 canonical t749 比较；不应把 tstep 工作重新变成 Wan action_dynamics。MiniMax 可作为 streaming-task replication。CogVideoX 只有 pilot 显示 timestep effect 后才值得补 extraction。

完整 `layer × step × prefix` cube 目前不能分析：只有 feature cells，没有 probe metrics。先做 layer20、p8/p24 的 reduced pilot，过 gate 后再加入 early/middle/late layers。

## 7. 原计划中真正缺失的实验

- target-conditioned planning/action output：task definition、INTACT mapping、data/metric/baseline 全缺。
- prediction feature A-series：feature definition、extraction mode 与任一 result 全缺。
- fair trained without-condition/visual-only/action-only C controls：只有 isolated examples/legacy interventions，没有跨模型 canonical grid。
- current streaming encoder-level shuffled-frame baseline：legacy A3 不能替代。
- diffusion-step probes：features 部分或全部存在，但 probe/eval 全缺。
- held-out test + multi-seed frozen confirmation：全缺。
- strict-zero-hidden B2 与完整 simple-prior baselines：selected audit 已明确当前 standard B2 不足。
- current 10-model cross-model plot/demo pack：旧 core figures 有，当前 full canonical 版本缺。

## 8. 下一批最小优先任务

详见 `next_experiment_queue.csv`。P0 顺序是：

1. 用现有 761 rows 生成 current plot pack；0 training。
2. staging/audit 31 个 existing B2 angle summaries，仅对 F3R p24 做 eval-only；不重训整套。
3. Wan tstep 的非 action pilot：t499/t999、layer20、p8/p24、A1+A2，共 8 个新 probe training jobs；t749 复用 canonical。
4. 先完成 planning/INTACT task construct contract；0 training。

之后才是一个 non-Wan matched no-condition C pilot、MiniMax timestep replication、prediction-feature minimal extraction，以及 frozen headline test。

## 9. 不值得继续扩 sweep 的任务

- **Wan p8 action-validity：停止。**Phase-1 gate 已给出明确答案，不进入 capacity-matched Phase 2，不再扩 layers/prefixes。
- 不补齐 162 个 historical provisional Wan/Cog rows，只在某个 frozen headline cell 不可替代时恢复。
- 不做 legacy non-streaming A3 full sweep。
- 不做未过 construct gate 的 full planning 或 prediction-feature Cartesian sweep。
- 不做 raw C1 best-layer/best-prefix sweep；任何扩展先看 gain-over-copy 与 matched controls。
- 不做完整 tstep cube；先过 reduced pilot。
- B1 不作为主线继续扩，因为其 mask/visibility design 只适合 upper-bound/diagnostic。

## 10. Wan action-validity 应怎样嵌入全局故事

把它放在方法论/limitations 中，作为一个清晰的 **case study**：

> Raw probing performance 不等于目标构念。Wan p8 的 correct action 确实显著优于 shuffled/wrong action，说明存在 weak action signal；但 layer 10/15 没有击败 last-state copy，所以原始中间层峰值不能直接解释成 action-conditioned dynamics。

这个 case study 支持三条全局规范：先报告 shortcut-normalized gain；中间层 peak 不能单独支撑机制解释；construct audit 可以在不修改 canonical 的情况下改变结论强度。它不应变成项目标题、主要 model ranking 或新的 Wan sweep 主线。

## 产物

- `reports/probing_spatial_global_restart_20260827.md`
- `results/analysis/global_restart/experiment_matrix.csv`
- `results/analysis/global_restart/planned_vs_completed.csv`
- `results/analysis/global_restart/robust_findings.md`
- `results/analysis/global_restart/next_experiment_queue.csv`

生成后脚本再次核验所有 source-of-truth SHA256；任何变化都会 fail closed。
"""
    return text


def main() -> None:
    protected = [REGISTRY] + [CANONICAL_DIR / name for name in CANONICAL_FILES]
    before = {str(path.relative_to(ROOT)): sha256(path) for path in protected}

    registry = read_csv(REGISTRY)
    complete = [row for row in registry if row["status"] == "COMPLETE"]
    all_results = read_csv(CANONICAL_DIR / "all_results.csv")
    assert len(registry) == 955
    assert len(complete) == 761
    assert len(all_results) == 761
    assert len({row["experiment_id"] for row in registry}) == len(registry)
    assert len({row["experiment_id"] for row in all_results}) == len(all_results)
    assert {row["experiment_id"] for row in complete} == {
        row["experiment_id"] for row in all_results
    }
    registry_by_id = {row["experiment_id"]: row for row in complete}
    for row in all_results:
        source = registry_by_id[row["experiment_id"]]
        for field in [
            "family",
            "probe",
            "model",
            "layer",
            "prefix",
            "metric_name",
            "metric_value",
            "raw_artifact_path",
        ]:
            assert row[field] == source[field]

    matrix, collision_count = build_matrix(complete)
    assert len(matrix) == 761
    assert collision_count == 21
    layer_counts = layer_peak_counts(complete)
    assert layer_counts == Counter({"intermediate": 41, "last": 32, "first": 26})
    tsteps = tstep_inventory()

    matrix_fields = list(matrix[0])
    write_csv(OUT_DIR / "experiment_matrix.csv", matrix, matrix_fields)

    planned = build_planned_rows()
    planned_fields = list(planned[0])
    write_csv(OUT_DIR / "planned_vs_completed.csv", planned, planned_fields)

    queue = build_queue_rows()
    queue_fields = list(queue[0])
    write_csv(OUT_DIR / "next_experiment_queue.csv", queue, queue_fields)

    (OUT_DIR / "robust_findings.md").write_text(
        robust_findings_md(complete, layer_counts), encoding="utf-8"
    )
    REPORT.write_text(
        report_md(
            registry,
            complete,
            matrix,
            collision_count,
            layer_counts,
            tsteps,
            before,
        ),
        encoding="utf-8",
    )

    after = {str(path.relative_to(ROOT)): sha256(path) for path in protected}
    assert before == after, "protected registry/canonical sources changed"
    print(
        json.dumps(
            {
                "ok": True,
                "matrix_rows": len(matrix),
                "planned_rows": len(planned),
                "queue_rows": len(queue),
                "scientific_config_collision_groups": collision_count,
                "protected_sources_unchanged": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
