#!/usr/bin/env python3
"""Build a reproducible analysis pack for the clean Active Spatial runs.

The source VAGEN-Lite tree is read-only. All derived tables and figures are
written under this repository so the analysis can be regenerated without
touching training artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DEFAULT_SOURCE = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite")
DEFAULT_OUTPUT = DEFAULT_SOURCE / "reports/active_spatial_v46_v50_cambrian_20260906"

RUNS = {
    "V46": {
        "family": "Qwen2.5-VL-7B",
        "variant": "W3, KL=0.30",
        "run": "qwen_v46_clean_d0pass_20260822_sco_renderer_r1_full",
        "train_logs": ["train.log"],
        "expected_steps": 700,
        "validation_contract": "ID-19 + OOD-25, 4 samples/task",
    },
    "V47": {
        "family": "Qwen2.5-VL-7B",
        "variant": "W3, KL=0.40",
        "run": "qwen_v47_clean_d0pass_20260826_recovery_r1_full",
        "train_logs": ["train.log"],
        "expected_steps": 700,
        "validation_contract": "ID-19 + OOD-25, 4 samples/task",
    },
    "V48": {
        "family": "Qwen2.5-VL-7B",
        "variant": "W1, KL=0.30",
        "run": "qwen_v48_clean_d0pass_20260823_full",
        "train_logs": [
            "../qwen_v48_clean_d0pass_20260823_full_pre_resume_archive_20260826/train.log",
            "train.log",
        ],
        "expected_steps": 700,
        "validation_contract": "ID-19 + OOD-25, 4 samples/task",
    },
    "V49": {
        "family": "Qwen2.5-VL-7B",
        "variant": "V26 replay, W1, KL=0.20",
        "run": "v49_7b_v26_repro_bad_render_20260723_010323",
        "train_logs": ["train.log"],
        "expected_steps": 700,
        "validation_contract": "invalid: renderer failure at step 0",
    },
    "V50": {
        "family": "Qwen2.5-VL-7B",
        "variant": "W3, entropy=0.001, short response, format=0.01",
        "run": "qwen_v50_clean_d0pass_20260824_full",
        "train_logs": ["train.log"],
        "expected_steps": 700,
        "validation_contract": "ID-19 + OOD-25, 4 samples/task",
    },
    "Cambrian-C8": {
        "family": "Cambrian-S-7B",
        "variant": "scaled reward + action-valid, deterministic val",
        "run": "cambrian_c8_clean_d0pass_20260824_sco_renderer_r3_full",
        "train_logs": ["train.log"],
        "expected_steps": 1000,
        "validation_contract": "ID-19 + OOD-25, 1 sample/task",
    },
    "Cambrian-C4": {
        "family": "Cambrian-S-7B",
        "variant": "high-variance reward, 4-sample val",
        "run": "cambrian_c4_clean_d0pass_20260822_h800_r5_full",
        "train_logs": ["train.log"],
        "expected_steps": 1000,
        "validation_contract": "legacy ID-21 + OOD-19, 4 samples/task",
    },
}

EVAL_DIRS = {
    "V46": "qwen_v46_clean_all_ckpts_id_ood_qa_20260828",
    "V47": "qwen_v47_clean_all_ckpts_id_ood_qa_20260828",
    "V48": "qwen_v48_clean_all_ckpts_id_ood_qa_20260828",
    "V50": "qwen_v50_clean_all_ckpts_id_ood_qa_20260828",
    "Cambrian-C8": "cambrian_c8_clean_all_ckpts_id_ood_qa_20260828",
    "Cambrian-C4": "cambrian_c4_clean_all_ckpts_id_ood_qa_20260828",
}

VALID_ACTIONS = (
    "move_forward",
    "move_backward",
    "move_left",
    "move_right",
    "turn_left",
    "turn_right",
)
ACTION_RE = re.compile(r"\b(" + "|".join(VALID_ACTIONS) + r")\b")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"\bstep:(\d+)\s+-")
NUMBER_RE = r"(?:np\.(?:float64|int32)\()?([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?|nan|inf|-inf)"

TRAIN_KEYS = (
    "episode/success_rate",
    "transition/invalid_action_rate/mean",
    "transition/missing_action_tag_rate/mean",
    "transition/reward_mean",
    "actor/entropy",
    "actor/grad_norm",
    "actor/kl_loss",
    "critic/score/mean",
    "critic/explained_var_valid",
    "rollout_corr/rollout_is_eff_sample_size",
    "rollout_corr/rollout_is_ratio_fraction_high",
    "rollout_corr/rollout_is_ratio_fraction_low",
    "response_length/mean",
    "response_length/clip_ratio",
    "perf/time_per_step",
)

TASK_ORDER = (
    "occlusion_alignment",
    "equidistance",
    "projective_relations",
    "fov_inclusion",
    "centering",
    "size_distance_invariance",
)
SUITE_ORDER = (
    "id_test",
    "ood_v2_centering",
    "ood_scene",
    "ood_instance",
    "ood_category",
    "ood_template",
    "ood_geometry",
)

COLORS = {
    "V46": "#157A6E",
    "V47": "#D1495B",
    "V48": "#3066BE",
    "V50": "#E09F3E",
    "Cambrian-C8": "#6A4C93",
    "Cambrian-C4": "#7A7A72",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def mean_value(rows: list[dict], key: str) -> float:
    values = [float(row.get(key, 0.0) or 0.0) for row in rows]
    return float(np.mean(values)) if values else math.nan


def wilson_interval(successes: float, total: int, z: float = 1.96) -> tuple[float, float]:
    if not total:
        return math.nan, math.nan
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return center - margin, center + margin


def parse_train_log(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    patterns = {key: re.compile(re.escape(key) + ":" + NUMBER_RE) for key in TRAIN_KEYS}
    with path.open(errors="replace") as handle:
        for raw_line in handle:
            line = ANSI_RE.sub("", raw_line)
            step_match = STEP_RE.search(line)
            if not step_match:
                continue
            row: dict[str, float | int] = {"step": int(step_match.group(1))}
            for key, pattern in patterns.items():
                match = pattern.search(line)
                if match:
                    try:
                        row[key] = float(match.group(1))
                    except ValueError:
                        row[key] = math.nan
            rows.append(row)
    return rows


def collect_training(source_root: Path) -> pd.DataFrame:
    experiment_root = source_root / "exps/vagen_active_spatial"
    collected: list[dict] = []
    for label, config in RUNS.items():
        run_root = experiment_root / config["run"]
        by_step: dict[int, dict] = {}
        for relative_log in config["train_logs"]:
            for row in parse_train_log(run_root / relative_log):
                by_step[int(row["step"])] = row
        for step in sorted(by_step):
            collected.append({"model": label, "family": config["family"], **by_step[step]})
    frame = pd.DataFrame(collected)
    if frame.empty:
        return frame
    frame = frame.sort_values(["model", "step"]).reset_index(drop=True)
    for key in (
        "episode/success_rate",
        "transition/invalid_action_rate/mean",
        "transition/missing_action_tag_rate/mean",
        "actor/entropy",
        "critic/score/mean",
        "critic/explained_var_valid",
    ):
        if key in frame:
            frame[key + "_ma25"] = frame.groupby("model")[key].transform(
                lambda values: values.rolling(25, min_periods=8, center=True).mean()
            )
    return frame


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def source_split(row: dict) -> str:
    source = row.get("data_source")
    if source in {"active_spatial_id_override", "active_spatial"}:
        return "ID"
    if source == "active_spatial_ood":
        return "OOD"
    return str(source or "unknown")


def collect_validation(source_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    experiment_root = source_root / "exps/vagen_active_spatial"
    curves: list[dict] = []
    tasks: list[dict] = []
    actions: list[dict] = []
    for label, config in RUNS.items():
        validation_root = experiment_root / config["run"] / "validation"
        if not validation_root.exists():
            continue
        files = sorted(validation_root.glob("*.jsonl"), key=lambda path: int(path.stem))
        for path in files:
            step = int(path.stem)
            rows = load_jsonl(path)
            for split in ("All", "ID", "OOD"):
                subset = rows if split == "All" else [row for row in rows if source_split(row) == split]
                successes = sum(float(row.get("traj_success", 0.0) or 0.0) for row in subset)
                ci_low, ci_high = wilson_interval(successes, len(subset))
                curves.append(
                    {
                        "model": label,
                        "family": config["family"],
                        "step": step,
                        "split": split,
                        "n": len(subset),
                        "success_rate": successes / len(subset) if subset else math.nan,
                        "success_ci_low": ci_low,
                        "success_ci_high": ci_high,
                        "final_score": mean_value(subset, "final_score"),
                        "score_improvement": mean_value(subset, "score_improvement"),
                        "invalid_action_rate": mean_value(subset, "invalid_action"),
                        "missing_action_tag_rate": mean_value(subset, "missing_action_tag"),
                        "strict_format_rate": mean_value(subset, "strict_format_correct_rate"),
                        "mean_primitive_steps": mean_value(subset, "n_primitive_steps"),
                    }
                )
            for task_type in TASK_ORDER:
                subset = [row for row in rows if row.get("task_type") == task_type]
                if not subset:
                    continue
                tasks.append(
                    {
                        "model": label,
                        "step": step,
                        "task_type": task_type,
                        "n": len(subset),
                        "success_rate": mean_value(subset, "traj_success"),
                        "final_score": mean_value(subset, "final_score"),
                        "score_improvement": mean_value(subset, "score_improvement"),
                        "invalid_action_rate": mean_value(subset, "invalid_action"),
                    }
                )
            counts: Counter[str] = Counter()
            for row in rows:
                counts.update(ACTION_RE.findall(str(row.get("action_sequence", ""))))
            total_actions = sum(counts.values())
            for action in VALID_ACTIONS:
                actions.append(
                    {
                        "model": label,
                        "step": step,
                        "action": action,
                        "count": counts[action],
                        "share": counts[action] / total_actions if total_actions else math.nan,
                    }
                )
    return pd.DataFrame(curves), pd.DataFrame(tasks), pd.DataFrame(actions)


def collect_offline_evaluation(source_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sweep_root = source_root / "evaluation/sweeps/active_spatial"
    navigation: list[dict] = []
    qa: list[dict] = []
    for label, directory in EVAL_DIRS.items():
        result_root = sweep_root / directory
        summary_path = result_root / "summary.csv"
        if summary_path.exists():
            with summary_path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    navigation.append(
                        {
                            "model": label,
                            "step": int(row["step"]),
                            "suite": row["suite"],
                            "status": row.get("status", ""),
                            "n": int(row["test_num_episodes"]) if row.get("test_num_episodes") else 0,
                            "success_rate": float(row["test_success_rate"])
                            if row.get("test_success_rate")
                            else math.nan,
                            "final_score": float(row["test_mean_final_score"])
                            if row.get("test_mean_final_score")
                            else math.nan,
                            "spl": float(row["test_spl"]) if row.get("test_spl") else math.nan,
                        }
                    )
        qa_path = result_root / "easi_results/easi_summary.json"
        if qa_path.exists():
            payload = json.loads(qa_path.read_text())
            for checkpoint_name, benchmark_values in payload.get("results", {}).items():
                step_match = re.search(r"_step(\d+)$", checkpoint_name)
                if not step_match:
                    continue
                step = int(step_match.group(1))
                for benchmark, value in benchmark_values.items():
                    qa.append(
                        {
                            "model": label,
                            "step": step,
                            "benchmark": benchmark,
                            "score": float(value),
                            "benchmarks_at_checkpoint": len(benchmark_values),
                        }
                    )
    return pd.DataFrame(navigation), pd.DataFrame(qa)


def build_inventory(training: pd.DataFrame, validation: pd.DataFrame, offline: pd.DataFrame, qa: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, config in RUNS.items():
        train_subset = training[training.model == label] if not training.empty else pd.DataFrame()
        val_subset = validation[(validation.model == label) & (validation.split == "All")]
        nav_subset = offline[(offline.model == label) & (offline.status == "ok")]
        qa_subset = qa[qa.model == label]
        max_train_step = int(train_subset.step.max()) if not train_subset.empty else 0
        max_val_step = int(val_subset.step.max()) if not val_subset.empty else 0
        rows.append(
            {
                "model": label,
                "family": config["family"],
                "variant": config["variant"],
                "run": config["run"],
                "expected_steps": config["expected_steps"],
                "max_train_step": max_train_step,
                "max_validation_step": max_val_step,
                "training_valid": label != "V49" and max_train_step >= config["expected_steps"],
                "validation_contract": config["validation_contract"],
                "offline_navigation_rows_ok": len(nav_subset),
                "easi_benchmarks_max": int(qa_subset.benchmarks_at_checkpoint.max()) if not qa_subset.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def build_model_summary(training: pd.DataFrame, validation: pd.DataFrame, offline: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in RUNS:
        val = validation[(validation.model == label) & (validation.split == "All")].sort_values("step")
        if val.empty:
            rows.append({"model": label, "valid_training": False})
            continue
        best = val.loc[val.success_rate.idxmax()]
        first, final = val.iloc[0], val.iloc[-1]
        train = training[training.model == label].sort_values("step")
        late = train[train.step > train.step.max() - 50] if not train.empty else train
        nav = offline[(offline.model == label) & (offline.status == "ok") & (offline.suite != "smoke")]
        id_rows = nav[nav.suite == "id_test"]
        ood_rows = nav[nav.suite.str.startswith("ood_", na=False)]
        offline_best_id = id_rows.loc[id_rows.success_rate.idxmax()] if not id_rows.empty else None
        ood_by_step = []
        for step, group in ood_rows.groupby("step"):
            total = group.n.sum()
            weighted = float((group.success_rate * group.n).sum() / total) if total else math.nan
            ood_by_step.append((int(step), weighted))
        offline_best_ood = max(ood_by_step, key=lambda item: item[1]) if ood_by_step else (math.nan, math.nan)
        rows.append(
            {
                "model": label,
                "valid_training": label != "V49",
                "val_baseline_success": first.success_rate,
                "val_peak_success": best.success_rate,
                "val_peak_step": int(best.step),
                "val_final_success": final.success_rate,
                "val_final_score": final.final_score,
                "val_final_invalid_rate": final.invalid_action_rate,
                "val_final_format_rate": final.strict_format_rate,
                "train_late_success_ma": late["episode/success_rate"].mean() if not late.empty else math.nan,
                "train_late_entropy": late["actor/entropy"].mean() if not late.empty else math.nan,
                "train_late_invalid_rate": late["transition/invalid_action_rate/mean"].mean()
                if not late.empty
                else math.nan,
                "offline_best_id_success": offline_best_id.success_rate if offline_best_id is not None else math.nan,
                "offline_best_id_step": int(offline_best_id.step) if offline_best_id is not None else math.nan,
                "offline_best_ood_weighted_success": offline_best_ood[1],
                "offline_best_ood_step": offline_best_ood[0],
            }
        )
    return pd.DataFrame(rows)


def configure_plots() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.titlesize": 14,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig: plt.Figure, figure_dir: Path, name: str) -> None:
    fig.savefig(figure_dir / f"{name}.png", facecolor="white")
    fig.savefig(figure_dir / f"{name}.pdf", facecolor="white")
    plt.close(fig)


def plot_validation_curves(validation: pd.DataFrame, figure_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    groups = ((axes[0], ["V46", "V47", "V48", "V50"], "Qwen clean variants"), (axes[1], ["Cambrian-C8", "Cambrian-C4"], "Cambrian-S clean variants"))
    for axis, labels, title in groups:
        for label in labels:
            data = validation[(validation.model == label) & (validation.split == "All")].sort_values("step")
            if data.empty:
                continue
            axis.plot(data.step, data.success_rate * 100, marker="o", ms=4, lw=2.2, color=COLORS[label], label=label)
            axis.fill_between(data.step, data.success_ci_low * 100, data.success_ci_high * 100, color=COLORS[label], alpha=0.09)
        axis.set_title(title)
        axis.set_xlabel("Training step")
        axis.set_ylabel("Validation success (%)")
        axis.set_ylim(5, 43)
        axis.legend(frameon=False)
    axes[0].text(0.01, 0.02, "n=176/checkpoint (4 samples per task)", transform=axes[0].transAxes, fontsize=9, color="#555555")
    axes[1].text(0.01, 0.02, "C8: n=44; C4: n=160, legacy suite", transform=axes[1].transAxes, fontsize=9, color="#555555")
    fig.suptitle("In-training validation: gains are modest and checkpoint-sensitive", y=1.03, fontsize=16, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, figure_dir, "01_validation_learning_curves")


def plot_offline_navigation(offline: pd.DataFrame, figure_dir: Path) -> None:
    data = offline[(offline.status == "ok") & (offline.model.isin(["V46", "V47", "V48", "V50"])) & (offline.suite != "smoke")].copy()
    rows = []
    for (model, step), group in data.groupby(["model", "step"]):
        id_group = group[group.suite == "id_test"]
        ood_group = group[group.suite.str.startswith("ood_")]
        if not id_group.empty:
            rows.append({"model": model, "step": step, "split": "ID", "success": id_group.success_rate.iloc[0]})
        if not ood_group.empty:
            total = ood_group.n.sum()
            rows.append({"model": model, "step": step, "split": "OOD (episode-weighted)", "success": (ood_group.success_rate * ood_group.n).sum() / total})
    aggregate = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    for axis, split in zip(axes, ("ID", "OOD (episode-weighted)")):
        subset = aggregate[aggregate.split == split]
        for label in ["V46", "V47", "V48", "V50"]:
            values = subset[subset.model == label].sort_values("step")
            axis.plot(values.step, values.success * 100, marker="o", lw=2.2, ms=5, color=COLORS[label], label=label)
        axis.axvline(60, ls="--", lw=1, color="#555555", alpha=0.6)
        axis.set_title(split)
        axis.set_xlabel("Checkpoint step")
        axis.set_ylabel("Offline navigation success (%)")
        axis.legend(frameon=False)
    axes[0].text(62, axes[0].get_ylim()[0] + 0.5, "actor updates begin", fontsize=8, color="#555555")
    fig.suptitle("Large-sample evaluation confirms navigation learning after critic warmup", y=1.03, fontsize=16, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, figure_dir, "02_offline_navigation")


def plot_ood_profile(offline: pd.DataFrame, figure_dir: Path) -> None:
    data = offline[(offline.status == "ok") & (offline.step == 700) & (offline.model.isin(["V46", "V47", "V48", "V50"]))]
    matrix = data.pivot(index="model", columns="suite", values="success_rate").reindex(index=["V46", "V47", "V48", "V50"], columns=SUITE_ORDER) * 100
    labels = ["ID", "Centering", "Scene", "Instance", "Category", "Template", "Geometry"]
    fig, axis = plt.subplots(figsize=(11.5, 4.0))
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="RdYlGn", vmin=10, vmax=60, linewidths=1, linecolor="white", cbar_kws={"label": "Success (%)"}, ax=axis)
    axis.set_xticklabels(labels, rotation=25, ha="right")
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.set_title("Step 700: transfer is strongest for scene/instance, weakest for category")
    fig.tight_layout()
    save_figure(fig, figure_dir, "03_ood_profile_step700")


def plot_training_stability(training: pd.DataFrame, figure_dir: Path) -> None:
    data = training[training.model.isin(["V46", "V47", "V48", "V50", "Cambrian-C8", "Cambrian-C4"])]
    specs = (
        ("episode/success_rate_ma25", "Train success, MA(25)", 100),
        ("actor/entropy_ma25", "Policy entropy, MA(25)", 1),
        ("transition/invalid_action_rate/mean_ma25", "Invalid action rate, MA(25)", 100),
        ("critic/explained_var_valid_ma25", "Critic explained variance, MA(25)", 1),
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
    for axis, (key, title, scale) in zip(axes.flat, specs):
        for label in ["V46", "V47", "V48", "V50", "Cambrian-C8", "Cambrian-C4"]:
            values = data[data.model == label].sort_values("step")
            if key not in values:
                continue
            y = values[key] * scale
            if "explained" in key:
                y = y.clip(-1, 1)
            axis.plot(values.step, y, lw=1.9, color=COLORS[label], label=label)
        axis.axvline(60, ls="--", lw=1, color="#555555", alpha=0.5)
        axis.set_title(title)
        axis.set_xlabel("Training step")
        if scale == 100:
            axis.set_ylabel("Percent")
    axes[0, 0].legend(ncol=2, frameon=False, loc="lower right")
    axes[1, 1].axhline(0, color="#222222", lw=0.8)
    fig.suptitle("Training dynamics: all runs learn, but entropy and syntax drift late", y=1.01, fontsize=16, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, figure_dir, "04_training_stability")


def plot_task_delta(tasks: pd.DataFrame, figure_dir: Path) -> None:
    rows = []
    for label in ["V46", "V47", "V48", "V50", "Cambrian-C8", "Cambrian-C4"]:
        model_tasks = tasks[tasks.model == label]
        if model_tasks.empty:
            continue
        first_step = int(model_tasks.step.min())
        final_step = int(model_tasks.step.max())
        first = model_tasks[model_tasks.step == first_step].set_index("task_type").final_score
        final = model_tasks[model_tasks.step == final_step].set_index("task_type").final_score
        for task in TASK_ORDER:
            if task in first.index and task in final.index:
                rows.append({"model": label, "task_type": task, "delta_pp": (final[task] - first[task]) * 100})
    delta = pd.DataFrame(rows).pivot(index="model", columns="task_type", values="delta_pp").reindex(index=["V46", "V47", "V48", "V50", "Cambrian-C8", "Cambrian-C4"], columns=TASK_ORDER)
    task_labels = ["Occlusion", "Equidistance", "Projective", "FoV", "Centering", "Size-distance"]
    fig, axis = plt.subplots(figsize=(11.5, 5.0))
    sns.heatmap(delta, annot=True, fmt="+.1f", center=0, cmap="vlag", vmin=-20, vmax=20, linewidths=1, linecolor="white", mask=delta.isna(), cbar_kws={"label": "Final - step 0 final score (pp)"}, ax=axis)
    axis.set_xticklabels(task_labels, rotation=25, ha="right")
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.set_title("Task-level score change is heterogeneous; aggregate gains hide regressions")
    axis.text(0.0, -0.29, "Validation task cells are small (n=4-72); use directionally, not as significance tests.", transform=axis.transAxes, fontsize=9, color="#555555")
    fig.tight_layout()
    save_figure(fig, figure_dir, "05_task_type_delta")


def plot_static_qa(qa: pd.DataFrame, figure_dir: Path) -> None:
    complete = qa[qa.benchmarks_at_checkpoint == 8]
    aggregate = complete.groupby(["model", "step"], as_index=False).score.mean()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for label in ["V46", "V47", "V48"]:
        values = aggregate[aggregate.model == label].sort_values("step")
        axes[0].plot(values.step, values.score * 100, marker="o", lw=2.2, ms=4, color=COLORS[label], label=label)
    axes[0].set_title("Complete EASI-8 mean")
    axes[0].set_xlabel("Checkpoint step")
    axes[0].set_ylabel("Mean benchmark score (%)")
    axes[0].legend(frameon=False)
    deltas = []
    for label in ["V46", "V47", "V48"]:
        values = aggregate[aggregate.model == label].sort_values("step")
        if len(values) >= 2:
            deltas.append({"model": label, "delta": (values.iloc[-1].score - values.iloc[0].score) * 100})
    delta_frame = pd.DataFrame(deltas)
    axes[1].bar(delta_frame.model, delta_frame.delta, color=[COLORS[x] for x in delta_frame.model], width=0.6)
    axes[1].axhline(0, color="#222222", lw=0.8)
    for index, row in delta_frame.iterrows():
        axes[1].text(index, row.delta + 0.03, f"{row.delta:+.2f} pp", ha="center", va="bottom", fontsize=10)
    axes[1].set_title("Step 50 to final transfer")
    axes[1].set_ylabel("EASI-8 change (percentage points)")
    axes[1].set_ylim(-0.2, max(0.8, delta_frame.delta.max() + 0.2))
    fig.suptitle("Embodied RL improves navigation, not broad static spatial QA", y=1.03, fontsize=16, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, figure_dir, "06_static_qa_transfer")


def fmt_pct(value: float) -> str:
    return "NA" if pd.isna(value) else f"{value * 100:.1f}%"


def write_report(output_dir: Path, inventory: pd.DataFrame, summary: pd.DataFrame, offline: pd.DataFrame, qa: pd.DataFrame) -> None:
    indexed = summary.set_index("model")
    qwen = [label for label in ["V46", "V47", "V48", "V50"] if label in indexed.index]
    final_nav = []
    for label in qwen:
        rows = offline[(offline.model == label) & (offline.step == 700) & (offline.status == "ok")]
        id_rows = rows[rows.suite == "id_test"]
        ood_rows = rows[rows.suite.str.startswith("ood_", na=False)]
        id_value = id_rows.success_rate.iloc[0] if not id_rows.empty else math.nan
        ood_value = (ood_rows.success_rate * ood_rows.n).sum() / ood_rows.n.sum() if not ood_rows.empty else math.nan
        final_nav.append((label, id_value, ood_value))
    nav_lines = "\n".join(f"| {label} | {fmt_pct(id_value)} | {fmt_pct(ood_value)} |" for label, id_value, ood_value in final_nav)
    val_lines = "\n".join(
        f"| {label} | {fmt_pct(indexed.loc[label, 'val_baseline_success'])} | "
        f"{fmt_pct(indexed.loc[label, 'val_peak_success'])} @ {int(indexed.loc[label, 'val_peak_step'])} | "
        f"{fmt_pct(indexed.loc[label, 'val_final_success'])} | "
        f"{fmt_pct(indexed.loc[label, 'val_final_invalid_rate'])} |"
        for label in ["V46", "V47", "V48", "V50", "Cambrian-C8", "Cambrian-C4"]
    )
    report = f"""# Active Spatial V46-V50 与 Cambrian-S 完整分析

生成日期：2026-09-06  
数据源：`/mnt/umm/users/yinbaiqiao/VAGEN-Lite`  
口径：只把 D0 old-logprob 修复后的 clean runs 作为训练结论；V49 单独标为无效。

## 一页结论

1. **训练确实学到了导航，但收益不大且不稳定。** 大样本离线评测中，Qwen 四版从 step 50 的 ID 约 15% 提升到 step 700 的 20%-23%，OOD 从约 20% 提升到 27%-34%。
2. **V47 的更强 KL=0.40 没有带来稳定性红利，反而是最弱终点。** step 700 ID 为 20.4%，明显低于 V46/V48/V50 的约 23.2%-23.4%；OOD 也只有 27.2%。
3. **W1 与 W3 没有形成清晰胜负。** V48-W1 与 V46-W3 的 step 700 ID 基本相同；V46 的中期峰值更强，但曲线在 step 300 明显回撤。单次 run 不足以支持“W3 必然更好”。
4. **V50 改善的是鲁棒性轮廓，不是总体 ID 上限。** 低 entropy、短 response 和微小 format reward 使最终 validation 格式最干净；离线 OOD 33.6% 为四版最高，但只比 V46 高 0.3pp，不能视为显著领先。
5. **真正的泛化瓶颈是 task semantics。** 所有模型在 scene/instance OOD 上明显更强，在 category OOD 上只有约 16%-20%。聚合 OOD 高于 ID 主要来自 suite 难度构成，不能解释成“模型在 OOD 上反而泛化更好”。
6. **RL 学到的是交互策略，不是通用静态空间能力。** 完整 EASI-8 上，V46/V47/V48 从 step 50 到最终仅变化约 0.4-0.6pp，几乎持平；与 navigation 的 8-13pp 增长形成鲜明对照。
7. **Cambrian-C8 有早期窗口，但后期退化。** 内部 validation 在 step 250 达到 31.8%，step 1000 回到 20.5%；末 50 步训练 entropy 约 4.10、invalid action 约 11.3%，不应直接使用 final checkpoint。
8. **Cambrian-C4 的高方差 reward bundle 更差。** validation 从 17.5% 到峰值 24.4%，且最终 invalid action 22.5%。C8 的 reward scaling + action-valid bundle 明显改善格式，但仍未解决后期 policy drift。

## 实验定义

| 版本 | 核心变量 | 有效性 |
| --- | --- | --- |
| V46 | Qwen-7B, W3, KL=0.30, entropy=0.005 | clean，有效，700 steps |
| V47 | V46 + KL 0.30 -> 0.40 | clean，有效，700 steps |
| V48 | V46 + window 3 -> 1 | clean，有效，700 steps |
| V49 | 复现 V26：W1, KL=0.20, legacy 7-type/OOD | **无效：bad renderer，仅 step 0** |
| V50 | V46 + entropy=0.001, response 384 -> 160, format reward=0.01 | clean，有效，700 steps |
| Cambrian-C8 | scaled reward + action-valid, deterministic validation | clean，有效，1000 steps |
| Cambrian-C4 | success reward=50, weaker shaping/action-valid, stochastic validation | clean，有效，1000 steps；与 C8 不是单因素对照 |

## 训练内验证

| 模型 | Step 0 | Peak | Final | Final invalid |
| --- | ---: | ---: | ---: | ---: |
{val_lines}

训练内 validation 的 Qwen 每点 n=176；Cambrian-C8 每点仅 n=44，置信区间更宽；Cambrian-C4 使用 legacy suite，不能直接按绝对值与 C8 排名。峰值是探索性 checkpoint selection，不能当作无偏估计。

![Validation curves](figures/01_validation_learning_curves.png)

## 大样本离线导航

Step 700 统一评测：

| 模型 | ID success | OOD weighted success |
| --- | ---: | ---: |
{nav_lines}

![Offline navigation](figures/02_offline_navigation.png)

![OOD profile](figures/03_ood_profile_step700.png)

V46 的全 checkpoint sweep 显示最佳 ID 是 step 250（24.3%），随后 step 300 掉到 18.4%，再逐步恢复。这个非单调性说明训练步数本身不是可靠 model-selection 规则，必须保留并评测中间 checkpoint。

## 训练稳定性

![Training stability](figures/04_training_stability.png)

- 所有 run 在 critic warmup=60 前 actor 都不更新，因此 step 50 近似共同 pretrained anchor；真正的版本差异应从 step 100 后判断。
- Qwen 后期 train success 仍高于 validation，但 entropy 和 invalid action 同时上升，存在 train-distribution overfit / sampling drift。
- Cambrian 的 critic explained variance 大量时间接近或低于 0，特别是高方差 C4；value model 对 return 的解释能力不足。
- rollout IS 的 ESS/N 长期接近 1，说明 D0 修复后的 runtime correction 本身不是这些曲线退化的主要原因。

## 任务与能力迁移

![Task deltas](figures/05_task_type_delta.png)

![Static QA](figures/06_static_qa_transfer.png)

任务级结果不支持“整体空间能力统一提升”：同一 checkpoint 往往在 projective/instance 类任务上提升，却在 occlusion、centering 或 category 上持平甚至回退。建议后续将 reward 与 validation 都改成按 task type 平衡，而不是只追总体 success。

## 数据完整性与不能下的结论

- V49 没有有效训练，不能补点、插值或与 V46-V50 连线。
- Cambrian-C8/C4 的恢复后离线 navigation 文件仍是 `missing`，因此不能宣称 Cambrian 优于或弱于 Qwen 的大样本 ID/OOD 表现。
- V50 最新 EASI 文件只含 3DSRBench 和 EmbSpatial 两项，不是 EASI-8；静态 QA 主结论只使用 V46/V47/V48 的完整 8 项结果。
- 目前每个 recipe 只有一个 seed。小于约 1pp 的版本差异不应包装成确定性结论。
- C4 与 C8 同时改变 reward scale、potential scale、format penalty、rollout 数和 validation sampling，不能把差异归因于单一超参。

## 组会建议主线

1. 先交代 D0 clean restart，历史旧 run 只作背景，不作方法证据。
2. 用图 02 证明“RL 有效”：critic warmup 后 navigation 明显上升。
3. 用图 03 说明“提升不均匀”：instance/scene 强，category 弱。
4. 用图 06 给出最重要 insight：交互导航能力提升，但静态空间 QA 几乎不动。
5. 用图 04 收束到下一步：需要 task-balanced reward、早停/中间 checkpoint 选择，以及 Cambrian 外部 eval 补齐，而不是继续单纯延长训练。

## 可复现方式

```bash
/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python \\
  scripts/analyze_active_spatial_training.py
```

机器可读表位于本目录 `tables/`；图同时提供 PNG 和 PDF。
"""
    (output_dir / "GROUP_MEETING_REPORT_CN.md").write_text(report)


def write_slide_outline(output_dir: Path) -> None:
    content = """# 组会展示提纲（8页）

## 1. 问题与实验矩阵

- 科学问题：不同 credit window、KL、格式约束和 backbone 是否改善 embodied spatial navigation？
- 强调只分析 D0 修复后的 clean runs；V49 因 renderer failure 无有效训练。
- 展示实验定义表，不先报结论。

## 2. RL 是否真的有效？

- 放 `figures/02_offline_navigation.png`。
- 讲法：step 50 仍在 critic warmup，之后 ID 提升约 5-8pp、OOD 提升约 7-14pp。
- 结论：有效，但曲线非单调，必须做 checkpoint selection。

## 3. Qwen 版本对比

- V47 强 KL 最弱；V48-W1 没有明显输给 V46-W3；V50 OOD 略好但差距很小。
- 不把单 seed、<1pp 差异解释成显著提升。

## 4. 泛化发生在哪里？

- 放 `figures/03_ood_profile_step700.png`。
- scene/instance 提升强，category 持续弱；OOD aggregate 受 suite 难度影响。

## 5. 训练为什么不稳定？

- 放 `figures/04_training_stability.png`。
- entropy 与 invalid action 后期上升；critic EV 弱；V46 step 250 后曾明显回撤。

## 6. Cambrian-S 的结论

- 放 `figures/01_validation_learning_curves.png` 右图。
- C8 step 250 是候选窗口，final 退化；C4 格式错误长期更高。
- 外部评测仍缺失，不做跨 backbone 排名。

## 7. 最关键 insight

- 放 `figures/06_static_qa_transfer.png`。
- Navigation 增长明显，EASI-8 几乎不变：RL 更像学会交互策略，而非提升通用空间表征。

## 8. 下一步决策

- 补齐 Cambrian-C8 step 100/250/1000 的统一离线 ID/OOD + EASI-8。
- 对 V46/V48/V50 最有希望的 checkpoint 做至少 3 seeds 或 paired bootstrap。
- 引入 task-balanced objective/selection，重点解决 category、occlusion、centering。
- 训练侧加入 entropy/invalid-action early-stop gate，并长期保留中间 checkpoint。
"""
    (output_dir / "SLIDE_OUTLINE_CN.md").write_text(content)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    training = collect_training(args.source_root)
    validation, tasks, actions = collect_validation(args.source_root)
    offline, qa = collect_offline_evaluation(args.source_root)
    inventory = build_inventory(training, validation, offline, qa)
    summary = build_model_summary(training, validation, offline)

    frames = {
        "run_inventory.csv": inventory,
        "model_summary.csv": summary,
        "training_curves.csv": training,
        "validation_curves.csv": validation,
        "validation_task_breakdown.csv": tasks,
        "validation_action_distribution.csv": actions,
        "offline_navigation.csv": offline,
        "easi_qa.csv": qa,
    }
    for name, frame in frames.items():
        frame.to_csv(table_dir / name, index=False)

    configure_plots()
    plot_validation_curves(validation, figure_dir)
    plot_offline_navigation(offline, figure_dir)
    plot_ood_profile(offline, figure_dir)
    plot_training_stability(training, figure_dir)
    plot_task_delta(tasks, figure_dir)
    plot_static_qa(qa, figure_dir)
    write_report(output_dir, inventory, summary, offline, qa)
    write_slide_outline(output_dir)

    print(f"Wrote analysis pack to {output_dir}")
    print(inventory.to_string(index=False))


if __name__ == "__main__":
    main()
