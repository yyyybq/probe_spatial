#!/usr/bin/env python3
"""Generate canonical result CSVs from experiments/registry.csv.

Only COMPLETE registry rows are eligible.  Best results are intentionally not
stored as lifecycle state; downstream analyses should filter these canonical
tables dynamically.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "experiments/registry.csv"
C_RAW = ROOT / "logs/inscene15k_streaming/results/c_eval_with_extra_20260728/c_series_raw_metrics_with_extra.csv"
WAN_COG = ROOT / "logs/probing_jobs_20260725_ab/recovery_20260730/perf_analysis/evaluation_results_all_probes.tsv"


def read_csv(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def parse_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"n/a", "nan", "none", "?"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def registry() -> list[dict[str, str]]:
    return [row for row in read_csv(REGISTRY) if row.get("status") == "COMPLETE"]


def c_lookup() -> dict[tuple[str, str, str, str], dict[str, str]]:
    out = {}
    for row in read_csv(C_RAW):
        key = (row.get("probe_type", ""), row.get("vfm_name", ""), row.get("layer", ""), row.get("prefix", ""))
        out[key] = row
    return out


def wan_cog_lookup() -> dict[str, dict[str, str]]:
    return {row.get("job_id", ""): row for row in read_csv(WAN_COG, delimiter="\t")}


def simple_rows(family: str) -> list[dict[str, object]]:
    out = []
    for row in registry():
        if row.get("family") != family:
            continue
        value = parse_float(row.get("metric_value"))
        if value is None:
            continue
        out.append(
            {
                "experiment_id": row["experiment_id"],
                "family": row["family"],
                "probe": row["probe"],
                "model": row["model"],
                "model_variant": row["model_variant"],
                "layer": row["layer"],
                "prefix": row["prefix"],
                "horizon": row["horizon"],
                "seed": row["seed"],
                "eval_split": row["eval_split"],
                "metric_name": row["metric_name"],
                "metric_value": value,
                "status": row["status"],
                "raw_artifact_path": row["raw_artifact_path"],
            }
        )
    return out


def build_a1(out_dir: Path) -> None:
    fields = [
        "experiment_id",
        "family",
        "probe",
        "model",
        "model_variant",
        "layer",
        "prefix",
        "horizon",
        "seed",
        "eval_split",
        "metric_name",
        "metric_value",
        "status",
        "raw_artifact_path",
    ]
    write_csv(out_dir / "a1_results.csv", simple_rows("A1"), fields)


def build_a2(out_dir: Path) -> None:
    fields = [
        "experiment_id",
        "family",
        "probe",
        "model",
        "model_variant",
        "layer",
        "prefix",
        "horizon",
        "seed",
        "eval_split",
        "metric_name",
        "metric_value",
        "status",
        "raw_artifact_path",
    ]
    write_csv(out_dir / "a2_results.csv", simple_rows("A2"), fields)


def build_b2(out_dir: Path, wan_rows: dict[str, dict[str, str]]) -> None:
    rows = []
    for row in registry():
        if row.get("family") != "B2" or row.get("metric_name") != "mean_ang_err_deg":
            continue
        value = parse_float(row.get("metric_value"))
        if value is None:
            continue
        raw = wan_rows.get(row.get("launcher_job_id", ""), {})
        rows.append(
            {
                "experiment_id": row["experiment_id"],
                "model": row["model"],
                "model_variant": row["model_variant"],
                "layer": row["layer"],
                "prefix": row["prefix"],
                "seed": row["seed"],
                "eval_split": row["eval_split"],
                "angular_error_mean": value,
                "angular_error_median": "",
                "top1": raw.get("top1", ""),
                "top3": raw.get("top3", ""),
                "mean_log_dist_err": raw.get("mean_log_dist_err", ""),
                "status": row["status"],
                "raw_artifact_path": row["raw_artifact_path"],
            }
        )
    fields = [
        "experiment_id",
        "model",
        "model_variant",
        "layer",
        "prefix",
        "seed",
        "eval_split",
        "angular_error_mean",
        "angular_error_median",
        "top1",
        "top3",
        "mean_log_dist_err",
        "status",
        "raw_artifact_path",
    ]
    write_csv(out_dir / "b2_results.csv", rows, fields)


def build_c1(out_dir: Path, lookup: dict[tuple[str, str, str, str], dict[str, str]]) -> None:
    rows = []
    for row in registry():
        if row.get("family") != "C1":
            continue
        raw = lookup.get((row["probe"], row["model"], row["layer"], row["prefix"]), {})
        global_r1 = parse_float(raw.get("global_R@1"))
        no_action = parse_float(raw.get("no_action_global_R@1"))
        shuffled = parse_float(raw.get("shuffled_action_global_R@1"))
        copy = parse_float(raw.get("last_observation_global_R@1"))
        rows.append(
            {
                "experiment_id": row["experiment_id"],
                "model": row["model"],
                "layer": row["layer"],
                "prefix": row["prefix"],
                "seed": row["seed"],
                "eval_split": row["eval_split"],
                "global_R@1": global_r1 if global_r1 is not None else "",
                "global_R@5": raw.get("global_R@5", ""),
                "no_action_R@1": no_action if no_action is not None else "",
                "shuffled_action_R@1": shuffled if shuffled is not None else "",
                "last_observation_R@1": copy if copy is not None else "",
                "action_gain_vs_no_action": global_r1 - no_action if global_r1 is not None and no_action is not None else "",
                "action_gain_vs_shuffle": global_r1 - shuffled if global_r1 is not None and shuffled is not None else "",
                "gain_over_copy": global_r1 - copy if global_r1 is not None and copy is not None else "",
                "action_gain_cos": raw.get("action_gain_cos", ""),
                "status": row["status"],
                "raw_artifact_path": row["raw_artifact_path"],
            }
        )
    fields = [
        "experiment_id",
        "model",
        "layer",
        "prefix",
        "seed",
        "eval_split",
        "global_R@1",
        "global_R@5",
        "no_action_R@1",
        "shuffled_action_R@1",
        "last_observation_R@1",
        "action_gain_vs_no_action",
        "action_gain_vs_shuffle",
        "gain_over_copy",
        "action_gain_cos",
        "status",
        "raw_artifact_path",
    ]
    write_csv(out_dir / "c1_results.csv", rows, fields)


def build_c2(out_dir: Path, lookup: dict[tuple[str, str, str, str], dict[str, str]]) -> None:
    rows = []
    for row in registry():
        if row.get("family") != "C2":
            continue
        raw = lookup.get((row["probe"], row["model"], row["layer"], row["prefix"]), {})
        rows.append(
            {
                "experiment_id": row["experiment_id"],
                "model": row["model"],
                "layer": row["layer"],
                "prefix": row["prefix"],
                "horizon": "all",
                "seed": row["seed"],
                "eval_split": row["eval_split"],
                "metric_name": "global_R@1",
                "metric_value": raw.get("global_R@1", ""),
                "global_R@5": raw.get("global_R@5", ""),
                "final_pose_error": raw.get("final_pose_error", ""),
                "drift_rate": raw.get("drift_rate", ""),
                "loop_closure_error": raw.get("loop_closure_error", ""),
                "status": row["status"],
                "raw_artifact_path": row["raw_artifact_path"],
            }
        )
        for horizon in ["1", "2", "4"]:
            value = raw.get(f"pose_error_h{horizon}", "")
            if value == "":
                continue
            rows.append(
                {
                    "experiment_id": row["experiment_id"],
                    "model": row["model"],
                    "layer": row["layer"],
                    "prefix": row["prefix"],
                    "horizon": horizon,
                    "seed": row["seed"],
                    "eval_split": row["eval_split"],
                    "metric_name": "pose_error",
                    "metric_value": value,
                    "global_R@5": "",
                    "final_pose_error": "",
                    "drift_rate": "",
                    "loop_closure_error": "",
                    "status": row["status"],
                    "raw_artifact_path": row["raw_artifact_path"],
                }
            )
    fields = [
        "experiment_id",
        "model",
        "layer",
        "prefix",
        "horizon",
        "seed",
        "eval_split",
        "metric_name",
        "metric_value",
        "global_R@5",
        "final_pose_error",
        "drift_rate",
        "loop_closure_error",
        "status",
        "raw_artifact_path",
    ]
    write_csv(out_dir / "c2_results.csv", rows, fields)


def build_c3(out_dir: Path, lookup: dict[tuple[str, str, str, str], dict[str, str]]) -> None:
    rows = []
    for row in registry():
        if row.get("family") != "C3":
            continue
        raw = lookup.get((row["probe"], row["model"], row["layer"], row["prefix"]), {})
        for metric in ["intervention_validity", "intervention_margin", "counterfactual_consistency", "global_R@1"]:
            rows.append(
                {
                    "experiment_id": row["experiment_id"],
                    "model": row["model"],
                    "layer": row["layer"],
                    "prefix": row["prefix"],
                    "horizon": "all",
                    "seed": row["seed"],
                    "eval_split": row["eval_split"],
                    "metric_name": metric,
                    "metric_value": raw.get(metric, ""),
                    "status": row["status"],
                    "raw_artifact_path": row["raw_artifact_path"],
                }
            )
        for horizon in ["1", "2", "4"]:
            value = raw.get(f"cf_cos_h{horizon}", "")
            if value == "":
                continue
            rows.append(
                {
                    "experiment_id": row["experiment_id"],
                    "model": row["model"],
                    "layer": row["layer"],
                    "prefix": row["prefix"],
                    "horizon": horizon,
                    "seed": row["seed"],
                    "eval_split": row["eval_split"],
                    "metric_name": "cf_cos",
                    "metric_value": value,
                    "status": row["status"],
                    "raw_artifact_path": row["raw_artifact_path"],
                }
            )
    fields = [
        "experiment_id",
        "model",
        "layer",
        "prefix",
        "horizon",
        "seed",
        "eval_split",
        "metric_name",
        "metric_value",
        "status",
        "raw_artifact_path",
    ]
    write_csv(out_dir / "c3_results.csv", rows, fields)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="experiments/registry.csv")
    parser.add_argument("--output-dir", default="results/canonical")
    args = parser.parse_args()
    global REGISTRY
    REGISTRY = ROOT / args.registry
    out_dir = ROOT / args.output_dir
    lookup = c_lookup()
    wan_rows = wan_cog_lookup()
    build_a1(out_dir)
    build_a2(out_dir)
    build_b2(out_dir, wan_rows)
    build_c1(out_dir, lookup)
    build_c2(out_dir, lookup)
    build_c3(out_dir, lookup)
    print(f"Wrote canonical results under {out_dir}")


if __name__ == "__main__":
    main()
