#!/usr/bin/env python3
"""Build a fail-closed three-seed Wan D1/D2 development report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

MANIFEST_SHA256 = "781e25848de48fa5219aad876bd10cb8f9445328b145a19193a6f6b2ab5efcc9"


LABEL = "development-only / Wan2.1-T2V-1.3B / three-seed"
METRICS = ("mae", "absrel", "delta1")
D1_CONDITIONS = {
    "true_target", "train_layout_prior", "within_scene_wrong_target",
    "across_scene_shuffled_target", "last_context", "zero", "random",
    "spatial_token_shuffle",
}
D2_CONDITIONS = {
    "true_motion", "identity_motion", "matched_horizon_shuffled_motion",
    "independently_trained_context_only", "last_context",
    "true_pooled_target_reference", "train_layout_prior",
}
COMPARISONS = {
    "D1 target-prior": ("D1", "true_target", "train_layout_prior"),
    "D1 target-wrong": ("D1", "true_target", "within_scene_wrong_target"),
    "D1 target-copy": ("D1", "true_target", "last_context"),
    "D2 motion-identity": ("D2", "true_motion", "identity_motion"),
    "D2 motion-shuffled": ("D2", "true_motion", "matched_horizon_shuffled_motion"),
    "D2 motion-context-only": ("D2", "true_motion", "independently_trained_context_only"),
}


def scene_macro(records, metric: str) -> float:
    grouped = defaultdict(list)
    for row in records:
        value = float(row[metric])
        if np.isfinite(value):
            grouped[row["scene_id"]].append(value)
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def scene_cluster_paired_bootstrap(records, lhs: str, rhs: str, *, metric: str,
                                   pair_key: str, n_boot: int, seed: int):
    pairs = defaultdict(dict)
    for row in records:
        if row["condition"] in {lhs, rhs}:
            pairs[(row["scene_id"], row[pair_key])][row["condition"]] = float(row[metric])
    scenes = defaultdict(list)
    for (scene, _), values in pairs.items():
        if lhs in values and rhs in values:
            scenes[scene].append(values[lhs] - values[rhs])
    values = np.asarray([np.mean(v) for v in scenes.values()], dtype=np.float64)
    if not len(values):
        return {"delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_scenes": 0}
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"delta": float(values.mean()), "ci_low": float(lo), "ci_high": float(hi), "n_scenes": int(len(values))}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_seed_runs(values: list[str]) -> dict[int, Path]:
    result = {}
    for value in values:
        seed_text, separator, root_text = value.partition("=")
        if not separator:
            raise ValueError(f"expected SEED=PATH, got {value!r}")
        seed = int(seed_text)
        if seed in result:
            raise ValueError(f"duplicate seed root: {seed}")
        result[seed] = Path(root_text).resolve()
    if set(result) != {0, 1, 2}:
        raise ValueError(f"exactly seeds 0,1,2 are required, got {sorted(result)}")
    return result


def load_stage(root: Path, seed: int, name: str, experiment: str, branch: str,
               rows_expected: int, conditions: set[str]):
    stage = root / name
    metrics_path, receipt_path, checkpoint_path = (
        stage / "per_sample_metrics.csv", stage / "run_receipt.json", stage / "checkpoint.pt"
    )
    for path in (metrics_path, receipt_path, checkpoint_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing or empty artifact: {path}")
    receipt = json.loads(receipt_path.read_text())
    expected = {
        "status": "complete_development_evaluation", "experiment": experiment,
        "model": "wan21_t2v_1p3b", "branch": branch, "seed": seed,
        "manifest_sha256": MANIFEST_SHA256, "n_dev": 552, "n_dev_scenes": 69,
        "test_scenes": 0,
    }
    if experiment == "D1":
        expected |= {"reference": "isolated_target", "depth_only": True}
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RuntimeError(f"receipt mismatch {key} in {receipt_path}: {receipt.get(key)!r} != {value!r}")
    with metrics_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != rows_expected or {int(row["seed"]) for row in rows} != {seed}:
        raise RuntimeError(f"row/seed mismatch in {metrics_path}")
    if {row["condition"] for row in rows} != conditions:
        raise RuntimeError(f"condition mismatch in {metrics_path}")
    for horizon in (2, 4):
        for condition in conditions:
            selected = [row for row in rows if int(row["horizon"]) == horizon and row["condition"] == condition]
            if len(selected) != 276 or len({row["scene_id"] for row in selected}) != 69:
                raise RuntimeError(f"incomplete h{horizon}/{condition} in {metrics_path}")
    return rows, {
        "seed": seed, "stage": name, "metrics": str(metrics_path), "metrics_sha256": sha256(metrics_path),
        "receipt": str(receipt_path), "receipt_sha256": sha256(receipt_path),
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
    }


def load(seed_runs: dict[int, Path]):
    records, sources = [], []
    for seed, root in sorted(seed_runs.items()):
        if not root.is_dir():
            raise RuntimeError(f"missing seed root: {root}")
        specs = (
            (f"d1_wan_spatial_s{seed}", "D1", "spatial", 4416, D1_CONDITIONS),
            (f"d1_wan_pooled_s{seed}", "D1", "pooled", 4416, D1_CONDITIONS),
            (f"d2_wan_pooled_s{seed}", "D2", "pooled", 3864, D2_CONDITIONS),
        )
        for spec in specs:
            rows, source = load_stage(root, seed, *spec)
            records.extend(rows)
            sources.append(source)
    return records, sources


def per_seed_rows(records):
    grouped = defaultdict(list)
    for row in records:
        if row["split"] != "dev" or row["manifest_sha256"] != MANIFEST_SHA256:
            raise RuntimeError(f"non-frozen row: {row.get('source_run')}")
        grouped[(row["experiment"], row["branch"], int(row["horizon"]), row["condition"], int(row["seed"]))].append(row)
    output = []
    for key, rows in sorted(grouped.items()):
        experiment, branch, horizon, condition, seed = key
        if len(rows) != 276 or len({row["scene_id"] for row in rows}) != 69:
            raise RuntimeError(f"incomplete grouped record: {key}")
        output.append({
            "label": LABEL, "experiment": experiment, "branch": branch, "horizon": horizon,
            "condition": condition, "seed": seed, "split": "dev", "n": 276, "scenes": 69,
            "manifest_sha256": MANIFEST_SHA256, "source_run": ";".join(sorted({row["source_run"] for row in rows})),
            **{f"scene_macro_{metric}": scene_macro(rows, metric) for metric in METRICS},
        })
    return output


def aggregate_rows(per_seed):
    grouped = defaultdict(list)
    for row in per_seed:
        grouped[tuple(row[field] for field in ("experiment", "branch", "horizon", "condition"))].append(row)
    output = []
    for key, rows in sorted(grouped.items()):
        if sorted(int(row["seed"]) for row in rows) != [0, 1, 2]:
            raise RuntimeError(f"missing seed: {key}")
        entry = dict(zip(("experiment", "branch", "horizon", "condition"), key))
        entry |= {"label": LABEL, "seeds": "0;1;2", "n_seeds": 3, "split": "dev", "manifest_sha256": MANIFEST_SHA256}
        for metric in METRICS:
            values = [float(row[f"scene_macro_{metric}"]) for row in rows]
            entry |= {f"{metric}_mean_across_seeds": statistics.mean(values),
                      f"{metric}_sd_across_seeds": statistics.stdev(values),
                      f"{metric}_min_seed": min(values), f"{metric}_max_seed": max(values)}
        output.append(entry)
    return output


def comparison_rows(records, per_seed):
    output = []
    for branch in ("spatial", "pooled"):
        for horizon in (2, 4):
            for name, (experiment, lhs, rhs) in COMPARISONS.items():
                if experiment == "D2" and branch != "pooled":
                    continue
                group = [dict(row, pair_id=f"{row['seed']}:{row['sample_id']}") for row in records
                         if row["experiment"] == experiment and row["branch"] == branch and int(row["horizon"]) == horizon]
                entry = {"label": LABEL, "experiment": experiment, "branch": branch, "horizon": horizon,
                         "comparison": name, "lhs_condition": lhs, "rhs_condition": rhs,
                         "difference_definition": "lhs_minus_rhs", "n_seeds": 3, "scenes": 69,
                         "split": "dev", "manifest_sha256": MANIFEST_SHA256}
                for metric in METRICS:
                    ci = scene_cluster_paired_bootstrap(group, lhs, rhs, metric=metric, pair_key="pair_id", n_boot=10_000, seed=20270923)
                    deltas = []
                    for seed in (0, 1, 2):
                        def get(condition):
                            return next(row for row in per_seed if row["experiment"] == experiment and row["branch"] == branch
                                        and row["horizon"] == horizon and row["condition"] == condition and row["seed"] == seed)
                        deltas.append(float(get(lhs)[f"scene_macro_{metric}"]) - float(get(rhs)[f"scene_macro_{metric}"]))
                    if ci["n_scenes"] != 69:
                        raise RuntimeError(f"incomplete paired comparison: {name}/h{horizon}")
                    entry |= {f"{metric}_delta": ci["delta"], f"{metric}_ci_low": ci["ci_low"],
                              f"{metric}_ci_high": ci["ci_high"], f"{metric}_seed_delta_mean": statistics.mean(deltas),
                              f"{metric}_seed_delta_sd": statistics.stdev(deltas)}
                output.append(entry)
    return output


def write_csv(path: Path, rows: list[dict]):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def write_markdown(path: Path, aggregates, comparisons):
    lines = [
        "# Wan2.1 D1/D2 three-seed development report", "",
        "**development-only — no held-out test split; all differences are lhs minus rhs.**", "",
        "## Paired comparisons", "",
        "Negative MAE/AbsRel and positive Delta1 favour the lhs. Intervals are 10,000-draw paired 69-scene cluster bootstraps over all three seeds.", "",
        "| experiment | branch | h | comparison | MAE delta [95% CI] | AbsRel delta [95% CI] | Delta1 delta [95% CI] |", "|---|---|---:|---|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(f"| {row['experiment']} | {row['branch']} | {row['horizon']} | {row['comparison']} | "
                     f"{row['mae_delta']:.6f} [{row['mae_ci_low']:.6f}, {row['mae_ci_high']:.6f}] | "
                     f"{row['absrel_delta']:.6f} [{row['absrel_ci_low']:.6f}, {row['absrel_ci_high']:.6f}] | "
                     f"{row['delta1_delta']:.6f} [{row['delta1_ci_low']:.6f}, {row['delta1_ci_high']:.6f}] |")
    lines += ["", "## Across-seed metric summary", "", "| experiment | branch | h | condition | MAE mean±SD | AbsRel mean±SD | Delta1 mean±SD |", "|---|---|---:|---|---:|---:|---:|"]
    for row in aggregates:
        lines.append(f"| {row['experiment']} | {row['branch']} | {row['horizon']} | {row['condition']} | "
                     f"{row['mae_mean_across_seeds']:.6f}±{row['mae_sd_across_seeds']:.6f} | "
                     f"{row['absrel_mean_across_seeds']:.6f}±{row['absrel_sd_across_seeds']:.6f} | "
                     f"{row['delta1_mean_across_seeds']:.6f}±{row['delta1_sd_across_seeds']:.6f} |")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-run", action="append", required=True, metavar="SEED=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    seed_runs = parse_seed_runs(args.seed_run)
    records, sources = load(seed_runs)
    per_seed = per_seed_rows(records)
    aggregates = aggregate_rows(per_seed)
    comparisons = comparison_rows(records, per_seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "per_seed_metrics.csv", per_seed)
    write_csv(args.output_dir / "across_seed_metrics.csv", aggregates)
    write_csv(args.output_dir / "paired_comparisons.csv", comparisons)
    summary = {"schema_version": 1, "status": "COMPLETE_DEVELOPMENT_ONLY", "label": LABEL,
               "manifest_sha256": MANIFEST_SHA256, "seed_runs": {str(k): str(v) for k, v in seed_runs.items()},
               "per_seed_rows": len(per_seed), "aggregate_rows": len(aggregates),
               "comparison_rows": len(comparisons), "sources": sources}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_markdown(args.output_dir / "report.md", aggregates, comparisons)
    print(json.dumps({"status": summary["status"], "aggregate_rows": len(aggregates), "comparison_rows": len(comparisons)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
