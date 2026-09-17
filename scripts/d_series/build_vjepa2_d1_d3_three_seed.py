#!/usr/bin/env python3
"""Build a fail-closed, development-only V-JEPA2 D1/D3 three-seed report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.d_series.feature_store import MANIFEST_SHA256
from vidfm3d.utils.layout_metrics import scene_cluster_paired_bootstrap, scene_macro


LABEL = "development-only / V-JEPA2 / three-seed"
METRICS = ("mae", "absrel", "delta1")
D1_CONDITIONS = {
    "true_target", "train_layout_prior", "within_scene_wrong_target",
    "across_scene_shuffled_target", "last_context", "zero", "random",
    "spatial_token_shuffle",
}
D3_AB_CONDITIONS = {
    "native_target_reference", "frozen_readout_transfer", "train_layout_prior",
    "within_scene_wrong_target", "last_context",
}
D3_FULL_CONDITIONS = D3_AB_CONDITIONS | {
    "tokenwise_linear_adapter", "retrained_same_capacity_readout",
}
COMPARISONS = {
    "B-A": ("frozen_readout_transfer", "native_target_reference"),
    "C-B": ("tokenwise_linear_adapter", "frozen_readout_transfer"),
    "D-B": ("retrained_same_capacity_readout", "frozen_readout_transfer"),
    "D-prior": ("retrained_same_capacity_readout", "train_layout_prior"),
}


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


def expected_stages(seed: int):
    for reference in ("isolated_target", "native_target"):
        for branch in ("spatial", "pooled"):
            yield f"d1_{reference.removesuffix('_target')}_{branch}_s{seed}", "D1", branch, reference, None, 4416, D1_CONDITIONS
    for branch in ("spatial", "pooled"):
        yield f"d3_ab_{branch}_s{seed}", "D3", branch, None, "AB", 2760, D3_AB_CONDITIONS
    for branch in ("spatial", "pooled"):
        yield f"d3_abcd_{branch}_s{seed}", "D3", branch, None, None, 3864, D3_FULL_CONDITIONS


def load_stage(
    root: Path, seed: int, name: str, experiment: str, branch: str,
    reference: str | None, phase: str | None, expected_rows: int,
    expected_conditions: set[str],
):
    stage = root / name
    receipt_path = stage / "run_receipt.json"
    metrics_path = stage / "per_sample_metrics.csv"
    checkpoint_path = stage / "checkpoint.pt"
    for path in (receipt_path, metrics_path, checkpoint_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing or empty required artifact: {path}")
    receipt = json.loads(receipt_path.read_text())
    expected = {
        "status": "complete_development_evaluation", "experiment": experiment,
        "model": "vjepa2_vitl256", "branch": branch, "seed": seed,
        "manifest_sha256": MANIFEST_SHA256, "n_dev": 552,
        "n_dev_scenes": 69, "test_scenes": 0,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RuntimeError(f"receipt mismatch {key} in {receipt_path}: {receipt.get(key)!r} != {value!r}")
    if reference is not None and receipt.get("reference") != reference:
        raise RuntimeError(f"reference mismatch in {receipt_path}")
    if phase is not None:
        if receipt.get("phase") != phase:
            raise RuntimeError(f"phase mismatch in {receipt_path}")
    elif "phase" in receipt:
        raise RuntimeError(f"unexpected phase in {receipt_path}")
    with metrics_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_rows:
        raise RuntimeError(f"row count mismatch in {metrics_path}: {len(rows)} != {expected_rows}")
    if {int(row["seed"]) for row in rows} != {seed}:
        raise RuntimeError(f"seed mismatch in {metrics_path}")
    if {row["condition"] for row in rows} != expected_conditions:
        raise RuntimeError(f"condition mismatch in {metrics_path}")
    for horizon in (2, 4):
        horizon_rows = [row for row in rows if int(row["horizon"]) == horizon]
        for condition in expected_conditions:
            selected = [row for row in horizon_rows if row["condition"] == condition]
            if len(selected) != 276 or len({row["scene_id"] for row in selected}) != 69:
                raise RuntimeError(f"incomplete h{horizon}/{condition} in {metrics_path}")
    source = {
        "seed": seed, "stage": name, "metrics": str(metrics_path),
        "metrics_sha256": sha256(metrics_path), "receipt": str(receipt_path),
        "receipt_sha256": sha256(receipt_path), "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
    }
    return rows, source


def load(seed_runs: dict[int, Path]):
    records, sources = [], []
    for seed, root in sorted(seed_runs.items()):
        if not root.is_dir():
            raise RuntimeError(f"missing seed run root: {root}")
        for spec in expected_stages(seed):
            rows, source = load_stage(root, seed, *spec)
            sources.append(source)
            # The full A/B/C/D stage is the authoritative D3 table. The AB
            # stage remains required and audited but is not loaded twice.
            if spec[4] != "AB":
                records.extend(rows)
    return records, sources


def seed_rows(records):
    grouped = defaultdict(list)
    for row in records:
        if row["split"] != "dev" or row["manifest_sha256"] != MANIFEST_SHA256:
            raise RuntimeError(f"non-frozen row: {row.get('source_run')}")
        key = (
            row["experiment"], row.get("reference", ""), row["branch"],
            int(row["horizon"]), row["condition"], int(row["seed"]),
        )
        grouped[key].append(row)
    output = []
    for key, rows in sorted(grouped.items()):
        experiment, reference, branch, horizon, condition, seed = key
        if len(rows) != 276 or len({row["scene_id"] for row in rows}) != 69:
            raise RuntimeError(f"incomplete grouped records: {key}")
        output.append({
            "label": LABEL, "experiment": experiment, "reference": reference,
            "branch": branch, "horizon": horizon, "condition": condition,
            "seed": seed, **{f"scene_macro_{metric}": scene_macro(rows, metric) for metric in METRICS},
            "n": len(rows), "scenes": 69, "split": "dev",
            "manifest_sha256": MANIFEST_SHA256,
            "source_run": ";".join(sorted({row["source_run"] for row in rows})),
        })
    if len(output) != 276:
        raise RuntimeError(f"expected 276 per-seed table rows, got {len(output)}")
    return output


def aggregate_rows(per_seed):
    grouped = defaultdict(list)
    for row in per_seed:
        key = tuple(row[field] for field in ("experiment", "reference", "branch", "horizon", "condition"))
        grouped[key].append(row)
    output = []
    for key, rows in sorted(grouped.items()):
        seeds = sorted(int(row["seed"]) for row in rows)
        if seeds != [0, 1, 2]:
            raise RuntimeError(f"missing seed in aggregate group {key}: {seeds}")
        result = dict(zip(("experiment", "reference", "branch", "horizon", "condition"), key))
        result |= {"label": LABEL, "seeds": "0;1;2", "n_seeds": 3, "split": "dev", "manifest_sha256": MANIFEST_SHA256}
        for metric in METRICS:
            values = [float(row[f"scene_macro_{metric}"]) for row in rows]
            result |= {
                f"{metric}_mean_across_seeds": statistics.mean(values),
                f"{metric}_sd_across_seeds": statistics.stdev(values),
                f"{metric}_min_seed": min(values), f"{metric}_max_seed": max(values),
                f"{metric}_range_across_seeds": max(values) - min(values),
            }
        output.append(result)
    if len(output) != 92:
        raise RuntimeError(f"expected 92 aggregate table rows, got {len(output)}")
    return output


def comparison_rows(records, per_seed):
    output = []
    d3 = [row for row in records if row["experiment"] == "D3"]
    for branch in ("pooled", "spatial"):
        for horizon in (2, 4):
            group = [dict(row, pair_id=f"{row['seed']}:{row['sample_id']}") for row in d3
                     if row["branch"] == branch and int(row["horizon"]) == horizon]
            for comparison, (lhs, rhs) in COMPARISONS.items():
                result = {
                    "label": LABEL, "experiment": "D3", "branch": branch,
                    "horizon": horizon, "comparison": comparison,
                    "lhs_condition": lhs, "rhs_condition": rhs,
                    "difference_definition": "lhs_minus_rhs", "n_seeds": 3,
                    "scenes": 69, "split": "dev", "manifest_sha256": MANIFEST_SHA256,
                }
                for metric in METRICS:
                    ci = scene_cluster_paired_bootstrap(
                        group, lhs, rhs, metric=metric, pair_key="pair_id",
                        n_boot=10_000, seed=20270301,
                    )
                    seed_deltas = []
                    for seed in (0, 1, 2):
                        lhs_row = next(row for row in per_seed if row["experiment"] == "D3" and row["branch"] == branch
                                       and row["horizon"] == horizon and row["condition"] == lhs and row["seed"] == seed)
                        rhs_row = next(row for row in per_seed if row["experiment"] == "D3" and row["branch"] == branch
                                       and row["horizon"] == horizon and row["condition"] == rhs and row["seed"] == seed)
                        seed_deltas.append(float(lhs_row[f"scene_macro_{metric}"]) - float(rhs_row[f"scene_macro_{metric}"]))
                    if ci["n_scenes"] != 69:
                        raise RuntimeError(f"incomplete comparison pairing: {branch}/h{horizon}/{comparison}/{metric}")
                    result |= {
                        f"{metric}_delta": ci["delta"], f"{metric}_ci_low": ci["ci_low"],
                        f"{metric}_ci_high": ci["ci_high"],
                        f"{metric}_seed_delta_mean": statistics.mean(seed_deltas),
                        f"{metric}_seed_delta_sd": statistics.stdev(seed_deltas),
                        f"{metric}_seed_deltas": ";".join(f"{value:.12g}" for value in seed_deltas),
                    }
                output.append(result)
    if len(output) != 16:
        raise RuntimeError(f"expected 16 D3 comparison rows, got {len(output)}")
    return output


def write_csv(path: Path, rows: list[dict]):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, aggregates, comparisons):
    lines = [
        "# V-JEPA2 D1/D3 three-seed development report", "",
        "**development-only — no held-out test split; comparison signs are reported without an assumed direction.**", "",
        "## D3 A/B/C/D comparisons", "",
        "Differences are lhs minus rhs. MAE/AbsRel below zero and Delta1 above zero favour the lhs; confidence intervals are paired 69-scene cluster bootstraps after averaging the three seeds within each scene.", "",
        "| branch | h | comparison | MAE delta [95% CI] | AbsRel delta [95% CI] | Delta1 delta [95% CI] | MAE seed SD |", "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['branch']} | {row['horizon']} | {row['comparison']} | "
            f"{row['mae_delta']:.6f} [{row['mae_ci_low']:.6f}, {row['mae_ci_high']:.6f}] | "
            f"{row['absrel_delta']:.6f} [{row['absrel_ci_low']:.6f}, {row['absrel_ci_high']:.6f}] | "
            f"{row['delta1_delta']:.6f} [{row['delta1_ci_low']:.6f}, {row['delta1_ci_high']:.6f}] | "
            f"{row['mae_seed_delta_sd']:.6f} |"
        )
    lines += ["", "## Across-seed metric summary", "", "| experiment | reference | branch | h | condition | MAE mean±SD | AbsRel mean±SD | Delta1 mean±SD |", "|---|---|---|---:|---|---:|---:|---:|"]
    for row in aggregates:
        lines.append(
            f"| {row['experiment']} | {row['reference']} | {row['branch']} | {row['horizon']} | {row['condition']} | "
            f"{row['mae_mean_across_seeds']:.6f}±{row['mae_sd_across_seeds']:.6f} | "
            f"{row['absrel_mean_across_seeds']:.6f}±{row['absrel_sd_across_seeds']:.6f} | "
            f"{row['delta1_mean_across_seeds']:.6f}±{row['delta1_sd_across_seeds']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-run", action="append", required=True, metavar="SEED=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    seed_runs = parse_seed_runs(args.seed_run)
    records, sources = load(seed_runs)
    per_seed = seed_rows(records)
    aggregates = aggregate_rows(per_seed)
    comparisons = comparison_rows(records, per_seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "per_seed_metrics.csv", per_seed)
    write_csv(args.output_dir / "across_seed_metrics.csv", aggregates)
    write_csv(args.output_dir / "d3_paired_comparisons.csv", comparisons)
    summary = {
        "schema_version": 1, "status": "COMPLETE_DEVELOPMENT_ONLY",
        "label": LABEL, "manifest_sha256": MANIFEST_SHA256,
        "seed_runs": {str(seed): str(path) for seed, path in seed_runs.items()},
        "per_seed_rows": len(per_seed), "aggregate_rows": len(aggregates),
        "comparison_rows": len(comparisons), "comparisons": comparisons,
        "sources": sources,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_markdown(args.output_dir / "report.md", aggregates, comparisons)
    print(json.dumps({"status": summary["status"], "output": str(args.output_dir), "rows": len(aggregates)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
