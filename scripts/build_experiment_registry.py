#!/usr/bin/env python3
"""Reconstruct the authoritative experiment registry from current artifacts.

This script does not infer scientific conclusions.  It records lifecycle state
for known run/evaluation configurations and keeps historical labels only in
status_reason/notes.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
A_B_TABLE = ROOT / "logs/eval_results/results_table.csv"
C_RAW = ROOT / "logs/inscene15k_streaming/results/c_eval_with_extra_20260728/c_series_raw_metrics_with_extra.csv"
WAN_COG = ROOT / "logs/probing_jobs_20260725_ab/recovery_20260730/perf_analysis/evaluation_results_all_probes.tsv"

REGISTRY_FIELDS = [
    "experiment_id",
    "family",
    "probe",
    "model",
    "backend",
    "model_variant",
    "layer",
    "prefix",
    "horizon",
    "seed",
    "train_split",
    "eval_split",
    "dataset_manifest",
    "config",
    "checkpoint",
    "launcher_job_id",
    "raw_artifact_path",
    "canonical_metric_path",
    "metric_name",
    "metric_value",
    "status",
    "status_reason",
    "supersedes",
    "git_commit",
    "notes",
]

VALID_STATUSES = {
    "PLANNED",
    "RUNNING",
    "COMPLETE",
    "FAILED",
    "INCOMPLETE",
    "INVALID",
    "SUPERSEDED",
    "ARCHIVED",
}


def read_rows(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        for row in rows:
            status = row.get("status", "")
            if status not in VALID_STATUSES:
                raise ValueError(f"Invalid status for {row.get('experiment_id')}: {status}")
            writer.writerow({field: row.get(field, "") for field in REGISTRY_FIELDS})


def write_missing(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "experiment_id",
        "family",
        "probe",
        "model",
        "model_variant",
        "layer",
        "prefix",
        "status",
        "status_reason",
        "launcher_job_id",
        "raw_artifact_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if row.get("status") == "COMPLETE":
                continue
            writer.writerow({field: row.get(field, "") for field in fields})


def rel(path: Path | str) -> str:
    text = str(path)
    if not text:
        return ""
    p = Path(text)
    if p.is_absolute():
        try:
            return str(p.relative_to(ROOT))
        except ValueError:
            return text
    return text


def git_commit() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.stdout.strip() or "UNKNOWN"


def clean(value: str | None) -> str:
    value = "" if value is None else str(value)
    value = value.strip().lower()
    value = value.replace("@", "at").replace("/", "_").replace(" ", "_")
    value = re.sub(r"[^a-z0-9_.+-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "na"


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"n/a", "nan", "none", "?"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def is_true(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def infer_layer(task: str, fallback: str = "") -> str:
    match = re.search(r"layer(-?\d+)", task)
    return match.group(1) if match else fallback


def infer_config(probe: str, model: str) -> str:
    stem = probe.split("/")[0]
    path = ROOT / "configs/experiment/inscene15k_streaming" / f"{stem}_{model}_v1.yaml"
    return rel(path) if path.exists() else ""


def canonical_path(family: str) -> str:
    return {
        "A1": "results/canonical/a1_results.csv",
        "A2": "results/canonical/a2_results.csv",
        "B2": "results/canonical/b2_results.csv",
        "C1": "results/canonical/c1_results.csv",
        "C2": "results/canonical/c2_results.csv",
        "C3": "results/canonical/c3_results.csv",
    }.get(family, "")


def registry_id(family: str, probe: str, model: str, variant: str, layer: str, prefix: str, horizon: str, seed: str) -> str:
    return "_".join(
        [
            clean(family),
            clean(probe),
            clean(model),
            clean(variant),
            f"l{clean(layer)}",
            f"p{clean(prefix)}",
            f"h{clean(horizon)}",
            f"s{clean(seed)}",
        ]
    )


def add_default_ab(rows: list[dict[str, str]], commit: str) -> None:
    for row in read_rows(A_B_TABLE):
        series = row.get("series", "").strip().lower()
        if series not in {"a1", "a2", "b2"}:
            continue
        model = row.get("model", "")
        probe = row.get("probe_type", "")
        prefix = row.get("prefix", "")
        layer = infer_layer(row.get("task", ""))
        metric_name = row.get("metric_name", "")
        metric_value = row.get("metric_value", "")
        value = parse_float(metric_value)
        family = series.upper()
        status = "COMPLETE"
        reason = "default eval_results row is numeric"
        notes = "source=logs/eval_results/results_table.csv"
        canonical = canonical_path(family)
        if family == "B2":
            status = "INCOMPLETE"
            reason = "default B2 table does not provide the canonical angle metric in a machine-readable source"
            canonical = ""
        elif row.get("status") != "done" or value is None:
            status = "INCOMPLETE"
            reason = f"source status={row.get('status')} metric_value={metric_value}"
        experiment_id = registry_id(family, probe, model, "default_streaming", layer, prefix, "all", "42")
        rows.append(
            {
                "experiment_id": experiment_id,
                "family": family,
                "probe": probe,
                "model": model,
                "backend": model,
                "model_variant": "default_streaming",
                "layer": layer,
                "prefix": prefix,
                "horizon": "all",
                "seed": "42",
                "train_split": "train",
                "eval_split": "val",
                "dataset_manifest": "configs/splits/inscene15k_v1.json",
                "config": infer_config(probe, model),
                "checkpoint": "",
                "launcher_job_id": row.get("task", ""),
                "raw_artifact_path": rel(A_B_TABLE),
                "canonical_metric_path": canonical,
                "metric_name": metric_name,
                "metric_value": metric_value if value is not None else "",
                "status": status,
                "status_reason": reason,
                "supersedes": "",
                "git_commit": commit,
                "notes": notes,
            }
        )


def c_family(task: str, probe: str) -> str:
    if task.startswith("C1") or probe == "action_dynamics":
        return "C1"
    if task.startswith("C2") or probe == "path_integration":
        return "C2"
    if task.startswith("C3") or probe == "counterfactual":
        return "C3"
    return "C"


def c_metric(row: dict[str, str], family: str) -> tuple[str, str]:
    if family == "C1":
        return "global_R@1", row.get("global_R@1", "")
    if family == "C2":
        return "global_R@1", row.get("global_R@1", "")
    if family == "C3":
        return "intervention_validity", row.get("intervention_validity", "")
    return row.get("primary_metric", ""), row.get("primary_value", "")


def c_manifest(prefix: str) -> str:
    path = ROOT / "manifests/c_retrain_20260722" / f"inscene_c_h124_prefix{prefix}.jsonl"
    if path.exists():
        return rel(path)
    path = ROOT / "manifests/c_retrain_20260722" / f"inscene_c_prefix{prefix}.jsonl"
    return rel(path) if path.exists() else ""


def add_c_series(rows: list[dict[str, str]], commit: str) -> None:
    for row in read_rows(C_RAW):
        probe = row.get("probe_type", "")
        family = c_family(row.get("task", ""), probe)
        model = row.get("vfm_name", "")
        prefix = row.get("prefix", "")
        layer = row.get("layer", "")
        metric_name, metric_value = c_metric(row, family)
        value = parse_float(metric_value)
        status = "COMPLETE" if value is not None else "INCOMPLETE"
        reason = "consolidated C metrics with extra VideoMAE/Qwen rows" if status == "COMPLETE" else "missing primary metric"
        experiment_id = registry_id(family, probe, model, "c_retrain_with_extra", layer, prefix, "all", "42")
        rows.append(
            {
                "experiment_id": experiment_id,
                "family": family,
                "probe": probe,
                "model": model,
                "backend": model,
                "model_variant": "c_retrain_with_extra",
                "layer": layer,
                "prefix": prefix,
                "horizon": "all",
                "seed": "42",
                "train_split": "train",
                "eval_split": "val",
                "dataset_manifest": c_manifest(prefix),
                "config": infer_config(probe, model),
                "checkpoint": rel(row.get("checkpoint_path") or row.get("ckpt_path") or ""),
                "launcher_job_id": row.get("job_name", row.get("key", "")),
                "raw_artifact_path": rel(C_RAW),
                "canonical_metric_path": canonical_path(family),
                "metric_name": metric_name,
                "metric_value": metric_value if value is not None else "",
                "status": status,
                "status_reason": reason,
                "supersedes": f"{clean(family)}_{clean(probe)}_{clean(model)}_legacy_results_table_l{clean(layer)}_p{clean(prefix)}",
                "git_commit": commit,
                "notes": f"eval_summary_path={rel(row.get('eval_summary_path', ''))}",
            }
        )


def wan_cog_family(probe: str) -> str:
    return {
        "streaming_depth": "A1",
        "view_consistency": "A2",
        "ego_belief": "B1",
        "ego_belief_v2": "B2",
    }.get(probe, "UNKNOWN")


def wan_cog_metric(row: dict[str, str]) -> tuple[str, str]:
    probe = row.get("probe", "")
    if probe == "ego_belief_v2":
        return "mean_ang_err_deg", row.get("mean_ang_err_deg", "")
    if probe == "ego_belief":
        return "mean_az_err_deg", row.get("mean_az_err_deg", "")
    if probe == "streaming_depth":
        return "mean_loss_depth", row.get("mean_loss_depth", "")
    if probe == "view_consistency":
        return "balanced_acc", row.get("balanced_acc", "")
    return "", ""


def add_wan_cog(rows: list[dict[str, str]], commit: str) -> None:
    for row in read_rows(WAN_COG, delimiter="\t"):
        probe = row.get("probe", "")
        family = wan_cog_family(probe)
        if family == "UNKNOWN":
            continue
        metric_name, metric_value = wan_cog_metric(row)
        value = parse_float(metric_value)
        provisional = is_true(row.get("provisional"))
        status = "COMPLETE" if (not provisional and value is not None) else "INCOMPLETE"
        reason = "formal/repaired checkpoint row accepted as canonical"
        if provisional:
            reason = f"historical provisional row excluded from canonical: kind={row.get('kind')} class={row.get('class')}"
        elif value is None:
            reason = f"metric missing for class={row.get('class')}"
        model = row.get("vfm", "")
        prefix = row.get("prefix", "")
        layer = row.get("layer", "")
        canonical = canonical_path(family) if status == "COMPLETE" and family in {"A1", "A2", "B2"} else ""
        experiment_id = registry_id(family, probe, model, "wan_cog_layer_sweep", layer, prefix, "all", "42")
        rows.append(
            {
                "experiment_id": experiment_id,
                "family": family,
                "probe": probe,
                "model": model,
                "backend": model,
                "model_variant": "wan_cog_layer_sweep",
                "layer": layer,
                "prefix": prefix,
                "horizon": "all",
                "seed": "42",
                "train_split": "train",
                "eval_split": "val",
                "dataset_manifest": "configs/splits/inscene15k_v1.json",
                "config": infer_config(probe, model),
                "checkpoint": "",
                "launcher_job_id": row.get("job_id", ""),
                "raw_artifact_path": rel(WAN_COG),
                "canonical_metric_path": canonical,
                "metric_name": metric_name,
                "metric_value": metric_value if value is not None else "",
                "status": status,
                "status_reason": reason,
                "supersedes": "",
                "git_commit": commit,
                "notes": f"historical_kind={row.get('kind')} historical_class={row.get('class')}",
            }
        )


def dedupe(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: dict[str, int] = {}
    out = []
    for row in rows:
        eid = row["experiment_id"]
        if eid in seen:
            seen[eid] += 1
            row = dict(row)
            row["experiment_id"] = f"{eid}_dup{seen[eid]}"
        else:
            seen[eid] = 0
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="experiments/registry.csv")
    args = parser.parse_args()
    rows: list[dict[str, str]] = []
    commit = git_commit()
    add_default_ab(rows, commit)
    add_c_series(rows, commit)
    add_wan_cog(rows, commit)
    rows = dedupe(rows)
    out_path = ROOT / args.output
    write_rows(out_path, rows)
    write_missing(ROOT / "experiments/missing_experiments.csv", rows)
    print(f"Wrote {len(rows)} registry rows to {out_path}")


if __name__ == "__main__":
    main()
