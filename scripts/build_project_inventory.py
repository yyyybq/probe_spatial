#!/usr/bin/env python3
"""Create archived project inventory CSV and Markdown report."""

from __future__ import annotations

import argparse
import csv
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIELDS = [
    "path",
    "category",
    "purpose",
    "producer",
    "consumer",
    "reproducible",
    "currently_referenced",
    "status",
    "recommended_action",
]


def run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    return proc.stdout


def git_status_map() -> dict[str, str]:
    out = {}
    for line in run(["git", "status", "--short"]).splitlines():
        if not line.strip():
            continue
        code = line[:2]
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        out[path] = code
    return out


def collect_paths() -> set[str]:
    paths: set[str] = set()
    for line in run(["git", "ls-files"]).splitlines():
        if line:
            paths.add(line)
    for line in run(["git", "status", "--short"]).splitlines():
        if line.strip():
            path = line[3:].strip()
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            if path.endswith("/"):
                for child in (ROOT / path).rglob("*"):
                    if child.is_file():
                        paths.add(str(child.relative_to(ROOT)))
            else:
                paths.add(path)
    known = [
        "logs/eval_results/results_table.csv",
        "logs/eval_results/results_summary.md",
        "logs/inscene15k_streaming/results/c_eval_with_extra_20260728/c_series_raw_metrics_with_extra.csv",
        "logs/inscene15k_streaming/results/c_eval_with_extra_20260728/c_series_final_metrics_with_extra.md",
        "logs/inscene15k_streaming/results/c_eval_with_extra_20260728/c_series_best_by_probe_vfm_with_extra.csv",
        "logs/probing_jobs_20260725_ab/recovery_20260730/perf_analysis/evaluation_results_all_probes.tsv",
        "logs/probing_jobs_20260725_ab/recovery_20260730/perf_analysis/evaluation_results_all_probes.md",
        "logs/probing_jobs_20260725_ab/recovery_20260730/perf_analysis/canonical_best_checkpoint_registry.tsv",
        "logs/probing_jobs_20260725_ab/recovery_20260730/perf_analysis/checkpoint_best_mapping.tsv",
        "logs/current_audit_20260817/c_shortcut_audit.csv",
        "logs/current_audit_20260817/b2_canonical_angle_formal_only.csv",
        "logs/current_audit_20260817/wan_cog_remaining_nonformal_jobs.tsv",
        "logs/current_audit_20260817/new_backend_artifact_audit.csv",
    ]
    for path in known:
        if (ROOT / path).exists():
            paths.add(path)
    for base in ["manifests", "experiments", "metadata", "results/canonical", "protocols", "archive"]:
        directory = ROOT / base
        if directory.exists():
            for child in directory.rglob("*"):
                if child.is_file():
                    paths.add(str(child.relative_to(ROOT)))
    return paths


def category(path: str) -> str:
    if path.startswith("vidfm3d/") or path.startswith("features/") or path.endswith(".py"):
        return "Source code"
    if path.startswith("scripts/"):
        return "Source code"
    if path.startswith("configs/"):
        return "Configs"
    if path.startswith("manifests/") or path.startswith("configs/splits/"):
        return "Data manifests"
    if path.startswith("logs/"):
        return "Experiment artifacts"
    if path.startswith("docs/") or path == "PROJECT_STATUS.md" or path.startswith("protocols/"):
        return "Derived reports"
    if path.startswith("results/canonical/"):
        return "Canonical results"
    if path.startswith("experiments/") or path.startswith("metadata/"):
        return "Registry metadata"
    if path.startswith("archive/"):
        return "Archive"
    return "Other"


def purpose(path: str, cat: str) -> str:
    name = Path(path).name
    if path == "experiments/registry.csv":
        return "Authoritative experiment lifecycle registry"
    if path.startswith("results/canonical/"):
        return "Canonical machine-readable metric table"
    if path.startswith("protocols/"):
        return "Scientific protocol definition"
    if "eval_diag" in path:
        return "Diagnostic evaluation and metric aggregation"
    if path.startswith("scripts/build_") or path.startswith("scripts/check_"):
        return "Consolidation or validation utility"
    if cat == "Configs":
        return "Hydra experiment/configuration file"
    if cat == "Data manifests":
        return "Dataset split or generated sample manifest"
    if cat == "Experiment artifacts":
        if name.endswith((".csv", ".tsv", ".json")):
            return "Raw or derived experiment metrics/artifact"
        return "Launcher, monitor, or report artifact"
    if cat == "Source code":
        return "Implementation code"
    if cat == "Derived reports":
        return "Human-readable project report or audit"
    return "Project file"


def producer(path: str, cat: str) -> str:
    if path.startswith("results/canonical/"):
        return "scripts/build_canonical_results.py"
    if path == "experiments/registry.csv":
        return "scripts/build_experiment_registry.py"
    if path == "archive/consolidation_history/20260817/project_inventory.csv":
        return "scripts/build_project_inventory.py"
    if path.startswith("logs/"):
        return "training/evaluation/aggregation jobs"
    if path.startswith("manifests/"):
        return "manifest builder scripts"
    return "manual/source control"


def consumer(path: str, cat: str) -> str:
    if path == "experiments/registry.csv":
        return "scripts/build_canonical_results.py; scripts/check_project_consistency.py; PROJECT_STATUS.md"
    if path.startswith("results/canonical/"):
        return "PROJECT_STATUS.md; reports; papers/slides"
    if path.startswith("logs/"):
        return "registry/canonical builders when explicitly referenced"
    if path.startswith("configs/"):
        return "training/evaluation launchers"
    if path.startswith("manifests/"):
        return "dataset loaders and launchers"
    return "developers"


def recommended(path: str, cat: str, status: str) -> str:
    if "__pycache__" in path or path.endswith((".pyc", ".tmp")):
        return "DELETE_GENERATED"
    if path.startswith("results/canonical/") or path == "experiments/registry.csv" or path.startswith("metadata/"):
        return "KEEP_ACTIVE"
    if path.startswith("scripts/") or path.startswith("configs/") or path.startswith("vidfm3d/") or path.startswith("features/"):
        return "KEEP_ACTIVE" if status.strip() != "??" else "INVESTIGATE"
    if path.startswith("protocols/") or path == "PROJECT_STATUS.md":
        return "KEEP_ACTIVE"
    if path.startswith("manifests/"):
        return "KEEP_ACTIVE"
    if path.startswith("logs/current_audit_20260817/"):
        return "ARCHIVE"
    if path.startswith("logs/") and any(token in path for token in ["best", "provisional", "progress", "handoff"]):
        return "ARCHIVE"
    if path.startswith("logs/"):
        return "ARCHIVE"
    if path.startswith("docs/"):
        return "KEEP_ACTIVE" if "20260817" in path else "ARCHIVE"
    if path.startswith("archive/"):
        return "ARCHIVE"
    return "INVESTIGATE"


def referenced(path: str) -> str:
    if path.startswith(("logs/", "docs/", "archive/")):
        return "no"
    if path in {"experiments/registry.csv", "archive/consolidation_history/20260817/project_inventory.csv"}:
        return "yes"
    if path.startswith(("scripts/", "configs/", "vidfm3d/", "features/", "manifests/", "results/canonical/", "protocols/")):
        return "yes"
    return "unknown"


def reproducible(path: str, cat: str) -> str:
    if cat in {"Source code", "Configs", "Data manifests", "Registry metadata", "Canonical results", "Derived reports"}:
        return "yes"
    if cat == "Experiment artifacts":
        return "partial"
    return "unknown"


def build_rows() -> list[dict[str, str]]:
    status_map = git_status_map()
    rows = []
    for path in sorted(collect_paths()):
        cat = category(path)
        status = status_map.get(path, "tracked")
        rows.append(
            {
                "path": path,
                "category": cat,
                "purpose": purpose(path, cat),
                "producer": producer(path, cat),
                "consumer": consumer(path, cat),
                "reproducible": reproducible(path, cat),
                "currently_referenced": referenced(path),
                "status": status,
                "recommended_action": recommended(path, cat, status),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_cat = Counter(row["category"] for row in rows)
    by_action = Counter(row["recommended_action"] for row in rows)
    by_status = Counter(row["status"] for row in rows)
    important = [row for row in rows if row["recommended_action"] in {"KEEP_ACTIVE", "MOVE_TO_CANONICAL", "INVESTIGATE"}]
    lines = [
        "# Project Inventory 2026-08-17",
        "",
        "This inventory is a classification snapshot. No files were deleted by the inventory builder.",
        "",
        "## Counts By Category",
        "",
    ]
    for key, value in sorted(by_cat.items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Counts By Recommended Action", ""])
    for key, value in sorted(by_action.items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Counts By Git Status", ""])
    for key, value in sorted(by_status.items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Important Files", ""])
    lines.append("| path | category | status | action | purpose |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in important[:240]:
        lines.append(
            f"| `{row['path']}` | {row['category']} | `{row['status']}` | {row['recommended_action']} | {row['purpose']} |"
        )
    if len(important) > 240:
        lines.append(f"| ... | ... | ... | ... | {len(important) - 240} additional important rows in archive/consolidation_history/20260817/project_inventory.csv |")
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="archive/consolidation_history/20260817/project_inventory.csv")
    parser.add_argument("--md", default="archive/consolidation_history/20260817/project_inventory_20260817.md")
    args = parser.parse_args()
    rows = build_rows()
    write_csv(ROOT / args.csv, rows)
    write_md(ROOT / args.md, rows)
    print(f"Wrote {len(rows)} inventory rows")


if __name__ == "__main__":
    main()
