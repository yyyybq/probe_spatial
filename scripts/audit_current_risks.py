#!/usr/bin/env python3
"""Build reproducibility/risk audit tables from existing probing_spatial artifacts.

The script is intentionally read-only with respect to experiment artifacts.  It
only writes derived CSV/Markdown reports under the requested output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
C_RAW = ROOT / "logs/inscene15k_streaming/results/c_eval_with_extra_20260728/c_series_raw_metrics_with_extra.csv"
WAN_COG_TSV = ROOT / "logs/probing_jobs_20260725_ab/recovery_20260730/perf_analysis/evaluation_results_all_probes.tsv"

BACKEND_CAMPAIGNS = [
    "probing_jobs_minimax_h3_abc_20260810_0109",
    "probing_jobs_minimax_h3_abc_20260810_0125",
    "probing_jobs_minimax_h3_abc_retry_20260812_1848",
    "probing_jobs_minimax_h3_a1_pose_mask_20260812_1905",
    "probing_jobs_probe_fill_lingbot_20260814_1928",
    "probing_jobs_probe_fill_fill_20260814_1822",
    "probing_jobs_probe_fill_fill_20260814_1928",
]


def campaign_tokens(name: str) -> list[str]:
    prefix = "probing_jobs_"
    token = name[len(prefix) :] if name.startswith(prefix) else name
    tokens = [token]
    if token.endswith("_0109"):
        tokens.append(token[:-5] + "_0125")
    return list(dict.fromkeys(tokens))


def read_csv(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def as_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def fmt(value: object, ndigits: int = 4) -> str:
    number = as_float(value)
    if number is None:
        return ""
    return f"{number:.{ndigits}f}"


def is_true(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def tail_line(path: Path) -> str:
    if not path.exists():
        return ""
    last = ""
    with path.open(errors="replace") as f:
        for line in f:
            if line.strip():
                last = line.strip()
    return last


def find_count(path: Path, name: str, timeout: int = 8) -> tuple[int | None, str]:
    if not path.exists():
        return 0, ""
    command = ["rg", "--files", str(path)]
    try:
        proc = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        command = ["find", str(path), "-name", name]
        try:
            proc = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return None, f"{' '.join(command[:2])} timed out after {timeout}s"
    except subprocess.TimeoutExpired:
        return None, f"{' '.join(command[:2])} timed out after {timeout}s"
    count = sum(1 for line in proc.stdout.splitlines() if Path(line.strip()).name == name)
    return count, proc.stderr.strip()


def markdown_table(rows: list[dict[str, object]], columns: list[str], limit: int | None = None) -> str:
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        return "_No rows._\n"
    out = []
    out.append("| " + " | ".join(columns) + " |")
    out.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        values = [str(row.get(col, "")) for col in columns]
        out.append("| " + " | ".join(values) + " |")
    return "\n".join(out) + "\n"


def audit_c_shortcuts(out_dir: Path) -> None:
    rows = read_csv(C_RAW)
    out_rows: list[dict[str, object]] = []
    eps = 1e-12
    for row in rows:
        global_r1 = as_float(row.get("global_R@1"))
        global_r5 = as_float(row.get("global_R@5"))
        baselines_r1 = {
            "no_action": as_float(row.get("no_action_global_R@1")),
            "shuffled_action": as_float(row.get("shuffled_action_global_R@1")),
            "last_observation": as_float(row.get("last_observation_global_R@1")),
        }
        baselines_r5 = {
            "no_action": as_float(row.get("no_action_global_R@5")),
            "shuffled_action": as_float(row.get("shuffled_action_global_R@5")),
            "last_observation": as_float(row.get("last_observation_global_R@5")),
        }
        available = {k: v for k, v in baselines_r1.items() if v is not None}
        best_name = ""
        best_value = None
        if available:
            best_name, best_value = max(available.items(), key=lambda item: item[1])
        delta = None
        if global_r1 is not None and best_value is not None:
            delta = global_r1 - best_value
        no_action = baselines_r1["no_action"]
        shuffled = baselines_r1["shuffled_action"]
        last_obs = baselines_r1["last_observation"]
        action_beats_all = bool(available) and global_r1 is not None and all(global_r1 > v + eps for v in available.values())
        shortcut_risk = bool(available) and global_r1 is not None and any(v >= global_r1 - eps for v in available.values())
        copy_beats_or_ties = last_obs is not None and global_r1 is not None and last_obs >= global_r1 - eps
        static_beats_or_ties = global_r1 is not None and any(
            v is not None and v >= global_r1 - eps for v in (no_action, shuffled)
        )
        out_rows.append(
            {
                "task": row.get("task", ""),
                "probe_type": row.get("probe_type", ""),
                "vfm_name": row.get("vfm_name", ""),
                "prefix": row.get("prefix", ""),
                "layer": row.get("layer", ""),
                "n": row.get("n", ""),
                "global_R@1": global_r1,
                "global_R@5": global_r5,
                "no_action_R@1": no_action,
                "shuffled_action_R@1": shuffled,
                "last_observation_R@1": last_obs,
                "best_baseline_R@1": best_value,
                "best_baseline_name": best_name,
                "delta_vs_best_baseline_R@1": delta,
                "action_beats_all_baselines": action_beats_all,
                "shortcut_risk_R@1": shortcut_risk,
                "copy_beats_or_ties_R@1": copy_beats_or_ties,
                "static_beats_or_ties_R@1": static_beats_or_ties,
                "action_gain_cos": row.get("action_gain_cos", ""),
                "intervention_validity": row.get("intervention_validity", ""),
                "intervention_margin": row.get("intervention_margin", ""),
                "counterfactual_consistency": row.get("counterfactual_consistency", ""),
                "no_action_R@5": baselines_r5["no_action"],
                "shuffled_action_R@5": baselines_r5["shuffled_action"],
                "last_observation_R@5": baselines_r5["last_observation"],
                "eval_summary_path": row.get("eval_summary_path", ""),
            }
        )

    fields = [
        "task",
        "probe_type",
        "vfm_name",
        "prefix",
        "layer",
        "n",
        "global_R@1",
        "global_R@5",
        "no_action_R@1",
        "shuffled_action_R@1",
        "last_observation_R@1",
        "best_baseline_R@1",
        "best_baseline_name",
        "delta_vs_best_baseline_R@1",
        "action_beats_all_baselines",
        "shortcut_risk_R@1",
        "copy_beats_or_ties_R@1",
        "static_beats_or_ties_R@1",
        "action_gain_cos",
        "intervention_validity",
        "intervention_margin",
        "counterfactual_consistency",
        "no_action_R@5",
        "shuffled_action_R@5",
        "last_observation_R@5",
        "eval_summary_path",
    ]
    write_csv(out_dir / "c_shortcut_audit.csv", out_rows, fields)

    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in out_rows:
        grouped[(str(row["task"]), str(row["probe_type"]), str(row["vfm_name"]))].append(row)
    summary_rows = []
    for key, group in sorted(grouped.items()):
        deltas = [as_float(r["delta_vs_best_baseline_R@1"]) for r in group]
        deltas = [v for v in deltas if v is not None]
        summary_rows.append(
            {
                "task": key[0],
                "probe_type": key[1],
                "vfm_name": key[2],
                "rows": len(group),
                "action_beats_all": sum(1 for r in group if r["action_beats_all_baselines"]),
                "shortcut_risk": sum(1 for r in group if r["shortcut_risk_R@1"]),
                "copy_beats_or_ties": sum(1 for r in group if r["copy_beats_or_ties_R@1"]),
                "static_beats_or_ties": sum(1 for r in group if r["static_beats_or_ties_R@1"]),
                "mean_delta_R@1": sum(deltas) / len(deltas) if deltas else "",
            }
        )
    summary_fields = [
        "task",
        "probe_type",
        "vfm_name",
        "rows",
        "action_beats_all",
        "shortcut_risk",
        "copy_beats_or_ties",
        "static_beats_or_ties",
        "mean_delta_R@1",
    ]
    write_csv(out_dir / "c_shortcut_summary_by_model.csv", summary_rows, summary_fields)

    best_support = sorted(
        [r for r in out_rows if as_float(r["delta_vs_best_baseline_R@1"]) is not None],
        key=lambda r: as_float(r["delta_vs_best_baseline_R@1"]) or -999,
        reverse=True,
    )
    worst_support = list(reversed(best_support))
    task_counts = Counter(str(r["task"]) for r in out_rows)
    risk_rows = [r for r in out_rows if r["shortcut_risk_R@1"]]
    md = [
        "# C-series Shortcut Audit",
        "",
        f"Source: `{C_RAW.relative_to(ROOT)}`",
        "",
        "Definition: `action_beats_all_baselines` means probe `global_R@1` is strictly above no-action, shuffled-action, and last-observation/copy baselines in the same row. `shortcut_risk_R@1` means at least one baseline ties or beats the probe.",
        "",
        f"- Rows audited: {len(out_rows)}",
        f"- Task rows: {dict(sorted(task_counts.items()))}",
        f"- Rows where action beats all available R@1 baselines: {sum(1 for r in out_rows if r['action_beats_all_baselines'])}",
        f"- Rows with R@1 shortcut risk: {len(risk_rows)}",
        f"- Rows where copy/last-observation ties or beats probe: {sum(1 for r in out_rows if r['copy_beats_or_ties_R@1'])}",
        f"- Rows where no-action or shuffled-action ties or beats probe: {sum(1 for r in out_rows if r['static_beats_or_ties_R@1'])}",
        "",
        "## Summary By Model",
        "",
        markdown_table(
            [
                {
                    **r,
                    "mean_delta_R@1": fmt(r["mean_delta_R@1"]),
                }
                for r in summary_rows
            ],
            summary_fields,
        ),
        "## Strongest Evidence Against Shortcut",
        "",
        markdown_table(
            [
                {
                    "task": r["task"],
                    "probe_type": r["probe_type"],
                    "vfm_name": r["vfm_name"],
                    "prefix": r["prefix"],
                    "layer": r["layer"],
                    "global_R@1": fmt(r["global_R@1"]),
                    "best_baseline_R@1": fmt(r["best_baseline_R@1"]),
                    "delta_vs_best_baseline_R@1": fmt(r["delta_vs_best_baseline_R@1"]),
                    "best_baseline_name": r["best_baseline_name"],
                }
                for r in best_support
            ],
            [
                "task",
                "probe_type",
                "vfm_name",
                "prefix",
                "layer",
                "global_R@1",
                "best_baseline_R@1",
                "delta_vs_best_baseline_R@1",
                "best_baseline_name",
            ],
            limit=12,
        ),
        "## Strongest Shortcut-risk Rows",
        "",
        markdown_table(
            [
                {
                    "task": r["task"],
                    "probe_type": r["probe_type"],
                    "vfm_name": r["vfm_name"],
                    "prefix": r["prefix"],
                    "layer": r["layer"],
                    "global_R@1": fmt(r["global_R@1"]),
                    "best_baseline_R@1": fmt(r["best_baseline_R@1"]),
                    "delta_vs_best_baseline_R@1": fmt(r["delta_vs_best_baseline_R@1"]),
                    "best_baseline_name": r["best_baseline_name"],
                }
                for r in worst_support
            ],
            [
                "task",
                "probe_type",
                "vfm_name",
                "prefix",
                "layer",
                "global_R@1",
                "best_baseline_R@1",
                "delta_vs_best_baseline_R@1",
                "best_baseline_name",
            ],
            limit=12,
        ),
        "",
    ]
    (out_dir / "c_shortcut_audit.md").write_text("\n".join(md))


def best_rows(rows: list[dict[str, str]], metric: str, higher_is_better: bool) -> list[dict[str, str]]:
    buckets: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[(row.get("probe", ""), row.get("vfm", ""), row.get("prefix", ""))].append(row)
    best = []
    for group in buckets.values():
        candidates = [r for r in group if as_float(r.get(metric)) is not None]
        if not candidates:
            continue
        best.append(
            sorted(candidates, key=lambda r: as_float(r.get(metric)) or 0, reverse=higher_is_better)[0]
        )
    return sorted(best, key=lambda r: (r.get("probe", ""), r.get("vfm", ""), int(r.get("prefix") or 0)))


def audit_wan_cog(out_dir: Path) -> None:
    rows = read_csv(WAN_COG_TSV, delimiter="\t")
    canonical_fields = [
        "job_id",
        "probe",
        "vfm",
        "prefix",
        "layer",
        "kind",
        "class",
        "provisional",
        "mean_ang_err_deg",
        "mean_az_err_deg",
        "mean_el_err_deg",
        "mean_log_dist_err",
        "top1",
        "top3",
        "balanced_acc",
        "roc_auc",
        "mean_overlap_mae",
        "hard_acc",
        "Auc_5",
        "Auc_10",
        "Rac_15",
        "Tac_15",
        "n",
    ]
    b2_rows = [r for r in rows if r.get("probe") == "ego_belief_v2"]
    write_csv(out_dir / "b2_canonical_angle.csv", b2_rows, canonical_fields)
    formal_b2 = [r for r in b2_rows if not is_true(r.get("provisional"))]
    write_csv(out_dir / "b2_canonical_angle_formal_only.csv", formal_b2, canonical_fields)
    b2_best_formal = best_rows(formal_b2, "mean_ang_err_deg", higher_is_better=False)
    write_csv(out_dir / "b2_best_angle_formal_by_vfm_prefix.csv", b2_best_formal, canonical_fields)

    nonformal = [r for r in rows if is_true(r.get("provisional")) or "MISSING" in str(r.get("class", ""))]
    write_csv(out_dir / "wan_cog_remaining_nonformal_jobs.tsv", nonformal, canonical_fields)
    no_ckpt = [r for r in rows if "NO_CKPT" in str(r.get("class", "")) or str(r.get("kind")) == "no_ckpt"]
    write_csv(out_dir / "wan_cog_no_checkpoint_jobs.tsv", no_ckpt, canonical_fields)

    group_fields = ["probe", "vfm", "prefix", "total", "formal", "provisional", "missing_class", "no_ckpt"]
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("probe", ""), row.get("vfm", ""), row.get("prefix", ""))].append(row)
    group_rows = []
    for key, group in sorted(grouped.items(), key=lambda item: item[0]):
        group_rows.append(
            {
                "probe": key[0],
                "vfm": key[1],
                "prefix": key[2],
                "total": len(group),
                "formal": sum(1 for r in group if not is_true(r.get("provisional"))),
                "provisional": sum(1 for r in group if is_true(r.get("provisional"))),
                "missing_class": sum(1 for r in group if "MISSING" in str(r.get("class", ""))),
                "no_ckpt": sum(1 for r in group if "NO_CKPT" in str(r.get("class", "")) or str(r.get("kind")) == "no_ckpt"),
            }
        )
    write_csv(out_dir / "wan_cog_layer_formalization_summary.csv", group_rows, group_fields)

    all_best_formal: list[dict[str, str]] = []
    metric_by_probe = {
        "ego_belief": ("mean_az_err_deg", False),
        "ego_belief_v2": ("mean_ang_err_deg", False),
        "streaming_depth": ("mean_loss_depth", False),
        "view_consistency": ("balanced_acc", True),
    }
    for probe, (metric, higher) in metric_by_probe.items():
        probe_formal = [r for r in rows if r.get("probe") == probe and not is_true(r.get("provisional"))]
        all_best_formal.extend(best_rows(probe_formal, metric, higher))
    write_csv(out_dir / "wan_cog_formal_best_by_probe_vfm_prefix.csv", all_best_formal, canonical_fields)

    counts = Counter(
        "formal" if not is_true(r.get("provisional")) else "provisional"
        for r in rows
    )
    classes = Counter(str(r.get("class", "")) for r in rows)
    md = [
        "# Wan/Cog Layer Sweep Formalization Audit",
        "",
        f"Source: `{WAN_COG_TSV.relative_to(ROOT)}`",
        "",
        f"- Total rows: {len(rows)}",
        f"- Formal rows: {counts.get('formal', 0)}",
        f"- Provisional rows: {counts.get('provisional', 0)}",
        f"- Class counts: {dict(sorted(classes.items()))}",
        f"- B2 rows: {len(b2_rows)}; B2 formal-only rows: {len(formal_b2)}",
        "",
        "This audit separates formal rows from latest/provisional rows. Formal reports should use `*_formal_only.csv` or `wan_cog_formal_best_by_probe_vfm_prefix.csv`; provisional rows remain useful for triage only.",
        "",
        "## B2 Formal Best By VFM/Prefix",
        "",
        markdown_table(
            [
                {
                    "job_id": r["job_id"],
                    "vfm": r["vfm"],
                    "prefix": r["prefix"],
                    "layer": r["layer"],
                    "class": r["class"],
                    "mean_ang_err_deg": fmt(r["mean_ang_err_deg"]),
                    "top1": fmt(r["top1"]),
                    "top3": fmt(r["top3"]),
                    "n": r["n"],
                }
                for r in b2_best_formal
            ],
            ["job_id", "vfm", "prefix", "layer", "class", "mean_ang_err_deg", "top1", "top3", "n"],
            limit=24,
        ),
        "## Formalization Summary",
        "",
        markdown_table(group_rows, group_fields, limit=40),
        "",
    ]
    (out_dir / "wan_cog_layer_formalization.md").write_text("\n".join(md))

    b2_md = [
        "# B2 Canonical Angle Metrics",
        "",
        f"Source: `{WAN_COG_TSV.relative_to(ROOT)}` filtered to `probe == ego_belief_v2`.",
        "",
        f"- Canonical B2 rows: {len(b2_rows)}",
        f"- Formal-only rows: {len(formal_b2)}",
        f"- Provisional rows: {len(b2_rows) - len(formal_b2)}",
        "",
        "Use `b2_canonical_angle_formal_only.csv` for formal claims and `b2_canonical_angle.csv` for complete triage.",
        "",
        "## Formal Best By VFM/Prefix",
        "",
        markdown_table(
            [
                {
                    "job_id": r["job_id"],
                    "vfm": r["vfm"],
                    "prefix": r["prefix"],
                    "layer": r["layer"],
                    "class": r["class"],
                    "mean_ang_err_deg": fmt(r["mean_ang_err_deg"]),
                    "mean_az_err_deg": fmt(r["mean_az_err_deg"]),
                    "mean_el_err_deg": fmt(r["mean_el_err_deg"]),
                    "top1": fmt(r["top1"]),
                    "top3": fmt(r["top3"]),
                    "n": r["n"],
                }
                for r in b2_best_formal
            ],
            [
                "job_id",
                "vfm",
                "prefix",
                "layer",
                "class",
                "mean_ang_err_deg",
                "mean_az_err_deg",
                "mean_el_err_deg",
                "top1",
                "top3",
                "n",
            ],
        ),
        "",
    ]
    (out_dir / "b2_canonical_angle.md").write_text("\n".join(b2_md))


def count_job_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(errors="replace") as f:
        lines = [line for line in f if line.strip()]
    if not lines:
        return 0
    return max(0, len(lines) - 1)


def audit_new_backends(out_dir: Path) -> None:
    rows = []
    for name in BACKEND_CAMPAIGNS:
        campaign = ROOT / "logs" / name
        if not campaign.exists():
            continue
        jobs_dir = campaign / "jobs"
        job_lists = sorted(jobs_dir.glob("*.tsv")) if jobs_dir.exists() else []
        all_tsv = jobs_dir / "all.tsv"
        job_rows = count_job_rows(all_tsv) if all_tsv.exists() else sum(count_job_rows(p) for p in job_lists)
        monitor_paths = sorted(campaign.glob("launcher/*monitor*.log"))
        latest_monitor = tail_line(monitor_paths[-1]) if monitor_paths else ""
        related_dirs = [campaign]
        for token in campaign_tokens(name):
            for base in [
                ROOT / "logs/inscene15k_streaming",
                ROOT / "logs/inscene15k_streaming/runs",
            ]:
                candidate = base / token
                if candidate.exists():
                    related_dirs.append(candidate)
        related_dirs = list(dict.fromkeys(related_dirs))
        val_count = 0
        scan_notes = []
        for directory in related_dirs:
            count, note = find_count(directory, "val_summary.json")
            if count is None:
                scan_notes.append(f"{directory.relative_to(ROOT)}: {note}")
            else:
                val_count += count
            if note and count is not None:
                scan_notes.append(f"{directory.relative_to(ROOT)}: {note}")
        final_like = []
        for directory in related_dirs:
            try:
                for p in directory.glob("*"):
                    if (
                        p.is_file()
                        and any(token in p.name.lower() for token in ["final_metrics", "validation", "result", "best"])
                        and p.suffix.lower() in {".csv", ".tsv", ".json", ".md"}
                    ):
                        final_like.append(p)
            except OSError as exc:
                scan_notes.append(f"{directory.relative_to(ROOT)}: {exc}")
        status = "not_reportable_no_stable_final_metrics_found"
        if final_like:
            status = "reportable_only_if_final_metrics_are_reviewed"
        if scan_notes:
            status = status + "_scan_notes"
        rows.append(
            {
                "campaign": name,
                "exists": True,
                "job_lists": len(job_lists),
                "declared_jobs": job_rows,
                "val_summary_files": val_count,
                "final_like_files": len(final_like),
                "latest_monitor_line": latest_monitor,
                "status": status,
                "related_dirs": ";".join(str(p.relative_to(ROOT)) for p in related_dirs),
                "scan_notes": "; ".join(scan_notes),
            }
        )
    fields = [
        "campaign",
        "exists",
        "job_lists",
        "declared_jobs",
        "val_summary_files",
        "final_like_files",
        "latest_monitor_line",
        "status",
        "related_dirs",
        "scan_notes",
    ]
    write_csv(out_dir / "new_backend_artifact_audit.csv", rows, fields)
    md = [
        "# New Backend Artifact Audit",
        "",
        "Campaigns inspected: MiniMax-H3, LingBot, and probe-fill runs currently present under `logs/`.",
        "",
        markdown_table(rows, fields),
        "",
        "Interpretation: job artifacts alone are not stable metrics. A backend becomes reportable only after the job set has a declared manifest, completed/evaluable summaries, and a canonical aggregate table checked into or archived with the run.",
        "",
    ]
    (out_dir / "new_backend_artifact_audit.md").write_text("\n".join(md))


def audit_dirty_workspace(out_dir: Path) -> None:
    try:
        proc = subprocess.run(
            ["git", "status", "--short"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        status = proc.stdout
        error = proc.stderr
    except subprocess.TimeoutExpired:
        status = ""
        error = "git status --short timed out after 60s"
    (out_dir / "dirty_artifact_inventory.txt").write_text(status + (("\nSTDERR:\n" + error) if error else ""))
    lines = [line for line in status.splitlines() if line.strip()]
    code_counts = Counter(line[:2] for line in lines)
    untracked = [line for line in lines if line.startswith("??")]
    modified = [line for line in lines if not line.startswith("??")]
    md = [
        "# Dirty Workspace Inventory",
        "",
        f"- Total dirty entries from `git status --short`: {len(lines)}",
        f"- Untracked entries: {len(untracked)}",
        f"- Modified/staged entries: {len(modified)}",
        f"- Status code counts: {dict(sorted(code_counts.items()))}",
        "",
        "Reproducibility-critical small files should be committed or archived with checksums. Large logs/checkpoints should usually be archived outside git with manifest paths and hashes.",
        "",
        "## First 80 Dirty Entries",
        "",
        "```text",
        "\n".join(lines[:80]),
        "```",
        "",
    ]
    if error:
        md.extend(["## Git Status Error", "", "```text", error, "```", ""])
    (out_dir / "dirty_artifact_inventory.md").write_text("\n".join(md))


def write_design_note(out_dir: Path) -> None:
    md = [
        "# Follow-up Experiment Design Notes",
        "",
        "## Q1: C-series high R@1 vs shortcut",
        "",
        "The first-pass verification is the baseline audit in `c_shortcut_audit.csv`: compare each C row against no-action, shuffled-action, and last-observation/copy baselines. Rows where the probe does not beat these controls should not be described as strong action/spatial reasoning evidence.",
        "",
        "Next experiments to make this stronger:",
        "",
        "1. Freeze the chosen C config, then evaluate on held-out test with the same three shortcut baselines.",
        "2. Add action-token ablations: zero action, shuffle actions within trajectory, shuffle actions across scenes, and reverse action order.",
        "3. Add copy/smoothness baselines: last observation, linear feature extrapolation, nearest neighbor in prefix feature space, and scene-level retrieval.",
        "4. Report per-horizon curves, because copy/smoothness shortcuts should be strongest at short horizons and decay with longer horizons.",
        "5. Stratify by motion magnitude and viewpoint change; real action reasoning should improve most when static retrieval is insufficient.",
        "",
        "## Q3: B2 without GT spatial mask",
        "",
        "The essence is identifiability: without a ground-truth spatial mask, a model may solve the object-location query by category/scene priors, query wording, visible-object bias, or dataset placement bias rather than by binding the queried object to hidden 3D location. The current B2 task can therefore measure a mixture of spatial grounding and dataset/query shortcuts.",
        "",
        "Experiments to separate these factors:",
        "",
        "1. Query shuffle within scene: keep the scene/video fixed but swap object queries. A real object-grounded probe should collapse; a scene-prior probe will remain high.",
        "2. Category-balanced splits: ensure train/val/test category and room/layout distributions do not leak location priors.",
        "3. Position-prior baseline: predict the most common hidden-object bearing/distance by category, room type, and prefix length without visual features.",
        "4. Same-category hard negatives: evaluate objects of the same class in different positions, so category alone is not enough.",
        "5. Query-only and feature-only baselines: train with text/query only and with visual features but masked query. Any strong score here is direct bias evidence.",
        "6. Synthetic/object-controlled subset: use scenes where object location is randomized or counterbalanced, even if small, as a bias stress test.",
        "",
        "## Validation vs test protocol",
        "",
        "Evaluation after training is only a test if the split was never used to choose model family, layer, prefix, checkpoint, seed, task formulation, or reportable hyperparameters. In this repo, most discovered summaries are named `val_summary.json`, so they are evaluation-mode validation results. A frozen test protocol should use validation for selection, freeze the config, then run exactly once on a held-out test manifest and repeat the whole train/select/evaluate procedure over multiple seeds.",
        "",
    ]
    (out_dir / "experiment_design_notes.md").write_text("\n".join(md))


def write_index(out_dir: Path) -> None:
    files = sorted(p.name for p in out_dir.iterdir() if p.is_file())
    md = [
        "# Current Risk Audit Index",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "Artifacts:",
        "",
    ]
    md.extend(f"- `{name}`" for name in files)
    md.append("")
    (out_dir / "README.md").write_text("\n".join(md))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="logs/current_audit_20260817", help="Output directory")
    args = parser.parse_args()
    out_dir = (ROOT / args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    audit_c_shortcuts(out_dir)
    audit_wan_cog(out_dir)
    audit_new_backends(out_dir)
    audit_dirty_workspace(out_dir)
    write_design_note(out_dir)
    write_index(out_dir)
    print(f"Wrote audit artifacts to {out_dir}")


if __name__ == "__main__":
    main()
