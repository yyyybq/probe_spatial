#!/usr/bin/env python3
"""Recover historical experiment universe without overwriting canonical registry.

Outputs:
  experiments/expected_experiments.csv
  experiments/registry_reconstructed_candidate.csv
  archive/consolidation_history/20260817/incomplete_reconciliation.csv
  results/reconstructed/default8_raw_index.csv
  archive/consolidation_history/20260817/coverage_*.md
  archive/consolidation_history/20260817/historical_reconstruction_report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_ARCHIVE = ROOT / "archive/consolidation_history/20260817"
A_B_TABLE = ROOT / "logs/eval_results/results_table.csv"
C_RAW = ROOT / "logs/inscene15k_streaming/results/c_eval_with_extra_20260728/c_series_raw_metrics_with_extra.csv"
WAN_COG = ROOT / "logs/probing_jobs_20260725_ab/recovery_20260730/perf_analysis/evaluation_results_all_probes.tsv"
WAN_COG_CKPT = ROOT / "logs/probing_jobs_20260725_ab/recovery_20260730/perf_analysis/canonical_best_checkpoint_registry.tsv"
OLD_REGISTRY = ROOT / "experiments/registry.csv"
OLD_INCOMPLETE = ROOT / "experiments/missing_experiments.csv"

EXPECTED_FIELDS = [
    "campaign",
    "model",
    "model_variant",
    "layer",
    "prefix",
    "probe",
    "horizon",
    "seed",
    "expected",
    "expectation_source",
    "notes",
]
CANDIDATE_FIELDS = [
    "experiment_id",
    *EXPECTED_FIELDS,
    "family",
    "old_registry_status",
    "reconstructed_status",
    "expected_evidence",
    "declared_evidence",
    "launched_evidence",
    "run_found",
    "train_started",
    "train_completed",
    "checkpoint_found",
    "checkpoint_path",
    "eval_started",
    "eval_completed",
    "raw_metric_found",
    "canonical_metric_found",
    "failure_evidence",
    "archaeology_notes",
    "evidence",
    "config",
    "dataset_manifest",
    "job_manifest",
    "launcher_job_id",
    "run_artifact_path",
    "raw_metric_path",
    "summary_path",
    "canonical_metric_path",
    "metric_name",
    "metric_value",
    "supersedes",
    "superseded_by",
    "rerun_needed",
    "reason",
]
RECON_STATUSES = [
    "EXPECTED_ONLY",
    "JOB_DECLARED",
    "JOB_LAUNCHED",
    "RUN_FOUND",
    "TRAIN_RUNNING",
    "TRAIN_RUNNING_OR_PARTIAL",
    "TRAIN_COMPLETE_EVAL_MISSING",
    "EVAL_COMPLETE_METRIC_UNRESOLVED",
    "COMPLETE_CANONICAL",
    "COMPLETE_RAW_ONLY",
    "COMPLETE_SUMMARY_ONLY",
    "SUPERSEDED",
    "FAILED",
    "UNKNOWN",
]
STATUS_RANK = {
    "COMPLETE_CANONICAL": 0,
    "COMPLETE_RAW_ONLY": 1,
    "COMPLETE_SUMMARY_ONLY": 2,
    "EVAL_COMPLETE_METRIC_UNRESOLVED": 3,
    "TRAIN_COMPLETE_EVAL_MISSING": 4,
    "TRAIN_RUNNING": 5,
    "TRAIN_RUNNING_OR_PARTIAL": 6,
    "RUN_FOUND": 7,
    "JOB_LAUNCHED": 8,
    "JOB_DECLARED": 9,
    "EXPECTED_ONLY": 10,
    "SUPERSEDED": 11,
    "FAILED": 12,
    "UNKNOWN": 99,
}

METRIC_KEYS = [
    "mean_ang_err_deg",
    "mean_az_err_deg",
    "mean_el_err_deg",
    "top1",
    "top3",
    "balanced_acc",
    "roc_auc",
    "mean_overlap_mae",
    "hard_acc",
    "Auc_5",
    "Auc_10",
    "Auc_30",
    "Rac_15",
    "Tac_15",
    "mean_loss_depth",
    "mean_loss_camera",
    "mean_loss_identity",
    "global_R@1",
    "intervention_validity",
]


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


def rel(path: str | Path) -> str:
    text = str(path or "")
    if not text:
        return ""
    p = Path(text)
    if p.is_absolute():
        try:
            return str(p.relative_to(ROOT))
        except ValueError:
            return text
    return text


def clean(value: object) -> str:
    text = str(value if value is not None else "").strip().lower()
    text = text.replace("@", "at").replace("/", "_").replace(" ", "_")
    text = re.sub(r"[^a-z0-9_.+-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "na"


def parse_float(value: object) -> float | None:
    text = str(value if value is not None else "").strip()
    if not text or text.lower() in {"n/a", "nan", "none", "?"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def bool_text(value: bool) -> str:
    return "True" if value else "False"


def is_true(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def exp_id(campaign: str, probe: str, model: str, variant: str, layer: str, prefix: str, horizon: str, seed: str) -> str:
    return "_".join(
        [
            clean(campaign),
            clean(probe),
            clean(model),
            clean(variant),
            f"l{clean(layer)}",
            f"p{clean(prefix)}",
            f"h{clean(horizon)}",
            f"s{clean(seed)}",
        ]
    )


def family(probe: str, campaign: str = "") -> str:
    if probe in {"streaming_depth", "streaming_camera", "streaming_mask"} or probe.startswith("a1_"):
        return "A1"
    if probe in {"view_consistency", "a2"}:
        return "A2"
    if probe == "ego_belief":
        return "B1"
    if probe in {"ego_belief_v2", "visible_ego_belief_v2", "b2"}:
        return "B2"
    if probe in {"action_dynamics", "c1"}:
        return "C1"
    if probe in {"path_integration", "c2"}:
        return "C2"
    if probe in {"counterfactual", "c3"}:
        return "C3"
    return "UNKNOWN"


def normalize_probe(probe: str, task: str = "") -> str:
    probe = probe.strip()
    mapping = {
        "streaming_depth/depth": "streaming_depth",
        "streaming_depth/camera": "streaming_depth",
        "streaming_depth/mask": "streaming_mask",
        "a1_depth": "streaming_depth",
        "a1_camera": "streaming_depth",
        "a1_mask": "streaming_mask",
        "a2": "view_consistency",
        "b2": "ego_belief_v2",
        "c1": "action_dynamics",
        "c2": "path_integration",
        "c3": "counterfactual",
    }
    if probe in mapping:
        return mapping[probe]
    if task.startswith("a1_depth") or task.startswith("a1_camera"):
        return "streaming_depth"
    if task.startswith("a1_mask"):
        return "streaming_mask"
    if task.startswith("a2_"):
        return "view_consistency"
    if task.startswith("b2_"):
        return "ego_belief_v2"
    if task.startswith("c1_"):
        return "action_dynamics"
    if task.startswith("c2_"):
        return "path_integration"
    if task.startswith("c3_"):
        return "counterfactual"
    return probe


def layer_from_task(task: str, fallback: str = "") -> str:
    match = re.search(r"layer(-?\d+)", task)
    return match.group(1) if match else fallback


def row_key(row: dict[str, object]) -> tuple[str, str, str, str, str, str]:
    return (
        str(row.get("campaign", "")),
        str(row.get("model", "")),
        str(row.get("layer", "")),
        str(row.get("prefix", "")),
        str(row.get("probe", "")),
        str(row.get("seed", "")),
    )


def add_expected(rows: dict[tuple[str, str, str, str, str, str, str], dict[str, object]], row: dict[str, object]) -> None:
    row.setdefault("horizon", "all")
    row.setdefault("seed", "42")
    row.setdefault("expected", "True")
    row.setdefault("notes", "")
    key = row_key(row)
    if key not in rows:
        rows[key] = row
    else:
        prev = rows[key]
        sources = set(filter(None, str(prev.get("expectation_source", "")).split(";")))
        sources.update(filter(None, str(row.get("expectation_source", "")).split(";")))
        prev["expectation_source"] = ";".join(sorted(sources))
        notes = [str(prev.get("notes", "")), str(row.get("notes", ""))]
        prev["notes"] = "; ".join(n for n in notes if n)


def default_expected(rows: dict[tuple[str, str, str, str, str, str, str], dict[str, object]]) -> None:
    for row in read_csv(A_B_TABLE):
        series = row.get("series", "").strip().lower()
        if series not in {"a1", "a2", "b2", "c1", "c2", "c3"}:
            continue
        task = row.get("task", "")
        probe = normalize_probe(row.get("probe_type", ""), task)
        add_expected(
            rows,
            {
                "campaign": "default8",
                "model": row.get("model", ""),
                "model_variant": "default_layer_benchmark",
                "layer": layer_from_task(task),
                "prefix": row.get("prefix", ""),
                "probe": probe,
                "horizon": "all",
                "seed": "42",
                "expected": "True",
                "expectation_source": rel(A_B_TABLE),
                "notes": f"task={task}; source_status={row.get('status')}; metric={row.get('metric_name')}",
            },
        )


def parse_job_id(job_id: str) -> tuple[str, str, str, str] | None:
    match = re.match(r"(.+?)_(aether|cogvideox|dino|f3r|qwen2_5_vl|videomae|vjepa2|wan|minimax_h3|lingbot)_p(\d+)_layer(-?\d+)$", job_id)
    if not match:
        return None
    return match.group(1), match.group(2), match.group(3), match.group(4)


def wan_cog_expected(rows: dict[tuple[str, str, str, str, str, str, str], dict[str, object]]) -> None:
    for row in read_csv(WAN_COG, delimiter="\t"):
        add_expected(
            rows,
            {
                "campaign": "wan_cog_layer_sweep",
                "model": row.get("vfm", ""),
                "model_variant": "multi_layer",
                "layer": row.get("layer", ""),
                "prefix": row.get("prefix", ""),
                "probe": row.get("probe", ""),
                "horizon": "all",
                "seed": "42",
                "expected": "True",
                "expectation_source": rel(WAN_COG),
                "notes": f"job_id={row.get('job_id')}; historical_kind={row.get('kind')}; historical_class={row.get('class')}",
            },
        )
    for row in read_csv(WAN_COG_CKPT, delimiter="\t"):
        parsed = parse_job_id(row.get("job_id", ""))
        if not parsed:
            continue
        probe, model, prefix, layer = parsed
        add_expected(
            rows,
            {
                "campaign": "wan_cog_layer_sweep",
                "model": model,
                "model_variant": "multi_layer",
                "layer": layer,
                "prefix": prefix,
                "probe": probe,
                "horizon": "all",
                "seed": "42",
                "expected": "True",
                "expectation_source": rel(WAN_COG_CKPT),
                "notes": f"job_id={row.get('job_id')}; checkpoint_class={row.get('checkpoint_class')}; eval_status={row.get('eval_status')}",
            },
        )


def campaign_for_model(model: str) -> str | None:
    if model in {"videomae", "vjepa2"}:
        return "videomae_vjepa_layer_sweep"
    if model == "minimax_h3":
        return "minimax_h3_layer_benchmark"
    if model == "lingbot":
        return "lingbot_video_layer_benchmark"
    if model in {"wan", "cogvideox"}:
        return "wan_cog_layer_sweep"
    return None


def job_campaign(path: Path) -> str | None:
    text = str(path)
    if "vjepa2_videomae" in text:
        return "videomae_vjepa_layer_sweep"
    if "minimax_h3" in text:
        return "minimax_h3_layer_benchmark"
    if "probe_fill_lingbot" in text or ("lingbot" in text and "probe_fill" in text):
        return "lingbot_video_layer_benchmark"
    if "probe_fill_fill" in text:
        # Multi-model fill campaign; resolve per-row via campaign_for_model().
        return "probe_fill_fill"
    if "lingbot" in text:
        return "lingbot_video_layer_benchmark"
    return None


def all_job_files() -> list[Path]:
    files = []
    for path in (ROOT / "logs").glob("probing_jobs_*/jobs/all.tsv"):
        if job_campaign(path):
            files.append(path)
    return sorted(set(files))


def parse_job_rows(path: Path) -> list[dict[str, str]]:
    lines = [line.rstrip("\n") for line in path.read_text(errors="replace").splitlines() if line.strip()]
    if not lines:
        return []
    first = lines[0].split("\t")
    rows = []
    if first and first[0] == "vfm":
        reader = csv.DictReader(lines, delimiter="\t")
        return list(reader)
    for line in lines:
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        row = {
            "vfm": parts[0],
            "probe": parts[1],
            "prefix": parts[2],
            "layer": parts[3],
            "manifest": parts[-2] if len(parts) >= 6 else "",
            "cfg": parts[-1] if len(parts) >= 6 else parts[4],
        }
        if len(parts) >= 7 and parts[-1].endswith(".ckpt"):
            row["checkpoint"] = parts[-1]
            row["cfg"] = parts[-2]
            row["manifest"] = parts[-3]
        if len(parts) >= 6 and parts[4].isdigit():
            row["disk_layer"] = parts[4]
        rows.append(row)
    return rows


def job_expected(rows: dict[tuple[str, str, str, str, str, str, str], dict[str, object]]) -> None:
    for path in all_job_files():
        campaign = job_campaign(path)
        if not campaign:
            continue
        # probe_fill_fill is a recovery launcher over the existing 1096 universe; do not
        # expand expected_experiments with duplicate multi_layer rows.
        if campaign == "probe_fill_fill":
            continue
        for row in parse_job_rows(path):
            model = row.get("vfm", "")
            probe = normalize_probe(row.get("probe", ""))
            add_expected(
                rows,
                {
                    "campaign": campaign,
                    "model": model,
                    "model_variant": "multi_layer",
                    "layer": row.get("layer", ""),
                    "prefix": row.get("prefix", ""),
                    "probe": probe,
                    "horizon": "all",
                    "seed": "42",
                    "expected": "True",
                    "expectation_source": rel(path),
                    "notes": f"manifest={rel(row.get('manifest', ''))}; cfg={rel(row.get('cfg', ''))}; disk_layer={row.get('disk_layer', '')}; checkpoint={rel(row.get('checkpoint', ''))}",
                },
            )


def run_dir_expected(rows: dict[tuple[str, str, str, str, str, str, str], dict[str, object]]) -> None:
    run_index = build_run_index()
    for key, items in run_index.items():
        campaign_guess, model, layer, prefix, probe = key
        if campaign_guess not in {"default8", "videomae_vjepa_layer_sweep", "minimax_h3_layer_benchmark", "lingbot_video_layer_benchmark", "wan_cog_layer_sweep"}:
            continue
        add_expected(
            rows,
            {
                "campaign": campaign_guess,
                "model": model,
                "model_variant": "run_dir_discovered",
                "layer": layer,
                "prefix": prefix,
                "probe": probe,
                "horizon": "all",
                "seed": "42",
                "expected": "True",
                "expectation_source": "run_dir",
                "notes": f"run_dirs={';'.join(rel(p) for p in items[:3])}",
            },
        )


def parse_run_name(name: str) -> tuple[str, str, str, str, str] | None:
    probe_map = {
        "a1_depth": "streaming_depth",
        "a1_camera": "streaming_depth",
        "a1_mask": "streaming_mask",
        "a2": "view_consistency",
        "b2": "ego_belief_v2",
        "c1": "action_dynamics",
        "c2": "path_integration",
        "c3": "counterfactual",
    }
    match = re.search(
        r"(streaming_depth|streaming_camera|streaming_mask|view_consistency|ego_belief_v2|ego_belief|action_dynamics|path_integration|counterfactual|a1_depth|a1_camera|a1_mask|a2|b2|c1|c2|c3)_(aether|cogvideox|dino|f3r|qwen2_5_vl|videomae|vjepa2|wan|minimax_h3|lingbot)_p(\d+)_layer(-?\d+)",
        name,
    )
    if not match:
        return None
    probe = probe_map.get(match.group(1), match.group(1))
    model = match.group(2)
    prefix = match.group(3)
    layer = match.group(4)
    if "vjepa2_videomae" in name:
        campaign = "videomae_vjepa_layer_sweep"
    elif "probe_fill_fill" in name or name.startswith("probe_fill_"):
        campaign = campaign_for_model(model) or "unknown_run_dir"
    elif "minimax_h3" in name:
        campaign = "minimax_h3_layer_benchmark"
    elif "lingbot" in name or "probe_fill_lingbot" in name:
        campaign = "lingbot_video_layer_benchmark"
    elif "wan_cog_layer" in name:
        campaign = "wan_cog_layer_sweep"
    elif name.startswith("inscene15k_streaming_"):
        campaign = "default8"
    else:
        campaign = campaign_for_model(model) or "unknown_run_dir"
    return campaign, model, layer, prefix, probe


def build_run_index() -> dict[tuple[str, str, str, str, str], list[Path]]:
    root = ROOT / "logs/inscene15k_streaming/runs"
    index: dict[tuple[str, str, str, str, str], list[Path]] = defaultdict(list)
    if not root.exists():
        return index
    nested_tokens = (
        "vjepa2_videomae",
        "minimax_h3",
        "probe_fill_lingbot",
        "probe_fill_fill",
        "wan_cog_layer",
    )
    for entry in os.scandir(root):
        if not entry.is_dir():
            continue
        parsed = parse_run_name(entry.name)
        if parsed:
            index[parsed].append(Path(entry.path))
        if any(token in entry.name for token in nested_tokens):
            try:
                for child in os.scandir(entry.path):
                    if child.is_dir():
                        parsed_child = parse_run_name(child.name)
                        if parsed_child:
                            index[parsed_child].append(Path(child.path))
            except OSError:
                pass
    return index


def build_summary_index(run_index: dict[tuple[str, str, str, str, str], list[Path]]) -> dict[tuple[str, str, str, str, str], Path]:
    out = {}
    for key, paths in run_index.items():
        for path in paths:
            candidates = [
                path / "eval/val_summary.json",
                path / "eval/test_summary.json",
                path / "val_summary.json",
            ]
            found = next((p for p in candidates if p.exists()), None)
            if found:
                out[key] = found
                break
    return out


def read_log_probe(path: Path, head_bytes: int = 65536, tail_bytes: int = 196608) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            head = f.read(min(head_bytes, size))
            if size > tail_bytes:
                f.seek(max(0, size - tail_bytes))
            tail = f.read(tail_bytes)
        return (head + b"\n" + tail).decode("utf-8", errors="replace")
    except OSError:
        return ""


def summary_payload(path: Path) -> dict[str, object]:
    try:
        with path.open() as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def primary_summary_metric(path: Path) -> tuple[str, str]:
    payload = summary_payload(path)
    for key in METRIC_KEYS:
        value = payload.get(key)
        if parse_float(value) is not None:
            return key, str(value)
    return "", ""


def inspect_run_dir(path: Path) -> dict[str, object]:
    candidates = [
        path / "eval/val_summary.json",
        path / "eval/test_summary.json",
        path / "val_summary.json",
        path / "test_summary.json",
    ]
    summary = next((p for p in candidates if p.exists()), None)
    metric_name, metric_value = primary_summary_metric(summary) if summary else ("", "")
    eval_dir = path / "eval"
    ckpts = sorted(path.glob("VideoProbe3D/*/checkpoints/*.ckpt"))
    if not ckpts:
        ckpts = sorted(path.glob("checkpoints/*.ckpt"))
    log_path = path / "inscene15k_streaming.log"
    log_text = read_log_probe(log_path) if log_path.exists() else ""
    failure_markers = [
        "Traceback (most recent call last)",
        "CUDA out of memory",
        "OutOfMemoryError",
        "UnpicklingError",
        "FileNotFoundError",
        "RuntimeError:",
        "Killed",
        "slurmstepd: error",
        "TIMEOUT",
    ]
    failure_hits = [marker for marker in failure_markers if marker in log_text]
    train_started = "Starting training" in log_text or "[probe-progress] event=train_epoch_start" in log_text
    train_completed = "`Trainer.fit` stopped" in log_text or ("max_epochs=" in log_text and "reached" in log_text)
    eval_completed = summary is not None and bool(metric_name)
    recently_active = False
    try:
        newest_mtime = path.stat().st_mtime
        if log_path.exists():
            newest_mtime = max(newest_mtime, log_path.stat().st_mtime)
        for ckpt in ckpts[-3:]:
            newest_mtime = max(newest_mtime, ckpt.stat().st_mtime)
        recently_active = (time.time() - newest_mtime) < 8 * 3600
    except OSError:
        recently_active = False
    notes = []
    if log_path.exists():
        notes.append("training_log")
    if (path / ".hydra/config.yaml").exists():
        notes.append("hydra_config")
    if ckpts:
        notes.append(f"checkpoint_count={len(ckpts)}")
    if summary:
        notes.append("eval_summary")
    if failure_hits:
        notes.append("failure_marker=" + "|".join(failure_hits[:3]))
    if recently_active:
        notes.append("recently_active")
    return {
        "run_found": True,
        "train_started": train_started,
        "train_completed": train_completed,
        "checkpoint_found": bool(ckpts),
        "checkpoint_path": rel(ckpts[-1]) if ckpts else "",
        "eval_started": eval_dir.exists() or summary is not None,
        "eval_completed": eval_completed,
        "summary_path": rel(summary) if summary else "",
        "summary_metric_name": metric_name,
        "summary_metric_value": metric_value,
        "failure_evidence": ";".join(failure_hits[:5]),
        "recently_active": recently_active,
        "archaeology_notes": ";".join(notes),
    }


def best_run_inspection(paths: list[Path]) -> dict[str, object]:
    if not paths:
        return {
            "run_found": False,
            "train_started": False,
            "train_completed": False,
            "checkpoint_found": False,
            "checkpoint_path": "",
            "eval_started": False,
            "eval_completed": False,
            "summary_path": "",
            "summary_metric_name": "",
            "summary_metric_value": "",
            "failure_evidence": "",
            "recently_active": False,
            "archaeology_notes": "",
        }
    inspected = [(path, inspect_run_dir(path)) for path in paths]

    def score(item: tuple[Path, dict[str, object]]) -> tuple[int, int, int, int, int]:
        _, info = item
        return (
            int(bool(info.get("summary_metric_value"))),
            int(bool(info.get("eval_completed"))),
            int(bool(info.get("train_completed"))),
            int(bool(info.get("checkpoint_found"))),
            int(bool(info.get("train_started"))),
        )

    path, info = sorted(inspected, key=score, reverse=True)[0]
    info["run_artifact_path"] = rel(path)
    if len(paths) > 1:
        info["archaeology_notes"] = ";".join(
            part for part in [str(info.get("archaeology_notes", "")), f"alternate_run_count={len(paths) - 1}"] if part
        )
    return info


def old_registry_index() -> dict[tuple[str, str, str, str, str], dict[str, str]]:
    out = {}
    for row in read_csv(OLD_REGISTRY):
        camp = {
            "default_streaming": "default8",
            "c_retrain_with_extra": "default8",
            "wan_cog_layer_sweep": "wan_cog_layer_sweep",
        }.get(row.get("model_variant", ""), row.get("model_variant", ""))
        key = (camp, row.get("model", ""), row.get("layer", ""), row.get("prefix", ""), normalize_probe(row.get("probe", "")))
        if row.get("status") == "COMPLETE" or key not in out:
            out[key] = row
    return out


def raw_metric_index() -> dict[tuple[str, str, str, str, str], dict[str, str]]:
    out = {}
    for row in read_csv(A_B_TABLE):
        series = row.get("series", "").strip().lower()
        if series not in {"a1", "a2", "b2", "c1", "c2", "c3"}:
            continue
        probe = normalize_probe(row.get("probe_type", ""), row.get("task", ""))
        key = ("default8", row.get("model", ""), layer_from_task(row.get("task", "")), row.get("prefix", ""), probe)
        metric = row.get("metric_value", "")
        if parse_float(metric) is not None:
            out[key] = {"path": rel(A_B_TABLE), "metric_name": row.get("metric_name", ""), "metric_value": metric}
    for row in read_csv(C_RAW):
        probe = normalize_probe(row.get("probe_type", ""))
        key = ("default8", row.get("vfm_name", ""), row.get("layer", ""), row.get("prefix", ""), probe)
        metric_name = {
            "action_dynamics": "global_R@1",
            "path_integration": "global_R@1",
            "counterfactual": "intervention_validity",
        }.get(probe, row.get("primary_metric", ""))
        metric_value = row.get(metric_name, "") or row.get("primary_value", "")
        if parse_float(metric_value) is not None:
            out[key] = {"path": rel(C_RAW), "metric_name": metric_name, "metric_value": metric_value}
    for row in read_csv(WAN_COG, delimiter="\t"):
        probe = normalize_probe(row.get("probe", ""))
        key = ("wan_cog_layer_sweep", row.get("vfm", ""), row.get("layer", ""), row.get("prefix", ""), probe)
        metric_name = {
            "ego_belief_v2": "mean_ang_err_deg",
            "ego_belief": "mean_az_err_deg",
            "streaming_depth": "mean_loss_depth",
            "view_consistency": "balanced_acc",
        }.get(probe, "")
        metric_value = row.get(metric_name, "")
        if parse_float(metric_value) is not None:
            out[key] = {"path": rel(WAN_COG), "metric_name": metric_name, "metric_value": metric_value, "provisional": row.get("provisional", "")}
    return out


def job_index() -> dict[tuple[str, str, str, str, str], dict[str, str]]:
    out = {}
    for path in all_job_files():
        campaign = job_campaign(path)
        if not campaign:
            continue
        for row in parse_job_rows(path):
            model = row.get("vfm", "")
            row_campaign = campaign_for_model(model) if campaign == "probe_fill_fill" else campaign
            if not row_campaign:
                continue
            key = (row_campaign, model, row.get("layer", ""), row.get("prefix", ""), normalize_probe(row.get("probe", "")))
            out[key] = {
                "path": rel(path),
                "manifest": rel(row.get("manifest", "")),
                "config": rel(row.get("cfg", "")),
                "checkpoint": rel(row.get("checkpoint", "")),
            }
    return out


def infer_config(probe: str, model: str) -> str:
    path = ROOT / "configs/experiment/inscene15k_streaming" / f"{probe}_{model}_v1.yaml"
    if path.exists():
        return rel(path)
    return ""


def classify_expected(expected: list[dict[str, object]]) -> list[dict[str, object]]:
    old = old_registry_index()
    raw = raw_metric_index()
    jobs = job_index()
    runs = build_run_index()
    out = []
    for row in expected:
        key = (str(row["campaign"]), str(row["model"]), str(row["layer"]), str(row["prefix"]), str(row["probe"]))
        old_row = old.get(key, {})
        raw_row = raw.get(key, {})
        job_row = jobs.get(key, {})
        run_paths = runs.get(key, [])
        run_info = best_run_inspection(run_paths)
        status = "UNKNOWN"
        reason = ""
        evidence_parts = []
        canonical = old_row.get("canonical_metric_path", "") if old_row.get("status") == "COMPLETE" else ""
        raw_metric_found = bool(raw_row.get("metric_value"))
        canonical_metric_found = bool(canonical)
        summary_metric_found = bool(run_info.get("summary_metric_value"))
        is_provisional_raw = raw_row.get("provisional", "").lower() == "true"
        if canonical:
            status = "COMPLETE_CANONICAL"
            reason = "matched COMPLETE row in current canonical registry"
            evidence_parts.append(f"registry={old_row.get('experiment_id')}")
        elif raw_metric_found and is_provisional_raw:
            status = "EVAL_COMPLETE_METRIC_UNRESOLVED"
            reason = "parseable historical Wan/Cog metric exists but source marks it provisional; do not use as formal best"
            evidence_parts.append(f"raw_metric={raw_row.get('path')}")
        elif raw_metric_found:
            status = "COMPLETE_RAW_ONLY"
            reason = "parseable raw aggregate metric exists but candidate is not canonical COMPLETE"
            evidence_parts.append(f"raw_metric={raw_row.get('path')}")
        elif summary_metric_found:
            status = "COMPLETE_SUMMARY_ONLY"
            reason = "run/eval summary exists but no canonical registry row was matched"
            evidence_parts.append(f"summary={run_info.get('summary_path')}")
        elif run_paths and run_info.get("failure_evidence") and not run_info.get("train_completed") and not run_info.get("checkpoint_found"):
            status = "FAILED"
            reason = "run directory contains failure evidence in training log"
            evidence_parts.append(f"failure={run_info.get('failure_evidence')}")
        elif run_paths and run_info.get("train_completed") and run_info.get("checkpoint_found"):
            status = "TRAIN_COMPLETE_EVAL_MISSING"
            reason = "training reached max epochs and checkpoint exists, but no eval summary/raw metric was matched"
            evidence_parts.append(f"run_dir={run_info.get('run_artifact_path')}")
        elif run_paths and run_info.get("train_started") and not run_info.get("train_completed") and run_info.get("recently_active"):
            status = "TRAIN_RUNNING"
            reason = "training log shows recent activity without completion marker"
            evidence_parts.append(f"run_dir={run_info.get('run_artifact_path')}")
        elif run_paths and run_info.get("train_started") and not run_info.get("train_completed"):
            status = "TRAIN_RUNNING_OR_PARTIAL"
            reason = "training log started but no completion marker or metric was found"
            evidence_parts.append(f"run_dir={run_info.get('run_artifact_path')}")
        elif run_paths and run_info.get("checkpoint_found") and not run_info.get("train_completed"):
            # Usable checkpoint from interrupted/retry run; treat as train-complete for eval queueing.
            status = "TRAIN_COMPLETE_EVAL_MISSING"
            reason = "checkpoint exists without Trainer.fit completion marker"
            evidence_parts.append(f"run_dir={run_info.get('run_artifact_path')}")
        elif run_paths:
            status = "RUN_FOUND"
            reason = "run directory exists, but train/eval lifecycle could not be resolved by shallow archaeology"
            evidence_parts.append(f"run_dir={run_info.get('run_artifact_path')}")
        elif job_row:
            status = "JOB_DECLARED"
            reason = "job manifest exists but no matching run/metric was found"
            evidence_parts.append(f"job_manifest={job_row.get('path')}")
        elif "run_dir" in str(row.get("expectation_source", "")):
            status = "UNKNOWN"
            reason = "run-dir-derived expectation could not be reconciled further"
        else:
            status = "EXPECTED_ONLY"
            reason = "expected by design/report but launch/run/metric evidence not found"

        run_artifact_path = str(run_info.get("run_artifact_path", ""))
        expected_evidence = str(row.get("expectation_source", ""))
        declared_evidence = job_row.get("path", "")
        launched_evidence = run_artifact_path or old_row.get("launcher_job_id", "")
        experiment_id = exp_id(
            str(row["campaign"]),
            str(row["probe"]),
            str(row["model"]),
            str(row["model_variant"]),
            str(row["layer"]),
            str(row["prefix"]),
            str(row.get("horizon", "all")),
            str(row.get("seed", "42")),
        )
        if status in {"FAILED", "EXPECTED_ONLY"}:
            rerun_needed = "True"
        elif status in {"JOB_DECLARED", "JOB_LAUNCHED", "RUN_FOUND", "TRAIN_RUNNING", "TRAIN_RUNNING_OR_PARTIAL", "TRAIN_COMPLETE_EVAL_MISSING", "EVAL_COMPLETE_METRIC_UNRESOLVED", "UNKNOWN"}:
            rerun_needed = "REVIEW"
        else:
            rerun_needed = "False"
        metric_name = raw_row.get("metric_name") or str(run_info.get("summary_metric_name", "")) or old_row.get("metric_name", "")
        metric_value = raw_row.get("metric_value") or str(run_info.get("summary_metric_value", "")) or old_row.get("metric_value", "")
        out.append(
            {
                "experiment_id": experiment_id,
                **row,
                "family": family(str(row["probe"]), str(row["campaign"])),
                "old_registry_status": old_row.get("status", ""),
                "reconstructed_status": status,
                "expected_evidence": expected_evidence,
                "declared_evidence": declared_evidence,
                "launched_evidence": launched_evidence,
                "run_found": bool_text(bool(run_info.get("run_found"))),
                "train_started": bool_text(bool(run_info.get("train_started"))),
                "train_completed": bool_text(bool(run_info.get("train_completed"))),
                "checkpoint_found": bool_text(bool(run_info.get("checkpoint_found"))),
                "checkpoint_path": run_info.get("checkpoint_path", ""),
                "eval_started": bool_text(bool(run_info.get("eval_started"))),
                "eval_completed": bool_text(bool(run_info.get("eval_completed"))),
                "raw_metric_found": bool_text(raw_metric_found or summary_metric_found),
                "canonical_metric_found": bool_text(canonical_metric_found),
                "failure_evidence": run_info.get("failure_evidence", ""),
                "archaeology_notes": run_info.get("archaeology_notes", ""),
                "evidence": ";".join(evidence_parts),
                "config": job_row.get("config", "") or infer_config(str(row["probe"]), str(row["model"])),
                "dataset_manifest": job_row.get("manifest", ""),
                "job_manifest": job_row.get("path", ""),
                "launcher_job_id": old_row.get("launcher_job_id", ""),
                "run_artifact_path": run_artifact_path,
                "raw_metric_path": raw_row.get("path", ""),
                "summary_path": run_info.get("summary_path", ""),
                "canonical_metric_path": canonical,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "supersedes": "",
                "superseded_by": "",
                "rerun_needed": rerun_needed,
                "reason": reason,
            }
        )
    return out


def expected_matrix() -> list[dict[str, object]]:
    rows: dict[tuple[str, str, str, str, str, str, str], dict[str, object]] = {}
    default_expected(rows)
    wan_cog_expected(rows)
    job_expected(rows)
    run_dir_expected(rows)
    return list(rows.values())


def reconcile_incomplete(candidate: list[dict[str, object]]) -> list[dict[str, object]]:
    cand_by_loose = defaultdict(list)
    for row in candidate:
        key = (row.get("campaign"), row.get("model"), row.get("layer"), row.get("prefix"), row.get("probe"))
        cand_by_loose[key].append(row)
    output = []
    for old in read_csv(OLD_INCOMPLETE):
        camp = {
            "default_streaming": "default8",
            "wan_cog_layer_sweep": "wan_cog_layer_sweep",
        }.get(old.get("model_variant", ""), "default8" if old.get("family") in {"B2"} else old.get("model_variant", ""))
        key = (camp, old.get("model"), old.get("layer"), old.get("prefix"), normalize_probe(old.get("probe", "")))
        matches = cand_by_loose.get(key, [])
        best = sorted(matches, key=lambda r: STATUS_RANK.get(str(r.get("reconstructed_status")), 99))[0] if matches else {}
        reconstructed = best.get("reconstructed_status", "UNKNOWN")
        output.append(
            {
                "experiment_id": old.get("experiment_id", ""),
                "current_status": old.get("status", ""),
                "reconstructed_status": reconstructed,
                "evidence": best.get("evidence", ""),
                "replacement_experiment_id": best.get("experiment_id", ""),
                "rerun_needed": best.get("rerun_needed", "UNKNOWN"),
                "reason": best.get("reason", "no candidate match found"),
                "raw_artifact_path": best.get("raw_metric_path", ""),
                "summary_path": best.get("summary_path", ""),
                "run_artifact_path": best.get("run_artifact_path", ""),
            }
        )
    return output


def aggregate_status(statuses: list[str]) -> str:
    if not statuses:
        return "NOT_RUN"
    return sorted(statuses, key=lambda s: STATUS_RANK.get(s, 99))[0]


def write_coverage(campaign: str, path: Path, candidate: list[dict[str, object]]) -> None:
    rows = [r for r in candidate if r.get("campaign") == campaign]
    probes = ["streaming_depth", "streaming_mask", "view_consistency", "ego_belief_v2", "action_dynamics", "path_integration", "counterfactual"]
    labels = {
        "streaming_depth": "A1-depth/camera",
        "streaming_mask": "A1-mask",
        "view_consistency": "A2",
        "ego_belief_v2": "B2",
        "action_dynamics": "C1",
        "path_integration": "C2",
        "counterfactual": "C3",
    }
    grouped: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    prefixes: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        key = (str(row.get("model", "")), str(row.get("layer", "")))
        grouped[key][str(row.get("probe", ""))].append(str(row.get("reconstructed_status", "")))
        prefixes[key].add(str(row.get("prefix", "")))
    lines = [
        f"# Coverage: {campaign}",
        "",
        f"Rows in reconstructed candidate: {len(rows)}",
        "",
        "| model/layer | prefixes | " + " | ".join(labels[p] for p in probes) + " |",
        "| --- | --- | " + " | ".join(["---"] * len(probes)) + " |",
    ]
    for key in sorted(grouped, key=lambda k: (k[0], int(k[1]) if str(k[1]).lstrip("-").isdigit() else 9999, k[1])):
        cells = [aggregate_status(grouped[key].get(p, [])) for p in probes]
        lines.append(f"| {key[0]}/L{key[1]} | {','.join(sorted(prefixes[key], key=lambda x: int(x) if x.isdigit() else 9999))} | " + " | ".join(cells) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_default_raw_index(candidate: list[dict[str, object]]) -> None:
    rows = [r for r in candidate if r.get("campaign") == "default8"]
    fields = [
        "experiment_id",
        "model",
        "layer",
        "prefix",
        "probe",
        "reconstructed_status",
        "raw_metric_path",
        "summary_path",
        "canonical_metric_path",
        "metric_name",
        "metric_value",
        "reason",
    ]
    write_csv(ROOT / "results/reconstructed/default8_raw_index.csv", rows, fields)


def write_recovered_raw_metrics(candidate: list[dict[str, object]]) -> None:
    rows = [
        {
            **r,
            "metric_provenance": (
                "canonical"
                if r.get("canonical_metric_found") == "True"
                else "raw_or_summary_provisional"
                if r.get("reconstructed_status") == "EVAL_COMPLETE_METRIC_UNRESOLVED"
                else "raw_or_summary_candidate"
            ),
            "formal_best_ready": "True" if r.get("reconstructed_status") in {"COMPLETE_CANONICAL", "COMPLETE_RAW_ONLY", "COMPLETE_SUMMARY_ONLY"} else "False",
        }
        for r in candidate
        if r.get("raw_metric_found") == "True" and r.get("metric_value")
    ]
    fields = [
        "experiment_id",
        "campaign",
        "model",
        "layer",
        "prefix",
        "probe",
        "reconstructed_status",
        "metric_name",
        "metric_value",
        "metric_provenance",
        "formal_best_ready",
        "raw_metric_path",
        "summary_path",
        "canonical_metric_path",
        "reason",
    ]
    write_csv(ROOT / "results/reconstructed/recovered_raw_metrics.csv", rows, fields)


def write_default8_b2_angle_candidate(candidate: list[dict[str, object]]) -> None:
    rows = []
    for row in candidate:
        if row.get("campaign") != "default8" or row.get("probe") != "ego_belief_v2":
            continue
        summary_path = ROOT / str(row.get("summary_path", ""))
        payload = summary_payload(summary_path) if row.get("summary_path") else {}
        rows.append(
            {
                "experiment_id": row.get("experiment_id", ""),
                "model": row.get("model", ""),
                "layer": row.get("layer", ""),
                "prefix": row.get("prefix", ""),
                "mean_ang_err_deg": payload.get("mean_ang_err_deg", row.get("metric_value") if row.get("metric_name") == "mean_ang_err_deg" else ""),
                "mean_az_err_deg": payload.get("mean_az_err_deg", ""),
                "mean_el_err_deg": payload.get("mean_el_err_deg", ""),
                "top1": payload.get("top1", ""),
                "top3": payload.get("top3", ""),
                "mean_log_dist_err": payload.get("mean_log_dist_err", ""),
                "n": payload.get("n", ""),
                "ckpt_path": rel(payload.get("ckpt_path", "")),
                "summary_path": row.get("summary_path", ""),
                "run_artifact_path": row.get("run_artifact_path", ""),
                "status": row.get("reconstructed_status", ""),
                "notes": "candidate angle table recovered from per-run eval summaries; validate split/protocol before promoting to canonical",
            }
        )
    fields = [
        "experiment_id",
        "model",
        "layer",
        "prefix",
        "mean_ang_err_deg",
        "mean_az_err_deg",
        "mean_el_err_deg",
        "top1",
        "top3",
        "mean_log_dist_err",
        "n",
        "ckpt_path",
        "summary_path",
        "run_artifact_path",
        "status",
        "notes",
    ]
    write_csv(ROOT / "results/reconstructed/default8_b2_angle_candidate.csv", rows, fields)


def write_unresolved_reclassified(candidate: list[dict[str, object]]) -> None:
    unresolved_statuses = {
        "EXPECTED_ONLY",
        "JOB_DECLARED",
        "JOB_LAUNCHED",
        "RUN_FOUND",
        "TRAIN_RUNNING",
        "TRAIN_RUNNING_OR_PARTIAL",
        "TRAIN_COMPLETE_EVAL_MISSING",
        "FAILED",
        "UNKNOWN",
    }
    rows = []
    for row in candidate:
        if row.get("reconstructed_status") not in unresolved_statuses:
            continue
        rows.append(
            {
                **row,
                "old_status": "INCOMPLETE",
                "new_status": row.get("reconstructed_status", ""),
                "manifest": row.get("dataset_manifest") or row.get("declared_evidence", ""),
                "run": row.get("run_artifact_path", ""),
                "checkpoint": row.get("checkpoint_path", ""),
                "eval": row.get("summary_path", "") or ("eval_dir_present" if row.get("eval_started") == "True" else ""),
                "raw_metric": row.get("raw_metric_path", "") or row.get("summary_path", ""),
            }
        )
    fields = [
        "experiment_id",
        "campaign",
        "model",
        "layer",
        "prefix",
        "probe",
        "old_status",
        "new_status",
        "reconstructed_status",
        "manifest",
        "run",
        "checkpoint",
        "eval",
        "raw_metric",
        "expected_evidence",
        "declared_evidence",
        "launched_evidence",
        "run_found",
        "train_started",
        "train_completed",
        "checkpoint_found",
        "checkpoint_path",
        "eval_started",
        "eval_completed",
        "raw_metric_found",
        "canonical_metric_found",
        "failure_evidence",
        "archaeology_notes",
        "run_artifact_path",
        "summary_path",
        "job_manifest",
        "rerun_needed",
        "reason",
    ]
    write_csv(HISTORICAL_ARCHIVE / "unresolved_660_reclassified.csv", rows, fields)


def write_supersession_candidates(candidate: list[dict[str, object]]) -> None:
    rows = []
    for row in candidate:
        if row.get("campaign") == "wan_cog_layer_sweep" and row.get("reconstructed_status") == "EVAL_COMPLETE_METRIC_UNRESOLVED":
            rows.append(
                {
                    "campaign": row.get("campaign", ""),
                    "model": row.get("model", ""),
                    "layer": row.get("layer", ""),
                    "prefix": row.get("prefix", ""),
                    "probe": row.get("probe", ""),
                    "old_experiment_id": row.get("experiment_id", ""),
                    "old_artifact": row.get("raw_metric_path", ""),
                    "new_experiment_id": "",
                    "new_artifact": "",
                    "evidence": row.get("evidence", ""),
                    "recommended_status": "NO_NEWER_COMPLETED_RERUN_FOUND__KEEP_METRIC_UNRESOLVED",
                    "reason": row.get("reason", ""),
                }
            )
    fields = [
        "campaign",
        "model",
        "layer",
        "prefix",
        "probe",
        "old_experiment_id",
        "old_artifact",
        "new_experiment_id",
        "new_artifact",
        "evidence",
        "recommended_status",
        "reason",
    ]
    write_csv(ROOT / "experiments/supersession_candidate.csv", rows, fields)


def write_artifact_layout_comparison(candidate: list[dict[str, object]]) -> None:
    examples = [
        (
            "Default B2",
            "default8",
            lambda r: r.get("campaign") == "default8" and r.get("probe") == "ego_belief_v2" and r.get("summary_path"),
            "Has explicit `eval/val_summary.json` and `eval/val_predictions.pt`; angle metrics are recoverable from per-run summaries.",
        ),
        (
            "Wan/Cog Layer Sweep",
            "wan_cog_layer_sweep",
            lambda r: r.get("campaign") == "wan_cog_layer_sweep" and r.get("run_artifact_path") and r.get("train_completed") == "True",
            "Training/checkpoint evidence is present, but historical aggregate rows are provisional unless matched to canonical registry.",
        ),
        (
            "VideoMAE/V-JEPA",
            "videomae_vjepa_layer_sweep",
            lambda r: r.get("campaign") == "videomae_vjepa_layer_sweep" and r.get("train_completed") == "True",
            "Hydra config, Lightning log, checkpoint, and visualizations exist; separate eval summaries are generally missing.",
        ),
        (
            "MiniMax-H3",
            "minimax_h3_layer_benchmark",
            lambda r: r.get("campaign") == "minimax_h3_layer_benchmark" and r.get("run_artifact_path"),
            "Run directories/logs are present, but representative runs show partial lifecycle without completion/checkpoint evidence.",
        ),
        (
            "LingBot Video",
            "lingbot_video_layer_benchmark",
            lambda r: r.get("campaign") == "lingbot_video_layer_benchmark" and r.get("train_completed") == "True",
            "Training-complete runs and checkpoints exist; no canonical or eval-summary metric surface was found in this pass.",
        ),
    ]
    lines = [
        "# Artifact Layout Comparison",
        "",
        "This file compares representative artifact layouts used by the reconstruction script. It intentionally does not promote any candidate into the authoritative registry.",
        "",
        "| area | representative run | config/log/checkpoint | eval/metric surface | interpretation |",
        "| --- | --- | --- | --- | --- |",
    ]
    for label, _campaign, predicate, interpretation in examples:
        row = next((r for r in candidate if predicate(r)), {})
        layout = []
        if row.get("run_artifact_path"):
            layout.append("run_dir")
        if row.get("train_started") == "True":
            layout.append("train_log")
        if row.get("checkpoint_found") == "True":
            layout.append("checkpoint")
        if row.get("summary_path"):
            layout.append("summary")
        if row.get("raw_metric_path"):
            layout.append("aggregate_metric")
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    f"`{row.get('run_artifact_path', '')}`",
                    ", ".join(layout) or "not resolved",
                    f"`{row.get('summary_path') or row.get('raw_metric_path') or ''}`",
                    interpretation,
                ]
            )
            + " |"
        )
    (HISTORICAL_ARCHIVE / "artifact_layout_comparison.md").write_text("\n".join(lines) + "\n")


def write_deep_archaeology(candidate: list[dict[str, object]]) -> None:
    status_counts = Counter(str(row.get("reconstructed_status", "")) for row in candidate)
    campaign_counts = Counter((str(row.get("campaign", "")), str(row.get("reconstructed_status", ""))) for row in candidate)
    unresolved = [r for r in candidate if r.get("reconstructed_status") in {
        "EXPECTED_ONLY",
        "JOB_DECLARED",
        "JOB_LAUNCHED",
        "RUN_FOUND",
        "TRAIN_RUNNING",
        "TRAIN_RUNNING_OR_PARTIAL",
        "TRAIN_COMPLETE_EVAL_MISSING",
        "FAILED",
        "UNKNOWN",
    }]
    default_b2_rows = [r for r in candidate if r.get("campaign") == "default8" and r.get("probe") == "ego_belief_v2"]
    minimax_rows = [r for r in candidate if r.get("campaign") == "minimax_h3_layer_benchmark"]
    minimax_probes = sorted({str(r.get("probe", "")) for r in minimax_rows})
    minimax_layers = sorted({str(r.get("layer", "")) for r in minimax_rows}, key=lambda x: int(x) if x.lstrip("-").isdigit() else 9999)
    minimax_prefixes = sorted({str(r.get("prefix", "")) for r in minimax_rows}, key=lambda x: int(x) if x.isdigit() else 9999)
    lines = [
        "# Deep Artifact Archaeology",
        "",
        "This pass performs lifecycle classification from shallow, fixed artifact locations: job manifests, run directories, Hydra config, Lightning logs, checkpoint folders, eval summaries, and aggregate metric tables.",
        "",
        "## Lifecycle Counts",
        "",
    ]
    for status, count in sorted(status_counts.items(), key=lambda item: STATUS_RANK.get(item[0], 99)):
        lines.append(f"- {status}: {count}")
    lines.extend(["", "## Campaign Counts", ""])
    for (campaign, status), count in sorted(campaign_counts.items()):
        lines.append(f"- {campaign} / {status}: {count}")
    lines.extend(
        [
            "",
            "## Representative Findings",
            "",
            "- VideoMAE/V-JEPA representative runs include `.hydra/config.yaml`, `inscene15k_streaming.log`, checkpoints, and visualizations. Logs show `Trainer.fit` reached max epochs, but no separate eval summary/metric artifact was found in the inspected layout.",
            "- MiniMax-H3 representative runs include configs/logs and early visualizations, but the inspected sample did not show a completion marker or checkpoint, so those rows are treated as partial/running until stronger evidence appears.",
            "- LingBot representative runs include training-complete logs and checkpoints, but no matched eval summary/canonical metric surface was found.",
            "- Wan/Cog representative unresolved runs include training-complete logs and checkpoints. Historical aggregate rows marked provisional are now classified as `EVAL_COMPLETE_METRIC_UNRESOLVED` rather than formal best.",
            "- Default B2 has recoverable per-run `eval/val_summary.json` angle metrics; these are written to `results/reconstructed/default8_b2_angle_candidate.csv` for protocol review.",
            "",
            "## MiniMax-H3 Cardinality",
            "",
            f"- MiniMax-H3 reconstructed rows: {len(minimax_rows)}.",
            f"- Distinct MiniMax-H3 probes: {', '.join(minimax_probes)}.",
            f"- Distinct MiniMax-H3 layers: {', '.join(minimax_layers)}.",
            f"- Distinct MiniMax-H3 prefixes: {', '.join(minimax_prefixes)}.",
            "- The 160-row count comes from the actual MiniMax-H3 job/run evidence universe, not from forcing the full 7-probe matrix. The 140/280 figures belong to campaigns whose design matrix is naturally `model x layer x prefix x 7 probes` or a doubled variant of that matrix. MiniMax-H3 therefore should be reported as its own campaign cardinality until a manifest explicitly declares the missing probe-family combinations.",
            "",
            "## Default B2",
            "",
            f"- Default B2 candidate rows inspected: {len(default_b2_rows)}.",
            f"- Rows with summary path: {sum(1 for r in default_b2_rows if r.get('summary_path'))}.",
            "- These are evaluation artifacts, but they still appear to be validation summaries; they should not be called held-out test results without a frozen split/protocol manifest.",
            "",
            "## Machine-Readable Outputs",
            "",
            "- `archive/consolidation_history/20260817/unresolved_660_reclassified.csv`",
            "- `experiments/supersession_candidate.csv`",
            "- `results/reconstructed/recovered_raw_metrics.csv`",
            "- `results/reconstructed/default8_b2_angle_candidate.csv`",
            "- `archive/consolidation_history/20260817/artifact_layout_comparison.md`",
            "",
            f"Rows in unresolved reclassification output: {len(unresolved)}.",
        ]
    )
    (HISTORICAL_ARCHIVE / "deep_artifact_archaeology.md").write_text("\n".join(lines) + "\n")


def write_action_lists(candidate: list[dict[str, object]]) -> None:
    agg_rows = [
        r for r in candidate
        if r.get("reconstructed_status") in {"COMPLETE_RAW_ONLY", "COMPLETE_SUMMARY_ONLY"}
    ]
    rerun_rows = [
        r for r in candidate
        if str(r.get("rerun_needed", "")) == "True"
    ]
    review_rows = [
        r for r in candidate
        if str(r.get("rerun_needed", "")) == "REVIEW"
    ]
    fields = [
        "experiment_id",
        "campaign",
        "model",
        "layer",
        "prefix",
        "probe",
        "reconstructed_status",
        "raw_metric_path",
        "summary_path",
        "run_artifact_path",
        "job_manifest",
        "train_started",
        "train_completed",
        "checkpoint_found",
        "eval_completed",
        "reason",
    ]
    write_csv(HISTORICAL_ARCHIVE / "aggregation_missing_candidate.csv", agg_rows, fields)
    write_csv(HISTORICAL_ARCHIVE / "rerun_needed_candidate.csv", rerun_rows, fields)
    write_csv(HISTORICAL_ARCHIVE / "rerun_review_candidate.csv", review_rows, fields)


def write_report(expected: list[dict[str, object]], candidate: list[dict[str, object]], reconciliation: list[dict[str, object]]) -> None:
    old_rows = read_csv(OLD_REGISTRY)
    old_loose = {
        (
            {
                "default_streaming": "default8",
                "c_retrain_with_extra": "default8",
                "wan_cog_layer_sweep": "wan_cog_layer_sweep",
            }.get(row.get("model_variant", ""), row.get("model_variant", "")),
            row.get("model", ""),
            row.get("layer", ""),
            row.get("prefix", ""),
            normalize_probe(row.get("probe", "")),
        )
        for row in old_rows
    }
    candidate_loose = {
        (row.get("campaign", ""), row.get("model", ""), row.get("layer", ""), row.get("prefix", ""), normalize_probe(str(row.get("probe", ""))))
        for row in candidate
    }
    status_counts = Counter(str(row.get("reconstructed_status", "")) for row in candidate)
    campaign_counts = Counter((str(row.get("campaign", "")), str(row.get("reconstructed_status", ""))) for row in candidate)
    recon_counts = Counter(str(row.get("reconstructed_status", "")) for row in reconciliation)
    lines = [
        "# Historical Experiment Reconstruction Report",
        "",
        "This is a candidate reconstruction. It does not overwrite `experiments/registry.csv` or canonical result tables.",
        "",
        "## Universe Size",
        "",
        f"- Expected configurations: {len(expected)}",
        f"- Candidate reconstructed rows: {len(candidate)}",
        f"- Existing registry rows: {len(old_rows)}",
        f"- Candidate loose configurations found in old registry: {len(candidate_loose & old_loose)}",
        f"- Candidate loose configurations not represented in old registry: {len(candidate_loose - old_loose)}",
        f"- Old-registry loose configurations not represented in candidate: {len(old_loose - candidate_loose)}",
        "",
        "## Reconstructed Status Counts",
        "",
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(f"- {status}: {count}")
    lines.extend(["", "## Campaign Coverage Counts", ""])
    for (campaign, status), count in sorted(campaign_counts.items()):
        lines.append(f"- {campaign} / {status}: {count}")
    lines.extend(["", "## Reclassification Of Previous INCOMPLETE Rows", ""])
    for status, count in sorted(recon_counts.items()):
        lines.append(f"- {status}: {count}")
    lines.extend(
        [
            "",
            "## Missing Vs Aggregation",
            "",
            "- `COMPLETE_RAW_ONLY` and `COMPLETE_SUMMARY_ONLY` are not rerun requirements by default; they require aggregation/canonicalization or provenance review.",
            "- `TRAIN_COMPLETE_EVAL_MISSING` usually means evaluation/canonicalization is missing, not training.",
            "- `TRAIN_RUNNING_OR_PARTIAL`, `JOB_DECLARED`, and `EXPECTED_ONLY` are rerun candidates only after manual review of the listed evidence.",
            "- `EVAL_COMPLETE_METRIC_UNRESOLVED` means a metric-like value exists but must not be used as formal best until provenance/protocol is resolved.",
            f"- Candidate rows that appear to need aggregation/canonicalization: {sum(1 for r in candidate if r.get('reconstructed_status') in {'COMPLETE_RAW_ONLY', 'COMPLETE_SUMMARY_ONLY'})}",
            f"- Candidate rows currently marked rerun_needed=True: {sum(1 for r in candidate if str(r.get('rerun_needed')) == 'True')}",
            f"- Candidate rows requiring deeper review before any rerun decision: {sum(1 for r in candidate if str(r.get('rerun_needed')) == 'REVIEW')}",
            "",
            "## Direct Evidence Example",
            "",
            "Example query: historical F3R default-layer C3 at prefix 8.",
            "",
        ]
    )
    for row in candidate:
        if row.get("campaign") == "default8" and row.get("model") == "f3r" and row.get("probe") == "counterfactual" and row.get("prefix") == "8":
            lines.extend(
                [
                    f"- experiment_id: `{row.get('experiment_id')}`",
                    f"- config: `{row.get('config')}`",
                    f"- run artifact: `{row.get('run_artifact_path')}`",
                    f"- raw metric: `{row.get('raw_metric_path')}`",
                    f"- summary: `{row.get('summary_path')}`",
                    f"- canonical status: `{row.get('reconstructed_status')}`",
                    f"- metric: `{row.get('metric_name')}={row.get('metric_value')}`",
                    "",
                ]
            )
            break
    lines.extend(
        [
            "## Canonical Completeness",
            "",
            "- Current canonical tables are complete for default A1/A2 and canonicalized C1/C2/C3 rows.",
            "- Current B2 canonical table is incomplete for the default 8-model benchmark angle metric; default B2 has raw/summary evidence but needs angle canonicalization review.",
            "- VideoMAE/V-JEPA and LingBot multi-layer campaigns mostly have training/checkpoint evidence but no matched eval summary/canonical rows in this archaeology pass.",
            "- MiniMax-H3 has job/run evidence but representative artifacts are partial; it should not be merged into the 7-probe matrix without an explicit manifest.",
            "- Wan/Cog provisional rows are separated as metric-unresolved and should not be described as formal best.",
            "",
            "Machine-readable files:",
            "",
            "- `experiments/expected_experiments.csv`",
            "- `experiments/registry_reconstructed_candidate.csv`",
            "- `archive/consolidation_history/20260817/incomplete_reconciliation.csv`",
            "- `archive/consolidation_history/20260817/unresolved_660_reclassified.csv`",
            "- `archive/consolidation_history/20260817/aggregation_missing_candidate.csv`",
            "- `archive/consolidation_history/20260817/rerun_needed_candidate.csv`",
            "- `archive/consolidation_history/20260817/rerun_review_candidate.csv`",
            "- `experiments/supersession_candidate.csv`",
            "- `results/reconstructed/default8_raw_index.csv`",
            "- `results/reconstructed/recovered_raw_metrics.csv`",
            "- `results/reconstructed/default8_b2_angle_candidate.csv`",
        ]
    )
    (HISTORICAL_ARCHIVE / "historical_reconstruction_report.md").write_text("\n".join(lines) + "\n")


def filter_to_frozen_1096(
    expected: list[dict[str, object]],
    candidate: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Keep the authoritative 1096-task universe from the frozen ledger.

    Probe-fill run discovery can surface extra Wan/Cog layer configs that are
    outside the design ledger; those are useful operationally but must not expand
    experiments/task_status_1096.csv.
    """
    frozen_path = ROOT / "experiments/task_status_1096.csv"
    frozen = read_csv(frozen_path)
    if len(frozen) != 1096:
        raise SystemExit(f"frozen ledger has {len(frozen)} rows, expected 1096: {frozen_path}")
    frozen_by_key = {
        (r["campaign"], r["model"], r["layer"], r["prefix"], r["probe_config"]): r for r in frozen
    }
    cand_by_key = {
        (str(r["campaign"]), str(r["model"]), str(r["layer"]), str(r["prefix"]), str(r["probe"])): r
        for r in candidate
    }
    exp_by_key = {
        (str(r["campaign"]), str(r["model"]), str(r["layer"]), str(r["prefix"]), str(r["probe"])): r
        for r in expected
    }
    out_cand: list[dict[str, object]] = []
    out_exp: list[dict[str, object]] = []
    missing = []
    for key, fr in frozen_by_key.items():
        cand = cand_by_key.get(key)
        exp = exp_by_key.get(key)
        if cand is None:
            missing.append(key)
            continue
        cand = dict(cand)
        cand["experiment_id"] = fr["experiment_id"]
        cand["model_variant"] = fr.get("model_variant", cand.get("model_variant", ""))
        out_cand.append(cand)
        if exp is not None:
            exp = dict(exp)
            exp["model_variant"] = fr.get("model_variant", exp.get("model_variant", ""))
            out_exp.append(exp)
        else:
            out_exp.append(
                {
                    "campaign": fr["campaign"],
                    "model": fr["model"],
                    "model_variant": fr.get("model_variant", ""),
                    "layer": fr["layer"],
                    "prefix": fr["prefix"],
                    "probe": fr["probe_config"],
                    "horizon": fr.get("horizon", "all"),
                    "seed": fr.get("seed", "42"),
                    "expected": "True",
                    "expectation_source": "frozen_task_status_1096",
                    "notes": "restored from frozen 1096 ledger",
                }
            )
    if missing:
        raise SystemExit(f"freeze filter missing {len(missing)} keys, e.g. {missing[:3]}")
    if len(out_cand) != 1096 or len(out_exp) != 1096:
        raise SystemExit(f"freeze filter produced cand={len(out_cand)} exp={len(out_exp)}")
    return out_exp, out_cand


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-freeze-1096",
        action="store_true",
        help="Allow expected/candidate to expand beyond the frozen 1096 universe",
    )
    args = parser.parse_args()
    expected = expected_matrix()
    candidate = classify_expected(expected)
    if not args.no_freeze_1096:
        expected, candidate = filter_to_frozen_1096(expected, candidate)
    reconciliation = reconcile_incomplete(candidate)
    write_csv(ROOT / "experiments/expected_experiments.csv", expected, EXPECTED_FIELDS)
    write_csv(ROOT / "experiments/registry_reconstructed_candidate.csv", candidate, CANDIDATE_FIELDS)
    write_csv(
        HISTORICAL_ARCHIVE / "incomplete_reconciliation.csv",
        reconciliation,
        [
            "experiment_id",
            "current_status",
            "reconstructed_status",
            "evidence",
            "replacement_experiment_id",
            "rerun_needed",
            "reason",
            "raw_artifact_path",
            "summary_path",
            "run_artifact_path",
        ],
    )
    write_default_raw_index(candidate)
    write_recovered_raw_metrics(candidate)
    write_default8_b2_angle_candidate(candidate)
    write_unresolved_reclassified(candidate)
    write_supersession_candidates(candidate)
    write_action_lists(candidate)
    HISTORICAL_ARCHIVE.mkdir(parents=True, exist_ok=True)
    write_coverage("default8", HISTORICAL_ARCHIVE / "coverage_default8.md", candidate)
    write_coverage("wan_cog_layer_sweep", HISTORICAL_ARCHIVE / "coverage_wan_cog.md", candidate)
    write_coverage("videomae_vjepa_layer_sweep", HISTORICAL_ARCHIVE / "coverage_videomae_vjepa.md", candidate)
    write_coverage("minimax_h3_layer_benchmark", HISTORICAL_ARCHIVE / "coverage_minimax_h3.md", candidate)
    write_coverage("lingbot_video_layer_benchmark", HISTORICAL_ARCHIVE / "coverage_lingbot_video.md", candidate)
    write_artifact_layout_comparison(candidate)
    write_deep_archaeology(candidate)
    write_report(expected, candidate, reconciliation)
    print(f"expected={len(expected)} candidate={len(candidate)} reconciliation={len(reconciliation)}")


if __name__ == "__main__":
    main()
