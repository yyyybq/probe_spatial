#!/usr/bin/env python3
"""Build the 1096-task lifecycle ledger and derived reporting tables.

This script is intentionally conservative: it reads existing reconstruction,
evaluation, and canonical artifacts, but does not overwrite the authoritative
registry or canonical result CSVs.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ROOT / "experiments/expected_experiments.csv"
RECON = ROOT / "experiments/registry_reconstructed_candidate.csv"
CANONICAL_DIR = ROOT / "results/canonical"
HISTORICAL_ARCHIVE = ROOT / "archive/consolidation_history/20260817"
WORK_QUEUE = ROOT / "experiments/work_queue"
EVAL_CAMPAIGNS = [
    {
        "evaluation_campaign_id": "missing_eval_20260817",
        "campaign_name": "missing_eval_20260817",
        "launcher": "scripts/submit_missing_eval_sco.sh",
        "runner": "scripts/run_missing_eval_shard.py",
        "manifest": "logs/probing_jobs_missing_eval_20260817/jobs/eval_jobs.tsv",
        "out_dir": "logs/probing_jobs_missing_eval_20260817",
        "run_root": "logs/inscene15k_streaming/runs/missing_eval_20260817",
        "log_root": "logs/inscene15k_streaming/eval_jobs",
        "result_file_pattern": "logs/inscene15k_streaming/runs/missing_eval_20260817/*/eval/val_summary.json",
        "notes": "central eval_diag campaign for A2/B2/C-family missing evals",
    },
    {
        "evaluation_campaign_id": "missing_a1_eval_20260817",
        "campaign_name": "missing_a1_eval_20260817",
        "launcher": "scripts/submit_missing_a1_eval_sco.sh",
        "runner": "scripts/run_missing_a1_eval_shard.py",
        "manifest": "logs/probing_jobs_missing_a1_eval_20260817/jobs/eval_jobs.tsv",
        "out_dir": "logs/probing_jobs_missing_a1_eval_20260817",
        "run_root": "logs/inscene15k_streaming/runs/missing_a1_eval_20260817",
        "log_root": "logs/inscene15k_streaming/eval_a1_jobs",
        "result_file_pattern": "logs/inscene15k_streaming/runs/missing_a1_eval_20260817/*/eval/val_summary.json",
        "notes": "central eval_a1_streaming campaign for A1 depth/camera/mask missing evals",
    },
]
PROBES = [
    "streaming_depth",
    "streaming_camera",
    "streaming_mask",
    "view_consistency",
    "ego_belief_v2",
    "ego_belief",
    "action_dynamics",
    "path_integration",
    "counterfactual",
]
MODELS = [
    "qwen2_5_vl",
    "minimax_h3",
    "cogvideox",
    "videomae",
    "vjepa2",
    "lingbot",
    "aether",
    "dino",
    "f3r",
    "wan",
]
SUMMARY_METRIC_ORDER = [
    "mean_loss_depth",
    "Auc_30",
    "mean_loss_identity",
    "balanced_acc",
    "mean_ang_err_deg",
    "global_R@1",
    "intervention_validity",
]


TASK_FIELDS = [
    "experiment_id",
    "campaign",
    "model",
    "model_variant",
    "layer",
    "prefix",
    "probe_config",
    "probe_family",
    "atomic_probe_outputs",
    "horizon",
    "seed",
    "expected_source",
    "train_config",
    "train_manifest",
    "train_job_id",
    "run_artifact_path",
    "checkpoint_path",
    "train_log_path",
    "train_started_at",
    "train_completed_at",
    "train_git_commit",
    "evaluation_campaign_id",
    "eval_config",
    "eval_manifest",
    "eval_job_id",
    "eval_log_path",
    "eval_output_path",
    "raw_metric_path",
    "canonical_metric_path",
    "eval_started_at",
    "eval_completed_at",
    "eval_split",
    "eval_git_commit",
    "training_status",
    "evaluation_status",
    "lifecycle_status",
    "canonical_status",
    "paper_table_eligible",
    "final_claim_ready",
    "status_reason",
    "train_start_evidence",
    "train_completion_evidence",
    "train_failure_evidence",
    "eval_start_evidence",
    "eval_completion_evidence",
    "eval_failure_evidence",
    "metric_evidence",
    "provenance_notes",
]


def read_csv(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
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


def is_true(value: str) -> bool:
    return str(value).lower() == "true"


def parse_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def parse_key(text: str) -> dict[str, str] | None:
    name = Path(text).name
    if name.endswith(".json"):
        name = Path(text).parent.parent.name
    for prefix_token in ["missing_a1_eval_20260817_", "missing_eval_20260817_"]:
        if name.startswith(prefix_token):
            name = name[len(prefix_token) :]
    for probe in sorted(PROBES, key=len, reverse=True):
        marker = probe + "_"
        if not name.startswith(marker):
            continue
        rest = name[len(marker) :]
        match = re.match(r"(?P<model>.+)_p(?P<prefix>\d+)_layer(?P<layer>[-\w.]+)$", rest)
        if not match:
            continue
        model = match.group("model")
        if model not in MODELS:
            continue
        return {
            "probe": normalize_probe(probe),
            "model": model,
            "prefix": match.group("prefix"),
            "layer": match.group("layer"),
        }
    return None


def normalize_probe(probe: str) -> str:
    mapping = {
        "streaming_depth/depth": "streaming_depth",
        "streaming_depth/camera": "streaming_depth",
        "streaming_depth/mask": "streaming_mask",
        "a1_depth": "streaming_depth",
        "a1_camera": "streaming_camera",
        "a1_mask": "streaming_mask",
        "b2": "ego_belief_v2",
        "c1": "action_dynamics",
        "c2": "path_integration",
        "c3": "counterfactual",
        "a2": "view_consistency",
    }
    return mapping.get(probe, probe)


def probe_family(probe: str) -> str:
    return {
        "streaming_depth": "A1",
        "streaming_camera": "A1",
        "streaming_mask": "A1",
        "view_consistency": "A2",
        "ego_belief_v2": "B2",
        "ego_belief": "B1",
        "action_dynamics": "C1",
        "path_integration": "C2",
        "counterfactual": "C3",
    }.get(probe, "UNKNOWN")


def atomic_outputs(probe: str) -> list[str]:
    if probe == "streaming_depth":
        return ["A1-depth", "A1-camera"]
    if probe == "streaming_camera":
        return ["A1-camera"]
    if probe == "streaming_mask":
        return ["A1-mask"]
    return {
        "view_consistency": ["A2"],
        "ego_belief_v2": ["B2"],
        "ego_belief": ["B1"],
        "action_dynamics": ["C1"],
        "path_integration": ["C2"],
        "counterfactual": ["C3"],
    }.get(probe, ["UNKNOWN"])


def primary_metric(probe: str, atomic: str | None = None) -> str:
    if atomic == "A1-depth":
        return "mean_loss_depth"
    if atomic == "A1-camera":
        return "Auc_30"
    if atomic == "A1-mask":
        return "mean_loss_identity"
    return {
        "streaming_depth": "mean_loss_depth",
        "streaming_camera": "Auc_30",
        "streaming_mask": "mean_loss_identity",
        "view_consistency": "balanced_acc",
        "ego_belief_v2": "mean_ang_err_deg",
        "ego_belief": "mean_az_err_deg",
        "action_dynamics": "global_R@1",
        "path_integration": "global_R@1",
        "counterfactual": "intervention_validity",
    }.get(probe, "")


def secondary_metrics(probe: str) -> list[str]:
    return {
        "streaming_depth": ["Auc_30", "Rac_15", "Tac_15", "mean_loss_camera"],
        "streaming_camera": ["mean_loss_camera", "Rac_15", "Tac_15"],
        "streaming_mask": [],
        "view_consistency": ["roc_auc", "mean_overlap_mae", "hard_acc"],
        "ego_belief_v2": ["mean_az_err_deg", "mean_el_err_deg", "top1", "top3", "mean_log_dist_err"],
        "action_dynamics": [
            "global_R@5",
            "no_action_global_R@1",
            "shuffled_action_global_R@1",
            "last_observation_global_R@1",
            "action_gain_cos",
        ],
        "path_integration": ["global_R@5", "final_pose_error", "drift_rate", "loop_closure_error", "pose_error_h1", "pose_error_h2", "pose_error_h4"],
        "counterfactual": ["intervention_margin", "counterfactual_consistency", "global_R@1", "cf_cos_h1", "cf_cos_h2", "cf_cos_h4"],
    }.get(probe, [])


def metric_from_summary(summary: dict[str, Any], probe: str, atomic: str | None = None) -> tuple[str, str]:
    metric = primary_metric(probe, atomic)
    if metric and summary.get(metric) not in (None, ""):
        return metric, str(summary.get(metric))
    mean_key = f"mean_{metric}"
    if metric and summary.get(mean_key) not in (None, ""):
        return metric, str(summary.get(mean_key))
    for key in SUMMARY_METRIC_ORDER:
        if summary.get(key) not in (None, ""):
            return key, str(summary.get(key))
    return "", ""


def timestamp_from_log(log_path: str, token: str) -> str:
    path = ROOT / log_path
    if not log_path or not path.exists():
        return ""
    pattern = re.compile(r"\[(?P<time>\d{4}-\d{2}-\d{2} [^,\]]+)")
    try:
        for line in path.read_text(errors="replace").splitlines():
            if token in line:
                match = pattern.search(line)
                return match.group("time") if match else ""
    except OSError:
        return ""
    return ""


def build_canonical_index() -> tuple[dict[tuple[str, str, str, str], list[dict[str, str]]], list[dict[str, str]]]:
    index: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    rows: list[dict[str, str]] = []
    for path in sorted(CANONICAL_DIR.glob("*.csv")):
        for row in read_csv(path):
            probe = normalize_probe(row.get("probe", "") or path.stem.replace("_results", ""))
            if path.name == "b2_results.csv":
                probe = "ego_belief_v2"
            elif path.name == "c1_results.csv":
                probe = "action_dynamics"
            elif path.name == "c2_results.csv":
                probe = "path_integration"
            elif path.name == "c3_results.csv":
                probe = "counterfactual"
            key = (row.get("model", ""), row.get("layer", ""), row.get("prefix", ""), probe)
            metric_name = row.get("metric_name", "") or primary_metric(probe, canonical_atomic(row, probe))
            metric_value = row.get("metric_value", "") or row.get(metric_name, "")
            if path.name == "b2_results.csv" and not metric_value:
                metric_name = "mean_ang_err_deg"
                metric_value = row.get("angular_error_mean", "")
            enriched = {
                **row,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "canonical_metric_path": rel(path),
                "probe_config": probe,
            }
            index[key].append(enriched)
            rows.append(enriched)
    return index, rows


def iter_status_jsonl(out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((out_dir / "launcher").glob("*status.jsonl")):
        for line in path.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                rows.append({**json.loads(line), "_status_path": rel(path)})
            except json.JSONDecodeError:
                rows.append({"event": "decode_error", "_status_path": rel(path), "raw": line})
    return rows


def build_eval_campaign_index() -> tuple[list[dict[str, Any]], dict[tuple[str, str, str, str], dict[str, Any]], list[dict[str, Any]]]:
    campaign_rows: list[dict[str, Any]] = []
    eval_index: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    recovered_metrics: list[dict[str, Any]] = []
    for spec in EVAL_CAMPAIGNS:
        manifest = ROOT / spec["manifest"]
        out_dir = ROOT / spec["out_dir"]
        jobs = read_csv(manifest, delimiter="\t")
        statuses = iter_status_jsonl(out_dir)
        done_by_key: dict[str, dict[str, Any]] = {}
        for status in statuses:
            if status.get("event") != "job_done":
                continue
            key = str(status.get("key", ""))
            if key not in done_by_key or str(status.get("time", "")) > str(done_by_key[key].get("time", "")):
                done_by_key[key] = status
        shard_starts = [s.get("time", "") for s in statuses if s.get("event") == "shard_start"]
        shard_dones = [s.get("time", "") for s in statuses if s.get("event") == "shard_done"]
        job_ids_path = out_dir / "launcher/sco_job_ids.txt"
        scheduler_ids = ",".join(job_ids_path.read_text(errors="replace").split()) if job_ids_path.exists() else ""
        run_ids = sorted({job.get("run_id", "") for job in jobs if job.get("run_id", "")})
        models = sorted({job.get("vfm", "") for job in jobs if job.get("vfm", "")})
        probes = sorted({job.get("probe", "") for job in jobs if job.get("probe", "")})
        prefixes = sorted({job.get("prefix", "") for job in jobs if job.get("prefix", "")}, key=lambda x: int(x) if x.isdigit() else 9999)
        layers = sorted({job.get("layer", "") for job in jobs if job.get("layer", "")}, key=lambda x: int(x) if x.lstrip("-").isdigit() else 9999)
        completed = failed = running = 0
        for job in jobs:
            key_text = job.get("job_key", "")
            parsed = parse_key(key_text)
            if not parsed:
                continue
            key = (parsed["model"], parsed["layer"], parsed["prefix"], parsed["probe"])
            status = done_by_key.get(key_text, {})
            summary_path = Path(status.get("summary", "")) if status.get("summary") else ROOT / spec["run_root"] / f"{spec['evaluation_campaign_id']}_{key_text}" / "eval/val_summary.json"
            summary = parse_json(summary_path) if summary_path.exists() else {}
            metric_name, metric_value = metric_from_summary(summary, parsed["probe"])
            rc = str(status.get("rc", ""))
            if rc == "0" and metric_value:
                completed += 1
                eval_status = "COMPLETE_RAW"
            elif status:
                failed += 1
                eval_status = "FAILED"
            else:
                running += 1
                eval_status = "RUNNING"
            log_path = status.get("log", "")
            if not log_path:
                log_path = str(ROOT / spec["log_root"] / f"{spec['evaluation_campaign_id']}_*_{key_text}.log")
            eval_row = {
                "evaluation_campaign_id": spec["evaluation_campaign_id"],
                "eval_manifest": rel(manifest),
                "eval_job_id": key_text,
                "eval_log_path": rel(log_path),
                "eval_output_path": rel(summary_path) if summary_path.exists() else "",
                "raw_metric_path": rel(summary_path) if metric_value else "",
                "eval_started_at": status.get("time", ""),
                "eval_completed_at": status.get("time", "") if eval_status == "COMPLETE_RAW" else "",
                "eval_split": "val",
                "eval_status": eval_status,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "summary": summary,
            }
            eval_index[key] = eval_row
            if metric_value:
                recovered_metrics.append(
                    {
                        "experiment_id": "",
                        "campaign": "",
                        "model": parsed["model"],
                        "layer": parsed["layer"],
                        "prefix": parsed["prefix"],
                        "probe_config": parsed["probe"],
                        "atomic_probe": ",".join(atomic_outputs(parsed["probe"])),
                        "evaluation_campaign_id": spec["evaluation_campaign_id"],
                        "eval_job_id": key_text,
                        "metric_name": metric_name,
                        "metric_value": metric_value,
                        "secondary_metrics": json.dumps(
                            {m: summary.get(m) for m in secondary_metrics(parsed["probe"]) if summary.get(m) not in (None, "")},
                            sort_keys=True,
                        ),
                        "eval_split": "val",
                        "checkpoint_path": rel(summary.get("ckpt_path", "")),
                        "raw_metric_path": rel(summary_path),
                        "eval_output_path": rel(summary_path),
                        "status": eval_status,
                        "recovered_note": "metric exists but latest job status was nonzero" if eval_status == "FAILED" else "",
                    }
                )
        current_status = "COMPLETE" if jobs and completed + failed == len(jobs) else "RUNNING" if running else "UNKNOWN"
        campaign_rows.append(
            {
                "evaluation_campaign_id": spec["evaluation_campaign_id"],
                "campaign_name": spec["campaign_name"],
                "launcher": spec["launcher"],
                "manifest": rel(manifest),
                "command": spec["runner"],
                "submitted_at": "",
                "started_at": min(shard_starts) if shard_starts else "",
                "completed_at": max(shard_dones) if shard_dones else "",
                "current_status": current_status,
                "scheduler_job_ids": scheduler_ids,
                "input_run_root": ";".join(run_ids),
                "input_checkpoint_pattern": "ckpt_path column in manifest",
                "output_root": spec["run_root"],
                "result_file_pattern": spec["result_file_pattern"],
                "expected_task_count": len(jobs),
                "matched_task_count": len([j for j in jobs if parse_key(j.get("job_key", ""))]),
                "completed_task_count": completed,
                "failed_task_count": failed,
                "running_task_count": running,
                "notes": spec["notes"] + f"; models={','.join(models)}; probes={','.join(probes)}; prefixes={','.join(prefixes)}; layers={','.join(layers)}",
            }
        )
    return campaign_rows, eval_index, recovered_metrics


def train_status(row: dict[str, str]) -> str:
    recon = row.get("reconstructed_status", "")
    if recon in {"COMPLETE_CANONICAL", "COMPLETE_RAW_ONLY", "COMPLETE_SUMMARY_ONLY", "EVAL_COMPLETE_METRIC_UNRESOLVED", "TRAIN_COMPLETE_EVAL_MISSING"}:
        return "COMPLETE"
    if recon == "TRAIN_RUNNING":
        return "RUNNING"
    if recon == "TRAIN_RUNNING_OR_PARTIAL":
        return "PARTIAL_OR_INTERRUPTED"
    if recon == "JOB_DECLARED":
        return "JOB_DECLARED"
    if recon == "FAILED":
        return "FAILED"
    if recon == "EXPECTED_ONLY":
        return "NOT_STARTED"
    if recon == "SUPERSEDED":
        return "SUPERSEDED"
    return "UNKNOWN"


def lifecycle(training_status: str, evaluation_status: str) -> str:
    if training_status == "COMPLETE" and evaluation_status in {"COMPLETE_RAW", "COMPLETE_CANONICAL", "METRIC_UNRESOLVED"}:
        return "TRAIN_AND_EVAL_COMPLETE"
    if training_status == "COMPLETE" and evaluation_status == "RUNNING":
        return "TRAIN_COMPLETE_EVAL_RUNNING"
    if training_status == "COMPLETE" and evaluation_status == "FAILED":
        return "TRAIN_COMPLETE_EVAL_FAILED"
    if training_status == "COMPLETE":
        return "TRAIN_COMPLETE_EVAL_NOT_STARTED"
    if training_status == "RUNNING":
        return "TRAIN_RUNNING"
    if training_status == "PARTIAL_OR_INTERRUPTED":
        return "TRAIN_PARTIAL_OR_INTERRUPTED"
    if training_status in {"NOT_STARTED", "JOB_DECLARED", "QUEUED"}:
        return "TRAIN_NOT_STARTED"
    if training_status == "FAILED":
        return "TRAIN_FAILED"
    if training_status == "SUPERSEDED" or evaluation_status == "SUPERSEDED":
        return "SUPERSEDED"
    return "UNKNOWN"


def atomic_cell_status(task: dict[str, Any], paper_ready: bool) -> str:
    if paper_ready:
        return "PAPER_READY"
    if task["evaluation_status"] == "COMPLETE_CANONICAL":
        return "CANONICAL"
    if task["evaluation_status"] == "COMPLETE_RAW":
        return "RAW_ONLY"
    if task["evaluation_status"] == "METRIC_UNRESOLVED":
        return "PROVISIONAL"
    if task["evaluation_status"] == "RUNNING":
        return "EVAL_RUNNING"
    if task["training_status"] == "FAILED":
        return "TRAIN_FAILED"
    if task["training_status"] == "PARTIAL_OR_INTERRUPTED":
        return "TRAIN_PARTIAL"
    if task["training_status"] in {"NOT_STARTED", "JOB_DECLARED"}:
        return "NOT_STARTED"
    if task["training_status"] == "COMPLETE":
        return "EVAL_MISSING"
    return "UNKNOWN"


def build_task_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    expected = read_csv(EXPECTED)
    recon = {r["experiment_id"]: r for r in read_csv(RECON)}
    canonical_index, canonical_rows = build_canonical_index()
    eval_campaigns, eval_index, recovered = build_eval_campaign_index()
    rows: list[dict[str, Any]] = []
    for row in read_csv(RECON):
        probe = normalize_probe(row.get("probe", ""))
        key = (row.get("model", ""), row.get("layer", ""), row.get("prefix", ""), probe)
        canonical_matches = canonical_index.get(key, [])
        eval_match = eval_index.get(key, {})
        training = train_status(row)
        canonical_path = row.get("canonical_metric_path", "")
        raw_path = row.get("raw_metric_path", "")
        metric_name = row.get("metric_name", "")
        metric_value = row.get("metric_value", "")
        evaluation_status = "NOT_APPLICABLE_TRAIN_INCOMPLETE"
        evaluation_campaign_id = ""
        eval_manifest = ""
        eval_job_id = ""
        eval_log_path = ""
        eval_output_path = row.get("summary_path", "")
        eval_started_at = ""
        eval_completed_at = ""
        eval_split = "val" if raw_path or canonical_path or eval_output_path else ""
        eval_failure = ""
        if training == "COMPLETE":
            if canonical_path or canonical_matches:
                evaluation_status = "COMPLETE_CANONICAL"
                canonical_path = canonical_path or canonical_matches[0].get("canonical_metric_path", "")
                raw_path = raw_path or canonical_matches[0].get("raw_artifact_path", "")
                metric_name = metric_name or canonical_matches[0].get("metric_name", "")
                metric_value = metric_value or canonical_matches[0].get("metric_value", "")
            elif eval_match:
                evaluation_status = eval_match["eval_status"]
                evaluation_campaign_id = eval_match["evaluation_campaign_id"]
                eval_manifest = eval_match["eval_manifest"]
                eval_job_id = eval_match["eval_job_id"]
                eval_log_path = eval_match["eval_log_path"]
                eval_output_path = eval_match["eval_output_path"]
                raw_path = eval_match["raw_metric_path"]
                metric_name = eval_match["metric_name"]
                metric_value = eval_match["metric_value"]
                eval_started_at = eval_match["eval_started_at"]
                eval_completed_at = eval_match["eval_completed_at"]
                eval_split = eval_match["eval_split"]
                if evaluation_status == "FAILED":
                    eval_failure = "central evaluation job returned non-zero rc"
            elif row.get("reconstructed_status") == "EVAL_COMPLETE_METRIC_UNRESOLVED":
                evaluation_status = "METRIC_UNRESOLVED"
            elif row.get("raw_metric_found") == "True" and (raw_path or row.get("summary_path")):
                evaluation_status = "COMPLETE_RAW"
                raw_path = raw_path or row.get("summary_path", "")
            else:
                evaluation_status = "NOT_STARTED"
        paper_eligible = (
            training == "COMPLETE"
            and evaluation_status == "COMPLETE_CANONICAL"
            and bool(raw_path)
            and bool(canonical_path)
            and bool(row.get("checkpoint_path"))
        )
        final_claim = "POLICY_NOT_FROZEN" if paper_eligible else "NO"
        life = lifecycle(training, evaluation_status)
        status_reason = row.get("reason", "")
        if eval_match and evaluation_status == "COMPLETE_RAW":
            status_reason = "central evaluation campaign produced parseable raw metric; candidate canonicalization still pending"
        elif paper_eligible:
            status_reason = "canonical metric exists with checkpoint/config provenance; final claim gate remains policy/seed dependent"
        train_log = f"{row.get('run_artifact_path')}/inscene15k_streaming.log" if row.get("run_artifact_path") else ""
        task = {
            "experiment_id": row["experiment_id"],
            "campaign": row.get("campaign", ""),
            "model": row.get("model", ""),
            "model_variant": row.get("model_variant", ""),
            "layer": row.get("layer", ""),
            "prefix": row.get("prefix", ""),
            "probe_config": probe,
            "probe_family": probe_family(probe),
            "atomic_probe_outputs": ",".join(atomic_outputs(probe)),
            "horizon": row.get("horizon", "all"),
            "seed": row.get("seed", "42"),
            "expected_source": row.get("expectation_source", ""),
            "train_config": row.get("config", ""),
            "train_manifest": row.get("dataset_manifest", "") or row.get("job_manifest", ""),
            "train_job_id": row.get("launcher_job_id", ""),
            "run_artifact_path": row.get("run_artifact_path", ""),
            "checkpoint_path": row.get("checkpoint_path", ""),
            "train_log_path": train_log,
            "train_started_at": timestamp_from_log(train_log, "Starting training"),
            "train_completed_at": timestamp_from_log(train_log, "`Trainer.fit` stopped"),
            "train_git_commit": "",
            "evaluation_campaign_id": evaluation_campaign_id,
            "eval_config": "",
            "eval_manifest": eval_manifest,
            "eval_job_id": eval_job_id,
            "eval_log_path": eval_log_path,
            "eval_output_path": eval_output_path,
            "raw_metric_path": raw_path,
            "canonical_metric_path": canonical_path,
            "eval_started_at": eval_started_at,
            "eval_completed_at": eval_completed_at,
            "eval_split": eval_split,
            "eval_git_commit": "",
            "training_status": training,
            "evaluation_status": evaluation_status,
            "lifecycle_status": life,
            "canonical_status": "COMPLETE" if evaluation_status == "COMPLETE_CANONICAL" else "NOT_CANONICAL",
            "paper_table_eligible": "YES" if paper_eligible else "NO",
            "final_claim_ready": final_claim,
            "status_reason": status_reason,
            "train_start_evidence": "train log start marker" if row.get("train_started") == "True" else "",
            "train_completion_evidence": "Trainer.fit normal stop + checkpoint" if training == "COMPLETE" else "",
            "train_failure_evidence": row.get("failure_evidence", "") if training == "FAILED" else "",
            "eval_start_evidence": eval_manifest or ("canonical/raw artifact" if evaluation_status in {"COMPLETE_RAW", "COMPLETE_CANONICAL", "METRIC_UNRESOLVED"} else ""),
            "eval_completion_evidence": eval_output_path or canonical_path if evaluation_status in {"COMPLETE_RAW", "COMPLETE_CANONICAL", "METRIC_UNRESOLVED"} else "",
            "eval_failure_evidence": eval_failure,
            "metric_evidence": metric_name + "=" + metric_value if metric_name and metric_value else "",
            "provenance_notes": row.get("archaeology_notes", ""),
        }
        rows.append(task)
        if eval_match:
            for rec in recovered:
                if rec["model"] == key[0] and rec["layer"] == key[1] and rec["prefix"] == key[2] and rec["probe_config"] == key[3]:
                    rec["experiment_id"] = row["experiment_id"]
                    rec["campaign"] = row.get("campaign", "")
    matched_by_campaign = Counter(t["evaluation_campaign_id"] for t in rows if t["evaluation_campaign_id"])
    for campaign in eval_campaigns:
        campaign["matched_task_count"] = matched_by_campaign.get(campaign["evaluation_campaign_id"], 0)
    if len(rows) != len(expected):
        raise SystemExit(f"expected {len(expected)} task rows, got {len(rows)}")
    if len({r["experiment_id"] for r in rows}) != len(rows):
        raise SystemExit("task experiment_id values are not unique")
    return rows, eval_campaigns, recovered, canonical_rows


def write_taxonomy() -> None:
    rows = []
    for probe in PROBES:
        if probe == "ego_belief":
            continue
        for atomic in atomic_outputs(probe):
            rows.append(
                {
                    "probe_config": probe,
                    "probe_family": probe_family(probe),
                    "atomic_probe": atomic,
                    "training_unit": "task_config",
                    "evaluation_unit": "summary_metric",
                    "primary_metric": primary_metric(probe, atomic),
                    "secondary_metrics": ",".join(secondary_metrics(probe)),
                    "horizon_expansion": "pose_error_h* rows" if probe == "path_integration" else "cf_cos_h* rows" if probe == "counterfactual" else "none",
                    "notes": "streaming_depth emits depth and camera atomic outputs" if probe == "streaming_depth" else "",
                }
            )
    write_csv(ROOT / "metadata/probe_taxonomy.csv", rows, ["probe_config", "probe_family", "atomic_probe", "training_unit", "evaluation_unit", "primary_metric", "secondary_metrics", "horizon_expansion", "notes"])


def write_atomic_coverage(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for task in tasks:
        for atomic in task["atomic_probe_outputs"].split(","):
            if not atomic:
                continue
            paper_ready = task["paper_table_eligible"] == "YES"
            rows.append(
                {
                    "experiment_id": task["experiment_id"],
                    "campaign": task["campaign"],
                    "model": task["model"],
                    "model_variant": task["model_variant"],
                    "layer": task["layer"],
                    "prefix": task["prefix"],
                    "probe_config": task["probe_config"],
                    "probe_family": task["probe_family"],
                    "atomic_probe": atomic,
                    "training_status": task["training_status"],
                    "evaluation_status": task["evaluation_status"],
                    "coverage_status": atomic_cell_status(task, paper_ready),
                    "metric_name": primary_metric(task["probe_config"], atomic),
                    "raw_metric_path": task["raw_metric_path"],
                    "canonical_metric_path": task["canonical_metric_path"],
                    "paper_table_eligible": task["paper_table_eligible"],
                    "final_claim_ready": task["final_claim_ready"],
                    "reason": task["status_reason"],
                }
            )
    write_csv(
        ROOT / "results/atomic_probe_coverage.csv",
        rows,
        [
            "experiment_id",
            "campaign",
            "model",
            "model_variant",
            "layer",
            "prefix",
            "probe_config",
            "probe_family",
            "atomic_probe",
            "training_status",
            "evaluation_status",
            "coverage_status",
            "metric_name",
            "raw_metric_path",
            "canonical_metric_path",
            "paper_table_eligible",
            "final_claim_ready",
            "reason",
        ],
    )
    return rows


def write_paper_ready(tasks: list[dict[str, Any]], canonical_rows: list[dict[str, str]]) -> None:
    task_by_key = {(t["model"], t["layer"], t["prefix"], t["probe_config"]): t for t in tasks}
    rows = []
    for row in canonical_rows:
        probe = row["probe_config"]
        key = (row.get("model", ""), row.get("layer", ""), row.get("prefix", ""), probe)
        task = task_by_key.get(key)
        if not task or task["paper_table_eligible"] != "YES":
            continue
        atomic = canonical_atomic(row, probe)
        secondary = {m: row.get(m) for m in secondary_metrics(probe) if row.get(m)}
        rows.append(
            {
                "experiment_id": task["experiment_id"],
                "campaign": task["campaign"],
                "model": task["model"],
                "model_variant": task["model_variant"],
                "layer": task["layer"],
                "prefix": task["prefix"],
                "atomic_probe": atomic,
                "horizon": row.get("horizon", task["horizon"]),
                "seed": row.get("seed", task["seed"]),
                "metric_name": row.get("metric_name", "") or primary_metric(probe, atomic),
                "metric_value": row.get("metric_value", "") or row.get(primary_metric(probe, atomic), ""),
                "secondary_metrics": json.dumps(secondary, sort_keys=True),
                "eval_split": row.get("eval_split", task["eval_split"]),
                "protocol_version": "candidate_v1_val_protocol",
                "checkpoint_path": task["checkpoint_path"],
                "raw_metric_path": row.get("raw_artifact_path", task["raw_metric_path"]),
                "canonical_metric_path": row.get("canonical_metric_path", task["canonical_metric_path"]),
                "selection_policy": "FULL_SWEEP",
                "paper_table_eligible": "YES",
                "final_claim_ready": task["final_claim_ready"],
                "readiness_note": "paper-table eligible; headline claims still require frozen protocol/seed/control gate",
            }
        )
    fields = [
        "experiment_id",
        "campaign",
        "model",
        "model_variant",
        "layer",
        "prefix",
        "atomic_probe",
        "horizon",
        "seed",
        "metric_name",
        "metric_value",
        "secondary_metrics",
        "eval_split",
        "protocol_version",
        "checkpoint_path",
        "raw_metric_path",
        "canonical_metric_path",
        "selection_policy",
        "paper_table_eligible",
        "final_claim_ready",
        "readiness_note",
    ]
    write_csv(ROOT / "results/paper_ready_results.csv", rows, fields)
    write_csv(ROOT / "results/reconstructed/paper_ready_results_candidate.csv", rows, fields)
    lines = [
        "# Paper-Ready Results",
        "",
        f"Rows: {len(rows)}",
        "",
        "These rows are canonicalized and provenance-linked. They are not best-layer selections; `selection_policy=FULL_SWEEP` keeps all completed layer/prefix rows.",
        "",
        "All rows are marked `paper_table_eligible=YES`; `final_claim_ready=POLICY_NOT_FROZEN` until the frozen protocol, seed policy, and control gate are reviewed.",
        "",
        "| campaign | rows |",
        "| --- | ---: |",
    ]
    for campaign, count in sorted(Counter(r["campaign"] for r in rows).items()):
        lines.append(f"| {campaign} | {count} |")
    lines.extend(["", "| atomic_probe | rows |", "| --- | ---: |"])
    for probe, count in sorted(Counter(r["atomic_probe"] for r in rows).items()):
        lines.append(f"| {probe} | {count} |")
    lines.extend(["", "## Canonical Sources", "", "| canonical_metric_path | rows |", "| --- | ---: |"])
    for path, count in sorted(Counter(r["canonical_metric_path"] for r in rows).items()):
        lines.append(f"| `{path}` | {count} |")
    (ROOT / "docs/paper_ready_results.md").write_text("\n".join(lines) + "\n")


def canonical_atomic(row: dict[str, str], probe: str) -> str:
    if probe == "streaming_depth":
        probe_text = row.get("probe", "")
        if "camera" in probe_text:
            return "A1-camera"
        if "mask" in probe_text:
            return "A1-mask"
        return "A1-depth"
    return atomic_outputs(probe)[0]


def write_recovered_metrics(recovered: list[dict[str, Any]]) -> None:
    fields = [
        "experiment_id",
        "campaign",
        "model",
        "layer",
        "prefix",
        "probe_config",
        "atomic_probe",
        "evaluation_campaign_id",
        "eval_job_id",
        "metric_name",
        "metric_value",
        "secondary_metrics",
        "eval_split",
        "checkpoint_path",
        "raw_metric_path",
        "eval_output_path",
        "status",
        "recovered_note",
    ]
    write_csv(ROOT / "results/reconstructed/evaluation_recovered_metrics.csv", recovered, fields)


def write_eval_missing_reconciliation(tasks: list[dict[str, Any]]) -> None:
    old_rows = read_csv(HISTORICAL_ARCHIVE / "unresolved_660_reclassified.csv")
    late_campaigns = {
        "videomae_vjepa_layer_sweep",
        "minimax_h3_layer_benchmark",
        "lingbot_video_layer_benchmark",
    }
    ids = {
        r["experiment_id"]
        for r in old_rows
        if r.get("new_status") == "TRAIN_COMPLETE_EVAL_MISSING" and r.get("campaign") in late_campaigns
    }
    task_by_id = {t["experiment_id"]: t for t in tasks}
    rows = []
    for exp_id in sorted(ids):
        task = task_by_id[exp_id]
        rows.append(
            {
                "experiment_id": exp_id,
                "old_evaluation_status": "TRAIN_COMPLETE_EVAL_MISSING",
                "matched_evaluation_campaign": task["evaluation_campaign_id"],
                "eval_job_id": task["eval_job_id"],
                "eval_output_path": task["eval_output_path"],
                "raw_metric_path": task["raw_metric_path"],
                "new_evaluation_status": task["evaluation_status"],
                "paper_table_eligible": task["paper_table_eligible"],
                "reason": task["status_reason"],
            }
        )
    write_csv(
        HISTORICAL_ARCHIVE / "eval_missing_372_reconciliation.csv",
        rows,
        [
            "experiment_id",
            "old_evaluation_status",
            "matched_evaluation_campaign",
            "eval_job_id",
            "eval_output_path",
            "raw_metric_path",
            "new_evaluation_status",
            "paper_table_eligible",
            "reason",
        ],
    )


def write_review_lists(tasks: list[dict[str, Any]]) -> None:
    WORK_QUEUE.mkdir(parents=True, exist_ok=True)
    failed = [t for t in tasks if t["training_status"] == "FAILED"]
    partial = [t for t in tasks if t["training_status"] == "PARTIAL_OR_INTERRUPTED"]
    declared = [t for t in tasks if t["training_status"] in {"JOB_DECLARED", "NOT_STARTED"}]
    eval_only = [t for t in tasks if t["training_status"] == "COMPLETE" and t["evaluation_status"] in {"NOT_STARTED", "FAILED", "RUNNING"}]
    fail_rows = [
        {
            **t,
            "failure_type": "traceback/runtime" if t["train_failure_evidence"] else "unknown",
            "last_completed_step": "",
            "checkpoint_available": "YES" if t["checkpoint_path"] else "NO",
            "resumable": "UNKNOWN" if t["checkpoint_path"] else "NO",
            "full_rerun_needed": "REVIEW",
            "evidence": t["train_failure_evidence"],
        }
        for t in failed
    ]
    partial_rows = [
        {
            **t,
            "partial_type": "usable_checkpoint_but_protocol_incomplete" if t["checkpoint_path"] else "unknown_or_interrupted",
            "resumable": "YES" if t["checkpoint_path"] else "UNKNOWN",
            "usable_checkpoint_but_protocol_incomplete": "YES" if t["checkpoint_path"] else "UNKNOWN",
            "must_restart": "UNKNOWN",
        }
        for t in partial
    ]
    declared_rows = [
        {
            **t,
            "not_started_type": "manifest_only" if t["train_manifest"] else "expected_only",
            "queued_but_never_launched": "UNKNOWN",
            "launch_evidence_missing": "YES",
            "genuinely_not_run": "UNKNOWN",
        }
        for t in declared
    ]
    write_csv(WORK_QUEUE / "training_failed_review.csv", fail_rows, TASK_FIELDS + ["failure_type", "last_completed_step", "checkpoint_available", "resumable", "full_rerun_needed", "evidence"])
    write_csv(WORK_QUEUE / "training_partial_review.csv", partial_rows, TASK_FIELDS + ["partial_type", "resumable", "usable_checkpoint_but_protocol_incomplete", "must_restart"])
    write_csv(WORK_QUEUE / "not_started_or_declared_review.csv", declared_rows, TASK_FIELDS + ["not_started_type", "queued_but_never_launched", "launch_evidence_missing", "genuinely_not_run"])
    write_csv(WORK_QUEUE / "evaluation_only_needed.csv", eval_only, TASK_FIELDS)


def write_summary(tasks: list[dict[str, Any]], atomic_rows: list[dict[str, Any]], eval_campaigns: list[dict[str, Any]]) -> None:
    high_statuses = [
        "TRAIN_AND_EVAL_COMPLETE",
        "TRAIN_COMPLETE_EVAL_RUNNING",
        "TRAIN_COMPLETE_EVAL_NOT_STARTED",
        "TRAIN_COMPLETE_EVAL_FAILED",
        "TRAIN_RUNNING",
        "TRAIN_PARTIAL_OR_INTERRUPTED",
        "TRAIN_NOT_STARTED",
        "TRAIN_FAILED",
        "SUPERSEDED",
        "UNKNOWN",
    ]
    rows: list[dict[str, Any]] = []

    def add_summary(dimension: str, key: str, subset: list[dict[str, Any]]) -> None:
        counts = Counter(t["lifecycle_status"] for t in subset)
        row = {"dimension": dimension, "key": key, "total": len(subset)}
        for status in high_statuses:
            row[status] = counts.get(status, 0)
        row["PAPER_TABLE_ELIGIBLE"] = sum(1 for t in subset if t.get("paper_table_eligible") == "YES")
        row["FINAL_CLAIM_READY"] = sum(1 for t in subset if t.get("final_claim_ready") == "YES")
        rows.append(row)

    add_summary("overall", "all", tasks)
    for field in ["campaign", "model", "probe_family", "layer", "prefix"]:
        for key in sorted({t[field] for t in tasks}):
            add_summary(field, key, [t for t in tasks if t[field] == key])
    for atomic in sorted({r["atomic_probe"] for r in atomic_rows}):
        exp_ids = {r["experiment_id"] for r in atomic_rows if r["atomic_probe"] == atomic}
        add_summary("atomic_probe", atomic, [t for t in tasks if t["experiment_id"] in exp_ids])
    fields = ["dimension", "key", "total", *high_statuses, "PAPER_TABLE_ELIGIBLE", "FINAL_CLAIM_READY"]
    write_csv(ROOT / "experiments/task_status_summary.csv", rows, fields)
    overall = Counter(t["lifecycle_status"] for t in tasks)
    lines = ["# Task Status Dashboard", "", "## Overall Status", "", "| lifecycle status | count | percentage |", "| --- | ---: | ---: |"]
    total = len(tasks)
    for status in high_statuses:
        count = overall.get(status, 0)
        if count:
            lines.append(f"| {status} | {count} | {count / total:.1%} |")
    lines.extend(["", "## Campaign x Lifecycle Status", "", "| campaign | " + " | ".join(high_statuses) + " |", "| --- | " + " | ".join(["---:"] * len(high_statuses)) + " |"])
    for campaign in sorted({t["campaign"] for t in tasks}):
        counts = Counter(t["lifecycle_status"] for t in tasks if t["campaign"] == campaign)
        lines.append("| " + campaign + " | " + " | ".join(str(counts.get(s, 0)) for s in high_statuses) + " |")
    lines.extend(["", "## Model x Atomic Probe Coverage", "", "| model | atomic_probe | PAPER_READY | CANONICAL | RAW_ONLY | PROVISIONAL | EVAL_MISSING | TRAIN_PARTIAL | TRAIN_FAILED | NOT_STARTED |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for model in sorted({r["model"] for r in atomic_rows}):
        for atomic in sorted({r["atomic_probe"] for r in atomic_rows if r["model"] == model}):
            counts = Counter(r["coverage_status"] for r in atomic_rows if r["model"] == model and r["atomic_probe"] == atomic)
            lines.append(f"| {model} | {atomic} | {counts.get('PAPER_READY',0)} | {counts.get('CANONICAL',0)} | {counts.get('RAW_ONLY',0)} | {counts.get('PROVISIONAL',0)} | {counts.get('EVAL_MISSING',0)} | {counts.get('TRAIN_PARTIAL',0)} | {counts.get('TRAIN_FAILED',0)} | {counts.get('NOT_STARTED',0)} |")
    lines.extend(["", "## Evaluation Campaign Recovery", "", "| campaign | expected | completed | running | failed | output root |", "| --- | ---: | ---: | ---: | ---: | --- |"])
    for row in eval_campaigns:
        lines.append(f"| {row['evaluation_campaign_id']} | {row['expected_task_count']} | {row['completed_task_count']} | {row['running_task_count']} | {row['failed_task_count']} | `{row['output_root']}` |")
    lines.extend(["", "## Paper-Ready Result Count", "", f"- paper_table_eligible tasks: {sum(1 for t in tasks if t['paper_table_eligible'] == 'YES')}", f"- final_claim_ready tasks: {sum(1 for t in tasks if t['final_claim_ready'] == 'YES')}", "", "## Remaining Actions", "", f"- evaluation-only tasks: {sum(1 for t in tasks if t['training_status'] == 'COMPLETE' and t['evaluation_status'] in {'NOT_STARTED','FAILED','RUNNING'})}", f"- failed training tasks: {sum(1 for t in tasks if t['training_status'] == 'FAILED')}", f"- partial/interrupted training tasks: {sum(1 for t in tasks if t['training_status'] == 'PARTIAL_OR_INTERRUPTED')}", f"- job declared/not started tasks: {sum(1 for t in tasks if t['training_status'] in {'JOB_DECLARED','NOT_STARTED'})}", ""])
    (ROOT / "docs/task_status_dashboard.md").write_text("\n".join(lines))


def write_atomic_doc(atomic_rows: list[dict[str, Any]]) -> None:
    lines = ["# Atomic Probe Coverage", "", "Status is per atomic probe output, not per historical task.", ""]
    for campaign in sorted({r["campaign"] for r in atomic_rows}):
        lines.extend([f"## {campaign}", "", "| model | layer | prefix | atomic_probe | status |", "| --- | ---: | ---: | --- | --- |"])
        for row in sorted([r for r in atomic_rows if r["campaign"] == campaign], key=lambda r: (r["model"], int(r["layer"]) if str(r["layer"]).lstrip("-").isdigit() else 9999, int(r["prefix"]) if str(r["prefix"]).isdigit() else 9999, r["atomic_probe"])):
            lines.append(f"| {row['model']} | {row['layer']} | {row['prefix']} | {row['atomic_probe']} | {row['coverage_status']} |")
        lines.append("")
    (ROOT / "docs/atomic_probe_coverage.md").write_text("\n".join(lines))


def main() -> None:
    tasks, eval_campaigns, recovered, canonical_rows = build_task_rows()
    write_csv(ROOT / "experiments/task_status_1096.csv", tasks, TASK_FIELDS)
    write_csv(
        ROOT / "experiments/evaluation_campaign_index.csv",
        eval_campaigns,
        [
            "evaluation_campaign_id",
            "campaign_name",
            "launcher",
            "manifest",
            "command",
            "submitted_at",
            "started_at",
            "completed_at",
            "current_status",
            "scheduler_job_ids",
            "input_run_root",
            "input_checkpoint_pattern",
            "output_root",
            "result_file_pattern",
            "expected_task_count",
            "matched_task_count",
            "completed_task_count",
            "failed_task_count",
            "running_task_count",
            "notes",
        ],
    )
    write_taxonomy()
    atomic_rows = write_atomic_coverage(tasks)
    write_recovered_metrics(recovered)
    write_paper_ready(tasks, canonical_rows)
    write_eval_missing_reconciliation(tasks)
    write_review_lists(tasks)
    write_summary(tasks, atomic_rows, eval_campaigns)
    write_atomic_doc(atomic_rows)
    print(f"tasks={len(tasks)} atomic_rows={len(atomic_rows)} recovered_eval_metrics={len(recovered)}")


if __name__ == "__main__":
    main()
