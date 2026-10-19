#!/usr/bin/env python3
"""Prepare, run, and finalize the V1 C3-from-C1 evaluation-only sweep.

The frozen h1/h2/h4 endpoints are a same-trajectory action-sensitivity
diagnostic.  They are deliberately not described as causal counterfactuals.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
PREFIXES = (8, 12, 16, 24)
HORIZONS = (1, 2, 4)
DATA_ROOT = "fj:s3://baiqiao/InsScene-15K/data/processed_scannetpp_v2/processed_scannetpp_v2"
STREAM_ROOT = "fj:s3://baiqiao/InsScene-15K/FEAT_STREAMING_BY_PREFIX"
TARGET_ROOT = "fj:s3://baiqiao/InsScene-15K/FEAT_TARGET_BY_PREFIX"
STORAGE_PARENT = "fj:s3://baiqiao/InsScene-15K/"
PYTHON_DEFAULT = "/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python"
ANALYSIS_METADATA = ROOT / "results/analysis/q00_1096_paper_no_c3_palette_20260923T000000Z/data/analysis_metadata.json"
REGISTRY = ROOT / "experiments/registry.csv"
BASE_MANIFEST = ROOT / "manifests/c_retrain_20260722"

MODEL_ORDER = (
    "dino",
    "f3r",
    "vjepa2",
    "videomae",
    "qwen2_5_vl",
    "wan",
    "cogvideox",
    "aether",
    "minimax_h3",
    "lingbot",
)
MODEL_DISPLAY = {
    "dino": "DINO",
    "f3r": "F3R",
    "vjepa2": "V-JEPA2",
    "videomae": "VideoMAE",
    "qwen2_5_vl": "Qwen2.5-VL",
    "wan": "Wan",
    "cogvideox": "CogVideoX",
    "aether": "Aether",
    "minimax_h3": "MiniMax-Hailuo-02",
    "lingbot": "LingBot-video",
}
EXPERIMENT = {
    model: f"inscene15k_streaming/action_dynamics_{model}_h124_v1"
    for model in MODEL_ORDER
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_action_set_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(record.get("sample_id")),
        str(record.get("split")),
        int(record.get("streaming_window_id", -1)),
        int(record.get("streaming_tail_raw", -1)),
        tuple(int(value) for value in record.get("streaming_indices", [])),
        tuple(int(value) for value in record.get("streaming_window_indices", [])),
    )


def freeze_manifests(output: Path) -> dict[int, dict[str, Any]]:
    frozen_dir = output / "action_sets"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[int, dict[str, Any]] = {}
    for prefix in PREFIXES:
        source = BASE_MANIFEST / f"inscene_c_h124_prefix{prefix}.jsonl"
        if not source.is_file():
            raise FileNotFoundError(f"missing h124 manifest: {source}")
        records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        val_records = [record for record in records if record.get("split") == "val"]
        if len(val_records) != 276:
            raise ValueError(f"prefix {prefix}: expected 276 val action sets, found {len(val_records)}")
        sample_ids = [str(record.get("sample_id", "")) for record in val_records]
        if any(not value for value in sample_ids) or len(sample_ids) != len(set(sample_ids)):
            raise ValueError(f"prefix {prefix}: empty or duplicate action-set sample IDs")
        destination = frozen_dir / f"c3_action_sensitivity_p{prefix}_h124_val.jsonl"
        with destination.open("w") as handle:
            for record in val_records:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        semantic_payload = json.dumps(
            [semantic_action_set_key(record) for record in val_records],
            separators=(",", ":"),
        ).encode()
        metadata[prefix] = {
            "prefix": prefix,
            "horizons": list(HORIZONS),
            "source_manifest": str(source.relative_to(ROOT)),
            "source_manifest_sha256": sha256_path(source),
            "frozen_manifest": str(destination.resolve()),
            "frozen_manifest_sha256": sha256_path(destination),
            "semantic_lineage_sha256": hashlib.sha256(semantic_payload).hexdigest(),
            "n_action_sets": len(val_records),
            "candidate_count_per_set": len(HORIZONS),
            "candidate_ids": list(HORIZONS),
            "interpretation": "same_trajectory_endpoint_action_sensitivity_not_causal_counterfactual",
        }
    return metadata


def registry_rows() -> list[dict[str, str]]:
    with REGISTRY.open(newline="") as handle:
        return list(csv.DictReader(handle))


def select_checkpoint(
    rows: list[dict[str, str]], model: str, layer: int, prefix: int
) -> tuple[dict[str, str] | None, str]:
    candidates = [
        row
        for row in rows
        if row.get("probe") == "action_dynamics"
        and row.get("model") == model
        and row.get("layer") == str(layer)
        and row.get("prefix") == str(prefix)
        and row.get("status") == "COMPLETE"
        and row.get("checkpoint")
    ]
    existing = [row for row in candidates if (ROOT / row["checkpoint"]).is_file()]
    if len(existing) == 1:
        return existing[0], ""
    if len(existing) > 1:
        raise ValueError(
            f"ambiguous C1 checkpoint for model={model}, layer={layer}, prefix={prefix}: "
            f"{[row['experiment_id'] for row in existing]}"
        )
    if candidates:
        return None, "registry checkpoint path is not present on disk"
    return None, "no COMPLETE V1 C1 checkpoint in registry"


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    representative = {
        key: int(value)
        for key, value in json.loads(ANALYSIS_METADATA.read_text())["representative_layers"].items()
    }
    if set(representative) != set(MODEL_ORDER):
        raise ValueError("representative-layer inventory differs from the ten-model paper inventory")

    frozen = freeze_manifests(output)
    rows = registry_rows()
    jobs: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        layer = representative[model]
        for prefix in PREFIXES:
            selected, reason = select_checkpoint(rows, model, layer, prefix)
            checkpoint = (ROOT / selected["checkpoint"]).resolve() if selected else None
            job_key = f"{model}_l{layer}_p{prefix}"
            jobs.append({
                "job_key": job_key,
                "model": model,
                "model_display": MODEL_DISPLAY[model],
                "representative_layer": layer,
                "prefix": prefix,
                "candidate_horizons": ";".join(map(str, HORIZONS)),
                "experiment": EXPERIMENT[model],
                "source_experiment_id": selected["experiment_id"] if selected else "",
                "checkpoint": str(checkpoint) if checkpoint else "",
                "checkpoint_bytes": checkpoint.stat().st_size if checkpoint else "",
                "checkpoint_sha256": sha256_path(checkpoint) if checkpoint and args.hash_checkpoints else "NOT_COMPUTED",
                "frozen_manifest": frozen[prefix]["frozen_manifest"],
                "frozen_manifest_sha256": frozen[prefix]["frozen_manifest_sha256"],
                "semantic_lineage_sha256": frozen[prefix]["semantic_lineage_sha256"],
                "n_action_sets_expected": frozen[prefix]["n_action_sets"],
                "streaming_feat_root": f"{STREAM_ROOT}/{model}_prefix_{prefix}_m0p35_r0p5",
                "target_feat_root": f"{TARGET_ROOT}/{model}_prefix_{prefix}_m0p35_r0p5",
                "input_status": "UNVERIFIED" if selected else "NOT_APPLICABLE",
                "input_reason": "" if selected else "C1 checkpoint unavailable",
                "output_dir": str((output / "cells" / job_key).resolve()),
                "status": "READY" if selected else "UNAVAILABLE",
                "reason": reason,
            })

    fields = list(jobs[0])
    write_csv(output / "sweep_manifest.csv", jobs, fields)
    write_csv(output / "checkpoint_selection.csv", jobs, fields)
    ready = sum(job["status"] == "READY" for job in jobs)
    action_set_metadata = {
        "protocol": "c3_action_sensitivity_from_frozen_c1_v1",
        "created_at_utc": utc_now(),
        "interpretation": "same_trajectory_endpoint_action_sensitivity_not_causal_counterfactual",
        "models": list(MODEL_ORDER),
        "representative_layers": representative,
        "prefix_rule": "arithmetic mean over available p8/p12/p16/p24; never best-prefix",
        "candidate_horizons": list(HORIZONS),
        "per_prefix": {str(key): value for key, value in frozen.items()},
        "jobs_total": len(jobs),
        "jobs_ready": ready,
        "jobs_unavailable": len(jobs) - ready,
        "source_registry": str(REGISTRY.relative_to(ROOT)),
        "source_registry_sha256": sha256_path(REGISTRY),
        "source_analysis_metadata": str(ANALYSIS_METADATA.relative_to(ROOT)),
        "source_analysis_metadata_sha256": sha256_path(ANALYSIS_METADATA),
    }
    (output / "frozen_action_set_metadata.json").write_text(
        json.dumps(action_set_metadata, indent=2) + "\n"
    )
    (output / "README.md").write_text(
        "# V1 C3-from-C1 action-sensitivity sweep\n\n"
        "This package evaluates frozen V1 C1 checkpoints only. It performs no training.\n\n"
        "The frozen candidates are h1/h2/h4 endpoints on the same observed trajectory. "
        "The results therefore support action-sensitivity language, not causal or "
        "counterfactual claims. Each prefix has 276 fixed validation anchors and three "
        "candidate endpoints. Model summaries use the paper's representative layer and "
        "the arithmetic mean over available p8/p12/p16/p24 values.\n\n"
        f"Prepared jobs: {ready}/{len(jobs)}.\n"
    )
    print(json.dumps({"output": str(output), "ready": ready, "total": len(jobs)}, indent=2))


def read_sweep(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _remote_child_names(client: Any, parent: str) -> list[str]:
    """List immediate child names while preserving storage errors."""
    names = []
    for value in client.list(parent):
        name = str(value).rstrip("/").rsplit("/", 1)[-1]
        if name:
            names.append(name)
    return sorted(set(names))


def preflight(args: argparse.Namespace) -> None:
    """Fail-closed availability audit for the frozen V1 feature namespace.

    ``aoss_fs.exists`` intentionally converts remote exceptions to ``False`` for
    training robustness.  That behavior is inappropriate for a sweep gate: a
    connection failure must remain distinguishable from an absent object.  This
    audit therefore calls the configured client directly and records the raw
    exception without launching evaluation.
    """
    from aoss_client.client import Client

    rows = read_sweep(args.manifest)
    client = Client(args.aoss_conf)
    checked_at = utc_now()
    try:
        root_children = _remote_child_names(client, STORAGE_PARENT)
        storage_error = ""
    except Exception as error:
        root_children = []
        storage_error = f"{type(error).__name__}: {error}"

    expected_roots = {
        STREAM_ROOT.rstrip("/").rsplit("/", 1)[-1],
        TARGET_ROOT.rstrip("/").rsplit("/", 1)[-1],
    }
    observed = set(root_children)
    base_roots_available = not storage_error and expected_roots <= observed
    for row in rows:
        if row.get("status") != "READY":
            row["input_status"] = "NOT_APPLICABLE"
            row["input_reason"] = row.get("reason") or "C1 checkpoint unavailable"
        elif storage_error:
            row["input_status"] = "PREFLIGHT_ERROR"
            row["input_reason"] = storage_error
        elif not base_roots_available:
            missing = sorted(expected_roots - observed)
            row["input_status"] = "UNAVAILABLE"
            row["input_reason"] = (
                "V1 feature namespace absent from storage root: " + ",".join(missing)
            )
        else:
            try:
                stream_ok = bool(client.isdir(row["streaming_feat_root"]))
                target_ok = bool(client.isdir(row["target_feat_root"]))
            except Exception as error:
                row["input_status"] = "PREFLIGHT_ERROR"
                row["input_reason"] = f"{type(error).__name__}: {error}"
            else:
                row["input_status"] = "AVAILABLE" if stream_ok and target_ok else "UNAVAILABLE"
                missing = []
                if not stream_ok:
                    missing.append("streaming")
                if not target_ok:
                    missing.append("target")
                row["input_reason"] = "" if not missing else (
                    "missing V1 model-prefix feature root(s): " + ",".join(missing)
                )

    fields = list(rows[0])
    for name in ("input_status", "input_reason"):
        if name not in fields:
            fields.append(name)
    write_csv(args.manifest, rows, fields)
    write_csv(args.manifest.parent / "checkpoint_selection.csv", rows, fields)
    counts: dict[str, int] = {}
    for row in rows:
        key = row["input_status"]
        counts[key] = counts.get(key, 0) + 1
    receipt = {
        "checked_at_utc": checked_at,
        "storage_parent": STORAGE_PARENT,
        "root_children": root_children,
        "expected_v1_roots": sorted(expected_roots),
        "storage_error": storage_error,
        "input_status_counts": counts,
        "v2_substitution_forbidden": True,
        "interpretation": (
            "Only the exact V1 namespaces selected by the historical C1 runs are eligible; "
            "canonical-v2 and seeded-v1-named V2 namespaces are inventory evidence only."
        ),
    }
    path = args.manifest.parent / "input_availability.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


def gate_rows(rows: list[dict[str, str]]) -> None:
    ready = [row for row in rows if row.get("status") == "READY"]
    blocked = [row for row in ready if row.get("input_status") != "AVAILABLE"]
    if blocked:
        counts: dict[str, int] = {}
        for row in blocked:
            key = row.get("input_status") or "UNVERIFIED"
            counts[key] = counts.get(key, 0) + 1
        raise RuntimeError(
            "C3 sweep input gate failed for "
            f"{len(blocked)}/{len(ready)} checkpoint-ready cells: {counts}. "
            "Run the preflight command after restoring the exact V1 feature roots; "
            "V2 substitution is forbidden."
        )


def gate(args: argparse.Namespace) -> None:
    rows = read_sweep(args.manifest)
    gate_rows(rows)
    print(json.dumps({
        "manifest": str(args.manifest),
        "checkpoint_ready_cells": sum(row.get("status") == "READY" for row in rows),
        "input_gate": "PASS",
    }, indent=2))


def audit(args: argparse.Namespace) -> None:
    rows = read_sweep(args.manifest)
    for row in rows:
        if row["status"] != "READY":
            continue
        checkpoint = Path(row["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(f"selected checkpoint disappeared: {checkpoint}")
        if checkpoint.stat().st_size != int(row["checkpoint_bytes"]):
            raise ValueError(f"selected checkpoint size changed: {checkpoint}")
        row["checkpoint_sha256"] = sha256_path(checkpoint)
        frozen_manifest = Path(row["frozen_manifest"])
        if sha256_path(frozen_manifest) != row["frozen_manifest_sha256"]:
            raise ValueError(f"frozen action-set manifest changed: {frozen_manifest}")
    fields = list(rows[0])
    write_csv(args.manifest, rows, fields)
    selection = args.manifest.parent / "checkpoint_selection.csv"
    write_csv(selection, rows, fields)
    receipt = {
        "audited_at_utc": utc_now(),
        "ready_checkpoint_count": sum(row["status"] == "READY" for row in rows),
        "unavailable_checkpoint_count": sum(row["status"] != "READY" for row in rows),
        "checkpoint_hash_algorithm": "sha256",
        "all_ready_checkpoints_hashed": all(
            row["checkpoint_sha256"] not in {"", "NOT_COMPUTED"}
            for row in rows if row["status"] == "READY"
        ),
    }
    (args.manifest.parent / "selection_audit.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt, indent=2))


def hydra_value(path: str) -> str:
    return path.replace("=", r"\=")


def run_one(job: dict[str, str], gpu: str, python: str, allow_cpu: bool) -> dict[str, Any]:
    output = Path(job["output_dir"])
    summary_path = output / "eval/val_c3_action_sensitivity_summary.json"
    prediction_path = output / "eval/val_c3_action_sensitivity_predictions.pt"
    log_path = output / "runner.log"
    output.mkdir(parents=True, exist_ok=True)
    if summary_path.is_file() and prediction_path.is_file():
        try:
            summary = json.loads(summary_path.read_text())
            if int(summary.get("n_valid_action_sets", 0)) == int(job["n_action_sets_expected"]):
                return {"job_key": job["job_key"], "returncode": 0, "skipped": True}
        except (ValueError, OSError, json.JSONDecodeError):
            pass

    env = os.environ.copy()
    env.update({
        "PROJECT_ROOT": str(ROOT),
        "PYTHONPATH": str(ROOT),
        "CUDA_VISIBLE_DEVICES": gpu,
        "INSCENE_DATA_ROOT": DATA_ROOT,
        "INSCENE_MANIFEST": job["frozen_manifest"],
        "INSCENE_STREAMING_FEAT_ROOT": job["streaming_feat_root"],
        "INSCENE_TARGET_FEAT_ROOT": job["target_feat_root"],
        "INSCENE_TRUST_EXPANDED_MANIFEST": "1",
        "Q00_EVALUATION_MODE": "LEGACY_AS_IS_LOOSE",
        "Q00_LEGACY_AS_IS_STRICT_CONSTANTS_COMPAT": "1",
        "Q00_REQUIRE_CUDA": "0" if allow_cpu else "1",
        "Q00_PROTOCOL": "c3_action_sensitivity_from_c1_v1",
        "Q00_MANIFEST_SHA256": job["frozen_manifest_sha256"],
        "Q00_UNIVERSE_ID": job["job_key"],
        "MPLCONFIGDIR": f"/tmp/matplotlib-c3-{job['job_key']}",
        "HYDRA_FULL_ERROR": "1",
    })
    cmd = [
        python,
        "vidfm3d/eval_diag.py",
        f"experiment={job['experiment']}",
        f"ckpt_path={hydra_value(job['checkpoint'])}",
        f"feature_layer={job['representative_layer']}",
        f"prefix_len={job['prefix']}",
        "action_horizons=[1,2,4]",
        "+evaluation_protocol=c3_action_sensitivity",
        "+eval_split=val",
        "+bootstrap_reps=1000",
        "train=false",
        "test=false",
        "data.data_module.num_workers_val=2",
        "data.data_module.batch_size_per_device_val=2",
        "logger.wandb.offline=true",
        f"job_name=c3_action_sensitivity_{job['job_key']}",
        f"hydra.run.dir={hydra_value(str(output))}",
    ]
    start = time.time()
    with log_path.open("w") as log:
        log.write(json.dumps({"started_at_utc": utc_now(), "gpu": gpu, "command": cmd}) + "\n")
        log.flush()
        process = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        log.write(json.dumps({
            "finished_at_utc": utc_now(),
            "returncode": process.returncode,
            "elapsed_seconds": time.time() - start,
        }) + "\n")
    return {
        "job_key": job["job_key"],
        "returncode": process.returncode,
        "skipped": False,
        "elapsed_seconds": time.time() - start,
        "summary": str(summary_path),
        "log": str(log_path),
    }


def run(args: argparse.Namespace) -> None:
    rows = read_sweep(args.manifest)
    gate_rows(rows)
    jobs = [row for row in rows if row["status"] == "READY"]
    jobs = [job for index, job in enumerate(jobs) if index % args.num_shards == args.shard_index]
    gpu_ids = [value.strip() for value in args.gpu_ids.split(",") if value.strip()]
    if not gpu_ids:
        raise ValueError("--gpu-ids must contain at least one GPU ID")
    status_path = args.manifest.parent / f"run_status_shard{args.shard_index:02d}.jsonl"

    def worker(slot: int, gpu: str) -> list[dict[str, Any]]:
        results = []
        for index, job in enumerate(jobs):
            if index % len(gpu_ids) == slot:
                result = run_one(job, gpu, args.python, args.allow_cpu)
                results.append(result)
                with status_path.open("a") as handle:
                    handle.write(json.dumps({"time": utc_now(), **result}) + "\n")
        return results

    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = [executor.submit(worker, slot, gpu) for slot, gpu in enumerate(gpu_ids)]
        for future in as_completed(futures):
            failures.extend(result for result in future.result() if result["returncode"] != 0)
    if failures:
        raise SystemExit(f"{len(failures)} C3 evaluation cells failed; see {status_path}")


def finite_mean(values: Iterable[float]) -> float:
    selected = [float(value) for value in values if math.isfinite(float(value))]
    return sum(selected) / len(selected) if selected else float("nan")


def finalize(args: argparse.Namespace) -> None:
    import torch

    from vidfm3d.eval_diag import (
        _bootstrap_c1_action_sensitivity,
        _summarize_c1_action_sensitivity,
    )

    jobs = [row for row in read_sweep(args.manifest) if row["status"] == "READY"]
    loaded: dict[str, tuple[dict[str, str], list[dict[str, Any]]]] = {}
    missing = []
    for job in jobs:
        path = Path(job["output_dir"]) / "eval/val_c3_action_sensitivity_predictions.pt"
        if not path.is_file():
            missing.append(job["job_key"])
            continue
        records = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(records, list):
            raise TypeError(f"{path} is not a list of action-set records")
        loaded[job["job_key"]] = (job, records)
    if missing and not args.allow_incomplete:
        raise FileNotFoundError(f"missing {len(missing)} prediction files; first: {missing[:5]}")

    result_dir = args.manifest.parent / "final"
    result_dir.mkdir(parents=True, exist_ok=True)
    common_by_prefix: dict[int, list[str]] = {}
    for prefix in PREFIXES:
        cell_sets = []
        for job, records in loaded.values():
            if int(job["prefix"]) == prefix:
                cell_sets.append({str(record["action_set_id"]) for record in records})
        common = sorted(set.intersection(*cell_sets)) if cell_sets else []
        if cell_sets and len(common) != int(next(
            job["n_action_sets_expected"] for job, _ in loaded.values() if int(job["prefix"]) == prefix
        )):
            raise ValueError(f"prefix {prefix}: completed cells do not share the full frozen action set")
        common_by_prefix[prefix] = common
        payload = {
            "prefix": prefix,
            "action_set_ids": common,
            "sha256": hashlib.sha256("\n".join(common).encode()).hexdigest(),
            "n": len(common),
        }
        (result_dir / f"frozen_action_set_ids_p{prefix}.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )

    metric_names = (
        "intervention_validity",
        "intervention_chance",
        "intervention_validity_chance_adjusted",
        "exact_candidate_validity",
        "exact_candidate_chance",
        "intervention_margin",
        "counterfactual_consistency",
        "action_sensitivity_cosine",
        "target_sensitivity_cosine",
        "target_to_prediction_sensitivity_pearson",
        "sensitivity_calibration_mae",
    )
    cell_rows: list[dict[str, Any]] = []
    for key, (job, records) in loaded.items():
        allowed = set(common_by_prefix[int(job["prefix"])])
        filtered = [record for record in records if str(record["action_set_id"]) in allowed]
        summary = _summarize_c1_action_sensitivity(filtered)
        summary["bootstrap_95ci"] = _bootstrap_c1_action_sensitivity(
            filtered, reps=args.bootstrap_reps, seed=args.seed
        )
        summary.update({
            "job_key": key,
            "model": job["model"],
            "model_display": job["model_display"],
            "representative_layer": int(job["representative_layer"]),
            "prefix": int(job["prefix"]),
            "checkpoint": job["checkpoint"],
            "checkpoint_sha256": job["checkpoint_sha256"],
            "frozen_manifest_sha256": job["frozen_manifest_sha256"],
            "action_set_id_sha256": hashlib.sha256("\n".join(sorted(allowed)).encode()).hexdigest(),
            "diagnostic_scope": "same_trajectory_endpoint_action_sensitivity_not_causal_counterfactual",
        })
        (result_dir / f"{key}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        cell_rows.append({
            "job_key": key,
            "model": job["model"],
            "model_display": job["model_display"],
            "representative_layer": job["representative_layer"],
            "prefix": job["prefix"],
            "n_action_sets": summary["n_valid_action_sets"],
            "n_candidates": summary["n_valid_candidates"],
            **{metric: summary.get(metric, float("nan")) for metric in metric_names},
        })

    if cell_rows:
        write_csv(result_dir / "c3_action_sensitivity_by_model_prefix.csv", cell_rows, list(cell_rows[0]))
    model_rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        selected = [row for row in cell_rows if row["model"] == model]
        model_rows.append({
            "model": model,
            "model_display": MODEL_DISPLAY[model],
            "representative_layer": (
                selected[0]["representative_layer"] if selected else ""
            ),
            "prefixes": ";".join(str(row["prefix"]) for row in sorted(selected, key=lambda row: int(row["prefix"]))),
            "n_prefix": len(selected),
            "prefix_aggregation": "arithmetic_mean_over_available_p8_p12_p16_p24_never_best",
            **{
                metric: finite_mean(float(row[metric]) for row in selected)
                for metric in metric_names
            },
        })
    write_csv(result_dir / "c3_action_sensitivity_by_model.csv", model_rows, list(model_rows[0]))
    completion = {
        "finalized_at_utc": utc_now(),
        "completed_cells": len(cell_rows),
        "ready_cells": len(jobs),
        "missing_prediction_cells": missing,
        "models": list(MODEL_ORDER),
        "prefixes": list(PREFIXES),
        "candidate_horizons": list(HORIZONS),
        "bootstrap_unit": "action_set",
        "bootstrap_reps": args.bootstrap_reps,
        "diagnostic_scope": "same_trajectory_endpoint_action_sensitivity_not_causal_counterfactual",
    }
    (result_dir / "completion.json").write_text(json.dumps(completion, indent=2) + "\n")
    print(json.dumps(completion, indent=2))


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(description=__doc__)
    commands = top.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--hash-checkpoints", action="store_true")
    prepare_parser.set_defaults(func=prepare)

    run_parser = commands.add_parser("run")
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--shard-index", type=int, default=0)
    run_parser.add_argument("--num-shards", type=int, default=1)
    run_parser.add_argument("--gpu-ids", default="0")
    run_parser.add_argument("--python", default=os.environ.get("PYTHON_BIN", PYTHON_DEFAULT))
    run_parser.add_argument("--allow-cpu", action="store_true")
    run_parser.set_defaults(func=run)

    audit_parser = commands.add_parser("audit")
    audit_parser.add_argument("--manifest", type=Path, required=True)
    audit_parser.set_defaults(func=audit)

    preflight_parser = commands.add_parser("preflight")
    preflight_parser.add_argument("--manifest", type=Path, required=True)
    preflight_parser.add_argument(
        "--aoss-conf",
        default=os.environ.get("AOSS_CONF", "/mnt/umm/users/yinbaiqiao/aoss/petreloss.conf"),
    )
    preflight_parser.set_defaults(func=preflight)

    gate_parser = commands.add_parser("gate")
    gate_parser.add_argument("--manifest", type=Path, required=True)
    gate_parser.set_defaults(func=gate)

    finalize_parser = commands.add_parser("finalize")
    finalize_parser.add_argument("--manifest", type=Path, required=True)
    finalize_parser.add_argument("--bootstrap-reps", type=int, default=1000)
    finalize_parser.add_argument("--seed", type=int, default=42)
    finalize_parser.add_argument("--allow-incomplete", action="store_true")
    finalize_parser.set_defaults(func=finalize)
    return top


def main() -> None:
    args = parser().parse_args()
    if getattr(args, "num_shards", 1) < 1:
        raise ValueError("--num-shards must be positive")
    if not 0 <= getattr(args, "shard_index", 0) < getattr(args, "num_shards", 1):
        raise ValueError("--shard-index must be in [0, num-shards)")
    args.func(args)


if __name__ == "__main__":
    main()
