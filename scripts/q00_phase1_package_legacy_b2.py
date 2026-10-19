#!/usr/bin/env python3
"""Package the 30 audited legacy B2 results into an isolated protocol namespace.

This script never reads or writes experiments/registry.csv or results/canonical/*.csv.
It promotes only rows explicitly classified PACKAGE by the frozen Q00 Wave-E audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "legacy_b2_shared_hidden_8_12_16_24_val_v1"
DEFAULT_INPUT = ROOT / (
    "reports/q00_phase1_runs/"
    "q00_phase1_wave_b_e_20260829T075939Z_2075f3eb/"
    "provenance/wave_e_resolution_inputs/b2_minimax.csv"
)
DEFAULT_OUTPUT = ROOT / "results/canonical_protocols" / PROTOCOL
PROTECTED = [
    ROOT / "experiments/registry.csv",
    *(ROOT / "results/canonical" / f"{name}_results.csv" for name in ("a1", "a2", "all", "b2", "c1", "c2", "c3")),
]
METRICS = (
    "mean_ang_err_deg",
    "mean_az_err_deg",
    "mean_el_err_deg",
    "top1",
    "top3",
    "mean_log_dist_err",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def prediction_contract(path: Path, summary: dict[str, Any]) -> tuple[str, dict[str, float]]:
    records = torch.load(path, map_location="cpu", weights_only=False)
    require(isinstance(records, list), f"prediction payload is not a list: {path}")
    require(len(records) == int(summary["n"]), f"prediction n mismatch: {path}")
    normalized: list[tuple[Any, ...]] = []
    columns: dict[str, list[float]] = defaultdict(list)
    for index, record in enumerate(records):
        require(isinstance(record, dict), f"prediction record {index} is not a dict: {path}")
        require(record.get("valid") is True, f"prediction record {index} is not valid: {path}")
        scene = Path(str(record["scene_path"])).name
        polar = torch.as_tensor(record["polar_gt"]).detach().cpu().reshape(-1)
        require(polar.numel() == 3 and bool(torch.isfinite(polar).all()), f"invalid polar_gt: {path}:{index}")
        target_bin = int(record["target_bin"])
        normalized.append((scene, target_bin, *(float(v).hex() for v in polar.tolist())))
        for name, source in (
            ("mean_ang_err_deg", "ang_err_deg"),
            ("mean_az_err_deg", "az_err_deg"),
            ("mean_el_err_deg", "el_err_deg"),
            ("top1", "top1"),
            ("top3", "top3"),
            ("mean_log_dist_err", "logd_err"),
        ):
            value = float(record[source])
            require(math.isfinite(value), f"non-finite prediction value {source}: {path}:{index}")
            columns[name].append(value)
    normalized.sort()
    identity = hashlib.sha256(
        json.dumps(normalized, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    recomputed = {name: sum(values) / len(values) for name, values in columns.items()}
    for name in METRICS:
        expected = float(summary[name])
        require(math.isfinite(expected), f"non-finite summary metric {name}: {path}")
        require(abs(expected - recomputed[name]) <= 1e-6, f"metric mismatch {name}: {path}")
    return identity, recomputed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    require(ROOT.resolve() in output.parents, f"output must be under repository: {output}")
    require(not output.exists(), f"refusing to overwrite existing namespace: {output}")
    before = {rel(path): {"sha256": sha256(path), "size_bytes": path.stat().st_size} for path in PROTECTED}

    with source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        source_rows = list(reader)
    rows = [row for row in source_rows if row["phase1_disposition"] == "PACKAGE"]
    require(len(rows) == 30, f"expected 30 PACKAGE rows, found {len(rows)}")
    require(len({row["universe_id"] for row in rows}) == 30, "duplicate universe_id in PACKAGE rows")
    require({row["campaign"] for row in rows} == {"default8"}, "unexpected campaign")
    require({row["family"] for row in rows} == {"B2"}, "unexpected family")
    require({row["probe"] for row in rows} == {"ego_belief_v2"}, "unexpected probe")

    result_rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    cohort_identities: dict[int, set[str]] = defaultdict(set)
    for row in sorted(rows, key=lambda value: (value["model"], int(value["prefix"]))):
        summary_path = ROOT / row["raw_summary_path"]
        prediction_path = ROOT / row["prediction_artifact_path"]
        checkpoint_path = ROOT / row["checkpoint_path"]
        hydra_path = ROOT / row["hydra_config_path"]
        run_manifest_path = hydra_path.parent.parent / "run_manifest.json"
        paths = {
            "summary": summary_path,
            "predictions": prediction_path,
            "checkpoint": checkpoint_path,
            "hydra_config": hydra_path,
            "run_manifest": run_manifest_path,
        }
        for kind, path in paths.items():
            require(path.is_file(), f"missing {kind}: {path}")
        require(sha256(summary_path) == row["raw_summary_sha256"], f"summary SHA mismatch: {summary_path}")
        summary = load_json(summary_path)
        run_manifest = load_json(run_manifest_path)
        require(int(summary["n"]) == int(row["n"]), f"summary n mismatch: {summary_path}")
        require(summary.get("probe_type") == "ego_belief_v2", f"wrong probe_type: {summary_path}")
        require(str(summary.get("vfm_name")) == row["model"], f"wrong model: {summary_path}")
        require(int(summary.get("feature_layer")) == int(row["layer"]), f"wrong layer: {summary_path}")
        require(Path(str(summary.get("ckpt_path"))).resolve() == checkpoint_path.resolve(), f"wrong checkpoint: {summary_path}")
        require(run_manifest.get("data", {}).get("split_manifest") is None, f"unexpected run-bound split: {run_manifest_path}")
        require(run_manifest.get("data", {}).get("split_manifest_sha256") is None, f"unexpected run-bound split SHA: {run_manifest_path}")
        config = run_manifest.get("config", {})
        require(config.get("model", {}).get("probe_type") == "ego_belief_v2", f"wrong run probe: {run_manifest_path}")
        require(int(config.get("feature_layer")) == int(row["layer"]), f"wrong run layer: {run_manifest_path}")
        require(int(config.get("prefix_len")) == int(row["prefix"]), f"wrong run prefix: {run_manifest_path}")
        require(config.get("streaming_hidden_prefix_lengths") == [8, 12, 16, 24], f"wrong hidden-prefix contract: {run_manifest_path}")
        require(int(row["final_epoch"]) == 49 and int(row["expected_max_epochs"]) == 50, f"wrong finality contract: {row['universe_id']}")
        require(row["checkpoint_status"] == "SUMMARY_BOUND_FINAL_EPOCH49", f"wrong checkpoint status: {row['universe_id']}")
        require("strict_pass=true" in row["clean_completion_status"], f"clean-completion failure: {row['universe_id']}")

        identity, recomputed = prediction_contract(prediction_path, summary)
        cohort_identities[int(row["prefix"])].add(identity)
        artifact_hashes: dict[str, str] = {}
        for kind, path in paths.items():
            digest = sha256(path)
            artifact_hashes[kind] = digest
            artifacts.append({
                "universe_id": row["universe_id"],
                "artifact_kind": kind,
                "path": rel(path),
                "sha256": digest,
                "size_bytes": path.stat().st_size,
            })
        result_rows.append({
            "protocol_canonical_id": f"{PROTOCOL}__{row['universe_id']}__mean_ang_err_deg",
            "universe_id": row["universe_id"],
            "campaign": row["campaign"],
            "model_variant": row["model_variant"],
            "protocol": PROTOCOL,
            "family": row["family"],
            "probe": row["probe"],
            "model": row["model"],
            "layer": row["layer"],
            "prefix": row["prefix"],
            "horizon": "all",
            "seed": "42",
            "n": int(summary["n"]),
            **{name: summary[name] for name in METRICS},
            "prediction_identity_sha256": identity,
            "reconstructed_signature": row["sample_label_signature"],
            "summary_path": rel(summary_path),
            "summary_sha256": artifact_hashes["summary"],
            "prediction_path": rel(prediction_path),
            "prediction_sha256": artifact_hashes["predictions"],
            "checkpoint_path": rel(checkpoint_path),
            "checkpoint_sha256": artifact_hashes["checkpoint"],
            "hydra_config_path": rel(hydra_path),
            "hydra_config_sha256": artifact_hashes["hydra_config"],
            "run_manifest_path": rel(run_manifest_path),
            "run_manifest_sha256": artifact_hashes["run_manifest"],
            "main_canonical_valid": "false",
            "protocol_canonical_valid": "true",
            "protocol_cohort_consistent": "true",
            "scientifically_comparable": "false",
            "analysis_eligible": "false",
            "comparability_scope": "artifact/label cohort only; raw-frame feature lineage is not run-bound",
            "split_manifest": "NULL",
            "git_commit": "unknown",
            "metric_recompute_max_abs_error": max(abs(float(summary[name]) - recomputed[name]) for name in METRICS),
        })

    expected_prefix_counts = {8: 8, 12: 8, 16: 7, 24: 7}
    actual_prefix_counts = Counter(int(row["prefix"]) for row in result_rows)
    require(dict(actual_prefix_counts) == expected_prefix_counts, f"unexpected prefix cohort: {actual_prefix_counts}")
    require(all(len(cohort_identities[prefix]) == 1 for prefix in expected_prefix_counts), "within-prefix label identity mismatch")
    require(len({row["protocol_canonical_id"] for row in result_rows}) == 30, "duplicate protocol canonical IDs")

    staging_parent = output.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{PROTOCOL}.stage.", dir=staging_parent))
    try:
        def write_csv(path: Path, values: list[dict[str, Any]]) -> None:
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)

        write_csv(stage / "b2_results.csv", result_rows)
        write_csv(stage / "artifact_inventory.csv", artifacts)
        namespace = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "parent_count": 30,
            "atomic_result_count": 30,
            "acceptance_basis": "User instructed PACKAGE rows to enter an independent legacy/protocol namespace on 2026-08-30.",
            "main_canonical_included": False,
            "protocol_cohort_consistency": "within this exact reconstructed legacy cohort only",
            "scientifically_comparable": False,
            "analysis_eligible": False,
            "limitations": [
                "No run-bound split manifest exists; split_manifest is NULL.",
                "Historical run manifests record git commit as unknown.",
                "Rows must not be mixed with strict inscene15k_v1 or historical_candidate_v1_val_protocol results.",
                "This package performs no new evaluation and does not validate the historical feature cache against canonical h124 manifests.",
                "Dedicated BY_PREFIX roots lack a run-bound raw-frame index; these rows are excluded from strict global analysis.",
            ],
            "source": rel(source),
            "source_sha256": sha256(source),
            "prefix_parent_counts": {str(k): v for k, v in sorted(actual_prefix_counts.items())},
            "prefix_prediction_identity_sha256": {str(k): next(iter(v)) for k, v in sorted(cohort_identities.items())},
            "protected_preimages": before,
        }
        (stage / "namespace.json").write_text(json.dumps(namespace, indent=2, sort_keys=True) + "\n")
        audit = {
            "status": "PASS",
            "protocol": PROTOCOL,
            "expected_parents": 30,
            "packaged_parents": len(result_rows),
            "duplicate_universe_ids": 0,
            "duplicate_protocol_canonical_ids": 0,
            "non_finite_metrics": 0,
            "artifact_count": len(artifacts),
            "artifact_hash_mismatches": 0,
            "summary_prediction_metric_mismatches": 0,
            "within_prefix_prediction_identity_mismatches": 0,
            "existing_main_canonical_writes": 0,
            "protocol_canonical_valid_parents": 30,
            "strict_analysis_eligible_parents": 0,
        }
        (stage / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
        os.replace(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    after = {rel(path): {"sha256": sha256(path), "size_bytes": path.stat().st_size} for path in PROTECTED}
    require(before == after, "protected source-of-truth file changed during packaging")
    print(json.dumps({"status": "PASS", "output": rel(output), "parents": 30, "artifacts": len(artifacts)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
