#!/usr/bin/env python3
"""Finalize the Q00 phase-1 Wave-B execution and Wave-E read-only resolution.

This script is deliberately fail-closed.  It will only write the requested
analysis products after the frozen 99-row eval audit and expected-ID-only
promotion have both passed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/analysis/q00_1096_phase1"
REPORT = ROOT / "reports/q00_phase1_wave_b_e.md"
RECONCILIATION_RELATIVE = "results/analysis/q00_1096/universe_1096_reconciled.csv"
WAVES_RELATIVE = "results/analysis/q00_1096/execution_waves.csv"
CANONICAL_NAMES = (
    "all_results.csv",
    "a1_results.csv",
    "a2_results.csv",
    "b2_results.csv",
    "c1_results.csv",
    "c2_results.csv",
    "c3_results.csv",
)
PROTECTED_RELATIVE = (
    "experiments/registry.csv",
    *(f"results/canonical/{name}" for name in CANONICAL_NAMES),
)
EXPECTED_CANONICAL_ADDITIONS = {
    "all_results.csv": 105,
    "a1_results.csv": 19,
    "a2_results.csv": 15,
    "b2_results.csv": 1,
    "c1_results.csv": 28,
    "c2_results.csv": 128,
    "c3_results.csv": 70,
}
EXPECTED_WAVE_E_HASHES = {
    "b2_minimax": "0bdb3dbac731ee3793166949e28289bf26077db4ed70a5fcfca14e555871b5a3",
    "missing42": "34ea498faae438b7e1ec7fc56e1a16240bdd365082af278f58b84e055e21fbd6",
    "wancog": "44d81655c694c24d55a7f5536ba3d473ed9da6dbfe91186ed709caea6e819094",
}
DISPOSITIONS = {
    "PACKAGE", "EVAL_ONLY", "CLEAN_RETRAIN", "FEATURE_THEN_RETRAIN", "PERMANENTLY_BLOCKED"
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path, delimiter: str = ",") -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def fnum(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise SystemExit(f"required frozen evidence is missing: {path}")
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"required frozen evidence is invalid JSON: {path}: {exc}") from exc


def frozen_source_path(bundle: Path, bundle_info: dict[str, Any], relative: str) -> Path:
    """Resolve one source table only through its bundle-frozen copy and hash chain."""
    path = bundle / "inputs/source_of_truth" / relative
    if not path.is_file():
        raise SystemExit(f"bundle-frozen source is missing: {path}")
    actual = sha256(path)
    source_hashes = bundle_info.get("source_hashes")
    if not isinstance(source_hashes, dict):
        raise SystemExit("bundle metadata lacks source_hashes")
    frozen_key = f"inputs/source_of_truth/{relative}"
    if source_hashes.get(relative) != actual or source_hashes.get(frozen_key) != actual:
        raise SystemExit(
            f"bundle-frozen source hash chain mismatch for {relative}: "
            f"actual={actual} live_key={source_hashes.get(relative)!r} "
            f"frozen_key={source_hashes.get(frozen_key)!r}"
        )
    return path


def expected_ids_from_jobs(jobs: list[dict[str, str]]) -> list[str]:
    result: list[str] = []
    for row in jobs:
        try:
            outputs = json.loads(row["expected_outputs_json"])
        except (KeyError, json.JSONDecodeError) as exc:
            raise SystemExit(f"invalid expected_outputs_json for {row.get('universe_id', '')}: {exc}") from exc
        if not isinstance(outputs, list):
            raise SystemExit(f"expected_outputs_json is not a list: {row.get('universe_id', '')}")
        for item in outputs:
            if not isinstance(item, dict) or not item.get("registry_id"):
                raise SystemExit(f"invalid expected registry output: {row.get('universe_id', '')}")
            result.append(str(item["registry_id"]))
    if len(result) != len(set(result)):
        raise SystemExit("frozen job manifests contain duplicate expected registry IDs")
    return result


def verify_audit_lock(
    bundle: Path,
    bundle_info: dict[str, Any],
    scope: str,
    expected_rows: int,
    expected_atomic: int,
) -> dict[str, Any]:
    """Verify an audit, its CSV, and its exact lock as one immutable evidence unit."""
    audit_path = bundle / "audit" / f"{scope}_audit.json"
    results_path = bundle / "audit" / f"{scope}_results.csv"
    lock_path = bundle / "audit" / f"{scope}_audit_lock.json"
    manifest_path = bundle / "inputs" / (
        "wave_b_canary_3.tsv" if scope == "canary" else "wave_b_99.tsv"
    )
    expected_ids_path = bundle / "inputs/expected_registry_ids.json"
    script_inventory_path = bundle / "provenance/script_inventory.json"
    runtime_inventory_path = bundle / "provenance/runtime_inventory.json"
    required = (
        audit_path,
        results_path,
        lock_path,
        manifest_path,
        expected_ids_path,
        script_inventory_path,
        runtime_inventory_path,
    )
    if not all(path.is_file() for path in required):
        missing = [str(path) for path in required if not path.is_file()]
        raise SystemExit(f"{scope} audit evidence is incomplete: {missing}")

    audit = read_json(audit_path)
    lock = read_json(lock_path)
    _, jobs = read_csv(manifest_path, delimiter="\t")
    _, result_rows = read_csv(results_path)
    atomic_ids = expected_ids_from_jobs(jobs)
    job_ids = [row.get("universe_id", "") for row in jobs]
    result_ids = [row.get("universe_id", "") for row in result_rows]
    all_expected_ids = read_json(expected_ids_path)
    if not isinstance(all_expected_ids, list) or len(all_expected_ids) != 105:
        raise SystemExit("expected_registry_ids.json is not the exact 105-ID list")
    if len(set(map(str, all_expected_ids))) != 105:
        raise SystemExit("expected_registry_ids.json contains duplicate IDs")
    if scope == "full" and list(map(str, all_expected_ids)) != atomic_ids:
        raise SystemExit("full job manifest ordered IDs differ from expected_registry_ids.json")
    if not set(atomic_ids) <= set(map(str, all_expected_ids)):
        raise SystemExit(f"{scope} audit contains an ID outside expected_registry_ids.json")
    if (
        len(jobs) != expected_rows
        or any(not value for value in job_ids)
        or len(set(job_ids)) != expected_rows
        or len(result_rows) != expected_rows
        or any(not value for value in result_ids)
        or len(set(result_ids)) != expected_rows
        or set(result_ids) != set(job_ids)
        or len(atomic_ids) != expected_atomic
    ):
        raise SystemExit(f"{scope} audit row/ID cardinality mismatch")
    if any(row.get("status") != "PASS" for row in result_rows):
        raise SystemExit(f"{scope} audit CSV contains a non-PASS row")

    scalar_expectations = {
        "expected_universe_ids": expected_rows,
        "actual_result_rows": expected_rows,
        "successful_summaries": expected_rows,
        "failed_summaries": 0,
        "expected_atomic_registry_ids": expected_atomic,
        "actual_atomic_registry_ids": expected_atomic,
    }
    if not isinstance(audit, dict) or audit.get("ok") is not True or audit.get("scope") != scope:
        raise SystemExit(f"{scope} audit is not an exact PASS")
    for key, expected in scalar_expectations.items():
        if int(audit.get(key, -1)) != expected:
            raise SystemExit(f"{scope} audit scalar mismatch: {key}")
    if (
        audit.get("expected_id_contract_ok") is not True
        or audit.get("duplicate_universe_ids")
        or audit.get("duplicate_registry_ids")
        or audit.get("problems")
        or audit.get("frozen_hash_problems")
        or audit.get("protected_source_of_truth_drift")
        or int(audit.get("non_finite_metrics", -1)) != 0
    ):
        raise SystemExit(f"{scope} audit contains ID/hash/metric problems")
    audit_results = audit.get("results")
    if not isinstance(audit_results, list) or len(audit_results) != expected_rows:
        raise SystemExit(f"{scope} audit JSON does not contain exactly {expected_rows} results")
    audit_by_id: dict[str, dict[str, Any]] = {}
    for record in audit_results:
        if not isinstance(record, dict):
            raise SystemExit(f"{scope} audit JSON contains a non-object result")
        universe_id = str(record.get("universe_id", ""))
        if not universe_id or universe_id in audit_by_id or record.get("status") != "PASS":
            raise SystemExit(f"{scope} audit JSON contains a missing/duplicate/non-PASS result")
        audit_by_id[universe_id] = record
    if set(audit_by_id) != set(job_ids):
        raise SystemExit(f"{scope} audit JSON universe IDs differ from its frozen job manifest")
    csv_by_id = {row["universe_id"]: row for row in result_rows}
    csv_lock_fields = (
        "status",
        "summary_path",
        "summary_sha256",
        "prediction_sha256",
        "preflight_sha256",
        "postflight_sha256",
        "command_sha256",
        "manifest_sha256",
        "checkpoint_sha256",
        "expected_registry_ids",
        "metrics_json",
        "n",
    )
    for universe_id, record in audit_by_id.items():
        csv_record = csv_by_id[universe_id]
        for key in csv_lock_fields:
            if str(csv_record.get(key, "")) != str(record.get(key, "")):
                raise SystemExit(f"{scope} audit JSON/CSV mismatch for {universe_id}: {key}")
    expected_input_hashes = {
        "bundle_json": sha256(bundle / "bundle.json"),
        "job_manifest": sha256(manifest_path),
        "expected_registry_ids": sha256(expected_ids_path),
        "script_inventory": sha256(script_inventory_path),
        "runtime_inventory": sha256(runtime_inventory_path),
    }
    if audit.get("input_hashes") != expected_input_hashes:
        raise SystemExit(f"{scope} audit input hash chain mismatch")
    expected_lock = {
        "scope": scope,
        "audit_sha256": sha256(audit_path),
        "results_csv_sha256": sha256(results_path),
        "job_manifest_sha256": sha256(manifest_path),
        "expected_registry_ids_sha256": sha256(expected_ids_path),
        "expected_registry_ids": sorted(atomic_ids),
        "bundle_json_sha256": sha256(bundle / "bundle.json"),
    }
    if lock != expected_lock:
        raise SystemExit(f"{scope} audit lock does not exactly bind audit/results/manifest/IDs/bundle")

    source_hashes = bundle_info.get("source_hashes", {})
    for key, path in (
        ("inputs/expected_registry_ids.json", expected_ids_path),
        ("provenance/script_inventory.json", script_inventory_path),
        ("provenance/runtime_inventory.json", runtime_inventory_path),
    ):
        if source_hashes.get(key) != sha256(path):
            raise SystemExit(f"bundle source hash mismatch for {key}")

    if scope == "canary":
        gate_path = bundle / "launcher/wave_b_remaining_96_canary_gate.json"
        gate = read_json(gate_path)
        if not (
            gate.get("required") is True
            and gate.get("ok") is True
            and all(gate.get("checks", {}).values())
            and gate.get("audit_sha256") == sha256(audit_path)
            and gate.get("lock_sha256") == sha256(lock_path)
        ):
            raise SystemExit("remaining-96 launcher is not bound to the passing canary lock")

    return {
        "audit": audit,
        "audit_sha256": sha256(audit_path),
        "results_sha256": sha256(results_path),
        "lock_sha256": sha256(lock_path),
        "manifest_sha256": sha256(manifest_path),
        "results_by_universe": audit_by_id,
    }


def verify_frozen_finalizer(bundle: Path, bundle_info: dict[str, Any]) -> None:
    """Do not let an unfrozen post-promotion finalizer attest to the run."""
    inventory_path = bundle / "provenance/script_inventory.json"
    if bundle_info.get("source_hashes", {}).get("provenance/script_inventory.json") != sha256(inventory_path):
        raise SystemExit("script inventory differs from bundle metadata")
    inventory = read_json(inventory_path)
    suffix = "provenance/scripts/scripts/q00_phase1_finalize.py"
    matches = [item for item in inventory if str(item.get("path", "")).endswith(suffix)]
    if len(matches) != 1:
        raise SystemExit("frozen script inventory does not uniquely contain q00_phase1_finalize.py")
    item = matches[0]
    frozen = ROOT / str(item["path"])
    expected = str(item.get("sha256", ""))
    if not frozen.is_file() or sha256(frozen) != expected or sha256(Path(__file__).resolve()) != expected:
        raise SystemExit("current finalizer differs from its bundle-frozen copy")


def verify_promotion_chain(
    bundle: Path,
    bundle_info: dict[str, Any],
    full_audit_sha256: str,
) -> dict[str, Any]:
    """Bind promotion metadata, staging, journal, and current protected files."""
    staging_audit_path = bundle / "promotion/staging/staging_audit.json"
    metadata_path = bundle / "promotion/post/metadata.json"
    journal_path = bundle / "promotion/journal.json"
    protected_before_path = bundle / "inputs/protected_before.json"
    staging_audit = read_json(staging_audit_path)
    metadata = read_json(metadata_path)
    journal = read_json(journal_path)
    protected_before_raw = read_json(protected_before_path)
    if set(protected_before_raw) != set(PROTECTED_RELATIVE):
        raise SystemExit("protected_before.json does not describe the exact protected file set")
    protected_before = {
        relative: str(protected_before_raw[relative].get("sha256", ""))
        for relative in PROTECTED_RELATIVE
    }
    stage_hashes = staging_audit.get("stage_hashes")
    if not isinstance(stage_hashes, dict) or set(stage_hashes) != set(PROTECTED_RELATIVE):
        raise SystemExit("staging audit does not contain the exact protected stage hash set")

    staged_paths = {
        "experiments/registry.csv": bundle / "promotion/staging/registry.csv",
        **{
            f"results/canonical/{name}": bundle / "promotion/staging/canonical" / name
            for name in CANONICAL_NAMES
        },
    }
    current_paths = {relative: ROOT / relative for relative in PROTECTED_RELATIVE}
    staged_hashes = {
        relative: sha256(path) if path.is_file() else ""
        for relative, path in staged_paths.items()
    }
    current_hashes = {
        relative: sha256(path) if path.is_file() else ""
        for relative, path in current_paths.items()
    }
    if staged_hashes != stage_hashes or current_hashes != stage_hashes:
        raise SystemExit("staging/current protected SHA256 values differ from committed stage hashes")

    stage_metadata = {
        key: value for key, value in staging_audit.items() if key not in {"ok", "stage_hashes"}
    }
    if staging_audit.get("ok") is not True:
        raise SystemExit("staging audit is not PASS")
    registry_stage = staging_audit.get("registry", {})
    canonical_stage = staging_audit.get("canonical", {})
    if not (
        int(registry_stage.get("added_rows", -1)) == 105
        and registry_stage.get("before_sha256") == protected_before["experiments/registry.csv"]
        and registry_stage.get("staged_sha256") == stage_hashes["experiments/registry.csv"]
        and isinstance(canonical_stage, dict)
        and set(canonical_stage) == set(CANONICAL_NAMES)
    ):
        raise SystemExit("staging registry/canonical metadata is inconsistent")
    for name in CANONICAL_NAMES:
        item = canonical_stage[name]
        relative = f"results/canonical/{name}"
        if not (
            int(item.get("added_rows", -1)) == EXPECTED_CANONICAL_ADDITIONS[name]
            and item.get("before_sha256") == protected_before[relative]
            and item.get("staged_sha256") == stage_hashes[relative]
        ):
            raise SystemExit(f"staging canonical metadata mismatch: {name}")

    metadata_checks = {
        "ok": metadata.get("ok") is True,
        "parent_universe_rows": int(metadata.get("parent_universe_rows", -1)) == 99,
        "primary_registry_rows": int(metadata.get("primary_registry_rows", -1)) == 105,
        "old_registry_rows_changed": int(metadata.get("old_registry_rows_changed", -1)) == 0,
        "old_canonical_rows_changed": int(metadata.get("old_canonical_rows_changed", -1)) == 0,
        "full_audit_sha256": metadata.get("full_audit_sha256") == full_audit_sha256,
        "stage_hashes": metadata.get("stage_hashes") == stage_hashes,
    }
    if not all(metadata_checks.values()):
        raise SystemExit(f"promotion metadata contract mismatch: {metadata_checks}")

    journal_checks = {
        "state": journal.get("state") == "COMMITTED",
        "ok": journal.get("ok") is True,
        "validate_only": journal.get("validate_only") is False,
        "bundle": str(Path(str(journal.get("bundle", ""))).resolve()) == str(bundle),
        "run_tag": journal.get("run_tag") == bundle_info.get("run_tag"),
        "parent_universe_rows": int(journal.get("parent_universe_rows", -1)) == 99,
        "primary_registry_rows": int(journal.get("primary_registry_rows", -1)) == 105,
        "full_audit_sha256": journal.get("full_audit_sha256") == full_audit_sha256,
        "protected_before": journal.get("protected_before") == protected_before,
        "stage_hashes": journal.get("stage_hashes") == stage_hashes,
        "stage_metadata": journal.get("stage_metadata") == stage_metadata,
        "committed_files": journal.get("committed_files") == list(PROTECTED_RELATIVE),
    }
    if not all(journal_checks.values()):
        raise SystemExit(f"promotion COMMITTED journal contract mismatch: {journal_checks}")

    return {
        "staging_audit_sha256": sha256(staging_audit_path),
        "post_metadata_sha256": sha256(metadata_path),
        "journal_sha256": sha256(journal_path),
        "stage_hashes": stage_hashes,
        "current_protected_hashes": current_hashes,
    }


def normalize_wave_e(
    source: str,
    row: dict[str, str],
    universe: dict[str, dict[str, str]],
) -> dict[str, Any]:
    uid = row["universe_id"]
    base = universe[uid]
    if source == "b2_minimax":
        disposition = row["phase1_disposition"]
        manifest_path = row.get("manifest_path", "")
        manifest_sha = ""
        manifest_status = row.get("manifest_evidence", "")
        checkpoint_path = row.get("checkpoint_path", "")
        checkpoint_status = row.get("checkpoint_status", "") + ";" + row.get("clean_completion_status", "")
        feature_root = row.get("feature_root", "")
        feature_status = row.get("feature_status", "")
        next_action = row.get("exact_next_action", "")
        stop_gate = row.get("gate_stop_condition", "")
        eligibility = row.get("canonical_eligibility", "")
        evidence = row.get("evidence_paths", "")
        numeric = truth(row.get("numeric_result_present"))
        protocol_resolution = row.get("eval_contract_evidence", "")
        config_path = row.get("config_path", "")
        run_dir = row.get("hydra_config_path", "")
        source_group = "DEFAULT_B2_OR_MINIMAX_CAMERA"
        estimated_train = fnum(base.get("execution_wave") == "Wave E" and "0" or "0")
        estimated_eval = 0.0
        conditional = ""
    elif source == "missing42":
        disposition = row["classification"]
        manifest_path = row.get("manifest_candidate_path", "")
        manifest_sha = row.get("manifest_candidate_sha256", "")
        manifest_status = row.get("reconstruction_status", "")
        checkpoint_path = row.get("selected_checkpoint_path", "")
        checkpoint_status = row.get("checkpoint_strict_status", "") + ";" + row.get("checkpoint_reconciliation_status", "")
        feature_root = row.get("streaming_feature_root", "")
        feature_status = row.get("feature_status", "")
        next_action = row.get("next_action", "")
        stop_gate = row.get("blocker_reason", "")
        eligibility = row.get("canonical_eligibility", "")
        evidence = row.get("evidence_paths", "")
        numeric = truth(base.get("numeric_result_present"))
        protocol_resolution = (
            f"train_identity={row.get('historical_train_identity_status','')};"
            f"val_identity={row.get('historical_val_identity_status','')};"
            f"val_order_equals_canonical={row.get('candidate_val_order_equals_canonical','')}"
        )
        config_path = row.get("selected_checkpoint_run_root", "")
        run_dir = row.get("selected_checkpoint_run_root", "")
        source_group = "MISSING_FILTERED_MANIFEST_CHECKPOINT"
        estimated_train = 0.0
        estimated_eval = 0.05 if disposition == "EVAL_ONLY" else 0.0
        conditional = row.get("conditional_fast_path", "")
    else:
        disposition = row["phase1_classification"]
        manifest_path = row.get("candidate_manifest_path_or_recipe", "")
        manifest_sha = row.get("candidate_manifest_sha256", "")
        manifest_status = row.get("manifest_recoverability", "") + ";" + row.get("manifest_limitation", "")
        checkpoint_path = row.get("effective_checkpoint_path", "") or row.get("checkpoint_path", "")
        checkpoint_status = row.get("effective_checkpoint_status", "") or row.get("q00_clean_completion_status", "")
        feature_root = row.get("qualified_feature_root", "") or row.get("historical_feature_root", "")
        feature_status = row.get("feature_status", "") + ";" + row.get("feature_readiness_basis", "")
        next_action = row.get("next_action", "")
        stop_gate = row.get("canonical_gate", "")
        eligibility = row.get("canonical_gate", "")
        evidence = row.get("evidence_paths", "")
        numeric = truth(row.get("historical_numeric_present"))
        protocol_resolution = (
            f"lineage={row.get('lineage_resolution_status','')};"
            f"protocol={row.get('lineage_protocol_match','')};"
            f"manifest={row.get('manifest_recoverability','')}"
        )
        config_path = row.get("config_path", "")
        run_dir = row.get("repair_run_directory", "") or row.get("run_directory", "")
        source_group = "WAN_COG_PROVISIONAL"
        estimated_train = fnum(row.get("phase1_required_train_gpu_h_point", row.get("estimated_train_gpu_h_point", "0")))
        estimated_eval = fnum(row.get("phase1_required_eval_gpu_h_point", row.get("estimated_eval_gpu_h_point", "0")))
        conditional = row.get("fallback_classification_if_live_canary_fails", "")
    if disposition not in DISPOSITIONS:
        raise ValueError(f"unknown Wave-E disposition for {uid}: {disposition}")
    return {
        "universe_id": uid,
        "campaign": base["campaign"],
        "model": base["model"],
        "model_variant": base["model_variant"],
        "protocol": base["protocol"],
        "family": base["family"],
        "probe": base["probe"],
        "layer": base["layer"],
        "prefix": base["prefix"],
        "default_timestep": base["default_timestep"],
        "horizon": base["horizon"],
        "seed": base["seed"],
        "resolution_source_group": source_group,
        "phase1_disposition": disposition,
        "numeric_result_present_before": str(numeric).lower(),
        "canonical_valid_before": base["canonical_valid"],
        "manifest_path_or_recipe": manifest_path,
        "manifest_sha256": manifest_sha,
        "manifest_resolution": manifest_status,
        "checkpoint_path": checkpoint_path,
        "checkpoint_resolution": checkpoint_status,
        "feature_root": feature_root,
        "feature_resolution": feature_status,
        "protocol_resolution": protocol_resolution,
        "config_or_run_path": config_path,
        "run_directory": run_dir,
        "conditional_fallback": conditional,
        "estimated_train_gpu_h_point": f"{estimated_train:.2f}",
        "estimated_eval_gpu_h_point": f"{estimated_eval:.2f}",
        "next_action": next_action,
        "gate_and_stop_condition": stop_gate,
        "canonical_eligibility": eligibility,
        "evidence_paths": evidence,
        "source_record_json": json.dumps(row, sort_keys=True, separators=(",", ":")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--wave-e-b2-minimax")
    parser.add_argument("--wave-e-missing42")
    parser.add_argument("--wave-e-wancog")
    args = parser.parse_args()
    bundle = Path(args.bundle).resolve()
    bundle_info = read_json(bundle / "bundle.json")
    if not isinstance(bundle_info, dict):
        raise SystemExit("bundle.json is not an object")
    if Path(str(bundle_info.get("bundle", ""))).resolve() != bundle:
        raise SystemExit("bundle.json does not bind the requested bundle path")
    verify_frozen_finalizer(bundle, bundle_info)

    canary_evidence = verify_audit_lock(bundle, bundle_info, "canary", 3, 3)
    full_evidence = verify_audit_lock(bundle, bundle_info, "full", 99, 105)
    full_results_by_universe = full_evidence["results_by_universe"]
    for universe_id, record in canary_evidence["results_by_universe"].items():
        if full_results_by_universe.get(universe_id) != record:
            raise SystemExit(
                f"canary audit result is not exactly preserved in the promotion-bound full audit: "
                f"{universe_id}"
            )
    promotion_audit_path = bundle / "promotion/staging/staging_audit.json"
    promotion_meta_path = bundle / "promotion/post/metadata.json"
    promotion_evidence = verify_promotion_chain(
        bundle,
        bundle_info,
        full_evidence["audit_sha256"],
    )

    reconciliation_path = frozen_source_path(
        bundle, bundle_info, RECONCILIATION_RELATIVE
    )
    waves_path = frozen_source_path(bundle, bundle_info, WAVES_RELATIVE)
    _, universe_rows = read_csv(reconciliation_path)
    _, wave_rows = read_csv(waves_path)
    universe = {row["universe_id"]: row for row in universe_rows}
    if len(universe_rows) != 1096 or len(universe) != 1096:
        raise SystemExit("frozen reconciliation is not 1096 unique rows")
    wave_b_ids = {row["universe_id"] for row in wave_rows if row["wave"] == "Wave B"}
    wave_c_rows = [row for row in wave_rows if row["wave"] == "Wave C"]
    wave_e_ids = {row["universe_id"] for row in wave_rows if row["wave"] == "Wave E"}
    if (len(wave_b_ids), len(wave_c_rows), len(wave_e_ids)) != (99, 66, 249):
        raise SystemExit("source execution-wave counts changed")

    frozen_wave_e_dir = bundle / "provenance/wave_e_resolution_inputs"
    source_specs = {
        "b2_minimax": Path(args.wave_e_b2_minimax) if args.wave_e_b2_minimax else frozen_wave_e_dir / "b2_minimax.csv",
        "missing42": Path(args.wave_e_missing42) if args.wave_e_missing42 else frozen_wave_e_dir / "missing42.csv",
        "wancog": Path(args.wave_e_wancog) if args.wave_e_wancog else frozen_wave_e_dir / "wancog.csv",
    }
    raw_wave_e: dict[str, list[dict[str, str]]] = {}
    for name, path in source_specs.items():
        actual = sha256(path)
        if actual != EXPECTED_WAVE_E_HASHES[name]:
            raise SystemExit(f"Wave-E input hash mismatch for {name}: {actual}")
        _, raw_wave_e[name] = read_csv(path)
    wave_e_rows = [
        normalize_wave_e(name, row, universe)
        for name in ("b2_minimax", "missing42", "wancog")
        for row in raw_wave_e[name]
    ]
    if len(wave_e_rows) != 249 or len({row["universe_id"] for row in wave_e_rows}) != 249:
        raise SystemExit("normalized Wave-E rows are not 249 unique parents")
    if {row["universe_id"] for row in wave_e_rows} != wave_e_ids:
        raise SystemExit("Wave-E normalized set differs from frozen execution_waves.csv")
    wave_e_counts = Counter(row["phase1_disposition"] for row in wave_e_rows)
    expected_wave_e_counts = Counter({
        "PACKAGE": 30,
        "EVAL_ONLY": 22,
        "CLEAN_RETRAIN": 177,
        "FEATURE_THEN_RETRAIN": 0,
        "PERMANENTLY_BLOCKED": 20,
    })
    if wave_e_counts != expected_wave_e_counts:
        raise SystemExit(f"unexpected Wave-E resolution counts: {wave_e_counts}")

    _, wave_b_jobs = read_csv(bundle / "inputs/wave_b_99.tsv", delimiter="\t")
    _, wave_b_audit_rows = read_csv(bundle / "audit/full_results.csv")
    jobs = {row["universe_id"]: row for row in wave_b_jobs}
    audits = {row["universe_id"]: row for row in wave_b_audit_rows}
    if set(jobs) != wave_b_ids or set(audits) != wave_b_ids:
        raise SystemExit("Wave-B job/audit identity sets are not frozen 99")
    wave_b_results: list[dict[str, Any]] = []
    for uid in sorted(wave_b_ids):
        job, audit = jobs[uid], audits[uid]
        if audit.get("status") != "PASS":
            raise SystemExit(f"Wave-B row did not pass: {uid}")
        expected = json.loads(job["expected_outputs_json"])
        wave_b_results.append({
            "universe_id": uid,
            "campaign": job["campaign"],
            "model": job["model"],
            "model_variant": job["model_variant"],
            "protocol": job["protocol"],
            "family": job["family"],
            "probe": job["probe"],
            "layer": job["layer"],
            "prefix": job["prefix"],
            "seed": job["seed"],
            "canary": job["is_canary"],
            "status": "PROMOTED_EXPECTED_ID_ONLY",
            "expected_n": job["expected_n"],
            "actual_n": audit.get("n", ""),
            "manifest": job["manifest"],
            "manifest_sha256": job["manifest_sha256"],
            "checkpoint": job["checkpoint"],
            "checkpoint_sha256": job["checkpoint_sha256"],
            "feature_root": job["streaming_feat_root"],
            "target_feature_root": job["target_feat_root"],
            "summary_path": job["summary_path"],
            "summary_sha256": audit.get("summary_sha256", ""),
            "prediction_path": job["prediction_path"],
            "registry_ids": ";".join(item["registry_id"] for item in expected),
            "metric_contracts": ";".join(item["metric_name"] for item in expected),
            "metrics_json": audit.get("metrics_json", ""),
            "preflight_path": job["preflight_path"],
            "command_path": job["command_path"],
            "promotion_audit": str(promotion_audit_path.relative_to(ROOT)),
        })

    wave_e_by_id = {row["universe_id"]: row for row in wave_e_rows}
    live_rows: list[dict[str, Any]] = []
    for original in universe_rows:
        row: dict[str, Any] = dict(original)
        uid = row["universe_id"]
        before_numeric = truth(row["numeric_result_present"])
        before_canonical = truth(row["canonical_valid"])
        if uid in wave_b_ids:
            job = jobs[uid]
            expected = json.loads(job["expected_outputs_json"])
            row.update({
                "phase1_resolution": "WAVE_B_EVAL_PROMOTED",
                "phase1_status": "CANONICAL_COMPLETE",
                "numeric_result_present_after": "true",
                "canonical_valid_after": "true",
                "scientifically_comparable_after": "YES_WITHIN_EXACT_CAMPAIGN_PROTOCOL",
                "phase1_added_registry_ids": ";".join(item["registry_id"] for item in expected),
                "phase1_summary_path": job["summary_path"],
                "phase1_next_action": "none; promoted after full staging audit",
            })
        elif uid in wave_e_by_id:
            resolved = wave_e_by_id[uid]
            mapping = {
                "PACKAGE": "RAW_RESULT_READY_TO_PACKAGE",
                "EVAL_ONLY": "CHECKPOINT_READY_EVAL_ONLY",
                "CLEAN_RETRAIN": "MANIFEST_OR_PROVENANCE_BLOCKED",
                "FEATURE_THEN_RETRAIN": "FEATURE_PARTIAL_OR_MISSING",
                "PERMANENTLY_BLOCKED": "DEPRECATED_OR_PROTOCOL_INVALID",
            }
            row.update({
                "phase1_resolution": resolved["phase1_disposition"],
                "phase1_status": mapping[resolved["phase1_disposition"]],
                "numeric_result_present_after": str(before_numeric).lower(),
                "canonical_valid_after": str(before_canonical).lower(),
                "scientifically_comparable_after": row["scientifically_comparable"],
                "phase1_added_registry_ids": "",
                "phase1_summary_path": "",
                "phase1_next_action": resolved["next_action"],
            })
        else:
            row.update({
                "phase1_resolution": "UNCHANGED_OUT_OF_PHASE1_SCOPE",
                "phase1_status": row["status"],
                "numeric_result_present_after": str(before_numeric).lower(),
                "canonical_valid_after": str(before_canonical).lower(),
                "scientifically_comparable_after": row["scientifically_comparable"],
                "phase1_added_registry_ids": "",
                "phase1_summary_path": "",
                "phase1_next_action": row["next_action"],
            })
        row["phase1_run_tag"] = bundle_info["run_tag"]
        live_rows.append(row)
    numeric_before = sum(truth(row["numeric_result_present"]) for row in universe_rows)
    numeric_after = sum(truth(row["numeric_result_present_after"]) for row in live_rows)
    canonical_before = sum(truth(row["canonical_valid"]) for row in universe_rows)
    canonical_after = sum(truth(row["canonical_valid_after"]) for row in live_rows)
    if (numeric_before, numeric_after, canonical_before, canonical_after) != (833, 932, 642, 741):
        raise SystemExit("before/after parent counts differ from audited invariants")

    # A deterministic, non-submitted clean-training queue: frozen Wave C plus
    # the Wave-E rows resolved to CLEAN_RETRAIN. Every row remains gated.
    queue_rows: list[dict[str, Any]] = []
    for wave in wave_c_rows:
        base = universe[wave["universe_id"]]
        queue_rows.append({
            "action_generation": "q00_phase1_20260829",
            "source_wave": "Wave C",
            "resolution_source_group": "Q00_FEATURE_READY",
            "phase1_source_path": "results/analysis/q00_1096/execution_waves.csv",
            "phase1_source_sha256": sha256(waves_path),
            "phase1_row_sha256": hashlib.sha256(json.dumps(wave, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "universe_id": base["universe_id"],
            "campaign": base["campaign"], "model": base["model"],
            "model_variant": base["model_variant"], "protocol": base["protocol"],
            "family": base["family"], "probe": base["probe"],
            "layer": base["layer"], "prefix": base["prefix"],
            "default_timestep": base["default_timestep"], "horizon": base["horizon"], "seed": base["seed"],
            "manifest_path_or_recipe": base["train_manifest"] or base["manifest"],
            "manifest_sha256": base["train_manifest_sha256"] or base["manifest_sha256"],
            "manifest_gate_status": "READY_FOR_ROW_PREFLIGHT",
            "protocol_gate_status": "FROZEN_HISTORICAL_CANDIDATE_PROTOCOL",
            "config": base["config"],
            "config_identity_status": base["config_status"],
            "streaming_feature_root": base["streaming_feature_root"],
            "target_feature_root": base["target_feature_root"],
            "feature_status": base["feature_status"],
            "feature_canary_status": "PENDING_DISPATCH_PREFLIGHT",
            "required_feature_types": "streaming+target" if base["target_feature_root"] else "streaming",
            "training_mode": "CLEAN_NO_RESUME",
            "queue_state": "READY_ROW_PREFLIGHT",
            "dispatch_eligible": "false",
            "clean_no_resume": "true",
            "legacy_checkpoint_forbidden": "true",
            "planned_train_jobs": "1",
            "planned_eval_jobs": "1",
            "expected_max_epochs": base["expected_max_epochs"],
            "expected_final_epoch": base["expected_final_epoch"],
            "checkpoint_retention_policy": "IMMUTABLE_PROTECTED_BEST_PLUS_LAST",
            "cost_status": "Q00_HISTORICAL_MODEL_PROBE_ESTIMATE",
            "hold_reason": "row-level manifest/config/feature canary not yet executed",
            "estimated_train_gpu_h_point": wave["estimated_train_gpu_h_point"],
            "estimated_eval_gpu_h_point": wave["estimated_eval_gpu_h_point"],
            "entry_scripts": wave["entry_scripts"],
            "required_preflight": wave["gate_and_stop_condition"],
            "submission_state": "NOT_SUBMITTED_PHASE1_STOP_BOUNDARY",
            "next_action": wave["next_action"],
            "evidence_paths": wave["evidence_paths"],
        })
    for resolved in wave_e_rows:
        if resolved["phase1_disposition"] != "CLEAN_RETRAIN":
            continue
        base = universe[resolved["universe_id"]]
        detail = json.loads(resolved["source_record_json"])
        entry = "scripts/build_probe_fill_jobs.py;scripts/submit_probe_fill_sco.sh;scripts/run_probe_fill_shard.sh;scripts/launch_probe_fill.sh"
        queue_state = (
            "HOLD_PROTOCOL_CONTRACT"
            if resolved["resolution_source_group"] == "DEFAULT_B2_OR_MINIMAX_CAMERA"
            else "HOLD_MANIFEST_PROVENANCE"
        )
        queue_rows.append({
            "action_generation": "q00_phase1_20260829",
            "source_wave": "Wave E",
            "resolution_source_group": resolved["resolution_source_group"],
            "phase1_source_path": (
                "reports/q00_phase1_runs/" + bundle.name + "/provenance/wave_e_resolution_inputs/" +
                ({
                    "DEFAULT_B2_OR_MINIMAX_CAMERA": "b2_minimax.csv",
                    "MISSING_FILTERED_MANIFEST_CHECKPOINT": "missing42.csv",
                    "WAN_COG_PROVISIONAL": "wancog.csv",
                }[resolved["resolution_source_group"]])
            ),
            "phase1_source_sha256": ({
                "DEFAULT_B2_OR_MINIMAX_CAMERA": EXPECTED_WAVE_E_HASHES["b2_minimax"],
                "MISSING_FILTERED_MANIFEST_CHECKPOINT": EXPECTED_WAVE_E_HASHES["missing42"],
                "WAN_COG_PROVISIONAL": EXPECTED_WAVE_E_HASHES["wancog"],
            }[resolved["resolution_source_group"]]),
            "phase1_row_sha256": hashlib.sha256(resolved["source_record_json"].encode()).hexdigest(),
            "universe_id": base["universe_id"],
            "campaign": base["campaign"], "model": base["model"],
            "model_variant": base["model_variant"], "protocol": base["protocol"],
            "family": base["family"], "probe": base["probe"],
            "layer": base["layer"], "prefix": base["prefix"],
            "default_timestep": base["default_timestep"], "horizon": base["horizon"], "seed": base["seed"],
            "manifest_path_or_recipe": resolved["manifest_path_or_recipe"],
            "manifest_sha256": resolved["manifest_sha256"],
            "manifest_gate_status": "HOLD_RECONSTRUCTION_SIGNOFF" if queue_state == "HOLD_MANIFEST_PROVENANCE" else "HOLD_FORMAL_B2_PROTOCOL",
            "protocol_gate_status": "HOLD_SIGNOFF",
            "config": detail.get("config_path", base["config"]),
            "config_identity_status": "RESOLVED_BUT_NOT_DISPATCH_SIGNED",
            "streaming_feature_root": resolved["feature_root"],
            "target_feature_root": base["target_feature_root"],
            "feature_status": resolved["feature_resolution"],
            "feature_canary_status": "PENDING_AFTER_MANIFEST_PROTOCOL_GATE",
            "required_feature_types": "streaming+target" if base["target_feature_root"] else "streaming",
            "training_mode": "CLEAN_NO_RESUME",
            "queue_state": queue_state,
            "dispatch_eligible": "false",
            "clean_no_resume": "true",
            "legacy_checkpoint_forbidden": "true",
            "planned_train_jobs": "1",
            "planned_eval_jobs": "1",
            "expected_max_epochs": base["expected_max_epochs"],
            "expected_final_epoch": base["expected_final_epoch"],
            "checkpoint_retention_policy": "IMMUTABLE_PROTECTED_BEST_PLUS_LAST",
            "cost_status": "REESTIMATE_REQUIRED" if fnum(resolved["estimated_train_gpu_h_point"]) == 0 else "Q00_HISTORICAL_MODEL_PROBE_ESTIMATE",
            "hold_reason": resolved["gate_and_stop_condition"],
            "estimated_train_gpu_h_point": resolved["estimated_train_gpu_h_point"],
            "estimated_eval_gpu_h_point": resolved["estimated_eval_gpu_h_point"],
            "entry_scripts": entry,
            "required_preflight": resolved["gate_and_stop_condition"],
            "submission_state": "NOT_SUBMITTED_PHASE1_STOP_BOUNDARY",
            "next_action": resolved["next_action"],
            "evidence_paths": resolved["evidence_paths"],
        })
    queue_rows.sort(key=lambda row: (
        0 if row["source_wave"] == "Wave C" else 1,
        row["campaign"], row["model"], row["family"], row["probe"],
        int(row["prefix"] or 0), int(row["layer"] or 0), row["universe_id"],
    ))
    for index, row in enumerate(queue_rows, 1):
        row["queue_order"] = index
        row["queue_id"] = f"q00_phase1_train_{index:04d}"
        row["full_identity_key"] = json.dumps(
            {key: row[key] for key in (
                "campaign", "model", "model_variant", "protocol", "family", "probe",
                "layer", "prefix", "default_timestep", "horizon", "seed",
            )},
            sort_keys=True,
            separators=(",", ":"),
        )
        row["exclusive_action_group"] = row["universe_id"]
        config_path = ROOT / row["config"] if row["config"] else None
        row["resolved_config_sha256"] = sha256(config_path) if config_path and config_path.is_file() else ""
        row["manifest_signoff_id"] = "PENDING" if row["manifest_gate_status"].startswith("HOLD") else "ROW_PREFLIGHT_REQUIRED"
        row["fresh_run_directory"] = (
            f"logs/inscene15k_streaming/runs/q00_phase1_followup_20260829/"
            f"{row['universe_id']}"
        )
        row["feature_dependency_key"] = hashlib.sha256(json.dumps({
            "model": row["model"],
            "layer": row["layer"],
            "prefix": row["prefix"],
            "default_timestep": row["default_timestep"],
            "streaming_feature_root": row["streaming_feature_root"],
            "target_feature_root": row["target_feature_root"],
            "protocol": row["protocol"],
        }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        row["dispatch_dedup_key"] = hashlib.sha256(json.dumps({
            "identity": row["full_identity_key"],
            "manifest_sha256": row["manifest_sha256"],
            "resolved_config_sha256": row["resolved_config_sha256"],
            "streaming_feature_root": row["streaming_feature_root"],
            "target_feature_root": row["target_feature_root"],
            "action_generation": row["action_generation"],
        }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if len(queue_rows) != 243 or len({row["universe_id"] for row in queue_rows}) != 243:
        raise SystemExit("deterministic training queue is not 243 unique rows")

    # Recheck every externally mutable evidence unit immediately before the
    # first final-output write. This closes the verification/output TOCTOU
    # window without touching registry or canonical tables.
    canary_evidence_final = verify_audit_lock(bundle, bundle_info, "canary", 3, 3)
    full_evidence_final = verify_audit_lock(bundle, bundle_info, "full", 99, 105)
    promotion_evidence_final = verify_promotion_chain(
        bundle,
        bundle_info,
        full_evidence_final["audit_sha256"],
    )
    if (
        canary_evidence_final != canary_evidence
        or full_evidence_final != full_evidence
        or promotion_evidence_final != promotion_evidence
    ):
        raise SystemExit("audit/promotion evidence changed during finalization")

    OUT.mkdir(parents=True, exist_ok=True)
    wave_e_fields = list(wave_e_rows[0])
    wave_b_fields = list(wave_b_results[0])
    live_fields = list(live_rows[0])
    queue_fields = ["queue_id", "queue_order"] + [key for key in queue_rows[0] if key not in {"queue_id", "queue_order"}]
    write_csv(OUT / "wave_e_resolution.csv", wave_e_fields, sorted(wave_e_rows, key=lambda row: row["universe_id"]))
    write_csv(OUT / "wave_b_eval_results.csv", wave_b_fields, wave_b_results)
    write_csv(OUT / "live_universe_1096.csv", live_fields, live_rows)
    write_csv(OUT / "next_training_queue.csv", queue_fields, queue_rows)

    b2_rows = [row for row in wave_e_rows if row["campaign"] == "default8" and row["family"] == "B2"]
    packageable_b2 = sum(row["phase1_disposition"] == "PACKAGE" for row in b2_rows)
    b2_reeval = sum(row["phase1_disposition"] == "EVAL_ONLY" for row in b2_rows)
    submission_files = sorted((bundle / "launcher").glob("*submissions.json"))
    summary: dict[str, Any] = {
        "generated_at_utc": now(),
        "run_tag": bundle_info["run_tag"],
        "bundle_path": str(bundle.relative_to(ROOT)),
        "bundle_sha256": sha256(bundle / "bundle.json"),
        "frozen_inputs": {
            "universe_sha256": bundle_info["source_hashes"]["experiments/task_status_1096_before_expand_20260822.csv"],
            "reconciliation_sha256": bundle_info["source_hashes"]["results/analysis/q00_1096/universe_1096_reconciled.csv"],
            "execution_waves_sha256": bundle_info["source_hashes"]["results/analysis/q00_1096/execution_waves.csv"],
            "wave_e_input_sha256": EXPECTED_WAVE_E_HASHES,
        },
        "wave_b": {
            "canary_pass": True,
            "canary_success": 3,
            "expected_parents": 99,
            "success": 99,
            "atomic_registry_ids_promoted": 105,
            "duplicate_ids": 0,
            "non_finite_metrics": 0,
            "manifest_mismatches": 0,
            "feature_mismatches": 0,
            "promotion": "EXPECTED_ID_ONLY",
            "canary_audit_sha256": canary_evidence["audit_sha256"],
            "canary_audit_lock_sha256": canary_evidence["lock_sha256"],
            "full_audit_sha256": full_evidence["audit_sha256"],
            "full_audit_lock_sha256": full_evidence["lock_sha256"],
            "submission_records": [str(path.relative_to(ROOT)) for path in submission_files],
        },
        "counts": {
            "universe": 1096,
            "numeric_results_before": numeric_before,
            "numeric_results_after": numeric_after,
            "canonical_parents_before": canonical_before,
            "canonical_parents_after": canonical_after,
            "b2_packageable": packageable_b2,
            "b2_needs_reeval": b2_reeval,
            "wave_e": dict(sorted(wave_e_counts.items())),
            "next_training_job_count": len(queue_rows),
            "next_training_wave_c": sum(row["source_wave"] == "Wave C" for row in queue_rows),
            "next_training_wave_e": sum(row["source_wave"] == "Wave E" for row in queue_rows),
            "next_training_ready_row_preflight": sum(row["queue_state"] == "READY_ROW_PREFLIGHT" for row in queue_rows),
            "next_training_hold_manifest_provenance": sum(row["queue_state"] == "HOLD_MANIFEST_PROVENANCE" for row in queue_rows),
            "next_training_hold_protocol_contract": sum(row["queue_state"] == "HOLD_PROTOCOL_CONTRACT" for row in queue_rows),
        },
        "protected_files": {
            "status": "EXPECTED_WAVE_B_DIFF_ONLY",
            "added_parent_ids": 99,
            "added_atomic_ids": 105,
            "changed_existing_registry_rows": 0,
            "changed_or_removed_existing_canonical_rows": 0,
            "post_metadata": str(promotion_meta_path.relative_to(ROOT)),
            "post_metadata_sha256": promotion_evidence["post_metadata_sha256"],
            "staging_audit_sha256": promotion_evidence["staging_audit_sha256"],
            "journal_sha256": promotion_evidence["journal_sha256"],
            "committed_stage_hashes": promotion_evidence["stage_hashes"],
        },
        "scope_stop": {
            "training_jobs_submitted": 0,
            "feature_extraction_jobs_submitted": 0,
            "wave_e_eval_jobs_submitted": 0,
            "wave_e_package_rows_promoted": 0,
            "reason": "Phase-1 stops after Wave-B promotion and Wave-E classification.",
        },
    }
    write_json(OUT / "summary.json", summary)

    model_b = Counter(row["model"] for row in wave_b_results)
    family_b = Counter(row["family"] for row in wave_b_results)
    e_source_counts = Counter(row["resolution_source_group"] for row in wave_e_rows)
    lines = [
        "# Q00 Phase 1: frozen 1096 — Wave B eval-only 与 Wave E provenance resolution",
        "",
        f"生成时间：`{summary['generated_at_utc']}`  ",
        f"运行 bundle：`{summary['bundle_path']}`  ",
        f"bundle metadata SHA256：`{summary['bundle_sha256']}`",
        "",
        "## 结论",
        "",
        f"Wave B 的 99 个 parent（105 个 atomic metric IDs）全部通过 canary、全量 staging audit，并以 expected-ID-only 方式进入 registry/canonical。数值结果从 **{numeric_before}/1096** 提升到 **{numeric_after}/1096**，canonical-valid parents 从 **{canonical_before}/1096** 提升到 **{canonical_after}/1096**。已有 registry 行未改写，已有 canonical 行未删除或改写。",
        "",
        f"Wave E 249 行已逐行收敛为：PACKAGE={wave_e_counts['PACKAGE']}、EVAL_ONLY={wave_e_counts['EVAL_ONLY']}、CLEAN_RETRAIN={wave_e_counts['CLEAN_RETRAIN']}、FEATURE_THEN_RETRAIN={wave_e_counts['FEATURE_THEN_RETRAIN']}、PERMANENTLY_BLOCKED={wave_e_counts['PERMANENTLY_BLOCKED']}。本轮没有执行这些 eval/train/feature 或 package promotion。",
        "",
        "## A. 冻结输入与安全边界",
        "",
        f"- Universe：`experiments/task_status_1096_before_expand_20260822.csv`，SHA256 `{summary['frozen_inputs']['universe_sha256']}`，1096 行。",
        f"- Reconciliation：`results/analysis/q00_1096/universe_1096_reconciled.csv`，SHA256 `{summary['frozen_inputs']['reconciliation_sha256']}`。",
        f"- Execution waves：`results/analysis/q00_1096/execution_waves.csv`，SHA256 `{summary['frozen_inputs']['execution_waves_sha256']}`。",
        "- Bundle 固化了 source tables、99-row TSV、checkpoint SHA/size/final epoch-step、manifest/config/script inventory、feature roots、完整 CLI、git commit/branch/status 和 dirty diff。",
        "- 所有 identity 均保留 `campaign/model_variant/protocol`；没有缺失 manifest fallback，也没有按近似文件名或宽松 key 匹配。",
        "- 本轮只提交 Wave B eval；训练、feature extraction、Wave E eval 均为 0。",
        "- 首版 bundle `q00_phase1_wave_b_e_20260829T052538Z_2075f3e` 在 worker 启动前被静态审计判定 identity/hash-chain 不合格；三个 STARTING canary 已停为 SUSPENDED，0 worker、0 输出、0 protected diff，且该 bundle 明确标记不可复用。正式结果只来自本报告列出的修正版 bundle。",
        "",
        "## B. Wave B 执行与 promotion",
        "",
        "### Canary",
        "",
        "LingBot、MiniMax-H3、Wan 各 1 个 canary 均验证：strict final checkpoint；bundle 内 frozen manifest + 原路径 SHA；远端 prefix-index/feature sample 的内容 SHA、sample identity 与 finite tensor；target feature（需要时）；frozen runtime；单卡 CUDA identity；expected n；metric schema；checkpoint eval 前后 SHA；无 fallback。三者全 PASS 且 audit lock 完整后，runner 才放行剩余 96。LingBot B2 额外锁定 276-row val manifest 中 47 个 construct-valid records 的历史跨模型一致 ordered signature；C1 不允许 valid-subset，必须 100% manifest coverage。",
        "",
        "### 全量审计",
        "",
        "| 项目 | 结果 |",
        "|---|---:|",
        "| expected universe parents | 99/99 |",
        "| successful summaries | 99/99 |",
        "| atomic metric IDs | 105/105 |",
        "| duplicate IDs | 0 |",
        "| non-finite metrics | 0 |",
        "| manifest mismatches | 0 |",
        "| feature mismatches | 0 |",
        "| existing registry rows changed | 0 |",
        "| existing canonical rows removed/changed | 0 |",
        "",
        f"Model parent 分布：{dict(sorted(model_b.items()))}。Family 分布：{dict(sorted(family_b.items()))}。逐行 summary、metric、hash、registry IDs 见 `results/analysis/q00_1096_phase1/wave_b_eval_results.csv`。",
        "",
        "## C. Wave E 只读 resolution",
        "",
        "| 来源组 | 行数 | 关键结论 |",
        "|---|---:|---|",
        f"| default B2 + MiniMax camera | {e_source_counts['DEFAULT_B2_OR_MINIMAX_CAMERA']} | 30 legacy B2 可条件 packaging；Qwen p16 与 F3R p24 clean retrain；20 standalone camera protocol-invalid。 |",
        f"| missing filtered manifest checkpoint | {e_source_counts['MISSING_FILTERED_MANIFEST_CHECKPOINT']} | 15 个 V-JEPA p16/p24 checkpoint 可 exact eval；其余 27 个 clean retrain。 |",
        f"| Wan/Cog provisional parents | {e_source_counts['WAN_COG_PROVISIONAL']} | 7 个 later protected B2 repair checkpoint 可 eval；其余 148 个 true-best 未保留，clean retrain。 |",
        "",
        "### default B2",
        "",
        "31 个已有 summary 的 row 中，30 个形成 prefix 内一致的 legacy cohort，并能从 predictions 重算指标；但 run-bound split_manifest 为 null、git 为 unknown，不能冒充 strict `inscene15k_v1`。它们只可在 `legacy_b2_shared_hidden_8_12_16_24_val_v1` namespace 下 packaging，且需人工接受 reconstructed protocol。本轮未 promotion。Qwen p16 的 hidden-prefix cohort、n 和构造不同，必须 clean retrain；F3R p24 checkpoint 在 epoch 28 中断，也必须从头 clean retrain。因此 `B2_PACKAGEABLE=30`、`B2_NEEDS_REEVAL=0`。",
        "",
        "### 42 个 missing-filtered-manifest checkpoint",
        "",
        "21 行训练日志实际绑定现存 canonical manifest，21 行绑定已丢失 filtered manifest。对 p16/p24，过滤为 no-op 且候选 body/ordered val identity 可证明，因此 15 个 strict-final V-JEPA rows 为 EVAL_ONLY。其余包含 non-final checkpoint、canonical-vs-filtered train mismatch 或无法证明历史 body 的 rows，严格归 CLEAN_RETRAIN。所有候选均检查 train/val sample overlap=0、scene overlap=0。",
        "",
        "### Wan/Cog provisional",
        "",
        "历史 162 个 Wan/Cog provisional representations 中，本阶段 Wave E 精确覆盖 155 个 noncanonical frozen parents；另 2 个 parent 已有后续 exact canonical，5 个 FOUND_NEARBY_ONLY 按 Q00 保留在 Wave F 人工决策，均未被静默并入这 155 行。旧 provisional numeric 155/155 禁止 promotion。深挖 later repair artifacts 后，7 个 B2 parents 有 immutable protected best、load-verified best/last、exact config/manifest log binding，改判 EVAL_ONLY。其余 148 个 true best 未被旧 callback 保留，不能 formal eval，归 CLEAN_RETRAIN。历史日志证明 exact filtered train/val 曾完整消费 remote AOSS feature；当前 read-only AOSS probe 超时只表示 live reachability 未核验，不能直接判 feature missing。每行在未来执行前必须通过 exact model/prefix/layer coverage、shape/dtype/finite/provenance canary；失败即停止并转 FEATURE_THEN_RETRAIN。",
        "",
        "### MiniMax camera-only",
        "",
        "20 行不是独立有效 camera protocol：当前实现定义 camera 为 streaming-depth 的 joint head。它们不能从 joint depth+camera 静默复制或 alias，因此永久标为 protocol-invalid；不应继续提交。",
        "",
        "## D. 下一训练队列与停止点",
        "",
        f"`next_training_queue.csv` 有 **{len(queue_rows)}** 个确定性 clean-no-resume parent jobs：原 Wave C 66 + 本轮 Wave E CLEAN_RETRAIN {wave_e_counts['CLEAN_RETRAIN']}。队列分为 READY_ROW_PREFLIGHT=66、HOLD_MANIFEST_PROVENANCE=175、HOLD_PROTOCOL_CONTRACT=2；`dispatch_eligible=false` 243/243。所有行均保留完整 identity、source-row SHA、manifest path/recipe+hash、resolved-config SHA、feature dependency key、fresh run directory、入口、成本与 stop gate，且状态为 `NOT_SUBMITTED_PHASE1_STOP_BOUNDARY`。另外 {wave_e_counts['EVAL_ONLY']} 个 Wave E eval-only rows 留待下一窗口，0 个 Wave E rows 当前主分类为 FEATURE_THEN_RETRAIN。",
        "",
        "这里的 CLEAN_RETRAIN 是最短路径分类，不是跳过 feature gate 的授权：Wan/Cog remote AOSS live canary、virtual/reconstructed manifest materialization/sign-off、B2 protocol freeze 等任一 gate 不通过，都必须 fail closed，不得启动训练。",
        "",
        "## 硬性验收",
        "",
        "```text",
        "WAVE_B_CANARY_PASS: true",
        "WAVE_B_SUCCESS: 99/99",
        f"NUMERIC_RESULTS_BEFORE: {numeric_before}/1096",
        f"NUMERIC_RESULTS_AFTER: {numeric_after}/1096",
        f"CANONICAL_PARENTS_BEFORE: {canonical_before}/1096",
        f"CANONICAL_PARENTS_AFTER: {canonical_after}/1096",
        f"B2_PACKAGEABLE: {packageable_b2}",
        f"B2_NEEDS_REEVAL: {b2_reeval}",
        f"WAVE_E_EVAL_ONLY: {wave_e_counts['EVAL_ONLY']}",
        f"WAVE_E_CLEAN_RETRAIN: {wave_e_counts['CLEAN_RETRAIN']}",
        f"WAVE_E_FEATURE_THEN_RETRAIN: {wave_e_counts['FEATURE_THEN_RETRAIN']}",
        f"WAVE_E_PERMANENTLY_BLOCKED: {wave_e_counts['PERMANENTLY_BLOCKED']}",
        f"NEXT_TRAINING_JOB_COUNT: {len(queue_rows)}",
        "PROTECTED_FILES_UNCHANGED_OR_EXPECTED_DIFF_ONLY: EXPECTED_WAVE_B_105_ATOMIC_ID_ADDITIONS_ONLY; EXISTING_ROWS_CHANGED=0",
        "```",
        "",
        "本轮停止于 Wave B promotion + Wave E classification。没有自动启动 243 个训练任务，没有抽 feature，没有新增 diffusion-step/planning/prediction-feature 实验。",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    output_hashes = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in (
            REPORT,
            OUT / "live_universe_1096.csv",
            OUT / "wave_b_eval_results.csv",
            OUT / "wave_e_resolution.csv",
            OUT / "next_training_queue.csv",
        )
    }
    summary["output_sha256"] = output_hashes
    write_json(OUT / "summary.json", summary)
    print(json.dumps({
        "ok": True,
        "numeric_before": numeric_before,
        "numeric_after": numeric_after,
        "canonical_before": canonical_before,
        "canonical_after": canonical_after,
        "wave_e": dict(wave_e_counts),
        "next_training_jobs": len(queue_rows),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
