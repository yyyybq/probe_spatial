#!/usr/bin/env python3
"""Build the fail-closed Q00 Phase-1 snapshot after direct-node lineage audits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/analysis/q00_1096_phase1"
REPORT = ROOT / "reports/q00_phase1_wave_b_e.md"
RECON = ROOT / "results/analysis/q00_1096/universe_1096_reconciled.csv"
WAVES = ROOT / "results/analysis/q00_1096/execution_waves.csv"
UNIVERSE = ROOT / "experiments/task_status_1096_before_expand_20260822.csv"
WAVE_B_BUNDLE = ROOT / (
    "reports/q00_phase1_runs/"
    "q00_phase1_wave_b_e_direct_shapecontract_20260830T040218Z_2075f3eb"
)
LEGACY_PACKAGE = ROOT / "results/canonical_protocols/legacy_b2_shared_hidden_8_12_16_24_val_v1"
PROTECTED_EXPECTED = {
    "experiments/registry.csv": "efc533bbe6286090d6c111c9d272ffb28cf7477b937f799f312eedf2a944431e",
    "results/canonical/a1_results.csv": "f7777676eafbc77dc8a8856e86a0901c98d6b5b3bdaddb568a975ef1c9f94d86",
    "results/canonical/a2_results.csv": "2613650bc7dcaa7a5bb5292c1953b9742657fbcb7967da3016b8c84c9f6ffa5f",
    "results/canonical/all_results.csv": "2a04a5e1f0165f1d7a06e7edcff38d956b845910ad95af3f7bbede3704133b17",
    "results/canonical/b2_results.csv": "0d4941709af83ba72820e4c463b96775d240096b44fca14bd4c5243e2dfa5f20",
    "results/canonical/c1_results.csv": "b88a37e239d150d297a4e97838d93888d5b6d3d0780df510e65b7f916b2ade1a",
    "results/canonical/c2_results.csv": "8c96a38500420af11427ed506bc92d4e1fc9d6b636bc34f0c4baf2d279fd9524",
    "results/canonical/c3_results.csv": "6cb2bec11abecb7cdf7549f14db84d58345e2bb5e9a0172f9435c80e5e2b7c5d",
}
EXPECTED_HASHES = {
    "universe": "9c7d6807b701de4b052819971a5c011e29c717e1c3e958ec9ae23116cdcfe779",
    "reconciliation": "3cda065533eb060e63cf01d3a7ab8252a081fcd3830d6bdbb1f51b6f1d9a4842",
    "wave_b": "8e9cb9c754a05bc2f794182353ce1ad4677ce91f249c963958fb24049dd70a81",
    "wave_e": "0ac2f426cab60f193c73f38a31794ed2384d5621af2c13d1f5aed4daa05c7313",
    "canonical_impact": "d6bfef88018130a4ef747b76a1fda32123c2b4ff46e4b0afd82cb8ba21467917",
}
PROBE_GPU_H = {
    "counterfactual": 14.94,
    "path_integration": 14.22,
    "action_dynamics": 9.68,
    "view_consistency": 20.95,
    "ego_belief_v2": 20.34,
    "streaming_mask": 31.22,
    "streaming_depth": 22.85,
    "ego_belief": 18.09,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def read_csv(path: Path, delimiter: str = ",") -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(bool(rows), f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wave-b", type=Path, default=Path("/tmp/q00_waveb_reclassification_99_v2.csv"))
    parser.add_argument("--wave-e", type=Path, default=Path("/tmp/q00_wavee_249_lineage_reclassified.csv"))
    parser.add_argument("--canonical-impact", type=Path, default=Path("/tmp/canonical_642_h124_shared24_lineage_impact_20260830.csv"))
    parser.add_argument("--canonical-impact-summary", type=Path, default=Path("/tmp/canonical_642_h124_shared24_lineage_impact_20260830_summary.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require(not OUT.exists(), f"refusing to overwrite existing output directory: {OUT}")
    require(not REPORT.exists(), f"refusing to overwrite existing report: {REPORT}")
    require(sha256(UNIVERSE) == EXPECTED_HASHES["universe"], "frozen universe hash mismatch")
    require(sha256(RECON) == EXPECTED_HASHES["reconciliation"], "reconciliation hash mismatch")
    require(sha256(args.wave_b) == EXPECTED_HASHES["wave_b"], "Wave-B audit hash mismatch")
    require(sha256(args.wave_e) == EXPECTED_HASHES["wave_e"], "Wave-E audit hash mismatch")
    require(sha256(args.canonical_impact) == EXPECTED_HASHES["canonical_impact"], "canonical impact hash mismatch")
    protected_before = {path: sha256(ROOT / path) for path in PROTECTED_EXPECTED}
    require(protected_before == PROTECTED_EXPECTED, "protected pre-image mismatch")

    _, universe_rows = read_csv(RECON)
    _, wave_rows = read_csv(WAVES)
    _, wave_b_rows = read_csv(args.wave_b)
    _, wave_e_rows = read_csv(args.wave_e)
    _, canonical_rows = read_csv(args.canonical_impact)
    _, jobs = read_csv(WAVE_B_BUNDLE / "inputs/wave_b_99.tsv", delimiter="\t")
    _, package_rows = read_csv(LEGACY_PACKAGE / "b2_results.csv")
    package_audit = json.loads((LEGACY_PACKAGE / "audit.json").read_text())
    canonical_summary = json.loads(args.canonical_impact_summary.read_text())
    universe = {row["universe_id"]: row for row in universe_rows}
    require(len(universe_rows) == len(universe) == 1096, "reconciliation is not 1096 unique rows")
    waves = defaultdict(list)
    for row in wave_rows:
        waves[row["wave"]].append(row)
    require({key: len(waves[key]) for key in ("Wave B", "Wave C", "Wave D", "Wave E", "Wave F")} == {
        "Wave B": 99, "Wave C": 66, "Wave D": 35, "Wave E": 249, "Wave F": 5
    }, "frozen wave cardinality mismatch")
    b_by_id = {row["universe_id"]: row for row in wave_b_rows}
    e_by_id = {row["universe_id"]: row for row in wave_e_rows}
    c_by_id = {row["universe_id"]: row for row in canonical_rows}
    jobs_by_id = {row["universe_id"]: row for row in jobs}
    require(len(b_by_id) == 99 and set(b_by_id) == {row["universe_id"] for row in waves["Wave B"]}, "Wave-B set mismatch")
    require(len(e_by_id) == 249 and set(e_by_id) == {row["universe_id"] for row in waves["Wave E"]}, "Wave-E set mismatch")
    require(len(c_by_id) == 642 and set(c_by_id) == {uid for uid, row in universe.items() if truth(row["canonical_valid"])}, "canonical impact set mismatch")
    require(len(jobs_by_id) == 99 and set(jobs_by_id) == set(b_by_id), "Wave-B job set mismatch")
    e_counts = Counter(row["resolved_classification"] for row in wave_e_rows)
    require(e_counts == Counter({"PACKAGE": 30, "CLEAN_RETRAIN": 17, "FEATURE_THEN_RETRAIN": 182, "PERMANENTLY_BLOCKED": 20}), f"Wave-E counts changed: {e_counts}")
    require(len(package_rows) == 30 and package_audit.get("status") == "PASS", "legacy package is not exact PASS")
    require(all(row["protocol_canonical_valid"] == "true" and row["analysis_eligible"] == "false" and row["scientifically_comparable"] == "false" for row in wave_e_rows if row["resolved_classification"] == "PACKAGE"), "legacy package eligibility contract mismatch")

    evidence_dir = OUT / "evidence"
    evidence_dir.mkdir(parents=True)
    evidence_sources = {
        "wave_b_reclassification_99.csv": args.wave_b,
        "wave_e_reclassification_249.csv": args.wave_e,
        "canonical_lineage_impact_642.csv": args.canonical_impact,
        "canonical_lineage_impact_summary.json": args.canonical_impact_summary,
        "wave_b_prefix_index_manifest_lineage_2208.csv": Path("/tmp/q00_wave_b_prefix_index_manifest_lineage_20260830.csv"),
        "wave_b_wan_p8_partial_reuse_6.csv": Path("/tmp/q00_waveb_wan_p8_exact_root_reuse_6.csv"),
        "wave_e_eval22_full_val_audit.json": Path("/tmp/q00_wavee_eval22_full_val_prefix_index_audit.json"),
        "wave_e_clean177_generic_audit.json": Path("/tmp/q00_wavee_clean177_generic_full_prefix_index_audit.json"),
        "wave_e_videomae_p12_target_audit.json": Path("/tmp/q00_wavee_videomae_p12_target_full_audit.json"),
    }
    evidence_hashes: dict[str, str] = {}
    for name, source in evidence_sources.items():
        require(source.is_file(), f"missing evidence file: {source}")
        shutil.copy2(source, evidence_dir / name)
        evidence_hashes[f"results/analysis/q00_1096_phase1/evidence/{name}"] = sha256(evidence_dir / name)

    wave_b_results: list[dict[str, Any]] = []
    atomic_count = 0
    for uid in sorted(b_by_id):
        audit = b_by_id[uid]
        job = jobs_by_id[uid]
        expected_outputs = json.loads(job["expected_outputs_json"])
        atomic_count += len(expected_outputs)
        is_canary = truth(job["is_canary"])
        failure = ROOT / job["output_directory"] / "eval/q00_failure.json"
        status_file = WAVE_B_BUNDLE / "launcher" / f"canary_{job['model']}_status.jsonl"
        wave_b_results.append({
            "universe_id": uid,
            "campaign": audit["campaign"],
            "model": audit["model"],
            "model_variant": audit["model_variant"],
            "protocol": audit["protocol"],
            "family": audit["family"],
            "probe": audit["probe"],
            "layer": audit["layer"],
            "prefix": audit["prefix"],
            "canary": str(is_canary).lower(),
            "execution_status": "CANARY_FAIL_PREFLIGHT_FEATURE_LINEAGE" if is_canary else "NOT_RUN_CANARY_GATE_FAILED",
            "evaluator_started": "false",
            "summary_present": str(Path(job["summary_path"]).is_file()).lower(),
            "prediction_present": str(Path(job["prediction_path"]).is_file()).lower(),
            "numeric_result_generated": "false",
            "promotion_allowed": "false",
            "expected_atomic_ids": ";".join(item["registry_id"] for item in expected_outputs),
            "expected_atomic_count": len(expected_outputs),
            "manifest_sha256": audit["manifest_sha256"],
            "checkpoint_sha256": job["checkpoint_sha256"],
            "checkpoint_reusable": audit["legacy_checkpoint_reusable_for_eval"],
            "generic_root_exact_val_n": audit["generic_root_exact_val_n"],
            "generic_root_mismatch_val_n": audit["generic_root_mismatch_val_n"],
            "generic_root_missing_key_n": audit["generic_root_missing_key_n"],
            "qualified_exact_streaming_root": audit["qualified_exact_streaming_root"],
            "qualified_exact_target_root": audit["qualified_exact_target_root"],
            "qualified_streaming_coverage": audit["qualified_streaming_coverage"],
            "qualified_target_required_horizon_coverage": audit["qualified_target_required_horizon_coverage"],
            "resolved_status": audit["status"],
            "resolved_bottleneck": audit["bottleneck"],
            "execution_class": audit["execution_class"],
            "failure_path": failure.relative_to(ROOT).as_posix() if failure.is_file() else "",
            "failure_sha256": sha256(failure) if failure.is_file() else "",
            "launcher_status_path": status_file.relative_to(ROOT).as_posix() if status_file.is_file() else "",
            "launcher_status_sha256": sha256(status_file) if status_file.is_file() else "",
            "next_action": audit["next_action"],
            "evidence_paths": audit["evidence"],
        })
    require(atomic_count == 105, f"expected 105 atomic IDs, found {atomic_count}")

    # Normalize Wave-E identity from the 1096 source and preserve the audit payload.
    wave_e_out: list[dict[str, Any]] = []
    for uid in sorted(e_by_id):
        audit = e_by_id[uid]
        base = universe[uid]
        wave_e_out.append({
            "universe_id": uid,
            "campaign": base["campaign"], "model": base["model"],
            "model_variant": base["model_variant"], "protocol": base["protocol"],
            "family": base["family"], "probe": base["probe"],
            "layer": base["layer"], "prefix": base["prefix"],
            "default_timestep": base["default_timestep"], "horizon": base["horizon"], "seed": base["seed"],
            **{key: value for key, value in audit.items() if key not in {"universe_id", "campaign", "model", "model_variant", "protocol", "family", "probe", "layer", "prefix"}},
        })

    # Build the live 1096 view without changing protected source-of-truth files.
    b_ids = set(b_by_id)
    c_ids = {row["universe_id"] for row in waves["Wave C"]}
    d_ids = {row["universe_id"] for row in waves["Wave D"]}
    package_ids = {row["universe_id"] for row in wave_e_rows if row["resolved_classification"] == "PACKAGE"}
    live_rows: list[dict[str, Any]] = []
    for original in universe_rows:
        row: dict[str, Any] = dict(original)
        uid = row["universe_id"]
        main_canonical = truth(row["canonical_valid"])
        numeric = truth(row["numeric_result_present"])
        phase_resolution = "UNCHANGED"
        status_after = row["status"]
        bottleneck_after = row["bottleneck"]
        protocol_canonical = main_canonical
        analysis_eligible = False
        scientifically_comparable = False
        lineage_class = "NOT_AUDITED_NONCANONICAL"
        next_action = row["next_action"]
        evidence = ""
        if uid in c_by_id:
            impact = c_by_id[uid]
            lineage_class = impact["impact_class"]
            analysis_eligible = truth(impact["strict_main_analysis_eligible"])
            scientifically_comparable = analysis_eligible
            phase_resolution = impact["result_disposition"]
            next_action = impact["recommended_action"]
            evidence = impact["evidence_paths"]
            if not analysis_eligible:
                bottleneck_after = "FEATURE_EXTRACTION" if truth(impact["error_proven"]) else "PROVENANCE"
        elif uid in b_ids or uid in c_ids or uid in d_ids:
            phase_resolution = "FEATURE_THEN_RETRAIN"
            status_after = "FEATURE_PARTIAL_OR_MISSING"
            bottleneck_after = "FEATURE_EXTRACTION"
            lineage_class = "GENERIC_SHARED_WINDOW_INVALID_OR_FEATURE_MISSING"
            if uid in b_by_id:
                next_action = b_by_id[uid]["next_action"]
                evidence = b_by_id[uid]["evidence"]
            elif uid in c_ids:
                next_action = "Create manifest-exact required streaming/target features, full audit, then clean no-resume train+eval; old checkpoint is forbidden."
                evidence = "reports/q00_phase1_direct_node_execution_log.md;reports/wan_p8_recovery/wan_p8_manifest_exact_recovery_20260821.md"
        elif uid in e_by_id:
            impact = e_by_id[uid]
            phase_resolution = impact["resolved_classification"]
            next_action = impact["next_action"]
            evidence = impact["evidence_paths"]
            lineage_class = impact["feature_lineage_status"]
            protocol_canonical = truth(impact["protocol_canonical_valid"])
            analysis_eligible = truth(impact["analysis_eligible"])
            scientifically_comparable = truth(impact["scientifically_comparable"])
            mapping = {
                "PACKAGE": ("RAW_RESULT_READY_TO_PACKAGE", "NONE"),
                "CLEAN_RETRAIN": ("FEATURE_READY_TRAIN_AND_EVAL", "TRAINING"),
                "FEATURE_THEN_RETRAIN": ("FEATURE_PARTIAL_OR_MISSING", "FEATURE_EXTRACTION"),
                "PERMANENTLY_BLOCKED": ("DEPRECATED_OR_PROTOCOL_INVALID", "PROTOCOL_INVALID"),
            }
            status_after, bottleneck_after = mapping[phase_resolution]
        row.update({
            "phase1_resolution": phase_resolution,
            "status_after_phase1": status_after,
            "bottleneck_after_phase1": bottleneck_after,
            "numeric_result_present_after": str(numeric).lower(),
            "main_canonical_valid_after": str(main_canonical).lower(),
            "protocol_canonical_valid_after": str(protocol_canonical).lower(),
            "scientifically_comparable_after": str(scientifically_comparable).lower(),
            "strict_main_analysis_eligible_after": str(analysis_eligible).lower(),
            "scientific_lineage_class": lineage_class,
            "phase1_next_action": next_action,
            "phase1_evidence_paths": evidence,
        })
        live_rows.append(row)

    numeric_before = sum(truth(row["numeric_result_present"]) for row in universe_rows)
    numeric_after = sum(truth(row["numeric_result_present_after"]) for row in live_rows)
    canonical_before = sum(truth(row["canonical_valid"]) for row in universe_rows)
    canonical_after = sum(truth(row["main_canonical_valid_after"]) for row in live_rows)
    protocol_canonical_after = sum(truth(row["protocol_canonical_valid_after"]) for row in live_rows)
    strict_eligible = sum(truth(row["strict_main_analysis_eligible_after"]) for row in live_rows)
    require((numeric_before, numeric_after, canonical_before, canonical_after, protocol_canonical_after, strict_eligible) == (833, 833, 642, 642, 672, 6), "phase counts mismatch")

    # Deterministic eventual training queue: confirmed-invalid canonical rows plus all
    # missing rows whose final disposition requires training. Unverified canonical rows
    # and Wave-F human-decision rows are deliberately excluded.
    queue_specs: dict[str, tuple[str, str, str]] = {}
    for uid, impact in c_by_id.items():
        if truth(impact["error_proven"]):
            queue_specs[uid] = ("CANONICAL_SCIENTIFIC_REMEDIATION", "FEATURE_THEN_RETRAIN", impact["recommended_action"])
    for uid in b_ids:
        queue_specs[uid] = ("MISSING_WAVE_B_RECLASSIFIED", "FEATURE_THEN_RETRAIN", b_by_id[uid]["next_action"])
    for uid in c_ids:
        queue_specs[uid] = ("MISSING_WAVE_C_RECLASSIFIED", "FEATURE_THEN_RETRAIN", "Build manifest-exact required feature roots; full audit; clean no-resume train+eval.")
    for uid in d_ids:
        queue_specs[uid] = ("MISSING_WAVE_D", "FEATURE_THEN_RETRAIN", universe[uid]["next_action"])
    for uid, impact in e_by_id.items():
        if impact["resolved_classification"] in {"CLEAN_RETRAIN", "FEATURE_THEN_RETRAIN"}:
            queue_specs[uid] = ("MISSING_WAVE_E_RESOLVED", impact["resolved_classification"], impact["next_action"])
    require(len(queue_specs) == 843, f"deterministic training queue must contain 843 rows, found {len(queue_specs)}")
    queue_rows: list[dict[str, Any]] = []
    for uid, (scope, execution_class, action) in queue_specs.items():
        base = universe[uid]
        impact = c_by_id.get(uid, {})
        eimpact = e_by_id.get(uid, {})
        manifest_path = base["train_manifest"] or base["manifest"]
        manifest_sha = base["train_manifest_sha256"] or base["manifest_sha256"]
        exact_root = eimpact.get("qualified_feature_root", "")
        if uid in b_by_id:
            exact_root = b_by_id[uid]["qualified_exact_streaming_root"]
        feature_group = hashlib.sha256(json.dumps({
            "campaign": base["campaign"], "model": base["model"], "prefix": base["prefix"],
            "layer": base["layer"], "timestep": base["default_timestep"],
            "manifest_sha256": manifest_sha, "protocol": base["protocol"],
        }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]
        queue_rows.append({
            "universe_id": uid,
            "source_scope": scope,
            "execution_class": execution_class,
            "campaign": base["campaign"], "model": base["model"],
            "model_variant": base["model_variant"], "protocol": base["protocol"],
            "family": base["family"], "probe": base["probe"],
            "layer": base["layer"], "prefix": base["prefix"],
            "default_timestep": base["default_timestep"], "horizon": base["horizon"], "seed": base["seed"],
            "manifest_path_or_recipe": manifest_path,
            "manifest_sha256": manifest_sha,
            "current_streaming_feature_root": base["streaming_feature_root"],
            "current_target_feature_root": base["target_feature_root"],
            "qualified_exact_feature_root": exact_root,
            "feature_dependency_group": feature_group,
            "legacy_checkpoint_forbidden": "true",
            "clean_no_resume": "true",
            "planned_feature_gate": "1" if execution_class == "FEATURE_THEN_RETRAIN" else "0",
            "planned_train_jobs": "1",
            "planned_eval_jobs": "1",
            "estimated_train_gpu_h_point": f"{PROBE_GPU_H.get(base['probe'], 15.0):.2f}",
            "estimated_eval_gpu_h_point": "0.05",
            "dispatch_eligible": "false",
            "queue_state": "FEATURE_REMEDIATION_REQUIRED" if execution_class == "FEATURE_THEN_RETRAIN" else "CLEAN_RETRAIN_BUNDLE_PREFLIGHT_REQUIRED",
            "required_gate": "immutable manifest/protocol; exact raw-ID feature coverage; finite/shape audit; clean final checkpoint; exact eval; expected-ID-only staging promotion",
            "next_action": action,
            "evidence_paths": impact.get("evidence_paths", eimpact.get("evidence_paths", base["evidence_paths"])),
        })
    # One operational canary candidate per campaign/model, preferring already-qualified
    # CLEAN_RETRAIN rows and non-action A1/A2 probes.
    family_rank = {"A1": 0, "A2": 1, "C2": 2, "C3": 3, "C1": 4, "B1": 5, "B2": 6}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in queue_rows:
        grouped[(row["campaign"], row["model"])].append(row)
    canary_ids: set[str] = set()
    for group in grouped.values():
        chosen = min(group, key=lambda row: (
            0 if row["execution_class"] == "CLEAN_RETRAIN" else 1,
            family_rank.get(row["family"], 9), int(row["prefix"]), abs(int(row["layer"])), row["universe_id"],
        ))
        canary_ids.add(chosen["universe_id"])
    for row in queue_rows:
        if row["universe_id"] in canary_ids:
            row["priority_tier"] = "A"
            row["priority_reason"] = "one mandatory train+eval canary for this campaign/model before any batch"
            row["canary_candidate"] = "true"
        elif row["source_scope"] in {"CANONICAL_SCIENTIFIC_REMEDIATION", "MISSING_WAVE_B_RECLASSIFIED", "MISSING_WAVE_C_RECLASSIFIED", "MISSING_WAVE_D"}:
            row["priority_tier"] = "B"
            row["priority_reason"] = "restores current scientific conclusions or critical model×task×layer×prefix coverage"
            row["canary_candidate"] = "false"
        else:
            row["priority_tier"] = "C"
            row["priority_reason"] = "historical provisional/table completion after higher-impact coverage"
            row["canary_candidate"] = "false"
    queue_rows.sort(key=lambda row: (row["priority_tier"], row["campaign"], row["model"], row["family"], row["probe"], int(row["prefix"]), int(row["layer"]), row["universe_id"]))
    for index, row in enumerate(queue_rows, 1):
        row["queue_order"] = index
        row["queue_id"] = f"q00_phase1_train_{index:04d}"

    write_csv(OUT / "wave_b_eval_results.csv", wave_b_results)
    write_csv(OUT / "wave_e_resolution.csv", wave_e_out)
    write_csv(OUT / "live_universe_1096.csv", live_rows)
    write_csv(OUT / "next_training_queue.csv", queue_rows)

    protected_after = {path: sha256(ROOT / path) for path in PROTECTED_EXPECTED}
    require(protected_after == protected_before, "protected files changed while building snapshot")
    train_point = sum(float(row["estimated_train_gpu_h_point"]) for row in queue_rows)
    eval_point = sum(float(row["estimated_eval_gpu_h_point"]) for row in queue_rows)
    completion_rows = [
        row for row in queue_rows
        if row["source_scope"] != "CANONICAL_SCIENTIFIC_REMEDIATION"
    ]
    completion_numeric_absent = sum(
        not truth(universe[row["universe_id"]]["numeric_result_present"])
        for row in completion_rows
    )
    completion_noncanonical_numeric = len(completion_rows) - completion_numeric_absent
    require(
        (len(completion_rows), completion_numeric_absent, completion_noncanonical_numeric)
        == (399, 243, 156),
        "missing/noncanonical completion split mismatch",
    )
    summary = {
        "generated_at_utc": now(),
        "universe": {"rows": 1096, "path": UNIVERSE.relative_to(ROOT).as_posix(), "sha256": EXPECTED_HASHES["universe"]},
        "counts": {
            "numeric_results_before": 833, "numeric_results_after": 833,
            "main_canonical_parents_before": 642, "main_canonical_parents_after": 642,
            "protocol_canonical_parents_after": 672,
            "strict_main_analysis_eligible": 6,
            "canonical_confirmed_lineage_error": 444,
            "canonical_lineage_unverified": 192,
            "legacy_protocol_packaged": 30,
            "wave_b_canary_pass": False, "wave_b_success": 0,
            "wave_e": dict(sorted(e_counts.items())),
            "next_training_job_count": 843,
            "missing_or_noncanonical_completion_parents": 399,
            "completion_numeric_absent_parents": completion_numeric_absent,
            "completion_provisional_or_protocol_unfrozen_numeric_parents": completion_noncanonical_numeric,
            "canonical_scientific_remediation_parents": 444,
            "feature_then_retrain": sum(row["execution_class"] == "FEATURE_THEN_RETRAIN" for row in queue_rows),
            "clean_retrain": sum(row["execution_class"] == "CLEAN_RETRAIN" for row in queue_rows),
            "canonical_unverified_audit_hold": 192,
            "wave_f_human_decision_hold": 5,
        },
        "compute_planning": {
            "training_gpu_h_point_using_historical_probe_medians": round(train_point, 2),
            "eval_gpu_h_point": round(eval_point, 2),
            "feature_extraction_gpu_h": "NOT_YET_BOUNDED; estimate by unique cache group after canary",
            "warning": "Old Q00 cost is invalid because 826 rows now require feature remediation before training.",
        },
        "wave_b": {
            "bundle": WAVE_B_BUNDLE.relative_to(ROOT).as_posix(),
            "bundle_sha256": sha256(WAVE_B_BUNDLE / "bundle.json"),
            "parents": 99, "expected_atomic_ids": 105, "summaries": 0,
            "remaining_96_executed": False, "promotion": False,
        },
        "legacy_package": {
            "path": LEGACY_PACKAGE.relative_to(ROOT).as_posix(),
            "parents": 30, "protocol_canonical_valid": True,
            "scientifically_comparable": False, "analysis_eligible": False,
            "audit_sha256": sha256(LEGACY_PACKAGE / "audit.json"),
        },
        "canonical_lineage_audit": canonical_summary,
        "protected_files": {"status": "UNCHANGED", "before": protected_before, "after": protected_after},
        "evidence_hashes": evidence_hashes,
        "stop_boundary": {
            "eval_jobs_completed": 0, "training_jobs_started": 0, "feature_extraction_jobs_started": 0,
            "reason": "Direct canaries proved a systematic temporal-window lineage collision; all downstream writers remained locked.",
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Q00 Phase 1：Wave B direct-node 执行与 Wave E lineage resolution",
        "",
        f"生成时间：`{summary['generated_at_utc']}`。本报告冻结 1096 universe，不含 diffusion-step sweep。",
        "",
        "## 结论",
        "",
        "Wave B 没有产生新结果。三个 direct-node canary 都在 evaluator 启动前发现 canonical h124 manifest 与 generic feature cache 指向不同 raw-frame window，严格 fail closed；剩余 96 个未运行，promotion 未发生。有限数值仍为 **833/1096**，主表 canonical parent 仍为 **642/1096**。",
        "",
        "这次失败揭示的是历史训练 lineage 问题，而不是临时执行错误：generic cache 使用共享 24-frame observation window，h124 manifest 分别使用 12/16/20/28-frame window；旧 loader 只以 `(window_id, tail)` 命中，不校验 raw IDs。99 个 Wave-B checkpoint 已在混合错配上训练，不能通过重新 eval 修复。",
        "",
        "Wave E 最终为 PACKAGE=30、EVAL_ONLY=0、CLEAN_RETRAIN=17、FEATURE_THEN_RETRAIN=182、PERMANENTLY_BLOCKED=20。30 个 legacy B2 已进入独立 namespace，但只表示 artifact-complete protocol package；由于 BY_PREFIX raw-frame lineage 未 run-bound，它们不进入严格全局分析。",
        "",
        "更重要的是，642 个现有 canonical parents 的表级 canonical-valid 状态没有被改写，但科学 lineage 审计只确认 **6** 个 Wan p8 action_dynamics rows 使用 manifest-exact versioned cache；444 个确认受 shared24↔h124 错配影响，另 192 个 BY_PREFIX 缺 raw-ID 证明。严格主分析因此暂时只能使用 6 个 exact-safe parent，其余 636 个 quarantine。Wan p8 的六行还受既有 action-validity construct audit 限制，只能作为 weak-signal case study，不能支撑全局空间结论。",
        "",
        "## Direct-node 执行证据",
        "",
        f"- Active successor bundle：`{summary['wave_b']['bundle']}`，SHA256 `{summary['wave_b']['bundle_sha256']}`。",
        "- 旧 SCO writers 在 direct execution 前均确认未启动并取消；direct host 为 8×H800 节点。",
        "- 三个 canary 的 checkpoint、manifest SHA、CUDA/runtime 和 tensor shape gate 通过；full prefix-index identity gate 分别在 LingBot、MiniMax-H3、Wan 上拒绝。",
        "- 2,208 个 val associations。按 prefix 聚合：p8 240/312、p12 276/276、p16 348/204、p24 342/210（exact/mismatch）；对应的每个 model×prefix group 模式分别为 120/156、138/138、174/102、171/105，8/8 groups 均非 exact。",
        "- evaluator started=0，summary=0，prediction=0，新增 numeric=0，canonical overwrite=0。",
        "",
        "## Wave B 99",
        "",
        "99/99 重分为 `FEATURE_PARTIAL_OR_MISSING / FEATURE_EXTRACTION / FEATURE_THEN_RETRAIN`。Wan p8 path_integration 六行可以复用 exact streaming 与 h1 target，但 h2/h4 target 覆盖为 0/2724；它们仍须补 target 后 clean retrain。其余 93 行没有 qualified full exact root。逐行见 `wave_b_eval_results.csv`。",
        "",
        "## Wave E 249",
        "",
        "- 30 PACKAGE：已写入 `results/canonical_protocols/legacy_b2_shared_hidden_8_12_16_24_val_v1/`；30 protocol IDs、150 artifact hashes、指标重算和 prefix 内 label identity 均 PASS。`protocol_canonical_valid=true`，但 `scientifically_comparable=false`、`analysis_eligible=false`。",
        "- 0 EVAL_ONLY：旧 22 行全部撤销；V-JEPA/Cog/Wan family canary 均复现 lineage mismatch。",
        "- 17 CLEAN_RETRAIN：均为 Wan p8 streaming-only rows，可用 manifest-exact versioned root；仍需新的 frozen train bundle 与每 campaign/model canary。",
        "- 182 FEATURE_THEN_RETRAIN：包含原 177 clean 中的 162 行、旧 eval-only 中的 20 行；F3R/Qwen BY_PREFIX 也因缺 run-bound raw-ID 证明 fail closed。",
        "- 20 PERMANENTLY_BLOCKED：MiniMax standalone camera contract 不得由 joint depth+camera 静默替代。",
        "",
        "## Canonical lineage impact",
        "",
        "| 分类 | Parents | 处理 |",
        "|---|---:|---|",
        "| exact/versioned Dataset-canary safe | 6 | 严格分析暂可保留，但需 Wan action-validity construct caveat |",
        "| confirmed generic shared24 vs h124 error | 444 | quarantine；manifest-exact feature + clean retrain + eval |",
        "| BYPREFIX raw-ID unverified | 192 | quarantine；先历史 index/provenance audit，不能直接判 safe |",
        "",
        "主 canonical 表仍是 642 parents；独立 legacy namespace 使 artifact-level protocol-canonical union 为 672，但不增加 strict analysis eligibility。",
        "",
        "## 下一训练队列",
        "",
        f"`next_training_queue.csv` 包含 **843** 个确定性 replacement jobs：444 个 confirmed-invalid canonical remediation + 399 个 missing-or-noncanonical completion。后 399 个中，243 个没有任何数值，156 个已有 provisional 或 protocol-unfrozen 数值但不能直接 canonicalize。执行类为 FEATURE_THEN_RETRAIN={summary['counts']['feature_then_retrain']}、CLEAN_RETRAIN={summary['counts']['clean_retrain']}。另外 192 个 canonical BYPREFIX 留在 lineage audit hold，5 个 Wave-F rows 留在人为 protocol decision hold。",
        "",
        "所有 843 行当前 `dispatch_eligible=false`。Priority A 只选每个 campaign/model 一个 train+eval canary；canary 必须先有 immutable exact feature/manifest bundle。历史 probe median 粗略训练点估为 " + f"**{train_point:.2f} GPU-h**，eval 点估 **{eval_point:.2f} GPU-h**；826 行还需 feature remediation，feature GPU-hours/磁盘必须按 unique cache group canary 后报价，旧 Q00 总成本已失效。",
        "",
        "## Protected files",
        "",
        "Registry 与七张 main canonical CSV 的 before/after SHA256 完全相同。唯一新增的 result namespace 是独立 legacy package；没有改写已有 canonical 行。",
        "",
        "## 验收",
        "",
        "```text",
        "WAVE_B_CANARY_PASS: false",
        "WAVE_B_SUCCESS: 0/99",
        "NUMERIC_RESULTS_BEFORE: 833",
        "NUMERIC_RESULTS_AFTER: 833",
        "CANONICAL_PARENTS_BEFORE: 642",
        "CANONICAL_PARENTS_AFTER: 642",
        "PROTOCOL_CANONICAL_PARENTS_AFTER: 672",
        "STRICT_MAIN_ANALYSIS_ELIGIBLE: 6",
        "CANONICAL_CONFIRMED_LINEAGE_ERROR: 444",
        "CANONICAL_LINEAGE_UNVERIFIED: 192",
        "B2_PACKAGEABLE: 30",
        "B2_NEEDS_REEVAL: 0",
        "WAVE_E_EVAL_ONLY: 0",
        "WAVE_E_CLEAN_RETRAIN: 17",
        "WAVE_E_FEATURE_THEN_RETRAIN: 182",
        "WAVE_E_PERMANENTLY_BLOCKED: 20",
        "NEXT_TRAINING_JOB_COUNT: 843",
        "PROTECTED_FILES_UNCHANGED_OR_EXPECTED_DIFF_ONLY: true (UNCHANGED)",
        "```",
        "",
        "## Stop point",
        "",
        "本轮在 fail-closed classification、legacy packaging 和确定性 queue 处停止：没有启动 remaining-96、Wave-E eval、训练或 feature extraction。下一步只可从 queue 的 Priority-A canary 开始，并先完成 exact feature namespace。diffusion-step、planning、prediction-feature 和新增 baseline 继续暂停。",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    protected_final = {path: sha256(ROOT / path) for path in PROTECTED_EXPECTED}
    require(protected_final == protected_before, "protected files changed after report write")
    print(json.dumps({
        "status": "PASS", "universe": 1096, "wave_b_success": 0,
        "wave_e": dict(e_counts), "main_canonical": 642,
        "strict_analysis_eligible": 6, "training_queue": 843,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
