#!/usr/bin/env python3
"""Validate and summarize the immutable P16 repair evidence bundle."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path("/mnt/umm/users/yinbaiqiao/probe_spatial")
RUN = ROOT / "analyses/probe_validity/evidence/p16_repair_20260917T070619Z_20277f5"
RUN_ID = RUN.name
MODELS = ("vjepa2", "videomae", "wan", "cogvideox")
JOBS = {
    "c1_vjepa2": ("pt-bzt2sjqs", "SUCCEEDED", "2026-09-17T07:10:23Z", "2026-09-17T07:16:13Z"),
    "c1_videomae": ("pt-euc45eai", "SUCCEEDED", "2026-09-17T07:17:02Z", "2026-09-17T07:22:17Z"),
    "c1_wan": ("pt-uq1trgbx", "SUCCEEDED", "2026-09-17T07:23:26Z", "2026-09-17T07:30:16Z"),
    "c1_cogvideox": ("pt-ptlbfp04", "SUCCEEDED", "2026-09-17T10:34:09Z", "2026-09-17T10:43:24Z"),
    "b2_recovery": ("pt-lxwdtkic", "SUCCEEDED", "2026-09-17T10:33:57Z", "2026-09-17T10:36:32Z"),
}
START_HASHES = {
    "experiments/task_status_1096.csv": "92b86c63c2fcc8062b36a6e44dd27a6acf5c02e5699ee591f9042d7d87bcb899",
    "experiments/registry.csv": "efc533bbe6286090d6c111c9d272ffb28cf7477b937f799f312eedf2a944431e",
    "results/paper_ready_results.csv": "212174a7473c064f1258a10994b6e01e3621cae164573a63c78dd01da73793d4",
    "results/atomic_probe_coverage.csv": "49d990bbed7ee99a9f7e1930548e33b7e18aaae2155712bbcab3e7df2c38340d",
    "analyses/probe_validity/c1_validity.csv": "d7f5ccf42631e27749728c87d41b2ce76a15c2142cecba90e6ad5439bf3e7388",
    "analyses/probe_validity/b2_validity.csv": "c970237270be3714195aa4981704df2eedb78f45b8a7c8861b8821c73ba7ffd8",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    receipts = {}
    for task in ("c1", "b2"):
        for model in MODELS:
            path = RUN / "cells" / f"{task}_{model}" / "receipt.json"
            if path.exists():
                receipts[f"{task}_{model}"] = json.loads(path.read_text())
    if len(receipts) != 8:
        raise RuntimeError(f"expected 8 receipts, found {sorted(receipts)}")

    # Revalidate every artifact written by each successful atomic cell.
    for model in MODELS:
        cell = RUN / "cells" / f"c1_{model}"
        receipt = receipts[f"c1_{model}"]
        if receipt["status"] != "PASS" or receipt["replay_gate"]["status"] != "PASS":
            raise RuntimeError(f"non-PASS C1 receipt: {model}")
        for name, expected in receipt["artifacts"].items():
            if sha256(cell / name) != expected["sha256"]:
                raise RuntimeError(f"artifact hash drift: {model}/{name}")

    # Candidate IDs and wrong-action mappings must be shared across models.
    mapping_hashes = {}
    for model in MODELS:
        with gzip.open(
            RUN / "cells" / f"c1_{model}" / "candidate_sample_diagnostics.csv.gz",
            "rt", encoding="utf-8", newline="",
        ) as handle:
            rows = list(csv.DictReader(handle))
        projection = [{
            key: row[key] for key in (
                "sample_id", "candidate_count", "candidate_indices",
                "candidate_sample_ids", "matched_wrong_index",
                "matched_wrong_source_sample_id",
            )
        } for row in rows]
        if len(rows) != 276 or {int(row["candidate_count"]) for row in rows} != {8}:
            raise RuntimeError(f"invalid hard-pool cardinality for {model}")
        mapping_hashes[model] = hashlib.sha256(
            json.dumps(projection, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    if len(set(mapping_hashes.values())) != 1:
        raise RuntimeError(f"cross-model candidate mapping drift: {mapping_hashes}")

    old_c1 = list(csv.DictReader((ROOT / "analyses/probe_validity/c1_validity.csv").open()))
    p12_global = {}
    p12_hard = {}
    for model in MODELS:
        global_rows = [row for row in old_c1 if (
            row["model"] == model and row["prefix"] == "12"
            and row["candidate_pool"] == "HISTORICAL_GLOBAL"
            and row["action_condition"] == "true"
        )]
        hard_rows = [row for row in old_c1 if (
            row["model"] == model and row["prefix"] == "12"
            and row["analysis_type"] == "paper_candidate_k_robustness"
            and row["candidate_pool"] == "MATCHED_TOTAL_POSE_DISPLACEMENT"
            and row["desired_candidate_count"] == "8"
            and row["candidate_seed"] == "42" and row["horizon"] == "1"
        )]
        if len(global_rows) != 1 or len(hard_rows) != 1:
            raise RuntimeError(f"ambiguous P12 comparator rows for {model}")
        p12_global[model], p12_hard[model] = global_rows[0], hard_rows[0]

    c1_rows = []
    for model in MODELS:
        receipt = receipts[f"c1_{model}"]
        aggregate = json.loads(
            (RUN / "cells" / f"c1_{model}" / "aggregates.json").read_text()
        )
        global_stats = aggregate["global"]
        hard = aggregate["matched_total_pose_k8_seed42"]
        high = aggregate["high_absolute_motion"]
        p12g, p12k = p12_global[model], p12_hard[model]
        c1_rows.append({
            "repair_run_id": RUN_ID,
            "result_origin": "P16_VERIFIED_REPAIR",
            "model": model,
            "layer": receipt["layer"], "prefix": 16, "horizon": 1,
            "checkpoint_sha256": receipt["checkpoint"]["sha256"],
            "execution_snapshot": "source_snapshot",
            "baseline_gate_status": receipt["replay_gate"]["status"],
            "baseline_max_absolute_difference": receipt["replay_gate"]["max_absolute_difference"],
            "p16_global_full_r1": global_stats["FULL"]["global_R@1"],
            "p16_global_copy_r1": global_stats["COPY"]["global_R@1"],
            "p16_global_zero_r1": global_stats["ZERO_ACTION"]["global_R@1"],
            "p16_global_shuffle_r1": global_stats["SHUFFLED_ACTION"]["global_R@1"],
            "p16_global_matched_wrong_r1": global_stats["MATCHED_WRONG"]["global_R@1"],
            "p16_global_true_minus_matched_r1": (
                global_stats["FULL"]["global_R@1"] - global_stats["MATCHED_WRONG"]["global_R@1"]
            ),
            "p16_hard_k8_full_r1": hard["FULL"]["r1"],
            "p16_hard_k8_matched_wrong_r1": hard["MATCHED_WRONG"]["r1"],
            "p16_hard_k8_gain_r1": hard["full_minus_matched_wrong"]["r1"],
            "p16_hard_k8_gain_ci_low": hard["full_minus_matched_wrong"]["r1_scene_bootstrap_95ci"][0],
            "p16_hard_k8_gain_ci_high": hard["full_minus_matched_wrong"]["r1_scene_bootstrap_95ci"][1],
            "p16_hard_k8_cosine_margin": hard["full_minus_matched_wrong"]["cosine"],
            "p16_high_motion_n": high["n"],
            "p16_high_motion_gain_r1": high["full_minus_matched_wrong"]["r1"],
            "p16_high_motion_ci_low": high["full_minus_matched_wrong"]["r1_scene_bootstrap_95ci"][0],
            "p16_high_motion_ci_high": high["full_minus_matched_wrong"]["r1_scene_bootstrap_95ci"][1],
            "p12_global_full_r1": p12g["r1"],
            "p12_global_copy_r1": p12g["copy_r1"],
            "p12_global_true_minus_matched_r1": p12g["true_minus_matched"],
            "p12_hard_k8_gain_r1_seed42": p12k["true_minus_matched"],
            "p12_hard_k8_ci_low_seed42": p12k["ci_low"],
            "p12_hard_k8_ci_high_seed42": p12k["ci_high"],
            "candidate_mapping_sha256": mapping_hashes[model],
            "evidence_path": f"cells/c1_{model}",
        })
    write_csv(RUN / "c1_verified_repair_results.csv", c1_rows)

    b2_reference_paths = {
        "vjepa2": ROOT / "logs/inscene15k_streaming/runs/inscene15k_streaming_b2_vjepa2_p16_layer23/eval/val_summary.json",
        "videomae": ROOT / "logs/inscene15k_streaming/runs/inscene15k_streaming_b2_videomae_p16_layer23/eval/val_summary.json",
        "wan": ROOT / "logs/inscene15k_streaming/runs/inscene15k_streaming_b2_wan_p16_layer20_heavy64_20260716_021635/eval/val_summary.json",
        "cogvideox": ROOT / "logs/inscene15k_streaming/runs/inscene15k_streaming_b2_cogvideox_p16_layer20_heavy64_20260716_021635/eval/val_summary.json",
    }
    old_b2 = list(csv.DictReader((ROOT / "analyses/probe_validity/b2_validity.csv").open()))
    cog_methods = {
        row["baseline_or_readout"]: row for row in old_b2
        if row["model"] == "cogvideox" and row["prefix"] == "16"
        and row["analysis_type"] == "paper_prior_or_order_control"
    }
    b2_rows = []
    for model in MODELS:
        receipt = receipts[f"b2_{model}"]
        reference = json.loads(b2_reference_paths[model].read_text())
        row = {
            "repair_run_id": RUN_ID, "model": model, "prefix": 16,
            "status": receipt["status"],
            "result_origin": (
                "EXISTING_NO_CONFLICT_RESULT" if model == "cogvideox" else "NO_RESULT_CHECKPOINT_UNAVAILABLE"
            ),
            "execution_snapshot": "source_snapshot",
            "checkpoint_sha256": "",
            "baseline_gate_status": receipt["baseline_gate_status"],
            "reference_full_n": reference["n"],
            "reference_full_mean_ang_err_deg": reference["mean_ang_err_deg"],
            "reference_full_top1": reference["top1"],
            "reference_full_top3": reference["top3"],
            "new_full_mean_ang_err_deg": "",
            "query_mask_delta_deg": "", "query_swap_delta_deg": "",
            "current_frame_delta_deg": "", "remove_visible_delta_deg": "",
            "content_shuffle_delta_deg": "", "content_reverse_delta_deg": "",
            "best_reusable_prior": "", "best_reusable_prior_delta_vs_full_deg": "",
            "evidence_level": (
                "SAME_BINDING_AND_RESULT_FINGERPRINT_SHA_UNAVAILABLE"
                if model == "cogvideox" else "BLOCKED_CHECKPOINT_UNAVAILABLE"
            ),
            "evidence_path": f"cells/b2_{model}/receipt.json",
        }
        if model == "cogvideox":
            row.update({
                "new_full_mean_ang_err_deg": cog_methods["FULL_HISTORY_PLUS_QUERY"]["mean_ang_err_deg"],
                "query_mask_delta_deg": cog_methods["HISTORY_WITHOUT_QUERY"]["delta_error_vs_full"],
                "query_swap_delta_deg": cog_methods["SAME_SCENE_QUERY_SWAP"]["delta_error_vs_full"],
                "current_frame_delta_deg": cog_methods["CURRENT_FRAME_PLUS_QUERY"]["delta_error_vs_full"],
                "remove_visible_delta_deg": cog_methods["REMOVE_TARGET_VISIBLE_FRAMES"]["delta_error_vs_full"],
                "content_shuffle_delta_deg": cog_methods["CONTENT_ONLY_SHUFFLE"]["delta_error_vs_full"],
                "content_reverse_delta_deg": cog_methods["REVERSE_CONTENT_ONLY"]["delta_error_vs_full"],
                "best_reusable_prior": "GLOBAL_CONSTANT",
                "best_reusable_prior_delta_vs_full_deg": cog_methods["GLOBAL_CONSTANT"]["delta_error_vs_full"],
            })
        b2_rows.append(row)
    write_csv(RUN / "b2_repair_status_results.csv", b2_rows)

    cell_rows = []
    for model in MODELS:
        receipt = receipts[f"c1_{model}"]
        job_id, scheduler, start, end = JOBS[f"c1_{model}"]
        cell_rows.append({
            "cell": f"{model}/C1", "formal_weight_identity": Path(receipt["checkpoint"]["path"]).parts[-3],
            "checkpoint_sha256": receipt["checkpoint"]["sha256"], "zoetrope_job_id": job_id,
            "scheduler_status": scheduler, "start_time": start, "complete_time": end,
            "baseline_gate": "PASS", "validity_eval": "PASS",
            "result_path": f"cells/c1_{model}", "current_status": "VERIFIED_REPAIR_COMPLETE",
        })
    for model in ("vjepa2", "videomae", "wan"):
        receipt = receipts[f"b2_{model}"]
        cell_rows.append({
            "cell": f"{model}/B2", "formal_weight_identity": Path(receipt["formal_checkpoint"]).parts[-3],
            "checkpoint_sha256": "", "zoetrope_job_id": "pt-lxwdtkic (recovery audit; no eval)",
            "scheduler_status": "SUCCEEDED", "start_time": JOBS["b2_recovery"][2],
            "complete_time": JOBS["b2_recovery"][3],
            "baseline_gate": "NOT_RUN", "validity_eval": "NOT_RUN",
            "result_path": f"cells/b2_{model}/receipt.json",
            "current_status": "BLOCKED_CHECKPOINT_UNAVAILABLE",
        })
    cell_rows.append({
        "cell": "cogvideox/B2", "formal_weight_identity": "wlqpdq8j",
        "checkpoint_sha256": "", "zoetrope_job_id": "NO_NEW_JOB",
        "scheduler_status": "NOT_APPLICABLE", "start_time": "", "complete_time": "",
        "baseline_gate": "NO_NEW_REPLAY_REQUIRED", "validity_eval": "EXISTING_RESULT_RETAINED",
        "result_path": "cells/b2_cogvideox/receipt.json",
        "current_status": "NO_CONFLICT_EXISTING_RESULT",
    })
    write_csv(RUN / "cell_status.csv", cell_rows)

    protected = {}
    for relative, expected in START_HASHES.items():
        observed = sha256(ROOT / relative)
        protected[relative] = {
            "start_sha256": expected, "end_sha256": observed,
            "byte_identical": observed == expected,
        }
    if not all(item["byte_identical"] for item in protected.values()):
        raise RuntimeError(f"protected input drift: {protected}")
    (RUN / "protected_hashes_end.json").write_text(
        json.dumps(protected, indent=2, sort_keys=True) + "\n"
    )
    summary = {
        "repair_run_id": RUN_ID,
        "status_counts": {
            "VERIFIED_REPAIR_COMPLETE": 4,
            "BLOCKED_CHECKPOINT_UNAVAILABLE": 3,
            "REPLAY_MISMATCH": 0,
            "NO_CONFLICT_EXISTING_RESULT": 1,
        },
        "counts": {
            "configured_cells": 8, "evaluation_attempts": 4,
            "recovery_attempts": 1, "total_attempts": 5,
            "evaluation_jobs": 4, "recovery_jobs": 1, "total_jobs": 5,
            "maximum_observed_concurrent_single_gpu_jobs": 2,
        },
        "candidate_mapping_sha256": next(iter(mapping_hashes.values())),
        "candidate_mapping_identical_across_four_models": True,
        "protected_inputs_byte_identical": True,
        "limitations": [
            "P12 and P16 target/query universes are not strictly paired.",
            "This is an evaluation-only repair, not an independent frozen confirmation.",
            "All repaired C1 results use one historical probe seed.",
            "Blocked B2 cells were not replaced with partial or neighboring checkpoints.",
        ],
    }
    (RUN / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    files = [path for path in RUN.rglob("*") if path.is_file() and path.name != "artifact_manifest.sha256"]
    with (RUN / "artifact_manifest.sha256").open("w", encoding="utf-8") as handle:
        for path in sorted(files):
            handle.write(f"{sha256(path)}  {path.relative_to(RUN)}\n")


if __name__ == "__main__":
    main()
