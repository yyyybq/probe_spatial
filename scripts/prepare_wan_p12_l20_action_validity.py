#!/usr/bin/env python3
"""Freeze inputs and prepare formal permutations for Wan P12/L20 validity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vidfm3d.utils.action_validity import (
    action_descriptors,
    aggregate_retrieval,
    frequency_blocks,
    make_block_derangement,
    retrieval_per_sample,
    tensor_sha256,
    validate_derangement,
)


RUN_TAG = "wan_p12_l20_action_validity_20260830"
MANIFEST_REL = Path("manifests/c_retrain_20260722/inscene_c_h124_prefix12.jsonl")
MANIFEST_SHA256 = "0ae3f4dbe26ddc2ba6b936b7a5bde5353a8d912da304287533cff6182d93851b"
CHECKPOINT_REL = Path(
    "logs/inscene15k_streaming/runs/c_retrain_safe_20260722_1435/"
    "c_retrain_safe_20260722_1435_action_dynamics_wan_p12_layer20/"
    "VideoProbe3D/6zf0wf3f/checkpoints/epoch=49-step=30600.ckpt"
)
CHECKPOINT_SHA256 = "87523ab72c43a3527eaaf45370e6f070e29c99d0122079cb56e443254d243567"
EVAL_REL = Path(
    "logs/inscene15k_streaming/runs/c_eval_20260724_1903/"
    "c_eval_20260724_1903_action_dynamics_wan_p12_layer20/eval"
)
PREDICTIONS_SHA256 = "ebd1dacfff2a2d6bd76d3c1de5addb0dda32548e3572867daff0036635f51529"
SUMMARY_SHA256 = "3060d0eeabf056623d9ffcbe2782b36d5e121c055441408d85fc37d6616337da"
PROTECTED = (
    "experiments/task_status_1096.csv",
    "experiments/registry.csv",
    "results/paper_ready_results.csv",
    "results/atomic_probe_coverage.csv",
    "results/canonical/a1_results.csv",
    "results/canonical/a2_results.csv",
    "results/canonical/all_results.csv",
    "results/canonical/b2_results.csv",
    "results/canonical/c1_results.csv",
    "results/canonical/c2_results.csv",
    "results/canonical/c3_results.csv",
    "metadata/probe_taxonomy.csv",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def condition_metrics(records: list[Dict[str, Any]], key: str, targets: torch.Tensor) -> Dict[str, float]:
    predictions = torch.stack([record[key].float() for record in records])
    return aggregate_retrieval(retrieval_per_sample(predictions, targets))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/analysis/wan_p12_l20_action_validity"),
    )
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=2026083000)
    args = parser.parse_args()

    root = args.project_root.resolve()
    output = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    manifest = root / MANIFEST_REL
    checkpoint = root / CHECKPOINT_REL
    predictions_path = root / EVAL_REL / "val_predictions.pt"
    summary_path = root / EVAL_REL / "val_summary.json"
    expected_hashes = {
        manifest: MANIFEST_SHA256,
        checkpoint: CHECKPOINT_SHA256,
        predictions_path: PREDICTIONS_SHA256,
        summary_path: SUMMARY_SHA256,
    }
    for path, expected in expected_hashes.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"frozen input hash changed: {path} expected={expected} actual={actual}")

    manifest_rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    val_manifest = [row for row in manifest_rows if row.get("split") == "val"]
    if len(manifest_rows) != 2724 or len(val_manifest) != 276:
        raise RuntimeError("P12 manifest counts changed")

    records = torch.load(predictions_path, map_location="cpu", weights_only=False)
    if len(records) != 276 or not all(bool(record.get("valid")) for record in records):
        raise RuntimeError("frozen prediction inventory is invalid")
    sample_ids = [str(row["sample_id"]) for row in val_manifest]
    manifest_scenes = [str(row["scene_dir"]) for row in val_manifest]
    record_scenes = [str(record["scene_path"]) for record in records]
    if manifest_scenes != record_scenes:
        raise RuntimeError("manifest and frozen prediction scene ordering differ")
    if len(set(sample_ids)) != 276 or len(set(record_scenes)) != 69:
        raise RuntimeError("unexpected P12 val sample/scene cardinality")
    if any(count != 4 for count in Counter(record_scenes).values()):
        raise RuntimeError("expected exactly four validation windows per scene")

    actions = torch.stack([record["action"].float() for record in records])
    targets = torch.stack([record["target"].float() for record in records])
    correct = condition_metrics(records, "pred", targets)
    null = condition_metrics(records, "pred_no_action", targets)
    shuffled = condition_metrics(records, "pred_shuffled_action", targets)
    copy = condition_metrics(records, "last_observation", targets)
    historical = json.loads(summary_path.read_text())
    comparisons: Dict[str, float] = {}
    for prefix, metrics in (
        ("", correct),
        ("no_action_", null),
        ("shuffled_action_", shuffled),
        ("last_observation_", copy),
    ):
        for metric, summary_key in (
            ("cos", "mean_cos"),
            ("r1", "global_R@1"),
            ("r5", "global_R@5"),
            ("rank0", "mean_rank"),
        ):
            key = prefix + summary_key
            comparisons[key] = abs(metrics[metric] - float(historical[key]))
    if max(comparisons.values()) > 1e-6:
        raise RuntimeError(f"frozen evaluator metric cross-check failed: {comparisons}")

    predictions = torch.stack([record["pred"].float() for record in records])
    no_action = torch.stack([record["pred_no_action"].float() for record in records])
    delta = predictions - no_action
    baseline = {
        "layer": 20,
        "prefix": 12,
        "correct": correct,
        "eval_null_action": null,
        "historical_batch_roll": shuffled,
        "copy_last_streaming_state": copy,
        "metric_crosscheck_abs_diff": comparisons,
        "actual_checkpoint_action_sensitivity": {
            "mean_l2_pred_minus_null": float(delta.norm(dim=1).mean()),
            "median_l2_pred_minus_null": float(delta.norm(dim=1).median()),
            "max_abs_pred_minus_null": float(delta.abs().max()),
            "fraction_changed_gt_1e-6": float((delta.norm(dim=1) > 1e-6).float().mean()),
        },
        "anchor_isolated_available": int(
            sum(bool(record.get("anchor_isolated_valid")) for record in records)
        ),
    }

    descriptor, labels, axes = action_descriptors(actions.numpy())
    blocks = frequency_blocks(descriptor, n_blocks=10)
    class_counts = dict(sorted(Counter(labels.tolist()).items()))
    inventory_path = output / "tables/action_inventory.csv"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    with inventory_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_index",
                "sample_id",
                "scene_path",
                "target_frame_idx",
                "frequency_block",
                "action_class",
            ]
            + [f"descriptor_{axis}" for axis in axes]
            + [f"action_{index}" for index in range(9)]
        )
        for index in range(276):
            writer.writerow(
                [
                    index,
                    sample_ids[index],
                    record_scenes[index],
                    int(records[index]["target_frame_idx"]),
                    int(blocks[index]),
                    labels[index],
                ]
                + descriptor[index].tolist()
                + actions[index].tolist()
            )

    permutation_dir = output / "permutations"
    permutation_dir.mkdir(parents=True, exist_ok=True)
    mapping_index = []
    for offset in range(args.permutations):
        seed = args.seed_base + offset
        donors = make_block_derangement(blocks, record_scenes, seed=seed)
        validate_derangement(donors, blocks, record_scenes)
        mapping = {
            "seed": seed,
            "method": "10_equal_frequency_motion_magnitude_blocks_scene_independent_derangement",
            "recipient_to_donor_index": donors.tolist(),
            "recipient_to_donor_sample_id": [sample_ids[index] for index in donors],
            "action_sha256_before": tensor_sha256(actions),
            "action_sha256_after": tensor_sha256(actions[torch.from_numpy(donors)]),
            "exact_marginal_preserved": True,
            "fixed_points": int(np.count_nonzero(donors == np.arange(276))),
            "same_scene_donors": int(
                np.count_nonzero(
                    np.asarray(record_scenes, dtype=object)[donors]
                    == np.asarray(record_scenes, dtype=object)
                )
            ),
        }
        path = permutation_dir / f"seed_{seed}.json"
        write_json(path, mapping)
        mapping_index.append({"seed": seed, "path": str(path), "sha256": sha256(path)})
    mapping_index_path = permutation_dir / "index.json"
    write_json(mapping_index_path, mapping_index)

    protected_snapshot = {
        rel: {"sha256": sha256(root / rel), "size": (root / rel).stat().st_size}
        for rel in PROTECTED
    }
    provenance = {
        "run_tag": RUN_TAG,
        "model": "wan",
        "prefix": 12,
        "layer": 20,
        "manifest": str(manifest),
        "manifest_sha256": MANIFEST_SHA256,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "predictions": str(predictions_path),
        "predictions_sha256": PREDICTIONS_SHA256,
        "summary": str(summary_path),
        "summary_sha256": SUMMARY_SHA256,
        "action_sha256": tensor_sha256(actions),
        "target_sha256": tensor_sha256(targets),
        "sample_id_source": "P12 val manifest in unchanged evaluator order",
        "protected_files_at_start": protected_snapshot,
    }
    phase0 = {
        "status": "PASS",
        "metric_implementation_valid": True,
        "action_input_reaches_probe": bool(
            baseline["actual_checkpoint_action_sensitivity"]["fraction_changed_gt_1e-6"] > 0
        ),
        "candidate_leakage": "NONE_FOUND",
        "candidate_pool_definition": "all 276 frozen P12 val target features in evaluator order",
        "positive_definition": "manifest sample-ID-paired target; diagonal in frozen evaluator order",
        "rank_definition": {"historical_mean_rank": "zero-based", "report_rank1": "one-based"},
        "training_probe_inputs": ["input_feat", "action"],
        "training_target_only": "target_feat pooled over feature H,W",
        "metadata_not_passed_to_probe": [
            "sample_id",
            "scene_path",
            "target_frame_idx",
            "streaming_window_id",
        ],
        "action_encoding": "relative pose: first two rows of R_rel (6D) plus t_rel (3D)",
        "native_action_classes": False,
        "stratification_classes": "dominant signed rotation-vector/translation component, RMS standardized on frozen val",
        "action_class_counts": class_counts,
        "permutation_count": args.permutations,
        "permutation_seed_base": args.seed_base,
        "permutation_index_sha256": sha256(mapping_index_path),
        "historical_batch_roll_is_formal_control": False,
        "historical_batch_roll_reason": "batch-local roll can retain scene structure; replaced by full-val scene-independent mappings",
        "copy_anchor_isolated_status": "UNAVAILABLE_NO_EXTRACTION_ALLOWED",
    }

    write_json(output / "input_provenance.json", provenance)
    write_json(output / "baseline_recomputed.json", baseline)
    write_json(output / "phase0_audit.json", phase0)
    write_json(
        output / "prepare_status.json",
        {
            "ok": True,
            "n": 276,
            "scenes": 69,
            "prefix": 12,
            "layer": 20,
            "permutations": args.permutations,
            "output": str(output),
            "phase0_sha256": sha256(output / "phase0_audit.json"),
            "provenance_sha256": sha256(output / "input_provenance.json"),
        },
    )
    print(json.dumps({"ok": True, "output": str(output), "classes": class_counts}, sort_keys=True))


if __name__ == "__main__":
    main()
