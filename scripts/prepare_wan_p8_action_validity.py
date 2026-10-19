#!/usr/bin/env python3
"""Freeze inputs and prepare Phase-1 permutations for the Wan p8 audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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


RUN_TAG = "wan_p8_action_dynamics_canonical_manifest_exact_20260821_200604_2075f3e"
MANIFEST_SHA256 = "772194f077c45cc85a942f17f46f50103125ae8f5b1c983c08881c1759607e0c"
LAYERS = (0, 5, 10, 15, 25, 29)
CHECKPOINTS = {
    0: ("jspcgevf", "014ee02384b1b54cbd5ebee3d8b0ff133e3eb8ff949459f5e7bda70af705c13b"),
    5: ("kttf6gu0", "9bc4a459008871199945cc287a65bb62c8906ee66aa4d10edd288380bdb4a093"),
    10: ("2oefz7nm", "d4fd33f068a396d5a024ac212f4f452a8cc43a1f12a9aeab8d29290243ef2343"),
    15: ("npn9jxyk", "bcfc7ccf508cba6b99dbe41e01a134a146d3ec557c147d639ad9a3f885043ed2"),
    25: ("jo3b0g4f", "c4bd2643fbf6a0f175e411849364967b4c2ed5dfef6a5b2ef3818a3a6c7ad1c3"),
    29: ("nmawq03h", "3712d78b5058008849a37cee01aa832947bdd2a7a9dab38f85fd05ac3336b35b"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def eval_dir(root: Path, layer: int) -> Path:
    name = f"{RUN_TAG}_canonical_clean_action_dynamics_wan_p8_layer{layer}"
    return root / "logs/inscene15k_streaming/runs" / f"{RUN_TAG}_eval_staging" / name / "eval"


def checkpoint_path(root: Path, layer: int) -> Path:
    wandb_id, _ = CHECKPOINTS[layer]
    name = f"{RUN_TAG}_canonical_clean_action_dynamics_wan_p8_layer{layer}"
    return (
        root
        / "logs/inscene15k_streaming/runs"
        / RUN_TAG
        / name
        / "VideoProbe3D"
        / wandb_id
        / "checkpoints/epoch=49-step=15300.ckpt"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/analysis/wan_p8_action_validity"),
    )
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=2026082500)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    manifest = root / "manifests/c_retrain_20260722/inscene_c_h124_prefix8.jsonl"
    if sha256(manifest) != MANIFEST_SHA256:
        raise RuntimeError("canonical manifest hash changed")
    manifest_rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    val_manifest = [row for row in manifest_rows if row.get("split") == "val"]
    if len(manifest_rows) != 2724 or len(val_manifest) != 276:
        raise RuntimeError("canonical manifest counts changed")

    baseline: Dict[str, Any] = {}
    provenance: Dict[str, Any] = {
        "run_tag": RUN_TAG,
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "manifest_counts": {"all": len(manifest_rows), "val": len(val_manifest)},
        "layers": {},
    }
    reference_ids = None
    reference_scenes = None
    reference_actions = None

    for layer in LAYERS:
        directory = eval_dir(root, layer)
        predictions_path = directory / "val_predictions.pt"
        summary_path = directory / "val_summary.json"
        checkpoint = checkpoint_path(root, layer)
        expected_checkpoint_sha = CHECKPOINTS[layer][1]
        if sha256(checkpoint) != expected_checkpoint_sha:
            raise RuntimeError(f"layer {layer} checkpoint hash changed")
        records = torch.load(predictions_path, map_location="cpu", weights_only=False)
        if len(records) != 276 or not all(bool(record.get("valid")) for record in records):
            raise RuntimeError(f"layer {layer} prediction inventory is invalid")
        sample_ids = [str(record["sample_id"]) for record in records]
        scenes = [str(record["scene_path"]) for record in records]
        actions = torch.stack([record["action"].float() for record in records])
        predictions = torch.stack([record["pred"].float() for record in records])
        no_action = torch.stack([record["pred_no_action"].float() for record in records])
        targets = torch.stack([record["target"].float() for record in records])
        copies = torch.stack([record["last_observation"].float() for record in records])

        if reference_ids is None:
            reference_ids, reference_scenes, reference_actions = sample_ids, scenes, actions
        else:
            if sample_ids != reference_ids or scenes != reference_scenes:
                raise RuntimeError(f"layer {layer} record ordering differs")
            torch.testing.assert_close(actions, reference_actions, rtol=0.0, atol=0.0)

        current = aggregate_retrieval(retrieval_per_sample(predictions, targets))
        null = aggregate_retrieval(retrieval_per_sample(no_action, targets))
        copy = aggregate_retrieval(retrieval_per_sample(copies, targets))
        summary = json.loads(summary_path.read_text())
        comparisons = {
            "mean_cos": abs(current["cos"] - float(summary["mean_cos"])),
            "global_R@1": abs(current["r1"] - float(summary["global_R@1"])),
            "global_R@5": abs(current["r5"] - float(summary["global_R@5"])),
            "mean_rank": abs(current["rank0"] - float(summary["mean_rank"])),
            "null_R@1": abs(null["r1"] - float(summary["no_action_global_R@1"])),
            "copy_R@1": abs(copy["r1"] - float(summary["last_observation_global_R@1"])),
        }
        if max(comparisons.values()) > 1e-6:
            raise RuntimeError(f"layer {layer} metric cross-check failed: {comparisons}")
        delta = predictions - no_action
        baseline[str(layer)] = {
            "correct": current,
            "eval_null_action": null,
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
        provenance["layers"][str(layer)] = {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": expected_checkpoint_sha,
            "predictions": str(predictions_path.resolve()),
            "predictions_sha256": sha256(predictions_path),
            "summary": str(summary_path.resolve()),
            "summary_sha256": sha256(summary_path),
            "action_sha256": tensor_sha256(actions),
            "target_sha256": tensor_sha256(targets),
        }

    assert reference_ids is not None and reference_scenes is not None and reference_actions is not None
    if len(set(reference_ids)) != 276 or len(set(reference_scenes)) != 69:
        raise RuntimeError("val sample/scene cardinalities changed")
    if any(count != 4 for count in Counter(reference_scenes).values()):
        raise RuntimeError("expected exactly four val samples per scene")

    descriptor, labels, axes = action_descriptors(reference_actions.numpy())
    blocks = frequency_blocks(descriptor, n_blocks=10)
    class_counts = dict(sorted(Counter(labels.tolist()).items()))
    inventory_path = output / "tables/action_inventory.csv"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    with inventory_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["sample_index", "sample_id", "scene_path", "frequency_block", "action_class"]
            + [f"descriptor_{axis}" for axis in axes]
            + [f"action_{i}" for i in range(9)]
        )
        for index in range(276):
            writer.writerow(
                [index, reference_ids[index], reference_scenes[index], int(blocks[index]), labels[index]]
                + descriptor[index].tolist()
                + reference_actions[index].tolist()
            )

    permutation_dir = output / "permutations"
    permutation_dir.mkdir(parents=True, exist_ok=True)
    mapping_index = []
    for offset in range(args.permutations):
        seed = args.seed_base + offset
        donors = make_block_derangement(blocks, reference_scenes, seed=seed)
        validate_derangement(donors, blocks, reference_scenes)
        mapping = {
            "seed": seed,
            "method": "10_equal_frequency_motion_magnitude_blocks_scene_independent_derangement",
            "recipient_to_donor_index": donors.tolist(),
            "recipient_to_donor_sample_id": [reference_ids[index] for index in donors],
            "action_sha256_before": tensor_sha256(reference_actions),
            "action_sha256_after": tensor_sha256(reference_actions[torch.from_numpy(donors)]),
            "exact_marginal_preserved": True,
            "fixed_points": int(np.count_nonzero(donors == np.arange(276))),
            "same_scene_donors": int(
                np.count_nonzero(np.asarray(reference_scenes, dtype=object)[donors] == np.asarray(reference_scenes, dtype=object))
            ),
        }
        path = permutation_dir / f"seed_{seed}.json"
        write_json(path, mapping)
        mapping_index.append({"seed": seed, "path": str(path.resolve()), "sha256": sha256(path)})

    phase0 = {
        "status": "PASS",
        "metric_implementation_valid": True,
        "action_input_reaches_probe": True,
        "candidate_leakage": "NONE_FOUND",
        "candidate_pool_definition": "all 276 canonical val target features in saved record order",
        "positive_definition": "sample_id-paired target; diagonal only in unchanged record order",
        "rank_definition": {"canonical_mean_rank": "zero-based", "report_mean_rank": "one-based"},
        "training_probe_inputs": ["input_feat", "action"],
        "training_target_only": "target_feat pooled over feature H,W",
        "metadata_not_passed_to_probe": [
            "sample_id",
            "scene_path",
            "target_frame_idx",
            "streaming_window_id",
            "streaming_tail_raw",
            "paired_sample_id",
        ],
        "action_encoding": "relative pose: first two rows of R_rel (6D) plus t_rel (3D)",
        "native_action_classes": False,
        "stratification_classes": "dominant signed component of rotation-vector plus translation, RMS standardized on frozen val",
        "action_class_counts": class_counts,
        "permutation_count": args.permutations,
        "permutation_seed_base": args.seed_base,
        "permutation_index_sha256_pending": True,
        "historical_batch_roll_is_formal_control": False,
        "historical_batch_roll_reason": "batch-local roll can retain scene structure and is replaced by frozen full-val mappings",
        "copy_anchor_isolated_available": 0,
        "copy_anchor_isolated_status": "UNAVAILABLE_NO_EXTRACTION_ALLOWED",
        "code_locations": {
            "dataset_return": "vidfm3d/data/components/inscene15k_dataset.py:1782",
            "training_step": "vidfm3d/models/probe_ext_module.py:297",
            "action_probe": "vidfm3d/models/components/probe_action_dynamics.py:54",
            "action_encoder": "vidfm3d/utils/spatial_diag.py:716",
            "historical_metric": "vidfm3d/eval_diag.py:345",
        },
    }
    mapping_index_path = output / "permutations/index.json"
    write_json(mapping_index_path, mapping_index)
    phase0["permutation_index_sha256_pending"] = False
    phase0["permutation_index_sha256"] = sha256(mapping_index_path)

    write_json(output / "input_provenance.json", provenance)
    write_json(output / "baseline_recomputed.json", baseline)
    write_json(output / "phase0_audit.json", phase0)
    write_json(
        output / "prepare_status.json",
        {
            "ok": True,
            "n": 276,
            "scenes": 69,
            "permutations": args.permutations,
            "output": str(output),
            "phase0_sha256": sha256(output / "phase0_audit.json"),
            "provenance_sha256": sha256(output / "input_provenance.json"),
        },
    )
    print(json.dumps({"ok": True, "output": str(output), "classes": class_counts}, sort_keys=True))


if __name__ == "__main__":
    main()
