#!/usr/bin/env python3
"""Preflight and, when eligible, run an observed-pair SeeSE3 M3/M4 sanity.

This is deliberately separate from the observed-order protocol and the legacy
``see_se3_smoke``.  A valid replication sanity requires (1) real observed frame
pairs, (2) an intra-scene temporal split with disjoint frame support, (3) an
exported representative middle layer, (4) enough pairs, and (5) several M4
seeds.  It refuses to silently relabel DINO layer 0 / a small scene-disjoint
smoke as a replication.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from analyses.se3_latent_mvp.geometry import se3_log
from analyses.se3_latent_mvp.run_mvp import load_and_audit_tuples
from analyses.se3_protocol_v2.protocol import dump_json, sha256
from analyses.se3_protocol_v2.see_se3_smoke import fit_m3, fit_m4, metrics


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must provide at least two distinct integers")
    return seeds


def exported_layer(path: Path) -> int | None:
    """Infer the feature layer from a C-series dump directory, not its model name."""
    match = re.search(r"_layer(\d+)(?:/|$)", str(path))
    return int(match.group(1)) if match else None


@dataclass(frozen=True)
class ObservedPair:
    scene_id: str
    source_frame: int
    target_frame: int
    source: torch.Tensor
    target: torch.Tensor
    twist: np.ndarray


def temporal_pairs(records: list[Any], temporal_gap: int) -> tuple[dict[str, list[ObservedPair]], dict[str, Any]]:
    """Split observed primitive pairs temporally within each scene.

    A pair belongs to a split only when both of its frame indices lie strictly
    within that temporal segment.  This gives train/val/test no shared raw
    frame support, while still using the same observed scenes as the paper-style
    temporal generalization check.
    """
    by_scene: dict[str, list[Any]] = {}
    for record in records:
        by_scene.setdefault(record.scene_id, []).append(record)
    result = {name: [] for name in ("train", "val", "test")}
    scene_audit: dict[str, Any] = {}
    for scene_id, rows in sorted(by_scene.items()):
        all_frames = [frame for row in rows for frame in row.raw_frame_indices]
        low, high = min(all_frames), max(all_frames)
        boundary_train = low + int(np.floor(0.60 * (high - low)))
        boundary_val = low + int(np.floor(0.80 * (high - low)))
        counts = {name: 0 for name in result}
        for row in rows:
            transforms = row.transforms
            for source_index, target_index, source, target, twist in (
                (row.raw_frame_indices[0], row.raw_frame_indices[1], row.features[0], row.features[1], transforms[0]),
                (row.raw_frame_indices[1], row.raw_frame_indices[2], row.features[1], row.features[2], transforms[1]),
            ):
                if target_index <= boundary_train:
                    split = "train"
                elif source_index > boundary_train + temporal_gap and target_index <= boundary_val:
                    split = "val"
                elif source_index > boundary_val + temporal_gap:
                    split = "test"
                else:
                    continue  # boundary buffer: do not leak frame support.
                result[split].append(ObservedPair(scene_id, int(source_index), int(target_index), source, target, se3_log(twist)))
                counts[split] += 1
        scene_audit[scene_id] = {
            "frame_range": [int(low), int(high)],
            "boundaries": {"train_end": int(boundary_train), "val_end": int(boundary_val), "gap": int(temporal_gap)},
            "pair_counts": counts,
        }
    supports = {
        split: {(pair.scene_id, frame) for pair in pairs for frame in (pair.source_frame, pair.target_frame)}
        for split, pairs in result.items()
    }
    disjoint = all(not (supports[left] & supports[right]) for left in supports for right in supports if left != right)
    return result, {
        "scheme": "within_scene_60_20_20_temporal_with_boundary_gap",
        "temporal_gap_frames": int(temporal_gap),
        "scene_audit": scene_audit,
        "pair_counts": {name: len(value) for name, value in result.items()},
        "scene_counts": {name: len({pair.scene_id for pair in value}) for name, value in result.items()},
        "raw_frame_support_disjoint": disjoint,
    }


def tensors(pairs: list[ObservedPair]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not pairs:
        raise RuntimeError("temporal split contained zero observed pairs")
    return (
        torch.stack([pair.source for pair in pairs]).float(),
        torch.stack([pair.target for pair in pairs]).float(),
        torch.from_numpy(np.stack([pair.twist for pair in pairs])).float(),
    )


def means(values: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        key: {"mean": float(np.mean([value[key] for value in values])), "std": float(np.std([value[key] for value in values], ddof=0))}
        for key in values[0]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--max-epochs", type=int, default=400)
    parser.add_argument("--temporal-gap", type=int, default=4)
    parser.add_argument("--minimum-train-pairs", type=int, default=1000)
    parser.add_argument("--minimum-eval-pairs", type=int, default=200)
    parser.add_argument("--manifest", default="manifests/c_retrain_20260722/inscene_c_h124_prefix8.jsonl")
    parser.add_argument("--c1-dump", default="logs/inscene15k_streaming/runs/c_eval_20260724_1903/c_eval_20260724_1903_action_dynamics_dino_p8_layer0/eval/val_predictions.pt")
    parser.add_argument("--c2-dump", default="logs/inscene15k_streaming/runs/c_eval_20260724_1903/c_eval_20260724_1903_path_integration_dino_p8_layer0/eval/val_predictions.pt")
    parser.add_argument("--raw-root", default="/mnt/umm/users/yinbaiqiao/InsScene-15K/data/processed_scannetpp_v2/processed_scannetpp_v2")
    args = parser.parse_args()
    output = resolve(args.output_dir)
    if output.exists():
        raise SystemExit(f"refusing to overwrite experimental output: {output}")
    output.mkdir(parents=True)
    seeds = parse_seeds(args.seeds)
    cfg = {
        "manifest": str(resolve(args.manifest)),
        "c1_dump": str(resolve(args.c1_dump)),
        "c2_dump": str(resolve(args.c2_dump)),
        "raw_scene_root": str(resolve(args.raw_root)),
        "alignment_atol": 2e-5,
        "pose_atol": 1e-5,
    }
    records, tuple_audit = load_and_audit_tuples(cfg)
    pairs, temporal_audit = temporal_pairs(records, args.temporal_gap)
    layer = exported_layer(Path(cfg["c2_dump"]))
    # The registered DINO extractor explicitly exports a single last-hidden-state
    # file as layer 0.  It cannot be substituted for a pre-registered middle
    # transformer layer in this sanity.
    layer_gate = layer is not None and layer > 0
    count_gate = (
        temporal_audit["pair_counts"]["train"] >= args.minimum_train_pairs
        and temporal_audit["pair_counts"]["val"] >= args.minimum_eval_pairs
        and temporal_audit["pair_counts"]["test"] >= args.minimum_eval_pairs
    )
    prerequisites = {
        "observed_frame_pairs": True,
        "paper_style_temporal_split": temporal_audit["raw_frame_support_disjoint"],
        "representative_exported_middle_layer": layer_gate,
        "enough_pairs": count_gate,
        "multiple_seeds_requested": len(seeds) >= 2,
    }
    output_data: dict[str, Any] = {
        "status": "eligible_to_run" if all(prerequisites.values()) else "blocked_missing_replication_prerequisites",
        "not_a_relabel_of_legacy_scene_disjoint_smoke": True,
        "feature": {"model": "dino", "exported_layer": layer, "role": "C2 isolated observed target"},
        "requirements": {
            **prerequisites,
            "minimum_train_pairs": args.minimum_train_pairs,
            "minimum_eval_pairs": args.minimum_eval_pairs,
            "requested_m4_seeds": list(seeds),
            "middle_layer_reason": "DINO C-series artifacts label the only exported last-hidden-state representation as layer 0; no middle-layer C1/C2 artifact was found or substituted.",
        },
        "temporal_split": temporal_audit,
        "tuple_alignment": tuple_audit["gates"],
        "provenance": {key: {"path": cfg[key], "sha256": sha256(Path(cfg[key]))} for key in ("manifest", "c1_dump", "c2_dump")},
        "training": "not_started_until_all_replication_prerequisites_pass",
    }
    if all(prerequisites.values()):
        torch.set_num_threads(4)
        train_s, train_t, train_xi = tensors(pairs["train"])
        val_s, val_t, val_xi = tensors(pairs["val"])
        test_s, test_t, test_xi = tensors(pairs["test"])
        mean, std = train_xi.mean(0), train_xi.std(0).clamp_min(1e-6)
        train_y, val_y, test_y = (train_xi - mean) / std, (val_xi - mean) / std, (test_xi - mean) / std
        m3_prediction, m3_meta = fit_m3(train_s, train_t, train_y, val_s, val_t, val_y, test_s, test_t)
        m3 = metrics(test_y, m3_prediction, mean, std)
        m4_per_seed = []
        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            prediction, meta = fit_m4(train_s, train_t, train_y, val_s, val_t, val_y, test_s, test_t, seed, args.max_epochs)
            m4_per_seed.append({"seed": seed, "metrics": metrics(test_y, prediction, mean, std), "training": {key: value for key, value in meta.items() if key != "history"}})
        m4_summary = means([row["metrics"] for row in m4_per_seed])
        m4_beats_m3 = m4_summary["overall_zscore_r2"]["mean"] > m3["overall_zscore_r2"]
        output_data.update({
            "M3": {"metrics": m3, "training": m3_meta},
            "M4": {"per_seed": m4_per_seed, "summary": m4_summary},
            "m4_vs_m3_gate": {
                "M4_overall_R2_strictly_exceeds_M3": m4_beats_m3,
                "status": "passed" if m4_beats_m3 else "failed_debug_implementation_or_training_before_interpreting",
            },
        })
        if not m4_beats_m3:
            output_data["status"] = "m4_not_better_than_m3_requires_debug"
    dump_json(output / "seese3_replication_sanity.json", output_data)
    (output / "REPORT.md").write_text(
        "# SeeSE3 replication sanity\n\n"
        f"Status: **{output_data['status']}**.  This file is intentionally separate from observed-order feature grounding.\n\n"
        "```json\n" + json.dumps(output_data["requirements"], indent=2, sort_keys=True) + "\n```\n"
    )
    print(json.dumps({"output": str(output), "status": output_data["status"], "requirements": prerequisites, "pair_counts": temporal_audit["pair_counts"]}, indent=2))


if __name__ == "__main__":
    main()
