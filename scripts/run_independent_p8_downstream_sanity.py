#!/usr/bin/env python3
"""Actual-data, non-paper sanity for the frozen final downstream interfaces."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("final_analysis", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    if args.output.exists(): raise SystemExit(f"refusing to overwrite {args.output}")
    repo = args.repo.resolve(); source = repo / "artifacts/independent_actionability_algebra_p8_20260831T091224Z_28bda217/manifests"
    old_root = repo / "features/independent_actionability_algebra_p8/independent_actionability_algebra_p8_20260831T091224Z_28bda217/dino_l0_science_pilot_v1"
    module = load_module(repo / "scripts/run_independent_p8_final_analysis.py")

    class PilotStore(module.Store):
        def _get(self, unit):
            if unit["unit_id"] not in self.cache:
                path = self.root / self.model / "scannetpp" / unit["scene_id"] / f"{unit['unit_id']}.safetensors"
                value = load_file(str(path))["feat"].float().reshape(-1, 1024).mean(0)
                self.cache[unit["unit_id"]] = value
            return self.cache[unit["unit_id"]]

    store = PilotStore(source / "science_pilot_subset_v1_feature_request_v2_dino_l0.jsonl", old_root, "dino")
    action = module.read_jsonl(source / "science_pilot_subset_v1_actionability_train.jsonl")[:32]
    algebra_rows = module.read_jsonl(source / "science_pilot_subset_v1_algebra_train.jsonl")
    # The old pilot requested only h1/h2/h4 in general; use the actual subset
    # whose h3 happened to be independently requested by another pilot role.
    algebra = [row for row in algebra_rows if row["anchor_raw_id"] in store.isolated
               and all(raw in store.isolated for raw in row["waypoint_raw_ids"])][:8]
    referenced = []
    for row in action:
        referenced.extend((store.context(row["context_raw_ids"]), store.vector(row["anchor_raw_id"]), store.vector(row["goal_raw_id"])))
    for row in algebra:
        referenced.append(store.vector(row["anchor_raw_id"])); referenced.extend(store.vector(x) for x in row["waypoint_raw_ids"])
    transform = module.FeatureTransform(torch.stack(referenced))

    x, y = module.scorer_features(action, store, transform, "raw", "direct")
    scenes = [row["scene_id"] for row in action]
    scorer, _, _ = module.train_scorer(x, y, x, y, scenes, 42, "cpu", epochs=120)
    with torch.no_grad():
        pred = scorer(x).argmax(1); swapped = scorer(x.flip(1)).argmax(1)
    scorer_accuracy = float((pred == y).float().mean())
    equivariance_failures = int(((1 - pred) != swapped).sum())

    dyn = module.dynamics_tensors(algebra, store)
    dynamics_model, _, _ = module.train_dynamics(
        dyn, dyn, transform, [row["scene_id"] for row in algebra], "full", 42, "cpu", epochs=120
    )
    rollout = module.unstandard_rollout(dynamics_model, transform, dyn[0], dyn[1], "cpu")
    dynamics_cosine = float(module.cosine_distance(rollout, dyn[2]).mean())
    result = {
        "status": "COMPLETE_VALID" if scorer_accuracy >= .90 and equivariance_failures == 0 else "FAILED_SANITY",
        "designation": "actual DINO pilot data; implementation sanity only; not a paper result",
        "samples": {"actionability": len(action), "dynamics": len(algebra)},
        "direct_train_overfit_accuracy": scorer_accuracy,
        "candidate_swap_equivariance_failures": equivariance_failures,
        "dynamics_train_cosine_distance": dynamics_cosine,
        "output_shapes": {"scorer_logits": list(scorer(x).shape), "dynamics_rollout": list(rollout.shape)},
        "input_leakage_contract": "scorer_features accepts frozen feature vectors and candidate 9D actions only; pose/utility/IDs are excluded",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True)); return 0 if result["status"] == "COMPLETE_VALID" else 2


if __name__ == "__main__": raise SystemExit(main())
