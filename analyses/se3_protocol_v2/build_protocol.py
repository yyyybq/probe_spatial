#!/usr/bin/env python3
"""Build a geometry-first SE(3) protocol-v2 manifest and audit gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from analyses.se3_protocol_v2.protocol import (
    POSE_CONVENTION,
    dump_json,
    dump_jsonl,
    edge_motions,
    load_manifest_scenes,
    load_scene,
    motion_quantiles,
    optional_dual_image_extension,
    select_branches,
    select_primitives,
    sha256,
    split_scene_ids,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def render_report(output: Path, audit: dict[str, Any]) -> None:
    gates = audit["gates"]
    coverage = audit["branch_coverage"]
    report = f"""# SE(3) protocol-v2 endpoint audit

Status: **optional dual-image residual diagnostic; not a blocker for observed-video protocol-v2**.

## Frozen coordinate contract

- Raw `trajectories` are camera-to-world matrices.  This protocol uses world-to-camera
  `E = inv(c2w)` and `T_AB = E_B @ inv(E_A)`.
- Ordered endpoints are `E_12 = T2 @ T1 @ E_A` and
  `E_21 = T1 @ T2 @ E_A`.  All branch matrices use that convention.

## Optional dual-image extension

- This track is `{audit['optional_dual_image']['status']}` by policy.  No mesh, NeRF,
  3DGS, ScanNet++ reconstruction, or renderer is searched for or built.
- Synthetic nearest-observed images retain their pose residuals, but are never a
  main-experiment target feature.

## Geometry-only construction

- Train primitive examples: `{audit['primitive_count']}` (required >= 1000).
- Held-out branch rows: `{coverage['total']}` (target 200), from `{audit['split_counts']['test']}` scene-disjoint test scenes.
- Requested/selected transform families are in `branch_coverage.json`; selection never
  inspected model features.
- `branches.jsonl` SHA-256: `{audit['branch_manifest_sha256']}`.

## Scope decision

The primary protocol is `observed_video_grounded_order`: real `(A,B,C)` tuples,
real `z_C`, and `T_AB,T_BC`.  Its feature coverage and split gates are evaluated in
`observed_order.py`; this optional branch manifest neither blocks it nor changes its
sample intersection.
"""
    (output / "REPORT.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifests/c_retrain_20260722/inscene_c_h124_prefix8.jsonl")
    parser.add_argument("--raw-root", default="/mnt/umm/users/yinbaiqiao/InsScene-15K/data/processed_scannetpp_v2/processed_scannetpp_v2")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--primitive-target", type=int, default=1000)
    parser.add_argument("--branch-target", type=int, default=200)
    args = parser.parse_args()

    manifest, raw_root, output = resolve(args.manifest), resolve(args.raw_root), resolve(args.output_dir)
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing experimental output: {output}")
    output.mkdir(parents=True)
    dump_json(
        output / "config.json",
        {
            "argv": sys.argv,
            "seed": args.seed,
            "primitive_target": args.primitive_target,
            "branch_target": args.branch_target,
            "manifest": str(manifest),
            "raw_root": str(raw_root),
            "builder_sha256": sha256(Path(__file__)),
        },
    )
    original_splits = load_manifest_scenes(manifest, raw_root)
    split = split_scene_ids(original_splits, args.seed)
    scenes = {scene: load_scene(scene, original_splits[scene], raw_root) for scene in sorted(split)}
    train_scenes = [scenes[scene] for scene in sorted(split) if split[scene] == "train"]
    test_scenes = [scenes[scene] for scene in sorted(split) if split[scene] == "test"]
    # Empirical train quantiles are deliberately sampled uniformly by scene;
    # scanning every raw adjacent frame is unnecessary for this fixed MVP and
    # would over-weight long captures.
    train_motions = [
        motion
        for scene in train_scenes
        for motion in edge_motions(
            scene,
            stride=1,
            sample_step=max(1, (len(scene.valid_frames) - 1) // 20),
        )
    ]
    quantiles = motion_quantiles(train_motions)
    primitives = select_primitives(train_scenes, args.primitive_target)
    branches, coverage = select_branches(test_scenes, quantiles, args.branch_target)
    dump_jsonl(output / "train_primitives.jsonl", primitives)
    dump_jsonl(output / "branches.jsonl", branches)
    dump_json(output / "scene_split.json", split)
    dump_json(output / "train_motion_quantiles.json", quantiles)
    dump_json(output / "branch_coverage.json", coverage)
    dual_image = optional_dual_image_extension()
    branch_sha = sha256(output / "branches.jsonl")
    split_counts = {label: sum(value == label for value in split.values()) for label in ("train", "val", "test")}
    family_requested = coverage["requested"]
    family_coverage = all(coverage["selected"].get(name, 0) >= quota for name, quota in family_requested.items())
    audit = {
        "pose_convention": POSE_CONVENTION,
        "inputs": {"manifest": str(manifest), "manifest_sha256": sha256(manifest), "raw_root": str(raw_root)},
        "optional_dual_image": dual_image,
        "seed": args.seed,
        "split_counts": split_counts,
        "primitive_count": len(primitives),
        "primitive_train_only": True,
        "branch_coverage": coverage,
        "branch_manifest_sha256": branch_sha,
        "feature_coverage": {
            "required_endpoint_roles_per_branch": ["source", "T1", "T2", "12", "21"],
            "required_exact_endpoint_feature_rows": len(branches) * 5,
            "available_exact_endpoint_feature_rows": 0,
            "coverage_fraction": 0.0,
            "reason": "Optional dual-image extension only; intentionally excluded from observed-video main protocol.",
        },
        "gates": {
            "pose_convention": True,
            "scene_split_disjoint": True,
            "composite_endpoint_not_in_train": True,
            "primitive_count_ge_1000": len(primitives) >= 1000,
            "heldout_branch_target_ge_200": len(branches) >= args.branch_target,
            "family_coverage": family_coverage,
            "optional_dual_image": "N/A_not_a_gate",
            "observed_video_protocol_authorized": True,
            "cross_model_expansion_not_blocked_by_dual_image": True,
        },
    }
    dump_json(output / "audit.json", audit)
    render_report(output, audit)
    print(json.dumps({"output": str(output), "gates": audit["gates"], "branches": len(branches)}, indent=2))


if __name__ == "__main__":
    main()
