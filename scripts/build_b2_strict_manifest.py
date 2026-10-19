#!/usr/bin/env python3
"""Build fixed-target, strict-zero-hidden B2 manifests.

All compared prefix lengths end at the same target observation.  A prefix of
length ``p`` is the suffix of the 24-observation motion window, so changing p
changes history length only.  Labels are produced once from raw ScanNet++
masks/depth/cameras and copied verbatim to every model-independent manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import rootutils
import torch

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from vidfm3d.data.components.inscene15k_dataset import (
    InsScene15KDataset,
    canonical_streaming_sample_id,
    depth_to_pointmap,
)
from vidfm3d.utils.spatial_diag import (
    compute_strict_hidden_object_target_for_id,
    select_strict_zero_hidden_object_id,
)


PREFIXES = (8, 12, 16, 24)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unknown"


def _dataset(args, split: str) -> InsScene15KDataset:
    return InsScene15KDataset(
        root=args.data_root,
        sources=["processed_scannetpp_v2"],
        split=split,
        split_manifest=args.split_manifest,
        num_views=max(PREFIXES),
        target_h=args.target_h,
        target_w=args.target_w,
        include_pmaps=True,
        streaming_prefix=True,
        prefix_lengths=[max(PREFIXES)],
        prefix_min_len=max(PREFIXES),
        prefix_max_len=max(PREFIXES),
        streaming_window_obs_len=max(PREFIXES),
        streaming_window_stride=args.window_stride,
        streaming_max_windows_per_scene=args.max_windows_per_scene,
        streaming_motion_step=args.motion_step,
        streaming_rotation_weight=args.rotation_weight,
    )


def _build_one(dataset: InsScene15KDataset, scene: dict, split: str, args):
    full_indices = torch.as_tensor(scene["streaming_indices"], dtype=torch.long)
    if int(full_indices.numel()) != max(PREFIXES):
        return None, "window_not_24"
    if scene["source"] != "scannetpp":
        return None, "not_scannetpp"
    try:
        _images, masks, depth, intrinsics, extrinsics = dataset._load_scannetpp_scene(
            scene, full_indices
        )
        _images, masks, depth, intrinsics = dataset._resize_to_target(
            _images, masks, depth, intrinsics
        )
    except Exception as exc:
        return None, f"load_error:{type(exc).__name__}"

    # The object must be queryable even in the shortest history suffix.
    target_idx = int(full_indices.numel()) - 1
    history_start = int(full_indices.numel()) - min(PREFIXES)
    object_id = select_strict_zero_hidden_object_id(
        masks,
        target_frame_idx=target_idx,
        min_visible_pixels=args.min_visible_pixels,
        history_start_idx=history_start,
    )
    if object_id is None:
        return None, "no_strict_zero_candidate_in_common_history"

    pointmaps = torch.stack([
        depth_to_pointmap(depth[i], intrinsics[i], extrinsics[i])
        for i in range(depth.shape[0])
    ])
    target = compute_strict_hidden_object_target_for_id(
        object_id,
        masks,
        pointmaps,
        (depth > 0).float(),
        extrinsics,
        min_visible_pixels=args.min_visible_pixels,
        target_frame_idx=target_idx,
    )
    if target is None:
        return None, "strict_target_geometry_invalid"
    if int(target["target_raw_pixels"].item()) != 0:
        return None, "target_pixels_nonzero_internal_error"

    pair_id = canonical_streaming_sample_id(
        scene,
        "b2_strict_pair",
        target_indices=[int(full_indices[-1].item())],
        object_id=object_id,
    )
    base = {
        "schema_version": 1,
        "split": split,
        "source": "scannetpp",
        "scene_dir": scene["scene_dir"],
        "scene_id": scene.get("scene_id"),
        "num_frames": int(scene["num_frames"]),
        "streaming_window_id": int(scene.get("streaming_window_id", 0)),
        "streaming_obs_start": int(scene.get("streaming_obs_start", 0)),
        "streaming_sampling": scene.get("streaming_sampling", "motion_uniform"),
        "streaming_window_indices": [int(v) for v in full_indices.tolist()],
        "streaming_tail_raw": int(full_indices[-1].item()),
        "strict_object_id": int(object_id),
        "strict_pair_id": pair_id,
        "strict_target_raw_pixels": 0,
        "strict_centroid_aggregation": "coordinatewise_median_all_valid_past_points",
        "strict_centroid_world": [float(v) for v in target["delta_world"].tolist()],
        "strict_polar": [float(v) for v in target["polar"].tolist()],
        "visible_history_frames": int(target["visible_history_indices"].numel()),
        "last_visible_frame": int(target["last_visible_frame"].item()),
        "last_visible_pixel_area": int(target["last_visible_pixel_area"].item()),
        "hidden_duration": int(target["hidden_duration"].item()),
    }
    records = {}
    for prefix in PREFIXES:
        record = dict(base)
        record["prefix"] = prefix
        record["streaming_indices"] = [int(v) for v in full_indices[-prefix:].tolist()]
        # Strict caches are extracted into one root per prefix.  Within that
        # root, the local prefix tail is p-1 even though raw target time is fixed.
        record["streaming_tail"] = prefix - 1
        record["sample_id"] = f"{pair_id}|prefix={prefix}"
        records[prefix] = record
    return records, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--min-visible-pixels", type=int, default=200)
    parser.add_argument("--target-h", type=int, default=288)
    parser.add_argument("--target-w", type=int, default=512)
    parser.add_argument("--window-stride", type=int, default=8)
    parser.add_argument("--max-windows-per-scene", type=int, default=4)
    parser.add_argument("--motion-step", type=float, default=0.35)
    parser.add_argument("--rotation-weight", type=float, default=0.5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted = {prefix: [] for prefix in PREFIXES}
    rejections = []
    for split in args.splits:
        dataset = _dataset(args, split)
        for scene in dataset.scenes:
            records, reason = _build_one(dataset, scene, split, args)
            if records is None:
                rejections.append({
                    "split": split,
                    "scene_id": scene.get("scene_id", ""),
                    "window_id": scene.get("streaming_window_id", ""),
                    "reason": reason,
                })
                continue
            for prefix, record in records.items():
                accepted[prefix].append(record)

    rejection_path = output_dir / "b2_strict_rejection_reasons.csv"
    with rejection_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "scene_id", "window_id", "reason"])
        writer.writeheader()
        writer.writerows(rejections)

    if not all(accepted[prefix] for prefix in PREFIXES):
        raise RuntimeError(
            "strict manifest gate failed: at least one prefix has zero accepted samples; "
            f"counts={ {prefix: len(rows) for prefix, rows in accepted.items()} }"
        )

    split_hash = _sha256(Path(args.split_manifest))
    for prefix in PREFIXES:
        rows = sorted(accepted[prefix], key=lambda row: row["sample_id"])
        manifest_path = output_dir / f"b2_strict_prefix{prefix}.jsonl"
        with manifest_path.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        meta = {
            "schema_version": 1,
            "run_tag": args.run_tag,
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--short")),
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "split_manifest": args.split_manifest,
            "split_manifest_sha256": split_hash,
            "prefix": prefix,
            "fixed_target_prefixes": list(PREFIXES),
            "n": len(rows),
            "n_by_split": dict(Counter(row["split"] for row in rows)),
            "target_raw_pixels_min": min(row["strict_target_raw_pixels"] for row in rows),
            "target_raw_pixels_max": max(row["strict_target_raw_pixels"] for row in rows),
            "min_visible_pixels": args.min_visible_pixels,
            "centroid_aggregation": "coordinatewise_median_all_valid_past_points",
            "data_root": args.data_root,
        }
        (output_dir / f"b2_strict_prefix{prefix}.meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
