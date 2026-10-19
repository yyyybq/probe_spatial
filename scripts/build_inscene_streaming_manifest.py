#!/usr/bin/env python
"""Build sample-level InsScene streaming manifests.

This is an offline indexing step for AOSS-backed training. The training dataset
can then read the local JSONL via INSCENE_MANIFEST and avoid listing remote
directories in every process.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from vidfm3d.utils import aoss_fs
from vidfm3d.utils.feature_layers import feature_filename
from vidfm3d.utils.temporal_sampling import temporal_windows_from_poses


def parse_int_list(value: str) -> list[int]:
    if not value:
        return []
    return [int(v) for v in value.replace(",", " ").split() if v.strip()]


def load_split_map(path: str | None) -> dict[str, str] | None:
    if not path:
        return None
    data = aoss_fs.load_json(path)
    splits = data.get("splits", data)
    out: dict[str, str] = {}
    for split, scene_ids in splits.items():
        for scene_id in scene_ids:
            out[str(scene_id)] = str(split)
    return out


def legacy_split(scene_ids: list[str], train_ratio: float) -> dict[str, str]:
    rng = np.random.default_rng(seed=42)
    indices = rng.permutation(len(scene_ids))
    split_idx = int(len(scene_ids) * train_ratio)
    train = {scene_ids[i] for i in indices[:split_idx]}
    return {scene_id: ("train" if scene_id in train else "val") for scene_id in scene_ids}


def collect_scannetpp_scene(scene_root: str, scene_id: str, min_frames: int) -> tuple[dict | None, str | None]:
    scene_dir = aoss_fs.join(scene_root, scene_id)
    if not aoss_fs.exists(scene_dir):
        return None, "missing_scene"
    img_dir = aoss_fs.join(scene_dir, "images")
    mask_dir = aoss_fs.join(scene_dir, "refined_ins_ids")
    meta_path = aoss_fs.join(scene_dir, "scene_iphone_metadata.npz")
    required = {
        "missing_images_dir": img_dir,
        "missing_masks_dir": mask_dir,
        "missing_metadata": meta_path,
    }
    for reason, path in required.items():
        if not aoss_fs.exists(path):
            return None, reason

    img_files = set(f for f in aoss_fs.listdir(img_dir) if f.endswith(".jpg"))
    mask_files = set(aoss_fs.listdir(mask_dir))
    meta = aoss_fs.load_npy(meta_path)
    meta_images = [str(f) for f in meta["images"].tolist()]
    valid_frames: list[str] = []
    valid_pose_indices: list[int] = []
    for meta_idx, frame_name in enumerate(meta_images):
        if frame_name in img_files and f"{frame_name}.npy" in mask_files:
            valid_frames.append(frame_name)
            valid_pose_indices.append(int(meta_idx))

    if len(valid_frames) < min_frames:
        return None, "not_enough_valid_frames"
    poses_c2w = meta["trajectories"][valid_pose_indices].astype(np.float32)
    return {
        "source": "scannetpp",
        "scene_id": scene_id,
        "scene_dir": scene_dir,
        "num_frames": len(valid_frames),
        "valid_frames": valid_frames,
        "valid_pose_indices": valid_pose_indices,
        "poses_c2w": poses_c2w,
    }, None


def expand_streaming_samples(
    scene: dict,
    split: str,
    prefix_lengths: list[int],
    action_horizons: list[int],
    motion_step: float,
    rotation_weight: float,
    window_stride: int,
    max_windows_per_scene: int,
    streaming_feature_root: str | None,
    target_feature_root: str | None,
) -> list[dict]:
    max_future = max(action_horizons, default=0)
    obs_len = max(prefix_lengths) + max_future
    windows = temporal_windows_from_poses(
        scene["poses_c2w"],
        observations_per_window=obs_len,
        motion_step=motion_step,
        rotation_weight=rotation_weight,
        window_stride=window_stride,
        max_windows_per_scene=max_windows_per_scene,
    )

    samples: list[dict] = []
    for window in windows:
        window_indices = [int(i) for i in window["indices"]]
        for prefix_len in prefix_lengths:
            tail = prefix_len - 1
            if tail + max_future >= len(window_indices):
                continue
            sample = {
                "sample_id": f"{scene['scene_id']}_w{int(window['window_id']):03d}_p{prefix_len}",
                "split": split,
                "source": scene["source"],
                "scene_id": scene["scene_id"],
                "scene_dir": scene["scene_dir"],
                "num_frames": int(scene["num_frames"]),
                "valid_frames": scene["valid_frames"],
                "valid_pose_indices": scene["valid_pose_indices"],
                "streaming_tail": tail,
                "streaming_tail_raw": int(window_indices[tail]),
                "streaming_indices": window_indices[:prefix_len],
                "streaming_window_indices": window_indices,
                "streaming_window_id": int(window["window_id"]),
                "streaming_obs_start": int(window.get("obs_start", 0)),
                "streaming_sampling": window.get("sampling", "motion_uniform"),
            }
            if streaming_feature_root:
                sample["streaming_feature_root"] = streaming_feature_root
            if target_feature_root:
                sample["target_feature_root"] = target_feature_root
            samples.append(sample)
    return samples


def streaming_feature_path(
    root: str,
    vfm_name: str,
    scene_id: str,
    window_id: int,
    tail: int,
    fname: str,
) -> str:
    return aoss_fs.join(
        root,
        vfm_name,
        "scannetpp",
        scene_id,
        f"window_{window_id:04d}",
        f"prefix_{tail:06d}",
        fname,
    )


def target_feature_paths(root: str, vfm_name: str, scene_id: str, fname: str) -> tuple[str, str]:
    scene_dir = aoss_fs.join(root, vfm_name, "scannetpp", scene_id)
    return aoss_fs.join(scene_dir, fname), aoss_fs.join(scene_dir, "target_indices.npy")


def feature_audit_reason(
    samples: list[dict],
    vfm_name: str,
    feature_layer: int | None,
    feature_timestep: int | None,
    streaming_feature_root: str | None,
    target_feature_root: str | None,
) -> str | None:
    if not samples:
        return "no_valid_prefix_sample"
    fname = feature_filename(
        vfm_name,
        feature_layer=feature_layer,
        feature_timestep=feature_timestep,
    )
    scene_id = samples[0]["scene_id"]
    if streaming_feature_root:
        scene_dir = aoss_fs.join(streaming_feature_root, vfm_name, "scannetpp", scene_id)
        if not aoss_fs.exists(aoss_fs.join(scene_dir, "prefix_index.npy")):
            return "missing_streaming_prefix_index"
        sample_keys = sorted(
            {
                (int(sample.get("streaming_window_id", 0)), int(sample["streaming_tail"]))
                for sample in samples
            }
        )
        for window_id, tail in sample_keys:
            path = streaming_feature_path(streaming_feature_root, vfm_name, scene_id, window_id, tail, fname)
            if not aoss_fs.exists(path):
                return "missing_streaming_feature"
    if target_feature_root:
        feat_path, index_path = target_feature_paths(target_feature_root, vfm_name, scene_id, fname)
        if not aoss_fs.exists(feat_path):
            return "missing_target_feature"
        if not aoss_fs.exists(index_path):
            return "missing_target_indices"
    return None


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-root",
        required=True,
        help="Directory that directly contains ScanNet++ scene directories.",
    )
    parser.add_argument("--output", required=True, help="Local JSONL manifest path.")
    parser.add_argument(
        "--known-scene",
        required=True,
        help="Scene id that must exist under --scene-root; prevents wrong roots.",
    )
    parser.add_argument("--prefix-lengths", default="8,12,16,24")
    parser.add_argument("--action-horizons", default="1,2,4")
    parser.add_argument("--streaming-motion-step", type=float, default=0.35)
    parser.add_argument("--streaming-rotation-weight", type=float, default=0.5)
    parser.add_argument("--streaming-window-stride", type=int, default=8)
    parser.add_argument("--streaming-max-windows-per-scene", type=int, default=4)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--force-split", default=None, help="Debug/controlled override for every scene split.")
    parser.add_argument("--streaming-feature-root", default=None)
    parser.add_argument("--target-feature-root", default=None)
    parser.add_argument("--vfm-name", default="dino")
    parser.add_argument("--feature-layer", type=int, default=0)
    parser.add_argument("--feature-timestep", type=int, default=749)
    parser.add_argument("--skip-feature-checks", action="store_true")
    parser.add_argument("--audit-dir", default=None)
    parser.add_argument("--max-scenes", type=int, default=0, help="Debug cap; <=0 keeps all scenes.")
    args = parser.parse_args()

    scene_root = args.scene_root.rstrip("/")
    known_path = aoss_fs.join(scene_root, args.known_scene)
    if not aoss_fs.exists(known_path):
        raise SystemExit(f"Invalid scene root: {scene_root}; missing {known_path}")

    prefix_lengths = parse_int_list(args.prefix_lengths)
    action_horizons = parse_int_list(args.action_horizons)
    if not prefix_lengths:
        raise SystemExit("--prefix-lengths must not be empty")
    min_frames = max(prefix_lengths) + max(action_horizons, default=0)

    scene_ids = sorted(
        scene_id
        for scene_id in aoss_fs.listdir(scene_root)
        if scene_id and not scene_id.startswith(".")
    )
    if args.max_scenes > 0:
        scene_ids = scene_ids[: args.max_scenes]
    split_map = load_split_map(args.split_manifest) or legacy_split(scene_ids, args.train_ratio)
    if args.force_split:
        split_map = {scene_id: args.force_split for scene_id in scene_ids}

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_dir = Path(args.audit_dir) if args.audit_dir else output.parent / f"{output.name}.audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    included_path = audit_dir / "included_scenes.txt"
    excluded_path = audit_dir / "excluded_scenes.jsonl"
    n_scenes = 0
    n_samples = 0
    split_counts: dict[str, int] = {}
    sample_ids: set[str] = set()
    duplicate_sample_ids: list[str] = []
    with output.open("w") as f, included_path.open("w") as included, excluded_path.open("w") as excluded:
        for scene_id in scene_ids:
            scene, reason = collect_scannetpp_scene(scene_root, scene_id, min_frames=min_frames)
            if scene is None:
                excluded.write(json.dumps({"scene_id": scene_id, "reason": reason}) + "\n")
                continue
            split = split_map.get(scene_id, "train")
            samples = expand_streaming_samples(
                scene,
                split=split,
                prefix_lengths=prefix_lengths,
                action_horizons=action_horizons,
                motion_step=args.streaming_motion_step,
                rotation_weight=args.streaming_rotation_weight,
                window_stride=args.streaming_window_stride,
                max_windows_per_scene=args.streaming_max_windows_per_scene,
                streaming_feature_root=args.streaming_feature_root,
                target_feature_root=args.target_feature_root,
            )
            reason = None
            if not samples:
                reason = "no_valid_prefix_sample"
            elif not args.skip_feature_checks:
                reason = feature_audit_reason(
                    samples,
                    vfm_name=args.vfm_name,
                    feature_layer=args.feature_layer,
                    feature_timestep=args.feature_timestep,
                    streaming_feature_root=args.streaming_feature_root,
                    target_feature_root=args.target_feature_root,
                )
            if reason:
                excluded.write(json.dumps({"scene_id": scene_id, "reason": reason}) + "\n")
                continue

            n_scenes += 1
            included.write(f"{scene_id}\n")
            for sample in samples:
                sample_id = sample["sample_id"]
                if sample_id in sample_ids:
                    duplicate_sample_ids.append(sample_id)
                sample_ids.add(sample_id)
                f.write(json.dumps(sample, separators=(",", ":")) + "\n")
                split_counts[sample["split"]] = split_counts.get(sample["split"], 0) + 1
            n_samples += len(samples)

    checksum = sha256_file(output)
    metadata = {
        "manifest_version": 1,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "output": str(output),
        "sha256": checksum,
        "scene_root": scene_root,
        "streaming_feature_root": args.streaming_feature_root,
        "target_feature_root": args.target_feature_root,
        "vfm_name": args.vfm_name,
        "feature_layer": args.feature_layer,
        "feature_timestep": args.feature_timestep,
        "prefix_lengths": prefix_lengths,
        "action_horizons": action_horizons,
        "num_candidate_scenes": len(scene_ids),
        "num_included_scenes": n_scenes,
        "num_samples": n_samples,
        "split_counts": split_counts,
        "duplicate_sample_ids": duplicate_sample_ids[:20],
        "num_duplicate_sample_ids": len(duplicate_sample_ids),
        "audit_dir": str(audit_dir),
        "included_scenes": str(included_path),
        "excluded_scenes": str(excluded_path),
    }
    meta_path = output.with_suffix(output.suffix + ".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"wrote {n_samples} samples from {n_scenes} scenes to {output}")
    print(f"sha256 {checksum}")
    print(f"audit_dir {audit_dir}")


if __name__ == "__main__":
    main()
