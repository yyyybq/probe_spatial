#!/usr/bin/env python3
"""Derive and freeze d_series_scannetpp_v1 from canonical-v2 prefix-8 rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


PROTOCOL = "d_series_scannetpp_v1"
PARENT_PROTOCOL = "inscene_streaming_canonical_v2_obs28_h124"
PARENT_SHA = "20976fc86abfdc96d40446a0fe15fc32af699369309c540f9a352e868ce89f91"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def frame_raw_id(name: str) -> int:
    return int(Path(name).stem.split("_")[-1])


def rotation_degrees(a: np.ndarray, b: np.ndarray) -> float:
    rel = a[:3, :3].T @ b[:3, :3]
    cosine = np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def relative_target_from_context(context_c2w: np.ndarray, target_c2w: np.ndarray) -> np.ndarray:
    """Transform points from the context camera into the target camera."""
    return np.linalg.inv(target_c2w) @ context_c2w


def directed_depth_overlap(
    context_depth_path: Path,
    target_depth_path: Path,
    context_k: np.ndarray,
    target_k: np.ndarray,
    context_c2w: np.ndarray,
    target_c2w: np.ndarray,
    stride: int = 8,
) -> float:
    """Fraction of sampled context depth points visible in the target view."""
    dc = np.asarray(Image.open(context_depth_path), dtype=np.float32) / 1000.0
    dt = np.asarray(Image.open(target_depth_path), dtype=np.float32) / 1000.0
    vv, uu = np.mgrid[0 : dc.shape[0] : stride, 0 : dc.shape[1] : stride]
    z = dc[::stride, ::stride]
    valid = np.isfinite(z) & (z > 0)
    if not valid.any():
        return float("nan")
    uv1 = np.stack((uu[valid], vv[valid], np.ones(valid.sum())), axis=1)
    xyz_context = (uv1 @ np.linalg.inv(context_k).T) * z[valid, None]
    rel = relative_target_from_context(context_c2w, target_c2w)
    xyz_target = xyz_context @ rel[:3, :3].T + rel[:3, 3]
    positive = xyz_target[:, 2] > 1e-6
    proj = xyz_target @ target_k.T
    u = np.rint(proj[:, 0] / np.maximum(proj[:, 2], 1e-6)).astype(np.int64)
    v = np.rint(proj[:, 1] / np.maximum(proj[:, 2], 1e-6)).astype(np.int64)
    inside = positive & (u >= 0) & (u < dt.shape[1]) & (v >= 0) & (v < dt.shape[0])
    visible = np.zeros_like(inside)
    indices = np.flatnonzero(inside)
    if indices.size:
        measured = dt[v[indices], u[indices]]
        predicted = xyz_target[indices, 2]
        tolerance = np.maximum(0.10, 0.05 * predicted)
        visible[indices] = (measured > 0) & (np.abs(measured - predicted) <= tolerance)
    return float(visible.sum() / valid.sum())


def load_original_roles(path: Path):
    payload = json.loads(path.read_text())
    roles = {}
    for split, values in payload["splits"].items():
        for value in values:
            if value.startswith("scannetpp/"):
                roles[value.split("/", 1)[1]] = split
    return roles


def build(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    original_roles = load_original_roles(args.original_split)
    records, exclusions = [], []
    scene_roles, scene_windows = {}, Counter()
    metadata_cache = {}

    with args.parent_manifest.open() as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if row.get("canonical_protocol") != PARENT_PROTOCOL or int(row["prefix_length"]) != 8:
                raise ValueError(f"unexpected parent row {line_number}")
            scene = row["scene_id"]
            split = "dev" if row["split"] == "val" else row["split"]
            scene_roles[scene] = split
            scene_windows[(scene, int(row["streaming_window_id"]))] += 1
            scene_dir = args.scene_root / scene
            metadata_path = scene_dir / "scene_iphone_metadata.npz"
            if scene not in metadata_cache:
                meta = np.load(metadata_path, allow_pickle=False)
                metadata_cache[scene] = {
                    "images": [str(v) for v in meta["images"]],
                    "trajectories": np.asarray(meta["trajectories"], dtype=np.float64),
                    "intrinsics": np.asarray(meta["intrinsics"], dtype=np.float64),
                }
            meta = metadata_cache[scene]
            meta_lookup = {name: i for i, name in enumerate(meta["images"])}
            obs = [int(v) for v in row["streaming_window_indices"]]
            if len(obs) != 28:
                exclusions.append({"scene_id": scene, "parent_sample_id": row["sample_id"], "reason": "not_obs28"})
                continue
            names = row["valid_frames"]
            prefix_names = [names[i] for i in obs[:8]]
            context_name = prefix_names[-1]
            for horizon in (2, 4):
                target_obs = 7 + horizon
                target_name = names[obs[target_obs]]
                if context_name not in meta_lookup or target_name not in meta_lookup:
                    exclusions.append({"scene_id": scene, "parent_sample_id": row["sample_id"], "horizon": horizon, "reason": "metadata_frame_missing"})
                    continue
                ci, ti = meta_lookup[context_name], meta_lookup[target_name]
                context_pose, target_pose = meta["trajectories"][ci], meta["trajectories"][ti]
                context_k, target_k = meta["intrinsics"][ci], meta["intrinsics"][ti]
                context_depth = scene_dir / "depth" / f"{Path(context_name).stem}.png"
                target_depth = scene_dir / "depth" / f"{Path(target_name).stem}.png"
                if not context_depth.is_file() or not target_depth.is_file():
                    exclusions.append({"scene_id": scene, "parent_sample_id": row["sample_id"], "horizon": horizon, "reason": "depth_missing"})
                    continue
                rel = relative_target_from_context(context_pose, target_pose)
                overlap = directed_depth_overlap(
                    context_depth, target_depth, context_k, target_k, context_pose, target_pose,
                    stride=args.overlap_stride,
                )
                sample_id = f"{row['sample_id']}_h{horizon}"
                records.append({
                    "protocol": PROTOCOL,
                    "sample_id": sample_id,
                    "parent_sample_id": row["sample_id"],
                    "split": split,
                    "scene_id": scene,
                    "window_id": int(row["streaming_window_id"]),
                    "prefix": 8,
                    "horizon": horizon,
                    "context_observation_index": 7,
                    "target_observation_index": target_obs,
                    "context_frame_index": int(obs[7]),
                    "target_frame_index": int(obs[target_obs]),
                    "prefix_frame_indices": obs[:8],
                    "prefix_frame_names": prefix_names,
                    "prefix_raw_frame_ids": [frame_raw_id(v) for v in prefix_names],
                    "context_frame_name": context_name,
                    "target_frame_name": target_name,
                    "context_raw_frame_id": frame_raw_id(context_name),
                    "target_raw_frame_id": frame_raw_id(target_name),
                    "raw_frame_delta": frame_raw_id(target_name) - frame_raw_id(context_name),
                    "timestamp_delta_seconds": None,
                    "translation_m": float(np.linalg.norm(target_pose[:3, 3] - context_pose[:3, 3])),
                    "rotation_deg": rotation_degrees(context_pose, target_pose),
                    "directed_depth_overlap": overlap,
                    "motion_target_from_context_3x4": rel[:3].reshape(-1).tolist(),
                    "original_inscene_split": original_roles.get(scene, "unlisted"),
                    "scene_dir_rel": scene,
                    "rgb_rel": f"{scene}/images/{target_name}",
                    "depth_rel": f"{scene}/depth/{Path(target_name).stem}.png",
                    "metadata_rel": f"{scene}/scene_iphone_metadata.npz",
                })

    records.sort(key=lambda x: (x["scene_id"], x["window_id"], x["horizon"]))
    manifest = out / "sample_manifest.jsonl"
    with manifest.open("w") as handle:
        for row in records:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    ledger = out / "exclusion_ledger.jsonl"
    with ledger.open("w") as handle:
        for row in exclusions:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    scenes_by_split = defaultdict(list)
    for scene, split in sorted(scene_roles.items()): scenes_by_split[split].append(scene)
    scenes_by_split.setdefault("test", [])
    for split in ("train", "dev", "test"):
        (out / f"{split}_scenes.txt").write_text("".join(f"{v}\n" for v in scenes_by_split[split]))
    (out / "scene_list.txt").write_text("".join(f"{v}\n" for v in sorted(scene_roles)))

    history = out / "scene_historical_use.csv"
    with history.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scene_id", "d_series_split", "canonical_c1_use", "original_inscene_split", "never_used"], lineterminator="\n")
        writer.writeheader()
        for scene, split in sorted(scene_roles.items()):
            writer.writerow({
                "scene_id": scene,
                "d_series_split": split,
                "canonical_c1_use": "train" if split == "train" else "validation_or_pilot",
                "original_inscene_split": original_roles.get(scene, "unlisted"),
                "never_used": False,
            })

    counts = Counter((r["split"], r["horizon"]) for r in records)
    summary = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "evaluation_status": "development_only_no_historically_unused_test_scenes",
        "parent_manifest": str(args.parent_manifest),
        "parent_manifest_expected_sha256": PARENT_SHA,
        "parent_manifest_actual_sha256": sha256(args.parent_manifest),
        "scene_counts": {k: len(v) for k, v in sorted(scenes_by_split.items())},
        "sample_counts": {f"{s}_h{h}": n for (s, h), n in sorted(counts.items())},
        "samples": len(records),
        "scenes": len(scene_roles),
        "exclusions": len(exclusions),
        "overlap_definition": {
            "direction": "context_to_target",
            "sample_stride_pixels": args.overlap_stride,
            "visibility_tolerance_m": "max(0.10,0.05*z_target_prediction)",
        },
    }
    (out / "manifest_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if summary["parent_manifest_actual_sha256"] != PARENT_SHA:
        raise ValueError("parent manifest hash changed")

    frozen = [
        manifest, ledger, history, out / "scene_list.txt", out / "train_scenes.txt",
        out / "dev_scenes.txt", out / "test_scenes.txt", out / "manifest_summary.json",
    ]
    (out / "hashes.sha256").write_text("".join(f"{sha256(p)}  {p.name}\n" for p in frozen))
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-manifest", type=Path, default=Path("artifacts/tstep_canonical_v2/manifests/inscene_c_h124_prefix8.jsonl"))
    parser.add_argument("--scene-root", type=Path, default=Path("/mnt/umm/users/yinbaiqiao/InsScene-15K/data/processed_scannetpp_v2/processed_scannetpp_v2"))
    parser.add_argument("--original-split", type=Path, default=Path("configs/splits/inscene15k_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("experiments/d_series_scannetpp_v1/data"))
    parser.add_argument("--overlap-stride", type=int, default=8)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
