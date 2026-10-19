#!/usr/bin/env python3
"""Build a VFM-specific C manifest by reusing an audited base manifest.

This is faster than rescanning raw scene metadata for each VFM. It preserves the
sample/split/window records from an existing C manifest, rewrites the
streaming/target feature roots, and audits that every included scene has the
required streaming and target feature files for the requested VFM/layer.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from vidfm3d.utils import aoss_fs
from vidfm3d.utils.feature_layers import feature_filename


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vfm-name", required=True)
    parser.add_argument("--feature-layer", type=int, required=True)
    parser.add_argument("--feature-timestep", type=int, default=749)
    parser.add_argument("--prefix-length", type=int, required=True)
    parser.add_argument("--streaming-feature-root", required=True)
    parser.add_argument("--target-feature-root", required=True)
    parser.add_argument("--audit-dir", default=None)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    start = time.time()
    base_manifest = Path(args.base_manifest)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_dir = Path(args.audit_dir) if args.audit_dir else output.parent / f"{output.name}.audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    fname = feature_filename(
        args.vfm_name,
        feature_layer=args.feature_layer,
        feature_timestep=args.feature_timestep,
    )
    records = [json.loads(line) for line in base_manifest.open()]
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_scene[str(record["scene_id"])].append(record)

    def check_scene(item: tuple[str, list[dict]]) -> tuple[str, str | None]:
        scene_id, scene_records = item
        target_feat = aoss_fs.join(
            args.target_feature_root,
            args.vfm_name,
            "scannetpp",
            scene_id,
            fname,
        )
        target_index = aoss_fs.join(
            args.target_feature_root,
            args.vfm_name,
            "scannetpp",
            scene_id,
            "target_indices.npy",
        )
        if not aoss_fs.exists(target_feat):
            return scene_id, "missing_target_feature"
        if not aoss_fs.exists(target_index):
            return scene_id, "missing_target_indices"

        for record in scene_records:
            feature_path = aoss_fs.join(
                args.streaming_feature_root,
                args.vfm_name,
                "scannetpp",
                scene_id,
                f"window_{int(record.get('streaming_window_id', 0)):04d}",
                f"prefix_{int(record['streaming_tail']):06d}",
                fname,
            )
            if not aoss_fs.exists(feature_path):
                return scene_id, "missing_streaming_feature"
        return scene_id, None

    missing: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(check_scene, item) for item in by_scene.items()]
        for i, future in enumerate(as_completed(futures), 1):
            scene_id, reason = future.result()
            if reason:
                missing[scene_id] = reason
            if i % 100 == 0:
                print(f"checked {i}/{len(futures)} scenes; missing={len(missing)}", flush=True)

    included_records: list[dict] = []
    with output.open("w") as f:
        for record in records:
            if str(record["scene_id"]) in missing:
                continue
            rewritten = dict(record)
            rewritten["streaming_feature_root"] = args.streaming_feature_root
            rewritten["target_feature_root"] = args.target_feature_root
            f.write(json.dumps(rewritten, separators=(",", ":")) + "\n")
            included_records.append(rewritten)

    included_scenes = sorted({str(record["scene_id"]) for record in included_records})
    (audit_dir / "included_scenes.txt").write_text("\n".join(included_scenes) + "\n")
    with (audit_dir / "excluded_scenes.jsonl").open("w") as f:
        for scene_id, reason in sorted(missing.items()):
            f.write(json.dumps({"scene_id": scene_id, "reason": reason}) + "\n")

    split_counts: dict[str, int] = {}
    duplicate_sample_ids: list[str] = []
    sample_ids: set[str] = set()
    for record in included_records:
        split_counts[str(record["split"])] = split_counts.get(str(record["split"]), 0) + 1
        sample_id = str(record.get("sample_id", ""))
        if sample_id in sample_ids:
            duplicate_sample_ids.append(sample_id)
        sample_ids.add(sample_id)

    metadata = {
        "manifest_version": 2,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_manifest": str(base_manifest),
        "output": str(output),
        "sha256": sha256_file(output),
        "vfm_name": args.vfm_name,
        "feature_layer": args.feature_layer,
        "feature_timestep": args.feature_timestep,
        "feature_filename": fname,
        "prefix_lengths": [args.prefix_length],
        "action_horizons": [1, 2, 4],
        "streaming_feature_root": args.streaming_feature_root,
        "target_feature_root": args.target_feature_root,
        "num_candidate_scenes": len(by_scene),
        "num_included_scenes": len(included_scenes),
        "num_excluded_scenes": len(missing),
        "num_samples": len(included_records),
        "split_counts": split_counts,
        "num_duplicate_sample_ids": len(duplicate_sample_ids),
        "duplicate_sample_ids": duplicate_sample_ids[:20],
        "audit_dir": str(audit_dir),
        "included_scenes": str(audit_dir / "included_scenes.txt"),
        "excluded_scenes": str(audit_dir / "excluded_scenes.jsonl"),
        "elapsed_sec": round(time.time() - start, 1),
    }
    output.with_suffix(output.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
