#!/usr/bin/env python3
"""Build deterministic, scene-disjoint inventories for Wan p8 recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CANONICAL_SHA256 = "772194f077c45cc85a942f17f46f50103125ae8f5b1c983c08881c1759607e0c"
EXPECTED_SPLITS = {"train": 2448, "val": 276}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scene_id(row: dict[str, Any]) -> str:
    return str(row.get("scene_id") or Path(str(row["scene_dir"])).name)


def target_raw(row: dict[str, Any]) -> tuple[int, str]:
    for field in ("target_raw_id", "target_raw_index", "target_frame_idx", "target_index"):
        if row.get(field) is not None:
            return int(row[field]), field
    position = int(row["streaming_tail"]) + 1
    return int(row["streaming_window_indices"][position]), f"streaming_window_indices[{position}]"


def write_text(path: Path, value: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(value)
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-shards", type=int, default=6)
    args = parser.parse_args()

    manifest = Path(args.manifest).resolve()
    out_dir = Path(args.out_dir).resolve()
    if sha256(manifest) != CANONICAL_SHA256:
        raise SystemExit("Canonical manifest SHA256 mismatch")
    if args.num_shards != 6:
        raise SystemExit("Wan p8 recovery requires exactly six shards")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"Refusing non-empty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    sample_ids = [str(row["sample_id"]) for row in rows]
    split_counts = Counter(str(row["split"]) for row in rows)
    if len(rows) != 2724 or len(set(sample_ids)) != 2724:
        raise SystemExit(f"Expected 2724 unique rows, got rows={len(rows)} unique={len(set(sample_ids))}")
    if dict(split_counts) != EXPECTED_SPLITS:
        raise SystemExit(f"Unexpected split counts: {dict(split_counts)}")

    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scene[scene_id(row)].append(row)
    scenes = sorted(by_scene)
    if len(scenes) != 681:
        raise SystemExit(f"Expected 681 scenes, got {len(scenes)}")

    mapping_lines = []
    for row in rows:
        target, source = target_raw(row)
        mapping_lines.append(
            json.dumps(
                {
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "scene_id": scene_id(row),
                    "streaming_window_id": int(row["streaming_window_id"]),
                    "streaming_tail": int(row["streaming_tail"]),
                    "streaming_raw_ids": [int(value) for value in row["streaming_indices"]],
                    "canonical_window_raw_ids": [int(value) for value in row["streaming_window_indices"]],
                    "target_raw_id": target,
                    "target_source": source,
                },
                sort_keys=True,
            )
        )
    mapping_path = out_dir / "sample_to_raw_id_mapping.jsonl"
    write_text(mapping_path, "\n".join(mapping_lines) + "\n")

    base, remainder = divmod(len(scenes), args.num_shards)
    offset = 0
    shard_records = []
    seen_scenes: set[str] = set()
    seen_samples: set[str] = set()
    for shard in range(args.num_shards):
        count = base + (1 if shard < remainder else 0)
        shard_scenes = scenes[offset : offset + count]
        offset += count
        shard_scene_set = set(shard_scenes)
        shard_rows = [row for row in rows if scene_id(row) in shard_scene_set]
        shard_samples = [str(row["sample_id"]) for row in shard_rows]
        overlap_scenes = seen_scenes & shard_scene_set
        overlap_samples = seen_samples & set(shard_samples)
        if overlap_scenes or overlap_samples:
            raise SystemExit(f"Shard overlap detected for shard {shard}")
        seen_scenes.update(shard_scene_set)
        seen_samples.update(shard_samples)

        selection_path = out_dir / f"shard_{shard:02d}_sample_ids.txt"
        write_text(selection_path, "\n".join(shard_samples) + "\n")
        record = {
            "shard": shard,
            "scene_count": len(shard_scenes),
            "sample_count": len(shard_rows),
            "split_counts": dict(Counter(str(row["split"]) for row in shard_rows)),
            "scenes": shard_scenes,
            "selection_file": str(selection_path),
            "selection_sha256": sha256(selection_path),
            "first_scene": shard_scenes[0],
            "last_scene": shard_scenes[-1],
        }
        shard_path = out_dir / f"shard_{shard:02d}_inventory.json"
        write_text(shard_path, json.dumps(record, indent=2, sort_keys=True) + "\n")
        record["inventory_file"] = str(shard_path)
        record["inventory_sha256"] = sha256(shard_path)
        shard_records.append(record)

    if seen_scenes != set(scenes) or seen_samples != set(sample_ids):
        raise SystemExit("Shard union does not equal canonical manifest")

    result = {
        "ok": True,
        "schema_version": 1,
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "sample_count": len(rows),
        "scene_count": len(scenes),
        "split_counts": dict(split_counts),
        "num_shards": args.num_shards,
        "scene_disjoint": True,
        "sample_disjoint": True,
        "union_sample_count": len(seen_samples),
        "mapping_file": str(mapping_path),
        "mapping_sha256": sha256(mapping_path),
        "shards": shard_records,
    }
    inventory_path = out_dir / "full_shard_inventory.json"
    write_text(inventory_path, json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": True, "scenes": len(scenes), "samples": len(rows), "shards": [item["sample_count"] for item in shard_records]}))


if __name__ == "__main__":
    main()
