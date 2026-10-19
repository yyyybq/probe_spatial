#!/usr/bin/env python3
"""Prove that a Wan streaming-prefix cache used exact canonical input frames."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_MANIFEST_SHA = "772194f077c45cc85a942f17f46f50103125ae8f5b1c983c08881c1759607e0c"
EXPECTED_COUNTS = {"train": 2448, "val": 276}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def scene_id(row: dict[str, Any]) -> str:
    return str(row.get("scene_id") or Path(str(row["scene_dir"])).name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--streaming-root", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    manifest = Path(args.manifest).resolve()
    root = Path(args.streaming_root).resolve()
    out_path = Path(args.out_json).resolve()
    rows = read_manifest(manifest)
    split_counts = Counter(str(row["split"]) for row in rows)
    problems: list[dict[str, Any]] = []
    metadata_cache: dict[Path, dict[tuple[int, int], dict[str, Any]]] = {}
    matched = Counter()

    if sha256(manifest) != EXPECTED_MANIFEST_SHA:
        problems.append({"error": "manifest_sha256_mismatch", "actual": sha256(manifest)})
    if dict(split_counts) != EXPECTED_COUNTS:
        problems.append({"error": "manifest_split_count_mismatch", "actual": dict(split_counts)})

    for row in rows:
        scene = scene_id(row)
        source = str(row.get("source", "scannetpp"))
        meta_path = root / "wan" / source / scene / "prefix_index.npy"
        if meta_path not in metadata_cache:
            try:
                records = np.load(meta_path, allow_pickle=True).tolist()
                metadata_cache[meta_path] = {
                    (int(record["window_id"]), int(record["tail"])): record
                    for record in records
                }
            except Exception as exc:
                problems.append({
                    "sample_id": row.get("sample_id"),
                    "error": "unreadable_prefix_index",
                    "path": str(meta_path),
                    "detail": repr(exc),
                })
                metadata_cache[meta_path] = {}

        key = (int(row["streaming_window_id"]), int(row["streaming_tail"]))
        record = metadata_cache[meta_path].get(key)
        if record is None:
            problems.append({
                "sample_id": row.get("sample_id"),
                "error": "missing_prefix_record",
                "path": str(meta_path),
                "key": list(key),
            })
            continue

        tail = int(row["streaming_tail"])
        manifest_window = [int(value) for value in row["streaming_window_indices"]]
        manifest_prefix = [int(value) for value in row.get("streaming_indices", manifest_window[: tail + 1])]
        actual_indices = [int(value) for value in record.get("indices", [])]
        actual_window = [int(value) for value in record.get("window_indices", actual_indices)]
        checks = {
            "manifest_prefix_is_window_prefix": manifest_prefix == manifest_window[: tail + 1],
            "actual_indices_match_manifest_prefix": actual_indices == manifest_prefix,
            "actual_window_matches_manifest_prefix": actual_window == manifest_window[: len(actual_window)],
            "valid_length_matches": int(record.get("valid_length", -1)) == len(manifest_prefix),
            "input_length_is_81": int(record.get("input_length", -1)) == 81,
            "pad_mode_repeat_tail": record.get("pad_mode") == "repeat_tail",
        }
        if all(checks.values()):
            matched[str(row["split"])] += 1
        else:
            problems.append({
                "sample_id": row.get("sample_id"),
                "scene_id": scene,
                "checks": checks,
                "manifest_prefix": manifest_prefix,
                "manifest_window": manifest_window,
                "actual_indices": actual_indices,
                "actual_window": actual_window,
                "record": record,
            })

    result = {
        "ok": not problems and dict(matched) == EXPECTED_COUNTS,
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "streaming_root": str(root),
        "rows": len(rows),
        "split_counts": dict(split_counts),
        "matched": dict(matched),
        "metadata_files_checked": len(metadata_cache),
        "problem_count": len(problems),
        "problems": problems,
        "semantic_rule": (
            "prefix_index record.indices must equal manifest streaming_indices and "
            "record.window_indices must equal the same leading portion of "
            "manifest streaming_window_indices"
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("ok", "rows", "matched", "problem_count")}))
    raise SystemExit(0 if result["ok"] else 2)


if __name__ == "__main__":
    main()
