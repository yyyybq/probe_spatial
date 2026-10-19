#!/usr/bin/env python3
"""Fail-closed full feature audit for Wan p8 manifest-exact recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open


CANONICAL_SHA256 = "772194f077c45cc85a942f17f46f50103125ae8f5b1c983c08881c1759607e0c"
LAYERS = [0, 5, 10, 15, 25, 29]
EXPECTED_SPLITS = {"train": 2448, "val": 276}
BAD_LOG = re.compile(r"Killed|out of memory|DataLoader worker.*Terminated|Traceback|uncaught exception", re.I)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scene_id(row: dict[str, Any]) -> str:
    return str(row.get("scene_id") or Path(str(row["scene_dir"])).name)


def target_raw(row: dict[str, Any]) -> int:
    for field in ("target_raw_id", "target_raw_index", "target_frame_idx", "target_index"):
        if row.get(field) is not None:
            return int(row[field])
    return int(row["streaming_window_indices"][int(row["streaming_tail"]) + 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--streaming-root", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--shard-inventory", required=True)
    parser.add_argument("--status-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--check-finite", action="store_true")
    args = parser.parse_args()

    manifest = Path(args.manifest).resolve()
    streaming_root = Path(args.streaming_root).resolve()
    target_root = Path(args.target_root).resolve()
    inventory_path = Path(args.shard_inventory).resolve()
    status_dir = Path(args.status_dir).resolve()
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    rows_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_scene[scene_id(row)].append(row)

    problems: list[dict[str, Any]] = []
    problem_count = 0

    def add(kind: str, **detail: Any) -> None:
        nonlocal problem_count
        problem_count += 1
        if len(problems) < 500:
            problems.append({"kind": kind, **detail})

    manifest_sha = sha256(manifest)
    split_counts = Counter(str(row["split"]) for row in rows)
    if manifest_sha != CANONICAL_SHA256:
        add("manifest_sha256", actual=manifest_sha)
    if len(rows) != 2724 or dict(split_counts) != EXPECTED_SPLITS:
        add("manifest_counts", rows=len(rows), splits=dict(split_counts))

    inventory = json.loads(inventory_path.read_text())
    if not (
        inventory.get("ok") is True
        and inventory.get("manifest_sha256") == CANONICAL_SHA256
        and inventory.get("sample_count") == 2724
        and inventory.get("scene_count") == 681
        and inventory.get("scene_disjoint") is True
        and inventory.get("sample_disjoint") is True
        and inventory.get("union_sample_count") == 2724
    ):
        add("shard_inventory_contract", inventory=str(inventory_path))

    mapping_path = Path(inventory.get("mapping_file", ""))
    if not mapping_path.is_file() or sha256(mapping_path) != inventory.get("mapping_sha256"):
        add("mapping_hash", path=str(mapping_path))
        mapping = {}
    else:
        mapping_rows = [json.loads(line) for line in mapping_path.read_text().splitlines() if line.strip()]
        mapping = {item["sample_id"]: item for item in mapping_rows}
        if len(mapping_rows) != 2724 or len(mapping) != 2724:
            add("mapping_count", rows=len(mapping_rows), unique=len(mapping))

    union_samples: set[str] = set()
    union_scenes: set[str] = set()
    for shard in inventory.get("shards", []):
        selection = Path(shard["selection_file"])
        shard_ids = {line.strip() for line in selection.read_text().splitlines() if line.strip()}
        shard_scenes = set(shard["scenes"])
        if sha256(selection) != shard["selection_sha256"]:
            add("selection_hash", shard=shard["shard"], path=str(selection))
        if union_samples & shard_ids or union_scenes & shard_scenes:
            add("shard_overlap", shard=shard["shard"])
        union_samples.update(shard_ids)
        union_scenes.update(shard_scenes)
        status_path = status_dir / f"shard_{int(shard['shard']):02d}.status.json"
        try:
            status = json.loads(status_path.read_text())
        except Exception as exc:
            add("shard_status", shard=shard["shard"], detail=repr(exc))
            status = {}
        if status.get("rc") != 0:
            add("shard_return_code", shard=shard["shard"], status=status)
        log_path = status_dir / f"shard_{int(shard['shard']):02d}.log"
        log_text = log_path.read_text(errors="replace") if log_path.is_file() else ""
        if "SHARD PASS" not in log_text or BAD_LOG.search(log_text):
            add("shard_log", shard=shard["shard"], path=str(log_path))

    expected_ids = {str(row["sample_id"]) for row in rows}
    expected_scenes = set(rows_by_scene)
    if union_samples != expected_ids or union_scenes != expected_scenes:
        add("shard_union", samples=len(union_samples), scenes=len(union_scenes))

    for row in rows:
        item = mapping.get(row["sample_id"])
        expected_stream = [int(value) for value in row["streaming_indices"]]
        if item is None or item.get("streaming_raw_ids") != expected_stream or item.get("target_raw_id") != target_raw(row):
            add("mapping_semantics", sample_id=row["sample_id"])

    stream_scene_root = streaming_root / "wan" / "scannetpp"
    target_scene_root = target_root / "wan" / "scannetpp"
    actual_stream_scenes = {path.name for path in stream_scene_root.iterdir() if path.is_dir()}
    actual_target_scenes = {path.name for path in target_scene_root.iterdir() if path.is_dir()}
    if actual_stream_scenes != expected_scenes:
        add("streaming_scene_set", actual=len(actual_stream_scenes), expected=len(expected_scenes))
    if actual_target_scenes != expected_scenes:
        add("target_scene_set", actual=len(actual_target_scenes), expected=len(expected_scenes))

    file_counts = {
        "streaming_sft": sum(1 for _ in streaming_root.rglob("*.sft")),
        "streaming_sidecars": sum(1 for _ in streaming_root.rglob("*.sft.manifest.json")),
        "target_sft": sum(1 for _ in target_root.rglob("*.sft")),
        "target_sidecars": sum(1 for _ in target_root.rglob("*.sft.manifest.json")),
        "prefix_indices": sum(1 for _ in streaming_root.rglob("prefix_index.npy")),
        "target_indices": sum(1 for _ in target_root.rglob("target_indices.npy")),
    }
    expected_file_counts = {
        "streaming_sft": 2724 * 6,
        "streaming_sidecars": 2724 * 6,
        "target_sft": 681 * 6,
        "target_sidecars": 681 * 6,
        "prefix_indices": 681,
        "target_indices": 681,
    }
    if file_counts != expected_file_counts:
        add("file_counts", actual=file_counts, expected=expected_file_counts)

    tensors_checked = 0
    finite_checked = 0

    def tensor_ok(path: Path, expected_shape: list[int]) -> bool:
        nonlocal tensors_checked, finite_checked
        if not path.is_file() or path.stat().st_size <= 0:
            add("missing_or_empty_tensor", path=str(path))
            return False
        try:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                keys = list(handle.keys())
                if keys != ["feat"]:
                    add("tensor_keys", path=str(path), keys=keys)
                    return False
                view = handle.get_slice("feat")
                shape = list(view.get_shape())
                dtype = str(view.get_dtype())
                if shape != expected_shape or dtype != "F16":
                    add("tensor_header", path=str(path), shape=shape, dtype=dtype, expected=expected_shape)
                    return False
                tensors_checked += 1
                if args.check_finite:
                    value = handle.get_tensor("feat")
                    if not bool(torch.isfinite(value).all().item()):
                        add("tensor_nonfinite", path=str(path))
                        return False
                    finite_checked += 1
                    if finite_checked % 250 == 0:
                        print(json.dumps({"finite_files_checked": finite_checked}), flush=True)
        except Exception as exc:
            add("tensor_unreadable", path=str(path), detail=repr(exc))
            return False
        return True

    target_scene_valid: dict[tuple[str, int], bool] = {}
    target_index_values: dict[str, list[int]] = {}
    for scene, scene_rows in sorted(rows_by_scene.items()):
        expected_targets = sorted({target_raw(row) for row in scene_rows})
        index_path = target_scene_root / scene / "target_indices.npy"
        try:
            actual_targets = np.load(index_path).astype(np.int64).tolist()
        except Exception as exc:
            add("target_index_unreadable", scene=scene, path=str(index_path), detail=repr(exc))
            actual_targets = []
        target_index_values[scene] = actual_targets
        if actual_targets != expected_targets:
            add("target_index_exact", scene=scene, expected=expected_targets, actual=actual_targets)
        for layer in LAYERS:
            path = target_scene_root / scene / f"feature_t749_layer{layer}.sft"
            valid = tensor_ok(path, [len(expected_targets), 30, 52, 1536])
            try:
                meta = json.loads(Path(f"{path}.manifest.json").read_text())
                valid = valid and (
                    meta.get("canonical_manifest_sha256") == CANONICAL_SHA256
                    and meta.get("mode") == "manifest_exact_target"
                    and meta.get("motion_window_generator_used") is False
                    and meta.get("target_raw_ids") == expected_targets
                    and meta.get("size_bytes") == path.stat().st_size
                )
                if not valid:
                    add("target_sidecar_contract", scene=scene, layer=layer, path=str(path))
            except Exception as exc:
                valid = False
                add("target_sidecar_unreadable", scene=scene, layer=layer, path=str(path), detail=repr(exc))
            target_scene_valid[(scene, layer)] = valid

    streaming_coverage = {layer: Counter() for layer in LAYERS}
    target_coverage = {layer: Counter() for layer in LAYERS}
    streaming_exact = 0
    target_exact = 0
    for scene, scene_rows in sorted(rows_by_scene.items()):
        prefix_path = stream_scene_root / scene / "prefix_index.npy"
        try:
            prefix_records = np.load(prefix_path, allow_pickle=True).tolist()
            prefix_by_sample = {item["sample_id"]: item for item in prefix_records}
        except Exception as exc:
            add("prefix_index_unreadable", scene=scene, path=str(prefix_path), detail=repr(exc))
            prefix_by_sample = {}
        if set(prefix_by_sample) != {row["sample_id"] for row in scene_rows}:
            add("prefix_index_sample_set", scene=scene)

        for row in scene_rows:
            split = str(row["split"])
            sample = str(row["sample_id"])
            expected_stream = [int(value) for value in row["streaming_indices"]]
            record = prefix_by_sample.get(sample, {})
            stream_raw_ok = record.get("actual_input_raw_ids") == expected_stream
            all_stream_layers = True
            window = int(row["streaming_window_id"])
            tail = int(row["streaming_tail"])
            for layer in LAYERS:
                path = stream_scene_root / scene / f"window_{window:04d}" / f"prefix_{tail:06d}" / f"feature_t749_layer{layer}.sft"
                valid = tensor_ok(path, [21, 30, 52, 1536])
                try:
                    meta = json.loads(Path(f"{path}.manifest.json").read_text())
                    valid = valid and (
                        meta.get("canonical_manifest_sha256") == CANONICAL_SHA256
                        and meta.get("mode") == "manifest_exact_streaming"
                        and meta.get("motion_window_generator_used") is False
                        and meta.get("sample_id") == sample
                        and meta.get("actual_input_raw_ids") == expected_stream
                        and meta.get("size_bytes") == path.stat().st_size
                    )
                    if not valid:
                        add("streaming_sidecar_contract", sample_id=sample, layer=layer, path=str(path))
                except Exception as exc:
                    valid = False
                    add("streaming_sidecar_unreadable", sample_id=sample, layer=layer, path=str(path), detail=repr(exc))
                if valid:
                    streaming_coverage[layer][split] += 1
                all_stream_layers = all_stream_layers and valid
                target_valid = target_scene_valid.get((scene, layer), False) and target_raw(row) in target_index_values[scene]
                if target_valid:
                    target_coverage[layer][split] += 1
            if stream_raw_ok and all_stream_layers:
                streaming_exact += 1
            if target_raw(row) in target_index_values[scene] and all(target_scene_valid.get((scene, layer), False) for layer in LAYERS):
                target_exact += 1

    coverage = {}
    for layer in LAYERS:
        coverage[str(layer)] = {
            "streaming": {split: int(streaming_coverage[layer][split]) for split in ("train", "val")},
            "target": {split: int(target_coverage[layer][split]) for split in ("train", "val")},
        }
        if coverage[str(layer)]["streaming"] != EXPECTED_SPLITS:
            add("streaming_coverage", layer=layer, actual=coverage[str(layer)]["streaming"])
        if coverage[str(layer)]["target"] != EXPECTED_SPLITS:
            add("target_coverage", layer=layer, actual=coverage[str(layer)]["target"])
    if streaming_exact != 2724:
        add("streaming_exact_raw", actual=streaming_exact, expected=2724)
    if target_exact != 2724:
        add("target_exact_raw", actual=target_exact, expected=2724)

    result = {
        "ok": problem_count == 0,
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha,
        "split_counts": dict(split_counts),
        "streaming_root": str(streaming_root),
        "target_root": str(target_root),
        "shard_inventory": str(inventory_path),
        "shard_inventory_sha256": sha256(inventory_path),
        "file_counts": file_counts,
        "expected_file_counts": expected_file_counts,
        "coverage": coverage,
        "streaming_exact_raw": streaming_exact,
        "target_exact_raw": target_exact,
        "streaming_mismatch": 2724 - streaming_exact,
        "target_mismatch": 2724 - target_exact,
        "tensors_checked": tensors_checked,
        "finite_check_requested": args.check_finite,
        "finite_files_checked": finite_checked,
        "problem_count": problem_count,
        "problems": problems,
    }
    out = Path(args.out_json).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    tmp.replace(out)
    print(json.dumps({key: result[key] for key in ("ok", "streaming_exact_raw", "target_exact_raw", "tensors_checked", "finite_files_checked", "problem_count")}))
    raise SystemExit(0 if result["ok"] else 2)


if __name__ == "__main__":
    main()
