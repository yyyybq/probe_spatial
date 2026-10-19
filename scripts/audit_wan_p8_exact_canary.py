#!/usr/bin/env python3
"""Fail-closed audit for the Wan p8 manifest-exact 40-sample canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file


CANONICAL_SHA256 = "772194f077c45cc85a942f17f46f50103125ae8f5b1c983c08881c1759607e0c"
LAYERS = [0, 5, 10, 15, 25, 29]


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


def selected_rows(manifest: Path, selection: Path) -> list[dict[str, Any]]:
    wanted = {
        line.strip()
        for line in selection.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    rows = [
        json.loads(line)
        for line in manifest.read_text().splitlines()
        if line.strip()
    ]
    result = [row for row in rows if row["sample_id"] in wanted]
    if len(wanted) != 40 or len(result) != 40:
        raise ValueError(f"Expected 40 unique selected rows, got wanted={len(wanted)} rows={len(result)}")
    return result


def tensor_audit(path: Path, expected_shape: list[int]) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "ok": False}
    if not path.is_file() or path.stat().st_size <= 0:
        result["error"] = "missing_or_empty"
        return result
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            shape = list(handle.get_slice("feat").get_shape()) if keys == ["feat"] else []
            dtype = str(handle.get_slice("feat").get_dtype()) if keys == ["feat"] else ""
        tensor = load_file(str(path))["feat"]
        finite = bool(torch.isfinite(tensor).all().item())
        result.update(
            {
                "keys": keys,
                "shape": shape,
                "dtype": dtype,
                "finite": finite,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
        result["ok"] = keys == ["feat"] and shape == expected_shape and dtype == "F16" and finite
        if not result["ok"]:
            result["error"] = "tensor_contract"
    except Exception as exc:
        result["error"] = "unreadable"
        result["detail"] = repr(exc)
    return result


def read_sidecar(path: Path) -> dict[str, Any]:
    return json.loads(Path(f"{path}.manifest.json").read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canary-tag", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--selection-file", required=True)
    parser.add_argument("--streaming-root", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--dataset-json", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    manifest = Path(args.manifest).resolve()
    selection = Path(args.selection_file).resolve()
    streaming_root = Path(args.streaming_root).resolve()
    target_root = Path(args.target_root).resolve()
    dataset_path = Path(args.dataset_json).resolve()
    problems: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []

    manifest_sha = sha256(manifest)
    if manifest_sha != CANONICAL_SHA256:
        problems.append({"error": "manifest_sha256", "actual": manifest_sha})
    rows = selected_rows(manifest, selection)
    split_counts = Counter(str(row["split"]) for row in rows)
    rows_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_scene[scene_id(row)].append(row)

    streaming_exact = 0
    target_exact = 0
    streaming_records: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}
    for scene, scene_rows in rows_by_scene.items():
        index_path = streaming_root / "wan" / "scannetpp" / scene / "prefix_index.npy"
        try:
            records = np.load(index_path, allow_pickle=True).tolist()
            streaming_records[scene] = {
                (int(record["window_id"]), int(record["tail"])): record
                for record in records
            }
        except Exception as exc:
            problems.append({"scene_id": scene, "error": "streaming_index", "path": str(index_path), "detail": repr(exc)})
            streaming_records[scene] = {}

        expected_targets = sorted({target_raw(row) for row in scene_rows})
        target_index_path = target_root / "wan" / "scannetpp" / scene / "target_indices.npy"
        try:
            actual_targets = np.load(target_index_path).astype(np.int64).tolist()
        except Exception as exc:
            problems.append({"scene_id": scene, "error": "target_index", "path": str(target_index_path), "detail": repr(exc)})
            actual_targets = []
        if actual_targets != expected_targets:
            problems.append(
                {
                    "scene_id": scene,
                    "error": "target_index_set_mismatch",
                    "expected": expected_targets,
                    "actual": actual_targets,
                    "path": str(target_index_path),
                }
            )

        for layer in LAYERS:
            path = target_root / "wan" / "scannetpp" / scene / f"feature_t749_layer{layer}.sft"
            audit = tensor_audit(path, [len(expected_targets), 30, 52, 1536])
            files.append({"kind": "target", "scene_id": scene, "layer": layer, **audit})
            if not audit["ok"]:
                problems.append({"scene_id": scene, "layer": layer, "error": "target_tensor", "path": str(path), "audit": audit})
            else:
                try:
                    meta = read_sidecar(path)
                    if (
                        meta.get("canonical_manifest_sha256") != CANONICAL_SHA256
                        or meta.get("target_raw_ids") != expected_targets
                        or meta.get("mode") != "manifest_exact_target"
                        or bool(meta.get("motion_window_generator_used", True))
                    ):
                        raise ValueError("target sidecar contract mismatch")
                except Exception as exc:
                    problems.append({"scene_id": scene, "layer": layer, "error": "target_sidecar", "path": str(path), "detail": repr(exc)})

    for row in rows:
        scene = scene_id(row)
        key = (int(row["streaming_window_id"]), int(row["streaming_tail"]))
        record = streaming_records[scene].get(key)
        expected_stream = [int(value) for value in row["streaming_indices"]]
        expected_target = target_raw(row)
        if record is None:
            problems.append({"sample_id": row["sample_id"], "error": "missing_streaming_record", "key": list(key)})
        elif [int(value) for value in record.get("actual_input_raw_ids", [])] == expected_stream:
            streaming_exact += 1
        else:
            problems.append(
                {
                    "sample_id": row["sample_id"],
                    "error": "streaming_raw_mismatch",
                    "expected": expected_stream,
                    "actual": record.get("actual_input_raw_ids"),
                }
            )

        target_index_path = target_root / "wan" / "scannetpp" / scene / "target_indices.npy"
        try:
            values = np.load(target_index_path).astype(np.int64).tolist()
        except Exception:
            values = []
        if values.count(expected_target) == 1:
            target_exact += 1
        else:
            problems.append(
                {
                    "sample_id": row["sample_id"],
                    "error": "target_raw_mismatch",
                    "expected": expected_target,
                    "actual": values,
                    "path": str(target_index_path),
                }
            )

        for layer in LAYERS:
            path = (
                streaming_root
                / "wan"
                / "scannetpp"
                / scene
                / f"window_{key[0]:04d}"
                / f"prefix_{key[1]:06d}"
                / f"feature_t749_layer{layer}.sft"
            )
            audit = tensor_audit(path, [21, 30, 52, 1536])
            files.append({"kind": "streaming", "sample_id": row["sample_id"], "layer": layer, **audit})
            if not audit["ok"]:
                problems.append({"sample_id": row["sample_id"], "layer": layer, "error": "streaming_tensor", "path": str(path), "audit": audit})
                continue
            try:
                meta = read_sidecar(path)
                if (
                    meta.get("canonical_manifest_sha256") != CANONICAL_SHA256
                    or meta.get("sample_id") != row["sample_id"]
                    or meta.get("actual_input_raw_ids") != expected_stream
                    or meta.get("mode") != "manifest_exact_streaming"
                    or bool(meta.get("motion_window_generator_used", True))
                ):
                    raise ValueError("streaming sidecar contract mismatch")
            except Exception as exc:
                problems.append({"sample_id": row["sample_id"], "layer": layer, "error": "streaming_sidecar", "path": str(path), "detail": repr(exc)})

    try:
        dataset = json.loads(dataset_path.read_text())
    except Exception as exc:
        dataset = {}
        problems.append({"error": "dataset_json", "path": str(dataset_path), "detail": repr(exc)})
    dataset_ok = bool(dataset.get("ok"))
    for layer in LAYERS:
        layer_result = dataset.get("layers", {}).get(str(layer), {})
        if not layer_result.get("ok") or int(layer_result.get("checked", -1)) != 40:
            problems.append({"error": "dataset_layer", "layer": layer, "result": layer_result})

    result = {
        "canary_tag": args.canary_tag,
        "ok": not problems,
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha,
        "selection_file": str(selection),
        "selection_sha256": sha256(selection),
        "sample_count": len(rows),
        "split_counts": dict(split_counts),
        "scene_count": len(rows_by_scene),
        "layers": LAYERS,
        "streaming_root": str(streaming_root),
        "target_root": str(target_root),
        "streaming_exact_raw": streaming_exact,
        "target_exact_raw": target_exact,
        "streaming_mismatch": len(rows) - streaming_exact,
        "target_mismatch": len(rows) - target_exact,
        "dataset_ok": dataset_ok,
        "dataset_json": str(dataset_path),
        "file_audits": files,
        "problem_count": len(problems),
        "problems": problems,
    }
    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_json.with_name(f".{out_json.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, out_json)

    status = "PASS" if result["ok"] else "FAIL"
    report = f"""# Wan p8 Manifest-Exact Canary Report

```text
CANARY_TAG: {args.canary_tag}
STATUS: {status}
CANONICAL_MANIFEST_SHA256: {manifest_sha}
SAMPLES: {len(rows)}/40
SCENES: {len(rows_by_scene)}
LAYERS: {','.join(str(layer) for layer in LAYERS)}
STREAMING_EXACT_RAW: {streaming_exact}/40
TARGET_EXACT_RAW: {target_exact}/40
STREAMING_MISMATCH: {len(rows) - streaming_exact}
TARGET_MISMATCH: {len(rows) - target_exact}
DATASET_DATALOADER_PASS: {str(dataset_ok).lower()}
PROBLEM_COUNT: {len(problems)}
STREAMING_ROOT: {streaming_root}
TARGET_ROOT: {target_root}
```

The JSON report at `{out_json}` contains every file path, tensor contract,
SHA256, finite-value check, and any sample-level failure evidence.
"""
    out_md = Path(args.out_md).resolve()
    tmp_md = out_md.with_name(f".{out_md.name}.tmp-{os.getpid()}")
    tmp_md.write_text(report)
    os.replace(tmp_md, out_md)
    print(json.dumps({"ok": result["ok"], "streaming_exact": streaming_exact, "target_exact": target_exact, "problems": len(problems)}))
    raise SystemExit(0 if result["ok"] else 2)


if __name__ == "__main__":
    main()
