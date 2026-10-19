#!/usr/bin/env python3
"""Verify Wan prefix-8 action_dynamics clean-rerun feature coverage.

The check is intentionally manifest-driven.  It only counts samples present in
the frozen canonical JSONL, and it verifies the exact file paths the dataset
will use for streaming-prefix inputs and isolated future targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_MANIFEST_SHA = "772194f077c45cc85a942f17f46f50103125ae8f5b1c983c08881c1759607e0c"
EXPECTED_COUNTS = {"train": 2448, "val": 276}
LAYERS = [0, 5, 10, 15, 25, 29]
VFM = "wan"
SOURCE = "scannetpp"
TIMESTEP = 749
CHANNELS = 1536


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def feature_name(layer: int) -> str:
    return f"feature_t{TIMESTEP}_layer{layer}.sft"


def scene_id(row: dict[str, Any]) -> str:
    return str(row.get("scene_id") or Path(str(row.get("scene_dir") or row["data_prefix"])).name)


def streaming_path(root: Path, row: dict[str, Any], layer: int) -> Path:
    return (
        root
        / VFM
        / str(row.get("source", SOURCE))
        / scene_id(row)
        / f"window_{int(row['streaming_window_id']):04d}"
        / f"prefix_{int(row['streaming_tail']):06d}"
        / feature_name(layer)
    )


def target_scene_dir(root: Path, row: dict[str, Any]) -> Path:
    return root / VFM / str(row.get("source", SOURCE)) / scene_id(row)


def target_path(root: Path, row: dict[str, Any], layer: int) -> Path:
    base = target_scene_dir(root, row)
    prefixed = base / f"prefix_{int(row['streaming_tail']):06d}"
    if prefixed.is_dir():
        return prefixed / feature_name(layer)
    return base / feature_name(layer)


def target_index_path(root: Path, row: dict[str, Any]) -> Path:
    base = target_scene_dir(root, row)
    prefixed = base / f"prefix_{int(row['streaming_tail']):06d}"
    if prefixed.is_dir():
        return prefixed / "target_indices.npy"
    return base / "target_indices.npy"


def target_raw_index(row: dict[str, Any], horizon: int) -> int:
    tail = int(row["streaming_tail"])
    window_indices = [int(x) for x in row["streaming_window_indices"]]
    return int(window_indices[tail + horizon])


def tensor_header(path: Path) -> dict[str, Any]:
    try:
        from safetensors import safe_open
    except Exception as exc:  # pragma: no cover - environment guard
        raise RuntimeError("safetensors is required in the probe-spatial environment") from exc
    with safe_open(str(path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        if keys != ["feat"]:
            return {"ok": False, "keys": keys, "error": "unexpected_keys"}
        sl = f.get_slice("feat")
        shape = list(sl.get_shape())
        dtype = str(sl.get_dtype())
        return {"ok": True, "keys": keys, "shape": shape, "dtype": dtype}


def valid_streaming_header(meta: dict[str, Any]) -> bool:
    shape = meta.get("shape") or []
    return (
        meta.get("ok")
        and len(shape) == 4
        and int(shape[-1]) == CHANNELS
        and str(meta.get("dtype")) in {"F16", "torch.float16", "float16"}
    )


def valid_target_header(meta: dict[str, Any]) -> bool:
    shape = meta.get("shape") or []
    return (
        meta.get("ok")
        and len(shape) == 4
        and int(shape[-1]) == CHANNELS
        and int(shape[0]) > 0
        and str(meta.get("dtype")) in {"F16", "torch.float16", "float16"}
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--streaming-root", required=True)
    ap.add_argument("--target-root", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--horizon", type=int, default=1)
    args = ap.parse_args()

    manifest = Path(args.manifest).resolve()
    streaming_root = Path(args.streaming_root).resolve()
    target_root = Path(args.target_root).resolve()
    out_path = Path(args.out_json).resolve()

    msha = sha256(manifest)
    rows = read_manifest(manifest)
    split_counts = Counter(str(r.get("split", "")) for r in rows)
    problems: list[dict[str, Any]] = []
    if msha != EXPECTED_MANIFEST_SHA:
        problems.append({"kind": "manifest_sha_mismatch", "actual": msha, "expected": EXPECTED_MANIFEST_SHA})
    if dict(split_counts) != EXPECTED_COUNTS:
        problems.append({"kind": "split_count_mismatch", "actual": dict(split_counts), "expected": EXPECTED_COUNTS})

    stream_seen = {layer: Counter() for layer in LAYERS}
    target_seen = {layer: Counter() for layer in LAYERS}
    header_cache: dict[Path, dict[str, Any]] = {}
    index_cache: dict[Path, set[int]] = {}
    sample_status: dict[str, dict[str, Any]] = {}

    for row in rows:
        split = str(row["split"])
        sid = str(row.get("sample_id") or f"{scene_id(row)}_w{int(row['streaming_window_id']):03d}_p8")
        target_raw = target_raw_index(row, args.horizon)
        sample_ok = True
        sample_errors = []

        idx_path = target_index_path(target_root, row)
        if idx_path not in index_cache:
            if not idx_path.exists() or idx_path.stat().st_size <= 0:
                index_cache[idx_path] = set()
                problems.append({"kind": "missing_target_index", "path": str(idx_path)})
            else:
                try:
                    index_cache[idx_path] = {int(x) for x in np.load(idx_path).astype(np.int64).tolist()}
                except Exception as exc:
                    index_cache[idx_path] = set()
                    problems.append({"kind": "bad_target_index", "path": str(idx_path), "error": str(exc)})
        if target_raw not in index_cache[idx_path]:
            sample_ok = False
            sample_errors.append(f"target_raw_missing:{target_raw}")

        for layer in LAYERS:
            sp = streaming_path(streaming_root, row, layer)
            if sp not in header_cache:
                if not sp.exists() or sp.stat().st_size <= 0:
                    header_cache[sp] = {"ok": False, "error": "missing_or_empty"}
                else:
                    try:
                        header_cache[sp] = tensor_header(sp)
                    except Exception as exc:
                        header_cache[sp] = {"ok": False, "error": str(exc)}
            if valid_streaming_header(header_cache[sp]):
                stream_seen[layer][split] += 1
            else:
                sample_ok = False
                sample_errors.append(f"streaming_layer{layer}:{header_cache[sp].get('error', 'bad_header')}")

            tp = target_path(target_root, row, layer)
            if tp not in header_cache:
                if not tp.exists() or tp.stat().st_size <= 0:
                    header_cache[tp] = {"ok": False, "error": "missing_or_empty"}
                else:
                    try:
                        header_cache[tp] = tensor_header(tp)
                    except Exception as exc:
                        header_cache[tp] = {"ok": False, "error": str(exc)}
            if valid_target_header(header_cache[tp]):
                target_seen[layer][split] += 1
            else:
                sample_ok = False
                sample_errors.append(f"target_layer{layer}:{header_cache[tp].get('error', 'bad_header')}")

        sample_status[sid] = {
            "ok": sample_ok,
            "split": split,
            "target_raw": target_raw,
            "errors": sample_errors,
        }

    expected = EXPECTED_COUNTS
    coverage = {}
    for layer in LAYERS:
        coverage[str(layer)] = {
            "streaming": {split: int(stream_seen[layer][split]) for split in ("train", "val")},
            "target": {split: int(target_seen[layer][split]) for split in ("train", "val")},
        }
        for kind, seen in (("streaming", stream_seen[layer]), ("target", target_seen[layer])):
            for split, exp in expected.items():
                if int(seen[split]) != exp:
                    problems.append({
                        "kind": "coverage_mismatch",
                        "feature_kind": kind,
                        "layer": layer,
                        "split": split,
                        "actual": int(seen[split]),
                        "expected": exp,
                    })

    bad_samples = {sid: st for sid, st in sample_status.items() if not st["ok"]}
    result = {
        "manifest": str(manifest),
        "manifest_sha256": msha,
        "streaming_root": str(streaming_root),
        "streaming_root_realpath": str(streaming_root.resolve()),
        "target_root": str(target_root),
        "target_root_realpath": str(target_root.resolve()),
        "layers": LAYERS,
        "expected_counts": expected,
        "split_counts": dict(split_counts),
        "coverage": coverage,
        "files_checked": len(header_cache),
        "target_index_files_checked": len(index_cache),
        "bad_sample_count": len(bad_samples),
        "bad_samples_preview": dict(list(bad_samples.items())[:20]),
        "problems": problems[:200],
        "ok": not problems and not bad_samples,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f".{out_path.name}.tmp")
    tmp.write_text(json.dumps(result, indent=2) + "\n")
    tmp.replace(out_path)
    print(json.dumps({"ok": result["ok"], "problems": len(problems), "bad_samples": len(bad_samples)}, indent=2))
    if not result["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
