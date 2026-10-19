#!/usr/bin/env python3
"""Expand targeted feature repairs into scene-granular atomic queue units."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODELS = {"cogvideox", "dino", "lingbot", "videomae", "vjepa2"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--canonical-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=sorted(MODELS))
    parser.add_argument("--expected-tasks", type=int, default=65_376)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite immutable queue: {output}")
    scope_path = args.scope.resolve()
    scope = json.loads(scope_path.read_text())
    source_path = args.canonical_source.resolve()
    scenes = [json.loads(line)["scene_id"] for line in source_path.open()]
    if len(scenes) != 2721:
        raise RuntimeError("canonical source row count is not 2721")
    # A scene can own several canonical windows.  The extractor's --start/end
    # address its sorted physical scene list, not manifest-row ordinals.
    scene_order = sorted(set(str(scene) for scene in scenes))
    if len(scene_order) != 681:
        raise RuntimeError("canonical source scene count is not 681")
    scene_index = {scene: index for index, scene in enumerate(scene_order)}

    atomic: list[dict[str, Any]] = []
    seen: set[str] = set()
    for binding in scope["worklists"]:
        if binding["model"] not in set(args.models):
            continue
        path = Path(binding["worklist"]).resolve()
        if sha256(path) != binding["file_sha256"]:
            raise RuntimeError(f"parent worklist SHA mismatch: {path}")
        worklist = json.loads(path.read_text())
        grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        if worklist["mode"] == "streaming_prefix":
            # Atomic identity: model/layer/prefix/scene.  Multiple windows for
            # that scene/prefix remain inside the same task.
            for entry in worklist["entries"]:
                for layer in entry["layers"]:
                    grouped[(int(layer), int(entry["prefix"]), str(entry["scene"]))].append({
                        "scene": str(entry["scene"]),
                        "window_id": int(entry["window_id"]),
                        "prefix": int(entry["prefix"]),
                        "layers": [int(layer)],
                        "reason": str(entry["reason"]),
                    })
        else:
            # Atomic identity: model/layer/scene.  All required target raw IDs
            # for that scene stay in the same task.
            for entry in worklist["entries"]:
                for layer in entry["layers"]:
                    grouped[(int(layer), str(entry["scene"]))].append({
                        "scene": str(entry["scene"]),
                        "frame_index": int(entry["frame_index"]),
                        "layers": [int(layer)],
                        "reason": str(entry["reason"]),
                    })
        for key, entries in sorted(grouped.items()):
            layer = int(key[0])
            prefix = int(key[1]) if worklist["mode"] == "streaming_prefix" else None
            scene = str(key[2]) if prefix is not None else str(key[1])
            identity = {
                "mode": worklist["mode"],
                "model": worklist["vfm"],
                "timestep": int(worklist["timestep"]),
                "layer": layer,
                "prefix": prefix,
                "scene_id": scene,
            }
            unique_key = canonical_sha(identity)
            if unique_key in seen:
                raise RuntimeError(f"duplicate atomic task: {identity}")
            seen.add(unique_key)
            entries.sort(key=lambda row: (
                row["scene"], row.get("window_id", -1), row.get("frame_index", -1)
            ))
            atomic.append({
                "schema": "PROBE528_ATOMIC_FEATURE_TASK_V1",
                "unique_key": unique_key,
                **identity,
                "scene_index": scene_index[scene],
                "entries": entries,
                "entry_count": len(entries),
                "layer_items": len(entries),
                "parent_worklist": {
                    "path": str(path),
                    "file_sha256": binding["file_sha256"],
                    "content_sha256": worklist["content_sha256"],
                },
            })
    atomic.sort(key=lambda row: (
        0 if row["model"] in {"cogvideox", "lingbot"} else 1,
        row["model"], row["mode"], row["layer"],
        -1 if row["prefix"] is None else row["prefix"], row["scene_index"],
    ))
    expected = args.expected_tasks
    if len(atomic) != expected or len(seen) != expected:
        raise RuntimeError(f"unexpected atomic task count: {len(atomic)}")
    output.mkdir(parents=True)
    queue_path = output / "queue.jsonl"
    write_exclusive(
        queue_path,
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in atomic),
    )
    counts = Counter((row["model"], row["mode"]) for row in atomic)
    manifest = {
        "schema": "PROBE528_ATOMIC_FEATURE_QUEUE_V1",
        "status": "FROZEN_NOT_SUBMITTED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "atomic_identity": {
            "streaming_prefix": ["model", "layer", "prefix", "scene_id"],
            "target_isolated": ["model", "layer", "scene_id"],
        },
        "scope": {"path": str(scope_path), "sha256": sha256(scope_path)},
        "canonical_source": {"path": str(source_path), "sha256": sha256(source_path)},
        "queue": {"path": str(queue_path), "sha256": sha256(queue_path), "tasks": len(atomic)},
        "counts": [
            {"model": model, "mode": mode, "tasks": count}
            for (model, mode), count in sorted(counts.items())
        ],
        "scheduler_policy": {
            "resource_pool": "zoetrope",
            "worker_spec": "N4lS.Iq.I80.1",
            "claim": "O_EXCL per unique_key",
            "writer_ownership": "one active claim per atomic task",
        },
    }
    manifest["content_sha256"] = canonical_sha(manifest)
    write_exclusive(output / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": manifest["status"], "tasks": len(atomic), "counts": manifest["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
