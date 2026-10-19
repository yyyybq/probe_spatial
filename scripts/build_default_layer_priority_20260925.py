#!/usr/bin/env python3
"""Build an additive, immutable default-layer queue for CogVideoX and LingBot.

This keeps the frozen valid7 worklists untouched.  It clones their exact
scene/window/frame allow-lists, changes only the selected output layer, and
uses the existing atomic-owner builder to make disjoint single-GPU workers.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/main_v2_probe_v2_816_20260917"
OUT = BASE / "default_layer_priority_20260925_r1"
GAPS = BASE / "targeted_reextract_worklists_20260922_r2/gap_worklists"


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def scene_indices() -> dict[str, int]:
    result: dict[str, int] = {}
    source = ROOT / "manifests/c_retrain_20260722/inscene_c_h124_prefix24.jsonl"
    for line in source.read_text().splitlines():
        row = json.loads(line)
        scene = str(row["scene_id"])
        if scene not in result:
            result[scene] = len(result)
    if len(result) != 681:
        raise RuntimeError(f"expected 681 scenes, got {len(result)}")
    return result


def rows_for(*, model: str, mode: str, layer: int, parent_path: Path, index: dict[str, int]) -> list[dict]:
    parent = json.loads(parent_path.read_text())
    reference = {
        "path": str(parent_path.resolve()),
        "file_sha256": file_digest(parent_path),
        "content_sha256": parent["content_sha256"],
    }
    rows = []
    for entry in parent["entries"]:
        scene = str(entry["scene"])
        copied = {key: value for key, value in entry.items() if key != "layers"}
        copied["layers"] = [layer]
        row = {
            "schema": "PROBE528_ATOMIC_FEATURE_TASK_V1",
            "model": model,
            "mode": mode,
            "layer": layer,
            "timestep": int(parent["timestep"]),
            "scene_id": scene,
            "scene_index": index[scene],
            "prefix": copied.get("prefix") if mode == "streaming_prefix" else None,
            "entry_count": 1,
            "layer_items": 1,
            "entries": [copied],
            "parent_worklist": reference,
        }
        row["unique_key"] = digest({key: row[key] for key in (
            "model", "mode", "layer", "timestep", "scene_id", "entries", "parent_worklist"
        )})
        rows.append(row)
    return rows


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite immutable output: {OUT}")
    OUT.mkdir(parents=True)
    index = scene_indices()
    definitions = [
        ("cogvideox", "streaming_prefix", 20, GAPS / "cogvideox-t749-streaming_prefix-c1f59d13ce52.json"),
        ("cogvideox", "target_isolated", 20, GAPS / "cogvideox-t749-target_isolated-eaa19f1aba6a.json"),
        ("lingbot", "target_isolated", 30, GAPS / "lingbot-t250-target_isolated-dddc629938e8.json"),
    ]
    all_rows = []
    for model, mode, layer, path in definitions:
        all_rows.extend(rows_for(model=model, mode=mode, layer=layer, parent_path=path, index=index))
    queue = OUT / "priority_default_layer_atomic_queue.jsonl"
    with queue.open("x", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    queue_sha = file_digest(queue)
    templates = [
        BASE / "atomic_feature_workers_four_models_20260923_r3/shards/p528-cogvideox-stream-00.json",
        BASE / "atomic_feature_workers_four_models_20260923_r3/shards/p528-cogvideox-target-00.json",
        BASE / "atomic_feature_workers_lingbot_20260923_r2/shards/p528-lingbot-target-00.json",
    ]
    command = [
        "/usr/bin/python3", str(ROOT / "scripts/build_probe528_atomic_worker_plan.py"),
        "--queue", str(queue), "--queue-sha256", queue_sha,
        "--output", str(OUT / "workers"),
    ]
    for template in templates:
        command.extend(["--template", str(template)])
    subprocess.run(command, check=True, cwd=ROOT)
    (OUT / "README.md").write_text(
        "# Default-layer priority recovery (2026-09-25)\n\n"
        "This is an additive isolated plan requested by the user. It does not alter the frozen valid7 worklist. "
        "CogVideoX uses L20 for the exact former Cog scene/window/frame allow-lists. LingBot L20 and L30 had equal "
        "confirmed coverage at selection time, so L30 was chosen as the deterministic deeper-layer tie break.\n"
    )
    print(json.dumps({"output": str(OUT), "queue_sha256": queue_sha, "rows": len(all_rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
