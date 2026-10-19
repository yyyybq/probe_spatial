#!/usr/bin/env python3
"""Freeze one representative pilot queue and five balanced production shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_QUEUE_SHA = "ad194b6ea5e53a2dd53ecd5e567b74c324b70d52219ab1a988b8bab69bc91726"
PILOT = {
    ("minimax_h3", "streaming_depth", 20, 8),
    ("videomae", "streaming_mask", 23, 24),
    ("cogvideox", "view_consistency", 0, 16),
    ("cogvideox", "ego_belief", 15, 8),
    ("lingbot", "ego_belief_v2", 48, 12),
    ("vjepa2", "action_dynamics", 23, 24),
    ("qwen2_5_vl", "path_integration", -1, 12),
    ("videomae", "counterfactual", 17, 16),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    return {"path": str(path.resolve()), "sha256": sha256(path), "rows": len(rows)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=5)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError(f"output already exists: {args.output_dir}")
    if args.shards != 5 or sha256(args.queue) != EXPECTED_QUEUE_SHA:
        raise ValueError("submission queue contract mismatch")
    rows = [json.loads(line) for line in args.queue.open(encoding="utf-8") if line.strip()]
    if len(rows) != 760 or len({row["unique_key"] for row in rows}) != 760:
        raise ValueError("input queue must contain 760 unique units")
    pilot = [row for row in rows if (row["model"], row["probe"], int(row["layer"]), int(row["prefix"])) in PILOT]
    if len(pilot) != 8:
        raise ValueError(f"pilot selector resolved to {len(pilot)} units")
    pilot_keys = {row["unique_key"] for row in pilot}
    remainder = [row for row in rows if row["unique_key"] not in pilot_keys]
    # The frozen queue is already longest-family first. Round-robin assignment
    # balances those heavy units while preserving their relative order inside
    # each dynamic eight-worker node queue.
    shards = [[] for _ in range(args.shards)]
    for index, row in enumerate(remainder):
        shards[index % args.shards].append(row)
    args.output_dir.mkdir(parents=True)
    bindings = {"pilot": write_jsonl(args.output_dir / "pilot_8.jsonl", pilot)}
    for index, shard in enumerate(shards):
        bindings[f"remainder_{index:02d}"] = write_jsonl(
            args.output_dir / f"remainder_{index:02d}.jsonl", shard
        )
    all_keys = [row["unique_key"] for row in pilot]
    for shard in shards:
        all_keys.extend(row["unique_key"] for row in shard)
    if len(all_keys) != 760 or len(set(all_keys)) != 760 or set(all_keys) != {row["unique_key"] for row in rows}:
        raise ValueError("submission partition is not an exact disjoint cover")
    manifest = {
        "schema": "MAIN_V2_PROBE816_SUBMISSION_PARTITION_V1",
        "status": "FROZEN_NOT_SUBMITTED",
        "input_queue": {"path": str(args.queue.resolve()), "sha256": EXPECTED_QUEUE_SHA, "rows": 760},
        "queues": bindings,
        "partition": {"pilot": 8, "remainder": 752, "remainder_shards": 5},
        "policy": "pilot gate, then five full-node dynamic queues; longest task families first",
    }
    path = args.output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

