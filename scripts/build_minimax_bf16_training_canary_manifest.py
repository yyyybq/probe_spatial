#!/usr/bin/env python3
"""Freeze the first canonical p8 train sample for MiniMax BF16 canaries."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


ROOT = Path("/mnt/umm/users/yinbaiqiao/probe_spatial")
SOURCE = ROOT / "artifacts/main_v2_probe_v2_816_20260917/frozen_queue_r2/manifests/owner_set_p8_5ff5e8302b7c.jsonl"
SOURCE_SHA = "16108949dc0a0551c49836a2e0938ff6fbef3621577ea7b0157919a4ab12b5d0"
OUT = ROOT / "artifacts/main_v2_probe_v2_816_20260917/minimax_bf16_v1/training_canary_p8.jsonl"
RECEIPT = OUT.with_suffix(".receipt.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o444)
    Path(temporary).replace(path)


def main() -> None:
    if sha256(SOURCE) != SOURCE_SHA:
        raise SystemExit("source manifest SHA mismatch")
    selected = None
    for line in SOURCE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (
            row.get("split") == "train"
            and row.get("scene_id") == "00777c41d4"
            and int(row.get("streaming_window_id", -1)) == 0
            and len(row.get("streaming_indices", [])) == 8
        ):
            selected = row
            break
    if selected is None:
        raise SystemExit("canonical canary sample not found")
    payload = (json.dumps(selected, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if OUT.exists() or RECEIPT.exists():
        raise SystemExit("refusing to overwrite existing canary manifest")
    atomic_bytes(OUT, payload)
    receipt = {
        "schema": "MINIMAX_H3_BF16_TRAINING_CANARY_MANIFEST_V1",
        "source_manifest": str(SOURCE),
        "source_manifest_sha256": SOURCE_SHA,
        "output_manifest": str(OUT),
        "output_manifest_sha256": sha256(OUT),
        "rows": 1,
        "sample_id": selected["sample_id"],
        "scene_id": selected["scene_id"],
        "streaming_window_id": selected["streaming_window_id"],
        "prefix": len(selected["streaming_indices"]),
        "ordered_raw_frame_ids": selected["streaming_indices"],
    }
    atomic_bytes(RECEIPT, (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
