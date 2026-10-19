#!/usr/bin/env python3
"""Freeze a small real canonical manifest for A1 movement runtime validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_PARENT_SHA256 = (
    "c8d16c48580f270b6a51cb90bd1e6aa126d2fe231f64104832ba043eee643901"
)
VERIFIED_SAMPLE_IDS = (
    "00777c41d4_w001_p8",
    "00777c41d4_w002_p8",
    "00777c41d4_w003_p8",
    "00dd871005_w001_p8",
    "00a231a370_w001_p8",
    "00a231a370_w002_p8",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def select_receipt_verified(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(row.get("sample_id")): row for row in rows}
    missing = [sample_id for sample_id in VERIFIED_SAMPLE_IDS if sample_id not in by_id]
    if missing:
        raise RuntimeError(f"receipt-verified sample ids absent from parent: {missing}")
    return [by_id[sample_id] for sample_id in VERIFIED_SAMPLE_IDS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=4)
    parser.add_argument("--val-samples", type=int, default=2)
    args = parser.parse_args()

    parent = args.parent_manifest.resolve()
    parent_sha = sha256(parent)
    if parent_sha != EXPECTED_PARENT_SHA256:
        raise RuntimeError(
            f"canonical parent SHA mismatch: {parent_sha} != {EXPECTED_PARENT_SHA256}"
        )
    rows = [json.loads(line) for line in parent.read_text().splitlines() if line]
    if (args.train_samples, args.val_samples) != (4, 2):
        raise ValueError("v1 receipt-verified canary is frozen at 4 train + 2 val")
    selected = select_receipt_verified(rows)
    splits = [row.get("split") for row in selected]
    if splits != ["train"] * 4 + ["val"] * 2:
        raise RuntimeError(f"unexpected receipt-verified split ordering: {splits}")
    if any(len(row.get("streaming_indices", [])) != 8 for row in selected):
        raise RuntimeError("selected row does not have the canonical p8 observation length")
    if any(not row.get("owner_unit_id") for row in selected):
        raise RuntimeError("selected row is not bound to an owner-backed feature unit")

    output_dir = args.output_dir.resolve()
    manifest = output_dir / "manifest.jsonl"
    content = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in selected
    )
    atomic_write(manifest, content)
    manifest_sha = sha256(manifest)
    receipt = {
        "schema": "A1_MOVEMENT_CANONICAL_CANARY_MANIFEST_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "inscene_streaming_canonical_v2_obs28_h124",
        "task": "streaming_movement",
        "vfm": "dino",
        "feature_timestep": 749,
        "feature_layer": 0,
        "prefix": 8,
        "parent_manifest": str(parent),
        "parent_manifest_sha256": parent_sha,
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha,
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "sample_ids": [row["sample_id"] for row in selected],
        "owner_unit_ids": [row["owner_unit_id"] for row in selected],
        "selection_basis": "probe816_receipt_union_dino_aoss_materialize_fetch",
        "prior_verified_manifest_sha256": "072cd68061f90187069bd7e192b5f2dedd199cd9b22510c78eb4ef0d0bfa46f5",
    }
    atomic_write(output_dir / "receipt.json", json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
