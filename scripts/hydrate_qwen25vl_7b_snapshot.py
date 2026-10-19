#!/usr/bin/env python3
"""Restore the exact Qwen2.5-VL-7B snapshot used by canonical-v2 extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
CONFIG_SHA256 = {
    "config.json": "77d9ec7321cc572e3579e2c84799c9cadaded63c49ce93b101733349fc330c43",
    "generation_config.json": "0a3aea82869fe29f20dc95ccf3e2bcff380eca1f5ad6447a4a4b37110b08e43e",
    "preprocessor_config.json": "f2058c716eef96ccaed1cc1e2d0c08306b62586d535b28d9d08e691b2fab7ca0",
    "tokenizer_config.json": "4abd3520120e266da84c0864fee064d1fb10806f02225911a47253dd38dc5f56",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=Path("/mnt/umm/users/yinbaiqiao/.cache/huggingface"),
    )
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    hub = args.hf_home.resolve() / "hub"
    snapshot = Path(snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        cache_dir=hub,
        local_files_only=False,
        max_workers=8,
    )).resolve()
    if snapshot.name != REVISION:
        raise RuntimeError(f"resolved unexpected Qwen revision: {snapshot}")
    observed = {name: sha256(snapshot / name) for name in CONFIG_SHA256}
    if observed != CONFIG_SHA256:
        raise RuntimeError(f"Qwen config identity mismatch: {observed}")

    model_root = hub / "models--Qwen--Qwen2.5-VL-7B-Instruct"
    atomic_text(model_root / "refs/main", REVISION + "\n")
    files = [path for path in snapshot.rglob("*") if path.is_file()]
    receipt = {
        "schema": "QWEN25VL_7B_EXACT_SNAPSHOT_HYDRATION_V1",
        "status": "PASS_EXACT_REVISION_AND_CONFIG_SHA",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "revision": REVISION,
        "hf_home": str(args.hf_home.resolve()),
        "snapshot": str(snapshot),
        "config_sha256": observed,
        "files": len(files),
        "logical_bytes": sum(path.stat().st_size for path in files),
    }
    receipt["content_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    atomic_text(args.receipt, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
