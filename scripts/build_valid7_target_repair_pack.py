#!/usr/bin/env python3
"""Freeze the exact extractor sources used by the valid7 target repairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


FILES = (
    "features/run_inscene15k.py",
    "features/noise_seed.py",
    "features/offline_hf.py",
    "features/canonical_gap_worklist.py",
    "vidfm3d/__init__.py",
    "vidfm3d/utils/__init__.py",
    "vidfm3d/utils/feature_layers.py",
    "vidfm3d/utils/temporal_sampling.py",
    "scripts/run_tstep_sync_loop.sh",
    "scripts/windowfix_v2/run_canonical_gap_shard.py",
    "scripts/windowfix_v2/canonical_gap_aoss_adapter.py",
    "protocols/inscene_streaming_canonical_v2.json",
    "manifests/c_retrain_20260722/inscene_c_h124_prefix24.jsonl",
)
TREES = ("features/cogvideox", "features/dino", "features/f3r")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"output exists: {output}")
    source = output / "source"
    source.mkdir(parents=True)
    for relative in FILES:
        src = root / relative
        dst = source / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    for relative in TREES:
        shutil.copytree(
            root / relative,
            source / relative,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    files = []
    for path in sorted(source.rglob("*")):
        if path.is_file():
            files.append({
                "relative_path": path.relative_to(source).as_posix(),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            })
            os.chmod(path, 0o555 if os.access(path, os.X_OK) else 0o444)
    content_sha = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = {
        "schema": "VALID7_TARGET_REPAIR_IMMUTABLE_SOURCE_PACK_V1",
        "status": "FROZEN",
        "source_root": str(source),
        "source_project_root": str(root),
        "content_sha256": content_sha,
        "files": files,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.chmod(manifest_path, 0o444)
    for directory in sorted((p for p in source.rglob("*") if p.is_dir()), reverse=True):
        os.chmod(directory, 0o555)
    os.chmod(source, 0o555)
    print(json.dumps({
        "status": "FROZEN", "files": len(files), "content_sha256": content_sha,
        "manifest": str(manifest_path),
    }, indent=2))


if __name__ == "__main__":
    main()
