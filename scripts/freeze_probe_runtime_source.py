#!/usr/bin/env python3
"""Hash and make one generated probing source tree immutable."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    if source.name != "source" or "runtime" not in source.parts or not source.is_dir():
        raise RuntimeError(f"refusing unexpected source tree: {source}")
    for cache in sorted(source.rglob("__pycache__"), reverse=True):
        shutil.rmtree(cache)
    rows = []
    for path in sorted(source.rglob("*")):
        if path.is_file():
            relative = path.relative_to(source).as_posix()
            rows.append((sha256(path), relative, path.stat().st_size))
            os.chmod(path, 0o555 if os.access(path, os.X_OK) else 0o444)
    checksum = source.parent / "source.sha256"
    checksum.write_text("".join(f"{digest}  {relative}\n" for digest, relative, _ in rows))
    os.chmod(checksum, 0o444)
    manifest = source.parent / "source_manifest.json"
    manifest.write_text(json.dumps({
        "schema": "MAIN_V2_PROBE_R7_IMMUTABLE_SOURCE_V1",
        "status": "FROZEN",
        "source": str(source),
        "files": len(rows),
        "bytes": sum(size for _, _, size in rows),
        "source_sha256_file": str(checksum),
        "source_sha256_file_sha256": sha256(checksum),
    }, indent=2, sort_keys=True) + "\n")
    os.chmod(manifest, 0o444)
    for directory in sorted((p for p in source.rglob("*") if p.is_dir()), reverse=True):
        os.chmod(directory, 0o555)
    os.chmod(source, 0o555)
    print(manifest.read_text())


if __name__ == "__main__":
    main()
