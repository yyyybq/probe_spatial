#!/usr/bin/env python3
"""Validate feature sidecars, sizes, checksums and safetensors readability."""

import argparse
import hashlib
import json
from pathlib import Path

from safetensors import safe_open


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--require-sidecar", action="store_true")
    parser.add_argument("--skip-checksum", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    files = sorted(root.rglob("*.sft"))
    failures = []
    for path in files:
        sidecar = Path(f"{path}.manifest.json")
        if not sidecar.exists():
            if args.require_sidecar:
                failures.append((path, "missing sidecar"))
            continue
        try:
            meta = json.loads(sidecar.read_text())
            if path.stat().st_size != meta["size_bytes"]:
                raise ValueError("size mismatch")
            if meta.get("sha256") and not args.skip_checksum and sha256(path) != meta["sha256"]:
                raise ValueError("checksum mismatch")
            with safe_open(path, framework="pt", device="cpu") as handle:
                actual = set(handle.keys())
            if actual != set(meta["tensors"]):
                raise ValueError(f"tensor keys mismatch: {actual}")
        except Exception as exc:
            failures.append((path, str(exc)))
    npy_files = sorted(root.rglob("*.npy"))
    for path in npy_files:
        sidecar = Path(f"{path}.manifest.json")
        if not sidecar.exists():
            if args.require_sidecar:
                failures.append((path, "missing sidecar"))
            continue
        try:
            meta = json.loads(sidecar.read_text())
            if path.stat().st_size != meta["size_bytes"]:
                raise ValueError("size mismatch")
            if meta.get("sha256") and not args.skip_checksum and sha256(path) != meta["sha256"]:
                raise ValueError("checksum mismatch")
        except Exception as exc:
            failures.append((path, str(exc)))
    print(f"checked={len(files) + len(npy_files)} failures={len(failures)}")
    for path, reason in failures[:50]:
        print(f"FAIL {path}: {reason}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
