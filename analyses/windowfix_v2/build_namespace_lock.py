#!/usr/bin/env python3
"""Bind frozen canonical manifests to new immutable feature namespaces.

The 53 MiB manifests repeat ``valid_frames`` in every row.  A namespace lock
avoids copying four otherwise-identical manifests merely to change two root
strings.  Consumers must verify both the manifest hashes and this lock before
overriding the row-level legacy roots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build(protocol_path: Path, manifest_dir: Path, base_uri: str) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    prefixes = [int(v) for v in protocol["streaming"]["prefix_lengths"]]
    manifest_sha: dict[str, str] = {}
    for prefix in prefixes:
        path = manifest_dir / f"inscene_c_h124_prefix{prefix}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        sidecar_path = Path(f"{path}.meta.json")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if sidecar.get("sha256") != digest:
            raise ValueError(f"manifest sidecar hash mismatch: {path}")
        manifest_sha[str(prefix)] = digest
    summary_path = manifest_dir / "canonical_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for prefix, digest in manifest_sha.items():
        if summary.get("manifests", {}).get(prefix, {}).get("sha256") != digest:
            raise ValueError(f"canonical summary hash mismatch for prefix {prefix}")
    namespaces = protocol["feature_namespaces"]
    base = base_uri.rstrip("/")
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_protocol": protocol["canonical_protocol"],
        "protocol_file": str(protocol_path),
        "protocol_file_sha256": sha256_file(protocol_path),
        "manifest_dir": str(manifest_dir),
        "manifest_sha256": manifest_sha,
        "canonical_universe_sha256": summary["universe_sha256"],
        "canonical_windows": summary["shared_unit_count"],
        "feature_roots": {
            "streaming": f"{base}/{namespaces['streaming']}",
            "target": f"{base}/{namespaces['target']}",
        },
        "override_policy": {
            "row_roots_are_not_authoritative": True,
            "require_manifest_sha256": True,
            "require_protocol_file_sha256": True,
            "fail_closed": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--base-uri", default="aoss://baiqiao/InsScene-15K")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and not args.force:
        parser.error(f"refusing to overwrite without --force: {output}")
    value = build(args.protocol.resolve(), args.manifest_dir.resolve(), args.base_uri)
    write_atomic(output, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
