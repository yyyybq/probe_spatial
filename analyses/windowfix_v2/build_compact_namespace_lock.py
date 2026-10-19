#!/usr/bin/env python3
"""Bind one frozen obs28 source to the isolated canonical-v2 namespaces."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _line(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def build(protocol_path: Path, source: Path, base_uri: str) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("canonical_protocol") != "inscene_streaming_canonical_v2_obs28_h124":
        raise ValueError("unexpected canonical protocol")
    if int(protocol.get("canonical_window", {}).get("observations", -1)) != 28:
        raise ValueError("canonical protocol is not obs28")
    source_sha = sha256_file(source)
    sidecar_path = Path(f"{source}.meta.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("sha256") != source_sha:
        raise ValueError("source manifest sidecar SHA mismatch")

    digest = hashlib.sha256()
    units: set[tuple[str, int]] = set()
    scenes: set[str] = set()
    with source.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            scene = str(row["scene_id"])
            window_id = int(row["streaming_window_id"])
            window = [int(value) for value in row["streaming_window_indices"]]
            source_prefix = [int(value) for value in row["streaming_indices"]]
            if len(window) != 28 or window != sorted(set(window)):
                raise ValueError(f"line {line_number}: source is not a strict obs28 window")
            if source_prefix != window[:24]:
                raise ValueError(f"line {line_number}: source is not p24=window[:24]")
            unit = (scene, window_id)
            if unit in units:
                raise ValueError(f"line {line_number}: duplicate scene/window")
            units.add(unit)
            scenes.add(scene)
            digest.update(
                _line(
                    {
                        "scene_id": scene,
                        "split": str(row["split"]),
                        "streaming_window_id": window_id,
                        "streaming_window_indices": window,
                    }
                )
            )
    if len(units) != 2721 or len(scenes) != 681:
        raise ValueError(f"unexpected source universe: windows={len(units)} scenes={len(scenes)}")
    namespaces = protocol["feature_namespaces"]
    base = base_uri.rstrip("/")
    return {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_protocol": protocol["canonical_protocol"],
        "protocol_file": str(protocol_path),
        "protocol_file_sha256": sha256_file(protocol_path),
        "canonical_source": {
            "path": str(source),
            "sha256": source_sha,
            "sidecar_path": str(sidecar_path),
            "sidecar_sha256": sha256_file(sidecar_path),
            "rows": len(units),
            "scenes": len(scenes),
            "canonical_obs_len": 28,
            "source_prefix": 24,
        },
        "canonical_universe_sha256": digest.hexdigest(),
        "feature_roots": {
            "streaming": f"{base}/{namespaces['streaming']}",
            "target": f"{base}/{namespaces['target']}",
        },
        "derivation": {
            "streaming": "canonical_window_indices[:prefix]",
            "anchor": "canonical_window_indices[prefix-1]",
            "future": "canonical_window_indices[prefix-1+horizon]",
            "prefixes": [8, 12, 16, 24],
            "horizons": [1, 2, 4],
        },
        "override_policy": {
            "row_roots_are_not_authoritative": True,
            "require_protocol_file_sha256": True,
            "require_source_manifest_sha256": True,
            "require_label_lineage_audit": True,
            "fail_closed": True,
        },
    }


def write_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--canonical-source", type=Path, required=True)
    parser.add_argument("--base-uri", default="aoss://baiqiao/InsScene-15K")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and not args.force:
        parser.error(f"refusing to overwrite without --force: {output}")
    value = build(
        args.protocol.resolve(), args.canonical_source.resolve(), args.base_uri
    )
    write_atomic(output, value)
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
