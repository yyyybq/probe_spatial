#!/usr/bin/env python3
"""Verify an independent feature namespace against exact v2 request units."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_create(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            # Preserve audit readability across SCO's squashed NFS identity.
            os.fchmod(handle.fileno(), 0o644)
        os.link(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def raw_scene(raw: str) -> str:
    prefix = "scannetpp:"
    if not raw.startswith(prefix) or "/" not in raw[len(prefix):]:
        raise ValueError(f"invalid raw ID: {raw}")
    return raw[len(prefix):].split("/", 1)[0]


def output_path(root: Path, model: str, unit: dict[str, Any]) -> Path:
    return root / model / "scannetpp" / raw_scene(unit["raw_frame_id"]) / f"{unit['unit_id']}.safetensors"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--science-tag", required=True)
    parser.add_argument("--request-manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--model", required=True, choices=("dino", "f3r", "vjepa2", "wan"))
    parser.add_argument("--artifact-output", type=Path, required=True)
    parser.add_argument("--max-units", type=int)
    args = parser.parse_args()
    request = args.request_manifest.resolve()
    rows = [json.loads(line) for line in request.read_text().splitlines() if line.strip()]
    wanted = [row for row in rows if row.get("model") == args.model]
    if args.max_units is not None:
        wanted = wanted[:args.max_units]
    if not wanted:
        raise SystemExit("no requested units")
    request_sha = sha256(request)
    problems: list[dict[str, Any]] = []
    seen_paths: Counter[str] = Counter()
    verified = 0
    for unit in wanted:
        feature = output_path(args.feature_root.resolve(), args.model, unit)
        sidecar = Path(f"{feature}.manifest.json")
        seen_paths[str(feature)] += 1
        if not feature.is_file() or not sidecar.is_file() or feature.is_symlink() or sidecar.is_symlink():
            problems.append({"unit_id": unit["unit_id"], "reason": "missing_feature_or_sidecar", "feature": str(feature)})
            continue
        try:
            meta = json.loads(sidecar.read_text())
        except Exception as exc:
            problems.append({"unit_id": unit["unit_id"], "reason": "unreadable_sidecar", "detail": repr(exc)})
            continue
        expected = {
            "science_tag": args.science_tag,
            "unit_id": unit["unit_id"],
            "model": args.model,
            "layer": unit["layer"],
            "timestep": unit["timestep"],
            "feature_role": unit["feature_role"],
            "raw_frame_id": unit["raw_frame_id"],
            "context_raw_ids": unit["context_raw_ids"],
            "source_manifest_sha256": unit["source_manifest_sha256"],
            "request_manifest_sha256": request_sha,
            "fallback": False,
        }
        differing = [key for key, value in expected.items() if meta.get(key) != value]
        if differing:
            problems.append({"unit_id": unit["unit_id"], "reason": "identity_mismatch", "fields": differing})
            continue
        if feature.stat().st_size != meta.get("size_bytes") or sha256(feature) != meta.get("feature_sha256"):
            problems.append({"unit_id": unit["unit_id"], "reason": "feature_receipt_mismatch"})
            continue
        if not meta.get("extractor_sha256") or not meta.get("model_revision") or not meta.get("processor_config"):
            problems.append({"unit_id": unit["unit_id"], "reason": "missing_extractor_or_model_provenance"})
            continue
        verified += 1
    duplicates = [path for path, count in seen_paths.items() if count != 1]
    status = "COMPLETE_VALID" if verified == len(wanted) and not duplicates else "FAILED_VALIDATION"
    result = {
        "schema_version": 1,
        "science_tag": args.science_tag,
        "status": status,
        "model": args.model,
        "request_manifest": str(request),
        "request_manifest_sha256": request_sha,
        "feature_root": str(args.feature_root.resolve()),
        "requested": len(wanted),
        "verified": verified,
        "missing_or_invalid": len(problems),
        "duplicate_feature_paths": duplicates,
        "problems": problems[:200],
        "coverage_contract": "requested == verified; missing == 0; duplicate == 0; fallback == 0; raw-ID mismatch == 0",
    }
    atomic_create(args.artifact_output.resolve(), result)
    print(json.dumps(result, indent=2))
    return 0 if status == "COMPLETE_VALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
