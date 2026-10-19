#!/usr/bin/env python3
"""Freeze an audit-only AOSS gate request subset for the 816 probe rows."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from datetime import datetime, timezone
from collections import Counter
from pathlib import Path


DISK_LAYER_OVERRIDES = {("lingbot", 48): 47, ("minimax_h3", 50): 49}


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_binding(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": file_sha(path),
        "size_bytes": path.stat().st_size,
    }


def read_object(path: Path, label: str) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def validate_lineage(
    requests: Path,
    source_manifest_path: Path,
    source_summary_path: Path,
    namespace_lock_path: Path,
) -> tuple[dict, dict, dict]:
    source_manifest = read_object(source_manifest_path, "source catalog manifest")
    source_summary = read_object(source_summary_path, "source catalog summary")
    namespace_lock = read_object(namespace_lock_path, "canonical namespace lock")
    outputs = source_manifest.get("outputs")
    if not isinstance(outputs, list):
        raise ValueError("source catalog manifest lacks outputs")
    request_matches = [item for item in outputs if item.get("file") == requests.name]
    summary_matches = [item for item in outputs if item.get("file") == source_summary_path.name]
    if len(request_matches) != 1 or len(summary_matches) != 1:
        raise ValueError("source manifest must bind source requests and summary exactly once")
    for path, binding, label in (
        (requests, request_matches[0], "source requests"),
        (source_summary_path, summary_matches[0], "source summary"),
    ):
        if binding.get("sha256") != file_sha(path) or int(binding.get("size_bytes", -1)) != path.stat().st_size:
            raise ValueError(f"{label} binding disagrees with source manifest")
    source_sha = str(namespace_lock.get("canonical_source", {}).get("sha256", ""))
    universe_sha = str(namespace_lock.get("canonical_universe_sha256", ""))
    if len(source_sha) != 64 or len(universe_sha) != 64:
        raise ValueError("canonical namespace lock lacks SHA256 lineage")
    for label, value in (("source manifest", source_manifest), ("source summary", source_summary)):
        if value.get("canonical_parent_source_sha256") != source_sha:
            raise ValueError(f"{label} canonical source lineage mismatch")
        if value.get("canonical_universe_sha256") != universe_sha:
            raise ValueError(f"{label} canonical universe lineage mismatch")
    declared_lock = Path(str(source_summary.get("canonical_namespace_lock", ""))).resolve()
    if declared_lock != namespace_lock_path.resolve():
        raise ValueError("source summary does not bind the supplied namespace lock")
    return source_manifest, source_summary, namespace_lock


def select(
    experiments: Path,
    requests: Path,
    output_dir: Path,
    source_manifest_path: Path | None = None,
    source_summary_path: Path | None = None,
    namespace_lock_path: Path | None = None,
) -> dict:
    if output_dir.exists():
        raise ValueError(f"output already exists: {output_dir}")
    source_manifest_path = source_manifest_path or requests.parent / "catalog_manifest.json"
    source_summary_path = source_summary_path or requests.parent / "summary.json"
    source_summary = read_object(source_summary_path, "source catalog summary")
    namespace_lock_path = namespace_lock_path or Path(str(source_summary.get("canonical_namespace_lock", "")))
    source_summary = validate_lineage(
        requests, source_manifest_path, source_summary_path, namespace_lock_path
    )[1]
    with experiments.open(newline="", encoding="utf-8") as handle:
        selected = list(csv.DictReader(handle))
    if len(selected) != 816 or len({r["experiment_id"] for r in selected}) != 816:
        raise ValueError("expected exactly 816 unique frozen experiment IDs")
    required = {
        (r["model"], DISK_LAYER_OVERRIDES.get((r["model"], int(r["layer"])), int(r["layer"])))
        for r in selected
    }
    output_dir.mkdir(parents=True)
    destination = output_dir / "remote_gate_requests_816.jsonl.gz"
    counts = Counter()
    seen = set()
    with gzip.open(requests, "rt", encoding="utf-8") as source, gzip.GzipFile(
        filename=str(destination), mode="wb", mtime=0
    ) as sink:
        for line in source:
            if not line.strip():
                raise ValueError("blank request line")
            value = json.loads(line)
            expected = value["expected"]
            spec = (str(expected["vfm"]), int(expected["feature_layer"]))
            if spec not in required:
                continue
            unit_id = str(value["unit_id"])
            if unit_id in seen:
                raise ValueError(f"duplicate request unit: {unit_id}")
            seen.add(unit_id)
            counts[f"{spec[0]}:L{spec[1]}"] += 1
            sink.write(line.encode("utf-8"))
    if not counts or not {model for model, _ in required} == {
        item.split(":", 1)[0] for item in counts
    }:
        raise ValueError("selected model coverage disagrees with request subset")
    created_at = datetime.now(timezone.utc).isoformat()
    selection_summary = {
        "schema": "canonical_v2_probe_gate_request_subset_v1",
        "status": "REVIEW_ONLY_NOT_EXECUTED",
        "audit_only": True,
        "created_at": created_at,
        "experiments": {**file_binding(experiments), "rows": len(selected)},
        "source_requests": file_binding(requests),
        "source_catalog_manifest": file_binding(source_manifest_path),
        "source_catalog_summary": file_binding(source_summary_path),
        "canonical_namespace_lock": file_binding(namespace_lock_path),
        "output": {**file_binding(destination), "rows": len(seen)},
        "counts_by_model_disk_layer": dict(sorted(counts.items())),
        "logical_to_disk_layer_overrides": {"lingbot:48": 47, "minimax_h3:50": 49},
        "scheduler_writes_performed": False,
    }
    selection_path = output_dir / "selection_summary.json"
    selection_path.write_text(
        json.dumps(selection_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    catalog_summary = {
        "schema": "CANONICAL_V2_PROBE816_STREAMING_GATE_SUBSET_SUMMARY_V1",
        "status": "REVIEW_ONLY_NOT_EXECUTED",
        "created_at": created_at,
        "canonical_namespace_lock": str(namespace_lock_path.resolve()),
        "canonical_parent_source_sha256": source_summary["canonical_parent_source_sha256"],
        "canonical_universe_sha256": source_summary["canonical_universe_sha256"],
        "canonical_protocol_lineage_file_sha256": source_summary.get("canonical_protocol_lineage_file_sha256"),
        "canonical_protocol_sha256": source_summary.get("canonical_protocol_sha256"),
        "claims": {
            "jobs_submitted": 0, "remote_reads_performed": 0,
            "payload_full_read_performed": False, "feature_payloads_copied": 0,
        },
        "counts": {
            "status": {"PENDING_REMOTE_GATE": len(seen)},
            "feature_specs": len(counts), "experiments": len(selected),
        },
        "remote_policy": {"pending_units": len(seen)},
        "subset_selection": {**file_binding(selection_path), "schema": selection_summary["schema"]},
        "source_catalog": {
            "manifest": file_binding(source_manifest_path),
            "summary": file_binding(source_summary_path), "requests": file_binding(requests),
        },
    }
    catalog_summary_path = output_dir / "catalog_summary.json"
    catalog_summary_path.write_text(
        json.dumps(catalog_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema": "CANONICAL_V2_PROBE816_STREAMING_GATE_SUBSET_MANIFEST_V1",
        "created_at": created_at,
        "canonical_parent_source_sha256": source_summary["canonical_parent_source_sha256"],
        "canonical_universe_sha256": source_summary["canonical_universe_sha256"],
        "source_catalog_manifest": file_binding(source_manifest_path),
        "outputs": [
            {"file": destination.name, "sha256": file_sha(destination), "size_bytes": destination.stat().st_size},
            {"file": selection_path.name, "sha256": file_sha(selection_path), "size_bytes": selection_path.stat().st_size},
            {"file": catalog_summary_path.name, "sha256": file_sha(catalog_summary_path), "size_bytes": catalog_summary_path.stat().st_size},
        ],
        "note": "catalog_manifest.json is excluded from its own non-circular digest set",
    }
    manifest_path = output_dir / "catalog_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**selection_summary, "catalog_summary": file_binding(catalog_summary_path), "catalog_manifest": file_binding(manifest_path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-catalog-manifest", type=Path)
    parser.add_argument("--source-catalog-summary", type=Path)
    parser.add_argument("--namespace-lock", type=Path)
    args = parser.parse_args()
    print(json.dumps(select(
        args.experiments, args.requests, args.output_dir, args.source_catalog_manifest,
        args.source_catalog_summary, args.namespace_lock,
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
