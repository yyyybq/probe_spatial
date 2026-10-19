#!/usr/bin/env python3
"""Build an audit-only route inventory for the frozen canonical-v2 subset.

This only indexes already audited routes.  It never copies feature payloads or
claims that a missing route is available. This inventory is NOT a loader root
or an executable overlay and cannot authorize a probing launch.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Iterable

DISK_LAYER_OVERRIDES = {
    ("lingbot", 48): 47,
    ("minimax_h3", 50): 49,
}
TARGET_PROBES = {"action_dynamics", "path_integration", "counterfactual"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with open(fd, "w", encoding="utf-8", closefd=True) as f:
            json.dump(value, f, indent=2, sort_keys=True, ensure_ascii=False)
            f.write("\n")
        Path(tmp).replace(path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def compact_route(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("unit_kind") == "target_row":
        fields = (
            "unit_kind", "spec_id", "source_feature_path", "source_index_path",
            "source_row_index", "canonical_row_index", "target_raw_id",
            "source_tensor_header_sha256", "source_size_bytes", "tensor_key",
        )
    else:
        fields = (
            "unit_id", "spec_id", "source_uri", "feature_filename",
            "window_id", "prefix", "tail", "canonical_raw_ids_sha256",
            "byte_level_certified", "payload_hash_performed",
        )
    return {name: row.get(name) for name in fields}


def canonical_key(row: dict[str, Any]) -> tuple[Any, ...]:
    mode = "target" if row.get("unit_kind") == "target_row" else "streaming"
    layer = row.get("layer", row.get("feature_layer"))
    prefix = row.get("prefix", row.get("prefix_len", -1))
    return (
        mode,
        route_vfm(row),
        str(row.get("source") or ""),
        str(row.get("scene_id") or ""),
        int(layer),
        None if row.get("timestep") is None else int(row["timestep"]),
        int(row.get("window_id", -1)) if mode == "streaming" else int(row.get("canonical_row_index", -1)),
        int(prefix) if mode == "streaming" else int(row.get("target_raw_id", -1)),
    )


def route_vfm(row: dict[str, Any]) -> str:
    value = str(row.get("vfm") or "").strip()
    if value:
        return value
    # Target audit rows carry the VFM in spec_id (target-dino-t749-l0).
    spec = str(row.get("spec_id") or "")
    if spec.startswith("target-"):
        parts = spec.split("-")
        if len(parts) >= 2:
            return parts[1]
    return ""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--experiments", type=Path, required=True)
    p.add_argument("--streaming-routes", type=Path, required=True)
    p.add_argument("--target-routes", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    selected = list(csv.DictReader(args.experiments.open(encoding="utf-8")))
    selected_models = {r["model"] for r in selected}
    selected_layers = {
        (r["model"], DISK_LAYER_OVERRIDES.get((r["model"], int(r["layer"])), int(r["layer"])))
        for r in selected
    }
    target_layers = {
        (r["model"], DISK_LAYER_OVERRIDES.get((r["model"], int(r["layer"])), int(r["layer"])))
        for r in selected if r["probe_config"] in TARGET_PROBES
    }
    # Route files use the canonical VFM names; experiments use the same names
    # in the frozen ledger.  Keep only the eight-model campaign explicitly.
    stream = []
    for row in rows(args.streaming_routes):
        vfm = route_vfm(row)
        layer = row.get("feature_layer", row.get("layer"))
        try:
            layer = int(layer)
        except (TypeError, ValueError):
            continue
        if (vfm, layer) in selected_layers:
            stream.append(row)
    target = []
    for row in rows(args.target_routes):
        vfm = route_vfm(row)
        layer = row.get("layer")
        try:
            layer = int(layer)
        except (TypeError, ValueError):
            continue
        if (vfm, layer) in target_layers:
            target.append(row)

    route_rows = stream + target
    owners: dict[tuple[Any, ...], dict[str, Any]] = {}
    duplicate_keys = []
    for row in route_rows:
        key = canonical_key(row)
        if key in owners:
            duplicate_keys.append(key)
        else:
            owners[key] = row

    # This is an index of available audited routes, not a claim of complete
    # feature coverage.  Completeness is checked by the launch gate against
    # the canonical unit manifest and remains false until all units are present.
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    route_path = out / "routes.jsonl.gz"
    with gzip.open(route_path, "wt", encoding="utf-8", compresslevel=6) as f:
        for key in sorted(owners, key=lambda x: json.dumps(x, default=str)):
            f.write(json.dumps({"key": list(key), "route": compact_route(owners[key])}, sort_keys=True, ensure_ascii=False) + "\n")
    manifest = {
        "schema": "canonical_v2_probe_route_inventory_v1",
        "status": "NOT_READY",
        "audit_only": True,
        "fail_closed": True,
        "payload_copy_performed": False,
        "payload_full_read_performed": False,
        "source_experiments": {"path": str(args.experiments.resolve()), "sha256": sha256(args.experiments), "rows": len(selected)},
        "source_routes": {
            "streaming": {"path": str(args.streaming_routes.resolve()), "sha256": sha256(args.streaming_routes), "rows_selected": len(stream)},
            "target": {"path": str(args.target_routes.resolve()), "sha256": sha256(args.target_routes), "rows_selected": len(target)},
        },
        "route_file": {"path": str(route_path.resolve()), "sha256": sha256(route_path), "rows": len(owners)},
        "counts": {"selected_models": sorted(selected_models), "selected_experiments": len(selected), "unique_routes": len(owners), "duplicate_route_keys": len(duplicate_keys)},
        "logical_to_disk_layer_overrides": {"lingbot:48": 47, "minimax_h3:50": 49},
        "readiness": {"complete_unit_coverage_verified": False, "reason": "legacy routes alone omit gap-delta receipts; no loader route resolution or training canary"},
    }
    atomic_json(out / "catalog_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
