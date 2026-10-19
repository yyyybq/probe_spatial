#!/usr/bin/env python3
"""Build positive streaming gap worklists from a frozen route catalog.

Only rows already classified as ``REEXTRACT_*`` are accepted.  Rows still in
``PENDING_REMOTE_GATE`` are deliberately excluded; a later fresh gate must
classify them as a reference or an explicit failed-gate reextract row first.
Every route is independently checked against the frozen canonical obs28 source.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.canonical_gap_worklist import (
    CANONICAL_PROTOCOL,
    CONTRACT,
    canonical_json_sha,
    sha256_file,
    streaming_worklist_preprocess_identity,
)


SCHEMA = "CANONICAL_V2_STREAMING_GAP_AUDIT_V1"
CATALOG_SCHEMA = "CANONICAL_V2_STREAMING_ROUTE_CATALOG_V1"
MANIFEST_SCHEMA = "CANONICAL_V2_STREAMING_ROUTE_CATALOG_MANIFEST_V1"
ALLOWED_STATUSES = {
    "REEXTRACT_RAW_ID_MISMATCH",
    "REEXTRACT_IDENTITY_MISMATCH",
    "REEXTRACT_UNVERIFIED",
}
PREFIXES = {8, 12, 16, 24}

PROMPT_CONTRACTS = {
    "wan": {"mode": "exact", "value": "."},
    "minimax_h3": {"mode": "exact", "value": "."},
    "lingbot": {"mode": "exact", "value": "."},
    "qwen2_5_vl": {
        "mode": "exact",
        "value": "Describe the spatial layout of this scene.",
    },
}
FIXED_PREPROCESS = {
    "wan": {"resize": [480, 832], "num_frames": 81},
    "cogvideox": {"resize": [480, 720], "num_frames": 97},
    "vjepa": {"resize": [480, 832], "num_frames": 76},
    "vjepa2": {"resize": [256, 256], "num_frames": 64},
    "dino": {"resize": [420, 728], "num_frames": 64},
    "videomae": {"resize": [224, 224], "num_frames": 16},
    "aether": {"resize": [480, 720], "num_frames": 81},
    "f3r": {"resize": [288, 512], "num_frames": 64},
    "minimax_h3": {"resize": [480, 832], "num_frames": 73},
    "lingbot": {"resize": [480, 832], "num_frames": 49},
    "qwen2_5_vl": {"resize": [336, 336], "num_frames": 8},
}

PREPROCESS = {
    vfm: streaming_worklist_preprocess_identity(
        vfm, value["resize"], int(value["num_frames"])
    )
    for vfm, value in FIXED_PREPROCESS.items()
}


class GapBuildError(ValueError):
    pass


def read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise GapBuildError(f"{label} must be a regular non-symlink file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GapBuildError(f"{label} must contain one JSON object")
    return value


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise GapBuildError(f"route file must be regular: {path}")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise GapBuildError(f"invalid route JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise GapBuildError(f"route is not an object at {path}:{line_number}")
            yield value


def canonical_windows(path: Path) -> tuple[dict[tuple[str, int], tuple[int, ...]], str]:
    digest = sha256_file(path)
    windows: dict[tuple[str, int], tuple[int, ...]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            scene = str(row["scene_id"])
            window_id = int(row["streaming_window_id"])
            values = tuple(int(value) for value in row["streaming_window_indices"])
            if len(values) != 28 or any(right <= left for left, right in zip(values, values[1:])):
                raise GapBuildError(f"canonical source line {line_number} is not strict obs28")
            key = (scene, window_id)
            if key in windows:
                raise GapBuildError(f"duplicate canonical window: {key}")
            windows[key] = values
    if not windows:
        raise GapBuildError("canonical source is empty")
    return windows, digest


def validate_catalog(directory: Path, protocol_file_sha: str) -> tuple[dict, dict, dict[str, str]]:
    manifest_path = directory / "catalog_manifest.json"
    summary_path = directory / "summary.json"
    manifest = read_json(manifest_path, "catalog manifest")
    summary = read_json(summary_path, "catalog summary")
    if manifest.get("schema") != MANIFEST_SCHEMA or summary.get("schema") != CATALOG_SCHEMA:
        raise GapBuildError("streaming catalog schema mismatch")
    if summary.get("current_protocol_file_sha256_observed") != protocol_file_sha:
        raise GapBuildError("catalog was not built against the current protocol file")
    outputs = {}
    for item in manifest.get("outputs", []):
        name = str(item.get("file", ""))
        path = directory / name
        if path.parent != directory or sha256_file(path) != item.get("sha256"):
            raise GapBuildError(f"catalog output SHA mismatch: {name}")
        if path.stat().st_size != item.get("size_bytes"):
            raise GapBuildError(f"catalog output size mismatch: {name}")
        outputs[name] = item["sha256"]
    if outputs.get("summary.json") != sha256_file(summary_path):
        raise GapBuildError("catalog manifest does not bind summary.json")
    return manifest, summary, outputs


def validate_route(
    route: Mapping[str, Any], windows: Mapping[tuple[str, int], tuple[int, ...]]
) -> dict[str, Any]:
    if route.get("schema") != CATALOG_SCHEMA:
        raise GapBuildError("route schema mismatch")
    status = str(route.get("status"))
    if status not in ALLOWED_STATUSES or route.get("action") != "REEXTRACT":
        raise GapBuildError(f"non-explicit reextract route entered gap builder: {status}")
    scene = str(route["scene_id"])
    window_id = int(route["window_id"])
    prefix = int(route["prefix"])
    layer = int(route["feature_layer"])
    if prefix not in PREFIXES or (scene, window_id) not in windows:
        raise GapBuildError("route has a non-canonical scene/window/prefix")
    expected = windows[(scene, window_id)][:prefix]
    actual = tuple(int(value) for value in route["canonical_raw_ids"])
    if actual != expected:
        raise GapBuildError(f"route canonical raw IDs disagree with source: {scene}/w{window_id}/p{prefix}")
    raw_sha = hashlib.sha256(
        json.dumps(list(expected), separators=(",", ":")).encode()
    ).hexdigest()
    if route.get("canonical_raw_ids_sha256") != raw_sha:
        raise GapBuildError("route canonical raw-ID SHA mismatch")
    reasons = route.get("reason_codes")
    if not isinstance(reasons, list) or not reasons:
        raise GapBuildError("reextract route has no reason code")
    return {
        "scene": scene,
        "window_id": window_id,
        "prefix": prefix,
        "layer": layer,
        "vfm": str(route["vfm"]),
        "model_id": str(route["model_id"]),
        "timestep": int(route["timestep"]),
        "reason": "+".join(sorted({status, *[str(value).upper() for value in reasons]})),
        "unit_id": str(route["unit_id"]),
    }


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def build(
    *,
    catalog_dir: Path,
    protocol_path: Path,
    canonical_source: Path,
    failed_gate_routes: Sequence[Path],
) -> dict[str, Any]:
    protocol = read_json(protocol_path, "canonical protocol")
    protocol_file_sha = sha256_file(protocol_path)
    if protocol.get("canonical_protocol") != CANONICAL_PROTOCOL:
        raise GapBuildError("canonical protocol identity mismatch")
    windows, source_sha = canonical_windows(canonical_source)
    manifest, summary, output_shas = validate_catalog(catalog_dir, protocol_file_sha)
    route_paths = [catalog_dir / "streaming_reextract.routes.jsonl.gz", *failed_gate_routes]
    normalized = []
    seen_units = set()
    route_bindings = []
    for path in route_paths:
        path = path.resolve()
        if path == (catalog_dir / "streaming_reextract.routes.jsonl.gz").resolve():
            if output_shas.get(path.name) != sha256_file(path):
                raise GapBuildError("catalog does not bind streaming reextract routes")
        route_bindings.append({"path": str(path), "sha256": sha256_file(path)})
        for raw in iter_jsonl(path):
            row = validate_route(raw, windows)
            if row["unit_id"] in seen_units:
                raise GapBuildError(f"duplicate reextract unit: {row['unit_id']}")
            seen_units.add(row["unit_id"])
            normalized.append(row)
    expected_base = sum(
        int(count)
        for status, count in summary.get("counts", {}).get("status", {}).items()
        if str(status) in {"REEXTRACT_RAW_ID_MISMATCH", "REEXTRACT_IDENTITY_MISMATCH"}
    )
    base_count = sum(
        1
        for row in iter_jsonl(catalog_dir / "streaming_reextract.routes.jsonl.gz")
    )
    if base_count != expected_base:
        raise GapBuildError(f"catalog reextract count mismatch: {base_count} != {expected_base}")

    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in normalized:
        groups[(row["vfm"], row["model_id"], row["timestep"])].append(row)
    worklists = []
    for (vfm, model_id, timestep), rows in sorted(groups.items()):
        if vfm not in PREPROCESS:
            raise GapBuildError(f"no frozen preprocessing identity for VFM {vfm}")
        entries_by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
        for row in rows:
            key = (row["scene"], row["window_id"], row["prefix"])
            entry = entries_by_key.setdefault(
                key,
                {
                    "scene": row["scene"], "window_id": row["window_id"],
                    "prefix": row["prefix"], "layers": [], "reasons": set(),
                },
            )
            if row["layer"] in entry["layers"]:
                raise GapBuildError(f"duplicate layer in stream gap key: {key}/{row['layer']}")
            entry["layers"].append(row["layer"])
            entry["reasons"].add(row["reason"])
        entries = []
        for key in sorted(entries_by_key):
            item = entries_by_key[key]
            entries.append(
                {
                    "scene": item["scene"], "window_id": item["window_id"],
                    "prefix": item["prefix"], "layers": sorted(item["layers"]),
                    "reason": "+".join(sorted(item["reasons"])),
                }
            )
        worklist = {
            "schema_version": 1, "contract": CONTRACT,
            "canonical_protocol": CANONICAL_PROTOCOL,
            "protocol_sha256": protocol["canonical_protocol_sha256"],
            "protocol_file_sha256": protocol_file_sha,
            "canonical_source_sha256": source_sha,
            "mode": "streaming_prefix", "vfm": vfm, "model_id": model_id,
            "timestep": timestep,
            "feature_namespace": "FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1",
            "output_layers": sorted({row["layer"] for row in rows}),
            "source": "scannetpp",
            "prompt_contract": PROMPT_CONTRACTS.get(
                vfm, {"mode": "not_applicable", "value": None}
            ),
            "preprocess_identity": PREPROCESS[vfm],
            "entries": entries,
            "counts": {
                "entries": len(entries),
                "layer_items": sum(len(item["layers"]) for item in entries),
                "scenes": len({item["scene"] for item in entries}),
            },
        }
        worklist["content_sha256"] = canonical_json_sha(worklist)
        worklists.append(worklist)
    return {
        "protocol": protocol,
        "protocol_path": str(protocol_path),
        "protocol_file_sha256": protocol_file_sha,
        "canonical_source": {"path": str(canonical_source), "sha256": source_sha},
        "catalog": {
            "directory": str(catalog_dir),
            "manifest_sha256": sha256_file(catalog_dir / "catalog_manifest.json"),
            "summary_sha256": sha256_file(catalog_dir / "summary.json"),
        },
        "route_bindings": route_bindings,
        "normalized": normalized,
        "worklists": worklists,
    }


def write_outputs(output_dir: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    if output_dir.exists():
        raise GapBuildError(f"refusing to replace immutable output directory: {output_dir}")
    staging = output_dir.with_name(f".{output_dir.name}.tmp-{os.getpid()}")
    staging.mkdir(parents=True)
    bindings = []
    for worklist in result["worklists"]:
        token = canonical_json_sha(
            [worklist["vfm"], worklist["model_id"], worklist["timestep"]]
        )[:12]
        name = f"{worklist['vfm']}-t{worklist['timestep']}-{token}.json"
        path = staging / "gap_worklists" / name
        payload = (json.dumps(worklist, indent=2, sort_keys=True) + "\n").encode()
        atomic_bytes(path, payload)
        bindings.append(
            {
                "path": str(output_dir / "gap_worklists" / name),
                "file_sha256": hashlib.sha256(payload).hexdigest(),
                "content_sha256": worklist["content_sha256"],
                "counts": worklist["counts"],
            }
        )
    audit = {
        "schema_version": 1, "audit_type": SCHEMA,
        "status": "INCOMPLETE" if bindings else "PASS",
        "ready_to_extract_gaps": bool(bindings),
        "canonical_protocol": CANONICAL_PROTOCOL,
        "protocol_file": result["protocol_path"],
        "protocol_file_sha256": result["protocol_file_sha256"],
        "canonical_source": result["canonical_source"],
        "catalog": result["catalog"], "route_bindings": result["route_bindings"],
        "counts": {
            "hard_errors": 0, "gap_worklists": len(bindings),
            "gap_entries": sum(item["counts"]["entries"] for item in bindings),
            "gap_layer_items": sum(item["counts"]["layer_items"] for item in bindings),
            "scenes_sum_nonunique": sum(item["counts"]["scenes"] for item in bindings),
        },
        "hard_errors": [], "outputs": {"gap_worklists": bindings},
        "notes": [
            "PENDING_REMOTE_GATE rows were not scheduled.",
            "Only explicit REEXTRACT_* routes are positive extraction work.",
        ],
    }
    audit["content_sha256"] = canonical_json_sha(audit)
    atomic_bytes(
        staging / "streaming_gap_audit.json",
        (json.dumps(audit, indent=2, sort_keys=True) + "\n").encode(),
    )
    os.replace(staging, output_dir)
    return audit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-dir", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--canonical-source", required=True, type=Path)
    parser.add_argument("--failed-gate-routes", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = build(
            catalog_dir=args.catalog_dir.resolve(),
            protocol_path=args.protocol.resolve(),
            canonical_source=args.canonical_source.resolve(),
            failed_gate_routes=[path.resolve() for path in args.failed_gate_routes],
        )
        audit = write_outputs(args.output_dir.resolve(), result)
    except GapBuildError as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}))
        return 2
    print(json.dumps({"status": audit["status"], "counts": audit["counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
