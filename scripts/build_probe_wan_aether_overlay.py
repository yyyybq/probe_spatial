#!/usr/bin/env python3
"""Build a Wan/Aether owner overlay from official-key gate receipts.

Owners come only from PASS receipts in this campaign. The frozen 816 overlay
and request SHA are never read or rewritten. READY requires a unique verified
owner for every official ``(vfm, scene, layer, t, prefix, window_id)`` key.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.owner_catalog import OWNER_ELIGIBLE, sha256_file, streaming_key


SHARD_DIR = re.compile(r"^gate-(\d+)-of-(\d+)$")
PASS_STATUS = "REFERENCE_SOURCE_READ_ONLY"
GAP_NAMESPACE = "FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1"


def official_key(row: dict[str, Any]) -> tuple[Any, ...] | None:
    expected = row.get("expected") or {}
    try:
        return streaming_key(
            vfm=str(expected["vfm"]),
            source=str(expected.get("source") or "scannetpp"),
            scene_id=str(expected["scene_id"]),
            feature_layer=int(expected["feature_layer"]),
            timestep=int(expected["timestep"]),
            window_id=int(expected["window_id"]),
            prefix=int(expected["prefix"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def jsonl(path: Path):
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def hole_entry(key: tuple[Any, ...], unit_id: str) -> dict[str, Any]:
    _, vfm, source, scene_id, layer, timestep, window_id, prefix = key
    return {
        "key": list(key),
        "unit_id": unit_id,
        "vfm": vfm,
        "source": source,
        "scene_id": scene_id,
        "feature_layer": int(layer),
        "timestep": None if timestep is None else int(timestep),
        "window_id": int(window_id),
        "prefix": int(prefix),
    }


def write_worklists(output_dir: Path, holes: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[tuple[str, int, int], list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for hole in holes:
        vfm = str(hole["vfm"])
        grouped[vfm][
            (str(hole["scene_id"]), int(hole["window_id"]), int(hole["prefix"]))
        ].append(int(hole["feature_layer"]))
    written = {}
    for vfm, units in grouped.items():
        entries = []
        layer_items = 0
        scenes = set()
        for (scene, window_id, prefix), layers in sorted(units.items()):
            unique_layers = sorted(set(layers))
            layer_items += len(unique_layers)
            scenes.add(scene)
            entries.append(
                {
                    "layers": unique_layers,
                    "prefix": prefix,
                    "reason": "OFFICIAL_KEY_GATE_FAIL_OR_MISSING",
                    "scene": scene,
                    "window_id": window_id,
                }
            )
        payload = {
            "canonical_protocol": "inscene_streaming_canonical_v2_obs28_h124",
            "canonical_source_sha256": "ebc496eb0587cda372ccf29a7c5244ef454096151f73c8eed9511967ded1902d",
            "contract": "CANONICAL_V2_GAP_WORKLIST_V1",
            "mode": "streaming_prefix",
            "vfm": vfm,
            "timestep": 749,
            "counts": {
                "entries": len(entries),
                "layer_items": layer_items,
                "scenes": len(scenes),
            },
            "entries": entries,
        }
        path = output_dir / f"{vfm}_official_key_holes.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written[vfm] = {
            "path": str(path),
            "entries": len(entries),
            "layer_items": layer_items,
            "scenes": len(scenes),
        }
    return written


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--gate-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=32)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"output already exists: {args.output_dir}")

    request_sha = sha256_file(args.requests)
    shard_dirs = sorted(
        path for path in args.gate_root.iterdir()
        if path.is_dir() and SHARD_DIR.match(path.name)
    )
    owners: dict[tuple[Any, ...], dict[str, Any]] = {}
    requested: dict[tuple[Any, ...], str] = {}
    conflicts: list[str] = []
    unkeyable: list[str] = []
    reasons: list[str] = []
    status_counts: Counter[str] = Counter()
    processed = 0
    seen: set[int] = set()

    for shard_dir in shard_dirs:
        match = SHARD_DIR.match(shard_dir.name)
        assert match is not None
        index = int(match.group(1))
        summary = json.loads((shard_dir / "summary.json").read_text(encoding="utf-8"))
        if summary.get("requests_sha256") != request_sha:
            reasons.append(f"shard {index} request SHA mismatch")
            continue
        if int(summary.get("shard_count", -1)) != args.expected_shards:
            reasons.append(f"shard {index} shard_count mismatch")
            continue
        seen.add(index)
        processed += int(summary.get("processed", 0))
        for row in jsonl(shard_dir / "receipts.jsonl.gz"):
            status = str(row.get("status") or "")
            status_counts[status] += 1
            unit_id = str(row.get("unit_id") or "")
            key = official_key(row)
            if key is None:
                unkeyable.append(unit_id or f"shard-{index}-unkeyed")
                continue
            if key in requested and requested[key] != unit_id:
                conflicts.append(str(key))
            requested[key] = unit_id
            if status != PASS_STATUS:
                continue
            uri = row.get("source_uri")
            if not uri:
                reasons.append(f"pass receipt missing source_uri: {unit_id}")
                continue
            if key in owners and owners[key].get("source_uri") != uri:
                conflicts.append(str(key))
                continue
            owners[key] = {
                "key": list(key),
                "origin": "gate_pass",
                "reason_codes": row.get("reason_codes"),
                "source_uri": uri,
                "status": status,
                "unit_id": unit_id,
            }

    missing = [index for index in range(args.expected_shards) if index not in seen]
    if missing:
        reasons.append(f"missing shards: {missing[:12]}")
    if unkeyable:
        reasons.append(f"unkeyable receipts: {len(unkeyable)}")
    if conflicts:
        reasons.append(f"owner key conflicts: {len(conflicts)}")
    hole_keys = [key for key in requested if key not in owners]
    hole_by_vfm = Counter(str(key[1]) for key in hole_keys)
    if hole_keys:
        reasons.append(f"unowned request keys: {len(hole_keys)} by_vfm={dict(hole_by_vfm)}")
    ready = (
        not reasons
        and not missing
        and not hole_keys
        and not unkeyable
        and not conflicts
        and len(owners) == len(requested)
        and processed == len(requested)
    )

    args.output_dir.mkdir(parents=True)
    route_path = args.output_dir / "routes.jsonl.gz"
    with gzip.GzipFile(filename=str(route_path), mode="wb", mtime=0) as sink:
        for key in sorted(owners, key=lambda item: json.dumps(item, default=str)):
            sink.write(
                (json.dumps(owners[key], sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
            )
    holes = [hole_entry(key, requested[key]) for key in hole_keys]
    if holes:
        hole_path = args.output_dir / "holes.jsonl.gz"
        with gzip.GzipFile(filename=str(hole_path), mode="wb", mtime=0) as sink:
            for hole in holes:
                sink.write((json.dumps(hole, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"))
        worklists = write_worklists(args.output_dir, holes)
    else:
        worklists = {}

    manifest = {
        "schema": "canonical_v2_probe_owner_overlay_v1",
        "status": "READY" if ready else "NOT_READY",
        "fail_closed": True,
        "audit_only": False if ready else True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_namespace": GAP_NAMESPACE,
        "campaign": "wan_aether_streaming_official_r1",
        "frozen_816_request_sha256": "5e0854c11f5141780ae797da3f1368cffec270a42b9ec7ab2d4cf31908597cd5",
        "note": "Wan/Aether are not part of the frozen 816 overlay.",
        "requests": {
            "path": str(args.requests.resolve()),
            "sha256": request_sha,
            "rows": len(requested),
        },
        "route_file": {
            "path": str(route_path.resolve()),
            "sha256": sha256_file(route_path),
            "rows": len(owners),
        },
        "readiness": {
            "reason": "unique complete coverage" if ready else "; ".join(reasons),
            "processed": processed,
            "requested": len(requested),
            "owners": len(owners),
            "holes": len(hole_keys),
            "holes_by_vfm": dict(hole_by_vfm),
            "pass_status": PASS_STATUS,
            "eligible_statuses": sorted(OWNER_ELIGIBLE),
            "status_counts": dict(status_counts),
        },
        "hole_worklists": worklists,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": manifest["status"],
        "requested": len(requested),
        "owners": len(owners),
        "holes": len(hole_keys),
        "holes_by_vfm": dict(hole_by_vfm),
        "status_counts": dict(status_counts),
        "reason": manifest["readiness"]["reason"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
