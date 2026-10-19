"""Build an executable 816 owner overlay from verified r2 gate receipts.

PASS gate receipts (REFERENCE_SOURCE_READ_ONLY) become owners first.
Remaining request holes are filled from gap-delta streaming objects
(PASS_REMOTE_VERIFIED). FAIL URIs, dummy/legacy fallbacks, and extra gap
objects that are not 816 request keys never become owners.

READY requires every keyed gate request to have exactly one eligible owner.
Incomplete shards, request SHA mismatch, unkeyable receipts, owner conflicts,
or leftover holes keep the catalog NOT_READY. Unreadable gap receipts that
are not 816 dependencies do not block. This script never copies payloads.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from scripts.owner_catalog import (
    OWNER_ELIGIBLE,
    disk_layer,
    sha256_file,
    streaming_key,
)

REQUEST_SHA = "5e0854c11f5141780ae797da3f1368cffec270a42b9ec7ab2d4cf31908597cd5"
SHARD_DIR = re.compile(r"^gate-(\d+)-of-(\d+)$")
VFM_RE = re.compile(
    r"^(?:cv2_)?(cogvideox|lingbot|minimax_h3|videomae|vjepa2|dino|f3r|qwen2_5_vl)"
)
# Gate unit_ids encode t749 for VFMs whose on-disk filenames have no timestep.
GATE_UNIT_NO_TIMESTEP = 749
AOSS_INSCENE = "aoss://baiqiao/InsScene-15K/"
PASS_STATUS = "REFERENCE_SOURCE_READ_ONLY"
GAP_STATUS = "PASS_REMOTE_VERIFIED"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with open(fd, "w", encoding="utf-8", closefd=True) as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
        Path(tmp).replace(path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def to_aoss_uri(root: str, relative_path: str) -> str:
    rel = str(relative_path).lstrip("/")
    base = str(root).rstrip("/")
    if not rel:
        raise ValueError("gap object is missing relative_path")
    if base.startswith("aoss://"):
        return f"{base}/{rel}"
    marker = "/InsScene-15K/"
    if marker not in base:
        raise ValueError(f"cannot map gap root to aoss://: {root}")
    return AOSS_INSCENE + base.split(marker, 1)[1].rstrip("/") + "/" + rel


def parse_layer_from_unit(unit: str) -> int | None:
    match = re.search(r"_lm(\d+)_", unit)
    if match:
        return -int(match.group(1))
    match = re.search(r"_l(-?\d+)_", unit)
    if match:
        return int(match.group(1))
    return None


def parse_timestep_from_name(name: str) -> int | None:
    match = re.search(r"_t(\d+)_", name)
    if match:
        return int(match.group(1))
    match = re.search(r"feature_t(\d+)", name)
    if match:
        return int(match.group(1))
    return None


def receipt_vfm(row: dict[str, Any]) -> str:
    expected = row.get("expected") or {}
    value = str(expected.get("vfm") or "").strip()
    if value:
        return value
    match = VFM_RE.search(str(row.get("unit_id") or ""))
    return match.group(1) if match else ""


def receipt_key(row: dict[str, Any]) -> tuple[Any, ...] | None:
    expected = row.get("expected") or {}
    unit = str(row.get("unit_id") or "")
    vfm = receipt_vfm(row)
    if not vfm:
        return None
    layer = expected.get("feature_layer", expected.get("layer"))
    source = expected.get("source") or "scannetpp"
    scene_id = expected.get("scene_id")
    window_id = expected.get("window_id")
    prefix = expected.get("prefix", expected.get("prefix_len"))
    if scene_id is None or window_id is None or prefix is None:
        # unit_id: cv2_cogvideox_t749_l0_p12_...:scene_w002_p12
        if ":" not in unit:
            return None
        _, rest = unit.split(":", 1)
        scene_id = rest.split("_w")[0]
        try:
            window_id = int(rest.split("_w")[1].split("_")[0])
            prefix = int(rest.rsplit("_p", 1)[1])
        except (IndexError, ValueError):
            return None
    try:
        layer = int(layer)
    except (TypeError, ValueError):
        parsed = parse_layer_from_unit(unit)
        if parsed is None:
            return None
        layer = parsed
    timestep = expected.get("timestep")
    if timestep is None:
        timestep = parse_timestep_from_name(unit)
    return streaming_key(
        vfm=vfm,
        source=str(source),
        scene_id=str(scene_id),
        feature_layer=int(layer),
        timestep=None if timestep is None else int(timestep),
        window_id=int(window_id),
        prefix=int(prefix),
    )


def gap_streaming_key(row: dict[str, Any]) -> tuple[Any, ...] | None:
    if row.get("mode") != "streaming_prefix":
        return None
    ident = row.get("identity") or {}
    if ident.get("kind") not in (None, "streaming_feature"):
        return None
    vfm = str(row.get("vfm") or ident.get("vfm") or "").strip()
    record = ident.get("gap_record_key") or {}
    scene_id = record.get("scene")
    window_id = record.get("window_id")
    prefix = record.get("prefix")
    layer = ident.get("layer")
    if not vfm or scene_id is None or window_id is None or prefix is None or layer is None:
        return None
    rel = str(row.get("relative_path") or "")
    timestep = ident.get("timestep")
    if timestep is None:
        timestep = parse_timestep_from_name(rel)
    if timestep is None:
        timestep = GATE_UNIT_NO_TIMESTEP
    source = "scannetpp"
    root = str(row.get("root") or "")
    if "/scannetpp" in root:
        source = "scannetpp"
    return streaming_key(
        vfm=vfm,
        source=source,
        scene_id=str(scene_id),
        feature_layer=int(layer),
        timestep=int(timestep),
        window_id=int(window_id),
        prefix=int(prefix),
    )


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                yield row


def _put_owner(
    owners: dict[tuple[Any, ...], dict[str, Any]],
    key: tuple[Any, ...],
    owner: dict[str, Any],
    conflicts: list[str],
) -> bool:
    existing = owners.get(key)
    if existing is None:
        owners[key] = owner
        return True
    if existing.get("source_uri") != owner.get("source_uri"):
        conflicts.append(str(key))
    return False


def _vfm_from_key(key: tuple[Any, ...]) -> str:
    return str(key[1]) if len(key) > 1 else "UNK"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-root", type=Path, required=True)
    parser.add_argument("--gap-index", type=Path, required=True)
    parser.add_argument("--recovery-receipts", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-request-sha256", default=REQUEST_SHA)
    parser.add_argument("--expected-shards", type=int, default=128)
    parser.add_argument("--expected-requests", type=int, default=207624)
    args = parser.parse_args()

    shard_dirs = sorted(
        path for path in args.gate_root.iterdir()
        if path.is_dir() and SHARD_DIR.match(path.name)
    )
    owners: dict[tuple[Any, ...], dict[str, Any]] = {}
    requested: dict[tuple[Any, ...], str] = {}
    conflicts: list[str] = []
    unkeyable: list[str] = []
    status_counts: dict[str, int] = {}
    processed = 0
    reasons: list[str] = []
    shard_summaries = []
    seen = set()

    for shard_dir in shard_dirs:
        match = SHARD_DIR.match(shard_dir.name)
        assert match is not None
        index = int(match.group(1))
        summary_path = shard_dir / "summary.json"
        receipts_path = shard_dir / "receipts.jsonl.gz"
        if summary_path.is_symlink() or receipts_path.is_symlink():
            reasons.append(f"symlink shard output: {shard_dir}")
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            reasons.append(f"unreadable summary {summary_path}: {exc}")
            continue
        if summary.get("requests_sha256") != args.expected_request_sha256:
            reasons.append(
                f"shard {index} request SHA {summary.get('requests_sha256')} "
                f"!= {args.expected_request_sha256}"
            )
            continue
        if int(summary.get("shard_count", -1)) != args.expected_shards:
            reasons.append(f"shard {index} shard_count mismatch")
            continue
        shard_summaries.append({
            "shard_index": index,
            "processed": int(summary.get("processed", 0)),
            "status_counts": summary.get("status_counts"),
            "receipts_sha256": summary.get("receipts_sha256"),
        })
        seen.add(index)
        processed += int(summary.get("processed", 0))
        try:
            for row in _jsonl(receipts_path):
                status = str(row.get("status") or "")
                status_counts[status] = status_counts.get(status, 0) + 1
                unit_id = str(row.get("unit_id") or "")
                key = receipt_key(row)
                if key is None:
                    unkeyable.append(unit_id or f"shard-{index}-unkeyed")
                    continue
                if key in requested and requested[key] != unit_id:
                    conflicts.append(str(key))
                requested[key] = unit_id
                if status not in OWNER_ELIGIBLE:
                    continue
                uri = row.get("source_uri")
                if not uri:
                    reasons.append(f"pass receipt missing source_uri: {unit_id}")
                    continue
                _put_owner(owners, key, {
                    "key": list(key),
                    "status": status,
                    "source_uri": uri,
                    "unit_id": unit_id,
                    "origin": "gate_pass",
                    "reason_codes": row.get("reason_codes"),
                }, conflicts)
        except OSError as exc:
            reasons.append(f"unreadable receipts {receipts_path}: {exc}")
        if (index + 1) % 16 == 0:
            print(
                f"[overlay] shards {index + 1}/{args.expected_shards} "
                f"requested={len(requested)} pass_owners={len(owners)}",
                file=sys.stderr,
            )

    missing = [index for index in range(args.expected_shards) if index not in seen]
    if missing:
        reasons.append(f"missing shards: {missing[:12]}{'...' if len(missing) > 12 else ''}")
    if unkeyable:
        reasons.append(f"unkeyable receipts: {len(unkeyable)}")
    if processed != args.expected_requests and not missing:
        reasons.append(f"processed {processed} != expected {args.expected_requests}")
    if len(requested) != args.expected_requests and not unkeyable and not missing:
        reasons.append(
            f"unique keyed receipts {len(requested)} != expected {args.expected_requests}"
        )

    gap_stats = {
        "objects": 0,
        "streaming": 0,
        "filled": 0,
        "unused": 0,
        "unkeyable": 0,
        "missing_uri": 0,
        "conflicts": 0,
        "sha256": sha256_file(args.gap_index),
    }
    holes_before_gap = sum(1 for key in requested if key not in owners)
    try:
        for row in _jsonl(args.gap_index):
            gap_stats["objects"] += 1
            key = gap_streaming_key(row)
            if key is None:
                if row.get("mode") == "streaming_prefix":
                    gap_stats["unkeyable"] += 1
                continue
            gap_stats["streaming"] += 1
            if key not in requested:
                gap_stats["unused"] += 1
                continue
            if key in owners:
                continue
            try:
                uri = to_aoss_uri(str(row.get("root") or ""), str(row.get("relative_path") or ""))
            except ValueError:
                gap_stats["missing_uri"] += 1
                continue
            if not row.get("payload_sha256"):
                gap_stats["missing_uri"] += 1
                continue
            added = _put_owner(owners, key, {
                "key": list(key),
                "status": GAP_STATUS,
                "source_uri": uri,
                "unit_id": requested[key],
                "origin": "gap_delta",
                "identity_sha256": row.get("identity_sha256"),
                "payload_sha256": row.get("payload_sha256"),
            }, conflicts)
            if added:
                gap_stats["filled"] += 1
            else:
                gap_stats["conflicts"] += 1
    except OSError as exc:
        reasons.append(f"unreadable gap index {args.gap_index}: {exc}")

    recovery_stats = {"rows": 0, "filled": 0, "unused": 0, "rejected": 0}
    if args.recovery_receipts is not None:
        try:
            for row in _jsonl(args.recovery_receipts):
                recovery_stats["rows"] += 1
                if row.get("status") != "PASS_METADATA_RECOVERED":
                    recovery_stats["rejected"] += 1
                    continue
                key = tuple(row.get("key") or ())
                uri = row.get("source_uri")
                if key not in requested or not uri:
                    recovery_stats["unused"] += 1
                    continue
                if key in owners:
                    continue
                added = _put_owner(owners, key, {
                    "key": list(key),
                    "status": "PASS_METADATA_RECOVERED",
                    "source_uri": uri,
                    "unit_id": requested[key],
                    "origin": "metadata_recovery",
                    "recovery_receipt_sha256": row.get("receipt_sha256"),
                    "recovery_basis": row.get("recovery_basis"),
                }, conflicts)
                if added:
                    recovery_stats["filled"] += 1
                else:
                    recovery_stats["rejected"] += 1
        except OSError as exc:
            reasons.append(f"unreadable recovery receipts {args.recovery_receipts}: {exc}")

    hole_keys = [key for key in requested if key not in owners]
    hole_by_vfm = Counter(_vfm_from_key(key) for key in hole_keys)
    origin_counts = Counter(str(row.get("origin") or "") for row in owners.values())
    if conflicts:
        reasons.append(f"owner key conflicts: {len(conflicts)}")
    if hole_keys:
        reasons.append(
            f"unowned request keys: {len(hole_keys)} by_vfm={dict(hole_by_vfm)}"
        )
    if len(owners) != args.expected_requests and not hole_keys:
        reasons.append(
            f"eligible owners {len(owners)} != expected {args.expected_requests}"
        )

    ready = (
        not reasons
        and not missing
        and not hole_keys
        and not unkeyable
        and not conflicts
        and len(owners) == args.expected_requests
        and processed == args.expected_requests
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    route_path = args.output_dir / "routes.jsonl.gz"
    with gzip.open(route_path, "wt", encoding="utf-8", compresslevel=6) as handle:
        for key in sorted(owners, key=lambda item: json.dumps(item, default=str)):
            handle.write(json.dumps(owners[key], sort_keys=True, ensure_ascii=False) + "\n")
    if hole_keys:
        hole_path = args.output_dir / "holes.jsonl.gz"
        with gzip.open(hole_path, "wt", encoding="utf-8", compresslevel=6) as handle:
            for key in hole_keys:
                handle.write(json.dumps({
                    "key": list(key),
                    "unit_id": requested[key],
                }, sort_keys=True, ensure_ascii=False) + "\n")
    manifest = {
        "schema": "canonical_v2_probe_owner_overlay_v1",
        "status": "READY" if ready else "NOT_READY",
        "fail_closed": True,
        "audit_only": not ready,
        "payload_copy_performed": False,
        "payload_full_read_performed": False,
        "request_sha256": args.expected_request_sha256,
        "logical_to_disk_layer_overrides": {
            "lingbot:48": disk_layer("lingbot", 48),
            "minimax_h3:50": disk_layer("minimax_h3", 50),
        },
        "route_file": {
            "path": str(route_path.resolve()),
            "sha256": sha256_file(route_path),
            "rows": len(owners),
        },
        "counts": {
            "shards_present": len(seen),
            "shards_expected": args.expected_shards,
            "processed": processed,
            "requested_keys": len(requested),
            "eligible_owners": len(owners),
            "pass_owners": int(origin_counts.get("gate_pass", 0)),
            "gap_filled_owners": int(origin_counts.get("gap_delta", 0)),
            "holes_before_gap": holes_before_gap,
            "holes": len(hole_keys),
            "holes_by_vfm": dict(hole_by_vfm),
            "unkeyable_receipts": len(unkeyable),
            "status": status_counts,
            "origin": dict(origin_counts),
            "gap_index": gap_stats,
            "metadata_recovery": recovery_stats,
        },
        "shard_summaries": shard_summaries,
        "readiness": {
            "complete_unit_coverage_verified": ready,
            "reason": "ok" if ready else "; ".join(reasons) or "incomplete",
        },
    }
    atomic_json(args.output_dir / "catalog_manifest.json", manifest)
    printable = {
        key: manifest[key]
        for key in (
            "schema", "status", "fail_closed", "audit_only",
            "request_sha256", "route_file", "counts", "readiness",
        )
    }
    print(json.dumps(printable, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
