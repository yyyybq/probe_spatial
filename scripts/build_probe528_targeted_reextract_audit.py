#!/usr/bin/env python3
"""Build the exact feature repair scope for the frozen seven-model 528 queue.

This is a local metadata audit.  It never reads model payloads, submits jobs, or
changes the frozen owner databases.  Streaming rows that still point at the two
explicitly retired V1 namespaces become repair items.  Target requirements are
derived from the actual C1/C2 h1/h2/h4 access pattern and compared with the
target owner database.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


OLD_STREAMING_NAMESPACES = {"FEAT_STREAMING", "FEAT_STREAMING_BY_PREFIX"}
OLD_TARGET_NAMESPACES = {"FEAT_TARGET", "FEAT_TARGET_BY_PREFIX"}
HORIZONS = (1, 2, 4)


def namespace(uri: str) -> str:
    marker = "InsScene-15K/"
    return uri.split(marker, 1)[1].split("/", 1)[0] if marker in uri else "LOCAL_OR_UNKNOWN"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", type=Path, required=True)
    ap.add_argument("--streaming-db", type=Path, required=True)
    ap.add_argument("--target-db", type=Path, required=True)
    ap.add_argument("--receipt-snapshot", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)

    queue = [json.loads(line) for line in args.queue.read_text().splitlines() if line.strip()]
    if len(queue) != 528 or len({r["unique_key"] for r in queue}) != 528:
        raise RuntimeError("queue is not the frozen 528-item scope")
    specs = sorted({
        (r["model"], int(r["disk_layer"]), int(r["timestep"]), int(r["prefix"]))
        for r in queue
    })

    stream = sqlite3.connect(f"file:{args.streaming_db.resolve()}?mode=ro", uri=True)
    stream.row_factory = sqlite3.Row
    stream_detail: list[dict] = []
    stream_summary: dict[tuple, Counter] = defaultdict(Counter)
    grouped_stream: dict[tuple, set[int]] = defaultdict(set)
    for model, layer, timestep, prefix in specs:
        rows = list(stream.execute(
            "SELECT vfm,source,scene_id,layer,timestep,window_id,prefix,raw_ids_json,"
            "source_uri,origin,status FROM streaming_owners WHERE vfm=? AND layer=? "
            "AND timestep=? AND prefix=? ORDER BY scene_id,window_id",
            (model, layer, timestep, prefix),
        ))
        if len(rows) != 2721:
            raise RuntimeError(f"unexpected streaming rows for {(model, layer, timestep, prefix)}: {len(rows)}")
        for row in rows:
            ns = namespace(row["source_uri"])
            needs_repair = ns in OLD_STREAMING_NAMESPACES
            stream_summary[(model, layer, timestep, prefix)][
                "repair" if needs_repair else "retained_v2"
            ] += 1
            if not needs_repair:
                continue
            item = {
                "model": model,
                "layer": layer,
                "timestep": timestep,
                "source": row["source"],
                "scene_id": row["scene_id"],
                "window_id": int(row["window_id"]),
                "prefix": prefix,
                "raw_frame_ids": json.loads(row["raw_ids_json"]),
                "retired_source_uri": row["source_uri"],
                "retired_namespace": ns,
                "reason": "SOURCE_NAMESPACE_RETIRED_AFTER_V2_PARTIAL_REUSE",
            }
            stream_detail.append(item)
            grouped_stream[(model, timestep, row["scene_id"], int(row["window_id"]), prefix)].add(layer)
    stream.close()

    # C1 and C2 have the same union of target identities because C2 consumes
    # all h1/h2/h4 targets.  Derive those identities from each frozen prefix
    # manifest rather than from a global feature inventory.
    manifests_by_prefix: dict[int, Path] = {}
    for row in queue:
        if row["probe"] in {"action_dynamics", "path_integration"}:
            p = Path(row["execution_manifest"]["path"])
            if sha256(p) != row["execution_manifest"]["sha256"]:
                raise RuntimeError(f"execution manifest SHA mismatch: {p}")
            previous = manifests_by_prefix.setdefault(int(row["prefix"]), p)
            if previous != p:
                raise RuntimeError(f"multiple manifests for prefix {row['prefix']}")
    manifest_rows = {
        prefix: [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for prefix, path in manifests_by_prefix.items()
    }
    if any(len(rows) != 2721 for rows in manifest_rows.values()):
        raise RuntimeError("execution manifest row count mismatch")

    target_required: dict[tuple, dict] = {}
    target_refs: dict[tuple, set[tuple]] = defaultdict(set)
    for model, layer, timestep, prefix in specs:
        for sample in manifest_rows[prefix]:
            window = [int(v) for v in sample["streaming_window_indices"]]
            tail = int(sample["streaming_tail"])
            for horizon in HORIZONS:
                raw_id = int(window[tail + horizon])
                key = (model, layer, timestep, str(sample["scene_id"]), raw_id)
                target_required.setdefault(key, {
                    "model": model, "layer": layer, "timestep": timestep,
                    "source": "scannetpp", "scene_id": str(sample["scene_id"]),
                    "target_raw_id": raw_id,
                })
                target_refs[key].add((prefix, int(sample["streaming_window_id"]), horizon))

    target = sqlite3.connect(f"file:{args.target_db.resolve()}?mode=ro", uri=True)
    target.row_factory = sqlite3.Row
    owner_index: dict[tuple, list[sqlite3.Row]] = defaultdict(list)
    selected = {(m, l, t) for m, l, t, _ in specs}
    for model, layer, timestep in selected:
        for row in target.execute(
            "SELECT * FROM target_owners WHERE vfm=? AND layer=? AND timestep=?",
            (model, layer, timestep),
        ):
            owner_index[(model, layer, timestep, row["scene_id"], int(row["target_raw_id"]))].append(row)
    target.close()

    # The owner DB predates some successful target extraction receipts.  Treat
    # those receipts as the later ground truth, keyed by exact model/layer/
    # scene/raw-frame identity.  This avoids re-extracting files that exist in
    # canonical-v2 but were never merged into the older owner DB.
    receipt_manifest_path = args.receipt_snapshot / "manifest.json"
    receipt_manifest = json.loads(receipt_manifest_path.read_text())
    receipt_target_index: dict[tuple, dict] = {}
    for binding in receipt_manifest["receipts"]:
        if binding.get("kind") != "target":
            continue
        path = Path(binding["snapshot_path"])
        if sha256(path) != binding["file_sha256"]:
            raise RuntimeError(f"target receipt SHA mismatch: {path}")
        receipt = json.loads(path.read_text())
        if receipt.get("status") != "PASS_REMOTE_VERIFIED":
            continue
        lineage = receipt["lineage"]
        model = str(lineage["vfm"]); timestep = int(lineage["timestep"])
        base_uri = str(receipt["remote"]["base_uri"]).replace(
            "s3://baiqiao.aoss-internal.cn-fz-01.fjscmsapi-oss.com/", "aoss://baiqiao/"
        ).rstrip("/")
        for relative, evidence in receipt["objects"].items():
            if not relative.endswith(".sft"):
                continue
            identity = evidence.get("canonical_unit_identity") or {}
            if identity.get("kind") != "target_delta_layer":
                continue
            layer = int(identity["layer"]); scene = str(identity["scene"])
            if (model, layer, timestep) not in selected:
                continue
            for row_index, raw_id in enumerate(identity["target_indices"]):
                key = (model, layer, timestep, scene, int(raw_id))
                owner = {
                    "source_uri": f"{base_uri}/{relative}",
                    "source_row_index": row_index,
                    "receipt_path": str(path.resolve()),
                    "receipt_sha256": binding["file_sha256"],
                    "payload_sha256": evidence["payload_sha256"],
                }
                previous = receipt_target_index.get(key)
                if previous is not None:
                    if previous["payload_sha256"] != owner["payload_sha256"]:
                        raise RuntimeError(f"conflicting target receipt payloads for {key}")
                    # Retry/shard receipts can prove the same bytes at multiple
                    # immutable paths.  Pick one deterministically.
                    owner = min((previous, owner), key=lambda value: value["source_uri"])
                receipt_target_index[key] = owner

    target_detail: list[dict] = []
    target_summary: dict[tuple, Counter] = defaultdict(Counter)
    for key, item in sorted(target_required.items()):
        owners = owner_index.get(key, [])
        receipt_owner = receipt_target_index.get(key)
        if receipt_owner is not None:
            category = "retained_v2_receipt"
            owner_uri = receipt_owner["source_uri"]
            owner_origin = "extraction_receipt_full_sha"
            owner_status = "PASS_REMOTE_VERIFIED"
        elif not owners:
            category = "repair_no_owner"
            owner_uri = None
            owner_origin = None
            owner_status = None
        else:
            uris = {str(row["source_uri"]) for row in owners}
            if len(uris) != 1:
                raise RuntimeError(f"conflicting target owners for {key}: {sorted(uris)}")
            owner = owners[0]
            owner_uri = str(owner["source_uri"])
            owner_origin = str(owner["origin"])
            owner_status = str(owner["status"])
            category = "owner_in_old_namespace" if namespace(owner_uri) in OLD_TARGET_NAMESPACES else "retained_v2"
        target_summary[(item["model"], item["layer"], item["timestep"])][category] += 1
        if category in {"retained_v2", "retained_v2_receipt"}:
            continue
        target_detail.append({
            **item,
            "required_by": [
                {"prefix": p, "window_id": w, "horizon": h}
                for p, w, h in sorted(target_refs[key])
            ],
            "classification": category,
            "current_owner_uri": owner_uri,
            "current_owner_origin": owner_origin,
            "current_owner_status": owner_status,
            "reason": (
                "NO_TARGET_OWNER_FOR_REQUIRED_C1_C2_RAW_FRAME"
                if category == "repair_no_owner"
                else "TARGET_OWNER_REMAINS_IN_V1_NAMESPACE_VERIFY_BEFORE_REPAIR"
            ),
        })

    stream_group_rows = [
        {
            "model": model, "timestep": timestep, "scene_id": scene,
            "window_id": window, "prefix": prefix, "layers": sorted(layers),
            "reason": "SOURCE_NAMESPACE_RETIRED_AFTER_V2_PARTIAL_REUSE",
        }
        for (model, timestep, scene, window, prefix), layers in sorted(grouped_stream.items())
    ]
    write_jsonl(args.output / "streaming_repair_units.jsonl", stream_detail)
    write_jsonl(args.output / "streaming_extractor_entries.jsonl", stream_group_rows)
    write_jsonl(args.output / "target_review_and_repair_units.jsonl", target_detail)

    summary_rows = []
    for key, counts in sorted(stream_summary.items()):
        summary_rows.append({
            "mode": "streaming", "model": key[0], "layer": key[1],
            "timestep": key[2], "prefix": key[3], "required": sum(counts.values()),
            "repair": counts["repair"], "retained_v2": counts["retained_v2"],
            "old_owner_verify": 0,
        })
    for key, counts in sorted(target_summary.items()):
        summary_rows.append({
            "mode": "target", "model": key[0], "layer": key[1],
            "timestep": key[2], "prefix": "h1,h2,h4 union",
            "required": sum(counts.values()), "repair": counts["repair_no_owner"],
            "retained_v2": counts["retained_v2"] + counts["retained_v2_receipt"],
            "old_owner_verify": counts["owner_in_old_namespace"],
        })
    with (args.output / "summary_by_spec.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader(); writer.writerows(summary_rows)

    totals = Counter()
    for row in summary_rows:
        totals[f"{row['mode']}_required"] += int(row["required"])
        totals[f"{row['mode']}_repair"] += int(row["repair"])
        totals[f"{row['mode']}_retained_v2"] += int(row["retained_v2"])
        totals[f"{row['mode']}_old_owner_verify"] += int(row["old_owner_verify"])
    manifest = {
        "schema": "PROBE528_TARGETED_REEXTRACT_AUDIT_V1",
        "status": "AUDIT_COMPLETE_TARGET_OLD_NAMESPACE_REQUIRES_REMOTE_CONFIRMATION",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "queue": {"path": str(args.queue.resolve()), "sha256": sha256(args.queue), "experiments": 528},
        "streaming_database": {"path": str(args.streaming_db.resolve()), "sha256": sha256(args.streaming_db)},
        "target_database": {"path": str(args.target_db.resolve()), "sha256": sha256(args.target_db)},
        "receipt_snapshot": {"path": str(receipt_manifest_path.resolve()), "sha256": sha256(receipt_manifest_path)},
        "policy": {
            "streaming_repair": sorted(OLD_STREAMING_NAMESPACES),
            "target_repair_now": "only exact required keys with no owner",
            "target_verify_before_repair": sorted(OLD_TARGET_NAMESPACES),
            "horizons": list(HORIZONS),
        },
        "totals": dict(totals),
        "files": {},
    }
    for name in ["streaming_repair_units.jsonl", "streaming_extractor_entries.jsonl", "target_review_and_repair_units.jsonl", "summary_by_spec.csv"]:
        p = args.output / name
        manifest["files"][name] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "totals": dict(totals)}, sort_keys=True))


if __name__ == "__main__":
    main()
