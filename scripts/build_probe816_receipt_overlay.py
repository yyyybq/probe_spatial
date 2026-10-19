#!/usr/bin/env python3
"""Rebind the frozen 8-model probe campaign to exact extraction receipts.

The frozen owner databases define the required composite keys.  This tool
does not discover samples or invent paths: it replaces a streaming owner only
when a PASS extraction receipt contains the exact same key and proves the raw
frame identity.  MiniMax stays on its separately validated BF16 route.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/umm/users/yinbaiqiao/probe_spatial")
DATA_ROOT = Path("/mnt/umm/users/yinbaiqiao/InsScene-15K")
PARENT_SHA = "ebc496eb0587cda372ccf29a7c5244ef454096151f73c8eed9511967ded1902d"
MODELS = {
    "cogvideox", "videomae", "dino", "f3r", "vjepa2", "qwen2_5_vl",
    "lingbot", "minimax_h3",
}
FEATURE_RE = re.compile(
    r"^(?P<scene>[^/]+)/window_(?P<window>\d+)/prefix_(?P<tail>\d+)/(?P<name>[^/]+\.sft)$"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def layer_from_name(name: str, model: str) -> int:
    match = re.search(r"(?:_layer|_l)(-?\d+)\.sft$", name)
    if match:
        return int(match.group(1))
    if model == "dino" and name == "feature.sft":
        return 0
    raise ValueError(f"cannot parse layer: {model}/{name}")


def aoss_root(receipt: dict[str, Any], model: str) -> str:
    namespace = str(receipt["lineage"]["feature_namespace"])
    return f"aoss://baiqiao/InsScene-15K/{namespace}/{model}/scannetpp"


def local_index_path(receipt: dict[str, Any], model: str, scene: str) -> Path:
    namespace = str(receipt["lineage"]["feature_namespace"])
    return DATA_ROOT / namespace / model / "scannetpp" / scene / "prefix_index.npy"


def checked_index_record(
    receipt: dict[str, Any], model: str, scene: str, window: int, prefix: int,
) -> dict[str, Any]:
    relative = f"{scene}/prefix_index.npy"
    evidence = receipt["objects"].get(relative)
    if not evidence:
        raise ValueError(f"receipt lacks {relative}")
    path = local_index_path(receipt, model, scene)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing local receipt-bound index: {path}")
    if path.stat().st_size != int(evidence["payload_size_bytes"]):
        raise ValueError(f"index size mismatch: {path}")
    if sha256_file(path) != str(evidence["payload_sha256"]):
        raise ValueError(f"index SHA mismatch: {path}")
    records = np.load(path, allow_pickle=True).tolist()
    if isinstance(records, dict):
        records = list(records.values())
    matches = [
        row for row in records
        if isinstance(row, dict)
        and int(row.get("window_id", -1)) == window
        and int(row.get("tail", -1)) == prefix - 1
    ]
    if len(matches) != 1:
        raise ValueError(
            f"index key cardinality mismatch: {model}/{scene}/w{window}/p{prefix}: {len(matches)}"
        )
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-streaming-db", type=Path, required=True)
    parser.add_argument("--target-db", type=Path, required=True)
    parser.add_argument("--receipt-snapshot", type=Path, required=True)
    parser.add_argument("--minimax-route", type=Path, required=True)
    parser.add_argument("--canonical-parent", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError(f"output exists: {output}")
    output.mkdir(parents=True)

    snapshot_manifest_path = args.receipt_snapshot / "manifest.json"
    snapshot = json.loads(snapshot_manifest_path.read_text(encoding="utf-8"))
    minimax = json.loads(args.minimax_route.read_text(encoding="utf-8"))
    if minimax.get("status") != "PASS":
        raise ValueError("MiniMax BF16 route is not PASS")
    if sha256_file(args.canonical_parent) != PARENT_SHA:
        raise ValueError("canonical parent manifest SHA mismatch")
    parent: dict[tuple[str, int], dict[str, Any]] = {}
    with args.canonical_parent.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (str(row["scene_id"]), int(row["streaming_window_id"]))
            if key in parent:
                raise ValueError(f"duplicate canonical parent key: {key}")
            parent[key] = row

    streaming_db = output / "streaming_owners.sqlite3"
    shutil.copy2(args.base_streaming_db, streaming_db)
    db = sqlite3.connect(streaming_db)
    db.row_factory = sqlite3.Row
    base_status = db.execute(
        "SELECT value FROM metadata WHERE key='status'"
    ).fetchone()
    if not base_status or base_status[0] != "READY":
        raise ValueError("base streaming DB is not READY")
    required_rows = db.execute("SELECT COUNT(*) FROM streaming_owners").fetchone()[0]
    if required_rows != 207_624:
        raise ValueError(f"unexpected streaming owner count: {required_rows}")
    specs = {
        (str(row[0]), int(row[1]), int(row[2]), int(row[3]))
        for row in db.execute(
            "SELECT DISTINCT vfm,layer,timestep,prefix FROM streaming_owners"
        )
    }
    templates = {}
    for spec in specs:
        row = db.execute(
            "SELECT tensor_shape_json,tensor_dtype FROM streaming_owners "
            "WHERE vfm=? AND layer=? AND timestep=? AND prefix=? LIMIT 1", spec,
        ).fetchone()
        templates[spec] = (str(row[0]), str(row[1]))

    receipt_owners: dict[tuple[Any, ...], dict[str, Any]] = {}
    index_cache: dict[tuple[str, str], dict[tuple[int, int], dict[str, Any]]] = {}
    index_sha_cache: dict[tuple[str, str], str] = {}
    receipt_files = []
    ignored_outside_campaign = 0
    for entry in snapshot["receipts"]:
        model = str(entry["vfm"])
        if entry["kind"] != "streaming" or model not in MODELS or model == "minimax_h3":
            continue
        receipt_path = Path(entry["snapshot_path"])
        if sha256_file(receipt_path) != entry["file_sha256"]:
            raise ValueError(f"receipt file SHA mismatch: {receipt_path}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") != "PASS_REMOTE_VERIFIED":
            raise ValueError(f"receipt is not PASS_REMOTE_VERIFIED: {receipt_path}")
        if receipt.get("lineage", {}).get("canonical_source_sha256") != PARENT_SHA:
            raise ValueError(f"receipt canonical source mismatch: {receipt_path}")
        receipt_files.append({"path": str(receipt_path), "sha256": entry["file_sha256"]})
        root = aoss_root(receipt, model)
        timestep = int(entry["timestep"])
        for relative, evidence in receipt["objects"].items():
            match = FEATURE_RE.fullmatch(relative)
            if not match:
                continue
            scene = match.group("scene")
            window = int(match.group("window"))
            prefix = int(match.group("tail")) + 1
            layer = layer_from_name(match.group("name"), model)
            key = (model, "scannetpp", scene, layer, timestep, window, prefix)
            spec = (model, layer, timestep, prefix)
            if spec not in specs:
                ignored_outside_campaign += 1
                continue
            existing = db.execute(
                "SELECT * FROM streaming_owners WHERE vfm=? AND source=? AND scene_id=? "
                "AND layer=? AND timestep=? AND window_id=? AND prefix=?", key,
            ).fetchone()
            parent_row = parent.get((scene, window))
            if parent_row is None:
                raise ValueError(f"receipt key absent from canonical parent: {key}")
            raw_ids = [int(v) for v in parent_row["streaming_window_indices"][:prefix]]
            if existing is not None and raw_ids != [int(v) for v in json.loads(existing["raw_ids_json"])]:
                raise ValueError(f"existing owner raw IDs differ from canonical parent: {key}")
            identity = evidence.get("canonical_unit_identity") or {}
            identity_sha = identity.get("gap_record_raw_ids_sha256")
            if identity_sha:
                if canonical_sha(raw_ids) != identity_sha:
                    raise ValueError(f"receipt raw-ID digest mismatch: {key}")
                input_length = int(identity["input_num_frames"])
                valid_length = len(raw_ids)
                pad_mode = str(identity["temporal_padding"])
                proof = "receipt_raw_ids_sha256"
            else:
                cache_key = (model, scene)
                if cache_key not in index_cache:
                    path = local_index_path(receipt, model, scene)
                    index_evidence = receipt["objects"].get(f"{scene}/prefix_index.npy")
                    if not index_evidence or path.is_symlink() or not path.is_file():
                        raise ValueError(f"missing receipt index: {path}")
                    index_sha_cache[cache_key] = sha256_file(path)
                    records = np.load(path, allow_pickle=True).tolist()
                    index_cache[cache_key] = {
                        (int(row["window_id"]), int(row["tail"]) + 1): row
                        for row in records
                    }
                record = index_cache[cache_key].get((window, prefix))
                if record is None or [int(v) for v in record["indices"]] != raw_ids:
                    raise ValueError(f"prefix index raw-ID mismatch: {key}")
                input_length = int(record["input_length"])
                valid_length = int(record["valid_length"])
                pad_mode = str(record["pad_mode"])
                proof = "current_prefix_index_exact_record"
            index_relative = f"{scene}/prefix_index.npy"
            index_evidence = receipt["objects"].get(index_relative)
            if not index_evidence:
                raise ValueError(f"receipt lacks index object: {key}")
            owner = {
                "source_uri": f"{root}/{relative}",
                "source_index_uri": f"{root}/{index_relative}",
                "payload_sha256": str(evidence["payload_sha256"]),
                "payload_size_bytes": int(evidence["payload_size_bytes"]),
                "payload_header_sha256": "GAP_RECEIPT_FULL_SHA",
                "source_index_sha256": index_sha_cache.get(
                    (model, scene), str(index_evidence["payload_sha256"])
                ),
                "input_length": input_length,
                "valid_length": valid_length,
                "pad_mode": pad_mode,
                "origin": "extraction_receipt_full_sha",
                "status": "PASS_REMOTE_VERIFIED",
                "identity_proof": proof,
                "receipt_sha256": entry["file_sha256"],
            }
            previous = receipt_owners.get(key)
            if previous is not None:
                route_fields = (
                    "source_uri", "source_index_uri", "payload_sha256",
                    "payload_size_bytes",
                )
                if any(previous[field] != owner[field] for field in route_fields):
                    raise ValueError(f"conflicting receipt owners: {key}")
                if (
                    previous["identity_proof"] != "receipt_raw_ids_sha256"
                    and owner["identity_proof"] == "receipt_raw_ids_sha256"
                ):
                    receipt_owners[key] = owner
                continue
            receipt_owners[key] = owner

    update_sql = """UPDATE streaming_owners SET
      source_uri=?,source_index_uri=?,payload_sha256=?,payload_size_bytes=?,
      payload_header_sha256=?,source_index_sha256=?,input_length=?,valid_length=?,
      pad_mode=?,origin=?,status=?
      WHERE vfm=? AND source=? AND scene_id=? AND layer=? AND timestep=?
        AND window_id=? AND prefix=?"""
    replacements = Counter()
    insertions = Counter()
    proofs = Counter()
    insert_sql = "INSERT INTO streaming_owners VALUES (" + ",".join("?" for _ in range(22)) + ")"
    for key, owner in sorted(receipt_owners.items()):
        values = [owner[name] for name in (
            "source_uri", "source_index_uri", "payload_sha256", "payload_size_bytes",
            "payload_header_sha256", "source_index_sha256", "input_length",
            "valid_length", "pad_mode", "origin", "status",
        )]
        cursor = db.execute(update_sql, (*values, *key))
        if cursor.rowcount == 1:
            replacements[str(key[0])] += 1
        elif cursor.rowcount == 0:
            model, source, scene, layer, timestep, window, prefix = key
            raw_ids = [int(v) for v in parent[(scene, window)]["streaming_window_indices"][:prefix]]
            shape_json, dtype = templates[(model, layer, timestep, prefix)]
            unit_id = canonical_sha([
                model, source, scene, layer, timestep, window, prefix, raw_ids,
            ])
            db.execute(insert_sql, (
                *key, unit_id, owner["source_uri"], owner["source_index_uri"],
                owner["payload_sha256"], owner["payload_size_bytes"],
                owner["payload_header_sha256"], owner["source_index_sha256"],
                shape_json, dtype, json.dumps(raw_ids, separators=(",", ":")),
                owner["input_length"], owner["valid_length"], owner["pad_mode"],
                owner["origin"], owner["status"],
            ))
            insertions[str(model)] += 1
        else:
            raise ValueError(f"receipt update cardinality mismatch: {key}/{cursor.rowcount}")
        proofs[owner["identity_proof"]] += 1

    db.execute("DELETE FROM metadata WHERE key IN ('receipt_snapshot_sha256','receipt_replacements')")
    db.executemany("INSERT INTO metadata(key,value) VALUES (?,?)", [
        ("receipt_snapshot_sha256", sha256_file(snapshot_manifest_path)),
        ("receipt_replacements", str(len(receipt_owners))),
    ])
    db.commit()
    duplicate_keys = db.execute(
        "SELECT COUNT(*) FROM (SELECT vfm,source,scene_id,layer,timestep,window_id,prefix,COUNT(*) n "
        "FROM streaming_owners GROUP BY 1,2,3,4,5,6,7 HAVING n != 1)"
    ).fetchone()[0]
    final_rows = db.execute("SELECT COUNT(*) FROM streaming_owners").fetchone()[0]
    by_model = dict(db.execute(
        "SELECT vfm,COUNT(*) FROM streaming_owners GROUP BY vfm ORDER BY vfm"
    ).fetchall())
    retained_origins = dict(db.execute(
        "SELECT origin,COUNT(*) FROM streaming_owners GROUP BY origin ORDER BY origin"
    ).fetchall())
    db.close()
    expected_keys = set()
    for model, layer, timestep, prefix in specs:
        if model == "minimax_h3":
            expected_keys.update(db_key for db_key in [])
            continue
        expected_keys.update(
            (model, "scannetpp", scene, layer, timestep, window, prefix)
            for scene, window in parent
        )
    check = sqlite3.connect(streaming_db)
    observed_non_minimax = set(check.execute(
        "SELECT vfm,source,scene_id,layer,timestep,window_id,prefix "
        "FROM streaming_owners WHERE vfm != 'minimax_h3'"
    ))
    minimax_rows = check.execute(
        "SELECT COUNT(*) FROM streaming_owners WHERE vfm='minimax_h3'"
    ).fetchone()[0]
    check.close()
    missing_keys = expected_keys - observed_non_minimax
    extra_keys = observed_non_minimax - expected_keys
    if duplicate_keys or missing_keys or extra_keys or minimax_rows != 33_880:
        raise ValueError(
            f"streaming closure failed: duplicates={duplicate_keys}, "
            f"missing={len(missing_keys)}, extra={len(extra_keys)}, minimax={minimax_rows}"
        )

    target = sqlite3.connect(f"file:{args.target_db.resolve()}?mode=ro&immutable=1", uri=True)
    target_rows = target.execute("SELECT COUNT(*) FROM target_owners").fetchone()[0]
    target_duplicates = target.execute(
        "SELECT COUNT(*) FROM (SELECT vfm,source,scene_id,layer,timestep,target_raw_id,COUNT(*) n "
        "FROM target_owners GROUP BY 1,2,3,4,5,6 HAVING n != 1)"
    ).fetchone()[0]
    target_holes = target.execute(
        "SELECT COUNT(*) FROM target_expected e LEFT JOIN target_owners o USING "
        "(vfm,source,scene_id,layer,timestep,canonical_row_index) WHERE o.source_uri IS NULL"
    ).fetchone()[0]
    target.close()
    if target_rows != 715_863 or target_duplicates or target_holes:
        raise ValueError("target owner database is not complete and unique")

    audit = {
        "schema": "PROBE816_RECEIPT_REBIND_AUDIT_V1",
        "status": "READY",
        "models": sorted(MODELS),
        "zero_holes": not missing_keys and target_holes == 0,
        "zero_duplicate_effective_owners": duplicate_keys == 0 and target_duplicates == 0,
        "streaming": {
            "required_rows": required_rows,
            "owner_rows": final_rows,
            "duplicate_keys": duplicate_keys,
            "receipt_replacements": len(receipt_owners),
            "replacements_by_model": dict(sorted(replacements.items())),
            "insertions_by_model": dict(sorted(insertions.items())),
            "newly_added_keys": sum(insertions.values()),
            "expected_non_minimax_keys": len(expected_keys),
            "missing_expected_keys": len(missing_keys),
            "extra_keys": len(extra_keys),
            "identity_proofs": dict(sorted(proofs.items())),
            "owners_by_model": by_model,
            "origins": retained_origins,
            "ignored_receipt_objects_outside_campaign": ignored_outside_campaign,
            "database": {"path": str(streaming_db), "sha256": sha256_file(streaming_db)},
        },
        "target": {
            "owner_rows": target_rows,
            "holes": target_holes,
            "duplicate_keys": target_duplicates,
            "database": {"path": str(args.target_db.resolve()), "sha256": sha256_file(args.target_db)},
        },
        "receipt_snapshot": {
            "path": str(snapshot_manifest_path.resolve()),
            "sha256": sha256_file(snapshot_manifest_path),
            "receipts_used": receipt_files,
        },
        "minimax_bf16_route": {
            "path": str(args.minimax_route.resolve()),
            "sha256": sha256_file(args.minimax_route),
            "identity_sha256": minimax["identity_sha256"],
        },
    }
    audit_path = output / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
