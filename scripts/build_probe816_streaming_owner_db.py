#!/usr/bin/env python3
"""Build the exact, fail-closed streaming owner DB for probe816 contract v2."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


# A reference-only route establishes a logical key, not a training-readable
# payload.  It must never win ownership over a full-SHA re-extraction and it
# cannot by itself make the database READY.
ELIGIBLE_GATE = "REFERENCE_SOURCE_READ_ONLY"
SCENE_RE = re.compile(r"/([^/]+)/window_\d+/prefix_\d+/")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def request_key(row: dict[str, Any]) -> tuple[Any, ...]:
    expected = row["expected"]
    uri = str(row["source_uri"])
    match = SCENE_RE.search(uri)
    if not match:
        raise ValueError(f"cannot parse scene from request URI: {uri}")
    return (
        str(expected["vfm"]), "scannetpp", match.group(1), int(expected["feature_layer"]),
        int(expected["timestep"]), int(row["source_index_record"]["window_id"]),
        len(expected["raw_ids"]),
    )


SCHEMA = """
PRAGMA journal_mode=DELETE;
PRAGMA synchronous=FULL;
CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
CREATE TABLE streaming_owners(
  vfm TEXT NOT NULL, source TEXT NOT NULL, scene_id TEXT NOT NULL,
  layer INTEGER NOT NULL, timestep INTEGER NOT NULL, window_id INTEGER NOT NULL,
  prefix INTEGER NOT NULL, unit_id TEXT NOT NULL UNIQUE,
  source_uri TEXT NOT NULL, source_index_uri TEXT NOT NULL,
  payload_sha256 TEXT, payload_size_bytes INTEGER NOT NULL,
  payload_header_sha256 TEXT NOT NULL, source_index_sha256 TEXT NOT NULL,
  tensor_shape_json TEXT NOT NULL, tensor_dtype TEXT NOT NULL,
  raw_ids_json TEXT NOT NULL, input_length INTEGER NOT NULL,
  valid_length INTEGER NOT NULL, pad_mode TEXT NOT NULL,
  origin TEXT NOT NULL, status TEXT NOT NULL,
  PRIMARY KEY(vfm,source,scene_id,layer,timestep,window_id,prefix)
) WITHOUT ROWID;
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=Path, required=True)
    ap.add_argument("--gate-root", type=Path, required=True)
    ap.add_argument("--metadata-recovery", type=Path, required=True)
    ap.add_argument("--minimax-correction", type=Path, required=True)
    ap.add_argument("--reextract-receipts", type=Path, action="append", default=[])
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.output_dir.exists():
        raise ValueError(f"output exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    requests = {}
    for row in jsonl(args.requests):
        unit = str(row["unit_id"])
        if unit in requests:
            raise ValueError(f"duplicate request: {unit}")
        requests[unit] = row
    if len(requests) != 207_624:
        raise ValueError(f"request count mismatch: {len(requests)}")

    gate = {}
    for path in sorted(args.gate_root.glob("gate-*/receipts.jsonl.gz")):
        for row in jsonl(path):
            unit = str(row["unit_id"])
            if unit in gate:
                raise ValueError(f"duplicate gate receipt: {unit}")
            gate[unit] = row
    missing_gate = set(requests) - set(gate)
    extra_gate = set(gate) - set(requests)
    if missing_gate:
        raise ValueError(f"gate lacks {len(missing_gate)} requested units")
    # The frozen gate run also covered 4,474 Aether units.  Aether is outside
    # this 8-model/816-row campaign; keep the immutable receipt tree intact and
    # select the exact requested subset here.
    gate = {unit: gate[unit] for unit in requests}

    metadata_units = {str(row["unit_id"]): row for row in jsonl(args.metadata_recovery)}
    minimax_units = {str(row["unit_id"]): row for row in jsonl(args.minimax_correction)}
    reextract_units = {}
    reextract_bindings = []
    for path in args.reextract_receipts:
        for row in jsonl(path):
            unit = str(row["unit_id"])
            if unit in reextract_units:
                raise ValueError(f"duplicate reextract receipt: {unit}")
            reextract_units[unit] = row
        reextract_bindings.append({"path": str(path.resolve()), "sha256": sha256_file(path)})

    db_path = args.output_dir / "streaming_owners.sqlite3"
    db = sqlite3.connect(db_path)
    db.executescript(SCHEMA)
    insert = "INSERT INTO streaming_owners VALUES (" + ",".join("?" for _ in range(22)) + ")"
    holes = []
    origins = Counter()
    missing_payload_sha = 0
    for index, (unit, request) in enumerate(requests.items(), 1):
        g = gate[unit]
        expected = request["expected"]
        key = request_key(request)
        evidence = g.get("evidence") or {}
        tensor = evidence.get("observed_tensor") or {}
        safe = evidence.get("safetensors") or {}
        sidecar = evidence.get("sidecar") or {}
        source_index = evidence.get("source_index") or {}
        owner = None
        # Evidence strength order is intentional.  Never let an older route
        # record shadow a materialized, full-SHA-verfied replacement.
        if unit in reextract_units:
            r = reextract_units[unit]
            if tuple(r["key"][1:]) != tuple(key):
                raise ValueError(f"reextract key mismatch for {unit}: {r['key']} != {key}")
            owner = {
                "source_uri": r["source_uri"], "source_index_uri": r["source_index_uri"],
                "payload_sha256": r["payload_sha256"], "payload_size_bytes": r["payload_size_bytes"],
                "header_sha256": r["safetensors_header_sha256"], "index_sha256": r["source_index_sha256"],
                "shape": r["tensor_shape"], "dtype": r["tensor_dtype"],
                "origin": "reextract_full_sha", "status": "PASS_LOCAL_FULL_SHA_VERIFIED",
                "input_length": r["input_length"], "valid_length": r["valid_length"],
                "pad_mode": r["pad_mode"],
            }
        elif unit in minimax_units:
            m = minimax_units[unit]
            owner = {
                "source_uri": m["source_uri"], "source_index_uri": m["source_index_uri"],
                "payload_sha256": m["payload_sha256"], "payload_size_bytes": m["payload_size_bytes"],
                "header_sha256": m["safetensors_header_sha256"], "index_sha256": m["source_index_sha256"],
                "shape": m["observed_tensor"]["shape"], "dtype": m["observed_tensor"]["dtype"],
                "origin": "minimax_contract_v2", "status": "PASS_CONTRACT_CORRECTED",
            }
        # Metadata recovery and REFERENCE_SOURCE_READ_ONLY rows deliberately
        # remain holes.  They did not establish payload readability and were
        # the source of the previous false READY overlay.
        if owner is None:
            holes.append({"unit_id": unit, "key": list(key), "gate_status": g.get("status")})
            continue
        if list(owner.get("shape") or []) != list(expected["expected_shape"]):
            raise ValueError(f"owner shape mismatch for {unit}")
        if str(owner.get("dtype")) != "float16":
            raise ValueError(f"owner dtype mismatch for {unit}")
        for field in ("source_uri", "source_index_uri", "payload_size_bytes", "header_sha256", "index_sha256"):
            if not owner.get(field):
                raise ValueError(f"owner missing {field} for {unit}")
        if not owner.get("payload_sha256"):
            missing_payload_sha += 1
        preprocess = expected["preprocess_identity"]
        input_length = int(owner.get("input_length", preprocess["num_frames"]))
        valid_length = int(owner.get("valid_length", len(expected["raw_ids"])))
        pad_mode = str(owner.get("pad_mode", preprocess["temporal_padding"]))
        db.execute(insert, (
            *key, unit, owner["source_uri"], owner["source_index_uri"], owner.get("payload_sha256"),
            int(owner["payload_size_bytes"]), owner["header_sha256"], owner["index_sha256"],
            json.dumps(expected["expected_shape"], separators=(",", ":")), "float16",
            json.dumps(expected["raw_ids"], separators=(",", ":")), input_length, valid_length,
            pad_mode, owner["origin"], owner["status"],
        ))
        origins[owner["origin"]] += 1
        if index % 20_000 == 0:
            db.commit()
    db.commit()
    owner_count = db.execute("SELECT count(*) FROM streaming_owners").fetchone()[0]
    duplicate_keys = db.execute(
        "SELECT count(*) FROM (SELECT vfm,source,scene_id,layer,timestep,window_id,prefix,count(*) n "
        "FROM streaming_owners GROUP BY 1,2,3,4,5,6,7 HAVING n != 1)"
    ).fetchone()[0]
    ready = owner_count == len(requests) and not holes and duplicate_keys == 0
    db.executemany("INSERT INTO metadata VALUES (?,?)", [
        ("schema", "CANONICAL_V2_PROBE816_STREAMING_OWNER_DB_V1"),
        ("status", "READY" if ready else "NOT_READY"),
        ("requests_sha256", sha256_file(args.requests)),
        ("rows", str(owner_count)),
    ])
    db.commit(); db.execute("VACUUM"); db.close()
    holes_path = args.output_dir / "holes.jsonl.gz"
    with holes_path.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        for row in holes:
            gz.write((json.dumps(row, sort_keys=True) + "\n").encode())
    manifest = {
        "schema": "CANONICAL_V2_PROBE816_STREAMING_OWNER_AUDIT_V1",
        "status": "READY" if ready else "NOT_READY",
        "fail_closed": True,
        "zero_holes": not holes,
        "zero_duplicate_effective_owners": duplicate_keys == 0,
        "requests": {"path": str(args.requests.resolve()), "sha256": sha256_file(args.requests), "rows": len(requests)},
        "database": {"path": str(db_path.resolve()), "sha256": sha256_file(db_path), "rows": owner_count},
        "holes": {"path": str(holes_path.resolve()), "sha256": sha256_file(holes_path), "rows": len(holes)},
        "counts": {"owners": owner_count, "holes": len(holes), "duplicate_keys": duplicate_keys,
                   "extra_gate_receipts_outside_campaign": len(extra_gate),
                   "owners_without_declared_payload_sha": missing_payload_sha, "by_origin": dict(origins)},
        "bindings": {
            "metadata_recovery": sha256_file(args.metadata_recovery),
            "minimax_correction": sha256_file(args.minimax_correction),
            "reextract": reextract_bindings,
        },
        "payload_policy": "declared SHA verified on materialization when present; otherwise exact size+header SHA+index SHA",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
