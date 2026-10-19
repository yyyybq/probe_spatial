#!/usr/bin/env python3
"""Build a fail-closed target-row owner database for the frozen probe816 set.

The database contains exactly the target rows consumed by A2/B/C probes in
the campaign.  It merges the audited legacy row routes with exact gap-delta
objects and refuses holes or conflicting owners.  Payloads are not copied;
their SHA/size bindings are carried into the immutable overlay contract.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PREFIXES = (8, 12, 16, 24)
HORIZONS = (1, 2, 4)
TARGET_PROBES = frozenset({"action_dynamics", "counterfactual", "path_integration"})
DISK_LAYER = {("lingbot", 48): 47, ("minimax_h3", 50): 49}
CANONICAL_SOURCE_SHA = "ebc496eb0587cda372ccf29a7c5244ef454096151f73c8eed9511967ded1902d"
CANONICAL_UNIVERSE_SHA = "e4e031ff892b5d356bafe038e905428fc2a2d1a1469fb845c0b8dadbc5911737"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL line {number}: {path}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL line {number}: {path}")
            yield row


def aoss_uri(root: str, relative: str) -> str:
    marker = "/InsScene-15K/"
    if root.startswith("aoss://"):
        return root.rstrip("/") + "/" + relative.lstrip("/")
    if marker not in root:
        raise ValueError(f"cannot map target gap root to AOSS: {root}")
    return "aoss://baiqiao/InsScene-15K/" + root.split(marker, 1)[1].rstrip("/") + "/" + relative.lstrip("/")


def selected_specs(
    experiments: Path,
    expected_experiment_rows: int,
    expected_selected_specs: int,
) -> set[tuple[str, int, int]]:
    with experiments.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if (
        len(rows) != expected_experiment_rows
        or len({row["experiment_id"] for row in rows}) != expected_experiment_rows
    ):
        raise ValueError(
            f"campaign is not exactly {expected_experiment_rows} unique experiments"
        )
    specs = set()
    for row in rows:
        if row["probe_config"] not in TARGET_PROBES:
            continue
        model = row["model"]
        logical = int(row["layer"])
        layer = DISK_LAYER.get((model, logical), logical)
        timestep = 250 if model in {"lingbot", "minimax_h3"} else 749
        specs.add((model, layer, timestep))
    if len(specs) != expected_selected_specs:
        raise ValueError(
            f"expected {expected_selected_specs} target feature specs, found {len(specs)}"
        )
    return specs


def canonical_universe(path: Path) -> dict[tuple[str, str], tuple[int, ...]]:
    if sha256_file(path) != CANONICAL_SOURCE_SHA:
        raise ValueError("canonical source SHA mismatch")
    targets: dict[tuple[str, str], set[int]] = defaultdict(set)
    windows: set[tuple[str, str, int]] = set()
    for row in jsonl(path):
        source, scene = str(row["source"]), str(row["scene_id"])
        window_id = int(row["streaming_window_id"])
        window = tuple(int(v) for v in row["streaming_window_indices"])
        prefix24 = tuple(int(v) for v in row["streaming_indices"])
        if len(window) != 28 or prefix24 != window[:24]:
            raise ValueError(f"invalid canonical obs28/p24 window: {source}/{scene}/{window_id}")
        key = (source, scene, window_id)
        if key in windows:
            raise ValueError(f"duplicate canonical window: {key}")
        windows.add(key)
        for prefix in PREFIXES:
            for pos in (prefix - 1,) + tuple(prefix - 1 + h for h in HORIZONS):
                targets[(source, scene)].add(window[pos])
    frozen = {key: tuple(sorted(values)) for key, values in sorted(targets.items())}
    payload = [
        {"source": source, "scene_id": scene, "target_raw_ids": list(values)}
        for (source, scene), values in frozen.items()
    ]
    if canonical_sha(payload) != CANONICAL_UNIVERSE_SHA:
        raise ValueError("canonical target universe SHA mismatch")
    if len(frozen) != 681 or sum(map(len, frozen.values())) != 37_677:
        raise ValueError("canonical target universe count mismatch")
    return frozen


SCHEMA = """
PRAGMA journal_mode=DELETE;
PRAGMA synchronous=FULL;
CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
CREATE TABLE target_expected(
  vfm TEXT NOT NULL, source TEXT NOT NULL, scene_id TEXT NOT NULL,
  layer INTEGER NOT NULL, timestep INTEGER NOT NULL,
  canonical_row_index INTEGER NOT NULL, target_raw_id INTEGER NOT NULL,
  PRIMARY KEY(vfm, source, scene_id, layer, timestep, canonical_row_index),
  UNIQUE(vfm, source, scene_id, layer, timestep, target_raw_id)
) WITHOUT ROWID;
CREATE TABLE target_owners(
  vfm TEXT NOT NULL, source TEXT NOT NULL, scene_id TEXT NOT NULL,
  layer INTEGER NOT NULL, timestep INTEGER NOT NULL,
  canonical_row_index INTEGER NOT NULL, target_raw_id INTEGER NOT NULL,
  source_uri TEXT NOT NULL, source_index_uri TEXT NOT NULL,
  source_row_index INTEGER NOT NULL, tensor_key TEXT NOT NULL,
  payload_sha256 TEXT NOT NULL, payload_size_bytes INTEGER NOT NULL,
  payload_header_sha256 TEXT NOT NULL,
  index_sha256 TEXT NOT NULL, index_size_bytes INTEGER,
  origin TEXT NOT NULL, status TEXT NOT NULL,
  PRIMARY KEY(vfm, source, scene_id, layer, timestep, canonical_row_index),
  UNIQUE(vfm, source, scene_id, layer, timestep, target_raw_id),
  FOREIGN KEY(vfm, source, scene_id, layer, timestep, canonical_row_index)
    REFERENCES target_expected(vfm, source, scene_id, layer, timestep, canonical_row_index)
) WITHOUT ROWID;
CREATE INDEX target_lookup_raw ON target_owners(vfm, source, scene_id, layer, timestep, target_raw_id);
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", type=Path, required=True)
    ap.add_argument("--canonical-source", type=Path, required=True)
    ap.add_argument("--legacy-routes", type=Path, required=True)
    ap.add_argument("--gap-index", type=Path, required=True)
    ap.add_argument("--payload-hash-receipts", type=Path)
    ap.add_argument("--expected-experiment-rows", type=int, default=816)
    ap.add_argument("--expected-selected-specs", type=int, default=19)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.output_dir.exists():
        raise ValueError(f"output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    specs = selected_specs(
        args.experiments,
        args.expected_experiment_rows,
        args.expected_selected_specs,
    )
    universe = canonical_universe(args.canonical_source)
    expected_rows = len(specs) * sum(map(len, universe.values()))

    db_path = args.output_dir / "target_owners.sqlite3"
    db = sqlite3.connect(db_path)
    db.executescript(SCHEMA)
    db.execute("PRAGMA foreign_keys=ON")
    db.executemany(
        "INSERT INTO metadata(key,value) VALUES (?,?)",
        [
            ("schema", "CANONICAL_V2_PROBE816_TARGET_OWNER_DB_V1"),
            ("canonical_source_sha256", CANONICAL_SOURCE_SHA),
            ("canonical_universe_sha256", CANONICAL_UNIVERSE_SHA),
            ("experiments_sha256", sha256_file(args.experiments)),
        ],
    )
    insert_expected = "INSERT INTO target_expected VALUES (?,?,?,?,?,?,?)"
    batch = []
    for vfm, layer, timestep in sorted(specs):
        for (source, scene), ids in universe.items():
            batch.extend((vfm, source, scene, layer, timestep, i, raw) for i, raw in enumerate(ids))
            if len(batch) >= 10_000:
                db.executemany(insert_expected, batch); batch.clear()
    if batch:
        db.executemany(insert_expected, batch)
    db.commit()

    hash_receipts: dict[str, dict[str, Any]] = {}
    if args.payload_hash_receipts is not None:
        for row in jsonl(args.payload_hash_receipts):
            if row.get("status") != "PASS_REMOTE_FULL_SHA":
                raise ValueError("payload hash receipt is not PASS_REMOTE_FULL_SHA")
            uri = str(row.get("source_uri") or "")
            digest = str(row.get("payload_sha256") or "")
            if not uri or len(digest) != 64 or uri in hash_receipts:
                raise ValueError(f"malformed/duplicate payload hash receipt: {uri}")
            hash_receipts[uri] = row

    insert_owner = """INSERT INTO target_owners VALUES
      (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""
    legacy_rows = 0
    skipped_legacy = 0
    pending_hash_objects: dict[str, dict[str, Any]] = {}
    for row in jsonl(args.legacy_routes):
        spec = (str(row["vfm"]), int(row["layer"]), int(row["timestep"]))
        if spec not in specs:
            skipped_legacy += 1
            continue
        payload_sha = str(row.get("source_payload_sha256") or "")
        index_sha = str(row.get("source_index_sha256") or "")
        source_uri = str(row["source_feature_path"])
        header_sha = str(row.get("source_tensor_header_sha256") or "")
        if len(index_sha) != 64 or len(header_sha) != 64:
            raise ValueError(f"legacy route lacks index/header SHA binding: {row.get('spec_id')}/{row.get('scene_id')}")
        status = "OWNER_SHA_BOUND_VERIFY_ON_MATERIALIZE"
        if len(payload_sha) != 64:
            receipt = hash_receipts.get(source_uri)
            if receipt is None:
                pending_hash_objects.setdefault(source_uri, {
                    "source_uri": source_uri,
                    "sidecar_uri": str(row.get("source_feature_sidecar_path") or ""),
                    "expected_size_bytes": int(row["source_size_bytes"]),
                    "expected_header_sha256": header_sha,
                    "vfm": spec[0], "layer": spec[1], "timestep": spec[2],
                })
                payload_sha = "PENDING"
                status = "PENDING_REMOTE_FULL_SHA"
            else:
                if (
                    int(receipt.get("payload_size_bytes", -1)) != int(row["source_size_bytes"])
                    or str(receipt.get("safetensors_header_sha256") or "") != header_sha
                ):
                    raise ValueError(f"payload hash receipt binding mismatch: {source_uri}")
                payload_sha = str(receipt["payload_sha256"])
                status = "PASS_REMOTE_FULL_SHA"
        values = (
            spec[0], str(row["source"]), str(row["scene_id"]), spec[1], spec[2],
            int(row["canonical_row_index"]), int(row["target_raw_id"]),
            source_uri, str(row["source_index_path"]),
            int(row["source_row_index"]), str(row.get("tensor_key") or "feat"),
            payload_sha, int(row["source_size_bytes"]), header_sha, index_sha, None,
            "legacy_route_audit", status,
        )
        try:
            db.execute(insert_owner, values)
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"duplicate/invalid legacy owner {values[:7]}: {exc}") from exc
        legacy_rows += 1
        if legacy_rows % 20_000 == 0:
            db.commit()
    db.commit()

    # Reduce gap receipts to one exact feature/index pair per selected unit.
    gap_objects: dict[tuple[str, int, int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    duplicate_identical = 0
    skipped_gap = 0
    for row in jsonl(args.gap_index):
        if row.get("mode") != "target_isolated":
            continue
        ident = row.get("identity") or {}
        spec = (str(row.get("vfm") or ident.get("vfm") or ""), int(ident.get("layer", -999)), int(ident.get("timestep", -999)))
        if spec not in specs:
            skipped_gap += 1
            continue
        if ident.get("canonical_source_sha256") != CANONICAL_SOURCE_SHA:
            raise ValueError(f"gap canonical source mismatch: {spec}")
        role = str(ident.get("payload_role") or "")
        if role not in {"feature", "target_indices"}:
            continue
        scene = str(ident["scene"])
        key = (spec[0], spec[1], spec[2], scene)
        candidate = {
            "uri": aoss_uri(str(row["root"]), str(row["relative_path"])),
            "sha256": str(row.get("payload_sha256") or ""),
            "size": int(row.get("payload_size_bytes", -1)),
            "target_indices": tuple(int(v) for v in ident.get("target_indices") or ()),
            "target_indices_sha256": str(ident.get("target_indices_sha256") or ""),
        }
        if len(candidate["sha256"]) != 64 or not candidate["target_indices"]:
            raise ValueError(f"incomplete gap object: {key}/{role}")
        prior = gap_objects[key].get(role)
        if prior is not None:
            if prior != candidate:
                raise ValueError(f"conflicting gap owners: {key}/{role}")
            duplicate_identical += 1
        else:
            gap_objects[key][role] = candidate

    gap_rows = 0
    gap_units = 0
    for (vfm, layer, timestep, scene), roles in sorted(gap_objects.items()):
        if set(roles) != {"feature", "target_indices"}:
            raise ValueError(f"incomplete target gap pair: {(vfm, layer, timestep, scene)} {set(roles)}")
        feature, index = roles["feature"], roles["target_indices"]
        if feature["target_indices"] != index["target_indices"]:
            raise ValueError(f"target gap feature/index identity mismatch: {(vfm, layer, timestep, scene)}")
        if index["sha256"] != index["target_indices_sha256"]:
            raise ValueError(f"target index SHA mismatch: {(vfm, layer, timestep, scene)}")
        canonical = universe[("scannetpp", scene)]
        canonical_pos = {raw: i for i, raw in enumerate(canonical)}
        for source_row, raw_id in enumerate(feature["target_indices"]):
            if raw_id not in canonical_pos:
                raise ValueError(f"gap target raw ID outside canonical universe: {scene}/{raw_id}")
            values = (
                vfm, "scannetpp", scene, layer, timestep, canonical_pos[raw_id], raw_id,
                feature["uri"], index["uri"], source_row, "feat", feature["sha256"],
                feature["size"], "GAP_RECEIPT_FULL_SHA", index["sha256"], index["size"],
                "gap_delta_exact", "PASS_REMOTE_VERIFIED",
            )
            try:
                db.execute(insert_owner, values)
            except sqlite3.IntegrityError as exc:
                # A gap is only eligible for an actually unowned expected row.
                existing = db.execute(
                    "SELECT source_uri,payload_sha256 FROM target_owners WHERE "
                    "vfm=? AND source=? AND scene_id=? AND layer=? AND timestep=? AND canonical_row_index=?",
                    values[:6],
                ).fetchone()
                raise ValueError(f"duplicate target writer {values[:7]} existing={existing}: {exc}") from exc
            gap_rows += 1
        gap_units += 1
        if gap_units % 500 == 0:
            db.commit()
    db.commit()

    counts = dict(db.execute(
        "SELECT (SELECT count(*) FROM target_expected), (SELECT count(*) FROM target_owners)"
    ).fetchone()) if False else {}
    expected_count = db.execute("SELECT count(*) FROM target_expected").fetchone()[0]
    owner_count = db.execute("SELECT count(*) FROM target_owners").fetchone()[0]
    holes = db.execute(
        "SELECT count(*) FROM target_expected e LEFT JOIN target_owners o USING"
        "(vfm,source,scene_id,layer,timestep,canonical_row_index) WHERE o.source_uri IS NULL"
    ).fetchone()[0]
    duplicates = db.execute(
        "SELECT count(*) FROM (SELECT vfm,source,scene_id,layer,timestep,canonical_row_index,count(*) n "
        "FROM target_owners GROUP BY 1,2,3,4,5,6 HAVING n != 1)"
    ).fetchone()[0]
    by_origin = dict(db.execute("SELECT origin,count(*) FROM target_owners GROUP BY origin"))
    by_spec = [
        {"vfm": vfm, "layer": layer, "timestep": timestep, "owners": count}
        for vfm, layer, timestep, count in db.execute(
            "SELECT vfm,layer,timestep,count(*) FROM target_owners GROUP BY vfm,layer,timestep ORDER BY 1,2,3"
        )
    ]
    ready = (
        expected_count == expected_rows and owner_count == expected_rows
        and holes == 0 and duplicates == 0 and not pending_hash_objects
    )
    if expected_count != expected_rows or owner_count != expected_rows or holes or duplicates:
        raise ValueError(
            f"target owner audit not ready expected={expected_count}/{expected_rows} "
            f"owners={owner_count} holes={holes} duplicates={duplicates}"
        )
    db.execute("INSERT INTO metadata VALUES (?,?)", ("status", "READY" if ready else "NOT_READY"))
    db.execute("INSERT INTO metadata VALUES (?,?)", ("rows", str(owner_count)))
    db.commit()
    db.execute("VACUUM")
    db.close()

    pending_path = args.output_dir / "pending_payload_hash_objects.jsonl.gz"
    with pending_path.open("wb") as raw, gzip.GzipFile(
        filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6
    ) as gz:
        for row in sorted(pending_hash_objects.values(), key=lambda x: x["source_uri"]):
            gz.write((json.dumps(row, sort_keys=True) + "\n").encode())
    manifest = {
        "schema": "CANONICAL_V2_PROBE816_TARGET_OWNER_AUDIT_V1",
        "status": "READY" if ready else "NOT_READY_PENDING_PAYLOAD_SHA",
        "fail_closed": True,
        "zero_holes": holes == 0,
        "zero_duplicate_effective_owners": duplicates == 0,
        "experiments": {
            "path": str(args.experiments.resolve()),
            "sha256": sha256_file(args.experiments),
            "rows": args.expected_experiment_rows,
        },
        "canonical_source": {"path": str(args.canonical_source.resolve()), "sha256": sha256_file(args.canonical_source)},
        "legacy_routes": {"path": str(args.legacy_routes.resolve()), "sha256": sha256_file(args.legacy_routes)},
        "gap_index": {"path": str(args.gap_index.resolve()), "sha256": sha256_file(args.gap_index)},
        "payload_hash_receipts": None if args.payload_hash_receipts is None else {
            "path": str(args.payload_hash_receipts.resolve()), "sha256": sha256_file(args.payload_hash_receipts),
            "rows": len(hash_receipts),
        },
        "pending_payload_hash_objects": {
            "path": str(pending_path.resolve()), "sha256": sha256_file(pending_path),
            "rows": len(pending_hash_objects),
        },
        "database": {"path": str(db_path.resolve()), "sha256": sha256_file(db_path), "size_bytes": db_path.stat().st_size},
        "counts": {
            "selected_specs": len(specs), "expected_rows": expected_count, "owner_rows": owner_count,
            "holes": holes, "duplicate_effective_owners": duplicates,
            "legacy_owner_rows": legacy_rows, "gap_owner_rows": gap_rows,
            "gap_units": gap_units, "identical_duplicate_receipts_collapsed": duplicate_identical,
            "skipped_legacy_rows": skipped_legacy, "skipped_gap_objects": skipped_gap,
            "pending_payload_hash_objects": len(pending_hash_objects),
            "by_origin": by_origin,
        },
        "specs": by_spec,
        "payload_policy": "FULL_SHA_REQUIRED_BEFORE_READY",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
