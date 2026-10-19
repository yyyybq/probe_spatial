#!/usr/bin/env python3
"""Replace only MiniMax owners with the verified BF16 V2 extraction outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np


STREAM_NAMESPACE = "FEAT_STREAMING_CANONICAL_V2_MINIMAX_BF16_V2"
TARGET_NAMESPACE = "FEAT_TARGET_CANONICAL_V2_MINIMAX_BF16_V2"
EXPECTED_STREAM_ROWS = 327_748
EXPECTED_TARGET_ROWS = 715_863
EXPECTED_MINIMAX_STREAM_ROWS = 33_880
EXPECTED_MINIMAX_TARGET_ROWS = 150_708


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_sidecar(payload: Path, namespace: str) -> dict[str, Any]:
    sidecar_path = Path(f"{payload}.manifest.json")
    if payload.is_symlink() or sidecar_path.is_symlink():
        raise RuntimeError(f"symlink rejected: {payload}")
    if not payload.is_file() or not sidecar_path.is_file():
        raise RuntimeError(f"missing payload or sidecar: {payload}")
    sidecar = json.loads(sidecar_path.read_text())
    if sidecar.get("feature_namespace") != namespace:
        raise RuntimeError(f"namespace mismatch: {payload}")
    if sidecar.get("feature_storage_contract") != "MINIMAX_H3_FINITE_BF16_V1":
        raise RuntimeError(f"storage contract mismatch: {payload}")
    if sidecar.get("feature_storage_dtype") != "torch.bfloat16":
        raise RuntimeError(f"storage dtype mismatch: {payload}")
    if int(sidecar.get("size_bytes", -1)) != payload.stat().st_size:
        raise RuntimeError(f"payload size mismatch: {payload}")
    digest = str(sidecar.get("sha256") or "")
    if len(digest) != 64:
        raise RuntimeError(f"payload SHA missing from sidecar: {payload}")
    return sidecar


def load_index(path: Path, namespace: str) -> tuple[Any, str, int]:
    sidecar_path = Path(f"{path}.manifest.json")
    if path.is_symlink() or sidecar_path.is_symlink():
        raise RuntimeError(f"index symlink rejected: {path}")
    if not path.is_file() or not sidecar_path.is_file():
        raise RuntimeError(f"missing index or sidecar: {path}")
    sidecar = json.loads(sidecar_path.read_text())
    if sidecar.get("feature_namespace") != namespace:
        raise RuntimeError(f"index namespace mismatch: {path}")
    if int(sidecar.get("size_bytes", -1)) != path.stat().st_size:
        raise RuntimeError(f"index size mismatch: {path}")
    observed = sha256(path)
    if sidecar.get("sha256") != observed:
        raise RuntimeError(f"index SHA mismatch: {path}")
    return np.load(path, allow_pickle=True), observed, path.stat().st_size


def require_production_pass(ledger_path: Path) -> dict[str, Any]:
    ledger = json.loads(ledger_path.read_text())
    if ledger.get("status") != "SUCCEEDED" or ledger.get("shard_count") != 8:
        raise RuntimeError("MiniMax production ledger is not SUCCEEDED")
    jobs = ledger.get("jobs", {})
    if set(jobs) != {f"{index:02d}" for index in range(8)}:
        raise RuntimeError("MiniMax production ledger does not contain exactly 8 shards")
    covered: list[tuple[int, int]] = []
    for key in sorted(jobs):
        record = jobs[key]
        if record.get("scheduler_state") != "SUCCEEDED":
            raise RuntimeError(f"scheduler shard is not SUCCEEDED: {key}")
        report_path = Path(record["verification_report"])
        if sha256(report_path) != record["verification_report_sha256"]:
            raise RuntimeError(f"verification report SHA mismatch: {key}")
        report = json.loads(report_path.read_text())
        if report.get("status") != "PASS" or report.get("shard_count") != 8:
            raise RuntimeError(f"verification report is not PASS: {key}")
        covered.append((int(report["scene_start"]), int(report["scene_end"])))
    if covered != [(i * 681 // 8, (i + 1) * 681 // 8) for i in range(8)]:
        raise RuntimeError(f"production shard coverage mismatch: {covered}")
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-overlay", type=Path, required=True)
    parser.add_argument("--production-ledger", type=Path, required=True)
    parser.add_argument("--stream-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError(f"output exists: {output}")
    output.mkdir(parents=True)

    base_path = args.base_overlay.resolve()
    base = json.loads(base_path.read_text())
    if base.get("status") != "READY" or base.get("fail_closed") is not True:
        raise RuntimeError("base overlay is not READY")
    ledger = require_production_pass(args.production_ledger.resolve())
    stream_root = args.stream_root.resolve() / "minimax_h3/scannetpp"
    target_root = args.target_root.resolve() / "minimax_h3/scannetpp"
    if STREAM_NAMESPACE not in args.stream_root.parts:
        raise RuntimeError("unexpected MiniMax streaming namespace")
    if TARGET_NAMESPACE not in args.target_root.parts:
        raise RuntimeError("unexpected MiniMax target namespace")

    stream_db = output / "streaming_owners.sqlite3"
    target_db = output / "target_owners.sqlite3"
    shutil.copy2(base["streaming_database"]["path"], stream_db)
    shutil.copy2(base["target_database"]["path"], target_db)

    stream = sqlite3.connect(stream_db)
    stream.row_factory = sqlite3.Row
    stream_rows = stream.execute("SELECT COUNT(*) FROM streaming_owners").fetchone()[0]
    minimax_stream_rows = stream.execute(
        "SELECT COUNT(*) FROM streaming_owners WHERE vfm='minimax_h3'"
    ).fetchone()[0]
    if stream_rows != EXPECTED_STREAM_ROWS or minimax_stream_rows != EXPECTED_MINIMAX_STREAM_ROWS:
        raise RuntimeError("unexpected streaming owner cardinality")
    index_cache: dict[str, tuple[dict[tuple[int, int], dict[str, Any]], str, int, Path]] = {}
    stream_payload_cache: dict[Path, dict[str, Any]] = {}
    for row in stream.execute(
        "SELECT * FROM streaming_owners WHERE vfm='minimax_h3' "
        "ORDER BY scene_id,window_id,prefix,layer"
    ).fetchall():
        scene = str(row["scene_id"])
        if scene not in index_cache:
            index_path = stream_root / scene / "prefix_index.npy"
            records_array, index_sha, index_size = load_index(index_path, STREAM_NAMESPACE)
            records = records_array.tolist()
            if not isinstance(records, list):
                raise RuntimeError(f"prefix index is not a list: {index_path}")
            keyed: dict[tuple[int, int], dict[str, Any]] = {}
            for record in records:
                key = (int(record["window_id"]), int(record["tail"]) + 1)
                if key in keyed:
                    raise RuntimeError(f"duplicate prefix index key: {scene}/{key}")
                keyed[key] = record
            index_cache[scene] = (keyed, index_sha, index_size, index_path)
        keyed, index_sha, _index_size, index_path = index_cache[scene]
        key = (int(row["window_id"]), int(row["prefix"]))
        record = keyed.get(key)
        if record is None:
            raise RuntimeError(f"missing exact prefix index record: {scene}/{key}")
        raw_ids = [int(value) for value in json.loads(row["raw_ids_json"])]
        if [int(value) for value in record["indices"]] != raw_ids:
            raise RuntimeError(f"raw frame IDs differ: {scene}/{key}")
        payload = (
            stream_root / scene / f"window_{key[0]:04d}"
            / f"prefix_{key[1] - 1:06d}"
            / f"feature_t250_layer{int(row['layer'])}.sft"
        )
        if payload not in stream_payload_cache:
            stream_payload_cache[payload] = load_sidecar(payload, STREAM_NAMESPACE)
        sidecar = stream_payload_cache[payload]
        shape = sidecar.get("tensors", {}).get("feat", {}).get("shape")
        if shape != [22, 15, 26, 5376]:
            raise RuntimeError(f"streaming tensor shape mismatch: {payload}")
        stream.execute(
            "UPDATE streaming_owners SET source_uri=?,source_index_uri=?,"
            "payload_sha256=?,payload_size_bytes=?,payload_header_sha256=?,"
            "source_index_sha256=?,tensor_shape_json=?,tensor_dtype=?,"
            "input_length=?,valid_length=?,pad_mode=?,origin=?,status=? "
            "WHERE vfm=? AND source=? AND scene_id=? AND layer=? AND timestep=? "
            "AND window_id=? AND prefix=?",
            (
                str(payload), str(index_path), str(sidecar["sha256"]),
                int(sidecar["size_bytes"]), "GAP_RECEIPT_FULL_SHA", index_sha,
                json.dumps(shape, separators=(",", ":")), "bfloat16",
                int(record["input_length"]), int(record["valid_length"]),
                str(record["pad_mode"]), "minimax_bf16_v2_extraction",
                "PASS_LOCAL_VERIFIED", row["vfm"], row["source"], scene,
                row["layer"], row["timestep"], row["window_id"], row["prefix"],
            ),
        )
    stream.executemany(
        "INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)",
        [
            ("minimax_bf16_v2_production_ledger_sha256", sha256(args.production_ledger)),
            ("minimax_bf16_v2_stream_namespace", STREAM_NAMESPACE),
        ],
    )
    stream.commit()
    stream_duplicates = stream.execute(
        "SELECT COUNT(*) FROM (SELECT vfm,source,scene_id,layer,timestep,"
        "window_id,prefix,COUNT(*) n FROM streaming_owners GROUP BY 1,2,3,4,5,6,7 "
        "HAVING n != 1)"
    ).fetchone()[0]
    stream.close()

    target = sqlite3.connect(target_db)
    target.row_factory = sqlite3.Row
    target_rows = target.execute("SELECT COUNT(*) FROM target_owners").fetchone()[0]
    minimax_target_rows = target.execute(
        "SELECT COUNT(*) FROM target_owners WHERE vfm='minimax_h3'"
    ).fetchone()[0]
    if target_rows != EXPECTED_TARGET_ROWS or minimax_target_rows != EXPECTED_MINIMAX_TARGET_ROWS:
        raise RuntimeError("unexpected target owner cardinality")
    target_index_cache: dict[str, tuple[dict[int, int], str, int, Path]] = {}
    target_payload_cache: dict[Path, dict[str, Any]] = {}
    for row in target.execute(
        "SELECT * FROM target_owners WHERE vfm='minimax_h3' "
        "ORDER BY scene_id,layer,target_raw_id"
    ).fetchall():
        scene = str(row["scene_id"])
        if scene not in target_index_cache:
            index_path = target_root / scene / "target_indices.npy"
            indices_array, index_sha, index_size = load_index(index_path, TARGET_NAMESPACE)
            indices = [int(value) for value in indices_array.astype(np.int64).tolist()]
            if len(indices) != len(set(indices)):
                raise RuntimeError(f"duplicate target raw ID: {index_path}")
            target_index_cache[scene] = (
                {raw_id: offset for offset, raw_id in enumerate(indices)},
                index_sha, index_size, index_path,
            )
        offsets, index_sha, index_size, index_path = target_index_cache[scene]
        raw_id = int(row["target_raw_id"])
        if raw_id not in offsets:
            raise RuntimeError(f"missing exact target raw ID: {scene}/{raw_id}")
        payload = target_root / scene / f"feature_t250_layer{int(row['layer'])}.sft"
        if payload not in target_payload_cache:
            target_payload_cache[payload] = load_sidecar(payload, TARGET_NAMESPACE)
        sidecar = target_payload_cache[payload]
        shape = sidecar.get("tensors", {}).get("feat", {}).get("shape")
        if not isinstance(shape, list) or shape[1:] != [15, 26, 5376]:
            raise RuntimeError(f"target tensor shape mismatch: {payload}")
        if offsets[raw_id] >= int(shape[0]):
            raise RuntimeError(f"target row index exceeds payload: {scene}/{raw_id}")
        target.execute(
            "UPDATE target_owners SET source_uri=?,source_index_uri=?,"
            "source_row_index=?,tensor_key='feat',payload_sha256=?,"
            "payload_size_bytes=?,payload_header_sha256=?,index_sha256=?,"
            "index_size_bytes=?,origin=?,status=? WHERE vfm=? AND source=? "
            "AND scene_id=? AND layer=? AND timestep=? AND canonical_row_index=?",
            (
                str(payload), str(index_path), offsets[raw_id], str(sidecar["sha256"]),
                int(sidecar["size_bytes"]), "GAP_RECEIPT_FULL_SHA", index_sha,
                index_size, "minimax_bf16_v2_extraction", "PASS_LOCAL_VERIFIED",
                row["vfm"], row["source"], scene, row["layer"], row["timestep"],
                row["canonical_row_index"],
            ),
        )
    target.executemany(
        "INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)",
        [
            ("minimax_bf16_v2_production_ledger_sha256", sha256(args.production_ledger)),
            ("minimax_bf16_v2_target_namespace", TARGET_NAMESPACE),
        ],
    )
    target.commit()
    target_duplicates = target.execute(
        "SELECT COUNT(*) FROM (SELECT vfm,source,scene_id,layer,timestep,"
        "target_raw_id,COUNT(*) n FROM target_owners GROUP BY 1,2,3,4,5,6 "
        "HAVING n != 1)"
    ).fetchone()[0]
    target_holes = target.execute(
        "SELECT COUNT(*) FROM target_expected e LEFT JOIN target_owners o USING "
        "(vfm,source,scene_id,layer,timestep,canonical_row_index) "
        "WHERE o.source_uri IS NULL"
    ).fetchone()[0]
    target.close()
    if stream_duplicates or target_duplicates or target_holes:
        raise RuntimeError("owner closure check failed")

    audit = {
        "schema": "PROBE816_MINIMAX_BF16_V2_REBIND_AUDIT_V1",
        "status": "READY",
        "models": base["models"],
        "zero_holes": True,
        "zero_duplicate_effective_owners": True,
        "streaming": {
            "owner_rows": EXPECTED_STREAM_ROWS,
            "duplicate_keys": stream_duplicates,
            "minimax_rows_rebound": EXPECTED_MINIMAX_STREAM_ROWS,
            "namespace": STREAM_NAMESPACE,
            "database": {"path": str(stream_db), "sha256": sha256(stream_db)},
        },
        "target": {
            "owner_rows": EXPECTED_TARGET_ROWS,
            "holes": target_holes,
            "duplicate_keys": target_duplicates,
            "minimax_rows_rebound": EXPECTED_MINIMAX_TARGET_ROWS,
            "namespace": TARGET_NAMESPACE,
            "database": {"path": str(target_db), "sha256": sha256(target_db)},
        },
        "minimax_bf16_route": {
            "path": str(args.production_ledger.resolve()),
            "sha256": sha256(args.production_ledger.resolve()),
            "identity_sha256": ledger["source_tree_sha256"],
        },
        "base_overlay": {"path": str(base_path), "sha256": sha256(base_path)},
        "physical_evidence": {
            "production_shards": 8,
            "scheduler_terminal_state": "SUCCEEDED",
            "streaming_payloads_checked_by_shard_verifiers": EXPECTED_MINIMAX_STREAM_ROWS,
            "target_payloads_checked_by_shard_verifiers": 681 * 4,
        },
    }
    audit_path = output / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
