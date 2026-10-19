#!/usr/bin/env python3
"""Build a tiny owner overlay directly from verified extraction receipts.

This intentionally does one thing: bind an exact
(model, scene, timestep, layer, window, prefix, raw-frame-ids) key to the URI
recorded by the previous extraction receipt.  It is for loader canaries, not
for inferring or filling missing features.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/umm/users/yinbaiqiao/probe_spatial")
PARENT = ROOT / "manifests/c_retrain_20260722/inscene_c_h124_prefix24.jsonl"
SNAPSHOT = ROOT / "artifacts/main_v2_probe_v2_816_20260917/extraction_receipt_snapshot_r1"
OLD_OVERLAY = ROOT / "artifacts/main_v2_probe_v2_816_20260917/mixed_owner_overlay_r2/manifest.json"
PARENT_SHA = "ebc496eb0587cda372ccf29a7c5244ef454096151f73c8eed9511967ded1902d"

SELECTORS = {
    "cogvideox": (749, 20, 8),
    "videomae": (749, 0, 8),
    "dino": (749, 0, 8),
    "f3r": (749, 24, 8),
    "vjepa2": (749, 0, 8),
    "qwen2_5_vl": (749, -1, 8),
    "lingbot": (250, 10, 8),
}

SKIPPED: dict[str, str] = {}

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


def aoss_root(receipt: dict[str, Any], model: str) -> str:
    namespace = str(receipt["lineage"]["feature_namespace"])
    return f"aoss://baiqiao/InsScene-15K/{namespace}/{model}/scannetpp"


def object_layer(name: str, model: str) -> int:
    match = re.search(r"_layer(-?\d+)\.sft$", name)
    if match:
        return int(match.group(1))
    match = re.search(r"_l(-?\d+)\.sft$", name)
    if match:
        return int(match.group(1))
    if name.endswith("/feature.sft") and model == "dino":
        return 0
    raise ValueError(f"cannot parse feature layer from {name}")


def parse_object_key(name: str, model: str) -> tuple[str, int, int, int]:
    match = re.fullmatch(
        r"([^/]+)/window_(\d+)/prefix_(\d+)/[^/]+\.sft", name
    )
    if not match:
        raise ValueError(f"invalid streaming receipt object path: {name}")
    scene, window, tail = match.groups()
    return scene, int(window), int(tail) + 1, object_layer(name, model)


def verified_index_record(
    uri: str,
    evidence: dict[str, Any],
    window_id: int,
    prefix: int,
) -> dict[str, Any]:
    """Read and SHA-check an old receipt's index before accepting its IDs."""
    import numpy as np

    sys.path.insert(0, str(ROOT / "runtime/main_v2_probe_v2_r5/source"))
    from vidfm3d.utils import aoss_fs

    payload = aoss_fs.read_bytes(uri)
    if len(payload) != int(evidence["payload_size_bytes"]):
        raise ValueError(f"index size mismatch: {uri}")
    if hashlib.sha256(payload).hexdigest() != evidence["payload_sha256"]:
        raise ValueError(f"index SHA mismatch: {uri}")
    records = np.load(io.BytesIO(payload), allow_pickle=True).tolist()
    if isinstance(records, dict):
        records = list(records.values())
    matches = [
        row for row in records
        if isinstance(row, dict)
        and int(row.get("window_id", -1)) == window_id
        and int(row.get("tail", -1)) == prefix - 1
    ]
    if len(matches) != 1:
        raise ValueError(
            f"index key cardinality mismatch for {uri}: "
            f"window={window_id}, prefix={prefix}, matches={len(matches)}"
        )
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    output.mkdir(parents=True)

    if sha256_file(PARENT) != PARENT_SHA:
        raise ValueError("canonical parent manifest SHA mismatch")
    parents: dict[tuple[str, int], dict[str, Any]] = {}
    with PARENT.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            parents[(str(row["scene_id"]), int(row["streaming_window_id"]))] = row

    snapshot_manifest_path = SNAPSHOT / "manifest.json"
    snapshot_manifest = json.loads(snapshot_manifest_path.read_text(encoding="utf-8"))
    selected: list[dict[str, Any]] = []
    for model, (timestep, layer, prefix) in SELECTORS.items():
        entries = [
            row for row in snapshot_manifest["receipts"]
            if row["kind"] == "streaming"
            and row["vfm"] == model
            and int(row["timestep"]) == timestep
        ]
        found = None
        for entry in entries:
            receipt_path = Path(entry["snapshot_path"])
            if sha256_file(receipt_path) != entry["file_sha256"]:
                raise ValueError(f"receipt SHA mismatch: {receipt_path}")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("status") != "PASS_REMOTE_VERIFIED":
                continue
            for relative, evidence in receipt["objects"].items():
                if not relative.endswith(".sft"):
                    continue
                scene, window, observed_prefix, observed_layer = parse_object_key(relative, model)
                if (observed_prefix, observed_layer) != (prefix, layer):
                    continue
                parent = parents.get((scene, window))
                if parent is None:
                    continue
                index_name = f"{scene}/prefix_index.npy"
                index_evidence = receipt["objects"].get(index_name)
                if not index_evidence:
                    raise ValueError(f"receipt lacks exact index object: {index_name}")
                root = aoss_root(receipt, model)
                index_uri = f"{root}/{index_name}"
                raw_ids = [int(v) for v in parent["streaming_window_indices"][:prefix]]
                identity = evidence.get("canonical_unit_identity") or {}
                expected_ids_sha = identity.get("gap_record_raw_ids_sha256")
                index_record = None
                if expected_ids_sha:
                    if canonical_sha(raw_ids) != expected_ids_sha:
                        continue
                    input_length = int(identity["input_num_frames"])
                    valid_length = prefix
                    pad_mode = str(identity["temporal_padding"])
                    identity_proof = "receipt_raw_ids_sha256"
                else:
                    index_record = verified_index_record(
                        index_uri, index_evidence, window, prefix
                    )
                    observed_ids = [int(v) for v in index_record["indices"]]
                    if observed_ids != raw_ids:
                        continue
                    input_length = int(index_record["input_length"])
                    valid_length = int(index_record["valid_length"])
                    pad_mode = str(index_record["pad_mode"])
                    identity_proof = "receipt_sha_bound_prefix_index_record"
                found = {
                    "model": model,
                    "source": "scannetpp",
                    "scene_id": scene,
                    "window_id": window,
                    "prefix": prefix,
                    "layer": layer,
                    "timestep": timestep,
                    "relative_path": relative,
                    "source_uri": f"{root}/{relative}",
                    "source_index_uri": index_uri,
                    "payload_sha256": evidence["payload_sha256"],
                    "payload_size_bytes": int(evidence["payload_size_bytes"]),
                    "source_index_sha256": index_evidence["payload_sha256"],
                    "raw_ids": raw_ids,
                    "input_length": input_length,
                    "valid_length": valid_length,
                    "pad_mode": pad_mode,
                    "identity_proof": identity_proof,
                    "receipt_path": str(receipt_path),
                    "receipt_sha256": entry["file_sha256"],
                    "receipt_content_sha256": receipt["content_sha256"],
                    "parent": parent,
                }
                break
            if found:
                break
        if found is None:
            raise ValueError(f"no exact verified receipt row for {model}:{timestep}:{layer}:p{prefix}")
        selected.append(found)

    db_path = output / "streaming_receipt_owners.sqlite3"
    db = sqlite3.connect(db_path)
    db.executescript(SCHEMA)
    insert = "INSERT INTO streaming_owners VALUES (" + ",".join("?" for _ in range(22)) + ")"
    manifests = output / "manifests"
    manifests.mkdir()
    public_rows = []
    for row in selected:
        unit_id = canonical_sha([
            row["model"], row["source"], row["scene_id"], row["layer"],
            row["timestep"], row["window_id"], row["prefix"], row["raw_ids"],
        ])
        db.execute(insert, (
            row["model"], row["source"], row["scene_id"], row["layer"],
            row["timestep"], row["window_id"], row["prefix"], unit_id,
            row["source_uri"], row["source_index_uri"], row["payload_sha256"],
            row["payload_size_bytes"], "GAP_RECEIPT_FULL_SHA", row["source_index_sha256"],
            "[]", "receipt_verified", json.dumps(row["raw_ids"], separators=(",", ":")),
            row["input_length"], row["valid_length"], row["pad_mode"],
            "extraction_receipt_full_sha", "PASS_REMOTE_VERIFIED",
        ))
        sample = dict(row.pop("parent"))
        sample["streaming_indices"] = row["raw_ids"]
        sample["streaming_tail"] = row["prefix"] - 1
        sample["streaming_tail_raw"] = row["raw_ids"][-1]
        sample["sample_id"] = f"{row['scene_id']}_w{row['window_id']:03d}_p{row['prefix']}"
        manifest_path = manifests / f"{row['model']}.jsonl"
        manifest_path.write_text(
            json.dumps(sample, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        row["manifest_path"] = str(manifest_path)
        row["manifest_sha256"] = sha256_file(manifest_path)
        public_rows.append(row)
    db.executemany("INSERT INTO metadata VALUES (?,?)", [
        ("schema", "PROBE816_RECEIPT_LOADER_CANARY_V1"),
        ("status", "READY"),
        ("rows", str(len(selected))),
        ("parent_manifest_sha256", PARENT_SHA),
        ("receipt_snapshot_sha256", sha256_file(snapshot_manifest_path)),
    ])
    db.commit()
    db.execute("VACUUM")
    db.close()

    old_overlay = json.loads(OLD_OVERLAY.read_text(encoding="utf-8"))
    overlay = {
        "schema": "CANONICAL_V2_PROBE816_MIXED_OWNER_OVERLAY_V1",
        "status": "READY",
        "fail_closed": True,
        "purpose": "receipt-exact real-loader canary only",
        "streaming_database": {
            "path": str(db_path), "sha256": sha256_file(db_path), "rows": len(selected),
        },
        "target_database": old_overlay["target_database"],
        "receipt_snapshot": {
            "path": str(snapshot_manifest_path), "sha256": sha256_file(snapshot_manifest_path),
        },
        "canonical_manifest": {"path": str(PARENT), "sha256": PARENT_SHA},
    }
    overlay_path = output / "overlay_manifest.json"
    overlay_path.write_text(json.dumps(overlay, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selection = {
        "schema": "PROBE816_RECEIPT_LOADER_CANARY_SELECTION_V1",
        "status": "READY",
        "overlay_manifest": {"path": str(overlay_path), "sha256": sha256_file(overlay_path)},
        "rows": public_rows,
        "skipped": SKIPPED,
    }
    (output / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(selection, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
