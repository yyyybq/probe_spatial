#!/usr/bin/env python3
"""Freeze the 168-item Wan probing scope and its complete streaming owner overlay."""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/main_v2_probe_v2_816_20260917"
OUT = BASE / "wan_probing_168_20260923_r2"
GATE = ROOT / "artifacts/wan_aether_streaming_official_r1"
REQUESTS = GATE / "requests/remote_gate_requests_wan_aether.jsonl.gz"
GAP_RECEIPT = Path("/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1/.windowfix_v2_gap_remote_receipts/e73413ab7dfea8e861092eec2d940affe8c4d7bf06409555413eaab4a47ce41e/remote_completion.json")
SOURCE = ROOT / "runtime/main_v2_probe_v2_r7_single_v3/source"
BASE_QUEUE = BASE / "valid7_user_scope_20260922_r4/queue_552_with_aether.jsonl"
LAYERS = (0, 5, 10, 15, 20, 25, 29)
PREFIXES = (8, 12, 16, 24)
PROBES = ("streaming_depth", "streaming_mask", "view_consistency", "ego_belief_v2", "action_dynamics", "path_integration")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def canonical(value) -> str:
    return sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == data:
            return
        raise RuntimeError(f"refusing to replace {path}")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)


def key_from_expected(e):
    return (e["vfm"], e["source"], e["scene_id"], int(e["feature_layer"]),
            int(e["timestep"]), int(e["window_id"]), int(e["prefix"]))


def owner_from_gate(row):
    e, ev = row["expected"], row["evidence"]
    k = key_from_expected(e)
    return k, dict(zip(("vfm", "source", "scene_id", "layer", "timestep", "window_id", "prefix"), k),
        unit_id=row["unit_id"], source_uri=row["source_uri"], source_index_uri=row["source_index_uri"],
        payload_sha256=ev["sidecar"].get("declared_payload_sha256_unverified"),
        payload_size_bytes=int(ev["safetensors"]["object_size_bytes"]),
        payload_header_sha256=ev["safetensors"]["header_sha256"],
        source_index_sha256=ev["source_index"]["sha256"],
        tensor_shape_json=json.dumps(e["expected_shape"]), tensor_dtype=e["expected_dtype"],
        raw_ids_json=json.dumps(e["raw_ids"]), input_length=int(e["preprocess_identity"]["num_frames"]),
        valid_length=len(e["raw_ids"]), pad_mode=e["preprocess_identity"]["temporal_padding"],
        origin="gate_pass", status=row["status"])


def build_overlay() -> dict:
    requested = {}
    with gzip.open(REQUESTS, "rt") as handle:
        for line in handle:
            row = json.loads(line); e = row["expected"]
            if e["vfm"] == "wan":
                requested[key_from_expected(e)] = row
    assert len(requested) == 76188

    old = {}
    for path in sorted((GATE / "gate").glob("gate-*/receipts.jsonl.gz")):
        with gzip.open(path, "rt") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("status") != "REFERENCE_SOURCE_READ_ONLY" or row.get("expected", {}).get("vfm") != "wan":
                    continue
                k, owner = owner_from_gate(row)
                assert k in requested and k not in old
                old[k] = owner
    assert len(old) == 40143

    receipt_raw = GAP_RECEIPT.read_bytes(); receipt = json.loads(receipt_raw)
    assert receipt["status"] == "PASS_REMOTE_VERIFIED"
    content = dict(receipt); assert content.pop("content_sha256") == canonical(content)
    objects = receipt["objects"]
    new = {}
    root = "aoss://baiqiao/InsScene-15K/FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1/wan/scannetpp/"
    for rel, obj in objects.items():
        identity = obj["canonical_unit_identity"]
        if identity.get("kind") != "streaming_feature":
            continue
        g = identity["gap_record_key"]
        k = ("wan", "scannetpp", g["scene"], int(identity["layer"]), 749,
             int(g["window_id"]), int(g["prefix"]))
        assert k in requested and k not in old and k not in new
        req = requested[k]; e = req["expected"]
        index_rel = f"{g['scene']}/prefix_index.npy"; index_obj = objects[index_rel]
        assert rel == req["source_uri"].split("/wan/scannetpp/", 1)[1]
        new[k] = dict(zip(("vfm", "source", "scene_id", "layer", "timestep", "window_id", "prefix"), k),
            unit_id=req["unit_id"], source_uri=root + rel, source_index_uri=root + index_rel,
            payload_sha256=obj["payload_sha256"], payload_size_bytes=int(obj["payload_size_bytes"]),
            payload_header_sha256="GAP_RECEIPT_FULL_SHA", source_index_sha256=index_obj["payload_sha256"],
            tensor_shape_json=json.dumps(e["expected_shape"]), tensor_dtype=e["expected_dtype"],
            raw_ids_json=json.dumps(e["raw_ids"]), input_length=int(e["preprocess_identity"]["num_frames"]),
            valid_length=len(e["raw_ids"]), pad_mode=e["preprocess_identity"]["temporal_padding"],
            origin="gap_remote_completion_exact_worklist", status="PASS_REMOTE_VERIFIED")
    assert len(new) == 36045
    owners = dict(old)
    owners.update(new)
    assert set(owners) == set(requested) and len(owners) == 76188
    assert Counter((k[3], k[6]) for k in owners) == Counter({(l, p): 2721 for l in LAYERS for p in PREFIXES})

    overlay_dir = OUT / "overlay"; overlay_dir.mkdir(parents=True, exist_ok=True)
    template_stream = BASE / "owner_overlay_r7_receipt_union/streaming_owners.sqlite3"
    db0 = sqlite3.connect(template_stream.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    stream_schema = db0.execute("SELECT sql FROM sqlite_master WHERE name='streaming_owners'").fetchone()[0]; db0.close()
    stream_path = overlay_dir / "streaming_owners.sqlite3"
    if stream_path.exists(): raise RuntimeError(f"refusing existing {stream_path}")
    db = sqlite3.connect(stream_path); db.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
    db.execute("INSERT INTO metadata VALUES('status','READY')"); db.execute(stream_schema)
    cols = [r[1] for r in db.execute("PRAGMA table_info(streaming_owners)")]
    db.executemany(f"INSERT INTO streaming_owners VALUES({','.join('?' for _ in cols)})",
                   ([o[c] for c in cols] for _, o in sorted(owners.items())))
    db.commit(); db.close()

    target_path = overlay_dir / "target_owners.sqlite3"
    if target_path.exists(): raise RuntimeError(f"refusing existing {target_path}")
    validation_target = BASE / "probe552_validation_stream_ab128_20260923_r1/overlay/target_owners.sqlite3"
    source_db = sqlite3.connect(validation_target.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    target_schema = source_db.execute("SELECT sql FROM sqlite_master WHERE name='target_owners'").fetchone()[0]; source_db.close()
    db = sqlite3.connect(target_path); db.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
    db.execute("INSERT INTO metadata VALUES('status','READY_STREAMING_ONLY')"); db.execute(target_schema); db.commit(); db.close()

    manifest = dict(schema="CANONICAL_V2_WAN_STREAMING_OWNER_OVERLAY_V1", status="READY_STREAMING_ONLY",
        fail_closed=True, created_utc=datetime.now(timezone.utc).isoformat(), models=["wan"],
        scope="WAN_7_LAYERS_4_PREFIXES_EXACT_2721_WINDOWS",
        streaming_database=dict(path=str(stream_path.resolve()), sha256=sha_file(stream_path), rows=len(owners)),
        target_database=dict(path=str(target_path.resolve()), sha256=sha_file(target_path), rows=0),
        counts=dict(required=76188, old_gate_verified=len(old), gap_receipt_verified=len(new), holes=0, duplicates=0),
        requests=dict(path=str(REQUESTS.resolve()), sha256=sha_file(REQUESTS)),
        gap_receipt=dict(path=str(GAP_RECEIPT), sha256=sha_bytes(receipt_raw)))
    manifest_path = overlay_dir / "manifest.json"
    write_new(manifest_path, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    return manifest


def build_queue() -> list:
    base_rows = [json.loads(line) for line in BASE_QUEUE.read_text().splitlines() if line]
    assert len(base_rows) == 552
    rows = []
    batch = {"streaming_depth": 1, "streaming_mask": 1, "view_consistency": 4,
             "ego_belief_v2": 1, "action_dynamics": 4, "path_integration": 4}
    for layer in LAYERS:
        for prefix in PREFIXES:
            for probe in PROBES:
                template = next(r for r in base_rows if r["model"] == "dino" and r["prefix"] == prefix and r["probe"] == probe)
                row = copy.deepcopy(template)
                config = f"inscene15k_streaming/{probe}_wan_v1"
                cfg = SOURCE / "configs/experiment" / (config + ".yaml")
                eid = f"user0923_{probe}_wan_l{layer}_p{prefix}_s42"
                row.update(model="wan", layer=layer, disk_layer=layer, timestep=749,
                    unique_key=f"wan__scannetpp__t749__l{layer}__p{prefix}__{probe}__seed42",
                    experiment_id=eid, experiment_ids=[eid], source_registry_rows=0,
                    experiment_config=config, experiment_config_sha256=sha_file(cfg),
                    config_provenance="wan_formal_config_plus_frozen_probe552_protocol",
                    old_checkpoint_path=None, old_run_artifact_path=None,
                    old_training_status="NEW_WAN_SCOPE_NOT_SUBMITTED", queue_ordinal=552 + len(rows),
                    launch_status="WAITING_VALIDATION" if not row["requires_target"] else "TARGET_PENDING")
                replacements = {"vfm_name": "wan", "video_channels": "1536", "feature_layer": str(layer),
                                "feat_postfix": f"_t749_layer{layer}",
                                "data.data_module.batch_size_per_device": str(batch[probe]),
                                "data.data_module.batch_size_per_device_val": str(batch[probe])}
                row["scientific_overrides"] = [
                    k + "=" + replacements[k] if (k := value.split("=", 1)[0]) in replacements else value
                    for value in row["scientific_overrides"]]
                rows.append(row)
    assert len(rows) == 168 and len({r["unique_key"] for r in rows}) == 168
    payload = b"".join((json.dumps(r, sort_keys=True) + "\n").encode() for r in rows)
    write_new(OUT / "queue_wan_168.jsonl", payload)
    write_new(OUT / "queue_wan_streaming_ab_112.jsonl", b"".join((json.dumps(r, sort_keys=True)+"\n").encode() for r in rows if not r["requires_target"]))
    write_new(OUT / "queue_720_with_wan.jsonl", BASE_QUEUE.read_bytes() + payload)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    overlay = build_overlay(); rows = build_queue()
    status = dict(schema="WAN_PROBING_168_STATUS_V1", updated_utc=datetime.now(timezone.utc).isoformat(),
        total=168, streaming_only_ready_for_validation=112, target_dependent_pending=56,
        by_stage={"WAITING_VALIDATION": 112, "TARGET_PENDING": 56},
        by_probe=dict(Counter(r["probe"] for r in rows)), overlay=overlay,
        queues={"wan168": str((OUT/'queue_wan_168.jsonl').resolve()),
                "streaming_ab112": str((OUT/'queue_wan_streaming_ab_112.jsonl').resolve()),
                "combined720": str((OUT/'queue_720_with_wan.jsonl').resolve())})
    write_new(OUT / "status.json", (json.dumps(status, indent=2, sort_keys=True)+"\n").encode())
    print(json.dumps({"total": 168, "streaming_ready": 112, "target_pending": 56,
                      "streaming_owners": overlay["streaming_database"]["rows"]}, sort_keys=True))


if __name__ == "__main__":
    main()
