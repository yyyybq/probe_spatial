"""Add Aether to the frozen 528 scope without changing its bytes or submitting jobs.

Coverage is taken only from the known Aether extraction receipts; an absent
receipt is NOT proof that the feature must be extracted again.
"""
import copy
import csv
import gzip
import hashlib
import io
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/main_v2_probe_v2_816_20260917"
OUT = BASE / "valid7_user_scope_20260922_r4"
DATA = Path("/mnt/umm/users/yinbaiqiao/InsScene-15K")
PARENT_SHA = "ebc496eb0587cda372ccf29a7c5244ef454096151f73c8eed9511967ded1902d"
JOBS = {2: "pt-0zal0qw4", 3: "pt-eav5rjmv", 4: "pt-f2xaf0tg",
        5: "pt-eh2id3e7", 6: "pt-1k63k7lp", 7: "pt-81b5sjqq"}
PROBES = ("streaming_depth", "streaming_mask", "view_consistency",
          "ego_belief_v2", "action_dynamics", "path_integration")

def sha(data):
    return hashlib.sha256(data).hexdigest()

def canonical(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())

def load(path):
    return json.loads(Path(path).read_text())

def write_new(path, text):
    if path.exists():
        if path.read_bytes() == text.encode():
            return
        raise RuntimeError(f"refusing to change existing artifact: {path}")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "w") as out:
        out.write(text)

def jsonl(rows):
    return "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)

def main():
    base_path = OUT / "queue_528.jsonl"
    base_bytes = base_path.read_bytes()
    assert sha(base_bytes) == load(OUT / "manifest.json")["queue"]["sha256"]
    rows = [json.loads(line) for line in base_bytes.splitlines()]
    assert len(rows) == 528
    stream_required, target_required, manifests = {}, set(), {}
    parent_metadata = load(BASE / "execution_manifest_r3_receipt_union/manifest.json")
    wanted_manifest_shas = {r["protocol"]["canonical_manifest_sha256"] for r in rows}
    manifest_bindings = {int(m["prefix"]): m for m in parent_metadata["availability_classes"]
                         if m["sha256"] in wanted_manifest_shas}
    assert len(manifest_bindings) == 4
    for prefix in (8, 12, 16, 24):
        path = Path(manifest_bindings[prefix]["path"])
        raw = path.read_bytes()
        assert sha(raw) == manifest_bindings[prefix]["sha256"]
        manifests[str(prefix)] = {"path": str(path), "sha256": sha(raw)}
        for line in raw.splitlines():
            item = json.loads(line)
            key = (item["scene_id"], int(item["streaming_window_id"]), prefix)
            assert key not in stream_required
            stream_required[key] = sha(json.dumps(item["streaming_indices"], separators=(",", ":")).encode())
            window = item["streaming_window_indices"]
            target_required.update((item["scene_id"], int(window[prefix - 1 + h])) for h in (0, 1, 2, 4))
    assert len(stream_required) == 10884 and len(target_required) == 37677

    covered_stream, covered_target, receipt_evidence = set(), set(), []
    for mode in ("STREAMING", "TARGET"):
        logs = DATA / f"FEAT_{mode}_CANONICAL_V2_GAP_DELTA_V1/.windowfix_v2_gap_logs/aether"
        receipt_paths = {load(p)["remote_completion_receipt"] for p in logs.glob("*/*/receipt.json")}
        for name in sorted(receipt_paths):
            path = Path(name)
            if not path.is_file():
                continue
            raw = path.read_bytes()
            value = json.loads(raw)
            lineage = value["lineage"]
            assert lineage["vfm"] == "aether" and lineage["timestep"] == 749
            assert lineage["canonical_source_sha256"] == PARENT_SHA
            assert value["status"] == "PASS_REMOTE_VERIFIED"
            content = dict(value)
            assert content.pop("content_sha256") == canonical(content)
            remote = value["remote"]["base_uri"]
            assert remote.endswith(f"/FEAT_{mode}_CANONICAL_V2_GAP_DELTA_V1/aether/scannetpp/")
            receipt_evidence.append({"path": name, "sha256": sha(raw), "remote_base": remote,
                                     "objects": len(value["objects"])})
            for relative, obj in value["objects"].items():
                identity = obj["canonical_unit_identity"]
                assert canonical(identity) == obj["canonical_unit_identity_sha256"]
                if identity.get("layer") != 1:
                    continue
                if identity.get("kind") == "streaming_feature":
                    k = identity["gap_record_key"]
                    key = (k["scene"], int(k["window_id"]), int(k["prefix"]))
                    if key in stream_required:
                        assert identity["gap_record_raw_ids_sha256"] == stream_required[key], key
                        covered_stream.add(key)
                elif identity.get("kind") == "target_delta_layer" and identity.get("payload_role") == "feature":
                    covered_target.update((identity["scene"], int(f)) for f in identity["target_indices"])

    # The remaining 146 keys were already read-verified before the positive
    # gap worklist was built. They were deliberately NOT re-extracted.
    gate_root = ROOT / "artifacts/wan_aether_streaming_official_r1/overlay"
    gate_manifest = load(gate_root / "manifest.json")
    gate_routes = ROOT / gate_manifest["route_file"]["path"]
    assert sha(gate_routes.read_bytes()) == gate_manifest["route_file"]["sha256"]
    previously_verified = set()
    with gzip.open(gate_routes, "rt") as source:
        for line in source:
            row = json.loads(line)
            k = row["key"]
            if k[:3] != ["streaming", "aether", "scannetpp"]:
                continue
            assert k[4:6] == [1, 749] and row["status"] == "REFERENCE_SOURCE_READ_ONLY"
            key = (k[3], int(k[6]), int(k[7]))
            assert row["source_uri"] == (
                f"aoss://baiqiao/InsScene-15K/FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1/aether/scannetpp/"
                f"{k[3]}/window_{k[6]:04d}/prefix_{k[7]-1:06d}/feature_t749_layer1.sft")
            if key in stream_required:
                previously_verified.add(key)
    assert len(previously_verified) == 146
    stream_all = covered_stream | previously_verified
    partition_root = ROOT / "runtime/main_v2_aether_release_v2/production_single_gpu_r1/plans/aether_target"
    owner = {}
    for part in load(partition_root / "partition_index.json")["partitions"]:
        contents = Path(part["scenes_path"]).read_bytes()
        assert sha(contents) == part["scenes_sha256"]
        scene_data = json.loads(contents)
        for scene in scene_data["scenes"]:
            assert scene not in owner
            owner[scene] = part["index"]
    pending = []
    for scene, frame in sorted(target_required - covered_target):
        part = owner[scene]
        pending.append({"model": "aether", "layer": 1, "timestep": 749,
                        "mode": "target_isolated", "scene_id": scene, "frame_index": frame,
                        "partition": part, "existing_job_id": JOBS.get(part),
                        "status": "NOT_YET_IN_REMOTE_RECEIPT_EXISTING_EXTRACTION_OWNER",
                        "submit_new_extraction": False})

    aether = []
    for prefix in (8, 12, 16, 24):
        for probe in PROBES:
            template = next(r for r in rows if r["model"] == "dino" and r["prefix"] == prefix and r["probe"] == probe)
            row = copy.deepcopy(template)
            config_probe = "streaming_depth" if probe == "streaming_mask" else probe
            config = f"inscene15k_streaming/{config_probe}_aether_{'h124_v1' if probe == 'action_dynamics' else 'v1'}"
            config_path = ROOT / "configs/experiment" / (config + ".yaml")
            eid = f"user0923_{probe}_aether_l1_p{prefix}_s42"
            row.update(model="aether", layer=1, disk_layer=1, timestep=749,
                       unique_key=f"aether__scannetpp__t749__l1__p{prefix}__{probe}__seed42",
                       experiment_id=eid, experiment_ids=[eid], source_registry_rows=0,
                       experiment_config=config, experiment_config_sha256=sha(config_path.read_bytes()),
                       config_provenance="user0923_aether_existing_formal_config_and_528_protocol",
                       old_checkpoint_path=None, old_run_artifact_path=None,
                       old_training_status="NEW_V2_SCOPE_NOT_SUBMITTED",
                       queue_ordinal=528 + len(aether),
                       launch_status="PLANNED_NOT_SUBMITTED_LOADER_CANARY_REQUIRED")
            replace = {"vfm_name": "aether", "video_channels": "3072", "feature_layer": "1",
                       "feat_postfix": "_t749_layer1"}
            row["scientific_overrides"] = [
                k + "=" + replace[k] if (k := v.split("=", 1)[0]) in replace else v
                for v in row["scientific_overrides"]]
            assert row["protocol"]["canonical_manifest_sha256"] == manifests[str(prefix)]["sha256"]
            aether.append(row)
    assert len(aether) == 24
    assert len({r["unique_key"] for r in rows + aether}) == 552
    combined = base_bytes.decode() + jsonl(aether)
    assert combined.encode()[:len(base_bytes)] == base_bytes
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "aether", "layer": 1, "timestep": 749,
        "base_queue": {"path": str(base_path), "sha256": sha(base_bytes), "unchanged": True},
        "combined_queue": {"path": str(OUT / "queue_552_with_aether.jsonl"), "sha256": sha(combined.encode()), "rows": 552},
        "aether_execution_units": 24, "aether_scientific_outputs": 28,
        "counts_by_probe": dict(Counter(r["probe"] for r in aether)),
        "frozen_manifests": manifests,
        "streaming": {"required": 10884, "remote_receipt_covered": len(covered_stream),
                      "previously_remote_read_verified": len(previously_verified),
                      "total_verified": len(stream_all),
                      "missing": len(set(stream_required) - stream_all),
                      "previous_read_verification": {"path": str(gate_routes), "sha256": gate_manifest["route_file"]["sha256"]},
                      "ordered_frame_id_hashes_checked": len(covered_stream)},
        "target": {"required_unique_anchor_and_h124_frames": 37677,
                   "remote_receipt_covered": len(target_required & covered_target),
                   "not_yet_in_remote_receipts": len(pending),
                   "pending_by_partition": dict(Counter(r["partition"] for r in pending)),
                   "note": "Pending receipts are NOT a re-extraction authorization; existing jobs own these IDs. Local/in-flight data may already exist."},
        "receipt_evidence": receipt_evidence,
        "submission_performed": False,
        "readiness": "FEATURE_COVERAGE_REPORT_ONLY_NOT_LOADER_OR_TRAINING_CANARY",
    }
    csv_out = io.StringIO()
    fields = ["unique_key", "model", "layer", "timestep", "prefix", "probe", "experiment_config", "requires_target"]
    writer = csv.DictWriter(csv_out, fieldnames=fields, extrasaction="ignore")
    writer.writeheader(); writer.writerows(aether)
    for name, text in {
        "queue_aether_24.jsonl": jsonl(aether), "queue_552_with_aether.jsonl": combined,
        "aether_experiments_24.csv": csv_out.getvalue(),
        "aether_target_pending_existing_jobs_r2.jsonl": jsonl(pending),
        "aether_feature_status_20260923_r2.json": json.dumps(summary, indent=2, sort_keys=True) + "\n",
    }.items():
        write_new(OUT / name, text)
    print(json.dumps({k: summary[k] for k in ["combined_queue", "streaming", "target"]}, indent=2))

if __name__ == "__main__":
    main()
