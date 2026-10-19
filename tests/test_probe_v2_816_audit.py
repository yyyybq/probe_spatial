"""Small synthetic checks for the canonical-v2 816 audit-only inventories."""

import csv
import gzip
import hashlib
import json

import pytest

from scripts.build_probe_overlay_catalog import DISK_LAYER_OVERRIDES, canonical_key
from scripts.filter_probe_gate_requests import select
from scripts.build_probe_owner_overlay import (
    gap_streaming_key,
    main as build_overlay,
    receipt_key,
    to_aoss_uri,
)
from scripts.resolve_probe816_holes import classify_row
from scripts.owner_catalog import (
    OwnerCatalogError,
    load_owner_catalog,
    resolve_owner,
    streaming_key,
)


def test_logical_layer_mapping_is_explicit():
    assert DISK_LAYER_OVERRIDES == {("lingbot", 48): 47, ("minimax_h3", 50): 49}
    assert canonical_key({
        "vfm": "lingbot", "source": "scannetpp", "scene_id": "scene1",
        "feature_layer": 47, "timestep": 250, "window_id": 0, "prefix": 8,
    }) == ("streaming", "lingbot", "scannetpp", "scene1", 47, 250, 0, 8)


def test_gate_subset_filters_model_and_disk_layer(tmp_path):
    experiments = tmp_path / "experiments.csv"
    with experiments.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["experiment_id", "model", "layer"])
        writer.writeheader()
        for index in range(816):
            model, layer = (("lingbot", "48") if index % 2 else ("minimax_h3", "50"))
            writer.writerow({"experiment_id": f"exp-{index}", "model": model, "layer": layer})
    requests = tmp_path / "requests.jsonl.gz"
    with gzip.open(requests, "wt", encoding="utf-8") as handle:
        for index, (model, layer) in enumerate((
            ("lingbot", 47), ("minimax_h3", 49), ("lingbot", 48), ("wan", 0)
        )):
            handle.write(json.dumps({
                "unit_id": f"unit-{index}",
                "expected": {"vfm": model, "feature_layer": layer},
            }) + "\n")
    namespace_lock = tmp_path / "namespace.json"
    namespace_lock.write_text(json.dumps({
        "canonical_source": {"sha256": "a" * 64},
        "canonical_universe_sha256": "b" * 64,
    }), encoding="utf-8")
    source_summary = tmp_path / "summary.json"
    source_summary.write_text(json.dumps({
        "canonical_namespace_lock": str(namespace_lock),
        "canonical_parent_source_sha256": "a" * 64,
        "canonical_universe_sha256": "b" * 64,
        "canonical_protocol_lineage_file_sha256": "c" * 64,
        "canonical_protocol_sha256": "d" * 64,
    }), encoding="utf-8")
    source_manifest = tmp_path / "catalog_manifest.json"
    source_manifest.write_text(json.dumps({
        "canonical_parent_source_sha256": "a" * 64,
        "canonical_universe_sha256": "b" * 64,
        "outputs": [
            {"file": requests.name, "sha256": hashlib.sha256(requests.read_bytes()).hexdigest(), "size_bytes": requests.stat().st_size},
            {"file": source_summary.name, "sha256": hashlib.sha256(source_summary.read_bytes()).hexdigest(), "size_bytes": source_summary.stat().st_size},
        ],
    }), encoding="utf-8")
    output = tmp_path / "output"
    summary = select(experiments, requests, output, source_manifest, source_summary, namespace_lock)
    assert summary["status"] == "REVIEW_ONLY_NOT_EXECUTED"
    assert summary["output"]["rows"] == 2
    assert summary["counts_by_model_disk_layer"] == {"lingbot:L47": 1, "minimax_h3:L49": 1}
    payload = output / "remote_gate_requests_816.jsonl.gz"
    assert hashlib.sha256(payload.read_bytes()).hexdigest() == summary["output"]["sha256"]
    with gzip.open(payload, "rt", encoding="utf-8") as handle:
        assert [json.loads(line)["unit_id"] for line in handle] == ["unit-0", "unit-1"]
    catalog_summary = json.loads((output / "catalog_summary.json").read_text())
    assert catalog_summary["counts"]["status"]["PENDING_REMOTE_GATE"] == 2
    manifest = json.loads((output / "catalog_manifest.json").read_text())
    assert {item["file"] for item in manifest["outputs"]} == {
        "remote_gate_requests_816.jsonl.gz", "selection_summary.json", "catalog_summary.json"
    }
    with pytest.raises(ValueError, match="already exists"):
        select(experiments, requests, output, source_manifest, source_summary, namespace_lock)


def test_gate_subset_rejects_broken_source_lineage(tmp_path):
    experiments = tmp_path / "experiments.csv"
    experiments.write_text("experiment_id,model,layer\n" + "\n".join(
        f"exp-{index},lingbot,48" for index in range(816)
    ) + "\n", encoding="utf-8")
    requests = tmp_path / "requests.jsonl.gz"
    with gzip.open(requests, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "unit_id": "unit-0", "expected": {"vfm": "lingbot", "feature_layer": 47}
        }) + "\n")
    namespace_lock = tmp_path / "namespace.json"
    namespace_lock.write_text(json.dumps({
        "canonical_source": {"sha256": "a" * 64},
        "canonical_universe_sha256": "b" * 64,
    }), encoding="utf-8")
    source_summary = tmp_path / "summary.json"
    source_summary.write_text(json.dumps({
        "canonical_namespace_lock": str(namespace_lock),
        "canonical_parent_source_sha256": "f" * 64,
        "canonical_universe_sha256": "b" * 64,
    }), encoding="utf-8")
    source_manifest = tmp_path / "catalog_manifest.json"
    source_manifest.write_text(json.dumps({
        "canonical_parent_source_sha256": "a" * 64,
        "canonical_universe_sha256": "b" * 64,
        "outputs": [
            {"file": requests.name, "sha256": hashlib.sha256(requests.read_bytes()).hexdigest(), "size_bytes": requests.stat().st_size},
            {"file": source_summary.name, "sha256": hashlib.sha256(source_summary.read_bytes()).hexdigest(), "size_bytes": source_summary.stat().st_size},
        ],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="source summary canonical source lineage mismatch"):
        select(experiments, requests, tmp_path / "output", source_manifest, source_summary, namespace_lock)


def _write_overlay(tmp_path, *, status="READY", audit_only=False, route_sha=None, extra_route=None):
    routes = tmp_path / "routes.jsonl.gz"
    key = list(streaming_key(
        vfm="lingbot", source="scannetpp", scene_id="scene1",
        feature_layer=47, timestep=None, window_id=0, prefix=8,
    ))
    rows = [{
        "key": key,
        "status": "REFERENCE_SOURCE_READ_ONLY",
        "source_uri": "aoss://bucket/lingbot/scene1/feature.sft",
        "unit_id": "cv2_lingbot_t0_l47_p8:scene1_w000_p8",
    }]
    if extra_route is not None:
        rows.append(extra_route)
    with gzip.open(routes, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    manifest = tmp_path / "catalog_manifest.json"
    payload = {
        "schema": "canonical_v2_probe_owner_overlay_v1",
        "status": status,
        "fail_closed": True,
        "audit_only": audit_only,
        "route_file": {
            "path": str(routes.resolve()),
            "sha256": route_sha or hashlib.sha256(routes.read_bytes()).hexdigest(),
            "rows": len(rows),
        },
        "readiness": {"complete_unit_coverage_verified": status == "READY", "reason": "test"},
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest, key


def test_owner_catalog_rejects_not_ready(tmp_path):
    manifest, _ = _write_overlay(tmp_path, status="NOT_READY", audit_only=True)
    with pytest.raises(OwnerCatalogError, match="not READY"):
        load_owner_catalog(manifest)


def test_owner_catalog_rejects_route_sha_mismatch_and_unverified(tmp_path):
    manifest, _ = _write_overlay(tmp_path, route_sha="0" * 64)
    with pytest.raises(OwnerCatalogError, match="route SHA mismatch"):
        load_owner_catalog(manifest)
    extra = {
        "key": ["streaming", "lingbot", "scannetpp", "scene2", 47, None, 0, 8],
        "status": "REEXTRACT_UNVERIFIED",
        "source_uri": "aoss://bucket/missing.sft",
    }
    unverified = tmp_path / "unverified"
    unverified.mkdir()
    manifest, _ = _write_overlay(unverified, extra_route=extra)
    with pytest.raises(OwnerCatalogError, match="not an eligible unique owner"):
        load_owner_catalog(manifest)


def test_owner_catalog_resolves_ready_owner_and_misses_fail_closed(tmp_path):
    manifest, key = _write_overlay(tmp_path)
    catalog = load_owner_catalog(manifest)
    assert resolve_owner(catalog, tuple(key), label="streaming") == (
        "aoss://bucket/lingbot/scene1/feature.sft"
    )
    with pytest.raises(OwnerCatalogError, match="no unique owner"):
        resolve_owner(
            catalog,
            streaming_key(
                vfm="lingbot", source="scannetpp", scene_id="missing",
                feature_layer=47, timestep=None, window_id=0, prefix=8,
            ),
            label="streaming",
        )


def test_receipt_key_parses_qwen_minus_layer_and_no_timestep_gap():
    key = receipt_key({
        "unit_id": "cv2_qwen2_5_vl_t749_lm1_p8_d9289dc71d:07f5b601ee_w000_p8",
        "status": "REEXTRACT_UNVERIFIED",
    })
    assert key == streaming_key(
        vfm="qwen2_5_vl", source="scannetpp", scene_id="07f5b601ee",
        feature_layer=-1, timestep=749, window_id=0, prefix=8,
    )
    gap_key = gap_streaming_key({
        "mode": "streaming_prefix",
        "vfm": "f3r",
        "root": "s3://baiqiao.aoss-internal.example/InsScene-15K/FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1/f3r/scannetpp",
        "relative_path": "0fa1fdd47b/window_0000/prefix_000007/feature_l24.sft",
        "payload_sha256": "ab" * 32,
        "identity": {
            "kind": "streaming_feature",
            "layer": 24,
            "gap_record_key": {"prefix": 8, "scene": "0fa1fdd47b", "window_id": 0},
        },
    })
    assert gap_key == streaming_key(
        vfm="f3r", source="scannetpp", scene_id="0fa1fdd47b",
        feature_layer=24, timestep=749, window_id=0, prefix=8,
    )
    assert to_aoss_uri(
        "s3://baiqiao.aoss-internal.example/InsScene-15K/FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1/f3r/scannetpp",
        "0fa1fdd47b/window_0000/prefix_000007/feature_l24.sft",
    ) == (
        "aoss://baiqiao/InsScene-15K/FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1/"
        "f3r/scannetpp/0fa1fdd47b/window_0000/prefix_000007/feature_l24.sft"
    )


def _write_gate_shard(root, *, index, rows, request_sha="a" * 64, shard_count=1):
    shard = root / f"gate-{index:04d}-of-{shard_count:04d}"
    shard.mkdir(parents=True)
    with gzip.open(shard / "receipts.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    (shard / "summary.json").write_text(json.dumps({
        "requests_sha256": request_sha,
        "shard_count": shard_count,
        "processed": len(rows),
        "status_counts": {},
        "receipts_sha256": "b" * 64,
    }), encoding="utf-8")


def test_overlay_pass_plus_gap_ready_only_when_every_request_has_owner(tmp_path, monkeypatch):
    gate = tmp_path / "gate"
    pass_row = {
        "unit_id": "cv2_dino_t749_l0_p8_c2ca877fed:scenea_w000_p8",
        "status": "REFERENCE_SOURCE_READ_ONLY",
        "source_uri": "aoss://baiqiao/InsScene-15K/FEAT_STREAMING_BY_PREFIX/dino/scenea/feature.sft",
        "reason_codes": ["LEGACY_EXACT_REFERENCE_V1_REMOTE_RANGE_GATE_PASS"],
    }
    fail_row = {
        "unit_id": "cv2_f3r_t749_l24_p8_e4fcdf3d95:sceneb_w000_p8",
        "status": "REEXTRACT_UNVERIFIED",
        "source_uri": "aoss://baiqiao/InsScene-15K/FEAT_STREAMING/f3r/sceneb/feature_l24.sft",
        "reason_codes": ["CHECK_FAILED:sidecar_model"],
    }
    _write_gate_shard(gate, index=0, rows=[pass_row, fail_row])
    gap = tmp_path / "gap.jsonl.gz"
    with gzip.open(gap, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "mode": "streaming_prefix",
            "vfm": "f3r",
            "root": "s3://host/InsScene-15K/FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1/f3r/scannetpp",
            "relative_path": "sceneb/window_0000/prefix_000007/feature_l24.sft",
            "payload_sha256": "cd" * 32,
            "identity_sha256": "ef" * 32,
            "identity": {
                "kind": "streaming_feature",
                "layer": 24,
                "gap_record_key": {"prefix": 8, "scene": "sceneb", "window_id": 0},
            },
        }) + "\n")
        handle.write(json.dumps({
            "mode": "target_isolated",
            "vfm": "f3r",
            "relative_path": "ignored.sft",
        }) + "\n")
    out = tmp_path / "overlay"
    monkeypatch.setattr("sys.argv", [
        "build_probe_owner_overlay.py",
        "--gate-root", str(gate),
        "--gap-index", str(gap),
        "--output-dir", str(out),
        "--expected-request-sha256", "a" * 64,
        "--expected-shards", "1",
        "--expected-requests", "2",
    ])
    assert build_overlay() == 0
    manifest = json.loads((out / "catalog_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "READY"
    assert manifest["audit_only"] is False
    assert manifest["counts"]["pass_owners"] == 1
    assert manifest["counts"]["gap_filled_owners"] == 1
    catalog = load_owner_catalog(out / "catalog_manifest.json")
    assert resolve_owner(catalog, receipt_key(fail_row), label="streaming").endswith(
        "FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1/f3r/scannetpp/sceneb/window_0000/prefix_000007/feature_l24.sft"
    )

    incomplete = tmp_path / "overlay_incomplete"
    empty_gap = tmp_path / "empty_gap.jsonl.gz"
    with gzip.open(empty_gap, "wt", encoding="utf-8") as handle:
        handle.write("{}\n")
    monkeypatch.setattr("sys.argv", [
        "build_probe_owner_overlay.py",
        "--gate-root", str(gate),
        "--gap-index", str(empty_gap),
        "--output-dir", str(incomplete),
        "--expected-request-sha256", "a" * 64,
        "--expected-shards", "1",
        "--expected-requests", "2",
    ])
    assert build_overlay() == 2
    not_ready = json.loads((incomplete / "catalog_manifest.json").read_text(encoding="utf-8"))
    assert not_ready["status"] == "NOT_READY"
    assert not_ready["counts"]["holes"] == 1
    with pytest.raises(OwnerCatalogError, match="not READY"):
        load_owner_catalog(incomplete / "catalog_manifest.json")


def test_hole_classifier_separates_metadata_from_missing_and_wrong_tensor():
    assert classify_row({
        "unit_id": "cv2_f3r_t749_l24_p8_x:scene_w000_p8",
        "reason_codes": ["CHECK_FAILED:sidecar_model"],
        "checks": {"sidecar_model": False, "header_shape_equals_expected": True},
    }) == "metadata_recoverable"
    assert classify_row({
        "unit_id": "cv2_minimax_h3_t250_l10_p8_x:scene_w000_p8",
        "reason_codes": ["CHECK_FAILED:header_shape_equals_expected"],
        "checks": {"header_shape_equals_expected": False, "sidecar_model": True},
    }) == "unusable_wrong_tensor"
    assert classify_row({
        "unit_id": "cv2_videomae_t749_l0_p24_x:scene_w000_p24",
        "reason_codes": ["DEFINITIVE_GATE_FAILURE:FileNotFoundError"],
    }) == "genuinely_missing"
