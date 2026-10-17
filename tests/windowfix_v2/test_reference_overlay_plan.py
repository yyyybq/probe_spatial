from __future__ import annotations

import importlib
import json
import struct
from pathlib import Path

import pytest

from analyses.windowfix_v2.build_hybrid_reuse_plan import CLASSIFIER_STATUSES, PlanError
from analyses.windowfix_v2.build_reference_overlay_plan import (
    REFERENCE_NAMESPACES,
    build,
    write_outputs,
)
from scripts.windowfix_v2.dry_run_reference_overlay import validate


PROTOCOL = "inscene_streaming_canonical_v2_obs28_h124"


def _write_safetensors(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = json.dumps(
        {"feat": {"dtype": "F16", "shape": [2, 3], "data_offsets": [0, 12]}},
        separators=(",", ":"),
    ).encode()
    header += b" " * ((8 - len(header) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(12))


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source = tmp_path / "legacy" / "feature_t749_layer20.sft"
    _write_safetensors(source)
    classification = tmp_path / "classification"
    classification.mkdir()
    exact = {
        "status": "REUSE_EXACT",
        "reason_codes": ["legacy_exact_reference_v1_identity_checks_passed"],
        "spec_id": "wan_t749_l20_p8",
        "vfm": "wan",
        "model_id": "model/a",
        "layer": 20,
        "timestep": 749,
        "tensor_key": "feat",
        "prefix": 8,
        "source": "scannetpp",
        "scene_id": "scene",
        "sample_id": "scene_w000_p8",
        "window_id": 0,
        "raw_ids": list(range(8)),
        "feature_path": str(source),
        "source_payload_sha256": None,
        "declared_source_payload_sha256": "0" * 64,
        "payload_sha256_verified": False,
        "source_size_bytes": source.stat().st_size,
        "legacy_root_id": "old",
        "expected_sidecar_identity": {
            "prompt": ".",
            "resize": [480, 832],
            "frame_index_schema": "image_mask_intersection_v1",
        },
    }
    for status in CLASSIFIER_STATUSES:
        rows = [exact] if status == "REUSE_EXACT" else []
        (classification / f"{status.lower()}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    (classification / "summary.json").write_text(
        json.dumps(
            {
                "payload_hash_policy": "reference",
                "generation_contract": "LEGACY_EXACT_REFERENCE_V1",
                "historical_sidecar_sha256_required": False,
                "payload_full_read_performed": False,
                "byte_level_certified": False,
                "canonical_protocol": PROTOCOL,
                "status_counts": {
                    status: int(status == "REUSE_EXACT") for status in CLASSIFIER_STATUSES
                },
            }
        )
        + "\n"
    )
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"canonical_protocol": PROTOCOL}) + "\n")
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "contract": "LEGACY_EXACT_REFERENCE_V1",
                "canonical_temporal_protocol": PROTOCOL,
                "route_namespaces": REFERENCE_NAMESPACES,
            }
        )
        + "\n"
    )
    return classification, protocol, policy, source


def test_reference_overlay_is_zero_copy_and_does_not_hash_payload(
    tmp_path: Path, monkeypatch
) -> None:
    classification, protocol, policy, source = _inputs(tmp_path)
    roots = {mode: str(tmp_path / name) for mode, name in REFERENCE_NAMESPACES.items()}
    plan, actions = build(
        classification_dir=classification,
        protocol_path=protocol,
        policy_path=policy,
        route_roots=roots,
    )
    output = tmp_path / "plan"
    write_outputs(output, plan, actions)

    audit = importlib.import_module("scripts.windowfix_v2.dry_run_reference_overlay")
    original_hash = audit.sha256_file

    def forbid_payload_hash(path: Path) -> str:
        if Path(path).suffix == ".sft":
            raise AssertionError("reference dry-run must not hash payloads")
        return original_hash(path)

    monkeypatch.setattr(audit, "sha256_file", forbid_payload_hash)
    report = validate(output)

    route = actions["reference_routes"][0]
    assert route["source_uri"] == source.as_uri()
    assert route["declared_source_payload_sha256"] == "0" * 64
    assert route["declared_sha256_verified"] is False
    assert plan["payload_copy_performed"] is False
    assert plan["byte_level_certified"] is False
    assert report["status"] == "PASS_TEMPORAL_PHYSICAL_REFERENCE_ONLY"
    assert report["payload_hash_performed"] is False
    assert not Path(roots["streaming"]).exists()


def test_reference_overlay_refuses_seeded_route_root(tmp_path: Path) -> None:
    classification, protocol, policy, _source = _inputs(tmp_path)
    with pytest.raises(PlanError, match="must end in"):
        build(
            classification_dir=classification,
            protocol_path=protocol,
            policy_path=policy,
            route_roots={
                "streaming": str(tmp_path / "FEAT_STREAMING_CANONICAL_V2_SEEDED_V1"),
                "target": str(tmp_path / REFERENCE_NAMESPACES["target"]),
            },
        )


def test_reference_overlay_rejects_classifier_that_claims_full_read(tmp_path: Path) -> None:
    classification, protocol, policy, _source = _inputs(tmp_path)
    summary_path = classification / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["payload_full_read_performed"] = True
    summary_path.write_text(json.dumps(summary) + "\n")
    roots = {mode: str(tmp_path / name) for mode, name in REFERENCE_NAMESPACES.items()}

    with pytest.raises(PlanError, match="unexpectedly reports a payload full read"):
        build(
            classification_dir=classification,
            protocol_path=protocol,
            policy_path=policy,
            route_roots=roots,
        )


def test_reference_overlay_routes_unverified_identity_to_reextract(tmp_path: Path) -> None:
    classification, protocol, policy, _source = _inputs(tmp_path)
    exact_path = classification / "reuse_exact.jsonl"
    row = json.loads(exact_path.read_text())
    row["status"] = "UNVERIFIED"
    row["reason_codes"] = ["spec_missing_expected_prompt_contract"]
    exact_path.write_text("")
    (classification / "unverified.jsonl").write_text(json.dumps(row) + "\n")
    summary_path = classification / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["status_counts"]["REUSE_EXACT"] = 0
    summary["status_counts"]["UNVERIFIED"] = 1
    summary_path.write_text(json.dumps(summary) + "\n")
    roots = {mode: str(tmp_path / name) for mode, name in REFERENCE_NAMESPACES.items()}

    plan, actions = build(
        classification_dir=classification,
        protocol_path=protocol,
        policy_path=policy,
        route_roots=roots,
    )

    assert actions["reference_routes"] == []
    assert actions["unverified_records"][0]["action"] == "REEXTRACT_UNVERIFIED"
    assert plan["counts"]["reextract"] == 1
