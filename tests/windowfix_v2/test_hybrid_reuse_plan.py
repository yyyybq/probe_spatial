from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest

from analyses.windowfix_v2.build_hybrid_reuse_plan import (
    CLASSIFIER_STATUSES,
    EXPECTED_NAMESPACES,
    PlanError,
    build,
    write_outputs,
)
from scripts.windowfix_v2.dry_run_hybrid_materialization import (
    MaterializationError,
    validate,
)


PROTOCOL = "inscene_streaming_canonical_v2_obs28_h124"


def _write_safetensors(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = json.dumps(
        {"feat": {"dtype": "F16", "shape": [2, 3], "data_offsets": [0, 12]}},
        separators=(",", ":"),
    ).encode()
    header += b" " * ((8 - len(header) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(12))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source = tmp_path / "legacy" / "feature_t749_layer20.sft"
    _write_safetensors(source)
    classification = tmp_path / "classification"
    classification.mkdir()
    exact = {
        "status": "REUSE_EXACT",
        "reason_codes": ["legacy_exact_import_v1_identity_checks_passed"],
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
        "source_payload_sha256": _sha(source),
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
    summary = {
        "payload_hash_policy": "legacy-import",
        "generation_contract": "LEGACY_EXACT_IMPORT_V1",
        "historical_sidecar_sha256_required": False,
        "canonical_protocol": PROTOCOL,
        "status_counts": {status: int(status == "REUSE_EXACT") for status in CLASSIFIER_STATUSES},
    }
    (classification / "summary.json").write_text(json.dumps(summary) + "\n")
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"canonical_protocol": PROTOCOL}) + "\n")
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "contract": "LEGACY_EXACT_IMPORT_V1",
                "canonical_temporal_protocol": PROTOCOL,
                "destination_namespaces": EXPECTED_NAMESPACES,
            }
        )
        + "\n"
    )
    return classification, protocol, policy, source


def test_hybrid_plan_and_dry_run_do_not_copy(tmp_path: Path) -> None:
    classification, protocol, policy, source = _write_inputs(tmp_path)
    destinations = {
        mode: str(tmp_path / namespace) for mode, namespace in EXPECTED_NAMESPACES.items()
    }
    plan, actions = build(
        classification_dir=classification,
        protocol_path=protocol,
        policy_path=policy,
        destination_roots=destinations,
    )
    output = tmp_path / "plan"
    write_outputs(output, plan, actions)
    report = validate(output, verify_source_sha=True)

    assert plan["counts"]["copy_whole_file"] == 1
    assert actions["import_files"][0]["source_path"] == str(source)
    assert report["status"] == "PASS_READY_FOR_SEPARATE_COPY_IMPLEMENTATION"
    assert report["copy_performed"] is False
    assert not Path(actions["import_files"][0]["destination_path"]).exists()


def test_hybrid_plan_refuses_seeded_namespace(tmp_path: Path) -> None:
    classification, protocol, policy, _source = _write_inputs(tmp_path)
    with pytest.raises(PlanError, match="must end in"):
        build(
            classification_dir=classification,
            protocol_path=protocol,
            policy_path=policy,
            destination_roots={
                "streaming": str(tmp_path / "FEAT_STREAMING_CANONICAL_V2_SEEDED_V1"),
                "target": str(tmp_path / EXPECTED_NAMESPACES["target"]),
            },
        )


def test_dry_run_rejects_tampered_worklist(tmp_path: Path) -> None:
    classification, protocol, policy, _source = _write_inputs(tmp_path)
    destinations = {
        mode: str(tmp_path / namespace) for mode, namespace in EXPECTED_NAMESPACES.items()
    }
    plan, actions = build(
        classification_dir=classification,
        protocol_path=protocol,
        policy_path=policy,
        destination_roots=destinations,
    )
    output = tmp_path / "plan"
    write_outputs(output, plan, actions)
    with (output / "import_files.jsonl").open("a") as handle:
        handle.write("{}\n")

    with pytest.raises(MaterializationError, match="worklist content SHA mismatch"):
        validate(output, verify_source_sha=False)
