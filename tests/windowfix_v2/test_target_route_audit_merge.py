from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from analyses.windowfix_v2.audit_target_row_routes import (
    CANONICAL_PROTOCOL,
    CONTRACT,
    MIXED_TARGET_NAMESPACE,
    audit_plan,
    canonical_json_sha,
    sha256_file,
    write_outputs,
)
from analyses.windowfix_v2.augment_target_merge_plan import _validate_route_audit
from analyses.windowfix_v2.merge_target_route_audit_shards import (
    ShardMergeError,
    merge_audits,
)
from scripts.windowfix_v2.prepare_sco_gap_jobs import validate_audits


PROJECT = Path(__file__).resolve().parents[2]
AUDITOR = PROJECT / "analyses/windowfix_v2/audit_target_row_routes.py"
SOURCE_READER = PROJECT / "analyses/windowfix_v2/target_row_source.py"
WORKER = PROJECT / "scripts/windowfix_v2/run_target_route_audit_worker.py"
SCHEMA = PROJECT / "scripts/windowfix_v2/target_route_audit_worker_binding_v1.schema.json"


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _with_content(**values) -> dict:
    result = dict(values)
    result["content_sha256"] = canonical_json_sha(result)
    return result


def _unit(vfm: str, layer: int) -> dict:
    return {
        "spec_id": f"{vfm}-t0-l{layer}",
        "vfm": vfm,
        "model_id": f"fixture/{vfm}",
        "accepted_model_ids": [f"fixture/{vfm}"],
        "layer": layer,
        "timestep": 0,
        "tensor_key": "feat",
        "source": "scannetpp",
        "scene_id": "scene-a",
        "expected_tail_shape": [2, 3],
        "expected_dtype": "torch.float16",
        "accepted_modes": ["target_isolated"],
        "prompt_contract": {"mode": "exact", "value": "."},
        "preprocess_identity": {"resize": [4, 6], "num_frames": 3},
        "destination_filename": f"feature_t0_layer{layer}.sft",
        "duplicate_owner_policy": "forbid",
        "sources": [],
    }


def _fixture(tmp_path: Path) -> dict[str, Path]:
    canonical = tmp_path / "canonical-p24.jsonl"
    canonical.write_text(
        json.dumps(
            {
                "sample_id": "scene-a_w000_p24",
                "source": "scannetpp",
                "scene_id": "scene-a",
                "streaming_window_id": 0,
                "streaming_obs_start": 0,
                "streaming_window_indices": list(range(100, 128)),
                "streaming_indices": list(range(100, 124)),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    protocol = tmp_path / "protocol.json"
    _json(
        protocol,
        {
            "schema_version": 2,
            "canonical_protocol": CANONICAL_PROTOCOL,
            "canonical_protocol_sha256": "a" * 64,
            "source": "scannetpp",
            "frame_index_schema": "image_mask_intersection_v1",
        },
    )
    plan = tmp_path / "parent-plan.json"
    plan_value = _with_content(
        schema_version=1,
        plan_type="canonical_v2_target_row_merge_plan",
        contract=CONTRACT,
        canonical_protocol=CANONICAL_PROTOCOL,
        protocol_file=str(protocol.resolve()),
        protocol_file_sha256=sha256_file(protocol),
        canonical_source={
            "path": str(canonical.resolve()),
            "sha256": sha256_file(canonical),
        },
        destination_root=str((tmp_path / MIXED_TARGET_NAMESPACE).resolve()),
        units=[_unit("dino", 0), _unit("wan", 5)],
    )
    _json(plan, plan_value)

    review = tmp_path / "review"
    (review / "shards").mkdir(parents=True)
    (review / "bindings").mkdir()
    (review / "outputs").mkdir()
    worker_commands = review / "worker_commands.txt"
    sco_commands = review / "sco_create_commands.txt"
    worker_commands.write_text("# fixture worker commands\n", encoding="utf-8")
    sco_commands.write_text("# fixture SCO commands\n", encoding="utf-8")

    manifest_shards = []
    for ordinal, vfm in enumerate(("dino", "wan"), 1):
        units = [unit for unit in plan_value["units"] if unit["vfm"] == vfm]
        shard_value = {
            key: value
            for key, value in plan_value.items()
            if key not in {"content_sha256", "units"}
        }
        shard_value["parent_plan"] = {
            "path": str(plan.resolve()),
            "file_sha256": sha256_file(plan),
            "content_sha256": plan_value["content_sha256"],
        }
        shard_value["units"] = units
        shard_value["content_sha256"] = canonical_json_sha(shard_value)
        shard_path = review / "shards" / f"{ordinal:02d}-{vfm}.json"
        _json(shard_path, shard_value)
        audit_output = (review / "outputs" / vfm).resolve()
        receipt_value = _with_content(
            schema_version=1,
            receipt_type="canonical_v2_target_route_audit_worker_binding",
            status="REVIEW_ONLY_NOT_EXECUTED",
            parent_plan={
                "path": str(plan.resolve()),
                "file_sha256": sha256_file(plan),
                "content_sha256": plan_value["content_sha256"],
            },
            shard={
                "path": str(shard_path.resolve()),
                "file_sha256": sha256_file(shard_path),
                "content_sha256": shard_value["content_sha256"],
            },
            auditor={"path": str(AUDITOR.resolve()), "file_sha256": sha256_file(AUDITOR)},
            source_reader={
                "path": str(SOURCE_READER.resolve()),
                "file_sha256": sha256_file(SOURCE_READER),
            },
            worker_wrapper={
                "path": str(WORKER.resolve()),
                "file_sha256": sha256_file(WORKER),
            },
            binding_schema={
                "path": str(SCHEMA.resolve()),
                "file_sha256": sha256_file(SCHEMA),
                "schema_id": "canonical_v2_target_route_audit_worker_binding_v1.schema.json",
            },
            audit_output=str(audit_output),
            audit_arguments={
                "remote_policy": "gate",
                "source_timeout_seconds": 30.0,
                "verify_source_sha": True,
            },
        )
        receipt_path = review / "bindings" / f"{ordinal:02d}-{vfm}.json"
        _json(receipt_path, receipt_value)
        result = audit_plan(
            shard_path.resolve(),
            verify_source_sha=True,
            remote_policy="gate",
            source_timeout_seconds=30.0,
        )
        report = write_outputs(audit_output, result)
        assert report["status"] == "INCOMPLETE"
        manifest_shards.append(
            {
                "ordinal": ordinal,
                "vfm": vfm,
                "plan": str(shard_path.resolve()),
                "plan_file_sha256": sha256_file(shard_path),
                "plan_content_sha256": shard_value["content_sha256"],
                "worker_binding_receipt": str(receipt_path.resolve()),
                "worker_binding_receipt_file_sha256": sha256_file(receipt_path),
                "worker_binding_receipt_content_sha256": receipt_value[
                    "content_sha256"
                ],
                "units": 1,
                "layers": [units[0]["layer"]],
                "scenes": 1,
                "audit_output": str(audit_output),
                "worker_command": "fixture",
                "sco_argv": ["fixture"],
            }
        )

    manifest_value = _with_content(
        schema_version=1,
        plan_type="canonical_v2_target_route_audit_sco_review",
        status="READY_FOR_REVIEW_NOT_SUBMITTED",
        submission_performed=False,
        feature_payload_written=False,
        source_plan={
            "path": str(plan.resolve()),
            "file_sha256": sha256_file(plan),
            "content_sha256": plan_value["content_sha256"],
        },
        auditor={"path": str(AUDITOR.resolve()), "sha256": sha256_file(AUDITOR)},
        source_reader={
            "path": str(SOURCE_READER.resolve()),
            "sha256": sha256_file(SOURCE_READER),
        },
        worker_wrapper={"path": str(WORKER.resolve()), "sha256": sha256_file(WORKER)},
        worker_binding_schema={
            "path": str(SCHEMA.resolve()),
            "sha256": sha256_file(SCHEMA),
            "schema_id": "canonical_v2_target_route_audit_worker_binding_v1.schema.json",
        },
        remote_policy="gate",
        source_timeout_seconds=30.0,
        verify_local_source_sha=True,
        counts={
            "shards": 2,
            "worker_binding_receipts": 2,
            "units": 2,
            "scenes_per_shard": 1,
            "sco_commands_rendered": 2,
            "sco_commands_executed": 0,
        },
        execution_preflight={"required_before_remote_reads": True, "fail_closed": True},
        shards=manifest_shards,
        worker_commands={"path": str(worker_commands.resolve()), "sha256": sha256_file(worker_commands)},
        sco_create_commands={"path": str(sco_commands.resolve()), "sha256": sha256_file(sco_commands)},
    )
    manifest = review / "manifest.json"
    _json(manifest, manifest_value)
    return {
        "plan": plan,
        "protocol": protocol,
        "canonical": canonical,
        "manifest": manifest,
        "review": review,
    }


def test_merge_outputs_bind_parent_and_are_directly_consumable(tmp_path: Path) -> None:
    values = _fixture(tmp_path)
    output = tmp_path / "unified-audit"
    report = merge_audits(
        plan_path=values["plan"],
        review_manifest_path=values["manifest"],
        output_dir=output,
    )
    report_path = output / "target_row_route_audit.json"
    assert report["status"] == "INCOMPLETE"
    assert report["plan_sha256"] == sha256_file(values["plan"])
    assert report["counts"]["units"] == 2
    assert report["counts"]["routed_rows"] == 0
    assert report["counts"]["missing_rows"] > 0
    assert report["shard_audit_merge"]["partition_verified_exact"] is True

    plan_value = json.loads(values["plan"].read_text(encoding="utf-8"))
    _audit, missing, worklists = _validate_route_audit(
        values["plan"].resolve(), plan_value, report_path
    )
    assert len(missing) == 2
    assert len(worklists) == 2
    bindings = validate_audits(
        [report_path],
        canonical_source_sha=sha256_file(values["canonical"]),
        protocol_file_sha=sha256_file(values["protocol"]),
    )
    assert len(bindings) == 2


def test_merge_rejects_incomplete_review_partition(tmp_path: Path) -> None:
    values = _fixture(tmp_path)
    manifest = json.loads(values["manifest"].read_text(encoding="utf-8"))
    manifest["shards"] = manifest["shards"][:1]
    manifest["counts"]["shards"] = 1
    manifest["counts"]["worker_binding_receipts"] = 1
    manifest["counts"]["sco_commands_rendered"] = 1
    manifest["content_sha256"] = canonical_json_sha(
        {key: value for key, value in manifest.items() if key != "content_sha256"}
    )
    _json(values["manifest"], manifest)
    with pytest.raises(ShardMergeError, match="VFM count"):
        merge_audits(
            plan_path=values["plan"],
            review_manifest_path=values["manifest"],
            output_dir=tmp_path / "unified-audit",
        )


def test_merge_rejects_tampered_bound_jsonl(tmp_path: Path) -> None:
    values = _fixture(tmp_path)
    missing = values["review"] / "outputs/dino/missing_target_units.jsonl"
    payload = bytearray(missing.read_bytes())
    position = payload.index(b'"missing_raw_ids":[') + len(b'"missing_raw_ids":[')
    assert 48 <= payload[position] <= 57
    payload[position] = ord("9") if payload[position] != ord("9") else ord("8")
    missing.write_bytes(payload)
    with pytest.raises(ShardMergeError, match="SHA256 mismatch"):
        merge_audits(
            plan_path=values["plan"],
            review_manifest_path=values["manifest"],
            output_dir=tmp_path / "unified-audit",
        )


def test_merge_rejects_tampered_worker_binding(tmp_path: Path) -> None:
    values = _fixture(tmp_path)
    receipt = values["review"] / "bindings/01-dino.json"
    receipt.write_bytes(receipt.read_bytes() + b" ")
    with pytest.raises(ShardMergeError, match="receipt file SHA256 mismatch"):
        merge_audits(
            plan_path=values["plan"],
            review_manifest_path=values["manifest"],
            output_dir=tmp_path / "unified-audit",
        )


def test_merge_rejects_rewritten_auditor_lineage(tmp_path: Path) -> None:
    values = _fixture(tmp_path)
    report_path = values["review"] / "outputs/dino/target_row_route_audit.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["auditor_source_sha256"] = "0" * 64
    report["content_sha256"] = canonical_json_sha(
        {key: value for key, value in report.items() if key != "content_sha256"}
    )
    _json(report_path, report)
    with pytest.raises(ShardMergeError, match="auditor_source_sha256 lineage"):
        merge_audits(
            plan_path=values["plan"],
            review_manifest_path=values["manifest"],
            output_dir=tmp_path / "unified-audit",
        )


def test_merge_refuses_to_overwrite_output(tmp_path: Path) -> None:
    values = _fixture(tmp_path)
    output = tmp_path / "unified-audit"
    output.mkdir()
    with pytest.raises(ShardMergeError, match="already exists"):
        merge_audits(
            plan_path=values["plan"],
            review_manifest_path=values["manifest"],
            output_dir=output,
        )
