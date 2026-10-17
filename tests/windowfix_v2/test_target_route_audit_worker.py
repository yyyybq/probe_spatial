from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.windowfix_v2.prepare_sco_target_route_audit import render
from scripts.windowfix_v2.run_target_route_audit_worker import (
    PLAN_CONTRACT,
    PLAN_TYPE,
    RECEIPT_STATUS,
    RECEIPT_TYPE,
    SCHEMA_ID,
    canonical_json_sha,
    sha256_file,
)

pytestmark = pytest.mark.integration


PROJECT = Path(__file__).resolve().parents[2]
REAL_WRAPPER = PROJECT / "scripts/windowfix_v2/run_target_route_audit_worker.py"
REAL_SCHEMA = (
    PROJECT / "scripts/windowfix_v2/target_route_audit_worker_binding_v1.schema.json"
)


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _content_object(**values) -> dict:
    result = dict(values)
    result["content_sha256"] = canonical_json_sha(result)
    return result


def _worker_fixture(tmp_path: Path) -> dict[str, Path | list[str]]:
    wrapper = tmp_path / "run_target_route_audit_worker.py"
    wrapper.write_bytes(REAL_WRAPPER.read_bytes())
    schema = tmp_path / REAL_SCHEMA.name
    schema.write_bytes(REAL_SCHEMA.read_bytes())

    parent = tmp_path / "parent.json"
    parent_value = _content_object(
        schema_version=1,
        plan_type=PLAN_TYPE,
        contract=PLAN_CONTRACT,
        units=[],
    )
    _json(parent, parent_value)
    parent_sha = sha256_file(parent)

    shard = tmp_path / "shard.json"
    shard_value = _content_object(
        schema_version=1,
        plan_type=PLAN_TYPE,
        contract=PLAN_CONTRACT,
        parent_plan={
            "path": str(parent.resolve()),
            "file_sha256": parent_sha,
            "content_sha256": parent_value["content_sha256"],
        },
        units=[],
    )
    _json(shard, shard_value)

    marker = tmp_path / "auditor-ran"
    auditor = tmp_path / "audit_target_row_routes.py"
    auditor.write_text(
        "from pathlib import Path\n"
        "Path(__file__).with_name('auditor-ran').write_text('fixture-audit')\n",
        encoding="utf-8",
    )
    source_reader = tmp_path / "target_row_source.py"
    source_reader.write_text("# fixture-source-reader\n", encoding="utf-8")
    output = tmp_path / "output"

    receipt = tmp_path / "binding.json"
    receipt_value = _content_object(
        schema_version=1,
        receipt_type=RECEIPT_TYPE,
        status=RECEIPT_STATUS,
        parent_plan={
            "path": str(parent.resolve()),
            "file_sha256": parent_sha,
            "content_sha256": parent_value["content_sha256"],
        },
        shard={
            "path": str(shard.resolve()),
            "file_sha256": sha256_file(shard),
            "content_sha256": shard_value["content_sha256"],
        },
        auditor={
            "path": str(auditor.resolve()),
            "file_sha256": sha256_file(auditor),
        },
        source_reader={
            "path": str(source_reader.resolve()),
            "file_sha256": sha256_file(source_reader),
        },
        worker_wrapper={
            "path": str(wrapper.resolve()),
            "file_sha256": sha256_file(wrapper),
        },
        binding_schema={
            "path": str(schema.resolve()),
            "file_sha256": sha256_file(schema),
            "schema_id": SCHEMA_ID,
        },
        audit_output=str(output.resolve()),
        audit_arguments={
            "remote_policy": "gate",
            "source_timeout_seconds": 30.0,
            "verify_source_sha": True,
        },
    )
    _json(receipt, receipt_value)
    argv = [
        sys.executable,
        str(wrapper.resolve()),
        "--binding-receipt",
        str(receipt.resolve()),
        "--binding-schema",
        str(schema.resolve()),
        "--expected-binding-receipt-sha256",
        sha256_file(receipt),
        "--expected-binding-schema-sha256",
        sha256_file(schema),
        "--expected-parent-plan-sha256",
        parent_sha,
        "--expected-shard-sha256",
        sha256_file(shard),
        "--expected-auditor-sha256",
        sha256_file(auditor),
        "--expected-source-reader-sha256",
        sha256_file(source_reader),
        "--expected-worker-wrapper-sha256",
        sha256_file(wrapper),
    ]
    return {
        "wrapper": wrapper,
        "schema": schema,
        "parent": parent,
        "shard": shard,
        "auditor": auditor,
        "source_reader": source_reader,
        "receipt": receipt,
        "marker": marker,
        "argv": argv,
    }


def _same_size_tamper(path: Path) -> None:
    payload = bytearray(path.read_bytes())
    for index, value in enumerate(payload):
        if value == ord(" "):
            payload[index] = ord("\t")
            break
    else:  # pragma: no cover - all fixtures contain at least one space
        raise AssertionError(f"fixture has no tamperable byte: {path}")
    path.write_bytes(payload)


def test_worker_runs_auditor_only_after_all_bindings_pass(tmp_path: Path) -> None:
    values = _worker_fixture(tmp_path)
    result = subprocess.run(values["argv"], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert values["marker"].read_text() == "fixture-audit"


@pytest.mark.parametrize(
    "target",
    [
        "parent",
        "shard",
        "auditor",
        "source_reader",
        "wrapper",
        "schema",
        "receipt",
    ],
)
def test_worker_rejects_same_size_tamper_before_auditor_runs(
    tmp_path: Path, target: str
) -> None:
    values = _worker_fixture(tmp_path)
    path = values[target]
    assert isinstance(path, Path)
    size = path.stat().st_size
    _same_size_tamper(path)
    assert path.stat().st_size == size

    result = subprocess.run(values["argv"], check=False, capture_output=True, text=True)
    assert result.returncode == 2
    assert "SHA256 mismatch" in result.stderr
    marker = values["marker"]
    assert isinstance(marker, Path)
    assert not marker.exists()


def test_renderer_emits_hash_bound_worker_receipts_and_cli(tmp_path: Path) -> None:
    vfms = (
        "aether",
        "cogvideox",
        "dino",
        "f3r",
        "lingbot",
        "minimax_h3",
        "qwen2_5_vl",
        "videomae",
        "vjepa2",
        "wan",
    )
    units = [
        {
            "vfm": vfms[index % len(vfms)],
            "layer": index % 50,
            "scene_id": f"scene-{index % 681:04d}",
        }
        for index in range(21_792)
    ]
    plan_value = _content_object(
        schema_version=1,
        plan_type=PLAN_TYPE,
        contract=PLAN_CONTRACT,
        units=units,
    )
    plan = tmp_path / "parent.json"
    _json(plan, plan_value)
    output = tmp_path / "review"
    result = render(
        plan_path=plan,
        output_dir=output,
        source_timeout_seconds=30.0,
        workspace="fixture",
        worker_spec="fixture-spec",
        image="fixture-image",
        storage_mount="fixture-mount",
    )

    assert result["counts"]["shards"] == 10
    assert result["counts"]["sco_commands_executed"] == 0
    assert result["worker_wrapper"]["sha256"] == sha256_file(REAL_WRAPPER)
    assert result["worker_binding_schema"]["sha256"] == sha256_file(REAL_SCHEMA)
    for shard in result["shards"]:
        receipt_path = Path(shard["worker_binding_receipt"])
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert sha256_file(receipt_path) == shard["worker_binding_receipt_file_sha256"]
        assert receipt["content_sha256"] == canonical_json_sha(
            {key: value for key, value in receipt.items() if key != "content_sha256"}
        )
        command = shard["worker_command"]
        for option in (
            "--expected-binding-receipt-sha256",
            "--expected-binding-schema-sha256",
            "--expected-parent-plan-sha256",
            "--expected-shard-sha256",
            "--expected-auditor-sha256",
            "--expected-source-reader-sha256",
            "--expected-worker-wrapper-sha256",
        ):
            assert option in command
        assert str(REAL_WRAPPER.resolve()) in command
