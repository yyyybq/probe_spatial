from __future__ import annotations

import gzip
import json
import stat
from pathlib import Path

import pytest

from analyses.windowfix_v2.build_aoss_reference_gate_shard_plan import (
    DEFAULT_SHARD_COUNT,
    EXPECTED_CANONICAL_SOURCE_SHA256,
    EXPECTED_CANONICAL_UNIVERSE_SHA256,
    PlanError,
    content_sha256,
    generate,
    sha256_file,
)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fixture(tmp_path: Path, records: int = 5) -> dict[str, Path | int | str]:
    catalog = tmp_path / "immutable-r5"
    catalog.mkdir(parents=True)
    namespace_lock = tmp_path / "canonical_namespace_lock.json"
    requests = catalog / "remote_gate_requests.jsonl.gz"
    with requests.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            for index in range(records):
                request = {
                    "schema": "LEGACY_EXACT_REFERENCE_REMOTE_GATE_REQUEST_V1",
                    "unit_id": f"unit-{index}",
                    "source_uri": f"aoss://bucket/f{index}.sft",
                    "sidecar_uri": f"aoss://bucket/f{index}.sft.manifest.json",
                    "source_index_uri": "aoss://bucket/prefix_index.npy",
                    "source_index_record": {"window_id": index, "tail": 7},
                    "expected": {"raw_ids": list(range(8))},
                }
                zipped.write(
                    (json.dumps(request, sort_keys=True, separators=(",", ":")) + "\n").encode()
                )

    summary = catalog / "summary.json"
    write_json(
        summary,
        {
            "canonical_namespace_lock": str(namespace_lock),
            "canonical_parent_source_sha256": EXPECTED_CANONICAL_SOURCE_SHA256,
            "canonical_universe_sha256": EXPECTED_CANONICAL_UNIVERSE_SHA256,
            "claims": {"jobs_submitted": 0},
            "counts": {"status": {"PENDING_REMOTE_GATE": records}},
            "remote_policy": {"pending_units": records},
        },
    )
    manifest = catalog / "catalog_manifest.json"
    write_json(
        manifest,
        {
            "canonical_parent_source_sha256": EXPECTED_CANONICAL_SOURCE_SHA256,
            "canonical_universe_sha256": EXPECTED_CANONICAL_UNIVERSE_SHA256,
            "outputs": [
                {
                    "file": requests.name,
                    "sha256": sha256_file(requests),
                    "size_bytes": requests.stat().st_size,
                },
                {
                    "file": summary.name,
                    "sha256": sha256_file(summary),
                    "size_bytes": summary.stat().st_size,
                },
            ]
        },
    )
    write_json(
        namespace_lock,
        {
            "canonical_source": {"sha256": EXPECTED_CANONICAL_SOURCE_SHA256},
            "canonical_universe_sha256": EXPECTED_CANONICAL_UNIVERSE_SHA256,
        },
    )
    gate = tmp_path / "gate.py"
    gate.write_text("# fixture gate; never executed\n", encoding="utf-8")
    schema = tmp_path / "receipt.schema.json"
    write_json(
        schema,
        {
            "properties": {
                "schema": {
                    "const": "LEGACY_EXACT_REFERENCE_REMOTE_GATE_RECEIPT_V1"
                }
            }
        },
    )
    return {
        "requests": requests,
        "catalog_manifest": manifest,
        "catalog_summary": summary,
        "namespace_lock": namespace_lock,
        "gate_script": gate,
        "receipt_schema": schema,
        "requests_sha": sha256_file(requests),
        "records": records,
    }


def render(tmp_path: Path, values: dict[str, Path | int | str], **overrides):
    args = {
        "requests": values["requests"],
        "catalog_manifest": values["catalog_manifest"],
        "catalog_summary": values["catalog_summary"],
        "namespace_lock": values["namespace_lock"],
        "gate_script": values["gate_script"],
        "receipt_schema": values["receipt_schema"],
        "output_dir": tmp_path / "plan",
        "receipt_output_root": tmp_path / "receipts",
        "aoss_conf": tmp_path / "not-read.conf",
        "shard_count": 3,
        "request_timeout_seconds": 30,
        "python": "python3",
        "plan_id": "fixture-r5-plan",
        "created_at": "2026-08-31T00:00:00+00:00",
    }
    args.update(overrides)
    return generate(**args)


def test_offline_renderer_binds_inputs_and_exact_modulo_counts(tmp_path: Path) -> None:
    values = fixture(tmp_path)
    plan = render(tmp_path, values)
    output = tmp_path / "plan"

    assert plan["status"] == "REVIEW_ONLY"
    assert plan["review_only"] is True
    assert plan["submissions"] == 0
    assert plan["remote_reads_performed"] == 0
    assert plan["commands_executed"] == 0
    assert plan["request_timeout_seconds"] == 30
    assert plan["counts"] == {
        "requests": 5,
        "shards": 3,
        "commands_rendered": 3,
        "expected_receipts": 5,
        "minimum_records_per_shard": 1,
        "maximum_records_per_shard": 2,
        "shards_with_extra_record": 2,
    }
    assert plan["shard_expected_records"] == [2, 2, 1]
    assert plan["inputs"]["requests"]["sha256"] == values["requests_sha"]
    assert plan["inputs"]["requests"]["records"] == 5
    assert (
        plan["inputs"]["canonical_namespace_lock"]["canonical_source_sha256"]
        == EXPECTED_CANONICAL_SOURCE_SHA256
    )
    assert (
        plan["inputs"]["canonical_namespace_lock"]["canonical_universe_sha256"]
        == EXPECTED_CANONICAL_UNIVERSE_SHA256
    )
    assert plan["content_sha256"] == content_sha256(plan)

    shards = [json.loads(line) for line in (output / "shards.jsonl").read_text().splitlines()]
    assert len(shards) == 3
    assert [item["expected_request_records"] for item in shards] == [2, 2, 1]
    assert all(item["submissions"] == 0 for item in shards)
    assert all(item["remote_reads_performed"] == 0 for item in shards)
    assert all(item["content_sha256"] == content_sha256(item) for item in shards)

    commands = (output / "commands.sh").read_text(encoding="utf-8")
    assert commands.count("--shard-index") == 3
    assert commands.count("--request-timeout-seconds 30") == 3
    assert "sco acp" not in commands
    assert "jobs create" not in commands
    assert not ((output / "commands.sh").stat().st_mode & stat.S_IXUSR)
    for name in ("shards.jsonl", "commands.sh", "README.md"):
        assert plan["outputs"][name]["sha256"] == sha256_file(output / name)


def test_renderer_rejects_request_bytes_not_bound_by_catalog(tmp_path: Path) -> None:
    values = fixture(tmp_path)
    requests = values["requests"]
    assert isinstance(requests, Path)
    requests.write_bytes(requests.read_bytes() + b"tamper")
    with pytest.raises(PlanError, match="request size binding|SHA-256 mismatch"):
        render(tmp_path, values)
    assert not (tmp_path / "plan").exists()


def test_renderer_rejects_noncanonical_lineage_or_timeout(tmp_path: Path) -> None:
    values = fixture(tmp_path)
    lock = values["namespace_lock"]
    assert isinstance(lock, Path)
    write_json(
        lock,
        {
            "canonical_source": {"sha256": "0" * 64},
            "canonical_universe_sha256": EXPECTED_CANONICAL_UNIVERSE_SHA256,
        },
    )
    with pytest.raises(PlanError, match="source SHA mismatch"):
        render(tmp_path, values)

    values = fixture(tmp_path / "timeout")
    with pytest.raises(PlanError, match="requires timeout=30"):
        render(tmp_path / "timeout", values, request_timeout_seconds=29)


def test_default_plan_has_128_shards() -> None:
    assert DEFAULT_SHARD_COUNT == 128
