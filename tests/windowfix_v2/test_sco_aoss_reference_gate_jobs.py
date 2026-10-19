from __future__ import annotations

import argparse
import gzip
import json
import stat
import sys
from pathlib import Path

import pytest
import jsonschema

from analyses.windowfix_v2.build_aoss_reference_gate_shard_plan import (
    EXPECTED_CANONICAL_SOURCE_SHA256,
    EXPECTED_CANONICAL_UNIVERSE_SHA256,
    content_sha256,
    generate,
    sha256_file,
)
from scripts.windowfix_v2 import prepare_sco_aoss_reference_gate_jobs as renderer
from scripts.windowfix_v2 import run_aoss_reference_gate_worker as worker


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def source_plan(tmp_path: Path, records: int = 5) -> tuple[Path, dict[str, Path]]:
    catalog = tmp_path / "catalog"
    catalog.mkdir(parents=True)
    namespace_lock = tmp_path / "namespace-lock.json"
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
            ],
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
    gate.write_text(
        "import argparse\n"
        "from pathlib import Path\n"
        "p=argparse.ArgumentParser()\n"
        "p.add_argument('--requests');p.add_argument('--conf');p.add_argument('--output-dir')\n"
        "p.add_argument('--shard-index');p.add_argument('--shard-count')\n"
        "p.add_argument('--request-timeout-seconds');a=p.parse_args()\n"
        "Path(a.output_dir).mkdir(parents=True)\n",
        encoding="utf-8",
    )
    receipt_schema = tmp_path / "receipt.schema.json"
    write_json(
        receipt_schema,
        {
            "properties": {
                "schema": {
                    "const": "LEGACY_EXACT_REFERENCE_REMOTE_GATE_RECEIPT_V1"
                }
            }
        },
    )
    aoss_conf = tmp_path / "petreloss.conf"
    aoss_conf.write_text("[fixture]\nendpoint=fake\n", encoding="utf-8")
    plan_dir = tmp_path / "gate-plan"
    generate(
        requests=requests,
        catalog_manifest=manifest,
        catalog_summary=summary,
        namespace_lock=namespace_lock,
        gate_script=gate,
        receipt_schema=receipt_schema,
        output_dir=plan_dir,
        receipt_output_root=tmp_path / "receipts",
        aoss_conf=aoss_conf,
        shard_count=3,
        request_timeout_seconds=30,
        python="python3",
        plan_id="fixture-gate-plan",
        created_at="2026-08-31T00:00:00+00:00",
    )
    return plan_dir, {
        "requests": requests,
        "gate": gate,
        "receipt_schema": receipt_schema,
        "aoss_conf": aoss_conf,
    }


def render(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    plan_dir, paths = source_plan(tmp_path)
    output = tmp_path / "sco-review"
    renderer.render(
        gate_plan_dir=plan_dir,
        output_dir=output,
        python=Path(sys.executable),
    )
    return plan_dir, output, paths


def worker_args(output: Path, index: int = 0) -> argparse.Namespace:
    binding_path = output / "bindings" / f"{index:04d}.json"
    binding = json.loads(binding_path.read_text())
    return argparse.Namespace(
        binding_receipt=binding_path,
        binding_schema=Path(binding["runtime"]["binding_schema"]["path"]),
        expected_binding_receipt_sha256=worker.sha256_file(binding_path),
        expected_binding_receipt_size=binding_path.stat().st_size,
        expected_binding_schema_sha256=binding["runtime"]["binding_schema"]["file_sha256"],
        expected_source_plan_sha256=binding["source_gate_plan"]["file_sha256"],
        expected_shard_sha256=binding["shard"]["file_sha256"],
        expected_worker_wrapper_sha256=binding["runtime"]["worker_wrapper"]["file_sha256"],
        expected_python_sha256=binding["runtime"]["python"]["file_sha256"],
    )


def flip_same_size(path: Path) -> None:
    data = bytearray(path.read_bytes())
    index = next(i for i, value in enumerate(data) if value not in b"\r\n \t")
    data[index] = ord("X") if data[index] != ord("X") else ord("Y")
    path.write_bytes(bytes(data))


def test_render_is_review_only_and_fully_locked(tmp_path: Path) -> None:
    plan_dir, output, _ = render(tmp_path)
    summary = json.loads((output / "summary.json").read_text())
    lock = json.loads((output / "PLAN.lock.json").read_text())
    assert summary["status"] == "READY_FOR_REVIEW_NOT_SUBMITTED"
    assert summary["submission_performed"] is False
    assert summary["remote_reads_performed"] == 0
    assert summary["counts"] == {
        "expected_receipts": 5,
        "jobspecs": 3,
        "requests": 5,
        "sco_commands_executed": 0,
        "sco_commands_rendered": 3,
        "shards": 3,
        "worker_bindings": 3,
    }
    assert summary["source_gate_plan"]["content_sha256"] == json.loads(
        (plan_dir / "plan.json").read_text()
    )["content_sha256"]
    assert summary["execution_preflight"]["same_size_tampering_is_rejected"] is True
    commands = (output / "sco_create_commands.txt").read_text()
    assert commands.count("sco acp jobs create") == 3
    assert commands.count("run_aoss_reference_gate_worker.py") == 3
    assert not ((output / "sco_create_commands.txt").stat().st_mode & stat.S_IXUSR)
    assert "summary.json" in lock["artifact_files"]
    assert "sco_create_commands.txt" in lock["artifact_files"]
    assert len([name for name in lock["artifact_files"] if name.startswith("bindings/")]) == 3
    assert lock["content_sha256"] == renderer.canonical_content_sha(lock)
    binding = json.loads((output / "bindings/0000.json").read_text())
    schema = json.loads(
        Path(binding["runtime"]["binding_schema"]["path"]).read_text()
    )
    jsonschema.validate(binding, schema)
    assert worker.verify_preflight(worker_args(output))["shard"]["shard_index"] == 0


@pytest.mark.parametrize("target", ["gate", "requests", "receipt_schema"])
def test_renderer_rejects_same_size_input_tamper(tmp_path: Path, target: str) -> None:
    plan_dir, paths = source_plan(tmp_path)
    flip_same_size(paths[target])
    with pytest.raises(renderer.RenderError, match="SHA256 mismatch"):
        renderer.render(
            gate_plan_dir=plan_dir,
            output_dir=tmp_path / "review",
            python=Path(sys.executable),
        )


def test_renderer_rejects_plan_content_or_output_tamper(tmp_path: Path) -> None:
    plan_dir, _ = source_plan(tmp_path / "plan-content")
    plan_path = plan_dir / "plan.json"
    plan_path.write_text(
        plan_path.read_text().replace('"review_only": true', '"review_only": fals'),
        encoding="utf-8",
    )
    with pytest.raises(renderer.RenderError):
        renderer.render(
            gate_plan_dir=plan_dir,
            output_dir=tmp_path / "bad-plan-review",
            python=Path(sys.executable),
        )

    plan_dir, _ = source_plan(tmp_path / "source-output")
    flip_same_size(plan_dir / "README.md")
    with pytest.raises(renderer.RenderError, match="SHA256 mismatch"):
        renderer.render(
            gate_plan_dir=plan_dir,
            output_dir=tmp_path / "bad-output-review",
            python=Path(sys.executable),
        )


def test_renderer_independently_rejects_shard_count_change(tmp_path: Path) -> None:
    plan_dir, _ = source_plan(tmp_path)
    shards_path = plan_dir / "shards.jsonl"
    shards = [json.loads(line) for line in shards_path.read_text().splitlines()]
    shards[0]["expected_request_records"] += 1
    shards[0]["content_sha256"] = content_sha256(shards[0])
    shards_path.write_text(
        "".join(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n" for item in shards),
        encoding="utf-8",
    )
    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["outputs"]["shards.jsonl"]["sha256"] = sha256_file(shards_path)
    plan["outputs"]["shards.jsonl"]["size_bytes"] = shards_path.stat().st_size
    plan["content_sha256"] = content_sha256(plan)
    write_json(plan_path, plan)
    with pytest.raises(renderer.RenderError, match="identity/count mismatch"):
        renderer.render(
            gate_plan_dir=plan_dir,
            output_dir=tmp_path / "review",
            python=Path(sys.executable),
        )


@pytest.mark.parametrize("target", ["requests", "commands"])
def test_worker_rejects_same_size_tamper_before_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    plan_dir, output, paths = render(tmp_path)
    args = worker_args(output)
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("gate subprocess must not start")

    monkeypatch.setattr(worker.subprocess, "run", forbidden)
    flip_same_size(paths["requests"] if target == "requests" else plan_dir / "commands.sh")
    with pytest.raises(worker.WorkerBindingError, match="SHA256 mismatch"):
        worker.run(args)
    assert called is False


def test_worker_runs_only_after_complete_preflight(tmp_path: Path) -> None:
    _, output, _ = render(tmp_path)
    args = worker_args(output)
    assert worker.run(args) == 0
    binding = json.loads(args.binding_receipt.read_text())
    assert Path(binding["gate_execution"]["output_dir"]).is_dir()
