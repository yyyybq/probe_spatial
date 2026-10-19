#!/usr/bin/env python3
"""Render review-only SCO jobs for an immutable AOSS streaming gate plan.

The input is the *directory* produced by
``build_aoss_reference_gate_shard_plan.py``.  This renderer validates the
plan, every file binding in ``inputs`` and ``outputs``, and every shard record
before writing one fail-closed worker binding per shard.  It never imports an
AOSS client, invokes ``sco``, or runs a gate command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLAN_SCHEMA = "LEGACY_EXACT_REFERENCE_REMOTE_GATE_SHARD_PLAN_V1"
SHARD_SCHEMA = "LEGACY_EXACT_REFERENCE_REMOTE_GATE_SHARD_V1"
RECEIPT_SCHEMA = "LEGACY_EXACT_REFERENCE_REMOTE_GATE_RECEIPT_V1"
PARTITION_RULE = "zero_based_jsonl_record_offset_mod_shard_count"
WORKER_BINDING_SCHEMA_ID = (
    "canonical_v2_aoss_reference_gate_worker_binding_v1.schema.json"
)
WORKER_BINDING_TYPE = "canonical_v2_aoss_reference_gate_worker_binding"
REVIEW_STATUS = "REVIEW_ONLY_NOT_EXECUTED"

DEFAULT_WORKSPACE = "aigc"
DEFAULT_AEC2_NAME = "h800"
DEFAULT_WORKER_SPEC = "N4lS.Iq.I80.8"
DEFAULT_PRIORITY = "HIGHEST"
DEFAULT_IMAGE = "registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617"
DEFAULT_STORAGE_MOUNT = "019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm"
DEFAULT_PYTHON = Path(
    "/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python"
)
HEX = frozenset("0123456789abcdef")


class RenderError(ValueError):
    """The source gate plan cannot safely be rendered as SCO review jobs."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_content_sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        canonical_json(
            {key: item for key, item in value.items() if key != "content_sha256"}
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: Any, label: str) -> str:
    text = str(value or "").lower()
    if len(text) != 64 or set(text) - HEX:
        raise RenderError(f"{label} must be a lowercase SHA256")
    return text


def _regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise RenderError(f"{label} must be a regular non-symlink file: {path}")
    return path.resolve()


def _directory(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise RenderError(f"{label} must be a directory and not a symlink: {path}")
    return path.resolve()


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RenderError(f"cannot parse {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RenderError(f"{label} must be a JSON object: {path}")
    return value


def _binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "file_sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _validate_bound_file(value: Any, label: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(value, dict):
        raise RenderError(f"{label} binding must be an object")
    for field in ("path", "sha256", "size_bytes"):
        if field not in value:
            raise RenderError(f"{label} binding lacks {field}")
    path = Path(str(value["path"]))
    if not path.is_absolute():
        raise RenderError(f"{label} path must be absolute: {path}")
    path = _regular(path, label)
    expected_sha = _sha(value["sha256"], f"{label} SHA256")
    try:
        expected_size = int(value["size_bytes"])
    except (TypeError, ValueError) as exc:
        raise RenderError(f"{label} size binding is invalid") from exc
    if expected_size < 0 or path.stat().st_size != expected_size:
        raise RenderError(
            f"{label} size mismatch: expected {expected_size}, found {path.stat().st_size}"
        )
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise RenderError(
            f"{label} SHA256 mismatch: expected {expected_sha}, found {actual_sha}"
        )
    normalized = dict(value)
    normalized["path"] = str(path)
    normalized["sha256"] = expected_sha
    normalized["size_bytes"] = expected_size
    return path, normalized


def _validate_plan(
    gate_plan_dir: Path,
) -> tuple[
    Path,
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
]:
    gate_plan_dir = _directory(gate_plan_dir, "gate plan directory")
    plan_path = _regular(gate_plan_dir / "plan.json", "gate plan")
    plan = _read_object(plan_path, "gate plan")
    if plan.get("schema") != PLAN_SCHEMA:
        raise RenderError("gate plan schema mismatch")
    if (
        plan.get("status") != "REVIEW_ONLY"
        or plan.get("review_only") is not True
        or plan.get("submissions") != 0
        or plan.get("remote_reads_performed") != 0
        or plan.get("commands_executed") != 0
    ):
        raise RenderError("gate plan does not prove review-only/no-execution status")
    recorded_content = _sha(plan.get("content_sha256"), "gate plan content SHA256")
    actual_content = canonical_content_sha(plan)
    if actual_content != recorded_content:
        raise RenderError(
            "gate plan content SHA256 mismatch: "
            f"recorded {recorded_content}, found {actual_content}"
        )

    raw_inputs = plan.get("inputs")
    if not isinstance(raw_inputs, dict) or not raw_inputs:
        raise RenderError("gate plan inputs must be a non-empty object")
    required_inputs = {"requests", "gate_script", "receipt_schema"}
    if not required_inputs <= set(raw_inputs):
        raise RenderError(
            f"gate plan lacks required inputs: {sorted(required_inputs - set(raw_inputs))}"
        )
    inputs: dict[str, dict[str, Any]] = {}
    for name, value in sorted(raw_inputs.items()):
        _, inputs[name] = _validate_bound_file(value, f"gate plan input {name}")

    if inputs["receipt_schema"].get("receipt_contract") != RECEIPT_SCHEMA:
        raise RenderError("gate plan receipt schema contract mismatch")
    request_records = inputs["requests"].get("records")
    try:
        request_records = int(request_records)
    except (TypeError, ValueError) as exc:
        raise RenderError("gate plan request binding lacks a valid records count") from exc
    if request_records <= 0:
        raise RenderError("gate plan request records must be positive")

    raw_outputs = plan.get("outputs")
    if not isinstance(raw_outputs, dict) or set(raw_outputs) != {
        "shards.jsonl",
        "commands.sh",
        "README.md",
    }:
        raise RenderError("gate plan output set mismatch")
    outputs: dict[str, dict[str, Any]] = {}
    output_paths: dict[str, Path] = {}
    for name, value in sorted(raw_outputs.items()):
        path, outputs[name] = _validate_bound_file(value, f"gate plan output {name}")
        if path != (gate_plan_dir / name).resolve():
            raise RenderError(f"gate plan output {name} escapes or aliases its plan directory")
        output_paths[name] = path

    shard_counts = plan.get("shard_expected_records")
    counts = plan.get("counts")
    if not isinstance(shard_counts, list) or not isinstance(counts, dict):
        raise RenderError("gate plan shard count metadata is malformed")
    try:
        declared_shards = int(counts["shards"])
        declared_requests = int(counts["requests"])
        declared_commands = int(counts["commands_rendered"])
        normalized_counts = [int(item) for item in shard_counts]
    except (KeyError, TypeError, ValueError) as exc:
        raise RenderError("gate plan count metadata is invalid") from exc
    if (
        declared_shards <= 0
        or len(normalized_counts) != declared_shards
        or declared_commands != declared_shards
        or declared_requests != request_records
        or sum(normalized_counts) != request_records
        or any(item < 0 for item in normalized_counts)
    ):
        raise RenderError("gate plan aggregate shard/request counts disagree")

    shards: list[dict[str, Any]] = []
    try:
        with output_paths["shards.jsonl"].open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise RenderError(f"shards.jsonl has a blank line at {line_number}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RenderError(f"shard line {line_number} is not an object")
                shards.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RenderError(f"cannot parse shards.jsonl: {exc}") from exc
    if len(shards) != declared_shards:
        raise RenderError(
            f"shard record count mismatch: expected {declared_shards}, found {len(shards)}"
        )

    seen_indices: set[int] = set()
    seen_ids: set[str] = set()
    plan_id = str(plan.get("plan_id", ""))
    if not plan_id:
        raise RenderError("gate plan lacks plan_id")
    for line_number, shard in enumerate(shards, 1):
        if shard.get("schema") != SHARD_SCHEMA or shard.get("plan_id") != plan_id:
            raise RenderError(f"shard line {line_number} schema/plan identity mismatch")
        recorded = _sha(
            shard.get("content_sha256"), f"shard line {line_number} content SHA256"
        )
        actual = canonical_content_sha(shard)
        if recorded != actual:
            raise RenderError(f"shard line {line_number} content SHA256 mismatch")
        if (
            shard.get("status") != "REVIEW_ONLY"
            or shard.get("review_only") is not True
            or shard.get("submissions") != 0
            or shard.get("remote_reads_performed") != 0
            or shard.get("commands_executed") != 0
            or shard.get("partition_rule") != PARTITION_RULE
        ):
            raise RenderError(f"shard line {line_number} is not fail-closed review-only")
        try:
            index = int(shard["shard_index"])
            shard_count = int(shard["shard_count"])
            expected_records = int(shard["expected_request_records"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RenderError(f"shard line {line_number} count metadata is invalid") from exc
        shard_id = str(shard.get("shard_id", ""))
        if (
            index < 0
            or index >= declared_shards
            or shard_count != declared_shards
            or expected_records != normalized_counts[index]
            or not shard_id
            or index in seen_indices
            or shard_id in seen_ids
        ):
            raise RenderError(f"shard line {line_number} identity/count mismatch")
        seen_indices.add(index)
        seen_ids.add(shard_id)
        for input_name in ("requests", "gate_script", "receipt_schema"):
            if shard.get(input_name) != inputs[input_name]:
                raise RenderError(
                    f"shard line {line_number} {input_name} binding differs from plan"
                )
        argv = shard.get("command_argv")
        if not isinstance(argv, list) or not argv or any(
            not isinstance(item, str) or not item for item in argv
        ):
            raise RenderError(f"shard line {line_number} command argv is invalid")
        expected_argv_tail = [
            str(inputs["gate_script"]["path"]),
            "--requests",
            str(inputs["requests"]["path"]),
            "--conf",
            str(plan.get("aoss_conf_path_unread", "")),
            "--output-dir",
            str(shard.get("output_dir", "")),
            "--shard-index",
            str(index),
            "--shard-count",
            str(declared_shards),
            "--request-timeout-seconds",
            str(plan.get("request_timeout_seconds")),
        ]
        if argv[1:] != expected_argv_tail:
            raise RenderError(f"shard line {line_number} command argv differs from plan")
        output_dir = Path(str(shard.get("output_dir", "")))
        if not output_dir.is_absolute():
            raise RenderError(f"shard line {line_number} output directory is not absolute")
        receipt_root = Path(str(plan.get("receipt_output_root", "")))
        if not receipt_root.is_absolute() or output_dir.parent != receipt_root:
            raise RenderError(f"shard line {line_number} output is outside receipt root")
    if seen_indices != set(range(declared_shards)):
        raise RenderError("gate shard indices are not complete")
    if int(counts.get("expected_receipts", -1)) != request_records:
        raise RenderError("gate plan expected_receipts differs from requests")
    return plan_path, plan, inputs, outputs, sorted(
        shards, key=lambda value: int(value["shard_index"])
    )


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _write(path: Path, data: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.chmod(mode)


def _staged_file_binding(stage_root: Path, output_dir: Path, relative: Path) -> dict[str, Any]:
    path = stage_root / relative
    return {
        "path": str(output_dir / relative),
        "file_sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _source_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(value["path"]),
        "file_sha256": str(value["sha256"]),
        "size_bytes": int(value["size_bytes"]),
    }


def _shell(argv: Sequence[str]) -> str:
    return shlex.join([str(item) for item in argv])


def render(
    *,
    gate_plan_dir: Path,
    output_dir: Path,
    python: Path,
    workspace: str = DEFAULT_WORKSPACE,
    aec2_name: str = DEFAULT_AEC2_NAME,
    worker_spec: str = DEFAULT_WORKER_SPEC,
    priority: str = DEFAULT_PRIORITY,
    container_image: str = DEFAULT_IMAGE,
    storage_mount: str = DEFAULT_STORAGE_MOUNT,
    worker_wrapper: Path | None = None,
    binding_schema: Path | None = None,
) -> dict[str, Any]:
    """Validate ``gate_plan_dir`` and atomically render review artifacts."""

    plan_path, plan, inputs, source_outputs, shards = _validate_plan(gate_plan_dir)
    output_dir = output_dir.resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise RenderError(f"output already exists; overwrite forbidden: {output_dir}")
    for label, value in (
        ("workspace", workspace),
        ("aec2_name", aec2_name),
        ("worker_spec", worker_spec),
        ("priority", priority),
        ("container_image", container_image),
        ("storage_mount", storage_mount),
    ):
        if not str(value).strip():
            raise RenderError(f"{label} must be non-empty")

    # The established conda ``python`` entry is a symlink.  Freeze its regular
    # resolved executable, which is also what ``sys.executable`` reports in the
    # worker and what the rendered command invokes.
    python = _regular(python.resolve(), "worker Python interpreter")
    wrapper = _regular(
        worker_wrapper
        or PROJECT_ROOT / "scripts/windowfix_v2/run_aoss_reference_gate_worker.py",
        "worker wrapper",
    )
    schema_path = _regular(
        binding_schema
        or PROJECT_ROOT
        / "scripts/windowfix_v2/aoss_reference_gate_worker_binding_v1.schema.json",
        "worker binding schema",
    )
    schema = _read_object(schema_path, "worker binding schema")
    if schema.get("$id") != WORKER_BINDING_SCHEMA_ID:
        raise RenderError("worker binding schema identity mismatch")
    aoss_conf = _regular(Path(str(plan.get("aoss_conf_path_unread", ""))), "AOSS config")

    source_plan_binding = {
        **_binding(plan_path),
        "content_sha256": str(plan["content_sha256"]),
    }
    source_input_bindings = {
        name: _source_binding(value) for name, value in sorted(inputs.items())
    }
    source_output_bindings = {
        name: _source_binding(value) for name, value in sorted(source_outputs.items())
    }
    runtime_bindings = {
        "python": _binding(python),
        "worker_wrapper": _binding(wrapper),
        "binding_schema": {
            **_binding(schema_path),
            "schema_id": WORKER_BINDING_SCHEMA_ID,
        },
        "aoss_conf": _binding(aoss_conf),
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        worker_commands: list[str] = []
        sco_commands: list[str] = []
        job_summaries: list[dict[str, Any]] = []
        width = max(4, len(str(len(shards) - 1)))
        for shard in shards:
            index = int(shard["shard_index"])
            shard_id = str(shard["shard_id"])
            relative_shard = Path("shards") / f"{index:0{width}d}.json"
            final_shard_path = output_dir / relative_shard
            _write(stage / relative_shard, _json_bytes(shard))
            shard_binding = {
                **_staged_file_binding(stage, output_dir, relative_shard),
                "content_sha256": str(shard["content_sha256"]),
                "source_line_number": index + 1,
                "shard_id": shard_id,
                "shard_index": index,
                "shard_count": int(shard["shard_count"]),
                "expected_request_records": int(shard["expected_request_records"]),
            }
            source_argv = list(shard["command_argv"])
            effective_argv = [str(python), *source_argv[1:]]
            binding: dict[str, Any] = {
                "schema_version": 1,
                "receipt_type": WORKER_BINDING_TYPE,
                "status": REVIEW_STATUS,
                "source_gate_plan": source_plan_binding,
                "source_gate_inputs": source_input_bindings,
                "source_gate_outputs": source_output_bindings,
                "shard": shard_binding,
                "runtime": runtime_bindings,
                "gate_execution": {
                    "source_command_argv": source_argv,
                    "effective_command_argv": effective_argv,
                    "python_override_is_explicit": effective_argv[0] != source_argv[0],
                    "output_dir": str(shard["output_dir"]),
                    "output_must_be_absent_before_start": True,
                    "request_timeout_seconds": int(plan["request_timeout_seconds"]),
                },
                "execution_policy": {
                    "review_only": True,
                    "sco_submission_performed": False,
                    "remote_reads_before_preflight": False,
                    "all_bound_files_require_sha256_and_size": True,
                },
            }
            binding["content_sha256"] = canonical_content_sha(binding)
            relative_binding = Path("bindings") / f"{index:0{width}d}.json"
            final_binding_path = output_dir / relative_binding
            _write(stage / relative_binding, _json_bytes(binding))
            binding_file = _staged_file_binding(stage, output_dir, relative_binding)

            worker_argv = [
                str(python),
                str(wrapper),
                "--binding-receipt",
                str(final_binding_path),
                "--binding-schema",
                str(schema_path),
                "--expected-binding-receipt-sha256",
                str(binding_file["file_sha256"]),
                "--expected-binding-receipt-size",
                str(binding_file["size_bytes"]),
                "--expected-binding-schema-sha256",
                str(runtime_bindings["binding_schema"]["file_sha256"]),
                "--expected-source-plan-sha256",
                str(source_plan_binding["file_sha256"]),
                "--expected-shard-sha256",
                str(shard_binding["file_sha256"]),
                "--expected-worker-wrapper-sha256",
                str(runtime_bindings["worker_wrapper"]["file_sha256"]),
                "--expected-python-sha256",
                str(runtime_bindings["python"]["file_sha256"]),
            ]
            worker_command = (
                "cd " + shlex.quote(str(PROJECT_ROOT)) + " && " + _shell(worker_argv)
            )
            job_name = f"bq-wf2-aoss-{str(plan['content_sha256'])[:8]}-{index:0{width}d}"
            if len(job_name) > 63:
                raise RenderError(f"generated SCO job name is too long: {job_name}")
            sco_argv = [
                "sco",
                "acp",
                "jobs",
                "create",
                f"--workspace-name={workspace}",
                f"--aec2-name={aec2_name}",
                f"--job-name={job_name}",
                f"--priority={priority}",
                f"--container-image-url={container_image}",
                f"--storage-mount={storage_mount}",
                "--training-framework=pytorch",
                "--worker-nodes=1",
                f"--worker-spec={worker_spec}",
                f"--command={worker_command}",
            ]
            jobspec: dict[str, Any] = {
                "schema_version": 1,
                "plan_type": "canonical_v2_aoss_reference_gate_sco_job_review",
                "status": "READY_FOR_REVIEW_NOT_SUBMITTED",
                "submission_performed": False,
                "source_gate_plan_content_sha256": str(plan["content_sha256"]),
                "shard_id": shard_id,
                "shard_index": index,
                "expected_request_records": int(shard["expected_request_records"]),
                "binding_receipt": binding_file,
                "worker_command": worker_command,
                "sco_argv": sco_argv,
                "sco_command": _shell(sco_argv),
            }
            jobspec["content_sha256"] = canonical_content_sha(jobspec)
            relative_jobspec = Path("jobspecs") / f"{index:0{width}d}.json"
            _write(stage / relative_jobspec, _json_bytes(jobspec))
            jobspec_binding = _staged_file_binding(stage, output_dir, relative_jobspec)
            worker_commands.append(worker_command)
            sco_commands.append(jobspec["sco_command"])
            job_summaries.append(
                {
                    "shard_id": shard_id,
                    "shard_index": index,
                    "expected_request_records": int(shard["expected_request_records"]),
                    "shard": shard_binding,
                    "binding_receipt": {
                        **binding_file,
                        "content_sha256": binding["content_sha256"],
                    },
                    "jobspec": {
                        **jobspec_binding,
                        "content_sha256": jobspec["content_sha256"],
                    },
                    "gate_output_dir": str(shard["output_dir"]),
                }
            )

        worker_rel = Path("worker_commands.txt")
        sco_rel = Path("sco_create_commands.txt")
        _write(
            stage / worker_rel,
            ("\n".join(worker_commands) + "\n").encode("utf-8"),
        )
        _write(
            stage / sco_rel,
            (
                "# REVIEW-ONLY ARTIFACT. No SCO command was invoked by the renderer.\n"
                + "\n".join(sco_commands)
                + "\n"
            ).encode("utf-8"),
        )

        summary: dict[str, Any] = {
            "schema_version": 1,
            "plan_type": "canonical_v2_aoss_reference_gate_sco_review",
            "status": "READY_FOR_REVIEW_NOT_SUBMITTED",
            "review_only": True,
            "submission_performed": False,
            "sco_commands_executed": 0,
            "remote_reads_performed": 0,
            "source_gate_plan": source_plan_binding,
            "source_gate_inputs": source_input_bindings,
            "source_gate_outputs": source_output_bindings,
            "runtime": runtime_bindings,
            "sco_environment": {
                "workspace": workspace,
                "aec2_name": aec2_name,
                "worker_spec": worker_spec,
                "priority": priority,
                "container_image": container_image,
                "storage_mount": storage_mount,
                "worker_nodes": 1,
                "training_framework": "pytorch",
            },
            "counts": {
                "requests": int(plan["counts"]["requests"]),
                "shards": len(shards),
                "worker_bindings": len(job_summaries),
                "jobspecs": len(job_summaries),
                "sco_commands_rendered": len(sco_commands),
                "sco_commands_executed": 0,
                "expected_receipts": sum(
                    int(item["expected_request_records"]) for item in shards
                ),
            },
            "execution_preflight": {
                "required_before_any_AOSS_read": True,
                "stdlib_only_worker_until_gate_exec": True,
                "checks_plan_file_and_content_sha256": True,
                "checks_every_source_input_sha256_and_size": True,
                "checks_every_source_output_sha256_and_size": True,
                "checks_all_shard_content_sha256_and_counts": True,
                "checks_request_gate_schema_current_sha256": True,
                "checks_AOSS_config_sha256_and_size": True,
                "same_size_tampering_is_rejected": True,
            },
            "jobs": job_summaries,
            "worker_commands": _staged_file_binding(stage, output_dir, worker_rel),
            "sco_create_commands": {
                **_staged_file_binding(stage, output_dir, sco_rel),
                "mode": "0644",
                "execution_policy": "review text only; renderer never invokes SCO",
            },
        }
        summary["content_sha256"] = canonical_content_sha(summary)
        summary_rel = Path("summary.json")
        _write(stage / summary_rel, _json_bytes(summary))

        artifact_files: dict[str, dict[str, Any]] = {}
        for path in sorted(item for item in stage.rglob("*") if item.is_file()):
            relative = path.relative_to(stage)
            artifact_files[str(relative)] = _staged_file_binding(
                stage, output_dir, relative
            )
        lock: dict[str, Any] = {
            "schema_version": 1,
            "plan_type": "canonical_v2_aoss_reference_gate_sco_review_lock",
            "status": "IMMUTABLE_REVIEW_ARTIFACT_NOT_SUBMITTED",
            "source_gate_plan": source_plan_binding,
            "renderer": _binding(Path(__file__).resolve()),
            "runtime": runtime_bindings,
            "counts": summary["counts"],
            "artifact_files": artifact_files,
        }
        lock["content_sha256"] = canonical_content_sha(lock)
        _write(stage / "PLAN.lock.json", _json_bytes(lock))
        os.replace(stage, output_dir)
        return {
            **summary,
            "summary_path": str(output_dir / summary_rel),
            "summary_file_sha256": sha256_file(output_dir / summary_rel),
            "plan_lock_path": str(output_dir / "PLAN.lock.json"),
            "plan_lock_file_sha256": sha256_file(output_dir / "PLAN.lock.json"),
        }
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--gate-plan-dir", required=True, type=Path)
    value.add_argument("--output-dir", required=True, type=Path)
    value.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    value.add_argument("--worker-wrapper", type=Path)
    value.add_argument("--binding-schema", type=Path)
    value.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    value.add_argument("--aec2-name", default=DEFAULT_AEC2_NAME)
    value.add_argument("--worker-spec", default=DEFAULT_WORKER_SPEC)
    value.add_argument("--priority", default=DEFAULT_PRIORITY)
    value.add_argument("--container-image", default=DEFAULT_IMAGE)
    value.add_argument("--storage-mount", default=DEFAULT_STORAGE_MOUNT)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        summary = render(
            gate_plan_dir=args.gate_plan_dir,
            output_dir=args.output_dir,
            python=args.python,
            workspace=args.workspace,
            aec2_name=args.aec2_name,
            worker_spec=args.worker_spec,
            priority=args.priority,
            container_image=args.container_image,
            storage_mount=args.storage_mount,
            worker_wrapper=args.worker_wrapper,
            binding_schema=args.binding_schema,
        )
    except (OSError, RenderError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": summary["status"],
                "submission_performed": summary["submission_performed"],
                "counts": summary["counts"],
                "summary": summary["summary_path"],
                "PLAN.lock": summary["plan_lock_path"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
