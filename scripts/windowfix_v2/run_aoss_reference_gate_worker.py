#!/usr/bin/env python3
"""Fail closed before running one reviewed AOSS streaming-reference gate shard.

Only Python's standard library is imported by this wrapper.  The AOSS-capable
gate subprocess is started only after the source plan's byte/content hash, all
of its input and output file SHA/size bindings, every shard content/count, and
the per-job binding receipt have been revalidated from current bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PLAN_SCHEMA = "LEGACY_EXACT_REFERENCE_REMOTE_GATE_SHARD_PLAN_V1"
SHARD_SCHEMA = "LEGACY_EXACT_REFERENCE_REMOTE_GATE_SHARD_V1"
RECEIPT_SCHEMA = "LEGACY_EXACT_REFERENCE_REMOTE_GATE_RECEIPT_V1"
PARTITION_RULE = "zero_based_jsonl_record_offset_mod_shard_count"
WORKER_BINDING_SCHEMA_ID = (
    "canonical_v2_aoss_reference_gate_worker_binding_v1.schema.json"
)
WORKER_BINDING_TYPE = "canonical_v2_aoss_reference_gate_worker_binding"
REVIEW_STATUS = "REVIEW_ONLY_NOT_EXECUTED"
HEX = frozenset("0123456789abcdef")


class WorkerBindingError(ValueError):
    """A launch dependency differs from the reviewed byte/content binding."""


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
        raise WorkerBindingError(f"{label} must be a lowercase SHA256")
    return text


def _size(value: Any, label: str) -> int:
    try:
        size = int(value)
    except (TypeError, ValueError) as exc:
        raise WorkerBindingError(f"{label} must be an integer") from exc
    if size < 0:
        raise WorkerBindingError(f"{label} must be non-negative")
    return size


def _regular(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise WorkerBindingError(
            f"{label} must be an absolute regular non-symlink file: {path}"
        )
    return path.resolve()


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkerBindingError(f"cannot parse {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise WorkerBindingError(f"{label} must be a JSON object: {path}")
    return value


def _verify_file_binding(
    value: Any,
    label: str,
    *,
    expected_sha: str | None = None,
    expected_size: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(value, dict):
        raise WorkerBindingError(f"{label} binding must be an object")
    for field in ("path", "file_sha256", "size_bytes"):
        if field not in value:
            raise WorkerBindingError(f"{label} binding lacks {field}")
    path = _regular(Path(str(value["path"])), label)
    recorded_sha = _sha(value["file_sha256"], f"{label} SHA256")
    recorded_size = _size(value["size_bytes"], f"{label} size")
    if expected_sha is not None and recorded_sha != _sha(expected_sha, f"expected {label} SHA256"):
        raise WorkerBindingError(f"{label} SHA256 differs from worker command")
    if expected_size is not None and recorded_size != expected_size:
        raise WorkerBindingError(f"{label} size differs from worker command")
    actual_size = path.stat().st_size
    if actual_size != recorded_size:
        raise WorkerBindingError(
            f"{label} size mismatch: expected {recorded_size}, found {actual_size}"
        )
    actual_sha = sha256_file(path)
    if actual_sha != recorded_sha:
        raise WorkerBindingError(
            f"{label} SHA256 mismatch: expected {recorded_sha}, found {actual_sha}"
        )
    return path, dict(value)


def _source_binding(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WorkerBindingError(f"{label} source-plan binding must be an object")
    for field in ("path", "sha256", "size_bytes"):
        if field not in value:
            raise WorkerBindingError(f"{label} source-plan binding lacks {field}")
    return {
        "path": str(Path(str(value["path"])).resolve()),
        "file_sha256": _sha(value["sha256"], f"{label} SHA256"),
        "size_bytes": _size(value["size_bytes"], f"{label} size"),
    }


def _content(value: Mapping[str, Any], label: str) -> str:
    recorded = _sha(value.get("content_sha256"), f"{label} content SHA256")
    actual = canonical_content_sha(value)
    if recorded != actual:
        raise WorkerBindingError(
            f"{label} content SHA256 mismatch: recorded {recorded}, found {actual}"
        )
    return actual


def _validate_binding_shape(binding: Mapping[str, Any]) -> None:
    expected = {
        "schema_version",
        "receipt_type",
        "status",
        "source_gate_plan",
        "source_gate_inputs",
        "source_gate_outputs",
        "shard",
        "runtime",
        "gate_execution",
        "execution_policy",
        "content_sha256",
    }
    if set(binding) != expected:
        raise WorkerBindingError("worker binding top-level fields mismatch")
    if binding.get("schema_version") != 1:
        raise WorkerBindingError("worker binding schema version mismatch")
    if binding.get("receipt_type") != WORKER_BINDING_TYPE:
        raise WorkerBindingError("worker binding receipt type mismatch")
    if binding.get("status") != REVIEW_STATUS:
        raise WorkerBindingError("worker binding status mismatch")
    policy = binding.get("execution_policy")
    if not isinstance(policy, dict) or policy != {
        "review_only": True,
        "sco_submission_performed": False,
        "remote_reads_before_preflight": False,
        "all_bound_files_require_sha256_and_size": True,
    }:
        raise WorkerBindingError("worker binding execution policy mismatch")


def _validate_source_plan(
    plan: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> tuple[dict[str, Path], dict[str, Path], list[dict[str, Any]]]:
    if plan.get("schema") != PLAN_SCHEMA:
        raise WorkerBindingError("source gate plan schema mismatch")
    if (
        plan.get("status") != "REVIEW_ONLY"
        or plan.get("review_only") is not True
        or plan.get("submissions") != 0
        or plan.get("remote_reads_performed") != 0
        or plan.get("commands_executed") != 0
    ):
        raise WorkerBindingError("source gate plan does not remain review-only")
    plan_content = _content(plan, "source gate plan")
    if binding["source_gate_plan"].get("content_sha256") != plan_content:
        raise WorkerBindingError("source gate plan content differs from binding receipt")

    plan_inputs = plan.get("inputs")
    receipt_inputs = binding.get("source_gate_inputs")
    if not isinstance(plan_inputs, dict) or not isinstance(receipt_inputs, dict):
        raise WorkerBindingError("source gate input bindings are malformed")
    if set(plan_inputs) != set(receipt_inputs):
        raise WorkerBindingError("source gate input binding names changed")
    required = {"requests", "gate_script", "receipt_schema"}
    if not required <= set(plan_inputs):
        raise WorkerBindingError("source gate plan lacks request/gate/schema input")
    input_paths: dict[str, Path] = {}
    for name in sorted(plan_inputs):
        expected = _source_binding(plan_inputs[name], f"source input {name}")
        if receipt_inputs[name] != expected:
            raise WorkerBindingError(f"source input {name} differs from worker binding")
        input_paths[name], _ = _verify_file_binding(expected, f"source input {name}")
    if plan_inputs["receipt_schema"].get("receipt_contract") != RECEIPT_SCHEMA:
        raise WorkerBindingError("source receipt schema contract mismatch")

    plan_outputs = plan.get("outputs")
    receipt_outputs = binding.get("source_gate_outputs")
    expected_output_names = {"shards.jsonl", "commands.sh", "README.md"}
    if (
        not isinstance(plan_outputs, dict)
        or not isinstance(receipt_outputs, dict)
        or set(plan_outputs) != expected_output_names
        or set(receipt_outputs) != expected_output_names
    ):
        raise WorkerBindingError("source gate output binding set changed")
    output_paths: dict[str, Path] = {}
    for name in sorted(plan_outputs):
        expected = _source_binding(plan_outputs[name], f"source output {name}")
        if receipt_outputs[name] != expected:
            raise WorkerBindingError(f"source output {name} differs from worker binding")
        output_paths[name], _ = _verify_file_binding(expected, f"source output {name}")

    counts = plan.get("counts")
    expected_counts = plan.get("shard_expected_records")
    if not isinstance(counts, dict) or not isinstance(expected_counts, list):
        raise WorkerBindingError("source gate count metadata is malformed")
    try:
        shard_count = int(counts["shards"])
        request_count = int(counts["requests"])
        request_records = int(plan_inputs["requests"]["records"])
        per_shard = [int(item) for item in expected_counts]
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkerBindingError("source gate count metadata is invalid") from exc
    if (
        shard_count <= 0
        or len(per_shard) != shard_count
        or any(item < 0 for item in per_shard)
        or sum(per_shard) != request_count
        or request_count != request_records
        or int(counts.get("commands_rendered", -1)) != shard_count
        or int(counts.get("expected_receipts", -1)) != request_count
    ):
        raise WorkerBindingError("source gate aggregate counts disagree")

    shards: list[dict[str, Any]] = []
    try:
        with output_paths["shards.jsonl"].open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise WorkerBindingError(
                        f"source shards stream has a blank line at {line_number}"
                    )
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise WorkerBindingError(
                        f"source shard line {line_number} is not an object"
                    )
                shards.append(item)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkerBindingError(f"cannot parse source shards stream: {exc}") from exc
    if len(shards) != shard_count:
        raise WorkerBindingError("source shard stream record count mismatch")
    seen_indices: set[int] = set()
    seen_ids: set[str] = set()
    for line_number, shard in enumerate(shards, 1):
        if shard.get("schema") != SHARD_SCHEMA or shard.get("plan_id") != plan.get("plan_id"):
            raise WorkerBindingError(f"source shard line {line_number} identity mismatch")
        _content(shard, f"source shard line {line_number}")
        if (
            shard.get("status") != "REVIEW_ONLY"
            or shard.get("review_only") is not True
            or shard.get("submissions") != 0
            or shard.get("remote_reads_performed") != 0
            or shard.get("commands_executed") != 0
            or shard.get("partition_rule") != PARTITION_RULE
        ):
            raise WorkerBindingError(f"source shard line {line_number} is not review-only")
        try:
            index = int(shard["shard_index"])
            item_shard_count = int(shard["shard_count"])
            item_records = int(shard["expected_request_records"])
        except (KeyError, TypeError, ValueError) as exc:
            raise WorkerBindingError(
                f"source shard line {line_number} count metadata is invalid"
            ) from exc
        shard_id = str(shard.get("shard_id", ""))
        if (
            index < 0
            or index >= shard_count
            or index in seen_indices
            or not shard_id
            or shard_id in seen_ids
            or item_shard_count != shard_count
            or item_records != per_shard[index]
        ):
            raise WorkerBindingError(f"source shard line {line_number} count mismatch")
        seen_indices.add(index)
        seen_ids.add(shard_id)
        for name in ("requests", "gate_script", "receipt_schema"):
            if shard.get(name) != plan_inputs[name]:
                raise WorkerBindingError(
                    f"source shard line {line_number} {name} binding mismatch"
                )
    if seen_indices != set(range(shard_count)):
        raise WorkerBindingError("source shard indices are incomplete")
    return input_paths, output_paths, shards


def verify_preflight(args: argparse.Namespace) -> dict[str, Any]:
    receipt_path = _regular(args.binding_receipt, "worker binding receipt")
    expected_receipt_size = _size(
        args.expected_binding_receipt_size, "expected worker binding receipt size"
    )
    receipt_stub = {
        "path": str(receipt_path),
        "file_sha256": args.expected_binding_receipt_sha256,
        "size_bytes": expected_receipt_size,
    }
    _verify_file_binding(
        receipt_stub,
        "worker binding receipt",
        expected_sha=args.expected_binding_receipt_sha256,
        expected_size=expected_receipt_size,
    )
    schema_path = _regular(args.binding_schema, "worker binding schema")
    binding = _read_object(receipt_path, "worker binding receipt")
    schema = _read_object(schema_path, "worker binding schema")
    if schema.get("$id") != WORKER_BINDING_SCHEMA_ID:
        raise WorkerBindingError("worker binding schema identity mismatch")
    _validate_binding_shape(binding)
    _content(binding, "worker binding receipt")

    runtime = binding.get("runtime")
    if not isinstance(runtime, dict) or set(runtime) != {
        "python",
        "worker_wrapper",
        "binding_schema",
        "aoss_conf",
    }:
        raise WorkerBindingError("worker runtime binding set mismatch")
    schema_binding = runtime["binding_schema"]
    if (
        not isinstance(schema_binding, dict)
        or schema_binding.get("schema_id") != WORKER_BINDING_SCHEMA_ID
        or Path(str(schema_binding.get("path", ""))).resolve() != schema_path
    ):
        raise WorkerBindingError("binding schema receipt identity/path mismatch")
    _verify_file_binding(
        schema_binding,
        "worker binding schema",
        expected_sha=args.expected_binding_schema_sha256,
    )
    wrapper_path, _ = _verify_file_binding(
        runtime["worker_wrapper"],
        "worker wrapper",
        expected_sha=args.expected_worker_wrapper_sha256,
    )
    if wrapper_path != Path(__file__).resolve():
        raise WorkerBindingError("bound worker wrapper is not the running script")
    python_path, _ = _verify_file_binding(
        runtime["python"],
        "worker Python interpreter",
        expected_sha=args.expected_python_sha256,
    )
    if python_path != Path(sys.executable).resolve():
        raise WorkerBindingError("bound Python interpreter is not the running interpreter")
    aoss_conf, _ = _verify_file_binding(runtime["aoss_conf"], "AOSS config")

    source_plan_binding = binding.get("source_gate_plan")
    plan_path, _ = _verify_file_binding(
        source_plan_binding,
        "source gate plan",
        expected_sha=args.expected_source_plan_sha256,
    )
    plan = _read_object(plan_path, "source gate plan")
    input_paths, output_paths, source_shards = _validate_source_plan(plan, binding)

    shard_binding = binding.get("shard")
    shard_path, _ = _verify_file_binding(
        shard_binding,
        "reviewed shard file",
        expected_sha=args.expected_shard_sha256,
    )
    shard = _read_object(shard_path, "reviewed shard file")
    shard_content = _content(shard, "reviewed shard file")
    if not isinstance(shard_binding, dict):  # explicit for type checkers/review
        raise WorkerBindingError("reviewed shard binding is malformed")
    try:
        index = int(shard_binding["shard_index"])
        source_line = int(shard_binding["source_line_number"])
        expected_records = int(shard_binding["expected_request_records"])
        shard_count = int(shard_binding["shard_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkerBindingError("reviewed shard binding count metadata is invalid") from exc
    if (
        source_line != index + 1
        or index < 0
        or index >= len(source_shards)
        or source_shards[index] != shard
        or shard_binding.get("content_sha256") != shard_content
        or shard_binding.get("shard_id") != shard.get("shard_id")
        or int(shard.get("shard_index", -1)) != index
        or int(shard.get("shard_count", -1)) != shard_count
        or int(shard.get("expected_request_records", -1)) != expected_records
    ):
        raise WorkerBindingError("reviewed shard differs from source shard line/count")

    execution = binding.get("gate_execution")
    if not isinstance(execution, dict) or set(execution) != {
        "source_command_argv",
        "effective_command_argv",
        "python_override_is_explicit",
        "output_dir",
        "output_must_be_absent_before_start",
        "request_timeout_seconds",
    }:
        raise WorkerBindingError("gate execution binding is malformed")
    source_argv = shard.get("command_argv")
    effective_argv = execution.get("effective_command_argv")
    if (
        execution.get("source_command_argv") != source_argv
        or not isinstance(source_argv, list)
        or not isinstance(effective_argv, list)
        or len(source_argv) < 2
        or effective_argv != [str(python_path), *source_argv[1:]]
        or execution.get("python_override_is_explicit")
        is not (str(python_path) != source_argv[0])
        or execution.get("output_must_be_absent_before_start") is not True
        or int(execution.get("request_timeout_seconds", -1))
        != int(plan.get("request_timeout_seconds", -2))
    ):
        raise WorkerBindingError("effective gate command differs from reviewed shard")
    if effective_argv[1] != str(input_paths["gate_script"]):
        raise WorkerBindingError("effective gate command does not invoke bound gate script")
    if "--requests" not in effective_argv or effective_argv[
        effective_argv.index("--requests") + 1
    ] != str(input_paths["requests"]):
        raise WorkerBindingError("effective gate command does not use bound requests")
    if "--conf" not in effective_argv or effective_argv[
        effective_argv.index("--conf") + 1
    ] != str(aoss_conf):
        raise WorkerBindingError("effective gate command does not use bound AOSS config")
    output_dir = Path(str(execution.get("output_dir", "")))
    if (
        not output_dir.is_absolute()
        or str(shard.get("output_dir")) != str(output_dir)
        or output_dir.exists()
        or output_dir.is_symlink()
    ):
        raise WorkerBindingError("gate output directory is not an absent reviewed path")

    return {
        "binding": binding,
        "plan": plan,
        "shard": shard,
        "effective_argv": effective_argv,
        "input_paths": input_paths,
        "output_paths": output_paths,
    }


def run(args: argparse.Namespace) -> int:
    verified = verify_preflight(args)
    completed = subprocess.run(verified["effective_argv"], check=False)
    return int(completed.returncode)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--binding-receipt", required=True, type=Path)
    value.add_argument("--binding-schema", required=True, type=Path)
    value.add_argument("--expected-binding-receipt-sha256", required=True)
    value.add_argument("--expected-binding-receipt-size", required=True, type=int)
    value.add_argument("--expected-binding-schema-sha256", required=True)
    value.add_argument("--expected-source-plan-sha256", required=True)
    value.add_argument("--expected-shard-sha256", required=True)
    value.add_argument("--expected-worker-wrapper-sha256", required=True)
    value.add_argument("--expected-python-sha256", required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return run(args)
    except (OSError, WorkerBindingError) as exc:
        print(
            json.dumps(
                {"status": "FAIL_PREFLIGHT", "error": str(exc)},
                sort_keys=True,
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
