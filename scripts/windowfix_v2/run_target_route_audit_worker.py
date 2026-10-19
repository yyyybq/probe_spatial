#!/usr/bin/env python3
"""Fail-closed launcher for one remote target-row route audit shard.

Only Python's standard library is imported until every launch dependency has
been checked against a SHA supplied on the command line.  In particular, the
auditor (which imports the AOSS-capable source reader) is started only after
the parent plan, shard, auditor, source reader, this wrapper, binding receipt,
and binding schema all pass their independent byte and content bindings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
RECEIPT_TYPE = "canonical_v2_target_route_audit_worker_binding"
RECEIPT_STATUS = "REVIEW_ONLY_NOT_EXECUTED"
SCHEMA_ID = "canonical_v2_target_route_audit_worker_binding_v1.schema.json"
PLAN_TYPE = "canonical_v2_target_row_merge_plan"
PLAN_CONTRACT = "CANONICAL_V2_TARGET_ROW_MERGE_V1"
HEX = frozenset("0123456789abcdef")


class WorkerBindingError(ValueError):
    """A worker launch input is not exactly the artifact that was reviewed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _sha(value: Any, label: str) -> str:
    text = str(value or "").lower()
    if len(text) != 64 or set(text) - HEX:
        raise WorkerBindingError(f"{label} must be a lowercase SHA256")
    return text


def _regular(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise WorkerBindingError(
            f"{label} must be an absolute regular non-symlink file: {path}"
        )
    return path.resolve()


def _verified_file(path: Path, expected_sha256: str, label: str) -> Path:
    path = _regular(path, label)
    expected = _sha(expected_sha256, f"expected {label} SHA256")
    actual = sha256_file(path)
    if actual != expected:
        raise WorkerBindingError(
            f"{label} SHA256 mismatch: expected {expected}, found {actual}"
        )
    return path


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkerBindingError(f"cannot parse {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise WorkerBindingError(f"{label} must contain a JSON object: {path}")
    return value


def _content_sha(value: Mapping[str, Any], label: str) -> str:
    recorded = _sha(value.get("content_sha256"), f"{label} content SHA256")
    actual = canonical_json_sha(
        {key: item for key, item in value.items() if key != "content_sha256"}
    )
    if recorded != actual:
        raise WorkerBindingError(
            f"{label} content SHA256 mismatch: recorded {recorded}, found {actual}"
        )
    return actual


def _binding(value: Any, label: str, *, require_content: bool = False) -> dict[str, str]:
    if not isinstance(value, dict):
        raise WorkerBindingError(f"binding receipt {label} must be an object")
    allowed = {"path", "file_sha256"}
    required = set(allowed)
    if require_content:
        allowed.add("content_sha256")
        required.add("content_sha256")
    if label == "binding_schema":
        allowed.add("schema_id")
        required.add("schema_id")
    if set(value) != allowed:
        raise WorkerBindingError(
            f"binding receipt {label} fields mismatch: expected {sorted(allowed)}"
        )
    if not required <= set(value):  # pragma: no cover - made explicit for review
        raise WorkerBindingError(f"binding receipt {label} is incomplete")
    path = Path(str(value["path"]))
    result = {
        "path": str(path),
        "file_sha256": _sha(value["file_sha256"], f"binding receipt {label} file SHA256"),
    }
    if require_content:
        result["content_sha256"] = _sha(
            value["content_sha256"], f"binding receipt {label} content SHA256"
        )
    if label == "binding_schema":
        if value.get("schema_id") != SCHEMA_ID:
            raise WorkerBindingError("binding receipt schema identity mismatch")
        result["schema_id"] = SCHEMA_ID
    return result


def _validate_receipt_shape(receipt: Mapping[str, Any]) -> None:
    expected_fields = {
        "schema_version",
        "receipt_type",
        "status",
        "parent_plan",
        "shard",
        "auditor",
        "source_reader",
        "worker_wrapper",
        "binding_schema",
        "audit_output",
        "audit_arguments",
        "content_sha256",
    }
    if set(receipt) != expected_fields:
        raise WorkerBindingError(
            "binding receipt top-level fields mismatch: "
            f"expected {sorted(expected_fields)}"
        )
    if receipt.get("schema_version") != SCHEMA_VERSION:
        raise WorkerBindingError("binding receipt schema version mismatch")
    if receipt.get("receipt_type") != RECEIPT_TYPE:
        raise WorkerBindingError("binding receipt type mismatch")
    if receipt.get("status") != RECEIPT_STATUS:
        raise WorkerBindingError("binding receipt status mismatch")
    arguments = receipt.get("audit_arguments")
    if not isinstance(arguments, dict) or set(arguments) != {
        "remote_policy",
        "source_timeout_seconds",
        "verify_source_sha",
    }:
        raise WorkerBindingError("binding receipt audit arguments are malformed")
    if arguments.get("remote_policy") != "gate" or arguments.get("verify_source_sha") is not True:
        raise WorkerBindingError("binding receipt weakens the remote/source SHA gate")
    timeout = arguments.get("source_timeout_seconds")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise WorkerBindingError("binding receipt source timeout must be positive")
    output = Path(str(receipt.get("audit_output", "")))
    if not output.is_absolute() or output.exists() or output.is_symlink():
        raise WorkerBindingError(
            "binding receipt audit_output must be an absent absolute path"
        )


def verify_preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Verify every launch dependency without importing AOSS-capable code."""
    receipt_path = _verified_file(
        args.binding_receipt,
        args.expected_binding_receipt_sha256,
        "worker binding receipt",
    )
    schema_path = _verified_file(
        args.binding_schema,
        args.expected_binding_schema_sha256,
        "worker binding schema",
    )
    receipt = _read_object(receipt_path, "worker binding receipt")
    schema = _read_object(schema_path, "worker binding schema")
    if schema.get("$id") != SCHEMA_ID:
        raise WorkerBindingError("worker binding schema $id mismatch")
    _validate_receipt_shape(receipt)
    receipt_content_sha = _content_sha(receipt, "worker binding receipt")

    expected = {
        "parent_plan": _sha(args.expected_parent_plan_sha256, "expected parent plan SHA256"),
        "shard": _sha(args.expected_shard_sha256, "expected shard SHA256"),
        "auditor": _sha(args.expected_auditor_sha256, "expected auditor SHA256"),
        "source_reader": _sha(
            args.expected_source_reader_sha256, "expected source reader SHA256"
        ),
        "worker_wrapper": _sha(
            args.expected_worker_wrapper_sha256, "expected worker wrapper SHA256"
        ),
        "binding_schema": _sha(
            args.expected_binding_schema_sha256, "expected binding schema SHA256"
        ),
    }
    bindings: dict[str, dict[str, str]] = {}
    for label in expected:
        bindings[label] = _binding(
            receipt.get(label),
            label,
            require_content=label in {"parent_plan", "shard"},
        )
        if bindings[label]["file_sha256"] != expected[label]:
            raise WorkerBindingError(
                f"binding receipt {label} SHA256 differs from worker command"
            )

    # The self-check is intentionally complete before any audited source can
    # be opened.  It catches ordinary or same-size edits made after rendering.
    wrapper_path = _regular(Path(bindings["worker_wrapper"]["path"]), "worker wrapper")
    if wrapper_path.resolve() != Path(__file__).resolve():
        raise WorkerBindingError("worker wrapper path does not name the running script")
    _verified_file(wrapper_path, expected["worker_wrapper"], "worker wrapper")
    if Path(bindings["binding_schema"]["path"]).resolve() != schema_path:
        raise WorkerBindingError("binding schema path differs from worker command")

    verified_paths: dict[str, Path] = {}
    for label in ("parent_plan", "shard", "auditor", "source_reader"):
        path = Path(bindings[label]["path"])
        verified_paths[label] = _verified_file(path, expected[label], label.replace("_", " "))
    expected_reader_path = verified_paths["auditor"].with_name("target_row_source.py")
    if verified_paths["source_reader"] != expected_reader_path:
        raise WorkerBindingError(
            "bound source reader is not the module imported by the bound auditor"
        )

    parent = _read_object(verified_paths["parent_plan"], "parent plan")
    shard = _read_object(verified_paths["shard"], "shard plan")
    parent_content = _content_sha(parent, "parent plan")
    shard_content = _content_sha(shard, "shard plan")
    if parent.get("plan_type") != PLAN_TYPE or parent.get("contract") != PLAN_CONTRACT:
        raise WorkerBindingError("parent plan contract/type mismatch")
    if shard.get("plan_type") != PLAN_TYPE or shard.get("contract") != PLAN_CONTRACT:
        raise WorkerBindingError("shard plan contract/type mismatch")
    if bindings["parent_plan"]["content_sha256"] != parent_content:
        raise WorkerBindingError("parent plan content SHA differs from binding receipt")
    if bindings["shard"]["content_sha256"] != shard_content:
        raise WorkerBindingError("shard content SHA differs from binding receipt")
    parent_record = shard.get("parent_plan")
    expected_parent_record = {
        "path": str(verified_paths["parent_plan"]),
        "file_sha256": expected["parent_plan"],
        "content_sha256": parent_content,
    }
    if parent_record != expected_parent_record:
        raise WorkerBindingError("shard parent_plan binding mismatch")

    return {
        "receipt": receipt,
        "receipt_path": receipt_path,
        "receipt_content_sha256": receipt_content_sha,
        "schema_path": schema_path,
        "paths": verified_paths,
    }


def run(args: argparse.Namespace) -> int:
    verified = verify_preflight(args)
    receipt = verified["receipt"]
    arguments = receipt["audit_arguments"]
    argv = [
        str(Path(sys.executable).resolve()),
        str(verified["paths"]["auditor"]),
        "--plan",
        str(verified["paths"]["shard"]),
        "--output-dir",
        str(receipt["audit_output"]),
        "--remote-policy",
        "gate",
        "--source-timeout-seconds",
        str(arguments["source_timeout_seconds"]),
        "--verify-source-sha",
    ]
    completed = subprocess.run(argv, check=False)
    return int(completed.returncode)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--binding-receipt", required=True, type=Path)
    value.add_argument("--binding-schema", required=True, type=Path)
    value.add_argument("--expected-binding-receipt-sha256", required=True)
    value.add_argument("--expected-binding-schema-sha256", required=True)
    value.add_argument("--expected-parent-plan-sha256", required=True)
    value.add_argument("--expected-shard-sha256", required=True)
    value.add_argument("--expected-auditor-sha256", required=True)
    value.add_argument("--expected-source-reader-sha256", required=True)
    value.add_argument("--expected-worker-wrapper-sha256", required=True)
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
