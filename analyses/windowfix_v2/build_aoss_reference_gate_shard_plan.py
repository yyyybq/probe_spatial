#!/usr/bin/env python3
"""Render a review-only shard plan for the canonical-v2 AOSS reference gate.

This renderer is deliberately offline.  It reads and validates the local,
gzip-compressed request catalogue, hashes the gate implementation and receipt
schema, and writes commands for later operator review.  It never imports an
AOSS client, contacts a remote service, executes a gate command, or submits a
job.

The gate runner partitions requests by zero-based JSONL record offset modulo
``shard_count``.  The plan records the exact expected count for every shard so
that later receipts can be checked for omissions before any route promotion.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shlex
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REQUESTS = (
    ROOT
    / "artifacts/windowfix_v2/reference_overlay_02e17c40b59a_r11/"
    "remote_gate_requests.jsonl.gz"
)
DEFAULT_CATALOG_MANIFEST = DEFAULT_REQUESTS.parent / "catalog_manifest.json"
DEFAULT_CATALOG_SUMMARY = DEFAULT_REQUESTS.parent / "summary.json"
DEFAULT_NAMESPACE_LOCK = (
    ROOT / "artifacts/windowfix_v2/canonical_namespace_lock.02e17c40b59a.json"
)
DEFAULT_GATE_SCRIPT = ROOT / "analyses/windowfix_v2/gate_aoss_streaming_references.py"
DEFAULT_RECEIPT_SCHEMA = (
    ROOT / "protocols/legacy_exact_reference_remote_gate_receipt_v1.schema.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "artifacts/windowfix_v2/"
    "aoss_reference_gate_plan_r11_hardened_r12_128shards_20260831"
)
DEFAULT_RECEIPT_OUTPUT_ROOT = (
    ROOT
    / "artifacts/windowfix_v2/"
    "aoss_reference_gate_receipts_r11_128shards_20260831"
)
DEFAULT_AOSS_CONF = Path("/mnt/umm/users/yinbaiqiao/aoss/petreloss.conf")
DEFAULT_SHARD_COUNT = 128
REQUEST_TIMEOUT_SECONDS = 30
EXPECTED_CANONICAL_SOURCE_SHA256 = (
    "ebc496eb0587cda372ccf29a7c5244ef454096151f73c8eed9511967ded1902d"
)
EXPECTED_CANONICAL_UNIVERSE_SHA256 = (
    "4771e2859c43889bd5fb75f5876bcf493758f4187336c9bd7eeca5432acbac9b"
)

PLAN_SCHEMA = "LEGACY_EXACT_REFERENCE_REMOTE_GATE_SHARD_PLAN_V1"
SHARD_SCHEMA = "LEGACY_EXACT_REFERENCE_REMOTE_GATE_SHARD_V1"
REQUEST_SCHEMA = "LEGACY_EXACT_REFERENCE_REMOTE_GATE_REQUEST_V1"
RECEIPT_SCHEMA = "LEGACY_EXACT_REFERENCE_REMOTE_GATE_RECEIPT_V1"
PARTITION_RULE = "zero_based_jsonl_record_offset_mod_shard_count"


class PlanError(RuntimeError):
    """Raised when an input cannot support a fail-closed gate plan."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_sha256(value: dict[str, Any]) -> str:
    unhashed = {key: item for key, item in value.items() if key != "content_sha256"}
    return sha256_bytes(canonical_json(unhashed).encode("utf-8"))


def _require_regular_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_file():
        raise PlanError(f"{label} is not a non-symlink regular file: {path}")
    return resolved


def scan_requests(
    path: Path, shard_count: int
) -> tuple[int, list[int], str, dict[str, int]]:
    """Validate the local request stream and derive exact modulo shard counts."""

    if shard_count <= 0:
        raise PlanError("shard_count must be positive")
    counts = [0] * shard_count
    schemas: dict[str, int] = {}
    seen_unit_ids: set[str] = set()
    logical_digest = hashlib.sha256()
    total = 0
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rb") as handle:
            for line_number, raw_line in enumerate(handle, 1):
                if not raw_line.strip():
                    raise PlanError(f"request catalogue has a blank line at {line_number}")
                logical_digest.update(raw_line)
                try:
                    request = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise PlanError(
                        f"request catalogue has invalid JSON at line {line_number}: {exc}"
                    ) from exc
                if not isinstance(request, dict):
                    raise PlanError(f"request line {line_number} is not an object")
                schema = str(request.get("schema", ""))
                schemas[schema] = schemas.get(schema, 0) + 1
                if schema != REQUEST_SCHEMA:
                    raise PlanError(
                        f"request line {line_number} has unexpected schema {schema!r}"
                    )
                unit_id = str(request.get("unit_id", ""))
                if not unit_id:
                    raise PlanError(f"request line {line_number} lacks unit_id")
                if unit_id in seen_unit_ids:
                    raise PlanError(f"duplicate request unit_id: {unit_id}")
                seen_unit_ids.add(unit_id)
                for field in (
                    "source_uri",
                    "sidecar_uri",
                    "source_index_uri",
                    "source_index_record",
                    "expected",
                ):
                    if field not in request:
                        raise PlanError(f"request line {line_number} lacks {field}")
                counts[total % shard_count] += 1
                total += 1
    except (OSError, EOFError) as exc:
        raise PlanError(f"cannot read request catalogue {path}: {exc}") from exc
    if total == 0:
        raise PlanError("request catalogue is empty")
    return total, counts, logical_digest.hexdigest(), schemas


def _render_command(argv: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in argv)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _input_binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanError(f"cannot parse {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PlanError(f"{label} is not a JSON object: {path}")
    return value


def _catalog_expectations(
    *,
    requests: Path,
    catalog_manifest: Path,
    catalog_summary: Path,
    namespace_lock: Path,
) -> tuple[str, int, dict[str, Any]]:
    """Derive immutable request expectations and validate canonical lineage."""

    manifest = _read_json_object(catalog_manifest, "catalog manifest")
    summary = _read_json_object(catalog_summary, "catalog summary")
    lock = _read_json_object(namespace_lock, "canonical namespace lock")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        raise PlanError("catalog manifest lacks outputs list")
    matches = [
        item
        for item in outputs
        if isinstance(item, dict) and item.get("file") == requests.name
    ]
    if len(matches) != 1:
        raise PlanError(
            f"catalog manifest must bind {requests.name} exactly once, found {len(matches)}"
        )
    expected_sha = str(matches[0].get("sha256", "")).lower()
    if len(expected_sha) != 64:
        raise PlanError("catalog manifest request binding lacks a valid SHA-256")
    if int(matches[0].get("size_bytes", -1)) != requests.stat().st_size:
        raise PlanError("catalog manifest request size binding disagrees with local file")
    summary_matches = [
        item
        for item in outputs
        if isinstance(item, dict) and item.get("file") == catalog_summary.name
    ]
    if len(summary_matches) != 1 or summary_matches[0].get("sha256") != sha256_file(
        catalog_summary
    ):
        raise PlanError("catalog manifest does not bind the supplied summary exactly")
    try:
        expected_count = int(summary["counts"]["status"]["PENDING_REMOTE_GATE"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PlanError("catalog summary lacks PENDING_REMOTE_GATE count") from exc
    if int(summary.get("remote_policy", {}).get("pending_units", -1)) != expected_count:
        raise PlanError("catalog summary remote pending count disagrees with status count")
    try:
        source_sha = str(lock["canonical_source"]["sha256"])
        universe_sha = str(lock["canonical_universe_sha256"])
    except (KeyError, TypeError) as exc:
        raise PlanError("canonical namespace lock lacks source/universe lineage") from exc
    if source_sha != EXPECTED_CANONICAL_SOURCE_SHA256:
        raise PlanError(
            "canonical namespace lock source SHA mismatch: "
            f"expected {EXPECTED_CANONICAL_SOURCE_SHA256}, found {source_sha}"
        )
    if universe_sha != EXPECTED_CANONICAL_UNIVERSE_SHA256:
        raise PlanError(
            "canonical namespace lock universe SHA mismatch: "
            f"expected {EXPECTED_CANONICAL_UNIVERSE_SHA256}, found {universe_sha}"
        )
    for label, value in (("catalog manifest", manifest), ("catalog summary", summary)):
        if value.get("canonical_parent_source_sha256") != source_sha:
            raise PlanError(f"{label} canonical parent SHA disagrees with namespace lock")
        if value.get("canonical_universe_sha256") != universe_sha:
            raise PlanError(f"{label} canonical universe SHA disagrees with namespace lock")
    declared_lock = summary.get("canonical_namespace_lock")
    if not declared_lock or Path(str(declared_lock)).resolve() != namespace_lock:
        raise PlanError("catalog summary does not bind the supplied canonical namespace lock")
    claims = summary.get("claims")
    if not isinstance(claims, dict) or int(claims.get("jobs_submitted", -1)) != 0:
        raise PlanError("catalog summary does not prove jobs_submitted=0")
    bindings = {
        "catalog_manifest": _input_binding(catalog_manifest),
        "catalog_summary": _input_binding(catalog_summary),
        "canonical_namespace_lock": {
            **_input_binding(namespace_lock),
            "canonical_source_sha256": source_sha,
            "canonical_universe_sha256": universe_sha,
        },
    }
    return expected_sha, expected_count, bindings


def generate(
    *,
    requests: Path,
    catalog_manifest: Path,
    catalog_summary: Path,
    namespace_lock: Path,
    gate_script: Path,
    receipt_schema: Path,
    output_dir: Path,
    receipt_output_root: Path,
    aoss_conf: Path,
    shard_count: int = DEFAULT_SHARD_COUNT,
    request_timeout_seconds: int = REQUEST_TIMEOUT_SECONDS,
    expected_request_count: int | None = None,
    expected_requests_sha256: str | None = None,
    python: str = "python3",
    plan_id: str = "aoss-reference-gate-r11-hardened-r12-128shards-20260831",
    created_at: str | None = None,
) -> dict[str, Any]:
    """Generate the offline review bundle and return its final plan object."""

    if request_timeout_seconds != REQUEST_TIMEOUT_SECONDS:
        raise PlanError(
            f"this plan contract requires timeout={REQUEST_TIMEOUT_SECONDS}, "
            f"found {request_timeout_seconds}"
        )
    if shard_count <= 0:
        raise PlanError("shard_count must be positive")
    if not plan_id.strip():
        raise PlanError("plan_id must be non-empty")

    requests = _require_regular_file(requests, "request catalogue")
    catalog_manifest = _require_regular_file(catalog_manifest, "catalog manifest")
    catalog_summary = _require_regular_file(catalog_summary, "catalog summary")
    namespace_lock = _require_regular_file(namespace_lock, "canonical namespace lock")
    gate_script = _require_regular_file(gate_script, "gate script")
    receipt_schema = _require_regular_file(receipt_schema, "receipt schema")
    if catalog_manifest.parent != requests.parent or catalog_summary.parent != requests.parent:
        raise PlanError("requests, catalog manifest, and catalog summary must share one directory")
    output_dir = output_dir.resolve()
    receipt_output_root = receipt_output_root.resolve()
    if output_dir.exists():
        raise PlanError(f"output already exists: {output_dir}")

    catalog_sha, catalog_count, catalog_bindings = _catalog_expectations(
        requests=requests,
        catalog_manifest=catalog_manifest,
        catalog_summary=catalog_summary,
        namespace_lock=namespace_lock,
    )
    if expected_requests_sha256 is not None and expected_requests_sha256.lower() != catalog_sha:
        raise PlanError("explicit request SHA disagrees with immutable catalog manifest")
    if expected_request_count is not None and expected_request_count != catalog_count:
        raise PlanError("explicit request count disagrees with immutable catalog summary")
    expected_requests_sha256 = catalog_sha
    expected_request_count = catalog_count
    observed_requests_sha = sha256_file(requests)
    if observed_requests_sha != expected_requests_sha256:
        raise PlanError(
            "request catalogue SHA-256 mismatch: "
            f"expected {expected_requests_sha256.lower()}, observed {observed_requests_sha}"
        )
    request_count, per_shard, logical_sha, request_schemas = scan_requests(
        requests, shard_count
    )
    if request_count != expected_request_count:
        raise PlanError(
            f"request count mismatch: expected {expected_request_count}, found {request_count}"
        )
    if sum(per_shard) != request_count or len(per_shard) != shard_count:
        raise PlanError("internal shard partition count mismatch")

    try:
        receipt_schema_value = json.loads(receipt_schema.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanError(f"cannot parse receipt schema: {exc}") from exc
    observed_receipt_contract = (
        receipt_schema_value.get("properties", {}).get("schema", {}).get("const")
        if isinstance(receipt_schema_value, dict)
        else None
    )
    if observed_receipt_contract != RECEIPT_SCHEMA:
        raise PlanError(
            "receipt schema contract mismatch: "
            f"expected {RECEIPT_SCHEMA}, found {observed_receipt_contract!r}"
        )

    created_at = created_at or datetime.now(timezone.utc).isoformat()
    inputs = {
        "plan_renderer": _input_binding(Path(__file__).resolve()),
        "requests": {
            **_input_binding(requests),
            "records": request_count,
            "uncompressed_jsonl_sha256": logical_sha,
            "record_schema_counts": request_schemas,
        },
        "gate_script": _input_binding(gate_script),
        "receipt_schema": {
            **_input_binding(receipt_schema),
            "receipt_contract": RECEIPT_SCHEMA,
        },
        **catalog_bindings,
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=str(output_dir.parent))
    )
    try:
        shards: list[dict[str, Any]] = []
        command_lines: list[str] = []
        width = max(4, len(str(shard_count - 1)))
        for shard_index, expected_records in enumerate(per_shard):
            shard_id = f"gate-{shard_index:0{width}d}-of-{shard_count:0{width}d}"
            shard_output = receipt_output_root / shard_id
            argv = [
                python,
                str(gate_script),
                "--requests",
                str(requests),
                "--conf",
                str(aoss_conf),
                "--output-dir",
                str(shard_output),
                "--shard-index",
                str(shard_index),
                "--shard-count",
                str(shard_count),
                "--request-timeout-seconds",
                str(request_timeout_seconds),
            ]
            shard = {
                "schema": SHARD_SCHEMA,
                "plan_id": plan_id,
                "shard_id": shard_id,
                "status": "REVIEW_ONLY",
                "review_only": True,
                "submissions": 0,
                "remote_reads_performed": 0,
                "commands_executed": 0,
                "partition_rule": PARTITION_RULE,
                "shard_index": shard_index,
                "shard_count": shard_count,
                "expected_request_records": expected_records,
                "request_timeout_seconds": request_timeout_seconds,
                "requests": inputs["requests"],
                "gate_script": inputs["gate_script"],
                "receipt_schema": inputs["receipt_schema"],
                "aoss_conf_path_unread": str(aoss_conf),
                "output_dir": str(shard_output),
                "expected_receipts_file": str(shard_output / "receipts.jsonl.gz"),
                "expected_summary_file": str(shard_output / "summary.json"),
                "command_argv": argv,
                "command": _render_command(argv),
            }
            shard["content_sha256"] = content_sha256(shard)
            shards.append(shard)
            command_lines.append(shard["command"])

        shards_path = stage / "shards.jsonl"
        shards_path.write_text(
            "".join(canonical_json(shard) + "\n" for shard in shards),
            encoding="utf-8",
        )
        commands_path = stage / "commands.sh"
        commands_path.write_text(
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    "# REVIEW-ONLY ARTIFACT. The generator did not execute these commands.",
                    "# Running a command below performs AOSS reads; do so only after review.",
                    "# No command submits an SCO job or starts probing/extraction.",
                    "set -euo pipefail",
                    "",
                    f"REQUESTS={shlex.quote(str(requests))}",
                    f"GATE_SCRIPT={shlex.quote(str(gate_script))}",
                    f"RECEIPT_SCHEMA={shlex.quote(str(receipt_schema))}",
                    "",
                    f"# requests sha256: {inputs['requests']['sha256']}",
                    f"# gate script sha256: {inputs['gate_script']['sha256']}",
                    f"# receipt schema sha256: {inputs['receipt_schema']['sha256']}",
                    f"[[ \"$(sha256sum \"$REQUESTS\" | cut -d ' ' -f 1)\" == "
                    f"\"{inputs['requests']['sha256']}\" ]] || "
                    "{ echo 'request SHA mismatch' >&2; exit 2; }",
                    f"[[ \"$(sha256sum \"$GATE_SCRIPT\" | cut -d ' ' -f 1)\" == "
                    f"\"{inputs['gate_script']['sha256']}\" ]] || "
                    "{ echo 'gate script SHA mismatch' >&2; exit 2; }",
                    f"[[ \"$(sha256sum \"$RECEIPT_SCHEMA\" | cut -d ' ' -f 1)\" == "
                    f"\"{inputs['receipt_schema']['sha256']}\" ]] || "
                    "{ echo 'receipt schema SHA mismatch' >&2; exit 2; }",
                    "",
                    *command_lines,
                    "",
                ]
            ),
            encoding="utf-8",
        )
        # It is intentionally a review document, not an executable submission script.
        commands_path.chmod(0o644)

        quotient, remainder = divmod(request_count, shard_count)
        readme_path = stage / "README.md"
        readme_path.write_text(
            "\n".join(
                [
                    f"# AOSS reference gate shard plan ({plan_id})",
                    "",
                    "This directory is an offline, review-only plan. Its generator made zero ",
                    "remote reads, executed zero gate commands, and submitted zero jobs.",
                    "",
                    f"- Requests: `{request_count}` records; SHA-256 `{observed_requests_sha}`.",
                    f"- Gate implementation SHA-256: `{inputs['gate_script']['sha256']}`.",
                    f"- Receipt schema SHA-256: `{inputs['receipt_schema']['sha256']}`.",
                    f"- Partition: `{PARTITION_RULE}` into `{shard_count}` shards.",
                    f"- Per-request timeout: `{request_timeout_seconds}` seconds.",
                    f"- Distribution: `{remainder}` shards contain `{quotient + 1}` records and "
                    f"`{shard_count - remainder}` contain `{quotient}` records.",
                    "",
                    "`shards.jsonl` is the machine-readable source of truth. `commands.sh` lists ",
                    "one direct gate command per shard and is intentionally mode 0644. Running ",
                    "those commands later will access AOSS; generation of this plan did not run ",
                    "them. The receipt output root is separate from this immutable plan directory.",
                    "",
                    "A shard result must contain exactly its `expected_request_records` before ",
                    "receipts are eligible for finalization. A network error remains pending, and ",
                    "no request is promoted merely because it appears in this plan.",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        outputs = {
            name: _input_binding(stage / name)
            for name in ("shards.jsonl", "commands.sh", "README.md")
        }
        # Store final paths rather than temporary staging paths in output bindings.
        for name, binding in outputs.items():
            binding["path"] = str(output_dir / name)
        plan = {
            "schema": PLAN_SCHEMA,
            "plan_id": plan_id,
            "created_at": created_at,
            "status": "REVIEW_ONLY",
            "review_only": True,
            "submissions": 0,
            "remote_reads_performed": 0,
            "commands_executed": 0,
            "feature_payloads_copied": 0,
            "probing_runs_started": 0,
            "partition_rule": PARTITION_RULE,
            "request_timeout_seconds": request_timeout_seconds,
            "aoss_conf_path_unread": str(aoss_conf),
            "receipt_output_root": str(receipt_output_root),
            "inputs": inputs,
            "counts": {
                "requests": request_count,
                "shards": shard_count,
                "commands_rendered": len(command_lines),
                "expected_receipts": request_count,
                "minimum_records_per_shard": min(per_shard),
                "maximum_records_per_shard": max(per_shard),
                "shards_with_extra_record": remainder,
            },
            "shard_expected_records": per_shard,
            "outputs": outputs,
            "execution_policy": {
                "submission_policy": "REVIEW_ONLY_NEVER_SUBMITTED_BY_RENDERER",
                "commands_file_mode": "0644",
                "remote_access_during_render": False,
                "operator_review_required_before_remote_gate": True,
                "receipt_count_must_equal_expected_before_finalization": True,
            },
        }
        plan["content_sha256"] = content_sha256(plan)
        _write_json(stage / "plan.json", plan)
        os.replace(stage, output_dir)
        return plan
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, default=DEFAULT_REQUESTS)
    parser.add_argument(
        "--catalog-manifest", type=Path, default=DEFAULT_CATALOG_MANIFEST
    )
    parser.add_argument("--catalog-summary", type=Path, default=DEFAULT_CATALOG_SUMMARY)
    parser.add_argument("--namespace-lock", type=Path, default=DEFAULT_NAMESPACE_LOCK)
    parser.add_argument("--gate-script", type=Path, default=DEFAULT_GATE_SCRIPT)
    parser.add_argument("--receipt-schema", type=Path, default=DEFAULT_RECEIPT_SCHEMA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--receipt-output-root", type=Path, default=DEFAULT_RECEIPT_OUTPUT_ROOT
    )
    parser.add_argument("--aoss-conf", type=Path, default=DEFAULT_AOSS_CONF)
    parser.add_argument("--shard-count", type=int, default=DEFAULT_SHARD_COUNT)
    parser.add_argument(
        "--request-timeout-seconds", type=int, default=REQUEST_TIMEOUT_SECONDS
    )
    parser.add_argument("--expected-request-count", type=int)
    parser.add_argument("--expected-requests-sha256")
    parser.add_argument("--python", default="python3")
    parser.add_argument(
        "--plan-id",
        default="aoss-reference-gate-r11-hardened-r12-128shards-20260831",
    )
    parser.add_argument("--created-at")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        plan = generate(
            requests=args.requests,
            catalog_manifest=args.catalog_manifest,
            catalog_summary=args.catalog_summary,
            namespace_lock=args.namespace_lock,
            gate_script=args.gate_script,
            receipt_schema=args.receipt_schema,
            output_dir=args.output_dir,
            receipt_output_root=args.receipt_output_root,
            aoss_conf=args.aoss_conf,
            shard_count=args.shard_count,
            request_timeout_seconds=args.request_timeout_seconds,
            expected_request_count=args.expected_request_count,
            expected_requests_sha256=args.expected_requests_sha256,
            python=args.python,
            plan_id=args.plan_id,
            created_at=args.created_at,
        )
    except PlanError as exc:
        raise SystemExit(f"ERROR: {exc}")
    print(json.dumps(plan["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
