#!/usr/bin/env python3
"""Join AOSS gate receipts to the lineage-bound frozen r11 pending routes.

This finalizer is deliberately partial-run safe.  A missing receipt or a
receipt whose gate status is still pending remains ``PENDING_REMOTE_GATE``.
Only a fully checked PASS becomes a read-only reference; a definitive FAIL
becomes a complete ``REEXTRACT_UNVERIFIED`` route consumable by
``build_streaming_gap_worklists.py --failed-gate-routes``.  Duplicate, extra,
or request-SHA-inconsistent receipts are hard errors.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import sqlite3
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, TextIO


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = ROOT / "artifacts/windowfix_v2/reference_overlay_02e17c40b59a_r11"
CATALOG_SCHEMA = "CANONICAL_V2_STREAMING_ROUTE_CATALOG_V1"
RECEIPT_SCHEMA = "LEGACY_EXACT_REFERENCE_REMOTE_GATE_RECEIPT_V1"
FINALIZER_SCHEMA = "LEGACY_EXACT_REFERENCE_REMOTE_GATE_FINALIZATION_V1"
NAMESPACE = "FEAT_STREAMING_CANONICAL_V2_MIXED_REFERENCE_V1"
CANONICAL_PARENT_SOURCE_SHA256 = (
    "ebc496eb0587cda372ccf29a7c5244ef454096151f73c8eed9511967ded1902d"
)
CANONICAL_UNIVERSE_SHA256 = (
    "4771e2859c43889bd5fb75f5876bcf493758f4187336c9bd7eeca5432acbac9b"
)
PASS = "REFERENCE_SOURCE_READ_ONLY"
FAIL = "REEXTRACT_UNVERIFIED"
PENDING = "PENDING_REMOTE_GATE"
DTYPE_ALIASES = {
    "F16": "float16",
    "BF16": "bfloat16",
    "F32": "float32",
    "F64": "float64",
    "I64": "int64",
    "I32": "int32",
    "I16": "int16",
    "I8": "int8",
    "U8": "uint8",
    "BOOL": "bool",
}

# A PASS is a provenance decision.  Requiring merely a non-empty mapping of
# true values is fail-open because a truncated receipt could otherwise be
# promoted.  This is the complete check contract emitted by
# gate_aoss_streaming_references.py.
REQUIRED_PASS_CHECKS = frozenset(
    {
        "source_index_record_exists",
        "source_index_raw_ids_exact",
        "sidecar_file",
        "sidecar_size",
        "sidecar_vfm",
        "sidecar_model",
        "sidecar_mode",
        "sidecar_frame_index_schema",
        "sidecar_resize",
        "sidecar_num_frames",
        "sidecar_timestep",
        "sidecar_prompt",
        "sidecar_tensor_shape_present",
        "sidecar_tensor_shape_equals_header",
        "header_shape_equals_expected",
        "sidecar_tensor_dtype_equals_header",
        "header_dtype_equals_expected",
    }
)


class FinalizeError(RuntimeError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def normalize_dtype(value: Any) -> str:
    text = str(value).strip()
    text = text.removeprefix("torch.").removeprefix("numpy.").removeprefix("np.")
    return DTYPE_ALIASES.get(text.upper(), text.lower())


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FinalizeError(f"{label} is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalizeError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FinalizeError(f"{label} is not a JSON object: {path}")
    return value


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise FinalizeError(f"JSONL input is not a regular file: {path}")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FinalizeError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise FinalizeError(f"non-object JSON at {path}:{line_number}")
            yield value


def open_gzip_text(path: Path) -> TextIO:
    raw = path.open("wb")
    zipped = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    return io.TextIOWrapper(zipped, encoding="utf-8", newline="\n")


def catalog_bindings(catalog: Path) -> dict[str, Any]:
    manifest = read_json(catalog / "catalog_manifest.json", "catalog manifest")
    summary = read_json(catalog / "summary.json", "catalog summary")
    if manifest.get("schema") != "CANONICAL_V2_STREAMING_ROUTE_CATALOG_MANIFEST_V1":
        raise FinalizeError("catalog manifest schema mismatch")
    if summary.get("schema") != CATALOG_SCHEMA or summary.get("namespace") != NAMESPACE:
        raise FinalizeError("catalog summary schema/namespace mismatch")
    for value, expected, label in (
        (
            summary.get("canonical_parent_source_sha256"),
            CANONICAL_PARENT_SOURCE_SHA256,
            "summary canonical parent",
        ),
        (
            summary.get("canonical_universe_sha256"),
            CANONICAL_UNIVERSE_SHA256,
            "summary canonical universe",
        ),
        (
            manifest.get("canonical_parent_source_sha256"),
            CANONICAL_PARENT_SOURCE_SHA256,
            "manifest canonical parent",
        ),
        (
            manifest.get("canonical_universe_sha256"),
            CANONICAL_UNIVERSE_SHA256,
            "manifest canonical universe",
        ),
    ):
        if value != expected:
            raise FinalizeError(f"{label} SHA mismatch: {value!r}")
    output_shas: dict[str, str] = {}
    for item in manifest.get("outputs", ()):
        name = str(item.get("file", ""))
        path = catalog / name
        if path.parent != catalog or not path.is_file():
            raise FinalizeError(f"invalid catalog output binding: {name}")
        actual = sha256_file(path)
        if actual != item.get("sha256") or path.stat().st_size != item.get("size_bytes"):
            raise FinalizeError(f"catalog output binding changed: {name}")
        output_shas[name] = actual
    required = {
        "remote_gate_requests.jsonl.gz",
        "streaming_pending_remote_gate.routes.jsonl.gz",
        "summary.json",
    }
    if not required <= output_shas.keys():
        raise FinalizeError(f"catalog lacks required outputs: {sorted(required-output_shas.keys())}")
    return {"manifest": manifest, "summary": summary, "output_shas": output_shas}


def make_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute(
        "CREATE TABLE expected ("
        "ordinal INTEGER NOT NULL, unit_id TEXT PRIMARY KEY, request_sha TEXT NOT NULL, "
        "source_uri TEXT NOT NULL, source_index_uri TEXT NOT NULL, "
        "expected_shape_json TEXT, expected_dtype TEXT NOT NULL, route_json TEXT)"
    )
    connection.execute(
        "CREATE TABLE receipts (unit_id TEXT PRIMARY KEY, receipt_json TEXT NOT NULL)"
    )
    return connection


def load_expected(
    connection: sqlite3.Connection, request_path: Path, pending_path: Path
) -> int:
    request_count = 0
    for ordinal, request in enumerate(iter_jsonl(request_path)):
        unit_id = str(request.get("unit_id", ""))
        if not unit_id:
            raise FinalizeError("gate request lacks unit_id")
        request_sha = sha256_bytes(canonical_json(request).encode("utf-8"))
        expected = request.get("expected")
        if not isinstance(expected, dict):
            raise FinalizeError(f"gate request lacks expected contract: {unit_id}")
        expected_shape = expected.get("expected_shape")
        if expected_shape is not None and (
            not isinstance(expected_shape, list)
            or not expected_shape
            or any(not isinstance(item, int) or item <= 0 for item in expected_shape)
        ):
            raise FinalizeError(f"gate request has invalid expected shape: {unit_id}")
        raw_expected_dtype = expected.get("expected_dtype")
        if not isinstance(raw_expected_dtype, str) or not raw_expected_dtype.strip():
            raise FinalizeError(f"gate request lacks expected dtype: {unit_id}")
        expected_dtype = normalize_dtype(raw_expected_dtype)
        try:
            connection.execute(
                "INSERT INTO expected VALUES (?, ?, ?, ?, ?, ?, ?, NULL)",
                (
                    ordinal,
                    unit_id,
                    request_sha,
                    str(request.get("source_uri", "")),
                    str(request.get("source_index_uri", "")),
                    canonical_json(expected_shape) if expected_shape is not None else None,
                    expected_dtype,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise FinalizeError(f"duplicate gate request unit_id: {unit_id}") from exc
        request_count += 1
    route_count = 0
    for route in iter_jsonl(pending_path):
        unit_id = str(route.get("unit_id", ""))
        if (
            route.get("schema") != CATALOG_SCHEMA
            or route.get("namespace") != NAMESPACE
            or route.get("status") != PENDING
            or route.get("action") != "WAIT_FOR_REMOTE_GATE"
        ):
            raise FinalizeError(f"invalid pending catalog route: {unit_id}")
        cursor = connection.execute(
            "UPDATE expected SET route_json=? WHERE unit_id=? AND route_json IS NULL",
            (canonical_json(route), unit_id),
        )
        if cursor.rowcount != 1:
            exists = connection.execute(
                "SELECT route_json FROM expected WHERE unit_id=?", (unit_id,)
            ).fetchone()
            reason = "extra" if exists is None else "duplicate"
            raise FinalizeError(f"{reason} pending route unit_id: {unit_id}")
        route_count += 1
    missing_route = connection.execute(
        "SELECT unit_id FROM expected WHERE route_json IS NULL LIMIT 1"
    ).fetchone()
    if missing_route is not None or route_count != request_count:
        raise FinalizeError(
            f"requests/pending routes are not one-to-one: requests={request_count}, "
            f"routes={route_count}, first_missing={missing_route}"
        )
    connection.commit()
    return request_count


def resolve_receipt_path(summary_path: Path, summary: Mapping[str, Any]) -> Path:
    value = str(summary.get("receipts_file", ""))
    path = Path(value)
    if not path.is_absolute():
        path = summary_path.parent / path
    path = path.resolve()
    if path.parent != summary_path.parent.resolve():
        raise FinalizeError(f"receipt file escapes shard directory: {path}")
    return path


def validate_receipt(
    receipt: Mapping[str, Any], expected: tuple[str, str, str, str | None, str]
) -> None:
    request_sha, source_uri, source_index_uri, expected_shape_json, expected_dtype = expected
    unit_id = str(receipt.get("unit_id", ""))
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise FinalizeError(f"receipt schema mismatch: {unit_id}")
    if receipt.get("request_sha256") != request_sha:
        raise FinalizeError(f"receipt request SHA mismatch: {unit_id}")
    if receipt.get("source_uri") != source_uri:
        raise FinalizeError(f"receipt source URI mismatch: {unit_id}")
    if receipt.get("source_index_uri") != source_index_uri:
        raise FinalizeError(f"receipt index URI mismatch: {unit_id}")
    if receipt.get("status") not in {PASS, FAIL, PENDING}:
        raise FinalizeError(f"receipt status invalid: {unit_id}")
    if receipt.get("payload_full_read_performed") is not False:
        raise FinalizeError(f"receipt claims a payload body read: {unit_id}")
    if receipt.get("byte_level_certified") is not False:
        raise FinalizeError(f"receipt incorrectly claims byte certification: {unit_id}")
    reasons = receipt.get("reason_codes")
    if not isinstance(reasons, list) or not reasons:
        raise FinalizeError(f"receipt lacks reason codes: {unit_id}")
    if receipt["status"] == PASS:
        checks = receipt.get("checks")
        if (
            not isinstance(checks, dict)
            or not REQUIRED_PASS_CHECKS.issubset(checks)
            or not all(value is True for value in checks.values())
        ):
            missing = (
                sorted(REQUIRED_PASS_CHECKS - set(checks))
                if isinstance(checks, dict)
                else sorted(REQUIRED_PASS_CHECKS)
            )
            raise FinalizeError(
                f"PASS receipt lacks the complete all-true check contract: "
                f"{unit_id}; missing={missing}"
            )
        evidence = receipt.get("evidence", {})
        if not isinstance(evidence, dict):
            raise FinalizeError(f"PASS receipt lacks evidence: {unit_id}")
        if not isinstance(evidence.get("source_index"), dict):
            raise FinalizeError(f"PASS receipt lacks source-index evidence: {unit_id}")
        if not isinstance(evidence.get("sidecar"), dict):
            raise FinalizeError(f"PASS receipt lacks sidecar evidence: {unit_id}")
        safetensors_evidence = evidence.get("safetensors")
        if not isinstance(safetensors_evidence, dict):
            raise FinalizeError(f"PASS receipt lacks safetensors evidence: {unit_id}")
        if safetensors_evidence.get("payload_body_read") is not False:
            raise FinalizeError(f"PASS receipt body-read claim is invalid: {unit_id}")
        observed_tensor = evidence.get("observed_tensor")
        if (
            not isinstance(observed_tensor, dict)
            or not isinstance(observed_tensor.get("shape"), list)
            or not observed_tensor.get("shape")
            or not isinstance(observed_tensor.get("dtype"), str)
            or not isinstance(observed_tensor.get("data_offsets"), list)
            or len(observed_tensor["data_offsets"]) != 2
        ):
            raise FinalizeError(f"PASS receipt lacks observed tensor evidence: {unit_id}")
        if expected_shape_json is None:
            raise FinalizeError(
                f"PASS receipt has no canary-bound expected tensor shape: {unit_id}"
            )
        expected_shape = json.loads(expected_shape_json)
        if observed_tensor["shape"] != expected_shape:
            raise FinalizeError(
                f"PASS receipt observed tensor shape differs from request: {unit_id}"
            )
        if normalize_dtype(observed_tensor["dtype"]) != expected_dtype:
            raise FinalizeError(
                f"PASS receipt observed tensor dtype differs from request: {unit_id}"
            )


def load_receipts(
    connection: sqlite3.Connection,
    summary_paths: Sequence[Path],
    request_file_sha: str,
) -> list[dict[str, Any]]:
    bindings: list[dict[str, Any]] = []
    shard_keys: set[tuple[int, int]] = set()
    for raw_path in summary_paths:
        summary_path = raw_path.resolve()
        summary = read_json(summary_path, "gate shard summary")
        if summary.get("schema") != "LEGACY_EXACT_REFERENCE_REMOTE_GATE_SUMMARY_V1":
            raise FinalizeError(f"gate shard summary schema mismatch: {summary_path}")
        if summary.get("requests_sha256") != request_file_sha:
            raise FinalizeError(f"gate shard request-file SHA mismatch: {summary_path}")
        if summary.get("payload_full_read_performed") is not False:
            raise FinalizeError(f"gate shard claims payload body reads: {summary_path}")
        shard_key = (int(summary["shard_index"]), int(summary["shard_count"]))
        if shard_key in shard_keys:
            raise FinalizeError(f"duplicate gate shard summary: {shard_key}")
        shard_keys.add(shard_key)
        receipt_path = resolve_receipt_path(summary_path, summary)
        receipt_sha = sha256_file(receipt_path)
        if receipt_sha != summary.get("receipts_sha256"):
            raise FinalizeError(f"gate receipt file SHA mismatch: {receipt_path}")
        observed = Counter()
        for receipt in iter_jsonl(receipt_path):
            unit_id = str(receipt.get("unit_id", ""))
            row = connection.execute(
                "SELECT request_sha,source_uri,source_index_uri,"
                "expected_shape_json,expected_dtype FROM expected WHERE unit_id=?",
                (unit_id,),
            ).fetchone()
            if row is None:
                raise FinalizeError(f"extra receipt unit_id: {unit_id}")
            validate_receipt(receipt, row)
            try:
                connection.execute(
                    "INSERT INTO receipts VALUES (?, ?)",
                    (unit_id, canonical_json(receipt)),
                )
            except sqlite3.IntegrityError as exc:
                raise FinalizeError(f"duplicate receipt unit_id: {unit_id}") from exc
            observed[receipt["status"]] += 1
        if sum(observed.values()) != int(summary.get("processed", -1)):
            raise FinalizeError(f"gate shard processed count mismatch: {summary_path}")
        if dict(sorted(observed.items())) != summary.get("status_counts"):
            raise FinalizeError(f"gate shard status counts mismatch: {summary_path}")
        bindings.append(
            {
                "summary": str(summary_path),
                "summary_sha256": sha256_file(summary_path),
                "receipts": str(receipt_path),
                "receipts_sha256": receipt_sha,
                "shard_index": shard_key[0],
                "shard_count": shard_key[1],
                "processed": sum(observed.values()),
            }
        )
    connection.commit()
    return bindings


def finalized_route(
    route: dict[str, Any], receipt: dict[str, Any] | None
) -> dict[str, Any]:
    result = dict(route)
    if receipt is None:
        result["reason_codes"] = [
            *route.get("reason_codes", []),
            "REMOTE_GATE_RECEIPT_MISSING",
        ]
        return result
    result["remote_reference_gate"] = receipt
    status = receipt["status"]
    if status == PASS:
        result.update(
            status=PASS,
            action="REFERENCE_SOURCE_READ_ONLY",
            reason_codes=list(receipt["reason_codes"]),
        )
    elif status == FAIL:
        result.update(
            status=FAIL,
            action="REEXTRACT",
            reason_codes=["REMOTE_REFERENCE_GATE_FAILED", *receipt["reason_codes"]],
        )
    else:
        result.update(
            status=PENDING,
            action="WAIT_FOR_REMOTE_GATE",
            reason_codes=["REMOTE_REFERENCE_GATE_STILL_PENDING", *receipt["reason_codes"]],
        )
    return result


def write_outputs(
    output_dir: Path,
    connection: sqlite3.Connection,
    *,
    catalog: Path,
    catalog_info: Mapping[str, Any],
    receipt_bindings: list[dict[str, Any]],
    expected_count: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FinalizeError(f"refusing to overwrite finalization: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.", dir=output_dir.parent) as tmp:
        stage = Path(tmp)
        paths = {
            "all": stage / "resolved_pending_universe.routes.jsonl.gz",
            "pass": stage / "reference_routes.jsonl.gz",
            "fail": stage / "failed_gate_reextract.routes.jsonl.gz",
            "pending": stage / "still_pending.routes.jsonl.gz",
        }
        counts: Counter[str] = Counter()
        writers: dict[str, TextIO] = {}
        try:
            for key, path in paths.items():
                writers[key] = open_gzip_text(path)
            cursor = connection.execute(
                "SELECT e.route_json,r.receipt_json FROM expected e "
                "LEFT JOIN receipts r ON r.unit_id=e.unit_id ORDER BY e.ordinal"
            )
            for route_json, receipt_json in cursor:
                route = json.loads(route_json)
                receipt = json.loads(receipt_json) if receipt_json is not None else None
                resolved = finalized_route(route, receipt)
                encoded = canonical_json(resolved) + "\n"
                writers["all"].write(encoded)
                status = resolved["status"]
                counts[status] += 1
                if status == PASS:
                    writers["pass"].write(encoded)
                elif status == FAIL:
                    writers["fail"].write(encoded)
                elif status == PENDING:
                    writers["pending"].write(encoded)
                else:  # pragma: no cover - finalized_route owns this enum
                    raise FinalizeError(f"unexpected resolved status: {status}")
        finally:
            for writer in writers.values():
                writer.close()
        if sum(counts.values()) != expected_count:
            raise FinalizeError("resolved route count changed")

        catalog_manifest = catalog / "catalog_manifest.json"
        catalog_summary = catalog / "summary.json"
        request_path = catalog / "remote_gate_requests.jsonl.gz"
        pending_path = catalog / "streaming_pending_remote_gate.routes.jsonl.gz"
        summary = {
            "schema": FINALIZER_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "COMPLETE" if counts[PENDING] == 0 else "INCOMPLETE",
            "namespace": NAMESPACE,
            "counts": {
                "expected_pending_units": expected_count,
                "receipts": connection.execute("SELECT COUNT(*) FROM receipts").fetchone()[0],
                "missing_receipts": expected_count
                - connection.execute("SELECT COUNT(*) FROM receipts").fetchone()[0],
                "resolved_status": dict(sorted(counts.items())),
            },
            "catalog_binding": {
                "directory": str(catalog),
                "manifest_sha256": sha256_file(catalog_manifest),
                "summary_sha256": sha256_file(catalog_summary),
                "requests_sha256": catalog_info["output_shas"][request_path.name],
                "pending_routes_sha256": catalog_info["output_shas"][pending_path.name],
            },
            "receipt_shards": receipt_bindings,
            "payload_full_read_performed": False,
            "outputs": {},
        }
        for key, path in paths.items():
            summary["outputs"][key] = {
                "file": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        (stage / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.rename(stage, output_dir)
    return summary


def finalize(
    *, catalog: Path, summary_paths: Sequence[Path], output_dir: Path
) -> dict[str, Any]:
    catalog = catalog.resolve()
    info = catalog_bindings(catalog)
    expected_from_summary = int(
        info["summary"].get("counts", {}).get("status", {}).get(PENDING, -1)
    )
    request_path = catalog / "remote_gate_requests.jsonl.gz"
    pending_path = catalog / "streaming_pending_remote_gate.routes.jsonl.gz"
    with tempfile.TemporaryDirectory(prefix="remote-gate-finalizer-db-") as tmp:
        connection = make_database(Path(tmp) / "join.sqlite3")
        try:
            count = load_expected(connection, request_path, pending_path)
            if count != expected_from_summary:
                raise FinalizeError(
                    f"catalog pending count mismatch: expected {expected_from_summary}, found {count}"
                )
            bindings = load_receipts(
                connection,
                summary_paths,
                info["output_shas"][request_path.name],
            )
            return write_outputs(
                output_dir.resolve(),
                connection,
                catalog=catalog,
                catalog_info=info,
                receipt_bindings=bindings,
                expected_count=count,
            )
        finally:
            connection.close()


def discover_summaries(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FinalizeError(f"receipt root is not a directory: {root}")
    return sorted(root.glob("**/summary.json"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-dir", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--receipt-root", type=Path)
    parser.add_argument("--shard-summary", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summaries = list(args.shard_summary)
    if args.receipt_root is not None:
        summaries.extend(discover_summaries(args.receipt_root))
    if not summaries:
        raise SystemExit("ERROR: no gate shard summaries supplied")
    try:
        result = finalize(
            catalog=args.catalog_dir,
            summary_paths=summaries,
            output_dir=args.output_dir,
        )
    except FinalizeError as exc:
        raise SystemExit(f"ERROR: {exc}")
    print(canonical_json({"status": result["status"], "counts": result["counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
