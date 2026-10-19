#!/usr/bin/env python3
"""Build the display-first, provenance-preserving loose result table for Q00.

This intentionally does *not* decide whether a historical value is fit for a
paper.  It makes every frozen parent configuration visible, retains the
evidence needed to assess it, and separates source precedence from scientific
validity.  It never writes the registry or any main canonical table.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]


SOURCE_PATHS = {
    "universe": "experiments/task_status_1096_before_expand_20260822.csv",
    "reconciliation": "results/analysis/q00_1096/universe_1096_reconciled.csv",
    "registry": "experiments/registry.csv",
    "canonical_all": "results/canonical/all_results.csv",
    "default_b2_raw": "results/reconstructed/default8_b2_angle_candidate.csv",
    "wan_cog_lineage": "experiments/wan_cog_lineage_resolution.csv",
    "legacy_b2": "results/canonical_protocols/legacy_b2_shared_hidden_8_12_16_24_val_v1/b2_results.csv",
    "legacy_b2_namespace": "results/canonical_protocols/legacy_b2_shared_hidden_8_12_16_24_val_v1/namespace.json",
    "window_audit": "results/analysis/main_table_1096_window_audit/window_audit_1096.csv",
    "phase1_live": "results/analysis/q00_1096_phase1/live_universe_1096.csv",
}


PRIMARY_METRIC = {
    "streaming_depth": ("mean_loss_depth", "min"),
    "streaming_camera": ("Auc_30", "max"),
    "streaming_mask": ("mean_loss_identity", "min"),
    "view_consistency": ("balanced_acc", "max"),
    "ego_belief": ("mean_az_err_deg", "min"),
    "ego_belief_v2": ("mean_ang_err_deg", "min"),
    "action_dynamics": ("global_R@1", "max"),
    "path_integration": ("global_R@1", "max"),
    "counterfactual": ("intervention_validity", "max"),
}

# These are result fields, rather than bookkeeping fields such as ``n`` or an
# epoch number, that can appear in historical val_summary.json payloads.
KNOWN_SUMMARY_METRICS = {metric for metric, _direction in PRIMARY_METRIC.values()} | {
    "Auc_30", "Auc_50", "balanced_acc", "top1", "top3",
    "mean_az_err_deg", "mean_el_err_deg", "mean_ang_err_deg",
    "mean_log_dist_err", "mean_loss_depth", "mean_loss_identity",
    "global_R@1", "global_R@5", "action_gain_cos", "intervention_validity",
    "pose_error", "mse", "mae", "accuracy",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def truth(value: Any) -> bool:
    return clean(value).lower() in {"1", "true", "yes", "y", "pass", "passed"}


def comparable_yes(value: Any) -> bool:
    """Interpret the reconciliation's deliberately descriptive comparator flag."""
    text = clean(value).upper()
    return text in {"1", "TRUE", "YES"} or text.startswith("YES_")


def finite(value: Any) -> float | None:
    try:
        number = float(clean(value))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def split_ids(value: Any) -> list[str]:
    return [part.strip() for part in clean(value).split(";") if part.strip()]


def metric_pair(value: str) -> tuple[str, float] | None:
    if "=" not in clean(value):
        return None
    name, raw = clean(value).split("=", 1)
    number = finite(raw)
    return (name.strip(), number) if name.strip() and number is not None else None


def json_cell(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def safe_repo_file(value: str) -> Path | None:
    """Return only regular paths physically inside this repository.

    No AOSS/remote URI or arbitrary absolute path is followed while building
    this CPU-only table.  This is deliberately narrower than historical
    evidence matching.
    """
    raw = clean(value)
    if not raw or raw.startswith(("aoss://", "s3://")):
        return None
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    try:
        normalized = candidate.resolve(strict=False)
    except OSError:
        return None
    try:
        normalized.relative_to(ROOT)
    except ValueError:
        return None
    return normalized if normalized.is_file() else None


def repo_display_path(path: Path) -> str:
    """Return a repository-relative path without requiring Python 3.9+."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def candidate(
    *,
    source: str,
    tier: str,
    metric_name: str,
    metric_value: float,
    path: str,
    identity_evidence: str,
    mapping_role: str = "primary",
    warning: str = "",
    candidate_protocol: str = "",
    source_id: str = "",
    source_sha256: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "source": source,
        "tier": tier,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "finite": True,
        "artifact_path": path,
        "source_sha256": source_sha256,
        "source_id": source_id,
        "identity_evidence": identity_evidence,
        "mapping_role": mapping_role,
        "candidate_protocol": candidate_protocol,
        "warning": warning,
    }
    if extra:
        row.update(extra)
    return row


def dedupe_candidates(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        key = json_cell(item)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def candidate_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    role = 0 if item.get("mapping_role") == "primary" else 1
    return (role, clean(item.get("source_id")), clean(item.get("metric_name")), clean(item.get("artifact_path")))


def expected_metric(probe: str) -> tuple[str, str]:
    try:
        return PRIMARY_METRIC[probe]
    except KeyError as exc:
        raise SystemExit(f"No primary metric contract for probe {probe!r}") from exc


# A deliberately small Parquet writer.  All columns are REQUIRED UTF8 strings;
# blank strings encode missing values.  This avoids optional-level ambiguity and
# permits writing a standards-compliant .parquet without a GPU or a pandas/
# pyarrow dependency in the frozen execution environment.
class CompactThrift:
    STOP = 0
    I32 = 5
    I64 = 6
    BINARY = 8
    LIST = 9
    STRUCT = 12

    def __init__(self) -> None:
        self.data = bytearray()
        self.last_field_id = 0

    @staticmethod
    def _uvarint(value: int) -> bytes:
        output = bytearray()
        while True:
            byte = value & 0x7F
            value >>= 7
            output.append(byte | (0x80 if value else 0))
            if not value:
                return bytes(output)

    @staticmethod
    def _zigzag32(value: int) -> int:
        return (value << 1) ^ (value >> 31)

    @staticmethod
    def _zigzag64(value: int) -> int:
        return (value << 1) ^ (value >> 63)

    def field(self, field_type: int, field_id: int) -> None:
        delta = field_id - self.last_field_id
        if 0 < delta <= 15:
            self.data.append((delta << 4) | field_type)
        else:
            self.data.append(field_type)
            self.data.extend(self._uvarint(field_id))
        self.last_field_id = field_id

    def i32(self, value: int) -> None:
        self.data.extend(self._uvarint(self._zigzag32(int(value))))

    def i64(self, value: int) -> None:
        self.data.extend(self._uvarint(self._zigzag64(int(value))))

    def binary(self, value: str | bytes) -> None:
        raw = value.encode("utf-8") if isinstance(value, str) else value
        self.data.extend(self._uvarint(len(raw)))
        self.data.extend(raw)

    def list_header(self, element_type: int, length: int) -> None:
        if length <= 14:
            self.data.append((length << 4) | element_type)
        else:
            self.data.append(0xF0 | element_type)
            self.data.extend(self._uvarint(length))

    def stop(self) -> None:
        self.data.append(self.STOP)


def thrift_struct(writer_fn: Callable[[CompactThrift], None]) -> bytes:
    writer = CompactThrift()
    writer_fn(writer)
    writer.stop()
    return bytes(writer.data)


def parquet_schema_element_root(name: str, children: int) -> bytes:
    return thrift_struct(lambda w: (w.field(CompactThrift.BINARY, 4), w.binary(name), w.field(CompactThrift.I32, 5), w.i32(children)))


def parquet_schema_element_string(name: str) -> bytes:
    # type BYTE_ARRAY=6; repetition REQUIRED=0; converted_type UTF8=0.
    return thrift_struct(
        lambda w: (
            w.field(CompactThrift.I32, 1), w.i32(6),
            w.field(CompactThrift.I32, 3), w.i32(0),
            w.field(CompactThrift.BINARY, 4), w.binary(name),
            w.field(CompactThrift.I32, 6), w.i32(0),
        )
    )


def parquet_page_header(body_size: int, values: int) -> bytes:
    def data_page_header(w: CompactThrift) -> None:
        # PLAIN=0, RLE=3.  Required fields have max def/rep level 0, hence
        # no encoded level bytes are present in the page body.
        w.field(CompactThrift.I32, 1); w.i32(values)
        w.field(CompactThrift.I32, 2); w.i32(0)
        w.field(CompactThrift.I32, 3); w.i32(3)
        w.field(CompactThrift.I32, 4); w.i32(3)

    def page(w: CompactThrift) -> None:
        # PageType.DATA_PAGE=0.
        w.field(CompactThrift.I32, 1); w.i32(0)
        w.field(CompactThrift.I32, 2); w.i32(body_size)
        w.field(CompactThrift.I32, 3); w.i32(body_size)
        w.field(CompactThrift.STRUCT, 5); w.data.extend(thrift_struct(data_page_header))

    return thrift_struct(page)


def parquet_column_metadata(name: str, values: int, page_offset: int, page_size: int) -> bytes:
    def write(w: CompactThrift) -> None:
        # Physical BYTE_ARRAY=6; PLAIN=0; UNCOMPRESSED=0.
        w.field(CompactThrift.I32, 1); w.i32(6)
        w.field(CompactThrift.LIST, 2); w.list_header(CompactThrift.I32, 1); w.i32(0)
        w.field(CompactThrift.LIST, 3); w.list_header(CompactThrift.BINARY, 1); w.binary(name)
        w.field(CompactThrift.I32, 4); w.i32(0)
        w.field(CompactThrift.I64, 5); w.i64(values)
        w.field(CompactThrift.I64, 6); w.i64(page_size)
        w.field(CompactThrift.I64, 7); w.i64(page_size)
        w.field(CompactThrift.I64, 9); w.i64(page_offset)
    return thrift_struct(write)


def parquet_column_chunk(name: str, values: int, page_offset: int, page_size: int) -> bytes:
    metadata = parquet_column_metadata(name, values, page_offset, page_size)
    def write(w: CompactThrift) -> None:
        w.field(CompactThrift.I64, 2); w.i64(page_offset)
        w.field(CompactThrift.STRUCT, 3); w.data.extend(metadata)
    return thrift_struct(write)


def parquet_row_group(chunks: list[bytes], total_size: int, rows: int) -> bytes:
    def write(w: CompactThrift) -> None:
        w.field(CompactThrift.LIST, 1); w.list_header(CompactThrift.STRUCT, len(chunks))
        for chunk in chunks:
            w.data.extend(chunk)
        w.field(CompactThrift.I64, 2); w.i64(total_size)
        w.field(CompactThrift.I64, 3); w.i64(rows)
    return thrift_struct(write)


def parquet_footer(columns: list[str], rows: int, chunks: list[bytes], total_size: int) -> bytes:
    schema = [parquet_schema_element_root("schema", len(columns))] + [parquet_schema_element_string(name) for name in columns]
    row_group = parquet_row_group(chunks, total_size, rows)
    def write(w: CompactThrift) -> None:
        w.field(CompactThrift.I32, 1); w.i32(1)
        w.field(CompactThrift.LIST, 2); w.list_header(CompactThrift.STRUCT, len(schema))
        for element in schema:
            w.data.extend(element)
        w.field(CompactThrift.I64, 3); w.i64(rows)
        w.field(CompactThrift.LIST, 4); w.list_header(CompactThrift.STRUCT, 1); w.data.extend(row_group)
        w.field(CompactThrift.BINARY, 6); w.binary("q00_1096_loose_builtin_parquet_writer")
    return thrift_struct(write)


def write_parquet(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Write a single-row-group uncompressed UTF8 Parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[bytes] = []
    total_size = 0
    with path.open("wb") as handle:
        handle.write(b"PAR1")
        for name in columns:
            raw_values = [clean(row.get(name, "")).encode("utf-8") for row in rows]
            body = b"".join(struct.pack("<I", len(value)) + value for value in raw_values)
            offset = handle.tell()
            header = parquet_page_header(len(body), len(raw_values))
            handle.write(header)
            handle.write(body)
            page_size = len(header) + len(body)
            chunks.append(parquet_column_chunk(name, len(raw_values), offset, page_size))
            total_size += page_size
        footer = parquet_footer(columns, len(rows), chunks, total_size)
        handle.write(footer)
        handle.write(struct.pack("<I", len(footer)))
        handle.write(b"PAR1")


def assert_parquet_envelope(path: Path) -> None:
    with path.open("rb") as handle:
        raw = handle.read()
    if len(raw) < 12 or raw[:4] != b"PAR1" or raw[-4:] != b"PAR1":
        raise RuntimeError(f"Parquet envelope invalid: {path}")
    footer_length = struct.unpack("<I", raw[-8:-4])[0]
    if not (0 < footer_length < len(raw) - 8):
        raise RuntimeError(f"Parquet footer length invalid: {path}")


def checkpoint_available(path_text: str) -> tuple[bool, str]:
    for candidate_path in split_ids(path_text):
        actual = safe_repo_file(candidate_path)
        if actual is not None:
            return True, str(actual.relative_to(ROOT))
    return False, ""


def source_hashes() -> dict[str, str]:
    paths = {name: ROOT / relative for name, relative in SOURCE_PATHS.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise SystemExit(f"Required loose-table source missing: {missing}")
    return {name: sha256(path) for name, path in paths.items()}


def add_canonical_candidates(
    rec: dict[str, str],
    canonical_by_id: dict[str, list[dict[str, str]]],
    target_metric: str,
    all_hash: str,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for column, role in (("canonical_id", "primary"), ("alternate_canonical_id", "alternate_collision_not_selected")):
        for atomic_id in split_ids(rec.get(column, "")):
            for raw in canonical_by_id.get(atomic_id, []):
                value = finite(raw.get("metric_value"))
                if value is None:
                    continue
                warning = "" if role == "primary" else "ALTERNATE_COLLISION_MAPPING_RETAINED_NOT_SELECTED"
                items.append(candidate(
                    source="canonical",
                    tier="T0_CANONICAL",
                    metric_name=clean(raw.get("metric_name")),
                    metric_value=value,
                    path=SOURCE_PATHS["canonical_all"],
                    source_sha256=all_hash,
                    source_id=atomic_id,
                    identity_evidence=f"reconciliation.{column} exact atomic-ID bridge",
                    mapping_role=role,
                    warning=warning,
                ))
    return dedupe_candidates(items)


def add_registry_candidates(
    rec: dict[str, str],
    registry_by_id: dict[str, list[dict[str, str]]],
    registry_hash: str,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for column, role in (("primary_registry_id", "primary"), ("alternate_registry_id", "alternate_collision_not_selected")):
        for atomic_id in split_ids(rec.get(column, "")):
            for raw in registry_by_id.get(atomic_id, []):
                value = finite(raw.get("metric_value"))
                if value is None:
                    continue
                items.append(candidate(
                    source="registry_index",
                    tier="INDEX_NOT_SELECTED",
                    metric_name=clean(raw.get("metric_name")),
                    metric_value=value,
                    path=SOURCE_PATHS["registry"],
                    source_sha256=registry_hash,
                    source_id=atomic_id,
                    identity_evidence=f"reconciliation.{column} exact atomic-ID bridge",
                    mapping_role=role,
                    warning="REGISTRY_INDEX_CANDIDATE_NOT_USED_FOR_SOURCE_PRECEDENCE",
                ))
    return dedupe_candidates(items)


def raw_b2_candidates(
    rec: dict[str, str],
    b2_by_id: dict[str, dict[str, str]],
    b2_hash: str,
    target_metric: str,
) -> list[dict[str, Any]]:
    row = b2_by_id.get(rec["universe_id"])
    if not row or rec.get("family") != "B2":
        return []
    value = finite(row.get(target_metric))
    if value is None:
        return []
    # This is a full-identity bridge: ID, run directory, and summary path were
    # reconciled row by row.  Refuse any replacement with a merely similar path.
    if clean(row.get("summary_path")) != clean(rec.get("raw_summary_path")):
        return []
    return [candidate(
        source="raw_historical_summary",
        tier="T1_RAW_HISTORICAL",
        metric_name=target_metric,
        metric_value=value,
        path=clean(row.get("summary_path")),
        source_sha256=b2_hash,
        source_id=rec["universe_id"],
        identity_evidence="exact universe_id + reconciled raw_summary_path + run_artifact_path",
        warning="DEFAULT8_B2_RAW_SUMMARY_PROTOCOL_UNFROZEN",
        extra={"n": clean(row.get("n")), "checkpoint_path": clean(row.get("ckpt_path"))},
    )]


def _summary_metric_values(payload: Any) -> dict[str, float]:
    """Extract known numeric metrics from a historical summary JSON.

    Summary files occasionally nest metric dictionaries; their metric names
    are still retained only when they belong to the known result contract.
    This intentionally does not treat counts, epochs, or arbitrary diagnostic
    scalars as probe results.
    """
    found: dict[str, float] = {}

    def walk(value: Any) -> None:
        if not isinstance(value, dict):
            return
        for key, nested in value.items():
            key_text = clean(key)
            number = finite(nested) if not isinstance(nested, (dict, list)) else None
            if key_text in KNOWN_SUMMARY_METRICS and number is not None:
                found[key_text] = number
            if isinstance(nested, dict):
                walk(nested)

    walk(payload)
    return dict(sorted(found.items()))


def raw_summary_candidates(
    rec: dict[str, str],
    target_metric: str,
) -> tuple[list[dict[str, Any]], str]:
    """Use only the reconciliation's exact per-parent summary path.

    No filename search is performed.  A supplied reconciliation SHA is checked
    when present, so an overwritten historical summary becomes an explicit
    rejected source instead of silently changing the loose table.
    """
    raw_path = clean(rec.get("raw_summary_path"))
    actual = safe_repo_file(raw_path)
    if actual is None:
        return [], "RAW_SUMMARY_UNAVAILABLE"
    try:
        payload = json.loads(actual.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return [], "RAW_SUMMARY_UNREADABLE"
    actual_hash = sha256(actual)
    expected_hash = clean(rec.get("raw_summary_sha256"))
    if expected_hash and expected_hash != actual_hash:
        return [], "RAW_SUMMARY_SHA256_MISMATCH"
    metrics = _summary_metric_values(payload)
    value = metrics.get(target_metric)
    if value is None:
        return [], "RAW_SUMMARY_NO_FINITE_PRIMARY_METRIC"
    relative = str(actual.relative_to(ROOT))
    return [candidate(
        source="raw_historical_summary",
        tier="T1_RAW_HISTORICAL",
        metric_name=target_metric,
        metric_value=value,
        path=relative,
        source_sha256=actual_hash,
        source_id=rec["universe_id"],
        identity_evidence="reconciliation.raw_summary_path exact full-parent bridge; SHA256 checked when frozen SHA recorded",
        warning="HISTORICAL_VAL_SUMMARY_EXACT_PARENT_PATH",
        extra={"all_metrics": metrics},
    )], "RAW_SUMMARY_PRIMARY_METRIC_FINITE"


def lineage_candidates(
    rec: dict[str, str],
    lineage_by_id: dict[str, list[dict[str, str]]],
    lineage_hash: str,
    target_metric: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_items: list[dict[str, Any]] = []
    provisional: list[dict[str, Any]] = []
    for row in lineage_by_id.get(rec["universe_id"], []):
        parsed = metric_pair(row.get("old_metric_value", ""))
        if not parsed:
            continue
        metric_name, value = parsed
        if metric_name != target_metric:
            continue
        resolution = clean(row.get("resolution_status"))
        item = candidate(
            source="historical_provisional" if resolution == "PROVISIONAL_ONLY_NO_REPLACEMENT" else "raw_historical_summary",
            tier="T2_PROVISIONAL" if resolution == "PROVISIONAL_ONLY_NO_REPLACEMENT" else "T1_RAW_HISTORICAL",
            metric_name=metric_name,
            metric_value=value,
            path=clean(row.get("old_metric_path")),
            source_sha256=lineage_hash,
            source_id=clean(row.get("lineage_id")),
            identity_evidence="wan_cog_lineage_resolution.old_experiment_id exact frozen universe_id; FULL_KEY_MATCH_TASK_STATUS",
            warning=(
                "LATEST_PROVISIONAL_NO_FORMAL_REPLACEMENT"
                if resolution == "PROVISIONAL_ONLY_NO_REPLACEMENT"
                else "FORMAL_HISTORICAL_LINEAGE_RECORD"
            ),
            extra={
                "resolution_status": resolution,
                "checkpoint_path": clean(row.get("old_checkpoint_path")),
                "identity_match_quality": clean(row.get("identity_match_quality")),
                "protocol_match": clean(row.get("protocol_match")),
            },
        )
        (provisional if resolution == "PROVISIONAL_ONLY_NO_REPLACEMENT" else raw_items).append(item)
    return dedupe_candidates(raw_items), dedupe_candidates(provisional)


def legacy_candidates(
    rec: dict[str, str],
    legacy_by_id: dict[str, dict[str, str]],
    legacy_hash: str,
    target_metric: str,
) -> list[dict[str, Any]]:
    row = legacy_by_id.get(rec["universe_id"])
    if not row:
        return []
    # Keep the frozen parent protocol.  The candidate's legacy protocol is
    # deliberately recorded inside the candidate instead of overwriting it.
    fields = ("campaign", "model_variant", "family", "probe", "model", "layer", "prefix", "horizon", "seed")
    if any(clean(row.get(field)) != clean(rec.get(field)) for field in fields):
        return []
    value = finite(row.get(target_metric))
    if value is None:
        return []
    return [candidate(
        source="legacy_packaged_namespace",
        tier="T3_LEGACY_PROTOCOL",
        metric_name=target_metric,
        metric_value=value,
        path=SOURCE_PATHS["legacy_b2"],
        source_sha256=legacy_hash,
        source_id=clean(row.get("protocol_canonical_id")),
        identity_evidence="exact universe_id plus full frozen identity; candidate protocol intentionally distinct",
        warning="LEGACY_PROTOCOL_NAMESPACE_NOT_STRICTLY_COMPARABLE",
        candidate_protocol=clean(row.get("protocol")),
        extra={"n": clean(row.get("n")), "protocol_canonical_valid": clean(row.get("protocol_canonical_valid"))},
    )]


def loose_eval_index(out_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Index terminal loose-eval receipts once per table refresh.

    The result table has 1,096 parents.  Rewalking the owned eval namespace for
    every parent is needlessly expensive on the shared filesystem and makes a
    CPU-only refresh scale with both parent and receipt count.  Only terminal
    receipts are eligible evidence, so scan those exactly once and retain the
    same candidate contract used by the per-row selector.
    """
    by_universe: dict[str, list[dict[str, Any]]] = defaultdict(list)
    runs = out_dir / "eval_runs"
    if not runs.is_dir():
        return by_universe
    receipt_paths: list[Path] = []
    for directory, _dirs, names in os.walk(runs, onerror=lambda _error: None):
        if "terminal_receipt.json" in names:
            receipt_paths.append(Path(directory) / "terminal_receipt.json")
    for path in sorted(receipt_paths):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        universe_id = clean(payload.get("universe_id"))
        if not universe_id:
            continue
        if clean(payload.get("evaluation_mode")) != "LEGACY_AS_IS_LOOSE":
            continue
        if clean(payload.get("status")) not in {"PASS", "COMPLETE", "SUCCEEDED"}:
            continue
        metric_name = clean(payload.get("metric_name"))
        value = finite(payload.get("metric_value"))
        if value is None:
            continue
        receipt_extra = {
            "n": payload.get("n"),
            "summary_path": clean(payload.get("summary_path")),
            "summary_sha256": clean(payload.get("summary_sha256")),
            "legacy_state_dict_compatibility": payload.get("legacy_state_dict_compatibility"),
            "legacy_state_dict_compatibility_retry": payload.get("legacy_state_dict_compatibility_retry"),
        }
        # Keep receipt-specific lineage visible in the display table without
        # serializing absent fields as misleading null provenance.
        receipt_extra = {key: value for key, value in receipt_extra.items() if value not in (None, "")}
        by_universe[universe_id].append(candidate(
            source="loose_reevaluation",
            tier="T4_KNOWN_MISMATCH_LOOSE_EVAL",
            metric_name=metric_name,
            metric_value=value,
            path=repo_display_path(path),
            source_sha256=sha256(path),
            source_id=universe_id,
            identity_evidence="terminal LEGACY_AS_IS_LOOSE receipt with exact universe_id",
            warning=clean(payload.get("warning")) or "LEGACY_AS_IS_LOOSE_NOT_SCIENTIFICALLY_COMPARABLE",
            extra=receipt_extra,
        ))
    return {universe_id: dedupe_candidates(items) for universe_id, items in by_universe.items()}


def loose_eval_candidates(
    loose_by_universe: dict[str, list[dict[str, Any]]],
    universe_id: str,
    target_metric: str,
) -> list[dict[str, Any]]:
    """Return eligible pre-indexed loose candidates for one frozen parent."""
    return [
        item for item in loose_by_universe.get(universe_id, [])
        if item.get("metric_name") == target_metric
    ]


def primary_candidates(items: list[dict[str, Any]], target_metric: str, include_alternate: bool = False) -> list[dict[str, Any]]:
    return sorted([
        item for item in items
        if item.get("metric_name") == target_metric
        and (include_alternate or item.get("mapping_role") == "primary")
    ], key=candidate_sort_key)


def build(args: argparse.Namespace) -> dict[str, Any]:
    sources = {name: ROOT / relative for name, relative in SOURCE_PATHS.items()}
    hashes_before = source_hashes()
    universe = read_csv(sources["universe"])
    reconciliation = read_csv(sources["reconciliation"])
    phase1 = read_csv(sources["phase1_live"])
    audit = read_csv(sources["window_audit"])
    if len(universe) != 1096 or len(reconciliation) != 1096:
        raise SystemExit(f"Frozen universe/reconciliation must both have 1096 rows: {len(universe)}, {len(reconciliation)}")
    universe_by_id = {row["experiment_id"]: row for row in universe}
    rec_by_id = {row["universe_id"]: row for row in reconciliation}
    phase1_by_id = {row["universe_id"]: row for row in phase1}
    audit_by_id = {row["experiment_id"]: row for row in audit}
    if len(universe_by_id) != 1096 or len(rec_by_id) != 1096 or set(universe_by_id) != set(rec_by_id):
        raise SystemExit("Universe/reconciliation identity set is not exactly the frozen 1096")
    if set(audit_by_id) != set(universe_by_id):
        raise SystemExit("Window-audit identity set is not exactly the frozen 1096")

    canonical_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(sources["canonical_all"]):
        canonical_by_id[clean(row.get("experiment_id"))].append(row)
    registry_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(sources["registry"]):
        registry_by_id[clean(row.get("experiment_id"))].append(row)
    b2_by_id = {row["experiment_id"]: row for row in read_csv(sources["default_b2_raw"])}
    lineage_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(sources["wan_cog_lineage"]):
        lineage_by_id[clean(row.get("old_experiment_id"))].append(row)
    legacy_by_id = {row["universe_id"]: row for row in read_csv(sources["legacy_b2"])}

    output_dir = Path(args.out_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    loose_by_universe = loose_eval_index(output_dir)
    rows: list[dict[str, Any]] = []
    tier_counts: Counter[str] = Counter()
    mismatch_rows = 0
    mismatch_selected = 0
    manifest_mismatch_rows = 0
    manifest_mismatch_selected = 0
    raw_summary_scanned = Counter()
    selected_warning_counter: Counter[str] = Counter()

    for universe_id in sorted(universe_by_id):
        frozen = universe_by_id[universe_id]
        rec = rec_by_id[universe_id]
        phase = phase1_by_id.get(universe_id, {})
        win = audit_by_id[universe_id]
        target_metric, direction = expected_metric(clean(rec.get("probe")))

        canonical = add_canonical_candidates(rec, canonical_by_id, target_metric, hashes_before["canonical_all"])
        registry = add_registry_candidates(rec, registry_by_id, hashes_before["registry"])
        summary_raw, raw_summary_state = raw_summary_candidates(rec, target_metric)
        raw_summary_scanned[raw_summary_state] += 1
        raw = summary_raw + raw_b2_candidates(rec, b2_by_id, hashes_before["default_b2_raw"], target_metric)
        lineage_raw, provisional = lineage_candidates(rec, lineage_by_id, hashes_before["wan_cog_lineage"], target_metric)
        raw = dedupe_candidates(raw + lineage_raw)
        legacy = legacy_candidates(rec, legacy_by_id, hashes_before["legacy_b2"], target_metric)
        loose = loose_eval_candidates(loose_by_universe, universe_id, target_metric)

        # Source precedence is intentionally independent of scientific status.
        # Alternate collision candidates are retained but never pooled/selected.
        source_order = [
            ("CANONICAL", "T0_CANONICAL", primary_candidates(canonical, target_metric)),
            ("RAW_HISTORICAL_SUMMARY", "T1_RAW_HISTORICAL", primary_candidates(raw, target_metric)),
            ("HISTORICAL_PROVISIONAL", "T2_PROVISIONAL", primary_candidates(provisional, target_metric)),
            ("LEGACY_B2_PROTOCOL", "T3_LEGACY_PROTOCOL", primary_candidates(legacy, target_metric)),
            ("LEGACY_AS_IS_LOOSE_EVAL", "T4_KNOWN_MISMATCH_LOOSE_EVAL", primary_candidates(loose, target_metric)),
        ]
        result_source = "NA"
        tier = "T5_NO_NUMERIC"
        selected: dict[str, Any] | None = None
        for source_name, candidate_tier, items in source_order:
            if items:
                result_source, tier, selected = source_name, candidate_tier, items[0]
                break

        error_proven = truth(win.get("error_proven"))
        exact_safe = truth(win.get("exact_safe"))
        if error_proven:
            mismatch = "TRUE"
            comparable = "false"
            lineage_status = clean(win.get("window_judgment")) or "CONFIRMED_MISMATCH"
            warning_parts = ["CONFIRMED_LABEL_FEATURE_WINDOW_MISMATCH_RETAINED_LOOSE"]
        elif exact_safe:
            mismatch = "FALSE"
            comparable = "true" if comparable_yes(rec.get("scientifically_comparable")) else "false"
            lineage_status = clean(win.get("window_judgment")) or "EXACT_SAFE"
            warning_parts = ["EXACT_SAFE_LINEAGE"]
        else:
            mismatch = "UNKNOWN"
            comparable = "unknown"
            lineage_status = clean(win.get("window_judgment")) or clean(phase.get("scientific_lineage_class")) or "LINEAGE_UNRESOLVED"
            warning_parts = [f"LINEAGE_UNVERIFIED_OR_HIGH_RISK:{lineage_status}"]

        train_manifest_status = clean(rec.get("train_manifest_status"))
        eval_manifest_status = clean(rec.get("eval_manifest_status"))
        if "MISSING" in {train_manifest_status, eval_manifest_status} or any(
            token in status for status in (train_manifest_status, eval_manifest_status)
            for token in ("MISMATCH", "RECONSTRUCTED", "AMBIGUOUS")
        ):
            manifest_mismatch = "TRUE"
            warning_parts.append("KNOWN_MANIFEST_PATH_OR_PROVENANCE_MISMATCH")
        elif train_manifest_status == "PRESENT" and eval_manifest_status in {"", "UNSPECIFIED", "PRESENT"}:
            manifest_mismatch = "FALSE"
        else:
            manifest_mismatch = "UNKNOWN"
            warning_parts.append("MANIFEST_PROVENANCE_UNSPECIFIED")
        if error_proven:
            mismatch_rows += 1
        if selected and error_proven:
            mismatch_selected += 1
        if manifest_mismatch == "TRUE":
            manifest_mismatch_rows += 1
        if selected and manifest_mismatch == "TRUE":
            manifest_mismatch_selected += 1
        if selected and tier == "T2_PROVISIONAL":
            warning_parts.append("HISTORICAL_PROVISIONAL_NOT_CANONICAL")
        if selected and tier == "T1_RAW_HISTORICAL":
            warning_parts.append("RAW_HISTORICAL_SUMMARY_PROTOCOL_UNFROZEN")
        if legacy:
            warning_parts.append("LEGACY_PROTOCOL_CANDIDATE_AVAILABLE")
        if clean(rec.get("registry_variant_collision_group")):
            warning_parts.append("CAMPAIGN_COLLISION_GROUP_PRESERVED_NO_MERGE")
        for status_name in ("train_manifest_status", "eval_manifest_status"):
            status = clean(rec.get(status_name))
            if status and status not in {"PRESENT_EXACT", "UNSPECIFIED"}:
                warning_parts.append(f"{status_name.upper()}={status}")
        warning = ";".join(dict.fromkeys(warning_parts))
        if selected:
            selected_warning_counter[warning] += 1

        checkpoint_ok, checkpoint_local_path = checkpoint_available(clean(rec.get("checkpoint_path")))
        all_candidates = dedupe_candidates(canonical + raw + provisional + legacy + loose)
        identity_fields = [
            "campaign", "model", "model_variant", "protocol", "family", "probe", "layer", "prefix", "horizon", "seed"
        ]
        # The original frozen CSV predates the explicit protocol namespace;
        # protocol is therefore preserved from reconciliation but cannot be
        # compared to a column that did not exist in the historical source.
        for field in ("campaign", "model", "model_variant", "family", "probe", "layer", "prefix", "horizon", "seed"):
            if clean(frozen.get({"probe": "probe_config", "family": "probe_family"}.get(field, field))) != clean(rec.get(field)):
                raise SystemExit(f"Full identity mismatch for {universe_id}: {field}")
        out: dict[str, Any] = {
            "universe_id": universe_id,
            **{field: clean(rec.get(field)) for field in identity_fields},
            "default_timestep": clean(rec.get("default_timestep")),
            "full_identity": clean(rec.get("full_identity")),
            "primary_target_metric": target_metric,
            "primary_target_direction": direction,
            "metric_selection_rule": "first finite primary-contract metric from the first exact primary mapping in source precedence; alternate collision mappings retained but never pooled",
            "selected_metric_name": clean(selected.get("metric_name")) if selected else "",
            "selected_result": repr(selected.get("metric_value")) if selected else "",
            "selected_result_finite": "true" if selected else "false",
            "result_source": result_source,
            "result_quality_tier": tier,
            "selected_result_path": clean(selected.get("artifact_path")) if selected else "",
            "selected_result_identity_evidence": clean(selected.get("identity_evidence")) if selected else "",
            "selected_result_candidate_json": json_cell(selected) if selected else "",
            "canonical_valid": "true" if truth(rec.get("canonical_valid")) else "false",
            "scientifically_comparable": comparable,
            "scientifically_comparable_reconciled": clean(rec.get("scientifically_comparable")),
            "known_label_feature_mismatch": mismatch,
            "known_manifest_mismatch": manifest_mismatch,
            "lineage_status": lineage_status,
            "scientific_lineage_class": clean(phase.get("scientific_lineage_class")),
            "phase1_resolution": clean(phase.get("phase1_resolution")),
            "window_evidence_grade": clean(win.get("evidence_grade")),
            "warning": warning,
            "registry_id": clean(rec.get("registry_id")),
            "registry_status": clean(rec.get("registry_status")),
            "canonical_id": clean(rec.get("canonical_id")),
            "alternate_canonical_id": clean(rec.get("alternate_canonical_id")),
            "raw_summary_path": clean(rec.get("raw_summary_path")),
            "raw_summary_sha256": clean(rec.get("raw_summary_sha256")),
            "raw_summary_candidate_status": raw_summary_state,
            "checkpoint_path": clean(rec.get("checkpoint_path")),
            "checkpoint_status": clean(rec.get("checkpoint_status")),
            "checkpoint_available": "true" if checkpoint_ok else "false",
            "checkpoint_local_path": checkpoint_local_path,
            "feature_status": clean(rec.get("feature_status")),
            "train_manifest": clean(rec.get("train_manifest")),
            "train_manifest_status": clean(rec.get("train_manifest_status")),
            "eval_manifest": clean(rec.get("eval_manifest")),
            "eval_manifest_status": clean(rec.get("eval_manifest_status")),
            "numeric_result_present_reconciled": clean(rec.get("numeric_result_present")),
            "numeric_result_kind_reconciled": clean(rec.get("numeric_result_kind")),
            "artifact_mapping_disposition": clean(rec.get("artifact_mapping_disposition")),
            "registry_variant_collision_group": clean(rec.get("registry_variant_collision_group")),
            "canonical_candidate_count": str(len(canonical)),
            "raw_historical_summary_candidate_count": str(len(raw)),
            "historical_provisional_candidate_count": str(len(provisional)),
            "legacy_packaged_candidate_count": str(len(legacy)),
            "loose_reeval_candidate_count": str(len(loose)),
            "registry_index_candidate_count": str(len(registry)),
            "all_numeric_candidate_count": str(len(all_candidates)),
            "canonical_candidates_json": json_cell(canonical),
            "raw_historical_summary_candidates_json": json_cell(raw),
            "historical_provisional_candidates_json": json_cell(provisional),
            "legacy_packaged_candidates_json": json_cell(legacy),
            "loose_reeval_candidates_json": json_cell(loose),
            "registry_index_candidates_json": json_cell(registry),
            "all_numeric_candidates_json": json_cell(all_candidates),
        }
        rows.append(out)
        tier_counts[tier] += 1

    if len(rows) != 1096 or len({row["universe_id"] for row in rows}) != 1096:
        raise SystemExit("Loose output does not contain one row for every frozen universe identity")
    if sum(tier_counts.values()) != 1096:
        raise SystemExit("Loose output tier accounting is not exhaustive")

    fields = list(rows[0])
    csv_path = output_dir / "all_results_1096_loose.csv"
    parquet_path = output_dir / "all_results_1096_loose.parquet"
    write_csv(csv_path, rows, fields)
    write_parquet(parquet_path, rows, fields)
    assert_parquet_envelope(parquet_path)

    missing_rows: list[dict[str, Any]] = []
    for row in rows:
        if row["selected_result_finite"] != "false":
            continue
        checkpoint_ok = row["checkpoint_available"] == "true"
        missing_rows.append({
            "universe_id": row["universe_id"],
            "campaign": row["campaign"],
            "protocol": row["protocol"],
            "model": row["model"],
            "model_variant": row["model_variant"],
            "family": row["family"],
            "probe": row["probe"],
            "layer": row["layer"],
            "prefix": row["prefix"],
            "horizon": row["horizon"],
            "seed": row["seed"],
            "primary_target_metric": row["primary_target_metric"],
            "checkpoint_available": row["checkpoint_available"],
            "checkpoint_path": row["checkpoint_path"],
            "checkpoint_local_path": row["checkpoint_local_path"],
            "checkpoint_status": row["checkpoint_status"],
            "feature_status": row["feature_status"],
            "known_label_feature_mismatch": row["known_label_feature_mismatch"],
            "known_manifest_mismatch": row["known_manifest_mismatch"],
            "lineage_status": row["lineage_status"],
            "warning": row["warning"],
            "evaluation_mode": "LEGACY_AS_IS_LOOSE" if checkpoint_ok else "",
            "requires_training_for_any_numeric": "false" if checkpoint_ok else "true",
            "next_action": "LEGACY_AS_IS_LOOSE_EVAL_PREFLIGHT" if checkpoint_ok else "REQUIRES_TRAINING_FOR_ANY_NUMERIC",
        })
    missing_path = output_dir / "missing_numeric.csv"
    write_csv(missing_path, missing_rows, list(missing_rows[0]) if missing_rows else ["universe_id"])

    hashes_after = source_hashes()
    if hashes_before != hashes_after:
        changed = {key: (hashes_before[key], hashes_after[key]) for key in hashes_before if hashes_before[key] != hashes_after[key]}
        raise SystemExit(f"A source-of-truth changed while building loose table: {changed}")

    finite_count = sum(row["selected_result_finite"] == "true" for row in rows)
    summary = {
        "mode": "DISPLAY_FIRST_LOOSE_RESULTS",
        "source_precedence": ["canonical finite", "raw historical summary", "historical provisional", "legacy packaged", "loose reevaluation", "NA"],
        "total_rows": len(rows),
        "finite_selected_results": finite_count,
        "canonical_results": tier_counts["T0_CANONICAL"],
        "raw_historical_results": tier_counts["T1_RAW_HISTORICAL"],
        "provisional_results": tier_counts["T2_PROVISIONAL"],
        "legacy_results": tier_counts["T3_LEGACY_PROTOCOL"],
        "loose_reeval_results": tier_counts["T4_KNOWN_MISMATCH_LOOSE_EVAL"],
        "no_numeric": tier_counts["T5_NO_NUMERIC"],
        "legacy_candidates_available": sum(row["legacy_packaged_candidate_count"] != "0" for row in rows),
        "known_mismatch_rows": mismatch_rows,
        "known_mismatch_results": mismatch_selected,
        "known_manifest_mismatch_rows": manifest_mismatch_rows,
        "known_manifest_mismatch_results": manifest_mismatch_selected,
        "raw_summary_paths_readable": (
            raw_summary_scanned["RAW_SUMMARY_PRIMARY_METRIC_FINITE"]
            + raw_summary_scanned["RAW_SUMMARY_NO_FINITE_PRIMARY_METRIC"]
        ),
        "raw_historical_summary_primary_candidates": raw_summary_scanned["RAW_SUMMARY_PRIMARY_METRIC_FINITE"],
        "exact_safe_rows": sum(row["known_label_feature_mismatch"] == "FALSE" for row in rows),
        "unknown_lineage_rows": sum(row["known_label_feature_mismatch"] == "UNKNOWN" for row in rows),
        "checkpoint_available_missing_numeric": sum(row["checkpoint_available"] == "true" for row in missing_rows),
        "requires_training_for_any_numeric": sum(row["requires_training_for_any_numeric"] == "true" for row in missing_rows),
        "tier_counts": dict(sorted(tier_counts.items())),
        "raw_summary_scan_status_counts": dict(sorted(raw_summary_scanned.items())),
        "source_sha256": hashes_before,
        "outputs": {
            "csv": str(csv_path.relative_to(ROOT)),
            "csv_sha256": sha256(csv_path),
            "parquet": str(parquet_path.relative_to(ROOT)),
            "parquet_sha256": sha256(parquet_path),
            "missing_numeric": str(missing_path.relative_to(ROOT)),
            "missing_numeric_sha256": sha256(missing_path),
        },
        "protected_source_hashes_unchanged": True,
        "notes": [
            "T0 is source precedence, not a claim of strict scientific validity.",
            "Known lineage/window errors and independent manifest provenance states are displayed as metadata and warnings, never hidden or merged.",
            "No registry or results/canonical table was modified.",
            "The Parquet file is an uncompressed UTF8 Parquet table written without GPU dependencies.",
        ],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def write_report(summary: dict[str, Any], report_path: Path) -> None:
    lines = [
        "# Q00 loose all-results table (frozen 1096)",
        "",
        "This is a display-first table. It retains canonical, raw historical, provisional, and legacy evidence even when lineage metadata warns that a value is not scientifically comparable. It does not modify the registry or canonical tables.",
        "",
        "## Counts",
        "",
        f"- TOTAL_ROWS: {summary['total_rows']}",
        f"- FINITE_SELECTED_RESULTS: {summary['finite_selected_results']}",
        f"- CANONICAL_RESULTS: {summary['canonical_results']}",
        f"- RAW_HISTORICAL_RESULTS: {summary['raw_historical_results']}",
        f"- PROVISIONAL_RESULTS: {summary['provisional_results']}",
        f"- LEGACY_RESULTS: {summary['legacy_results']}",
        f"- LOOSE_REEVAL_RESULTS: {summary['loose_reeval_results']}",
        f"- NO_NUMERIC: {summary['no_numeric']}",
        f"- KNOWN_MISMATCH_RESULTS: {summary['known_mismatch_results']}",
        f"- KNOWN_MANIFEST_MISMATCH_RESULTS: {summary['known_manifest_mismatch_results']}",
        f"- HISTORICAL_VAL_SUMMARY_PATHS_READABLE: {summary['raw_summary_paths_readable']}",
        f"- HISTORICAL_VAL_SUMMARY_PRIMARY_CANDIDATES: {summary['raw_historical_summary_primary_candidates']}",
        f"- LEGACY_CANDIDATES_AVAILABLE: {summary['legacy_candidates_available']}",
        "",
        "## Selection and safeguards",
        "",
        "Selection is per frozen parent and uses only exact identity bridges: canonical atomic IDs from reconciliation, exact default-B2 summary/run paths, Wan/Cog lineage `old_experiment_id`, and exact legacy `universe_id` plus full identity. The source order is canonical finite → raw historical summary → historical provisional → legacy packaged → loose reevaluation → NA.",
        "",
        "T0 is an evidence-source tier rather than an endorsement of the result's scientific validity. `known_label_feature_mismatch`, `known_manifest_mismatch`, `lineage_status`, `scientifically_comparable`, and `warning` make the existing limitations explicit. Collision candidates remain separate JSON records and are never pooled.",
        "",
        f"The table has {summary['legacy_candidates_available']} rows with a legacy B2 candidate, but T1 raw summaries outrank all of them; therefore LEGACY_RESULTS can be zero while legacy evidence remains visible.",
        f"It also contains {summary['loose_reeval_results']} finite T4 legacy-as-is evaluations. These are display-only values and retain their mismatch warnings; they do not alter canonical validity or scientific comparability.",
        "",
        "## Outputs",
        "",
    ]
    for label, item in summary["outputs"].items():
        if label.endswith("sha256"):
            continue
        lines.append(f"- `{item}`")
    lines.extend(["", "## Frozen input hashes", ""])
    for name, digest in sorted(summary["source_sha256"].items()):
        lines.append(f"- `{name}`: `{digest}`")
    lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(ROOT / "results/analysis/q00_1096_loose"))
    parser.add_argument("--report", default=str(ROOT / "reports/q00_all_results_1096_loose.md"))
    args = parser.parse_args()
    summary = build(args)
    write_report(summary, Path(args.report).resolve())
    print(json.dumps({
        "TOTAL_ROWS": summary["total_rows"],
        "FINITE_SELECTED_RESULTS": summary["finite_selected_results"],
        "CANONICAL_RESULTS": summary["canonical_results"],
        "RAW_HISTORICAL_RESULTS": summary["raw_historical_results"],
        "PROVISIONAL_RESULTS": summary["provisional_results"],
        "LEGACY_RESULTS": summary["legacy_results"],
        "NO_NUMERIC": summary["no_numeric"],
        "KNOWN_MISMATCH_RESULTS": summary["known_mismatch_results"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
