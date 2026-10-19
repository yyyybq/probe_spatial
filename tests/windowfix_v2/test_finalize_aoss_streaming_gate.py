from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from analyses.windowfix_v2.finalize_aoss_streaming_gate import (
    CANONICAL_PARENT_SOURCE_SHA256,
    CANONICAL_UNIVERSE_SHA256,
    FAIL,
    NAMESPACE,
    PASS,
    PENDING,
    REQUIRED_PASS_CHECKS,
    FinalizeError,
    canonical_json,
    finalize,
    sha256_bytes,
    sha256_file,
    validate_receipt,
)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl_gz(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def request(unit_id: str) -> dict:
    return {
        "schema": "LEGACY_EXACT_REFERENCE_REMOTE_GATE_REQUEST_V1",
        "unit_id": unit_id,
        "source_uri": f"aoss://bucket/{unit_id}/feature.sft",
        "source_index_uri": f"aoss://bucket/{unit_id}/prefix_index.npy",
        "expected": {"expected_shape": [8, 1], "expected_dtype": "float16"},
    }


def route(unit_id: str) -> dict:
    return {
        "schema": "CANONICAL_V2_STREAMING_ROUTE_CATALOG_V1",
        "namespace": NAMESPACE,
        "unit_id": unit_id,
        "status": PENDING,
        "action": "WAIT_FOR_REMOTE_GATE",
        "source_uri": f"aoss://bucket/{unit_id}/feature.sft",
        "canonical_raw_ids": list(range(8)),
        "vfm": "dino",
        "model_id": "facebook/dinov2-large",
        "feature_layer": 0,
        "timestep": 749,
        "prefix": 8,
        "scene_id": "scene",
        "window_id": 0,
        "tail": 7,
        "reason_codes": ["REMOTE_PAYLOAD_GATE_REQUIRED"],
    }


def receipt(req: dict, status: str) -> dict:
    value = {
        "schema": "LEGACY_EXACT_REFERENCE_REMOTE_GATE_RECEIPT_V1",
        "unit_id": req["unit_id"],
        "request_sha256": sha256_bytes(canonical_json(req).encode("utf-8")),
        "source_uri": req["source_uri"],
        "source_index_uri": req["source_index_uri"],
        "payload_full_read_performed": False,
        "byte_level_certified": False,
        "status": status,
        "reason_codes": ["TEST_GATE_RESULT"],
    }
    if status == PASS:
        value["checks"] = {key: True for key in REQUIRED_PASS_CHECKS}
        value["evidence"] = {
            "source_index": {"sha256": "0" * 64},
            "sidecar": {"sha256": "1" * 64},
            "safetensors": {"payload_body_read": False},
            "observed_tensor": {
                "shape": [8, 1],
                "dtype": "F16",
                "data_offsets": [0, 16],
            },
        }
    return value


def make_catalog(root: Path, unit_ids: list[str]) -> tuple[Path, list[dict]]:
    root.mkdir()
    requests = [request(unit_id) for unit_id in unit_ids]
    write_jsonl_gz(root / "remote_gate_requests.jsonl.gz", requests)
    write_jsonl_gz(
        root / "streaming_pending_remote_gate.routes.jsonl.gz",
        [route(unit_id) for unit_id in unit_ids],
    )
    summary = {
        "schema": "CANONICAL_V2_STREAMING_ROUTE_CATALOG_V1",
        "namespace": NAMESPACE,
        "canonical_parent_source_sha256": CANONICAL_PARENT_SOURCE_SHA256,
        "canonical_universe_sha256": CANONICAL_UNIVERSE_SHA256,
        "counts": {"status": {PENDING: len(unit_ids)}},
    }
    write_json(root / "summary.json", summary)
    outputs = []
    for name in (
        "remote_gate_requests.jsonl.gz",
        "streaming_pending_remote_gate.routes.jsonl.gz",
        "summary.json",
    ):
        path = root / name
        outputs.append(
            {"file": name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    write_json(
        root / "catalog_manifest.json",
        {
            "schema": "CANONICAL_V2_STREAMING_ROUTE_CATALOG_MANIFEST_V1",
            "canonical_parent_source_sha256": CANONICAL_PARENT_SOURCE_SHA256,
            "canonical_universe_sha256": CANONICAL_UNIVERSE_SHA256,
            "outputs": outputs,
        },
    )
    return root, requests


def make_shard(root: Path, requests_file: Path, rows: list[dict], index: int = 0) -> Path:
    root.mkdir()
    receipts_path = root / "receipts.jsonl.gz"
    write_jsonl_gz(receipts_path, rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    summary_path = root / "summary.json"
    write_json(
        summary_path,
        {
            "schema": "LEGACY_EXACT_REFERENCE_REMOTE_GATE_SUMMARY_V1",
            "requests_sha256": sha256_file(requests_file),
            "shard_index": index,
            "shard_count": 2,
            "processed": len(rows),
            "status_counts": dict(sorted(counts.items())),
            "payload_full_read_performed": False,
            "receipts_file": receipts_path.name,
            "receipts_sha256": sha256_file(receipts_path),
        },
    )
    return summary_path


def read_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def test_finalizer_promotes_pass_routes_fails_mismatches_and_keeps_missing_pending(
    tmp_path: Path,
) -> None:
    catalog, requests = make_catalog(tmp_path / "catalog", ["pass", "fail", "missing"])
    shard = make_shard(
        tmp_path / "shard-0",
        catalog / "remote_gate_requests.jsonl.gz",
        [receipt(requests[0], PASS), receipt(requests[1], FAIL)],
    )
    output = tmp_path / "finalized"
    summary = finalize(catalog=catalog, summary_paths=[shard], output_dir=output)

    assert summary["status"] == "INCOMPLETE"
    assert summary["counts"]["resolved_status"] == {FAIL: 1, PASS: 1, PENDING: 1}
    assert read_jsonl_gz(output / "reference_routes.jsonl.gz")[0]["unit_id"] == "pass"
    failed = read_jsonl_gz(output / "failed_gate_reextract.routes.jsonl.gz")[0]
    assert failed["unit_id"] == "fail"
    assert failed["action"] == "REEXTRACT"
    pending = read_jsonl_gz(output / "still_pending.routes.jsonl.gz")[0]
    assert pending["unit_id"] == "missing"
    assert "REMOTE_GATE_RECEIPT_MISSING" in pending["reason_codes"]


@pytest.mark.parametrize("kind", ["duplicate", "extra"])
def test_finalizer_rejects_duplicate_or_extra_receipts(tmp_path: Path, kind: str) -> None:
    catalog, requests = make_catalog(tmp_path / "catalog", ["one"])
    first = make_shard(
        tmp_path / "shard-0",
        catalog / "remote_gate_requests.jsonl.gz",
        [receipt(requests[0], PASS)],
        0,
    )
    if kind == "duplicate":
        row = receipt(requests[0], PASS)
    else:
        unknown = request("unknown")
        row = receipt(unknown, PASS)
    second = make_shard(
        tmp_path / "shard-1",
        catalog / "remote_gate_requests.jsonl.gz",
        [row],
        1,
    )
    with pytest.raises(FinalizeError, match=kind):
        finalize(
            catalog=catalog,
            summary_paths=[first, second],
            output_dir=tmp_path / "never-written",
        )


def test_pass_receipt_requires_real_boolean_true_checks() -> None:
    req = request("strict")
    value = receipt(req, PASS)
    value["checks"]["source_index_raw_ids_exact"] = "true"
    expected = (
        value["request_sha256"],
        value["source_uri"],
        value["source_index_uri"],
        canonical_json(req["expected"]["expected_shape"]),
        req["expected"]["expected_dtype"],
    )
    with pytest.raises(FinalizeError, match="all-true"):
        validate_receipt(value, expected)


def test_pass_receipt_rejects_truncated_all_true_check_mapping() -> None:
    req = request("truncated")
    value = receipt(req, PASS)
    value["checks"] = {"source_index_record_exists": True}
    expected = (
        value["request_sha256"],
        value["source_uri"],
        value["source_index_uri"],
        canonical_json(req["expected"]["expected_shape"]),
        req["expected"]["expected_dtype"],
    )
    with pytest.raises(FinalizeError, match="complete all-true check contract"):
        validate_receipt(value, expected)


def test_pass_receipt_must_match_request_exact_shape() -> None:
    req = request("wrong-shape")
    value = receipt(req, PASS)
    value["evidence"]["observed_tensor"]["shape"] = [4, 2]
    expected = (
        value["request_sha256"],
        value["source_uri"],
        value["source_index_uri"],
        canonical_json(req["expected"]["expected_shape"]),
        req["expected"]["expected_dtype"],
    )
    with pytest.raises(FinalizeError, match="shape differs"):
        validate_receipt(value, expected)


def test_finalizer_rejects_catalog_without_parent_lineage(tmp_path: Path) -> None:
    catalog, requests = make_catalog(tmp_path / "catalog", ["one"])
    summary = json.loads((catalog / "summary.json").read_text(encoding="utf-8"))
    summary["canonical_parent_source_sha256"] = "0" * 64
    write_json(catalog / "summary.json", summary)
    shard = make_shard(
        tmp_path / "shard",
        catalog / "remote_gate_requests.jsonl.gz",
        [receipt(requests[0], PASS)],
    )
    with pytest.raises(FinalizeError, match="canonical parent"):
        finalize(catalog=catalog, summary_paths=[shard], output_dir=tmp_path / "out")
