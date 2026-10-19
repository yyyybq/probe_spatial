#!/usr/bin/env python3
"""Freeze the corrected MiniMax-H3 streaming tensor contract for probe816.

This is deliberately a versioned derivative: the original request file and
gate receipts remain immutable.  A correction is emitted only when *every*
MiniMax gate receipt failed solely because the old request expected T=19,
while all identity, sidecar, dtype, and raw-frame checks passed and the
observed tensor/sidecar shape is exactly T=22.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


OLD_SHAPE = [19, 15, 26, 5376]
NEW_SHAPE = [22, 15, 26, 5376]
MINIMAX = "minimax_h3"
EXPECTED_UNITS = 33_880


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL line {number}: {path}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL line {number}: {path}")
            yield row


def atomic_json(path: Path, value: Any) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with open(fd, "w", encoding="utf-8", closefd=True) as f:
            json.dump(value, f, indent=2, sort_keys=True, ensure_ascii=False)
            f.write("\n")
        Path(tmp).replace(path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=Path, required=True)
    ap.add_argument("--gate-root", type=Path, required=True)
    ap.add_argument("--extractor", type=Path, required=True)
    ap.add_argument("--featurizer", type=Path, required=True)
    ap.add_argument("--blocker", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.output_dir.exists():
        raise ValueError(f"output already exists: {args.output_dir}")

    original_requests_sha = sha256_file(args.requests)

    request_by_unit: dict[str, dict[str, Any]] = {}
    request_rows: list[dict[str, Any]] = []
    for row in jsonl(args.requests):
        unit = str(row["unit_id"])
        if unit in request_by_unit:
            raise ValueError(f"duplicate request unit: {unit}")
        request_by_unit[unit] = row
        request_rows.append(row)

    proofs: dict[str, dict[str, Any]] = {}
    failed_check_counts: Counter[tuple[str, ...]] = Counter()
    observed_shapes: Counter[tuple[int, ...]] = Counter()
    for receipt_path in sorted(args.gate_root.glob("gate-*/receipts.jsonl.gz")):
        for row in jsonl(receipt_path):
            unit = str(row.get("unit_id") or "")
            req = request_by_unit.get(unit)
            if req is None or (req.get("expected") or {}).get("vfm") != MINIMAX:
                continue
            if unit in proofs:
                raise ValueError(f"duplicate gate receipt for {unit}")
            failed = tuple(sorted(k for k, ok in (row.get("checks") or {}).items() if not ok))
            failed_check_counts[failed] += 1
            observed = ((row.get("evidence") or {}).get("observed_tensor") or {})
            shape = tuple(int(v) for v in observed.get("shape") or ())
            observed_shapes[shape] += 1
            expected = req.get("expected") or {}
            if expected.get("expected_shape") != OLD_SHAPE:
                raise ValueError(f"old MiniMax request shape changed for {unit}")
            if row.get("status") != "REEXTRACT_UNVERIFIED":
                raise ValueError(f"unexpected old gate status for {unit}: {row.get('status')}")
            if failed != ("header_shape_equals_expected",):
                raise ValueError(f"non-shape gate failure for {unit}: {failed}")
            if list(shape) != NEW_SHAPE or observed.get("dtype") != "float16":
                raise ValueError(f"unexpected observed tensor for {unit}: {observed}")
            sidecar = ((row.get("evidence") or {}).get("sidecar") or {})
            safetensors = ((row.get("evidence") or {}).get("safetensors") or {})
            if not sidecar.get("declared_payload_sha256_unverified"):
                raise ValueError(f"missing declared payload SHA for {unit}")
            proofs[unit] = row

    if len(proofs) != EXPECTED_UNITS:
        raise ValueError(f"MiniMax proof count {len(proofs)} != {EXPECTED_UNITS}")

    args.output_dir.mkdir(parents=True)
    corrected_path = args.output_dir / "remote_gate_requests_816_contract_v2.jsonl.gz"
    receipt_path = args.output_dir / "minimax_contract_correction_receipts.jsonl.gz"
    corrected_count = 0
    with corrected_path.open("wb") as raw, gzip.GzipFile(
        filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6
    ) as gz:
        for original in request_rows:
            row = json.loads(json.dumps(original))
            expected = row.get("expected") or {}
            if expected.get("vfm") == MINIMAX:
                expected["expected_shape"] = list(NEW_SHAPE)
                expected["shape_contract"] = (
                    "minimax_h3_packing_v2_sidecar_and_header_equal_exact_shape"
                )
                row["schema"] = "CANONICAL_V2_PROBE816_REMOTE_GATE_REQUEST_V2"
                row["contract_correction"] = {
                    "schema": "MINIMAX_H3_TEMPORAL_TOKEN_CONTRACT_CORRECTION_V1",
                    "old_expected_shape": list(OLD_SHAPE),
                    "corrected_expected_shape": list(NEW_SHAPE),
                    "input_num_frames": 73,
                    "reason": "MiniMax-H3 packing maps 73 input frames to 22 video-token frames",
                    "original_request_file_sha256": original_requests_sha,
                }
                corrected_count += 1
            gz.write((json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode())

    with receipt_path.open("wb") as raw, gzip.GzipFile(
        filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6
    ) as gz:
        for unit in sorted(proofs):
            gate = proofs[unit]
            req = request_by_unit[unit]
            evidence = gate["evidence"]
            row = {
                "schema": "CANONICAL_V2_PROBE816_MINIMAX_CONTRACT_CORRECTION_RECEIPT_V1",
                "status": "PASS_CONTRACT_CORRECTED",
                "unit_id": unit,
                "source_uri": gate["source_uri"],
                "source_index_uri": gate["source_index_uri"],
                "sidecar_uri": req["sidecar_uri"],
                "original_expected_shape": list(OLD_SHAPE),
                "corrected_expected_shape": list(NEW_SHAPE),
                "observed_tensor": evidence["observed_tensor"],
                "all_original_checks_except_old_shape_passed": True,
                "failed_checks_before_correction": ["header_shape_equals_expected"],
                "payload_sha256": evidence["sidecar"]["declared_payload_sha256_unverified"],
                "payload_sha256_verification": "VERIFY_ON_FIRST_MATERIALIZATION",
                "payload_size_bytes": evidence["safetensors"]["object_size_bytes"],
                "safetensors_header_sha256": evidence["safetensors"]["header_sha256"],
                "source_index_sha256": evidence["source_index"]["sha256"],
                "raw_ids": req["expected"]["raw_ids"],
                "raw_ids_sha256": req["expected"]["raw_ids_sha256"],
                "feature_layer": req["expected"]["feature_layer"],
                "timestep": req["expected"]["timestep"],
                "source_index_record": req["source_index_record"],
            }
            row["receipt_sha256"] = canonical_sha(row)
            gz.write((json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode())

    if corrected_count != EXPECTED_UNITS:
        raise ValueError(f"corrected request rows {corrected_count} != {EXPECTED_UNITS}")
    manifest = {
        "schema": "CANONICAL_V2_PROBE816_STREAMING_CONTRACT_V2_MANIFEST",
        "status": "READY_FOR_OWNER_MERGE",
        "original_requests": {
            "path": str(args.requests.resolve()),
            "sha256": original_requests_sha,
            "rows": len(request_rows),
        },
        "corrected_requests": {
            "path": str(corrected_path.resolve()),
            "sha256": sha256_file(corrected_path),
            "rows": len(request_rows),
        },
        "correction_receipts": {
            "path": str(receipt_path.resolve()),
            "sha256": sha256_file(receipt_path),
            "rows": corrected_count,
        },
        "proof_bindings": {
            "gate_root": str(args.gate_root.resolve()),
            "extractor": {"path": str(args.extractor.resolve()), "sha256": sha256_file(args.extractor)},
            "featurizer": {"path": str(args.featurizer.resolve()), "sha256": sha256_file(args.featurizer)},
            "blocker": {"path": str(args.blocker.resolve()), "sha256": sha256_file(args.blocker)},
        },
        "audit": {
            "minimax_units": len(proofs),
            "failed_checks": {"|".join(k): v for k, v in failed_check_counts.items()},
            "observed_shapes": {"x".join(map(str, k)): v for k, v in observed_shapes.items()},
            "old_shape": OLD_SHAPE,
            "corrected_shape": NEW_SHAPE,
            "original_artifacts_modified": False,
        },
    }
    atomic_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
