#!/usr/bin/env python3
"""Audit exact D-series reuse of the immutable independent-P8 V-JEPA2 store."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


MODEL = "vjepa2"
MODEL_REVISION = "b3c1679b7c34d3255ef3547f27c7b226aefab26f"
REQUEST_SHA256 = "fb1dbd2d71878edae7eae9034a16ee9809fc0c0cfb67352e0a301ad0edbed369"
D_MANIFEST_SHA256 = "781e25848de48fa5219aad876bd10cb8f9445328b145a19193a6f6b2ab5efcc9"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_id(scene: str, frame_name: str) -> str:
    return f"scannetpp:{scene}/{frame_name}"


def feature_path(root: Path, unit: dict[str, Any]) -> Path:
    return root / MODEL / "scannetpp" / unit["scene_id"] / f"{unit['unit_id']}.safetensors"


def validate_candidate(root: Path, unit: dict[str, Any], role: str) -> tuple[Path, dict[str, Any]]:
    payload = feature_path(root, unit)
    sidecar = Path(f"{payload}.manifest.json")
    if not payload.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"missing V2 pair for {unit['unit_id']}: {payload}")
    meta = json.loads(sidecar.read_text())
    expected = {
        "unit_id": unit["unit_id"],
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "layer": 23,
        "feature_role": role,
        "raw_frame_id": unit["raw_frame_id"],
        "context_raw_ids": unit["context_raw_ids"],
        "request_manifest_sha256": REQUEST_SHA256,
        "shape": [1024],
        "dtype": "torch.float32",
        "fallback": False,
    }
    mismatches = {key: [meta.get(key), value] for key, value in expected.items() if meta.get(key) != value}
    if mismatches:
        raise ValueError(f"V2 identity mismatch for {unit['unit_id']}: {mismatches}")
    if payload.stat().st_size != meta.get("size_bytes"):
        raise ValueError(f"V2 size mismatch for {unit['unit_id']}")
    if meta.get("pooling") != "FP32 mean over every returned temporal and spatial token":
        raise ValueError(f"V2 pooling mismatch for {unit['unit_id']}")
    return payload, meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-manifest", type=Path, required=True)
    parser.add_argument("--v2-request", type=Path, required=True)
    parser.add_argument("--v2-root", type=Path, required=True)
    parser.add_argument("--coverage-receipt", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--mapping-csv", type=Path, required=True)
    args = parser.parse_args()

    if sha256(args.d_manifest) != D_MANIFEST_SHA256:
        raise ValueError("D-series manifest hash mismatch")
    if sha256(args.v2_request) != REQUEST_SHA256:
        raise ValueError("V2 request manifest hash mismatch")
    coverage = json.loads(args.coverage_receipt.read_text())
    expected_coverage = {
        "status": "COMPLETE_VALID",
        "model": MODEL,
        "scope": "full_model_request",
        "request_manifest_sha256": REQUEST_SHA256,
        "model_revision": {MODEL_REVISION: 12358},
        "requested": 12358,
        "verified": 12358,
        "missing_or_invalid": 0,
    }
    bad_coverage = {key: [coverage.get(key), value] for key, value in expected_coverage.items() if coverage.get(key) != value}
    if bad_coverage:
        raise ValueError(f"V2 coverage receipt mismatch: {bad_coverage}")

    v2_root = args.v2_root.resolve()
    request_rows = [json.loads(line) for line in args.v2_request.open() if line.strip()]
    vjepa = [row for row in request_rows if row.get("model") == MODEL]
    contexts = {tuple(row["context_raw_ids"]): row for row in vjepa if row["feature_role"] == "streaming_context_p8"}
    isolated = {row["raw_frame_id"]: row for row in vjepa if row["feature_role"] == "isolated_frame"}
    if len(contexts) + len(isolated) != len(vjepa):
        raise ValueError("duplicate or unsupported V2 V-JEPA2 unit")

    d_rows = [json.loads(line) for line in args.d_manifest.open() if line.strip()]
    parents: dict[str, dict[str, Any]] = {}
    mapping: list[dict[str, Any]] = []
    actual_reads: list[Path] = []
    adapted = Counter()
    missing = Counter()

    for row in d_rows:
        parent_id = row["parent_sample_id"]
        if parent_id not in parents:
            key = tuple(raw_id(row["scene_id"], name) for name in row["prefix_frame_names"])
            unit = contexts.get(key)
            if unit is None:
                missing["streaming_context"] += 1
                parents[parent_id] = {"status": "missing"}
            else:
                payload, meta = validate_candidate(v2_root, unit, "streaming_context_p8")
                parents[parent_id] = {"status": "adapted", "unit": unit, "payload": payload, "meta": meta}
                adapted["streaming_context"] += 1
        target = raw_id(row["scene_id"], row["target_frame_name"])
        unit = isolated.get(target)
        if unit is None:
            missing["isolated_target"] += 1
            target_entry = {"status": "missing"}
        else:
            payload, meta = validate_candidate(v2_root, unit, "isolated_frame")
            target_entry = {"status": "adapted", "unit": unit, "payload": payload, "meta": meta}
            adapted["isolated_target"] += 1
        context_entry = parents[parent_id]
        for d_role, entry in (("streaming_context", context_entry), ("isolated_target", target_entry)):
            record = {
                "sample_id": row["sample_id"],
                "parent_sample_id": parent_id,
                "scene_id": row["scene_id"],
                "split": row["split"],
                "horizon": row["horizon"],
                "d_series_role": d_role,
                "status": entry["status"],
                "branch": "pooled",
                "adapter": "safetensors key feat -> D-series pooled; no numeric transform",
                "v2_role": entry.get("unit", {}).get("feature_role", ""),
                "v2_unit_id": entry.get("unit", {}).get("unit_id", ""),
                "v2_payload": str(entry.get("payload", "")),
                "v2_payload_sha256": entry.get("meta", {}).get("feature_sha256", ""),
            }
            mapping.append(record)
            if entry.get("payload") is not None:
                actual_reads.append(entry["payload"])

    # Deterministic representative content reads. Full payload identity is inherited
    # from the one-time full coverage receipt rather than rehashed before training.
    unique_reads = sorted(set(actual_reads))
    probes = [unique_reads[index] for index in sorted(set([0, len(unique_reads) // 4, len(unique_reads) // 2, 3 * len(unique_reads) // 4, len(unique_reads) - 1]))] if unique_reads else []
    from safetensors.torch import load_file
    actual_read_evidence = []
    for path in probes:
        value = load_file(str(path))["feat"]
        actual_read_evidence.append({
            "path": str(path), "shape": list(value.shape), "dtype": str(value.dtype),
            "finite": bool(value.isfinite().all()), "sha256": sha256(path),
        })
        if list(value.shape) != [1024] or str(value.dtype) != "torch.float32" or not value.isfinite().all():
            raise ValueError(f"representative V2 read failed: {path}")

    windows = len(parents)
    samples = len(d_rows)
    roles = {
        "streaming_context": {
            "v2_actual_role": "streaming_context_p8", "requested_windows": windows,
            "adapted_windows": adapted["streaming_context"], "missing_windows": missing["streaming_context"],
            "classification": "adapted" if adapted["streaming_context"] else "missing",
            "compatibility": "pooled only; exact ordered p8 identity; tensor key rename only",
        },
        "isolated_target": {
            "v2_actual_role": "isolated_frame", "requested_samples": samples,
            "adapted_samples": adapted["isolated_target"], "missing_samples": missing["isolated_target"],
            "classification": "adapted" if adapted["isolated_target"] else "missing",
            "compatibility": "pooled only; exact target raw-frame identity; tensor key rename only",
        },
        "native_target_reference": {
            "v2_actual_role": None, "requested_samples": samples, "reused_samples": 0,
            "missing_samples": samples, "classification": "missing",
            "compatibility": "not present; isolated_frame is not a native full-teacher target reference",
        },
        "native_predictor_output": {
            "v2_actual_role": None, "requested_samples": samples, "reused_samples": 0,
            "missing_samples": samples, "classification": "missing",
            "compatibility": "not present; no predictor output role exists in V2",
        },
    }
    result = {
        "schema_version": 1,
        "status": "COMPLETE_VALID",
        "scope": "D-series to independent-P8 V2 V-JEPA2 exact reuse audit",
        "d_series_manifest": str(args.d_manifest),
        "d_series_manifest_sha256": D_MANIFEST_SHA256,
        "expected": {"scenes": 681, "windows": 2721, "samples": 5442, "train_samples": 4890, "dev_samples": 552},
        "resolved": {
            "scenes": len({row["scene_id"] for row in d_rows}), "windows": windows, "samples": samples,
            "train_samples": sum(row["split"] == "train" for row in d_rows),
            "dev_samples": sum(row["split"] == "dev" for row in d_rows),
        },
        "v2": {
            "request_manifest": str(args.v2_request), "request_manifest_sha256": REQUEST_SHA256,
            "coverage_receipt": str(args.coverage_receipt), "coverage_receipt_sha256": sha256(args.coverage_receipt),
            "feature_root": str(v2_root), "vjepa2_units": len(vjepa),
            "checkpoint": "facebook/vjepa2-vitl-fpc64-256", "revision": MODEL_REVISION,
            "layer_zero_based": 23, "raw_layout": [32, 16, 16, 1024],
            "stored_layout": [1024], "pooling": "FP32 mean over every returned temporal and spatial token",
            "preprocessing": "resize shortest edge 292, center crop 256, ImageNet normalization",
            "context_padding": "repeat_tail to 64 frames", "isolated_padding": "repeat_single_frame to 64 frames",
        },
        "roles": roles,
        "spatial_branch": {
            "classification": "incompatible", "reused_samples": 0,
            "reason": "V2 deliberately stores only a pooled vector; spatial tokens cannot be reconstructed reversibly",
        },
        "actual_read_evidence": actual_read_evidence,
        "mapping_csv": str(args.mapping_csv),
        "selection_dependency_on_overlap_geometry": False,
        "notes": [
            "Exact matches are adapted, not called original-layout reuse, because the V2 tensor key/path schema differs.",
            "No teacher/target latent is mapped to predictor input.",
            "Unmatched and native roles must be extracted only as missing-role fills.",
        ],
    }
    if result["resolved"] != result["expected"]:
        raise ValueError(f"D-series coverage mismatch: {result['resolved']}")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with args.mapping_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mapping[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(mapping)
    md = f"""# V-JEPA2 V2 reuse audit\n\nStatus: `COMPLETE_VALID`\n\nD-series frozen coverage resolved exactly: **681 scenes, 2,721 windows, 5,442 samples** (4,890 train; 552 dev). Source manifest SHA256: `{D_MANIFEST_SHA256}`.\n\n| D-series role | Actual V2 role | Classification | Exact compatible coverage | Missing |\n|---|---|---:|---:|---:|\n| streaming/context | `streaming_context_p8` | adapted (pooled only) | {adapted['streaming_context']} / {windows} windows | {missing['streaming_context']} |\n| isolated target | `isolated_frame` | adapted (pooled only) | {adapted['isolated_target']} / {samples} samples | {missing['isolated_target']} |\n| native target reference | absent | missing | 0 / {samples} samples | {samples} |\n| native predictor output | absent | missing | 0 / {samples} samples | {samples} |\n| spatial tokens | pooled payload only | incompatible | 0 | all required spatial samples |\n\nThe adapter is identity-preserving: it reads safetensors key `feat` as the D-series pooled vector and performs no numeric transform. V2 uses the exact checkpoint revision `{MODEL_REVISION}`, layer 23, 64-frame repeat-tail/repeat-single inputs, 256 crop, ImageNet normalization, and FP32 mean over all 32x16x16 tokens. It cannot be relabelled as native target or native prediction.\n\nThe one-time full V2 coverage receipt verified all **12,358/12,358** V-JEPA2 request units; D-series training should cite that receipt and this mapping rather than rehashing the store. Five deterministic mapped payloads were independently loaded here.\n"""
    args.output_md.write_text(md)
    print(json.dumps({"status": result["status"], "roles": roles, "mapping_rows": len(mapping)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
