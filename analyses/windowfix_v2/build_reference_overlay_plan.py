#!/usr/bin/env python3
"""Build a zero-copy canonical-v2 legacy feature routing overlay.

The input must come from ``classify_feature_reuse.py`` with
``--payload-hash-policy reference``.  Exact records become read-only source
routes under ``LEGACY_EXACT_REFERENCE_V1``.  The planner does not read tensor
payload bytes, copy features, write route roots, call SCO, or start probing.

This is explicitly a temporal/physical-identity plan, not byte certification.
Unknown, missing, and mismatched candidates are all routed to re-extraction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from analyses.windowfix_v2.build_hybrid_reuse_plan import (
    CLASSIFIER_STATUSES,
    PlanError,
    _identity_key,
    _jsonl,
    _join_root,
    _preprocess_identity,
    _root_namespace,
    _safe_component,
    _write_atomic,
    canonical_sha,
    read_json,
    read_jsonl,
    sha256_file,
)


SCHEMA_VERSION = 1
REFERENCE_CONTRACT = "LEGACY_EXACT_REFERENCE_V1"
REFERENCE_NAMESPACES = {
    "streaming": "FEAT_STREAMING_CANONICAL_V2_MIXED_REFERENCE_V1",
    "target": "FEAT_TARGET_CANONICAL_V2_MIXED_REFERENCE_V1",
}


def _source_uri(row: Mapping[str, Any]) -> str:
    raw = str(row.get("source_uri") or row.get("feature_path") or "").strip()
    if not raw:
        raise PlanError("exact reference row has no source path/URI")
    parsed = urlparse(raw)
    if parsed.scheme:
        if row.get("source_exists_verified") is not True:
            raise PlanError("remote exact reference lacks source_exists_verified=true")
        return raw
    path = Path(raw)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise PlanError(f"exact reference source is not a local regular file: {path}")
    if path.stat().st_size != int(row.get("source_size_bytes", -1)):
        raise PlanError(f"exact reference source size changed: {path}")
    return path.as_uri()


def _validate_exact_reference(row: Mapping[str, Any]) -> None:
    required = (
        "spec_id",
        "vfm",
        "model_id",
        "layer",
        "scene_id",
        "source",
        "source_size_bytes",
        "raw_ids",
    )
    missing = [field for field in required if row.get(field) in (None, "")]
    if missing:
        raise PlanError(f"exact reference row lacks fields: {missing}")
    if int(row["source_size_bytes"]) <= 0:
        raise PlanError("exact reference row has invalid source_size_bytes")
    if row.get("source_payload_sha256") not in (None, ""):
        raise PlanError("reference classifier unexpectedly claims a recomputed payload SHA")
    if row.get("payload_sha256_verified") is not False:
        raise PlanError("reference classifier unexpectedly claims payload SHA verification")
    expected = row.get("expected_sidecar_identity")
    if not isinstance(expected, dict) or "prompt" not in expected:
        raise PlanError("exact reference lacks expected prompt identity")
    if not _preprocess_identity(expected):
        raise PlanError("exact reference lacks expected preprocessing identity")
    raw_ids = row.get("raw_ids")
    if not isinstance(raw_ids, list) or not raw_ids:
        raise PlanError("exact reference has no raw_ids")
    if str(row.get("unit_kind", "streaming_file")) == "target_row":
        if row.get("source_row_index") is None or row.get("target_raw_id") is None:
            raise PlanError("target reference lacks source_row_index/target_raw_id")


def _route_key(row: Mapping[str, Any]) -> str:
    unit_kind = str(row.get("unit_kind", "streaming_file"))
    mode = "target" if unit_kind == "target_row" else "streaming"
    vfm = _safe_component(row.get("vfm"), field="vfm")
    source = _safe_component(row.get("source"), field="source")
    scene = _safe_component(row.get("scene_id"), field="scene_id")
    timestep = row.get("timestep")
    layer = int(row["layer"])
    feature_name = (
        f"feature_layer{layer}.sft"
        if timestep is None
        else f"feature_t{int(timestep)}_layer{layer}.sft"
    )
    if mode == "target":
        parts = [vfm, source, scene, feature_name, f"raw_{int(row['target_raw_id']):06d}"]
    else:
        parts = [
            vfm,
            source,
            scene,
            f"window_{int(row['window_id']):04d}",
            f"prefix_{int(row['prefix']) - 1:06d}",
            feature_name,
        ]
    return str(PurePosixPath(*parts))


def build(
    *,
    classification_dir: Path,
    protocol_path: Path,
    policy_path: Path,
    route_roots: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    summary_path = classification_dir / "summary.json"
    summary = read_json(summary_path)
    protocol = read_json(protocol_path)
    policy = read_json(policy_path)
    if summary.get("payload_hash_policy") != "reference":
        raise PlanError("classifier must run with --payload-hash-policy reference")
    if summary.get("generation_contract") != REFERENCE_CONTRACT:
        raise PlanError("classifier contract is not LEGACY_EXACT_REFERENCE_V1")
    if summary.get("payload_full_read_performed") is not False:
        raise PlanError("reference classifier unexpectedly reports a payload full read")
    if summary.get("byte_level_certified") is not False:
        raise PlanError("reference classifier unexpectedly reports byte certification")
    canonical_protocol = str(protocol.get("canonical_protocol", ""))
    if not canonical_protocol or summary.get("canonical_protocol") != canonical_protocol:
        raise PlanError("classifier and temporal protocol do not match")
    if policy.get("contract") != REFERENCE_CONTRACT:
        raise PlanError("invalid reference policy contract")
    if policy.get("canonical_temporal_protocol") != canonical_protocol:
        raise PlanError("reference policy and temporal protocol do not match")
    if policy.get("route_namespaces") != REFERENCE_NAMESPACES:
        raise PlanError("unexpected reference route namespace names")
    for mode, namespace in REFERENCE_NAMESPACES.items():
        root = route_roots[mode]
        if _root_namespace(root) != namespace:
            raise PlanError(f"{mode} route root must end in {namespace}, found {root}")
        if "SEEDED_V1" in root:
            raise PlanError("reference overlay must never target SEEDED_V1")

    actions: dict[str, list[dict[str, Any]]] = {
        "reference_routes": [],
        "reextract_records": [],
        "unverified_records": [],
    }
    seen_identities: set[tuple[Any, ...]] = set()
    seen_routes: set[tuple[str, str]] = set()
    observed = {status: 0 for status in CLASSIFIER_STATUSES}
    classifier_files: dict[str, dict[str, Any]] = {}
    for status in CLASSIFIER_STATUSES:
        path = classification_dir / f"{status.lower()}.jsonl"
        classifier_files[status] = {"path": str(path), "sha256": sha256_file(path)}
        for row in read_jsonl(path):
            if row.get("status") != status:
                raise PlanError(f"{path}: row status disagrees with worklist")
            identity = _identity_key(row)
            if identity in seen_identities:
                raise PlanError(f"duplicate canonical identity: {identity}")
            seen_identities.add(identity)
            observed[status] += 1
            unit_kind = str(row.get("unit_kind", "streaming_file"))
            mode = "target" if unit_kind == "target_row" else "streaming"
            route_key = _route_key(row)
            base = {
                "schema_version": SCHEMA_VERSION,
                "contract": REFERENCE_CONTRACT,
                "byte_level_certified": False,
                "payload_full_read_performed": False,
                "payload_copy_performed": False,
                "canonical_protocol": canonical_protocol,
                "spec_id": row.get("spec_id"),
                "unit_kind": unit_kind,
                "vfm": row.get("vfm"),
                "model_id": row.get("model_id"),
                "layer": row.get("layer"),
                "timestep": row.get("timestep"),
                "tensor_key": row.get("tensor_key", "feat"),
                "source": row.get("source"),
                "scene_id": row.get("scene_id"),
                "window_id": row.get("window_id"),
                "prefix": row.get("prefix"),
                "raw_ids": row.get("raw_ids"),
                "route_mode": mode,
                "route_root": route_roots[mode],
                "route_key": route_key,
                "logical_route_uri": _join_root(route_roots[mode], route_key),
                "classifier_status": status,
                "classifier_reason_codes": row.get("reason_codes", []),
            }
            if status == "REUSE_EXACT":
                _validate_exact_reference(row)
                source_uri = _source_uri(row)
                expected = row["expected_sidecar_identity"]
                base.update(
                    {
                        "action": "REFERENCE_SOURCE_READ_ONLY",
                        "source_uri": source_uri,
                        "source_size_bytes": int(row["source_size_bytes"]),
                        "declared_source_payload_sha256": row.get(
                            "declared_source_payload_sha256"
                        ),
                        "declared_sha256_verified": False,
                        "legacy_root_id": row.get("legacy_root_id"),
                        "prompt": expected["prompt"],
                        "preprocess_identity": _preprocess_identity(expected),
                    }
                )
                if unit_kind == "target_row":
                    base.update(
                        {
                            "source_row_index": int(row["source_row_index"]),
                            "target_raw_id": int(row["target_raw_id"]),
                        }
                    )
                route_identity = (route_roots[mode], route_key)
                if route_identity in seen_routes:
                    raise PlanError(f"duplicate logical route: {route_identity}")
                seen_routes.add(route_identity)
                actions["reference_routes"].append(base)
            elif status == "UNVERIFIED":
                base["action"] = "REEXTRACT_UNVERIFIED"
                actions["unverified_records"].append(base)
            else:
                base["action"] = "REEXTRACT"
                actions["reextract_records"].append(base)

    expected_counts = summary.get("status_counts")
    if not isinstance(expected_counts, dict) or observed != {
        status: int(expected_counts.get(status, 0)) for status in CLASSIFIER_STATUSES
    }:
        raise PlanError("classifier summary counts disagree with worklists")
    for rows in actions.values():
        rows.sort(
            key=lambda row: json.dumps(
                _identity_key(row), separators=(",", ":"), ensure_ascii=False
            )
        )
    plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "plan_type": "canonical_v2_legacy_reference_overlay",
        "contract": REFERENCE_CONTRACT,
        "status": "DRY_RUN_REFERENCE_PLAN_ONLY",
        "byte_level_certified": False,
        "payload_full_read_performed": False,
        "payload_copy_performed": False,
        "route_roots_written": False,
        "sco_submission_performed": False,
        "probing_training_in_scope": False,
        "canonical_protocol": canonical_protocol,
        "canonical_protocol_file": str(protocol_path),
        "canonical_protocol_file_sha256": sha256_file(protocol_path),
        "reference_policy_file": str(policy_path),
        "reference_policy_file_sha256": sha256_file(policy_path),
        "classification_summary": {
            "path": str(summary_path),
            "sha256": sha256_file(summary_path),
        },
        "classification_worklists": classifier_files,
        "route_roots": dict(route_roots),
        "route_namespaces": REFERENCE_NAMESPACES,
        "seeded_namespace_written": False,
        "counts": {
            "canonical_records": sum(observed.values()),
            "reference_routes": len(actions["reference_routes"]),
            "reextract": len(actions["reextract_records"])
            + len(actions["unverified_records"]),
            "reextract_due_to_unverified": len(actions["unverified_records"]),
            "classifier_statuses": observed,
        },
        "action_worklists": {
            f"{name}.jsonl": {
                "records": len(rows),
                "content_sha256": hashlib.sha256(_jsonl(rows).encode()).hexdigest(),
            }
            for name, rows in sorted(actions.items())
        },
    }
    plan["content_sha256"] = canonical_sha(plan)
    return plan, actions


def write_outputs(
    output_dir: Path,
    plan: Mapping[str, Any],
    actions: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    _write_atomic(
        output_dir / "reference_overlay_plan.json",
        json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    for name, rows in actions.items():
        _write_atomic(output_dir / f"{name}.jsonl", _jsonl(rows))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("protocols/inscene_feature_reference_legacy_exact_v1.json"),
    )
    parser.add_argument("--streaming-route-root", required=True)
    parser.add_argument("--target-route-root", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    plan, actions = build(
        classification_dir=args.classification_dir.resolve(),
        protocol_path=args.protocol.resolve(),
        policy_path=args.policy.resolve(),
        route_roots={
            "streaming": args.streaming_route_root,
            "target": args.target_route_root,
        },
    )
    write_outputs(args.output_dir.resolve(), plan, actions)
    print(json.dumps(plan["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
