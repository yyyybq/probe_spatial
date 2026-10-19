#!/usr/bin/env python3
"""Build a new immutable target merge plan from completed gap deltas.

The input plan and route audit are never modified.  Every missing raw-ID row
must be represented by the audit's hash-bound gap worklists and by one exact
per-scene/per-layer delta payload.  Completed gap-run shard receipts are also
required.  The resulting plan remains non-materializing; the target merge
writer must freshly audit it and reach PASS before it may publish anything.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyses.windowfix_v2.audit_target_row_routes import (
    CANONICAL_PROTOCOL,
    CONTRACT,
    GAP_CONTRACT,
    GAP_TARGET_NAMESPACE,
    PLAN_TYPE,
    RouteAuditError,
    _source_gate,
    audit_plan,
    canonical_json_sha,
    canonical_target_unions,
    read_json,
    sha256_file,
    validate_destination_root,
)


class AugmentError(ValueError):
    """Raised before an incomplete or ambiguous augmented plan is published."""


def _content_sha(value: Mapping[str, Any]) -> str:
    return canonical_json_sha(
        {key: item for key, item in value.items() if key != "content_sha256"}
    )


def _require_content_sha(value: Mapping[str, Any], label: str) -> str:
    recorded = value.get("content_sha256")
    actual = _content_sha(value)
    if recorded != actual:
        raise AugmentError(f"{label} content SHA mismatch")
    return actual


def _regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise AugmentError(f"{label} must be a regular non-symlink file: {path}")
    return path.resolve()


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    path = _regular(path, label)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AugmentError(f"{label} line {line_number} is invalid JSON") from exc
            if not isinstance(value, dict):
                raise AugmentError(f"{label} line {line_number} is not an object")
            rows.append(value)
    return rows


def _unit_key(value: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(value["vfm"]),
        str(value["model_id"]),
        int(value["timestep"]),
        int(value["layer"]),
        str(value["source"]),
        str(value["scene_id"]),
    )


def _worklist_unit_key(
    worklist: Mapping[str, Any], scene: str, layer: int
) -> tuple[Any, ...]:
    return (
        str(worklist["vfm"]),
        str(worklist["model_id"]),
        int(worklist["timestep"]),
        int(layer),
        str(worklist["source"]),
        str(scene),
    )


def _validate_route_audit(
    plan_path: Path, plan: Mapping[str, Any], audit_path: Path
) -> tuple[dict[str, Any], dict[tuple[Any, ...], tuple[int, ...]], list[dict[str, Any]]]:
    audit_path = _regular(audit_path, "target route audit")
    report = read_json(audit_path, "target route audit")
    _require_content_sha(report, "target route audit")
    if report.get("audit_type") != "canonical_v2_target_row_route_audit":
        raise AugmentError("target route audit type mismatch")
    if report.get("contract") != CONTRACT or report.get("status") == "FAIL":
        raise AugmentError("target route audit is failed or has the wrong contract")
    if report.get("plan_sha256") != sha256_file(plan_path):
        raise AugmentError("target route audit does not bind the input plan")
    canonical = plan.get("canonical_source")
    if not isinstance(canonical, dict):
        raise AugmentError("input plan lacks canonical_source")
    if report.get("canonical_source", {}).get("sha256") != canonical.get("sha256"):
        raise AugmentError("target route audit canonical source mismatch")

    outputs = report.get("outputs")
    if not isinstance(outputs, dict):
        raise AugmentError("target route audit has no immutable outputs")
    missing_binding = outputs.get("missing_target_units")
    if not isinstance(missing_binding, dict):
        raise AugmentError("target route audit lacks missing_target_units binding")
    missing_path = _regular(Path(str(missing_binding.get("path", ""))), "missing units")
    if sha256_file(missing_path) != missing_binding.get("sha256"):
        raise AugmentError("missing_target_units SHA mismatch")
    missing_rows = _read_jsonl(missing_path, "missing target units")
    missing: dict[tuple[Any, ...], tuple[int, ...]] = {}
    for row in missing_rows:
        key = _unit_key(row)
        raw_ids = tuple(int(value) for value in row.get("missing_raw_ids", []))
        if key in missing or not raw_ids or len(raw_ids) != len(set(raw_ids)):
            raise AugmentError(f"duplicate/empty missing target unit: {key}")
        if tuple(sorted(raw_ids)) != raw_ids:
            raise AugmentError(f"missing raw IDs are not sorted: {key}")
        missing[key] = raw_ids

    worklists: list[dict[str, Any]] = []
    derived: dict[tuple[Any, ...], set[int]] = defaultdict(set)
    bindings = outputs.get("gap_worklists")
    if not isinstance(bindings, list):
        raise AugmentError("target route audit gap_worklists binding is malformed")
    for binding in bindings:
        if not isinstance(binding, dict):
            raise AugmentError("gap worklist binding is not an object")
        path = _regular(Path(str(binding.get("path", ""))), "gap worklist")
        if sha256_file(path) != binding.get("file_sha256"):
            raise AugmentError("gap worklist file SHA mismatch")
        worklist = read_json(path, "gap worklist")
        content_sha = _require_content_sha(worklist, "gap worklist")
        if content_sha != binding.get("content_sha256"):
            raise AugmentError("route audit/worklist content SHA mismatch")
        if (
            worklist.get("schema_version") != 1
            or worklist.get("contract") != GAP_CONTRACT
            or worklist.get("canonical_protocol") != CANONICAL_PROTOCOL
            or worklist.get("canonical_source_sha256") != canonical.get("sha256")
            or worklist.get("mode") != "target_isolated"
            or worklist.get("feature_namespace") != GAP_TARGET_NAMESPACE
        ):
            raise AugmentError("gap worklist semantic contract mismatch")
        worklist = dict(worklist)
        worklist["_path"] = str(path)
        worklist["_file_sha256"] = binding["file_sha256"]
        worklists.append(worklist)
        output_layers = {int(value) for value in worklist.get("output_layers", [])}
        for entry in worklist.get("entries", []):
            if not isinstance(entry, dict):
                raise AugmentError("gap worklist entry is not an object")
            scene = str(entry.get("scene", ""))
            raw_id = int(entry["frame_index"])
            layers = {int(value) for value in entry.get("layers", [])}
            if not scene or not layers or not layers <= output_layers:
                raise AugmentError("gap worklist entry scene/layer mismatch")
            for layer in layers:
                key = _worklist_unit_key(worklist, scene, layer)
                if raw_id in derived[key]:
                    raise AugmentError(f"duplicate worklist raw ID: {key}:{raw_id}")
                derived[key].add(raw_id)
    frozen_derived = {key: tuple(sorted(values)) for key, values in derived.items()}
    if frozen_derived != missing:
        missing_keys = sorted(set(missing) - set(frozen_derived))
        extra_keys = sorted(set(frozen_derived) - set(missing))
        changed = sorted(
            key
            for key in set(missing) & set(frozen_derived)
            if missing[key] != frozen_derived[key]
        )
        raise AugmentError(
            "gap worklists do not exactly equal audited missing rows: "
            f"missing_keys={missing_keys[:3]} extra_keys={extra_keys[:3]} changed={changed[:3]}"
        )
    if not missing:
        raise AugmentError("route audit contains no definite target gaps to augment")
    return report, missing, worklists


def _validate_completion_bindings(
    bindings: Sequence[tuple[Path, Path]],
    *,
    worklists: Sequence[Mapping[str, Any]],
    audit_path: Path,
    audit_report: Mapping[str, Any],
    canonical_source_sha256: str,
    delta_root: Path,
) -> list[dict[str, Any]]:
    required = {str(item["_file_sha256"]): item for item in worklists}
    coverage: dict[str, list[tuple[int, int]]] = defaultdict(list)
    result: list[dict[str, Any]] = []
    seen_shards: set[str] = set()
    for raw_shard, raw_receipt in bindings:
        shard_path = _regular(raw_shard, "gap shard")
        receipt_path = _regular(raw_receipt, "gap completion receipt")
        shard = read_json(shard_path, "gap shard")
        shard_content = _require_content_sha(shard, "gap shard")
        receipt = read_json(receipt_path, "gap completion receipt")
        worklist_sha = str(shard.get("worklist_file_sha256", ""))
        worklist = required.get(worklist_sha)
        if worklist is None:
            raise AugmentError("completion binding references an unaudited worklist")
        shard_id = str(shard.get("shard_id", ""))
        if not shard_id or shard_id in seen_shards:
            raise AugmentError("duplicate or empty gap shard_id")
        seen_shards.add(shard_id)
        expected = {
            "schema_version": 1,
            "status": "READY",
            "contract": GAP_CONTRACT,
            "mode": "target_isolated",
            "feature_namespace": GAP_TARGET_NAMESPACE,
            "feature_root": str(delta_root),
            "canonical_source_sha256": canonical_source_sha256,
            "worklist_file_sha256": worklist_sha,
            "worklist_content_sha256": worklist["content_sha256"],
            "source_audit_sha256": sha256_file(audit_path),
            "source_audit_content_sha256": audit_report["content_sha256"],
            "vfm": worklist["vfm"],
            "model_id": worklist["model_id"],
            "timestep": worklist["timestep"],
            "source": worklist["source"],
        }
        for field, value in expected.items():
            if shard.get(field) != value:
                raise AugmentError(f"gap shard {field} mismatch: {shard_id}")
        if {int(value) for value in shard.get("layers", [])} != {
            int(value) for value in worklist.get("output_layers", [])
        }:
            raise AugmentError(f"gap shard layer set mismatch: {shard_id}")
        if receipt.get("status") != "COMPLETE":
            raise AugmentError(f"gap receipt is not COMPLETE: {receipt_path}")
        receipt_expected = {
            "shard_id": shard_id,
            "shard_content_sha256": shard_content,
            "worklist_file_sha256": worklist_sha,
            "worklist_content_sha256": worklist["content_sha256"],
            "canonical_source_sha256": canonical_source_sha256,
        }
        for field, value in receipt_expected.items():
            if receipt.get(field) != value:
                raise AugmentError(f"gap receipt {field} mismatch: {receipt_path}")
        start, end = int(shard.get("scene_start", -1)), int(shard.get("scene_end", -1))
        if start < 0 or end <= start:
            raise AugmentError(f"invalid gap shard scene range: {shard_id}")
        processes = receipt.get("processes")
        if not isinstance(processes, list) or not processes:
            raise AugmentError(f"gap receipt has no process coverage: {receipt_path}")
        ranges = sorted(
            (int(item["start"]), int(item["end"]))
            for item in processes
            if isinstance(item, dict) and int(item.get("returncode", -1)) == 0
        )
        cursor = start
        for left, right in ranges:
            if left != cursor or right <= left or right > end:
                raise AugmentError(f"gap receipt process coverage is not exact: {receipt_path}")
            cursor = right
        if cursor != end or len(ranges) != len(processes):
            raise AugmentError(f"gap receipt process coverage is incomplete: {receipt_path}")
        coverage[worklist_sha].append((start, end))
        result.append(
            {
                "shard_path": str(shard_path),
                "shard_file_sha256": sha256_file(shard_path),
                "shard_content_sha256": shard_content,
                "receipt_path": str(receipt_path),
                "receipt_file_sha256": sha256_file(receipt_path),
                "worklist_file_sha256": worklist_sha,
                "scene_start": start,
                "scene_end": end,
            }
        )
    if not result:
        raise AugmentError("at least one completed gap shard receipt is required")
    complete_scene_ends: set[int] = set()
    for worklist_sha in required:
        ranges = sorted(coverage.get(worklist_sha, []))
        cursor = 0
        for left, right in ranges:
            if left != cursor:
                raise AugmentError(
                    f"gap shard receipts do not exactly cover all scenes: {worklist_sha}"
                )
            cursor = right
        if cursor <= 0:
            raise AugmentError(
                f"gap shard receipts do not exactly cover all scenes: {worklist_sha}"
            )
        complete_scene_ends.add(cursor)
    if len(complete_scene_ends) != 1:
        raise AugmentError("gap shard receipts disagree on the dataset scene range")
    return sorted(result, key=lambda item: (item["worklist_file_sha256"], item["scene_start"]))


def _atomic_json_noreplace(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    if path.exists() or path.is_symlink():
        raise AugmentError(f"augmented plan already exists; overwrite forbidden: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise AugmentError(
                f"augmented plan concurrently appeared; overwrite forbidden: {path}"
            ) from exc
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def augment_plan(
    *,
    initial_plan_path: Path,
    route_audit_path: Path,
    delta_root: Path,
    completion_bindings: Sequence[tuple[Path, Path]],
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists() or output_path.is_symlink():
        raise AugmentError(
            f"augmented plan already exists; overwrite forbidden: {output_path}"
        )
    initial_plan_path = _regular(initial_plan_path, "initial target merge plan")
    route_audit_path = _regular(route_audit_path, "target route audit")
    initial = read_json(initial_plan_path, "initial target merge plan")
    if (
        initial.get("schema_version") != 1
        or initial.get("plan_type") != PLAN_TYPE
        or initial.get("contract") != CONTRACT
        or initial.get("canonical_protocol") != CANONICAL_PROTOCOL
    ):
        raise AugmentError("initial target merge plan contract/type mismatch")
    initial_content = _require_content_sha(initial, "initial target merge plan")
    if delta_root.is_symlink():
        raise AugmentError("delta root may not be a symlink")
    delta_root = delta_root.resolve()
    if (
        delta_root.name != GAP_TARGET_NAMESPACE
        or "SEEDED_V1" in str(delta_root)
        or not delta_root.is_dir()
    ):
        raise AugmentError(f"delta root must be an existing {GAP_TARGET_NAMESPACE} directory")
    report, missing, worklists = _validate_route_audit(
        initial_plan_path, initial, route_audit_path
    )
    canonical_spec = initial["canonical_source"]
    canonical_path = Path(str(canonical_spec["path"]))
    canonical_targets, canonical_meta = canonical_target_unions(
        canonical_path, str(canonical_spec["sha256"])
    )
    receipts = _validate_completion_bindings(
        completion_bindings,
        worklists=worklists,
        audit_path=route_audit_path,
        audit_report=report,
        canonical_source_sha256=canonical_meta["sha256"],
        delta_root=delta_root,
    )
    worklist_by_unit: dict[tuple[Any, ...], dict[str, Any]] = {}
    for worklist in worklists:
        for entry in worklist["entries"]:
            for layer in entry["layers"]:
                key = _worklist_unit_key(worklist, str(entry["scene"]), int(layer))
                prior = worklist_by_unit.setdefault(key, worklist)
                if prior["_file_sha256"] != worklist["_file_sha256"]:
                    raise AugmentError(f"one target unit is split across worklists: {key}")

    augmented = copy.deepcopy(initial)
    added = 0
    added_rows = 0
    additions: list[dict[str, Any]] = []
    units_by_key = {_unit_key(unit): unit for unit in augmented.get("units", [])}
    if len(units_by_key) != len(augmented.get("units", [])):
        raise AugmentError("initial plan contains duplicate units")
    output_root = validate_destination_root(augmented.get("destination_root"))
    for key, raw_ids in sorted(missing.items()):
        unit = units_by_key.get(key)
        if unit is None:
            raise AugmentError(f"missing audit unit is absent from initial plan: {key}")
        if unit.get("duplicate_owner_policy") == "require_bitwise_equal_then_priority":
            raise AugmentError(
                "DINO conditional multi-prefix routes must first be bitwise-validated and "
                "frozen into one unambiguous legacy source before adding a gap delta"
            )
        worklist = worklist_by_unit[key]
        scene_dir = delta_root / key[0] / key[4] / key[5]
        delta_dir = (
            scene_dir
            / ".canonical_v2_target_delta"
            / str(worklist["_file_sha256"])[:16]
        )
        feature = delta_dir / f"feature_t{key[2]}_layer{key[3]}.delta.sft"
        index = delta_dir / f"target_indices_layer{key[3]}.npy"
        source_id = f"canonical_v2_gap_{str(worklist['_file_sha256'])[:16]}"
        source = {
            "source_id": source_id,
            "kind": "gap_delta",
            "priority": 1000,
            "candidate_gate_status": "COMPLETE_GAP_DELTA_VERIFIED",
            "feature_path": str(feature),
            "index_path": str(index),
            "feature_sidecar_path": f"{feature}.manifest.json",
            "index_sidecar_path": f"{index}.manifest.json",
            "gap_worklist_path": str(worklist["_path"]),
            "gap_worklist_file_sha256": worklist["_file_sha256"],
            "gap_worklist_content_sha256": worklist["content_sha256"],
        }
        if source_id in {str(item.get("source_id")) for item in unit.get("sources", [])}:
            raise AugmentError(f"gap source_id already exists in unit: {key}")
        expected_scene = canonical_targets[(key[4], key[5])]
        try:
            routes, evidence = _source_gate(
                source,
                unit,
                expected_raw_ids=expected_scene,
                canonical_protocol=CANONICAL_PROTOCOL,
                canonical_source_sha256=canonical_meta["sha256"],
                output_root=output_root,
                verify_source_sha=True,
            )
        except (OSError, KeyError, TypeError, ValueError, RouteAuditError) as exc:
            raise AugmentError(f"gap delta gate failed for {key}: {exc}") from exc
        routed_ids = tuple(int(item["target_raw_id"]) for item in routes)
        if routed_ids != raw_ids or evidence.get("ignored_noncanonical_rows") != 0:
            raise AugmentError(
                f"gap delta raw IDs are not the exact audited missing set: {key}"
            )
        if evidence.get("payload_sha256_verified") is not True:
            raise AugmentError(f"gap delta payload did not receive a full SHA gate: {key}")
        source["expected_payload_sha256"] = evidence["payload_sha256"]
        source["expected_index_sha256"] = evidence["index_sha256"]
        unit.setdefault("sources", []).append(source)
        added += 1
        added_rows += len(raw_ids)
        additions.append(
            {
                "unit": list(key),
                "rows": len(raw_ids),
                "source_id": source_id,
                "feature_path": str(feature),
                "feature_sha256": evidence["payload_sha256"],
                "index_path": str(index),
                "index_sha256": evidence["index_sha256"],
                "worklist_file_sha256": worklist["_file_sha256"],
            }
        )

    augmented.pop("content_sha256", None)
    augmented["augmentation"] = {
        "contract": "CANONICAL_V2_TARGET_DELTA_AUGMENT_V1",
        "parent_plan": {
            "path": str(initial_plan_path),
            "file_sha256": sha256_file(initial_plan_path),
            "content_sha256": initial_content,
        },
        "route_audit": {
            "path": str(route_audit_path),
            "file_sha256": sha256_file(route_audit_path),
            "content_sha256": report["content_sha256"],
        },
        "delta_root": str(delta_root),
        "augmenter_source": str(Path(__file__).resolve()),
        "augmenter_source_sha256": sha256_file(Path(__file__).resolve()),
        "completion_bindings": receipts,
        "counts": {
            "gap_sources_added": added,
            "gap_row_layer_items_added": added_rows,
            "worklists": len(worklists),
            "completion_receipts": len(receipts),
        },
        "additions": additions,
    }
    augmented["content_sha256"] = canonical_json_sha(augmented)

    # A fresh whole-plan preflight catches structural conflicts.  Remote
    # sources remain explicitly pending and are never contacted here.
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(augmented, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.preflight-", dir=output_path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
        preflight = audit_plan(
            temporary,
            verify_source_sha=False,
            remote_policy="pending",
            source_timeout_seconds=None,
        )
    finally:
        temporary.unlink(missing_ok=True)
    if preflight["report"]["status"] == "FAIL" or preflight["report"]["counts"][
        "hard_errors"
    ]:
        raise AugmentError("augmented plan failed structural route preflight")
    _atomic_json_noreplace(output_path, augmented)
    return {
        "status": "AUGMENTED_PLAN_WRITTEN_NOT_MATERIALIZED",
        "output": str(output_path),
        "output_file_sha256": sha256_file(output_path),
        "output_content_sha256": augmented["content_sha256"],
        "preflight_status": preflight["report"]["status"],
        "preflight_counts": preflight["report"]["counts"],
        **augmented["augmentation"]["counts"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-plan", required=True, type=Path)
    parser.add_argument("--route-audit", required=True, type=Path)
    parser.add_argument("--delta-root", required=True, type=Path)
    parser.add_argument(
        "--completion-binding",
        required=True,
        action="append",
        nargs=2,
        metavar=("SHARD_JSON", "COMPLETE_RECEIPT_JSON"),
        type=Path,
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = augment_plan(
            initial_plan_path=args.initial_plan,
            route_audit_path=args.route_audit,
            delta_root=args.delta_root,
            completion_bindings=[tuple(value) for value in args.completion_binding],
            output_path=args.output,
        )
    except (AugmentError, RouteAuditError, OSError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
