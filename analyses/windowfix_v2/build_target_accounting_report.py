#!/usr/bin/env python3
"""Build immutable, set-based accounting for the canonical-v2 target plan.

This is deliberately a local/read-only analysis.  It binds the frozen target
merge plan and its offline route audit, then compares the canonical raw-ID
universe with the *actual* historical flat and manifest-exact index files.
It never contacts AOSS, writes feature payloads, or changes an existing plan.

The important distinction made by this report is between:

* the current exhaustive route-audit partition (routed/pending/missing), and
* the older pre-gate best-case forecast, which assumes every remote cache has
  the same future-only raw-ID coverage proven by the local flat indexes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE = PROJECT_ROOT / "artifacts/windowfix_v2/target_row_merge_initial"
DEFAULT_PLAN = BASE / "target_merge_plan.json"
DEFAULT_SUMMARY = BASE / "target_merge_plan.summary.json"
DEFAULT_AUDIT_DIR = BASE / "route_audit_offline_pending_r5"
DEFAULT_OUTPUT_JSON = BASE / "target_accounting_r6_20260831.json"
DEFAULT_OUTPUT_MD = BASE / "target_accounting_r6_20260831.md"

FLAT_MINIMAX_ROOT = Path(
    "/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_TARGET/minimax_h3/scannetpp"
)
FLAT_WAN_ROOT = Path(
    "/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_TARGET/wan/scannetpp"
)
WAN_DOT_ROOT = Path(
    "/mnt/umm/users/yinbaiqiao/InsScene-15K/FEATURES_VERSIONED/"
    "wan_p8_action_dynamics_canonical_manifest_exact_20260821_200604_2075f3e/"
    "target/wan/scannetpp"
)

PREFIXES = (8, 12, 16, 24)
HORIZONS = (1, 2, 4)
WAN_DOT_LAYERS = (0, 5, 10, 15, 25, 29)
EXPECTED_REMOTE_SPECS = {
    ("dino", 0),
    ("f3r", 24),
    *(("lingbot", layer) for layer in (10, 20, 30, 40, 47)),
    *(("minimax_h3", layer) for layer in (10, 20, 30, 40, 49)),
    *(("videomae", layer) for layer in (0, 5, 11, 17, 23)),
    *(("vjepa2", layer) for layer in (0, 5, 11, 17, 23)),
}
EXPECTED_NO_SOURCE_SPECS = {
    ("aether", 1),
    ("cogvideox", 20),
    ("qwen2_5_vl", -1),
    ("wan", 20),
}


class AccountingError(ValueError):
    """Raised when a frozen input or an accounting identity is inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise AccountingError(f"{label} must be a regular non-symlink file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AccountingError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AccountingError(f"{label} must contain a JSON object: {path}")
    return value


def read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise AccountingError(f"{label} must be a regular non-symlink file: {path}")
    result: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AccountingError(
                    f"malformed {label} line {line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise AccountingError(f"non-object {label} line {line_number}")
            result.append(value)
    return result


def _check_embedded_content_sha(value: Mapping[str, Any], label: str) -> None:
    recorded = value.get("content_sha256")
    actual = canonical_sha({key: item for key, item in value.items() if key != "content_sha256"})
    if recorded != actual:
        raise AccountingError(f"{label} content_sha256 mismatch: {recorded} != {actual}")


def canonical_sets(
    path: Path,
) -> tuple[
    dict[str, set[int]],
    dict[str, set[int]],
    dict[str, set[int]],
    dict[str, dict[int, set[str]]],
    int,
]:
    """Return all/future/anchor sets plus raw-ID role evidence per scene."""
    all_rows: dict[str, set[int]] = defaultdict(set)
    future_rows: dict[str, set[int]] = defaultdict(set)
    anchor_rows: dict[str, set[int]] = defaultdict(set)
    roles: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    windows: set[tuple[str, int]] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source") != "scannetpp":
                raise AccountingError(f"unexpected source on canonical line {line_number}")
            scene = str(row["scene_id"])
            window_id = int(row["streaming_window_id"])
            key = (scene, window_id)
            if key in windows:
                raise AccountingError(f"duplicate canonical window: {key}")
            windows.add(key)
            indices = tuple(int(item) for item in row["streaming_window_indices"])
            prefix24 = tuple(int(item) for item in row["streaming_indices"])
            if len(indices) != 28 or prefix24 != indices[:24]:
                raise AccountingError(f"canonical line {line_number} is not p24/obs28")
            if any(right <= left for left, right in zip(indices, indices[1:])):
                raise AccountingError(f"canonical line {line_number} is not strictly increasing")
            for prefix in PREFIXES:
                raw_id = indices[prefix - 1]
                all_rows[scene].add(raw_id)
                anchor_rows[scene].add(raw_id)
                roles[scene][raw_id].add(f"p{prefix}:anchor")
                for horizon in HORIZONS:
                    raw_id = indices[prefix - 1 + horizon]
                    all_rows[scene].add(raw_id)
                    future_rows[scene].add(raw_id)
                    roles[scene][raw_id].add(f"p{prefix}:h{horizon}")
    return dict(all_rows), dict(future_rows), dict(anchor_rows), dict(roles), len(windows)


def set_payload_sha(rows: Mapping[str, set[int]]) -> str:
    return canonical_sha(
        [
            {"scene_id": scene, "target_raw_ids": sorted(values)}
            for scene, values in sorted(rows.items())
        ]
    )


def load_index_root(
    root: Path,
    scenes: Iterable[str],
    *,
    label: str,
) -> tuple[dict[str, set[int]], dict[str, Any]]:
    """Load and sidecar-bind every target_indices.npy under one root."""
    rows: dict[str, set[int]] = {}
    file_receipts: list[dict[str, Any]] = []
    for scene in sorted(scenes):
        path = root / scene / "target_indices.npy"
        sidecar_path = Path(f"{path}.manifest.json")
        if path.is_symlink() or not path.is_file():
            raise AccountingError(f"missing {label} index: {path}")
        sidecar = read_json(sidecar_path, f"{label} index sidecar")
        file_sha = sha256_file(path)
        if sidecar.get("file") != path.name:
            raise AccountingError(f"{label} sidecar filename mismatch: {sidecar_path}")
        if sidecar.get("sha256") != file_sha:
            raise AccountingError(f"{label} sidecar SHA mismatch: {sidecar_path}")
        if int(sidecar.get("size_bytes", -1)) != path.stat().st_size:
            raise AccountingError(f"{label} sidecar size mismatch: {sidecar_path}")
        try:
            array = np.load(path, allow_pickle=False)
        except Exception as exc:
            raise AccountingError(f"cannot load {label} index {path}: {exc}") from exc
        if array.ndim != 1 or array.dtype != np.dtype("int64"):
            raise AccountingError(f"{label} index must be one-dimensional int64: {path}")
        values = tuple(int(item) for item in array.tolist())
        if any(right <= left for left, right in zip(values, values[1:])):
            raise AccountingError(f"{label} index must be sorted and unique: {path}")
        if sidecar.get("shape") != [len(values)] or sidecar.get("dtype") != "int64":
            raise AccountingError(f"{label} sidecar shape/dtype mismatch: {sidecar_path}")
        if sidecar.get("target_raw_ids") is not None and sidecar["target_raw_ids"] != list(values):
            raise AccountingError(f"{label} sidecar target_raw_ids mismatch: {sidecar_path}")
        rows[scene] = set(values)
        file_receipts.append(
            {
                "scene_id": scene,
                "index_sha256": file_sha,
                "sidecar_sha256": sha256_file(sidecar_path),
                "rows": len(values),
            }
        )
    metadata = {
        "label": label,
        "root": str(root),
        "scenes": len(rows),
        "rows": sum(len(value) for value in rows.values()),
        "raw_id_set_sha256": set_payload_sha(rows),
        "file_receipts_sha256": canonical_sha(file_receipts),
        "every_index_sha_and_sidecar_verified": True,
    }
    return rows, metadata


def sum_sets(rows: Mapping[str, set[int]]) -> int:
    return sum(len(value) for value in rows.values())


def _spec_key(value: Mapping[str, Any]) -> tuple[str, int]:
    return str(value["vfm"]), int(value["layer"])


def _aggregate_audit_units(units: Iterable[Mapping[str, Any]]) -> dict[tuple[str, int], dict[str, int]]:
    aggregate: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
    for unit in units:
        key = _spec_key(unit)
        aggregate[key].update(
            expected_rows=int(unit["expected_rows"]),
            routed_rows=int(unit["routed_rows"]),
            pending_rows=int(unit["pending_rows"]),
            missing_rows=int(unit["missing_rows"]),
            duplicate_rows=int(unit["duplicate_rows"]),
            conditional_duplicate_rows=int(unit["conditional_duplicate_rows"]),
            units=1,
        )
    return {key: dict(counts) for key, counts in aggregate.items()}


def _input_receipt(path: Path, *, content_sha: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "file_sha256": sha256_file(path)}
    if content_sha is not None:
        result["content_sha256"] = content_sha
    return result


def build_report(
    *,
    plan_path: Path,
    summary_path: Path,
    audit_dir: Path,
) -> dict[str, Any]:
    plan = read_json(plan_path, "target merge plan")
    summary = read_json(summary_path, "target merge plan summary")
    audit_path = audit_dir / "target_row_route_audit.json"
    audit = read_json(audit_path, "offline target route audit")
    _check_embedded_content_sha(plan, "target merge plan")
    _check_embedded_content_sha(audit, "offline target route audit")

    plan_sha = sha256_file(plan_path)
    if summary.get("plan_file_sha256") != plan_sha or audit.get("plan_sha256") != plan_sha:
        raise AccountingError("summary/audit are not bound to the supplied target merge plan")
    if summary.get("plan_content_sha256") != plan.get("content_sha256"):
        raise AccountingError("summary plan content binding mismatch")
    canonical_path = Path(plan["canonical_source"]["path"])
    if sha256_file(canonical_path) != plan["canonical_source"]["sha256"]:
        raise AccountingError("canonical source SHA mismatch")

    routes_path = audit_dir / "target_row_routes.jsonl"
    missing_path = audit_dir / "missing_target_units.jsonl"
    pending_path = audit_dir / "pending_target_units.jsonl"
    routes = read_jsonl(routes_path, "target routes")
    missing_units = read_jsonl(missing_path, "missing target units")
    pending_units = read_jsonl(pending_path, "pending target units")
    for key, path, rows in (
        ("target_row_routes", routes_path, routes),
        ("missing_target_units", missing_path, missing_units),
        ("pending_target_units", pending_path, pending_units),
    ):
        declared = audit["outputs"][key]
        if declared["sha256"] != sha256_file(path) or int(declared["records"]) != len(rows):
            raise AccountingError(f"audit output binding mismatch for {key}")

    all_rows, future_rows, anchor_rows, roles, windows = canonical_sets(canonical_path)
    if set(all_rows) != {str(unit["scene_id"]) for unit in plan["units"]}:
        raise AccountingError("plan scenes and canonical scenes differ")
    canonical_count = sum_sets(all_rows)
    future_count = sum_sets(future_rows)
    if windows != 2721 or len(all_rows) != 681 or canonical_count != 37677:
        raise AccountingError("unexpected canonical target universe cardinality")
    if audit["canonical_source"]["target_universe_sha256"] != canonical_sha(
        [
            {"source": "scannetpp", "scene_id": scene, "target_raw_ids": sorted(values)}
            for scene, values in sorted(all_rows.items())
        ]
    ):
        raise AccountingError("audit canonical target universe binding mismatch")

    flat_minimax, flat_minimax_meta = load_index_root(
        FLAT_MINIMAX_ROOT, all_rows, label="flat_minimax_h3"
    )
    flat_wan, flat_wan_meta = load_index_root(FLAT_WAN_ROOT, all_rows, label="flat_wan")
    wan_dot, wan_dot_meta = load_index_root(WAN_DOT_ROOT, all_rows, label="wan_dot_exact")
    for scene in sorted(all_rows):
        if flat_minimax[scene] != future_rows[scene]:
            raise AccountingError(f"flat Minimax index is not canonical future-only: {scene}")
        if flat_wan[scene] != flat_minimax[scene]:
            raise AccountingError(f"flat Wan and Minimax indexes differ: {scene}")
        if not wan_dot[scene] <= flat_wan[scene]:
            raise AccountingError(f"Wan dot exact index is not a flat-index subset: {scene}")
    if future_count != 32350 or sum_sets(wan_dot) != 2724:
        raise AccountingError("unexpected historical target-index cardinality")

    missing_anchor_rows: dict[str, set[int]] = {
        scene: all_rows[scene] - future_rows[scene] for scene in sorted(all_rows)
    }
    if sum_sets(missing_anchor_rows) != 5327:
        raise AccountingError("canonical-minus-future index gap must contain 5,327 rows")
    role_signatures: Counter[tuple[str, ...]] = Counter()
    p8_anchor: set[tuple[str, int]] = set()
    p24_anchor: set[tuple[str, int]] = set()
    future_role_gap = 0
    for scene, raw_ids in missing_anchor_rows.items():
        for raw_id in raw_ids:
            signature = tuple(sorted(roles[scene][raw_id]))
            role_signatures[signature] += 1
            if "p8:anchor" in signature:
                p8_anchor.add((scene, raw_id))
            if "p24:anchor" in signature:
                p24_anchor.add((scene, raw_id))
            if any(":h" in role for role in signature):
                future_role_gap += 1
    if future_role_gap:
        raise AccountingError("future-only index gap unexpectedly contains future roles")

    physical: dict[tuple[str, int], dict[str, Any]] = {}
    for unit in plan["units"]:
        key = _spec_key(unit)
        source_contracts = [
            {
                "source_id": source.get("source_id"),
                "kind": source.get("kind"),
                "priority": int(source.get("priority", 0)),
                "candidate_gate_status": source.get("candidate_gate_status"),
            }
            for source in unit["sources"]
        ]
        if key in physical:
            prior = physical[key]
            if (
                prior["spec_id"] != unit["spec_id"]
                or prior["source_contracts"] != source_contracts
            ):
                raise AccountingError(f"non-uniform plan spec across scenes: {key}")
        else:
            physical[key] = {
                "vfm": key[0],
                "layer": key[1],
                "timestep": int(unit["timestep"]),
                "spec_id": str(unit["spec_id"]),
                "source_contracts": source_contracts,
            }
    if len(physical) != 32:
        raise AccountingError(f"expected 32 physical specs, found {len(physical)}")
    remote_specs = {
        key for key, item in physical.items() if item["source_contracts"] and key[0] != "wan"
    }
    no_source_specs = {key for key, item in physical.items() if not item["source_contracts"]}
    wan_dot_specs = {
        key for key, item in physical.items() if item["source_contracts"] and key[0] == "wan"
    }
    if remote_specs != EXPECTED_REMOTE_SPECS:
        raise AccountingError(f"unexpected remote spec set: {sorted(remote_specs)}")
    if no_source_specs != EXPECTED_NO_SOURCE_SPECS:
        raise AccountingError(f"unexpected no-source spec set: {sorted(no_source_specs)}")
    if wan_dot_specs != {("wan", layer) for layer in WAN_DOT_LAYERS}:
        raise AccountingError(f"unexpected Wan dot spec set: {sorted(wan_dot_specs)}")

    by_spec = _aggregate_audit_units(audit["units"])
    universe = canonical_count * len(physical)
    certified_local = len(routes)
    pending_remote = sum(item["pending_rows"] for item in by_spec.values())
    definite_gap = sum(item["missing_rows"] for item in by_spec.values())
    duplicate_rows = int(audit["counts"]["duplicate_rows"])
    conflicting_rows = int(audit["counts"]["duplicate_rows_requiring_bitwise_check"])
    if (universe, certified_local, pending_remote, definite_gap) != (
        1_205_664,
        16_344,
        828_894,
        360_426,
    ):
        raise AccountingError("offline audit partition cardinality changed")
    if universe != certified_local + pending_remote + definite_gap:
        raise AccountingError("offline audit status partition does not sum to universe")
    if duplicate_rows or conflicting_rows or audit["counts"]["hard_errors"]:
        raise AccountingError("offline audit contains duplicate/conflicting/hard-error rows")

    route_identities = {
        (str(row["vfm"]), int(row["layer"]), str(row["scene_id"]), int(row["target_raw_id"]))
        for row in routes
    }
    expected_route_identities = {
        ("wan", layer, scene, raw_id)
        for layer in WAN_DOT_LAYERS
        for scene, raw_ids in wan_dot.items()
        for raw_id in raw_ids
    }
    if route_identities != expected_route_identities:
        raise AccountingError("certified local routes are not exactly the Wan dot index expansion")

    no_source_rows = canonical_count * len(no_source_specs)
    wan_dot_raw_id_absent = (canonical_count - sum_sets(wan_dot)) * len(wan_dot_specs)
    remote_anchor_gap = sum_sets(missing_anchor_rows) * len(remote_specs)
    lower_bound = no_source_rows + wan_dot_raw_id_absent + remote_anchor_gap
    difference = lower_bound - definite_gap
    remote_candidate_ceiling = future_count * len(remote_specs)
    declared_candidate_ceiling = certified_local + remote_candidate_ceiling
    if (
        no_source_rows,
        wan_dot_raw_id_absent,
        remote_anchor_gap,
        lower_bound,
        difference,
        remote_candidate_ceiling,
        declared_candidate_ceiling,
    ) != (150_708, 209_718, 117_194, 477_620, 117_194, 711_700, 728_044):
        raise AccountingError("best-case lower-bound accounting changed")
    old_lower_bound = int(
        summary["counts"]["must_reextract_row_layer_items_best_case_lower_bound"]
    )
    if old_lower_bound != lower_bound:
        raise AccountingError("frozen summary lower bound cannot be reproduced")

    # Exact overlap accounting for the excluded empty-prompt Wan flat cache.
    prompt_candidate_rows = future_count * len(WAN_DOT_LAYERS)
    prompt_candidate_with_compatible_alternative = certified_local
    prompt_only_definite_gaps = prompt_candidate_rows - certified_local
    wan_flat_anchor_gap = sum_sets(missing_anchor_rows) * len(WAN_DOT_LAYERS)
    if prompt_only_definite_gaps + wan_flat_anchor_gap != wan_dot_raw_id_absent:
        raise AccountingError("Wan missing-row semantic/raw-ID partition is inconsistent")

    model_breakdown: list[dict[str, Any]] = []
    for vfm in sorted({key[0] for key in remote_specs}):
        layers = sorted(layer for model, layer in remote_specs if model == vfm)
        model_breakdown.append(
            {
                "vfm": vfm,
                "layers": layers,
                "layer_count": len(layers),
                "rows_per_layer": sum_sets(missing_anchor_rows),
                "row_layer_items": sum_sets(missing_anchor_rows) * len(layers),
                "reason": "HISTORICAL_FUTURE_ONLY_INDEX_EXCLUDES_CANONICAL_P8_P24_ANCHORS",
                "remote_gate_status": "PENDING",
            }
        )
    layer_breakdown = [
        {
            "vfm": vfm,
            "layer": layer,
            "timestep": physical[(vfm, layer)]["timestep"],
            "spec_id": physical[(vfm, layer)]["spec_id"],
            "row_layer_items": sum_sets(missing_anchor_rows),
            "reason": "HISTORICAL_FUTURE_ONLY_INDEX_EXCLUDES_CANONICAL_P8_P24_ANCHORS",
            "remote_gate_status": "PENDING",
        }
        for vfm, layer in sorted(remote_specs)
    ]

    per_spec = []
    for key in sorted(physical):
        item = physical[key]
        counts = by_spec[key]
        per_spec.append(
            {
                "vfm": key[0],
                "layer": key[1],
                "timestep": item["timestep"],
                "spec_id": item["spec_id"],
                "declared_source_count_per_scene": len(item["source_contracts"]),
                **counts,
            }
        )

    report: dict[str, Any] = {
        "schema_version": 1,
        "report_type": "canonical_v2_target_row_layer_accounting",
        "report_id": "target_accounting_r6_20260831",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "PENDING_REMOTE_AUDIT",
        "scope": {
            "aoss_contacted": False,
            "feature_payload_written": False,
            "target_materialized": False,
            "old_artifacts_modified": False,
        },
        "input_receipts": {
            "plan": _input_receipt(plan_path, content_sha=plan["content_sha256"]),
            "plan_summary": _input_receipt(summary_path),
            "offline_audit": _input_receipt(audit_path, content_sha=audit["content_sha256"]),
            "routes": _input_receipt(routes_path),
            "missing_units": _input_receipt(missing_path),
            "pending_units": _input_receipt(pending_path),
            "canonical_source": _input_receipt(canonical_path),
            "builder": _input_receipt(Path(__file__).resolve()),
        },
        "actual_index_evidence": {
            "canonical_all_roles": {
                "scenes": len(all_rows),
                "windows": windows,
                "rows": canonical_count,
                "raw_id_set_sha256": set_payload_sha(all_rows),
            },
            "canonical_future_only": {
                "rows": future_count,
                "raw_id_set_sha256": set_payload_sha(future_rows),
            },
            "canonical_anchor_union": {
                "rows": sum_sets(anchor_rows),
                "raw_id_set_sha256": set_payload_sha(anchor_rows),
            },
            "flat_minimax_h3": flat_minimax_meta,
            "flat_wan": flat_wan_meta,
            "wan_dot_manifest_exact": wan_dot_meta,
            "equalities_proved": {
                "flat_minimax_equals_canonical_future_only_every_scene": True,
                "flat_wan_equals_flat_minimax_every_scene": True,
                "wan_dot_is_subset_of_flat_wan_every_scene": True,
            },
        },
        "current_exhaustive_status_partition": {
            "sets_are_pairwise_disjoint": True,
            "row_layer_universe": universe,
            "certified_local_rows": certified_local,
            "certified_remote_rows_current": 0,
            "pending_remote_candidate_rows": pending_remote,
            "remote_rows_reusable_after_full_audit": None,
            "remote_rows_reusable_after_full_audit_status": "PENDING_10_SHARD_REMOTE_AUDIT",
            "definite_gap_rows": definite_gap,
            "duplicate_owner_rows": duplicate_rows,
            "conflicting_owner_rows": conflicting_rows,
            "hard_error_rows": 0,
            "formula": "1205664 = 16344 certified_local + 828894 pending_remote + 360426 definite_gap",
        },
        "definite_gap_partition": {
            "sets_are_pairwise_disjoint": True,
            "no_declared_compatible_source_rows": no_source_rows,
            "no_declared_compatible_source_specs": [
                {"vfm": vfm, "layer": layer, "rows": canonical_count}
                for vfm, layer in sorted(no_source_specs)
            ],
            "wan_dot_compatible_source_raw_id_absent_rows": wan_dot_raw_id_absent,
            "formula": "360426 = 150708 no-source + 209718 absent-from-Wan-dot-index",
        },
        "compatibility_accounting": {
            "note": "Candidate-rejection sets below are not all mutually exclusive with the primary status partition; explicit overlaps are reported.",
            "prompt_semantic_incompatible_candidate_rows": prompt_candidate_rows,
            "prompt_semantic_reason": "legacy flat Wan prompt='' but canonical prompt='.'",
            "prompt_candidate_rows_with_compatible_wan_dot_owner": prompt_candidate_with_compatible_alternative,
            "prompt_incompatible_rows_without_alternative_owner": prompt_only_definite_gaps,
            "raw_id_absent_from_declared_compatible_wan_dot_source_rows": wan_dot_raw_id_absent,
            "anchor_raw_ids_absent_from_actual_future_only_flat_index": sum_sets(missing_anchor_rows),
            "anchor_row_layer_items_across_22_remote_specs": remote_anchor_gap,
            "anchor_row_layer_items_across_6_excluded_wan_flat_specs": wan_flat_anchor_gap,
            "anchor_candidate_row_layer_items_total": remote_anchor_gap + wan_flat_anchor_gap,
            "anchor_gap_contains_future_role_rows": future_role_gap,
            "anchor_role_membership": {
                "p8_anchor_rows": len(p8_anchor),
                "p24_anchor_rows": len(p24_anchor),
                "p8_and_p24_anchor_overlap_rows": len(p8_anchor & p24_anchor),
                "union_rows": sum_sets(missing_anchor_rows),
                "role_signatures": [
                    {"roles": list(signature), "rows": count}
                    for signature, count in sorted(
                        role_signatures.items(), key=lambda item: (-item[1], item[0])
                    )
                ],
            },
            "certified_shape_compatible_rows": certified_local,
            "known_shape_incompatible_rows": 0,
            "remote_rows_shape_unchecked": pending_remote,
            "known_other_semantic_incompatible_rows": 0,
            "remote_prompt_model_preprocess_shape_gate_status": "PENDING",
        },
        "best_case_reextract_lower_bound": {
            "value": lower_bound,
            "classification": "CONDITIONAL_PRE_GATE_FORECAST",
            "authoritative_as_current_definite_gap": False,
            "condition": "all 22 remote specs pass identity/payload gates and their canonical intersection equals the proven 32350-row historical future-only index",
            "sets_are_pairwise_disjoint_under_condition": True,
            "components": {
                "no_declared_compatible_source_rows": no_source_rows,
                "wan_dot_compatible_source_raw_id_absent_rows": wan_dot_raw_id_absent,
                "remote_future_only_anchor_gap_rows": remote_anchor_gap,
            },
            "formula_components": "477620 = 150708 + 209718 + 117194",
            "formula_candidate_ceiling": "477620 = 1205664 - (16344 certified_local + 711700 remote_candidate_ceiling)",
            "remote_candidate_ceiling_rows": remote_candidate_ceiling,
            "total_declared_candidate_ceiling_including_local": declared_candidate_ceiling,
            "final_exact_reextract_rows": None,
            "final_exact_reextract_status": "PENDING_FULL_REMOTE_AUDIT",
        },
        "difference_477620_minus_360426": {
            "value": difference,
            "reason": "The offline audit conservatively keeps every row of 22 remote specs PENDING. The forecast subtracts the exact 5327-row p8/p24 anchor complement of the historical future-only index once per remote spec.",
            "raw_id_rows_per_layer": sum_sets(missing_anchor_rows),
            "remote_layer_count": len(remote_specs),
            "formula": "117194 = 5327 * 22",
            "by_model": model_breakdown,
            "by_layer": layer_breakdown,
        },
        "legacy_field_disposition": {
            "field": "counts.must_reextract_row_layer_items_best_case_lower_bound",
            "old_value": old_lower_bound,
            "stale": False,
            "reproduced_exactly": True,
            "correct_scope": "conditional pre-gate best-case forecast",
            "incorrect_uses": [
                "current definite gap count",
                "completed remote audit result",
                "authorization to convert PENDING rows into extraction work",
            ],
        },
        "per_physical_spec": per_spec,
        "execution_rule": "Keep all 828894 remote rows PENDING until all ten audit shards merge; add only definite FAIL/unreadable/unrouted rows, never timeouts, to later extraction worklists.",
    }
    report["content_sha256"] = canonical_sha(report)
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    current = report["current_exhaustive_status_partition"]
    lower = report["best_case_reextract_lower_bound"]
    difference = report["difference_477620_minus_360426"]
    lines = [
        "# canonical-v2 target row-layer accounting r6",
        "",
        f"Status: **{report['status']}**. This report is local/read-only and did not contact AOSS or materialize target features.",
        "",
        "## Authoritative current partition",
        "",
        "| Set | Row-layer items | Current meaning |",
        "|---|---:|---|",
        f"| Universe | {current['row_layer_universe']:,} | 32 physical layers × 37,677 canonical raw-ID rows |",
        f"| Certified local | {current['certified_local_rows']:,} | Six Wan layers, payload/index SHA verified |",
        f"| Certified remote | {current['certified_remote_rows_current']:,} | No remote shard has been merged yet |",
        f"| Pending remote | {current['pending_remote_candidate_rows']:,} | 22 layers; not FAIL and not schedulable as gaps |",
        f"| Definite gap | {current['definite_gap_rows']:,} | Safe first-stage extraction set |",
        f"| Duplicate/conflicting owner | {current['duplicate_owner_rows'] + current['conflicting_owner_rows']:,} | None |",
        "",
        f"The three primary sets are pairwise disjoint: `{current['formula']}`.",
        "",
        "`remote_rows_reusable_after_full_audit` is deliberately `null`: all ten remote shards must merge before that number exists.",
        "",
        "## Why 360,426 is definite now",
        "",
        "`360,426 = 150,708 + 209,718`:",
        "",
        "- 150,708 = four layers with no declared compatible source: Aether l1, CogVideoX l20, Qwen2.5-VL l-1, and Wan l20.",
        "- 209,718 = six Wan dot-prompt layers × (37,677 canonical rows − 2,724 certified local rows).",
        "",
        "## Exact 117,194 difference",
        "",
        f"`{difference['formula']}`. The actual Minimax flat indexes and actual Wan flat indexes were SHA/sidecar checked for all 681 scenes. Both equal the canonical future-only set exactly (32,350 rows). The canonical all-role set has 37,677 rows; its 5,327-row complement is entirely p8/p24 anchors and contains zero future-role rows.",
        "",
        "| Model | Layers | Rows/layer | Row-layer items |",
        "|---|---|---:|---:|",
    ]
    for item in difference["by_model"]:
        layers = ",".join(str(layer) for layer in item["layers"])
        lines.append(
            f"| {item['vfm']} | {layers} | {item['rows_per_layer']:,} | {item['row_layer_items']:,} |"
        )
    lines.extend(
        [
            "",
            "These 117,194 rows are still inside the offline audit's PENDING set. They are not an additional already-rendered first-stage gap worklist.",
            "",
            "## 477,620 disposition",
            "",
            f"The old value is reproducible and therefore is **not stale**, but its scope is conditional: **{lower['classification']}**. It is not the current definite-gap count and not a completed remote-audit result.",
            "",
            f"- Component formula: `{lower['formula_components']}`.",
            f"- Candidate-ceiling formula: `{lower['formula_candidate_ceiling']}`.",
            f"- Condition: {lower['condition']}.",
            "- The exact final re-extraction count remains unknown until all ten remote audit shards merge.",
            "",
            "## Compatibility sets",
            "",
            "Candidate rejection counts are not all mutually exclusive with the primary status partition:",
            "",
            f"- Prompt-semantic mismatch: {report['compatibility_accounting']['prompt_semantic_incompatible_candidate_rows']:,} Wan flat candidate rows. Of these, {report['compatibility_accounting']['prompt_candidate_rows_with_compatible_wan_dot_owner']:,} have a certified dot-prompt alternative and {report['compatibility_accounting']['prompt_incompatible_rows_without_alternative_owner']:,} do not.",
            f"- Raw-ID/anchor mismatch: 5,327 unique canonical rows are absent from the actual future-only flat index. Expansion is {report['compatibility_accounting']['anchor_row_layer_items_across_22_remote_specs']:,} over the 22 pending remote layers and {report['compatibility_accounting']['anchor_row_layer_items_across_6_excluded_wan_flat_specs']:,} over six excluded Wan-flat layers.",
            f"- Shape mismatch currently proved: {report['compatibility_accounting']['known_shape_incompatible_rows']:,}. Shape remains unchecked for {report['compatibility_accounting']['remote_rows_shape_unchecked']:,} remote-PENDING rows.",
            f"- Duplicate/conflicting owner rows: {current['duplicate_owner_rows'] + current['conflicting_owner_rows']:,}.",
            "",
            "## Execution consequence",
            "",
            report["execution_rule"],
            "",
            f"JSON content SHA-256: `{report['content_sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_new(path: Path, payload: bytes) -> None:
    """Atomically create an immutable artifact; never replace an existing path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise AccountingError(f"refusing to overwrite immutable output: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()
    report = build_report(
        plan_path=args.plan.resolve(),
        summary_path=args.summary.resolve(),
        audit_dir=args.audit_dir.resolve(),
    )
    json_payload = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    markdown_payload = render_markdown(report).encode("utf-8")
    write_new(args.output_json.resolve(), json_payload)
    write_new(args.output_markdown.resolve(), markdown_payload)
    print(
        json.dumps(
            {
                "status": report["status"],
                "json": str(args.output_json.resolve()),
                "markdown": str(args.output_markdown.resolve()),
                "content_sha256": report["content_sha256"],
                "definite_gap_rows": report["current_exhaustive_status_partition"]["definite_gap_rows"],
                "best_case_reextract_lower_bound": report["best_case_reextract_lower_bound"]["value"],
                "difference": report["difference_477620_minus_360426"]["value"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
