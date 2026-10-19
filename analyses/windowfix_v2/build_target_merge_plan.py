#!/usr/bin/env python3
"""Build the frozen canonical-v2 target row merge plan.

This builder performs no AOSS reads and never writes a feature cache.  It
expands the 32 physical target identities in the frozen 1096-experiment
inventory over all 681 canonical ScanNet++ scenes.  Candidate legacy sources
are declarations only; :mod:`audit_target_row_routes` must freshly gate every
declared index, sidecar, safetensors header and identity before a row can be
routed.

The current canonical semantics use prompt ``'.'`` for Wan.  Consequently the
large historical flat Wan target cache (prompt ``''``) is deliberately absent
from ``sources``.  Only the six-layer, dot-prompt manifest-exact cache is
declared for Wan, and it remains payload-SHA gated before materialization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyses.windowfix_v2.audit_target_row_routes import (
    CANONICAL_PROTOCOL,
    CONTRACT,
    MIXED_TARGET_NAMESPACE,
    PLAN_TYPE,
    canonical_json_sha,
    canonical_target_unions,
    sha256_file,
)


INVENTORY = PROJECT_ROOT / (
    "artifacts/windowfix_v2/feature_inventory/"
    "canonical_v2_feature_specs.windowfix_v2_b34c099e010d.json"
)
PROTOCOL = PROJECT_ROOT / "protocols/inscene_streaming_canonical_v2.json"
CANONICAL_SOURCE = (
    PROJECT_ROOT / "manifests/c_retrain_20260722/inscene_c_h124_prefix24.jsonl"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts/windowfix_v2/target_row_merge_initial/target_merge_plan.json"
)
DEFAULT_DESTINATION = (
    Path("/mnt/umm/users/yinbaiqiao/InsScene-15K") / MIXED_TARGET_NAMESPACE
)

FLAT_AOSS_ROOT = "aoss://baiqiao/InsScene-15K/FEAT_TARGET"
DINO_PREFIX_AOSS_ROOT = "aoss://baiqiao/InsScene-15K/FEAT_TARGET_BY_PREFIX"
FLAT_SOURCE_VFMS = frozenset(
    {"f3r", "lingbot", "minimax_h3", "videomae", "vjepa2"}
)
WAN_REUSABLE_LAYERS = frozenset({0, 5, 10, 15, 25, 29})
WAN_VERSIONED_ROOT = Path(
    "/mnt/umm/users/yinbaiqiao/InsScene-15K/FEATURES_VERSIONED/"
    "wan_p8_action_dynamics_canonical_manifest_exact_20260821_200604_2075f3e/"
    "target/wan/scannetpp"
)
WAN_SNAPSHOT_MODEL = (
    "/mnt/umm/users/yinbaiqiao/.cache/huggingface/hub/"
    "models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/"
    "0fad780a534b6463e45facd96134c9f345acfa5b"
)

# Target-isolated rows contain one spatial grid per raw frame.  These tails
# are frozen from the extractor reshape contracts cited in ``evidence`` below.
TAIL_SHAPES: dict[str, list[int]] = {
    "aether": [30, 45, 3072],
    "cogvideox": [30, 45, 3072],
    "dino": [30, 52, 1024],
    "f3r": [18, 32, 1024],
    "lingbot": [30, 52, 2048],
    "minimax_h3": [15, 26, 5376],
    "qwen2_5_vl": [12, 12, 3584],
    "videomae": [14, 14, 1024],
    "vjepa2": [16, 16, 1024],
    "wan": [30, 52, 1536],
}

PROMPT_CONTRACTS: dict[str, dict[str, Any]] = {
    "aether": {"mode": "exact", "value": ""},
    "cogvideox": {"mode": "exact", "value": ""},
    "dino": {"mode": "not_applicable", "value": None},
    "f3r": {"mode": "not_applicable", "value": None},
    "lingbot": {"mode": "exact", "value": "."},
    "minimax_h3": {"mode": "exact", "value": "."},
    "qwen2_5_vl": {
        "mode": "exact",
        "value": "Describe the spatial layout of this scene.",
    },
    "videomae": {"mode": "not_applicable", "value": None},
    "vjepa2": {"mode": "not_applicable", "value": None},
    "wan": {"mode": "exact", "value": "."},
}

PREPROCESS: dict[str, dict[str, Any]] = {
    "aether": {"resize": [480, 720], "num_frames": 81},
    "cogvideox": {"resize": [480, 720], "num_frames": 97},
    "dino": {"resize": [420, 728], "num_frames": 64},
    "f3r": {"resize": [288, 512], "num_frames": 64},
    "lingbot": {"resize": [480, 832], "num_frames": 49},
    "minimax_h3": {"resize": [480, 832], "num_frames": 73},
    "qwen2_5_vl": {"resize": [336, 336], "num_frames": 8},
    "videomae": {"resize": [224, 224], "num_frames": 16},
    "vjepa2": {"resize": [256, 256], "num_frames": 64},
    "wan": {"resize": [480, 832], "num_frames": 81},
}

SHAPE_EVIDENCE = {
    "aether": "features/aether/aether_feature.py:2-4,126",
    "cogvideox": "features/cogvideox/cogvideox_feature.py:2-4,114",
    "dino": "features/dino/extract_features.py:6,159-170",
    "f3r": "features/f3r/extract_features.py:5,48-50,193-202",
    "lingbot": "features/lingbot/lingbot_feature.py:111-123",
    "minimax_h3": "features/minimax_h3/minimax_h3_feature.py:89-132",
    "qwen2_5_vl": "features/run_inscene15k_mllm.py:430-456; vidfm3d/utils/feature_layers.py:65-70",
    "videomae": "features/run_inscene15k.py:374-425",
    "vjepa2": "features/vjepa2/vjepa2_feature.py:1-10,83-98",
    "wan": "features/run_inscene15k.py:2460-2490; manifest-exact sidecars",
}


class PlanBuildError(ValueError):
    pass


def _json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PlanBuildError(f"input must be a regular non-symlink file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PlanBuildError(f"input is not a JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _physical_specs(inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_identity: dict[tuple[Any, ...], dict[str, Any]] = {}
    for raw in inventory.get("specs", []):
        if raw.get("requires_target") != "true":
            continue
        key = (
            str(raw["vfm"]),
            str(raw["model_id"]),
            int(raw["feature_layer"]),
            int(raw["timestep"]),
        )
        prior = by_identity.get(key)
        if prior is not None and prior["feature_filename"] != raw["feature_filename"]:
            raise PlanBuildError(f"physical target filename conflict: {key}")
        by_identity[key] = dict(raw)
    result = [by_identity[key] for key in sorted(by_identity)]
    if len(result) != 32:
        raise PlanBuildError(f"expected 32 physical target specs, found {len(result)}")
    return result


def _legacy_source(spec: Mapping[str, Any], scene: str) -> list[dict[str, Any]]:
    vfm = str(spec["vfm"])
    layer = int(spec["feature_layer"])
    filename = str(spec["feature_filename"])
    if vfm == "wan" and layer in WAN_REUSABLE_LAYERS:
        directory = WAN_VERSIONED_ROOT / scene
        return [
            {
                "source_id": "wan_dot_manifest_exact_local",
                "kind": "legacy",
                "candidate_gate_status": "PENDING_LOCAL_PAYLOAD_SHA_GATE",
                "feature_path": str((directory / filename).resolve()),
                "index_path": str((directory / "target_indices.npy").resolve()),
                "expected_source_filename": filename,
            }
        ]
    if vfm in FLAT_SOURCE_VFMS:
        directory = f"{FLAT_AOSS_ROOT}/{vfm}/scannetpp/{scene}"
        return [
            {
                "source_id": "legacy_flat_aoss",
                "kind": "legacy",
                "candidate_gate_status": "PENDING_AOSS_PAYLOAD_GATE",
                "feature_path": f"{directory}/{filename}",
                "index_path": f"{directory}/target_indices.npy",
                "expected_source_filename": filename,
            }
        ]
    if vfm == "dino":
        # No single historical DINO prefix cache is asserted to own the full
        # canonical union.  Gate all four roots, prefer the longest prefix,
        # and require bitwise equality for every overlapping raw ID.
        result = []
        for priority, prefix in enumerate((24, 16, 12, 8)):
            directory = (
                f"{DINO_PREFIX_AOSS_ROOT}/dino_prefix_{prefix}_m0p35_r0p5/"
                f"dino/scannetpp/{scene}"
            )
            result.append(
                {
                    "source_id": f"dino_prefix{prefix}_aoss",
                    "kind": "legacy",
                    "priority": priority,
                    "candidate_gate_status": "PENDING_AOSS_PAYLOAD_GATE",
                    "feature_path": f"{directory}/{filename}",
                    "index_path": f"{directory}/target_indices.npy",
                    "expected_source_filename": filename,
                }
            )
        return result
    return []


def _local_index_evidence(
    canonical_targets: Mapping[tuple[str, str], tuple[int, ...]],
) -> dict[str, Any]:
    """Cheap raw-ID evidence only; feature payloads remain separately gated."""
    evidence: dict[str, Any] = {}
    roots = {
        "flat_union_proxy_minimax_h3": Path(
            "/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_TARGET/minimax_h3/scannetpp"
        ),
        "wan_dot_manifest_exact": WAN_VERSIONED_ROOT,
    }
    for label, root in roots.items():
        rows = intersection = extra = duplicates = missing = 0
        for (source, scene), expected in canonical_targets.items():
            if source != "scannetpp":
                continue
            path = root / scene / "target_indices.npy"
            if path.is_symlink() or not path.is_file():
                missing += 1
                continue
            values = np.load(path, allow_pickle=False)
            if values.ndim != 1 or values.dtype.kind not in "iu":
                raise PlanBuildError(f"invalid local target index: {path}")
            native = [int(value) for value in values]
            if native != sorted(native):
                raise PlanBuildError(f"local target index is not sorted: {path}")
            rows += len(native)
            duplicates += len(native) - len(set(native))
            expected_set = set(expected)
            intersection += len(set(native) & expected_set)
            extra += len(set(native) - expected_set)
        evidence[label] = {
            "root": str(root),
            "scenes_expected": len(canonical_targets),
            "scenes_missing": missing,
            "index_rows": rows,
            "canonical_intersection_rows": intersection,
            "extra_rows": extra,
            "duplicate_rows": duplicates,
            "scope": (
                "raw_id_index_evidence_only; does not certify that every AOSS VFM/layer "
                "has the same index or a valid feature payload"
            ),
        }
    return evidence


def build_plan(
    *,
    inventory_path: Path,
    protocol_path: Path,
    canonical_source_path: Path,
    destination_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    inventory_path = inventory_path.resolve()
    protocol_path = protocol_path.resolve()
    canonical_source_path = canonical_source_path.resolve()
    inventory = _json(inventory_path)
    protocol = _json(protocol_path)
    if protocol.get("canonical_protocol") != CANONICAL_PROTOCOL:
        raise PlanBuildError("current protocol file is not canonical obs28/h124")
    if destination_root.name != MIXED_TARGET_NAMESPACE or "SEEDED_V1" in str(destination_root):
        raise PlanBuildError("destination must be the mixed target namespace, never SEEDED_V1")

    canonical_sha = sha256_file(canonical_source_path)
    canonical_targets, canonical_meta = canonical_target_unions(
        canonical_source_path, canonical_sha
    )
    physical = _physical_specs(inventory)
    if set(TAIL_SHAPES) != {str(spec["vfm"]) for spec in physical}:
        raise PlanBuildError("shape registry and physical target VFM set differ")

    units: list[dict[str, Any]] = []
    for spec in physical:
        vfm = str(spec["vfm"])
        layer = int(spec["feature_layer"])
        model_id = str(spec["model_id"])
        accepted_model_ids = [model_id]
        if vfm == "wan":
            accepted_model_ids.append(WAN_SNAPSHOT_MODEL)
        modes = ["target_isolated"]
        if vfm == "wan" and layer in WAN_REUSABLE_LAYERS:
            modes.append("manifest_exact_target")
        physical_id = f"target-{vfm}-t{int(spec['timestep'])}-l{layer}"
        for source, scene in sorted(canonical_targets):
            if source != "scannetpp":
                raise PlanBuildError(f"unexpected canonical source: {source}")
            units.append(
                {
                    "spec_id": physical_id,
                    "inventory_spec_id": str(spec["spec_id"]),
                    "vfm": vfm,
                    "model_id": model_id,
                    "accepted_model_ids": accepted_model_ids,
                    "layer": layer,
                    "timestep": int(spec["timestep"]),
                    "tensor_key": "feat",
                    "source": source,
                    "scene_id": scene,
                    "expected_tail_shape": TAIL_SHAPES[vfm],
                    "expected_dtype": "torch.float16",
                    "accepted_modes": modes,
                    "prompt_contract": PROMPT_CONTRACTS[vfm],
                    "preprocess_identity": PREPROCESS[vfm],
                    "destination_filename": str(spec["feature_filename"]),
                    "duplicate_owner_policy": (
                        "require_bitwise_equal_then_priority" if vfm == "dino" else "forbid"
                    ),
                    "sources": _legacy_source(spec, scene),
                }
            )

    expected_units = len(physical) * len(canonical_targets)
    if len(canonical_targets) != 681 or len(units) != expected_units or len(units) != 21_792:
        raise PlanBuildError(
            f"unexpected target plan cardinality: scenes={len(canonical_targets)} units={len(units)}"
        )
    source_units = Counter(
        source["candidate_gate_status"]
        for unit in units
        for source in unit["sources"]
    )
    index_evidence = _local_index_evidence(canonical_targets)
    flat_rows = int(
        index_evidence["flat_union_proxy_minimax_h3"]["canonical_intersection_rows"]
    )
    wan_rows = int(
        index_evidence["wan_dot_manifest_exact"]["canonical_intersection_rows"]
    )
    flat_layers = sum(1 for spec in physical if spec["vfm"] in FLAT_SOURCE_VFMS)
    no_source_specs = [
        spec
        for spec in physical
        if spec["vfm"] not in FLAT_SOURCE_VFMS
        and spec["vfm"] != "dino"
        and not (
            spec["vfm"] == "wan"
            and int(spec["feature_layer"]) in WAN_REUSABLE_LAYERS
        )
    ]
    # The DINO four-prefix remote union has no fresh local gate.  32,350 is an
    # explicit upper bound inherited from the historical flat target-union
    # index, not a claim that any DINO row is presently reusable.
    dino_remote_candidate_upper_bound = flat_rows
    raw_candidate_max = (
        flat_rows * flat_layers
        + wan_rows * len(WAN_REUSABLE_LAYERS)
        + dino_remote_candidate_upper_bound
    )
    hard_reextract_min = canonical_meta["target_rows"] * len(physical) - raw_candidate_max
    plan: dict[str, Any] = {
        "schema_version": 1,
        "plan_type": PLAN_TYPE,
        "contract": CONTRACT,
        "canonical_protocol": CANONICAL_PROTOCOL,
        "protocol_file": str(protocol_path),
        "protocol_file_sha256": sha256_file(protocol_path),
        "canonical_source": {
            "path": str(canonical_source_path),
            "sha256": canonical_sha,
        },
        "destination_root": str(destination_root.resolve()),
        "build_inputs": {
            "builder_source": str(Path(__file__).resolve()),
            "builder_source_sha256": sha256_file(Path(__file__).resolve()),
            "inventory_path": str(inventory_path),
            "inventory_file_sha256": sha256_file(inventory_path),
            "inventory_sha256": inventory.get("inventory_sha256"),
        },
        "shape_contract_evidence": SHAPE_EVIDENCE,
        "excluded_legacy_candidates": [
            {
                "vfm": "wan",
                "layers": sorted(WAN_REUSABLE_LAYERS),
                "root": f"{FLAT_AOSS_ROOT}/wan/scannetpp",
                "reason": "PROMPT_SEMANTIC_MISMATCH: legacy prompt='' but canonical prompt='.'",
                "raw_id_candidate_rows_per_layer_before_semantic_gate": flat_rows,
            },
        ],
        "units": units,
    }
    plan["content_sha256"] = canonical_json_sha(plan)
    summary = {
        "schema_version": 1,
        "status": "PENDING_AOSS_AND_PAYLOAD_GATES",
        "plan_contract": CONTRACT,
        "plan_content_sha256": plan["content_sha256"],
        "protocol_file_sha256": plan["protocol_file_sha256"],
        "canonical": canonical_meta,
        "counts": {
            "physical_target_specs": len(physical),
            "scenes": len(canonical_targets),
            "plan_units": len(units),
            "canonical_target_rows_per_full_layer_universe": canonical_meta["target_rows"],
            "canonical_target_row_layer_items": canonical_meta["target_rows"] * len(physical),
            "units_by_candidate_gate": dict(sorted(source_units.items())),
            "declared_flat_aoss_layers": flat_layers,
            "declared_wan_dot_layers": len(WAN_REUSABLE_LAYERS),
            "declared_dino_prefix_sources": 4,
            "no_declared_legacy_source_layers": len(no_source_specs),
            "no_declared_legacy_source_physical_ids": [
                f"{spec['vfm']}:t{int(spec['timestep'])}:l{int(spec['feature_layer'])}"
                for spec in no_source_specs
            ],
            "raw_id_candidate_row_layer_items_flat_aoss_max_before_gate": (
                flat_rows * flat_layers
            ),
            "raw_id_candidate_row_layer_items_wan_dot_before_payload_gate": (
                wan_rows * len(WAN_REUSABLE_LAYERS)
            ),
            "raw_id_candidate_rows_dino_prefix_union_upper_bound_before_gate": (
                dino_remote_candidate_upper_bound
            ),
            "raw_id_candidate_row_layer_items_total_max_before_gate": raw_candidate_max,
            "must_reextract_row_layer_items_best_case_lower_bound": hard_reextract_min,
            "wan_flat_empty_prompt_raw_only_blocked_row_layer_items": (
                flat_rows * len(WAN_REUSABLE_LAYERS)
            ),
        },
        "local_index_evidence": index_evidence,
        "interpretation": {
            "flat_aoss_candidate_count_is_upper_bound": True,
            "wan_dot_candidate_count_is_not_payload_certified": True,
            "materialization_allowed": False,
            "best_case_lower_bound_assumes_every_declared_candidate_passes": True,
            "required_next_gate": (
                "run audit_target_row_routes.py where AOSS is reachable; then fill emitted "
                "gap worklists and rerun until COMPLETE before materialization"
            ),
        },
    }
    return plan, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=INVENTORY)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--canonical-source", type=Path, default=CANONICAL_SOURCE)
    parser.add_argument("--destination-root", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plan, summary = build_plan(
        inventory_path=args.inventory,
        protocol_path=args.protocol,
        canonical_source_path=args.canonical_source,
        destination_root=args.destination_root,
    )
    output = args.output.resolve()
    _atomic_json(output, plan)
    summary["plan_path"] = str(output)
    summary["plan_file_sha256"] = sha256_file(output)
    audit_output = output.parent / "route_audit_remote_gate"
    summary["pending_audit"] = {
        "status": "PENDING_AOSS_AND_PAYLOAD_GATES",
        "output_directory": str(audit_output),
        "expected_outputs": [
            str(audit_output / "target_row_route_audit.json"),
            str(audit_output / "target_row_routes.jsonl"),
            str(audit_output / "missing_target_units.jsonl"),
            str(audit_output / "gap_worklists"),
        ],
        "command": (
            f"{sys.executable} {PROJECT_ROOT / 'analyses/windowfix_v2/audit_target_row_routes.py'} "
            f"--plan {output} --output-dir {audit_output} --remote-policy gate "
            "--source-timeout-seconds 30 --verify-source-sha"
        ),
        "note": (
            "Run where AOSS metadata and byte-range reads are available.  "
            "--verify-source-sha fully verifies local payloads; aggregate remote payloads "
            "remain header/size/index gated and selected remote rows are range-hashed during "
            "materialization, without downloading whole remote tensors."
        ),
    }
    summary_path = output.with_name("target_merge_plan.summary.json")
    _atomic_json(summary_path, summary)
    print(json.dumps({"plan": str(output), "summary": str(summary_path), **summary["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
