#!/usr/bin/env python3
"""Build the conservative canonical-v2 feature generation ledger.

This is a planning tool only.  It never copies features, calls SCO, or starts
an extractor.  The plan is deliberately fail-closed: the inventory and compact
label receipt must be bound to the exact current protocol file, and unproven
legacy tensors cannot become canonical simply because their raw frame IDs
match.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


STRICT_BACKEND_STATUS = {
    "wan": ("READY", "sample-local seeded streaming and target paths"),
    "cogvideox": ("READY", "sample-local seeded streaming and target paths"),
    "minimax_h3": ("READY", "fixed-seed deterministic streaming and target paths"),
    "dino": (
        "READY",
        "deterministic forward with explicit deterministic_forward_no_noise_v1 provenance",
    ),
    "vjepa2": (
        "READY",
        "deterministic forward with explicit deterministic_forward_no_noise_v1 provenance",
    ),
    "videomae": (
        "READY",
        "deterministic forward with explicit deterministic_forward_no_noise_v1 provenance",
    ),
    "aether": (
        "BLOCKED_GPU_CANARY",
        "sample-local generator/provenance implemented; cross-process/order GPU equality canary still required",
    ),
    "f3r": (
        "BLOCKED_GPU_CANARY",
        "random image-index generator is now sample-local; cross-rank/order GPU equality canary still required",
    ),
    "lingbot": (
        "BLOCKED_GPU_CANARY",
        "VAE and diffusion draws now use one sample-local generator; GPU byte/tensor equality canary still required",
    ),
    "qwen2_5_vl": (
        "BLOCKED_GPU_CANARY",
        "canonical obs28 MLLM extractor and strict no-noise provenance are implemented; cross-process/order GPU equality canary still required",
    ),
}

DEFAULT8_RAW_ID_UPPER_BOUND = {
    8: (786, 1935),
    12: (883, 1838),
    16: (1111, 1610),
    24: (1694, 1027),
}
DEFAULT8_MODELS = (
    "aether",
    "cogvideox",
    "dino",
    "f3r",
    "qwen2_5_vl",
    "videomae",
    "vjepa2",
    "wan",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"required input must be a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def build(
    *,
    inventory_path: Path,
    protocol_path: Path,
    label_receipt_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    inventory = read_json(inventory_path)
    protocol = read_json(protocol_path)
    receipt = read_json(label_receipt_path)
    protocol_sha = sha256_file(protocol_path)
    if inventory.get("source", {}).get("protocol_file_sha256") != protocol_sha:
        raise ValueError("feature inventory is stale relative to the current protocol file")
    if receipt.get("status") != "pass" or receipt.get("problem_count") != 0:
        raise ValueError("compact canonical label receipt is not a zero-problem PASS")
    if receipt.get("protocol_file_sha256") != protocol_sha:
        raise ValueError("compact label receipt is stale relative to the current protocol file")
    source = Path(str(receipt.get("source_manifest", "")))
    if sha256_file(source) != receipt.get("source_manifest_sha256"):
        raise ValueError("compact label receipt source manifest changed")
    if protocol.get("acceptance", {}).get("legacy_features_valid") is not False:
        raise ValueError("this conservative plan requires legacy_features_valid=false")
    if protocol.get("acceptance", {}).get("audit_gate") != "fail_closed":
        raise ValueError("this plan requires a fail-closed audit gate")
    if int(receipt.get("counts", {}).get("canonical_windows", -1)) != 2721:
        raise ValueError("unexpected canonical window count")

    specs = inventory.get("specs")
    if not isinstance(specs, list) or len(specs) != 160:
        raise ValueError("expected exactly 160 deduplicated feature specs")

    spec_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, int], dict[str, Any]] = {}
    target_layer_keys: set[tuple[str, str, int, int]] = set()
    for source_row in specs:
        row = dict(source_row)
        vfm = str(row["vfm"])
        if vfm not in STRICT_BACKEND_STATUS:
            raise ValueError(f"no strict backend decision for {vfm}")
        backend_status, blocker = STRICT_BACKEND_STATUS[vfm]
        prefix = int(row["prefix"])
        layer = int(row["feature_layer"])
        timestep = int(row["timestep"])
        model_id = str(row["model_id"])
        requires_target = _bool(row.get("requires_target"))
        row.update(
            {
                "action": "REEXTRACT",
                "canonical_requirement": "REEXTRACT_REQUIRED",
                "reuse_decision": "REJECT_LEGACY_UNDER_SEEDED_V1",
                "reuse_reason": (
                    "protocol rejects legacy payloads without deterministic-noise provenance; "
                    "raw-ID equality alone is insufficient"
                ),
                "backend_status": backend_status,
                "backend_blocker": blocker,
                "required_streaming_files": 2721,
                "required_target_scene_files": 681 if requires_target else 0,
            }
        )
        spec_rows.append(row)
        group_key = (vfm, model_id, timestep)
        group = grouped.setdefault(
            group_key,
            {
                "vfm": vfm,
                "model_id": model_id,
                "timestep": timestep,
                "layers": set(),
                "target_layers": set(),
                "spec_ids": [],
                "backend_status": backend_status,
                "backend_blocker": blocker,
            },
        )
        group["layers"].add(layer)
        group["spec_ids"].append(str(row["spec_id"]))
        if requires_target:
            group["target_layers"].add(layer)
            target_layer_keys.add((vfm, model_id, timestep, layer))

    group_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        group = grouped[key]
        layers = sorted(group.pop("layers"))
        target_layers = sorted(group.pop("target_layers"))
        spec_ids = sorted(group.pop("spec_ids"))
        group_rows.append(
            {
                **group,
                "layers": layers,
                "target_layers": target_layers,
                "prefixes": [8, 12, 16, 24],
                "streaming_files": len(layers) * 4 * 2721,
                "target_scene_files": len(target_layers) * 681,
                "spec_count": len(spec_ids),
                "spec_ids_sha256": canonical_json_sha(spec_ids),
                "launchable": group["backend_status"] == "READY",
            }
        )

    by_vfm: dict[str, dict[str, Any]] = {}
    for vfm in sorted(STRICT_BACKEND_STATUS):
        rows = [row for row in spec_rows if row["vfm"] == vfm]
        target_keys = [key for key in target_layer_keys if key[0] == vfm]
        status, blocker = STRICT_BACKEND_STATUS[vfm]
        by_vfm[vfm] = {
            "backend_status": status,
            "blocker": blocker,
            "prefix_level_specs": len(rows),
            "streaming_files": sum(int(row["required_streaming_files"]) for row in rows),
            "target_layer_groups": len(target_keys),
            "target_scene_files": len(target_keys) * 681,
        }

    status_counts = Counter(row["backend_status"] for row in spec_rows)
    plan = {
        "schema_version": 1,
        "plan_type": "canonical_v2_feature_reextract",
        "status": "READY_FOR_PARTIAL_SCO_RENDER",
        "submission_performed": False,
        "probing_training_in_scope": False,
        "protocol": protocol["canonical_protocol"],
        "protocol_file": str(protocol_path),
        "protocol_file_sha256": protocol_sha,
        "inventory": str(inventory_path),
        "inventory_file_sha256": sha256_file(inventory_path),
        "inventory_version": inventory["inventory_version"],
        "label_receipt": str(label_receipt_path),
        "label_receipt_sha256": sha256_file(label_receipt_path),
        "label_mapping_sha256": receipt["input_anchor_future_mapping_sha256"],
        "source_manifest_sha256": receipt["source_manifest_sha256"],
        "feature_namespaces": protocol["feature_namespaces"],
        "legacy_reuse_policy": {
            "accepted_legacy_payloads": 0,
            "reason": "legacy_features_valid=false and seeded provenance is mandatory",
            "raw_id_candidates_retained_as_diagnostics": True,
        },
        "counts": {
            "experiments_covered": 1096,
            "prefix_level_specs": len(spec_rows),
            "streaming_layer_groups": len({(r["vfm"], r["model_id"], r["timestep"], r["feature_layer"]) for r in spec_rows}),
            "target_layer_groups": len(target_layer_keys),
            "vfm_batches": len(group_rows),
            "streaming_files": sum(int(row["required_streaming_files"]) for row in spec_rows),
            "target_scene_files": len(target_layer_keys) * 681,
            "specs_by_backend_status": dict(sorted(status_counts.items())),
        },
        "by_vfm": by_vfm,
        "batch_groups": group_rows,
    }
    plan["content_sha256"] = canonical_json_sha(plan)

    candidates: list[dict[str, Any]] = []
    for model in DEFAULT8_MODELS:
        for prefix, (exact, mismatch) in DEFAULT8_RAW_ID_UPPER_BOUND.items():
            candidates.append(
                {
                    "legacy_campaign": "default8",
                    "vfm": model,
                    "prefix": prefix,
                    "raw_id_exact_candidates": exact,
                    "raw_id_mismatch": mismatch,
                    "canonical_units": exact + mismatch,
                    "accepted_payload_reuse": 0,
                    "decision": "DIAGNOSTIC_ONLY_REJECTED_BY_SEEDED_V1",
                }
            )
    return plan, spec_rows, candidates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--label-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan, rows, candidates = build(
        inventory_path=args.inventory.resolve(),
        protocol_path=args.protocol.resolve(),
        label_receipt_path=args.label_receipt.resolve(),
    )
    output = args.output_dir.resolve()
    atomic_json(output / "reextract_plan.json", plan)
    fieldnames = list(rows[0])
    for extra in (
        "action",
        "canonical_requirement",
        "reuse_decision",
        "reuse_reason",
        "backend_status",
        "backend_blocker",
        "required_streaming_files",
        "required_target_scene_files",
    ):
        if extra not in fieldnames:
            fieldnames.append(extra)
    atomic_csv(output / "reextract_specs.csv", rows, fieldnames)
    atomic_csv(
        output / "legacy_raw_id_candidates.csv",
        candidates,
        list(candidates[0]),
    )
    print(json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
