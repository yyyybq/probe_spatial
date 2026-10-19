#!/usr/bin/env python3
"""Freeze exact, positive extraction worklists for the seven-model 528 scope.

The input audit has already reconciled the frozen probing manifests, owner
databases, and later extraction receipts.  This step deliberately performs no
feature discovery: it only converts the exact repair rows into the existing
canonical-v2 gap extractor contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STREAM_TEMPLATE_DIR = Path(
    "artifacts/windowfix_v2/"
    "streaming_gap_worklists_ebc496eb0587_catalog_r12_qwen_pass_20260831/"
    "gap_worklists"
)
TARGET_TEMPLATE_DIR = Path(
    "artifacts/windowfix_v2/target_row_merge_initial/"
    "route_audit_aoss_merged_r2_20260831/gap_worklists"
)
TARGET_ROUTE_AUDIT = Path(
    "artifacts/windowfix_v2/target_row_merge_initial/"
    "route_audit_aoss_merged_r2_20260831/target_row_route_audit.json"
)
EXPECTED_STREAM_LAYER_ITEMS = 128_393
EXPECTED_TARGET_LAYER_ITEMS = 634_384
RETIRED_TARGET_CLASSES = {"repair_no_owner", "owner_in_old_namespace"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def templates(directory: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in directory.glob("*.json"):
        value = read_json(path)
        vfm = str(value["vfm"])
        if vfm in result:
            raise ValueError(f"duplicate template for {vfm}")
        result[vfm] = value
    return result


def base_from_template(template: dict[str, Any], layers: list[int]) -> dict[str, Any]:
    fields = (
        "schema_version", "contract", "canonical_protocol", "protocol_sha256",
        "protocol_file_sha256", "canonical_source_sha256", "mode", "vfm",
        "model_id", "timestep", "feature_namespace", "source",
        "prompt_contract", "preprocess_identity",
    )
    value = {field: template[field] for field in fields}
    value["output_layers"] = sorted(layers)
    return value


def finish_worklist(value: dict[str, Any], entries: list[dict[str, Any]]) -> dict[str, Any]:
    if not entries:
        raise ValueError(f"empty worklist for {value['vfm']} {value['mode']}")
    value["entries"] = entries
    value["counts"] = {
        "entries": len(entries),
        "layer_items": sum(len(entry["layers"]) for entry in entries),
        "scenes": len({entry["scene"] for entry in entries}),
    }
    value["content_sha256"] = canonical_json_sha(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", required=True, type=Path)
    parser.add_argument("--target-root-query", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    audit_dir = args.audit_dir.resolve()
    root_query_path = args.target_root_query.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise ValueError(f"refusing to overwrite immutable output: {output_dir}")

    audit_manifest = read_json(audit_dir / "manifest.json")
    root_query = read_json(root_query_path)
    roots = {
        str(item.get("root", "")).rstrip("/").rsplit("/", 1)[-1]: item
        for item in root_query.get("roots", [])
        if isinstance(item, dict)
    }
    for namespace in ("FEAT_TARGET", "FEAT_TARGET_BY_PREFIX"):
        evidence = roots.get(namespace, {})
        if (
            evidence.get("isdir") is not False
            or evidence.get("error") is not None
            or evidence.get("first_entries") != []
        ):
            raise ValueError(f"retired target root is not proven absent: {namespace}")

    stream_entries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stream_layers: dict[str, set[int]] = defaultdict(set)
    stream_source = audit_dir / "streaming_extractor_entries.jsonl"
    with stream_source.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            vfm = str(row["model"])
            layers = sorted(int(value) for value in row["layers"])
            stream_layers[vfm].update(layers)
            stream_entries[vfm].append({
                "scene": str(row["scene_id"]),
                "window_id": int(row["window_id"]),
                "prefix": int(row["prefix"]),
                "layers": layers,
                "reason": "RETIRED_V1_STREAMING_NAMESPACE",
            })

    target_grouped: dict[str, dict[tuple[str, int], set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    target_source = audit_dir / "target_review_and_repair_units.jsonl"
    with target_source.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            classification = str(row["classification"])
            if classification not in RETIRED_TARGET_CLASSES:
                raise ValueError(f"unexpected target classification: {classification}")
            vfm = str(row["model"])
            key = (str(row["scene_id"]), int(row["target_raw_id"]))
            target_grouped[vfm][key].add(int(row["layer"]))

    stream_templates = templates(STREAM_TEMPLATE_DIR.resolve())
    target_templates = templates(TARGET_TEMPLATE_DIR.resolve())
    output_dir.mkdir(parents=True)
    worklist_dir = output_dir / "gap_worklists"
    worklist_dir.mkdir()

    bindings: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []

    def emit(value: dict[str, Any]) -> None:
        stem = f"{value['vfm']}-t{value['timestep']}-{value['mode']}"
        filename = f"{stem}-{value['content_sha256'][:12]}.json"
        path = (worklist_dir / filename).resolve()
        write_json(path, value)
        binding = {
            "path": str(path),
            "file_sha256": sha256_file(path),
            "content_sha256": value["content_sha256"],
            "mode": value["mode"],
            "vfm": value["vfm"],
            "timestep": value["timestep"],
            "counts": value["counts"],
        }
        bindings.append(binding)
        summary.append({
            "model": value["vfm"],
            "mode": value["mode"],
            "layers": value["output_layers"],
            **value["counts"],
            "worklist": str(path),
            "file_sha256": binding["file_sha256"],
        })

    for vfm in sorted(stream_entries):
        entries = sorted(
            stream_entries[vfm],
            key=lambda row: (row["scene"], row["window_id"], row["prefix"]),
        )
        emit(finish_worklist(
            base_from_template(stream_templates[vfm], sorted(stream_layers[vfm])),
            entries,
        ))

    for vfm in sorted(target_grouped):
        grouped = target_grouped[vfm]
        layers = sorted({layer for values in grouped.values() for layer in values})
        entries = [
            {
                "scene": scene,
                "frame_index": frame_index,
                "layers": sorted(grouped[(scene, frame_index)]),
                "reason": "MISSING_V2_TARGET_AFTER_RETIRED_V1_NAMESPACE",
            }
            for scene, frame_index in sorted(grouped)
        ]
        emit(finish_worklist(base_from_template(target_templates[vfm], layers), entries))

    stream_total = sum(
        row["layer_items"] for row in summary if row["mode"] == "streaming_prefix"
    )
    target_total = sum(
        row["layer_items"] for row in summary if row["mode"] == "target_isolated"
    )
    if stream_total != EXPECTED_STREAM_LAYER_ITEMS:
        raise ValueError(f"stream total mismatch: {stream_total}")
    if target_total != EXPECTED_TARGET_LAYER_ITEMS:
        raise ValueError(f"target total mismatch: {target_total}")

    protocol_path = Path(
        "/mnt/umm/users/yinbaiqiao/probe_spatial/"
        "protocols/inscene_streaming_canonical_v2.json"
    ).resolve()
    canonical_source_path = Path(
        "/mnt/umm/users/yinbaiqiao/probe_spatial/"
        "manifests/c_retrain_20260722/inscene_c_h124_prefix24.jsonl"
    ).resolve()
    common_source_shas = {read_json(Path(item["path"]))["canonical_source_sha256"] for item in bindings}
    # Worklist files all bind the same canonical source.  Read them back from
    # disk so the source audit proves the exact immutable payloads it lists.
    if common_source_shas != {sha256_file(canonical_source_path)}:
        raise ValueError("canonical-source binding mismatch")

    target_route_audit = read_json(TARGET_ROUTE_AUDIT.resolve())

    def emit_source_audit(filename: str, selected: list[dict[str, Any]]) -> dict[str, Any]:
        selected_stream = sum(
            item["counts"]["layer_items"]
            for item in selected if item["mode"] == "streaming_prefix"
        )
        selected_target = sum(
            item["counts"]["layer_items"]
            for item in selected if item["mode"] == "target_isolated"
        )
        source_audit = {
            "schema_version": 1,
            "audit_type": "PROBE528_TARGETED_REEXTRACT_POSITIVE_ALLOWLIST_V1",
            "status": "PASS",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "canonical_protocol": "inscene_streaming_canonical_v2_obs28_h124",
            "protocol_file": str(protocol_path),
            "protocol_file_sha256": sha256_file(protocol_path),
            "canonical_source": {
                "path": str(canonical_source_path),
                "sha256": sha256_file(canonical_source_path),
            },
            # The existing target-canary runner validates tensor shape/dtype
            # against this immutable canonical-v2 merge plan.
            "plan": target_route_audit["plan"],
            "plan_sha256": target_route_audit["plan_sha256"],
            "plan_content_sha256": target_route_audit["plan_content_sha256"],
            "inputs": {
                "targeted_audit_manifest": {
                    "path": str((audit_dir / "manifest.json").resolve()),
                    "sha256": sha256_file(audit_dir / "manifest.json"),
                },
                "streaming_entries": {"path": str(stream_source), "sha256": sha256_file(stream_source)},
                "target_entries": {"path": str(target_source), "sha256": sha256_file(target_source)},
                "target_root_query": {"path": str(root_query_path), "sha256": sha256_file(root_query_path)},
                "target_tensor_contract_audit": {
                    "path": str(TARGET_ROUTE_AUDIT.resolve()),
                    "sha256": sha256_file(TARGET_ROUTE_AUDIT.resolve()),
                },
            },
            "policy": {
                "selection": "exact positive rows only; never infer a full extraction",
                "streaming": "repair exact owners in deliberately retired V1 namespaces",
                "target": "repair no-owner rows plus exact owners in proven-empty retired V1 roots",
                "later_v2_receipts": "retained and excluded",
            },
            "counts": {
                "hard_errors": 0,
                "worklists": len(selected),
                "streaming_layer_items": selected_stream,
                "target_layer_items": selected_target,
            },
            "hard_errors": [],
            "outputs": {"gap_worklists": selected},
        }
        source_audit["content_sha256"] = canonical_json_sha(source_audit)
        path = output_dir / filename
        write_json(path, source_audit)
        return {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "content_sha256": source_audit["content_sha256"],
        }

    streaming_audit = emit_source_audit(
        "source_audit_streaming.json",
        [item for item in bindings if item["mode"] == "streaming_prefix"],
    )
    target_audit = emit_source_audit(
        "source_audit_target.json",
        [item for item in bindings if item["mode"] == "target_isolated"],
    )

    final_scope = {
        "schema": "PROBE528_TARGETED_REEXTRACT_SCOPE_V1",
        "status": "DETAILED_SCOPE_FROZEN_READY_FOR_CANARY",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_probing_experiments": audit_manifest["queue"],
        "streaming_layer_items_to_extract": stream_total,
        "target_layer_items_to_extract": target_total,
        "worklists": summary,
        "source_audits": [streaming_audit, target_audit],
    }
    final_scope["content_sha256"] = canonical_json_sha(final_scope)
    write_json(output_dir / "final_scope.json", final_scope)
    print(json.dumps({
        "status": final_scope["status"],
        "worklists": len(bindings),
        "streaming_layer_items": stream_total,
        "target_layer_items": target_total,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
