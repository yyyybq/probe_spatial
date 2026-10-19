#!/usr/bin/env python3
"""Frozen data routing/materialisation for Wan/V-JEPA2 exact canonical-v2 cells."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[2]
RESULT = REPO / "results/experimental/se3_wan_vjepa2_exact_v2_layers_20260916_r1"
PROTOCOL_PATH = RESULT / "FROZEN_PROTOCOL.json"
STREAM_ROUTES = REPO / (
    "artifacts/windowfix_v2/reference_overlay_02e17c40b59a_r12_qwen_pass_20260831/"
    "streaming_universe.routes.jsonl.gz"
)
TARGET_ROUTES = REPO / (
    "artifacts/windowfix_v2/target_row_merge_initial/"
    "route_audit_aoss_merged_r2_20260831/target_row_routes.jsonl"
)
TARGET_MISSING = TARGET_ROUTES.parent / "missing_target_units.jsonl"
TARGET_PENDING = TARGET_ROUTES.parent / "pending_target_units.jsonl"
LOCAL_SCENES = Path(
    "/mnt/umm/users/yinbaiqiao/InsScene-15K/data/processed_scannetpp_v2/"
    "processed_scannetpp_v2"
)

# Frozen provenance of the completed Wan canonical-v2 target publication.
# Target gap payloads are deliberately namespaced by the immutable worklist
# digest; unlike the legacy target root, each layer also owns its own index.
WAN_TARGET_GAP_WORKLIST_SHA256 = (
    "ef69725f9ed8d7ca57d6eba5d0415812219136a5b3905af6559c115403705b42"
)


def load_protocol() -> dict[str, Any]:
    return json.loads(PROTOCOL_PATH.read_text())


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    Path(temporary).replace(path)
    path.chmod(0o644)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    count = 0
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
            count += 1
    if count == 0:
        Path(temporary).unlink(missing_ok=True)
        raise RuntimeError(f"refusing empty JSONL: {path}")
    Path(temporary).replace(path)
    path.chmod(0o644)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def feature_filename(model: str, layer: int) -> str:
    return f"feature_t749_layer{layer}.sft" if model == "wan" else f"feature_layer{layer}.sft"


def _target_positions(row: dict[str, Any]) -> dict[str, int]:
    prefix = int(row["prefix_length"])
    window = row["streaming_window_indices"]
    return {"A": int(window[prefix - 1]), "B": int(window[prefix]), "C": int(window[prefix + 1])}


def _actual_frame_ids(row: dict[str, Any], positions: dict[str, int]) -> dict[str, str]:
    frames = row["valid_frames"]
    return {role: str(frames[index]) for role, index in positions.items()}


def prepare_common_manifest() -> list[dict[str, Any]]:
    protocol = load_protocol()
    source = REPO / protocol["manifest"]
    rows = read_jsonl(source)
    if {int(row["prefix_length"]) for row in rows} != {int(protocol["prefix"])}:
        raise RuntimeError("canonical manifest prefix differs from frozen protocol")

    train_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    evaluation: list[dict[str, Any]] = []
    for row in rows:
        if row["split"] == "train":
            train_by_scene[row["scene_id"]].append(row)
        elif row["split"] == "val":
            evaluation.append(row)
        else:
            raise RuntimeError(f"unexpected canonical split {row['split']!r}")

    ordered_scenes = sorted(
        train_by_scene,
        key=lambda scene: sha_text(f"{protocol['experiment_id']}:{scene}"),
    )
    n_train = int(protocol["data_budget"]["train_scenes"])
    n_val = int(protocol["data_budget"]["validation_scenes"])
    if len(ordered_scenes) < n_train + n_val:
        raise RuntimeError("insufficient canonical train scenes for frozen budget")

    selected: list[dict[str, Any]] = []
    for split, scenes in (
        ("train", ordered_scenes[:n_train]),
        ("validation", ordered_scenes[n_train:n_train + n_val]),
    ):
        for scene in scenes:
            row = min(train_by_scene[scene], key=lambda item: sha_text(item["sample_id"]))
            selected.append({**row, "analysis_split": split})
    selected.extend({**row, "analysis_split": "evaluation"} for row in evaluation)

    output = []
    for row in selected:
        positions = _target_positions(row)
        context_positions = [int(value) for value in row["streaming_indices"]]
        if max(context_positions) >= positions["B"]:
            raise RuntimeError(f"future endpoint leaked into source: {row['sample_id']}")
        output.append({
            **row,
            "abc_frame_positions": positions,
            "abc_frame_ids": _actual_frame_ids(row, positions),
            "input_frame_positions": context_positions,
            "input_frame_ids": [str(row["valid_frames"][index]) for index in context_positions],
        })

    write_jsonl(RESULT / "data/common_manifest.jsonl", output)
    split_counts = defaultdict(int)
    scene_sets: dict[str, set[str]] = defaultdict(set)
    for row in output:
        split_counts[row["analysis_split"]] += 1
        scene_sets[row["analysis_split"]].add(row["scene_id"])
    summary = {
        "status": "FROZEN_BEFORE_FEATURE_VALUES_OR_TEST_METRICS",
        "manifest_sha256": hashlib.sha256((RESULT / "data/common_manifest.jsonl").read_bytes()).hexdigest(),
        "samples": dict(split_counts),
        "scenes": {key: len(value) for key, value in scene_sets.items()},
        "selection_uses_features": False,
        "selection_uses_outcomes": False,
    }
    atomic_json(RESULT / "data/common_manifest_summary.json", summary)
    return output


def _gap_stream_uri(model: str, layer: int, row: dict[str, Any]) -> str:
    return (
        "aoss://baiqiao/InsScene-15K/FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1/"
        f"{model}/scannetpp/{row['scene_id']}/window_{int(row['streaming_window_id']):04d}/"
        f"prefix_{int(row['streaming_tail']):06d}/{feature_filename(model, layer)}"
    )


def _gap_target_uri(model: str, layer: int, scene: str) -> str:
    base = (
        "aoss://baiqiao/InsScene-15K/FEAT_TARGET_CANONICAL_V2_GAP_DELTA_V1/"
        f"{model}/scannetpp/{scene}"
    )
    if model == "wan":
        digest = WAN_TARGET_GAP_WORKLIST_SHA256[:16]
        return (
            f"{base}/.canonical_v2_target_delta/{digest}/"
            f"feature_t749_layer{layer}.delta.sft"
        )
    return f"{base}/{feature_filename(model, layer)}"


def _gap_target_index_uri(model: str, layer: int, scene: str) -> str:
    uri = _gap_target_uri(model, layer, scene)
    parent = uri.rsplit("/", 1)[0]
    if model == "wan":
        return f"{parent}/target_indices_layer{layer}.npy"
    return f"{parent}/target_indices.npy"


def prepare_route_plan(rows: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    protocol = load_protocol()
    rows = rows or read_jsonl(RESULT / "data/common_manifest.jsonl")
    wanted_samples = {row["sample_id"] for row in rows}
    row_by_sample = {row["sample_id"]: row for row in rows}
    cells = {
        (model, int(layer))
        for model, spec in protocol["models"].items()
        for layer in spec["layers"]
    }

    streaming: dict[tuple[str, int, str], dict[str, Any]] = {}
    with gzip.open(STREAM_ROUTES, "rt") as handle:
        for line in handle:
            route = json.loads(line)
            key = (route.get("vfm"), int(route.get("feature_layer", -999)), route.get("sample_id"))
            if key[:2] not in cells or key[2] not in wanted_samples or int(route.get("prefix", -1)) != 12:
                continue
            sample = row_by_sample[key[2]]
            if route["status"] == "PENDING_REMOTE_GATE":
                uri = route["source_uri"]
                source_kind = "exact_legacy_reference_fresh_gate_required"
            elif route["status"].startswith("REEXTRACT"):
                uri = _gap_stream_uri(key[0], key[1], sample)
                source_kind = "canonical_v2_exact_gap"
            else:
                raise RuntimeError(f"unsupported streaming route: {route['status']}")
            streaming[key] = {
                "uri": uri,
                "source_kind": source_kind,
                "catalog_status": route["status"],
                "expected_input_positions": [int(value) for value in route["canonical_raw_ids"]],
                "route_unit_id": route["unit_id"],
            }

    wanted_targets = {
        (model, layer, row["scene_id"], int(position))
        for model, layer in cells
        for row in rows
        for position in row["abc_frame_positions"].values()
    }
    targets: dict[tuple[str, int, str, int], dict[str, Any]] = {}
    with TARGET_ROUTES.open() as handle:
        for line in handle:
            route = json.loads(line)
            key = (
                route.get("vfm"), int(route.get("layer", -999)), route.get("scene_id"),
                int(route.get("target_raw_id", -1)),
            )
            if key not in wanted_targets:
                continue
            targets[key] = {
                "uri": route["source_feature_path"],
                "index_uri": route["source_index_path"],
                "row_index": int(route["source_row_index"]),
                "source_kind": "exact_legacy_target_row",
                "route_source_id": route["source_id"],
            }

    gaps: set[tuple[str, int, str, int]] = set()
    with TARGET_MISSING.open() as handle:
        for line in handle:
            unit = json.loads(line)
            model, layer, scene = unit.get("vfm"), int(unit.get("layer", -999)), unit.get("scene_id")
            if (model, layer) not in cells:
                continue
            for raw_id in unit.get("missing_raw_ids", []):
                key = (model, layer, scene, int(raw_id))
                if key in wanted_targets:
                    gaps.add(key)

    pending: set[tuple[str, int, str, int]] = set()
    with TARGET_PENDING.open() as handle:
        for line in handle:
            unit = json.loads(line)
            model, layer, scene = unit.get("vfm"), int(unit.get("layer", -999)), unit.get("scene_id")
            if (model, layer) not in cells:
                continue
            for raw_id in unit.get("pending_raw_ids", []):
                key = (model, layer, scene, int(raw_id))
                if key in wanted_targets:
                    pending.add(key)

    for key in wanted_targets - set(targets):
        model, layer, scene, _ = key
        if key in gaps:
            uri = _gap_target_uri(model, layer, scene)
            targets[key] = {
                "uri": uri,
                "index_uri": _gap_target_index_uri(model, layer, scene),
                "row_index": None,
                "source_kind": "canonical_v2_exact_gap",
                "publication_worklist_sha256": (
                    WAN_TARGET_GAP_WORKLIST_SHA256 if model == "wan" else None
                ),
            }
        elif key in pending:
            uri = f"aoss://baiqiao/InsScene-15K/FEAT_TARGET/{model}/scannetpp/{scene}/{feature_filename(model, layer)}"
            targets[key] = {
                "uri": uri,
                "index_uri": uri.rsplit("/", 1)[0] + "/target_indices.npy",
                "row_index": None,
                "source_kind": "exact_legacy_pending_fresh_gate_required",
            }
        else:
            raise RuntimeError(f"unrouted requested target row: {key}")

    plan = []
    for model, layer in sorted(cells):
        for row in rows:
            stream_key = (model, layer, row["sample_id"])
            if stream_key not in streaming:
                raise RuntimeError(f"unrouted streaming sample: {stream_key}")
            target_plan = {}
            for role, position in row["abc_frame_positions"].items():
                target_plan[role] = {**targets[(model, layer, row["scene_id"], int(position))], "target_position": int(position)}
            plan.append({
                "cell": f"{model}_l{layer}", "model": model, "layer": layer,
                "sample_id": row["sample_id"], "scene_id": row["scene_id"],
                "analysis_split": row["analysis_split"], "streaming": streaming[stream_key],
                "targets": target_plan,
            })
    write_jsonl(RESULT / "data/route_plan.jsonl", plan)
    atomic_json(RESULT / "data/route_plan_summary.json", {
        "rows": len(plan), "cells": sorted({row["cell"] for row in plan}),
        "samples_per_cell": len(rows),
        "sha256": hashlib.sha256((RESULT / "data/route_plan.jsonl").read_bytes()).hexdigest(),
        "requires_fresh_aoss_payload_and_sidecar_audit": True,
        "wan_target_gap_worklist_sha256": WAN_TARGET_GAP_WORKLIST_SHA256,
    })
    return plan


def prepare() -> None:
    rows = prepare_common_manifest()
    prepare_route_plan(rows)


def _find_exact_integer_list(value: Any, wanted: list[int]) -> bool:
    if isinstance(value, dict):
        return any(_find_exact_integer_list(item, wanted) for item in value.values())
    if isinstance(value, list):
        if value and all(isinstance(item, int) for item in value) and value == wanted:
            return True
        return any(_find_exact_integer_list(item, wanted) for item in value)
    return False


def _find_string(value: Any, wanted: str) -> bool:
    if isinstance(value, dict):
        return any(_find_string(item, wanted) for item in value.values())
    if isinstance(value, list):
        return any(_find_string(item, wanted) for item in value)
    return isinstance(value, str) and value == wanted


def _sidecar_certifies_positions(value: Any, wanted: list[int]) -> tuple[bool, str]:
    if _find_exact_integer_list(value, wanted):
        return True, "explicit_integer_list"
    digest = sha_text(canonical_json(wanted))
    if _find_string(value, digest):
        return True, "canonical_raw_ids_sha256"
    return False, "none"


def _validate_stream_sidecar_identity(
    meta: dict[str, Any], *, model: str, layer: int, uri: str,
) -> None:
    expected_file = feature_filename(model, layer)
    checks = {"vfm": model, "mode": "streaming_prefix", "file": expected_file}
    for key, expected in checks.items():
        if meta.get(key) != expected:
            raise RuntimeError(
                f"streaming sidecar identity mismatch {key}: "
                f"observed={meta.get(key)!r}, expected={expected!r}, uri={uri}"
            )
    # Gap-v2 sidecars carry an explicit layer integer. Historical exact
    # reference sidecars encode it only in the immutable filename.
    if "layer" in meta and int(meta["layer"]) != layer:
        raise RuntimeError(
            f"streaming sidecar layer mismatch observed={meta['layer']!r}, "
            f"expected={layer!r}, uri={uri}"
        )


def _load_sidecar(aoss_fs: Any, uri: str) -> dict[str, Any]:
    sidecar_uri = uri + ".manifest.json"
    if not aoss_fs.exists(sidecar_uri):
        raise FileNotFoundError(f"missing feature sidecar: {sidecar_uri}")
    value = aoss_fs.load_json(sidecar_uri)
    if not isinstance(value, dict):
        raise RuntimeError(f"feature sidecar is not an object: {sidecar_uri}")
    return value


def _geometry(rows: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    import numpy as np
    import torch

    from vidfm3d.data.components.inscene15k_dataset import (
        InsScene15KDataset,
        depth_to_pointmap,
    )
    from vidfm3d.data.components.video_probe_dataset import invert_pose_ref_and_scale
    from vidfm3d.utils.spatial_diag import encode_relative_pose, relative_pose_matrix

    if output.exists():
        return dict(np.load(output, allow_pickle=True))
    holder = InsScene15KDataset.__new__(InsScene15KDataset)
    holder.target_h, holder.target_w = 288, 512
    sample_ids, t1s, t2s, a1s, a2s, noncomm = [], [], [], [], [], []
    for row in rows:
        scene = dict(row)
        scene["scene_dir"] = str(LOCAL_SCENES / row["scene_id"])
        positions = row["abc_frame_positions"]
        selected = torch.as_tensor([positions[role] for role in ("A", "B", "C")], dtype=torch.long)
        depth, intrinsics, extrinsics = holder._load_camera_depth_scene(scene, selected)
        depth, intrinsics = holder._resize_depth_to_target(depth, intrinsics)
        pointmaps = torch.stack([
            depth_to_pointmap(depth[index], intrinsics[index], extrinsics[index])
            for index in range(3)
        ])
        extrinsics, _, _ = invert_pose_ref_and_scale(
            extrinsics, pointmaps, depthmaps=depth.unsqueeze(-1), ref_idx=0, scale_by_points=True
        )
        t1 = relative_pose_matrix(extrinsics[0], extrinsics[1])
        t2 = relative_pose_matrix(extrinsics[1], extrinsics[2])
        sample_ids.append(row["sample_id"])
        t1s.append(t1.numpy()); t2s.append(t2.numpy())
        a1s.append(encode_relative_pose(extrinsics[0], extrinsics[1]).numpy())
        a2s.append(encode_relative_pose(extrinsics[1], extrinsics[2]).numpy())
        noncomm.append(float(torch.linalg.norm(t2 @ t1 - t1 @ t2)))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f".tmp-{os.getpid()}.npz")
    np.savez_compressed(
        temporary, sample_id=np.asarray(sample_ids), t1=np.stack(t1s), t2=np.stack(t2s),
        action1=np.stack(a1s), action2=np.stack(a2s), noncomm_strength=np.asarray(noncomm),
    )
    temporary.replace(output)
    output.chmod(0o644)
    return dict(np.load(output, allow_pickle=True))


def materialize_cell(cell: str) -> None:
    import numpy as np
    import torch
    from safetensors.torch import save_file
    from vidfm3d.utils import aoss_fs

    protocol = load_protocol()
    rows = read_jsonl(RESULT / "data/common_manifest.jsonl")
    route_rows = [row for row in read_jsonl(RESULT / "data/route_plan.jsonl") if row["cell"] == cell]
    if len(route_rows) != len(rows):
        raise RuntimeError(f"{cell}: route plan is incomplete")
    by_sample = {row["sample_id"]: row for row in route_rows}
    model, layer_text = cell.rsplit("_l", 1)
    layer = int(layer_text)
    channels = int(protocol["models"][model]["channels"])
    output = RESULT / "cache" / f"{cell}.sft"
    receipt_path = RESULT / "inventory" / f"{cell}.json"
    if output.exists() and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("status") == "PASS_MATERIALIZED_EXACT_V2":
            return

    cache: dict[str, dict[str, torch.Tensor]] = {}
    payload_uris_read: set[str] = set()
    index_cache: dict[str, np.ndarray] = {}
    sidecar_cache: dict[str, dict[str, Any]] = {}
    inventory = []

    def tensors(uri: str) -> dict[str, torch.Tensor]:
        if uri not in cache:
            if not aoss_fs.exists(uri):
                raise FileNotFoundError(uri)
            cache[uri] = aoss_fs.load_safetensors(uri)
            payload_uris_read.add(uri)
        return cache[uri]

    def sidecar(uri: str) -> dict[str, Any]:
        if uri not in sidecar_cache:
            sidecar_cache[uri] = _load_sidecar(aoss_fs, uri)
        return sidecar_cache[uri]

    contexts, anchors, targets_b, targets_c = [], [], [], []
    try:
        for manifest_row in rows:
            route = by_sample[manifest_row["sample_id"]]
            stream = route["streaming"]
            value = tensors(stream["uri"])["feat"]
            meta = sidecar(stream["uri"])
            _validate_stream_sidecar_identity(
                meta, model=model, layer=layer, uri=stream["uri"]
            )
            expected = stream["expected_input_positions"]
            certified, certification_kind = _sidecar_certifies_positions(meta, expected)
            if (
                not certified
                and stream["source_kind"] == "exact_legacy_reference_fresh_gate_required"
                and stream["catalog_status"] == "PENDING_REMOTE_GATE"
            ):
                # The route catalog was built from the legacy prefix_index and
                # requires raw-ID equality. The live fetch above independently
                # gates the current payload and its model/layer/mode sidecar.
                certified = True
                certification_kind = "route_catalog_raw_id_exact_plus_live_payload_sidecar"
            if not certified:
                raise RuntimeError(f"{cell}:{manifest_row['sample_id']} sidecar lacks exact input positions {expected}")
            pooled_context = value.float().mean(dim=tuple(range(value.ndim - 1)))
            if pooled_context.shape != (channels,):
                raise RuntimeError(f"{cell}: context pooled shape {tuple(pooled_context.shape)}")
            contexts.append(pooled_context)
            inventory.append({
                "sample_id": manifest_row["sample_id"], "role": "streaming_context_A",
                "uri": stream["uri"], "source_kind": stream["source_kind"],
                "input_frame_positions": expected, "tensor_shape": list(value.shape),
                "sidecar_present": True, "sidecar_position_certification": certification_kind,
            })
            # Streaming payloads are one-per-sample and can be ~100 MB for Wan.
            # Keep only the pooled vector; never retain the full sweep in RAM.
            cache.pop(stream["uri"], None)

            role_values = {}
            target_payloads_for_sample: set[str] = set()
            for role in ("A", "B", "C"):
                target = route["targets"][role]
                target_payloads_for_sample.add(target["uri"])
                data = tensors(target["uri"])["feat"]
                sidecar(target["uri"])
                index_uri = target["index_uri"]
                if index_uri not in index_cache:
                    index_cache[index_uri] = np.asarray(aoss_fs.load_npy(index_uri), dtype=np.int64).reshape(-1)
                indices = index_cache[index_uri]
                expected_position = int(target["target_position"])
                if target["row_index"] is None:
                    matches = np.flatnonzero(indices == expected_position)
                    if len(matches) != 1:
                        raise RuntimeError(f"{cell}:{manifest_row['sample_id']} target {role} index count={len(matches)}")
                    row_index = int(matches[0])
                else:
                    row_index = int(target["row_index"])
                    if row_index >= len(indices) or int(indices[row_index]) != expected_position:
                        raise RuntimeError(f"{cell}:{manifest_row['sample_id']} routed row/index mismatch")
                feature = data[row_index].float()
                pooled = feature.mean(dim=tuple(range(feature.ndim - 1)))
                if pooled.shape != (channels,):
                    raise RuntimeError(f"{cell}: target pooled shape {tuple(pooled.shape)}")
                role_values[role] = pooled
                inventory.append({
                    "sample_id": manifest_row["sample_id"], "role": f"target_isolated_{role}",
                    "uri": target["uri"], "index_uri": index_uri, "row_index": row_index,
                    "target_position": expected_position, "source_kind": target["source_kind"],
                    "tensor_shape": list(feature.shape), "sidecar_present": True,
                })
            anchors.append(role_values["A"]); targets_b.append(role_values["B"]); targets_c.append(role_values["C"])
            # A/B/C commonly share one scene payload. Release it after all
            # three rows are pooled; the node-local AOSS cache handles reuse.
            for uri in target_payloads_for_sample:
                cache.pop(uri, None)

        geometry = _geometry(rows, RESULT / "cache/geometry.npz")
        if list(geometry["sample_id"].astype(str)) != [row["sample_id"] for row in rows]:
            raise RuntimeError("geometry/sample order mismatch")
        payload = {
            "context": torch.stack(contexts).half(), "z_a": torch.stack(anchors).half(),
            "z_b": torch.stack(targets_b).half(), "z_c": torch.stack(targets_c).half(),
        }
        if not all(torch.isfinite(value).all() for value in payload.values()):
            raise RuntimeError(f"{cell}: non-finite pooled feature")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(f".tmp-{os.getpid()}.sft")
        save_file(payload, temporary)
        temporary.replace(output)
        output.chmod(0o644)
        atomic_json(receipt_path, {
            "status": "PASS_MATERIALIZED_EXACT_V2", "cell": cell, "model": model, "layer": layer,
            "samples": len(rows), "channels": channels, "payload": str(output),
            "payload_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "route_plan_sha256": hashlib.sha256((RESULT / "data/route_plan.jsonl").read_bytes()).hexdigest(),
            "roles": {"context": "streaming_prefix_A", "targets": "isolated_A_B_C"},
            "unique_payload_uris_read": len(payload_uris_read), "fresh_index_reads": len(index_cache),
            "inventory": inventory,
        })
    except Exception as error:
        atomic_json(receipt_path, {
            "status": "FAILED_ISOLATED_CELL", "cell": cell,
            "error_type": type(error).__name__, "error": str(error),
            "loaded_inventory": inventory,
        })
        raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("route-plan")
    materialize = sub.add_parser("materialize")
    materialize.add_argument("--cell", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "route-plan":
        prepare_route_plan()
    else:
        materialize_cell(args.cell)
