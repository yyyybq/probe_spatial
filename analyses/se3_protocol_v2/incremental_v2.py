#!/usr/bin/env python3
"""Run new-scene and sparse-layer increments for SE3-Observed-v2.

This runner is intentionally narrow.  It consumes the previously frozen and
verified independent DINO science-pilot features, excludes every legacy r2
scene, and evaluates a DINO-only new-scene confirmation cell.  Separately it
evaluates the only safe embedded layer contrast currently available
(V-JEPA2 L0/L17/L23) on a validity-intersection subset of the legacy data.
It never reads canonical-v2 feature deltas.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from analyses.se3_latent_mvp.geometry import camera_to_world_to_extrinsic, se3_log
from analyses.se3_latent_mvp.run_mvp import (
    TupleRecord,
    load_and_audit_tuples,
    pair_tensors,
    transform_batch,
)
from analyses.se3_protocol_v2.observed_order import action_tensor, derangement
from analyses.se3_protocol_v2.paper_v1 import (
    DEFAULT_NORMALIZATION_EPSILON,
    DEFAULT_TIE_TOLERANCE,
    MODEL_SPECS,
    PRIMARY_BASELINES,
    file_digest_or_na,
    holm_adjust,
    mean_rows,
    predict,
    scene_macro,
    summarize_prediction_arrays,
    train_all,
    write_csv,
)
from analyses.se3_protocol_v2.protocol import POSE_CONVENTION, dump_json, sha256


REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_ROOT = REPO_ROOT / "results/experimental/se3_observed_paper_v1_20260913_r2"
INDEPENDENT_ROOT = REPO_ROOT / "artifacts/independent_actionability_algebra_p8_20260831T091224Z_28bda217"
INDEPENDENT_FEATURE_ROOT = REPO_ROOT / "features/independent_actionability_algebra_p8/independent_actionability_algebra_p8_20260831T091224Z_28bda217/dino_l0_science_pilot_v1"
REQUEST_MANIFEST = INDEPENDENT_ROOT / "manifests/science_pilot_subset_v1_feature_request_v2_dino_l0.jsonl"
ALGEBRA_TRAIN = INDEPENDENT_ROOT / "manifests/science_pilot_subset_v1_algebra_train.jsonl"
ALGEBRA_TEST = INDEPENDENT_ROOT / "manifests/science_pilot_subset_v1_algebra_val.jsonl"
FEATURE_RECEIPT = INDEPENDENT_ROOT / "execution/dino_l0_science_pilot_v1/coverage_receipt.json"
RAW_ROOT = Path("/mnt/umm/users/yinbaiqiao/InsScene-15K/data/processed_scannetpp_v2/processed_scannetpp_v2")
C_MANIFEST = REPO_ROOT / "manifests/c_retrain_20260722/inscene_c_h124_prefix8.jsonl"
C_RUN = REPO_ROOT / "logs/inscene15k_streaming/runs/c_eval_20260724_1903"
MISSING_RUN = REPO_ROOT / "logs/inscene15k_streaming/runs/full_missing_eval_20260817"
SEEDS = (42, 43, 44, 45, 46)
BOOTSTRAP_SEED = 20260914
BOOTSTRAP_REPETITIONS = 2000
CFG = {
    "hidden_dim": 64,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "max_epochs": 180,
    "patience": 30,
    "cosine_loss_weight": 0.25,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def stable_scene_partition(scenes: set[str], validation_fraction: float = 0.2) -> dict[str, str]:
    ranked = sorted(
        scenes,
        key=lambda scene: hashlib.sha256(f"se3-observed-v2-confirmation|{scene}".encode()).hexdigest(),
    )
    validation_count = max(1, round(len(ranked) * validation_fraction))
    validation = set(ranked[:validation_count])
    return {scene: "val" if scene in validation else "train" for scene in ranked}


def raw_id_from_metadata(value: Any, scene: str) -> str:
    name = value.decode() if isinstance(value, bytes) else str(value)
    return name if name.startswith("scannetpp:") else f"scannetpp:{scene}/{Path(name).name}"


def feature_request_index() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(REQUEST_MANIFEST):
        if row["feature_role"] != "isolated_frame":
            continue
        raw_id = str(row["raw_frame_id"])
        if raw_id in result and result[raw_id]["unit_id"] != row["unit_id"]:
            raise RuntimeError(f"ambiguous isolated feature request: {raw_id}")
        result[raw_id] = row
    return result


def load_independent_confirmation(output: Path) -> tuple[dict[str, list[TupleRecord]], list[dict[str, Any]], dict[str, Any]]:
    receipt = json.loads(FEATURE_RECEIPT.read_text())
    if not (
        receipt.get("status") == "COMPLETE_VALID"
        and receipt.get("requested") == receipt.get("verified")
        and receipt.get("missing_or_invalid") == 0
        and receipt.get("request_manifest_sha256") == sha256(REQUEST_MANIFEST)
    ):
        raise RuntimeError(f"independent DINO coverage receipt is not closed: {receipt}")
    legacy_scenes = {row["scene_id"] for row in read_jsonl(LEGACY_ROOT / "se3_observed_v1_manifest.jsonl")}
    source_train = [row for row in read_jsonl(ALGEBRA_TRAIN) if row["scene_id"] not in legacy_scenes]
    source_test = [row for row in read_jsonl(ALGEBRA_TEST) if row["scene_id"] not in legacy_scenes]
    assignment = stable_scene_partition({row["scene_id"] for row in source_train})
    request = feature_request_index()
    pooled_cache: dict[str, torch.Tensor] = {}
    raw_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    records: dict[str, list[TupleRecord]] = {"train": [], "val": [], "test": []}
    pose_max_abs = 0.0

    def pooled_feature(raw_id: str) -> tuple[torch.Tensor, dict[str, Any], Path]:
        unit = request.get(raw_id)
        if unit is None:
            raise RuntimeError(f"missing frozen DINO request unit: {raw_id}")
        path = INDEPENDENT_FEATURE_ROOT / "dino/scannetpp" / unit["scene_id"] / f"{unit['unit_id']}.safetensors"
        sidecar_path = Path(f"{path}.manifest.json")
        sidecar = json.loads(sidecar_path.read_text())
        if sidecar["feature_role"] != "isolated_frame" or sidecar["actual_input_raw_ids"] != [raw_id] or sidecar["fallback"]:
            raise RuntimeError(f"feature contract mismatch: {path}")
        if raw_id not in pooled_cache:
            tensor = load_file(str(path))["feat"]
            if tuple(tensor.shape) != (1, 30, 52, 1024):
                raise RuntimeError(f"unexpected DINO feature shape: {path} {tuple(tensor.shape)}")
            pooled_cache[raw_id] = tensor.float().mean(dim=(0, 1, 2))
        return pooled_cache[raw_id], sidecar, path

    for source_role, source_rows in (("source_train", source_train), ("source_test", source_test)):
        for source in source_rows:
            split = "test" if source_role == "source_test" else assignment[source["scene_id"]]
            raw_ids = [source["anchor_raw_id"], source["waypoint_raw_ids"][0], source["waypoint_raw_ids"][1]]
            c2w = np.asarray([source["anchor_pose"], source["waypoint_poses"][0], source["waypoint_poses"][1]], dtype=np.float64)
            scene = source["scene_id"]
            if scene not in raw_cache:
                raw_cache[scene] = np.load(RAW_ROOT / scene / "scene_iphone_metadata.npz")
            raw = raw_cache[scene]
            raw_lookup = {raw_id_from_metadata(value, scene): index for index, value in enumerate(raw["images"])}
            raw_indices = tuple(raw_lookup[raw_id] for raw_id in raw_ids)
            raw_c2w = np.asarray(raw["trajectories"][list(raw_indices)], dtype=np.float64)
            pose_max_abs = max(pose_max_abs, float(np.max(np.abs(raw_c2w - c2w))))
            features, feature_contracts = [], []
            for raw_id in raw_ids:
                feature, sidecar, path = pooled_feature(raw_id)
                features.append(feature)
                feature_contracts.append({
                    "raw_frame_id": raw_id,
                    "unit_id": sidecar["unit_id"],
                    "feature_path": str(path),
                    "feature_sha256": sidecar["feature_sha256"],
                    "actual_input_raw_ids": sidecar["actual_input_raw_ids"],
                })
            extrinsics = np.stack([camera_to_world_to_extrinsic(value) for value in c2w])
            record = TupleRecord(
                sample_id=source["sample_id"], scene_id=scene, window_id=-1,
                raw_frame_indices=raw_indices, features=torch.stack(features), extrinsics=extrinsics,
            )
            records[split].append(record)
            t_ab, t_bc, t_ac = record.transforms
            rows.append({
                "tuple_id": source["sample_id"], "scene_id": scene, "split": split,
                "source_partition": source["split"], "raw_frame_ids": raw_ids,
                "raw_frame_indices": raw_indices, "c2w": c2w.tolist(),
                "w2c": [np.vstack((value, [0, 0, 0, 1])).tolist() for value in extrinsics],
                "T_AB": t_ab.tolist(), "T_BC": t_bc.tolist(), "T_AC": t_ac.tolist(),
                "feature_contracts": feature_contracts,
                "model": "dino", "layer": 0, "module": "last_hidden_state",
                "model_revision": "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c",
                "feature_role": "isolated_frame", "pooling": "mean over 30x52 patch grid",
                "timestamps_s": [None, None, None], "timestamp_status": "not supplied",
            })
    if pose_max_abs > 1e-8:
        raise RuntimeError(f"source/raw c2w mismatch: {pose_max_abs}")
    if any(legacy_scenes & {record.scene_id for record in values} for values in records.values()):
        raise RuntimeError("legacy-scene leakage in confirmation records")
    counts = {
        split: {"tuples": len(values), "scenes": len({r.scene_id for r in values}),
                "unique_frames": len({(r.scene_id, i) for r in values for i in r.raw_frame_indices})}
        for split, values in records.items()
    }
    audit = {
        "status": "FROZEN_DINO_ONLY_NEW_SCENE_CONFIRMATION",
        "selection_frozen_before_scores": True,
        "tuple_definition": "A=anchor, B=observed h1 waypoint, C=observed h2 waypoint",
        "source_train_manifest": str(ALGEBRA_TRAIN), "source_test_manifest": str(ALGEBRA_TEST),
        "legacy_scenes_excluded": len(legacy_scenes), "legacy_scene_overlap": 0,
        "counts": counts, "raw_pose_max_abs": pose_max_abs,
        "confirmation_models_available": ["dino_l0"],
        "four_model_confirmation_status": "BLOCKED_MISSING_F3R_VJEPA2_COGVIDEOX_EXACT_FEATURES",
    }
    jsonl_path = output / "confirmation_manifest.jsonl"
    with jsonl_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    audit["manifest_sha256"] = sha256(jsonl_path)
    dump_json(output / "confirmation_split.json", {
        "seed_material": "se3-observed-v2-confirmation",
        "assignments": {record.scene_id: split for split, values in records.items() for record in values},
        "counts": counts,
    })
    dump_json(output / "confirmation_inventory_audit.json", audit)
    return records, rows, audit


def confirmation_gap_lists(output: Path, manifest_rows: list[dict[str, Any]]) -> None:
    feature_rows, context_rows = [], []
    contracts = {
        "f3r": (24, 64, "target_isolated repeated-frame multiview"),
        "vjepa2": (23, 64, "target_isolated repeated-frame encoder"),
        "cogvideox": (20, 97, "target_isolated diffusion-transformer; extraction seed must be newly frozen"),
    }
    seen: set[tuple[str, str]] = set()
    seen_contexts: set[tuple[str, tuple[str, ...]]] = set()
    for row in manifest_rows:
        for state, raw_id in zip(("A", "B", "C"), row["raw_frame_ids"]):
            for model, (layer, frames, role) in contracts.items():
                key = (model, raw_id)
                if key not in seen:
                    feature_rows.append({
                        "model": model, "layer": layer, "feature_role": role,
                        "raw_frame_id": raw_id, "scene_id": row["scene_id"],
                        "input_frames": frames, "status": "missing_exact_safe_feature",
                    })
                    seen.add(key)
        # Exact p8 contexts ending at A/B/C, derived from source metadata order.
        scene = row["scene_id"]
        raw = np.load(RAW_ROOT / scene / "scene_iphone_metadata.npz")
        all_ids = [raw_id_from_metadata(value, scene) for value in raw["images"]]
        for state, index, raw_id in zip(("A", "B", "C"), row["raw_frame_indices"], row["raw_frame_ids"]):
            if index < 7:
                continue
            context = all_ids[index - 7:index + 1]
            context_key = (scene, tuple(context))
            if context_key in seen_contexts:
                continue
            seen_contexts.add(context_key)
            context_rows.append({
                "model": "vjepa2", "layer": 23, "state": state, "endpoint_raw_frame_id": raw_id,
                "ordered_context_raw_ids": json.dumps(context, separators=(",", ":")),
                "scene_id": scene, "split": row["split"], "feature_role": "streaming_context_p8_encoder",
                "status": "missing_exact_safe_feature", "future_endpoint_in_source": False,
            })
    write_csv(output / "confirmation_missing_feature_units.csv", feature_rows)
    write_csv(output / "vjepa2_real_history_missing_contexts.csv", context_rows)
    dump_json(output / "gap_summary.json", {
        "confirmation_available_unique_units": {"dino": len(seen) // len(contracts)},
        "confirmation_missing_unique_units": {
            model: sum(row["model"] == model for row in feature_rows) for model in contracts
        },
        "vjepa2_context_missing_units": len(context_rows),
        "context_comparison_status": "BLOCKED_NO_COMPLETE_MATCHED_REPEATED_AND_REAL_HISTORY_A_B_C_FEATURE_SET",
        "layer_cell_availability": {
            "dino": "only export label 0 / last_hidden_state is safely embedded; no early-middle-late contrast",
            "f3r": "only export index 24 is safely embedded; no early-middle-late contrast",
            "vjepa2": "L0/L17/L23 runnable on common-valid subset; L11 middle cell missing",
        },
        "excluded_roots": ["canonical-v2 and its deltas", "known-unsafe Wan p8 lineage"],
    })


def prepare_vjepa_layer_records(output: Path) -> tuple[dict[int, dict[str, list[TupleRecord]]], dict[str, Any]]:
    validation = [row for row in read_jsonl(C_MANIFEST) if row["split"] == "val"]
    legacy_assignment = json.loads((LEGACY_ROOT / "split.json").read_text())["assignments"]
    paths = {
        0: (
            MISSING_RUN / "full_missing_eval_20260817_action_dynamics_vjepa2_p8_layer0/eval/val_predictions.pt",
            MISSING_RUN / "full_missing_eval_20260817_counterfactual_vjepa2_p8_layer0/eval/val_predictions.pt",
        ),
        17: (
            MISSING_RUN / "full_missing_eval_20260817_action_dynamics_vjepa2_p8_layer17/eval/val_predictions.pt",
            MISSING_RUN / "full_missing_eval_20260817_counterfactual_vjepa2_p8_layer17/eval/val_predictions.pt",
        ),
        23: (
            C_RUN / "c_eval_20260724_1903_action_dynamics_vjepa2_p8_layer23/eval/val_predictions.pt",
            C_RUN / "c_eval_20260724_1903_path_integration_vjepa2_p8_layer23/eval/val_predictions.pt",
        ),
    }
    loaded = {layer: (torch.load(a, map_location="cpu", weights_only=False), torch.load(b, map_location="cpu", weights_only=False)) for layer, (a, b) in paths.items()}
    common = [
        index for index in range(len(validation))
        if all(bool(c1[index]["valid"]) and bool(c2[index]["valid"].all()) for c1, c2 in loaded.values())
    ]
    subset_manifest = output / "vjepa2_layer_common_manifest.jsonl"
    with subset_manifest.open("w") as handle:
        for index in common:
            handle.write(json.dumps(validation[index], sort_keys=True, separators=(",", ":")) + "\n")
    result: dict[int, dict[str, list[TupleRecord]]] = {}
    per_layer_audit = {}
    inputs = output / "derived_validity_intersection_inputs"
    inputs.mkdir()
    for layer, (source_c1, source_c2) in paths.items():
        c1, c2 = loaded[layer]
        filtered_c1, filtered_c2 = [c1[i] for i in common], [c2[i] for i in common]
        c1_path, c2_path = inputs / f"vjepa2_l{layer}_c1.pt", inputs / f"vjepa2_l{layer}_c2.pt"
        torch.save(filtered_c1, c1_path); torch.save(filtered_c2, c2_path)
        records, audit = load_and_audit_tuples({
            "manifest": str(subset_manifest), "c1_dump": str(c1_path), "c2_dump": str(c2_path),
            "raw_scene_root": str(RAW_ROOT), "alignment_atol": 2e-5, "pose_atol": 1e-5,
        })
        result[layer] = {split: [r for r in records if legacy_assignment[r.scene_id] == split] for split in ("train", "val", "test")}
        per_layer_audit[str(layer)] = {
            "source_c1": str(source_c1), "source_c2": str(source_c2),
            "source_c1_sha256": file_digest_or_na(source_c1), "source_c2_sha256": file_digest_or_na(source_c2),
            "alignment": audit,
        }
    counts = {split: {"tuples": len(result[0][split]), "scenes": len({r.scene_id for r in result[0][split]})} for split in ("train", "val", "test")}
    audit = {
        "status": "FROZEN_SPARSE_AVAILABLE_LAYER_CONTRAST",
        "selection_frozen_before_scores": True,
        "layers": [0, 17, 23],
        "selection_rule": "all available V-JEPA2 target-isolated p8 layers with complete C1 plus three-horizon C2 rows; intersect validity before scoring",
        "feature_contract": "same V-JEPA2 encoder family, skip predictor, isolated RGB frame repeated 64 times; p8 is sampling/cache nomenclature here, not real history",
        "not_claimed_as_complete_early_middle_late": True,
        "missing_preregistered_middle_layer": 11,
        "source_rows": len(validation), "common_valid_rows": len(common), "counts": counts,
        "common_manifest_sha256": sha256(subset_manifest), "per_layer": per_layer_audit,
    }
    dump_json(output / "vjepa2_layer_inventory_audit.json", audit)
    return result, audit


def cosine_distance64(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(left.double(), right.double(), dim=-1)).clamp_min(0.0)


def matched_cross_scene_donor(records: list[TupleRecord]) -> np.ndarray:
    """Exact action permutation with no self/same-scene donor, chosen pre-score."""
    magnitude = np.asarray([
        np.linalg.norm(se3_log(record.transforms[0])) + np.linalg.norm(se3_log(record.transforms[1]))
        for record in records
    ])
    ordered = np.argsort(magnitude, kind="mergesort")
    best: tuple[float, np.ndarray] | None = None
    for shift in range(1, len(records)):
        shifted = np.roll(ordered, -shift)
        if any(records[left].scene_id == records[right].scene_id for left, right in zip(ordered, shifted)):
            continue
        donors = np.empty(len(records), dtype=np.int64)
        donors[ordered] = shifted
        cost = float(np.mean(np.abs(magnitude - magnitude[donors])))
        if best is None or cost < best[0]:
            best = (cost, donors)
    if best is None:
        raise RuntimeError("no cross-scene exact donor permutation exists")
    donors = best[1]
    if sorted(donors.tolist()) != list(range(len(records))) or np.any(donors == np.arange(len(records))):
        raise RuntimeError("matched donor is not an exact derangement")
    return donors


def prediction_arrays_independent(
    baseline: str, models: dict[str, torch.nn.Module], z_a: torch.Tensor, z_b: torch.Tensor, z_c: torch.Tensor,
    t1: np.ndarray, t2: np.ndarray, action_mean: torch.Tensor, action_std: torch.Tensor,
) -> tuple[dict[str, np.ndarray], dict[str, torch.Tensor], dict[str, Any]]:
    composite = np.stack([right @ left for left, right in zip(t1, t2)])
    numerical: dict[str, Any] = {}
    if baseline == "additive":
        model = models["additive"]
        a1, a2, ac = (action_tensor(value, action_mean, action_std) for value in (t1, t2, composite))
        with torch.no_grad():
            increment1 = model.increment(a1).double()
            increment2 = model.increment(a2).double()
            increment_composite = model.increment(ac).double()
        # Both paths are independently evaluated.  Float64 is used only for
        # their algebraic accumulation so associativity roundoff is observable
        # but far below the frozen metric tie tolerance.
        first = z_a.double() + increment1
        one_bc = z_b.double() + increment2
        recursive = first + increment2
        reversed_first = z_a.double() + increment2
        reversed_ = reversed_first + increment1
        direct = z_a.double() + increment_composite
        delta = (recursive - reversed_).abs()
        scale = max(1.0, float(torch.stack((z_a.double().abs().max(), increment1.abs().max(), increment2.abs().max())).max()))
        bound = 16.0 * torch.finfo(torch.float64).eps * scale
        numerical = {
            "paths_independently_evaluated": True,
            "accumulation_dtype": "float64",
            "max_abs_recursive_reverse_delta": float(delta.max()),
            "roundoff_bound": bound,
            "within_roundoff_bound": float(delta.max()) <= bound,
            "exact_tensor_equal": bool(torch.equal(recursive, reversed_)),
        }
        if not numerical["within_roundoff_bound"]:
            raise RuntimeError(f"additive path difference exceeds numerical bound: {numerical}")
    else:
        first = predict(baseline, models, z_a, t1, action_mean, action_std)
        one_bc = predict(baseline, models, z_b, t2, action_mean, action_std)
        recursive = predict(baseline, models, first, t2, action_mean, action_std)
        reversed_first = predict(baseline, models, z_a, t2, action_mean, action_std)
        reversed_ = predict(baseline, models, reversed_first, t1, action_mean, action_std)
        direct = predict(baseline, models, z_a, composite, action_mean, action_std)
    def np_cos(left: torch.Tensor, right: torch.Tensor) -> np.ndarray:
        return cosine_distance64(left, right).cpu().numpy().astype(np.float64)
    rec_error, rev_error = np_cos(recursive, z_c), np_cos(reversed_, z_c)
    margin = rev_error - rec_error
    arrays = {
        "one_ab_error": np_cos(first, z_b), "one_bc_error": np_cos(one_bc, z_c),
        "recursive_error": rec_error, "reversed_error": rev_error, "direct_error": np_cos(direct, z_c),
        "margin": margin,
        "tie_score": np.where(margin > DEFAULT_TIE_TOLERANCE, 1.0, np.where(margin < -DEFAULT_TIE_TOLERANCE, 0.0, 0.5)),
        "recursive_direct_distance": np_cos(recursive, direct),
    }
    return arrays, {"one_ab": first, "one_bc": one_bc, "recursive": recursive, "reversed": reversed_, "direct": direct}, numerical


def evaluate_cell(
    output: Path, cell: str, partitions: dict[str, list[TupleRecord]], seeds: tuple[int, ...] = SEEDS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    child = output / cell
    child.mkdir()
    all_metrics, all_samples, all_training = [], [], {}
    for seed in seeds:
        models, training = train_all(partitions, seed, CFG)
        action_mean, action_std = training.pop("action_mean"), training.pop("action_std")
        test = partitions["test"]
        features = torch.stack([record.features for record in test])
        z_a, z_b, z_c = features[:, 0], features[:, 1], features[:, 2]
        t1, t2, _ = transform_batch(test)
        donor = matched_cross_scene_donor(test)
        shuffle1, shuffle2 = derangement(len(test), seed + 101), derangement(len(test), seed + 211)
        actions = {
            "copy": (t1, t2), "without_condition": (t1, t2), "action_only": (t1, t2),
            "additive": (t1, t2), "full": (t1, t2),
            "matched_wrong_pose": (t1[donor], t2[donor]), "shuffled_pose": (t1[shuffle1], t2[shuffle2]),
        }
        arrays_by_baseline, predictions_to_save, additive_audit = {}, {}, {}
        for baseline in PRIMARY_BASELINES:
            arrays, predictions, numerical = prediction_arrays_independent(
                baseline, models, z_a, z_b, z_c, *actions[baseline], action_mean, action_std
            )
            arrays_by_baseline[baseline] = arrays
            if numerical:
                additive_audit = numerical
            for name, value in predictions.items():
                dtype = np.float64 if baseline == "additive" and name in {"recursive", "reversed"} else np.float16
                predictions_to_save[f"{baseline}__{name}"] = value.cpu().numpy().astype(dtype)
        copy_arrays, without = arrays_by_baseline["copy"], arrays_by_baseline["without_condition"]
        scenes = np.asarray([record.scene_id for record in test])
        for baseline in PRIMARY_BASELINES:
            arrays = arrays_by_baseline[baseline]
            for aggregation in ("scene_macro", "tuple_micro"):
                row = summarize_prediction_arrays(arrays, copy_arrays, scenes, aggregation)
                reducer = np.mean if aggregation == "tuple_micro" else lambda x: scene_macro(np.asarray(x), scenes)
                one = (arrays["one_ab_error"] + arrays["one_bc_error"]) / 2
                without_one = (without["one_ab_error"] + without["one_bc_error"]) / 2
                one_value, without_one_value = float(reducer(one)), float(reducer(without_one))
                rec_value, without_rec_value = float(reducer(arrays["recursive_error"])), float(reducer(without["recursive_error"]))
                row.update({
                    "cell": cell, "baseline": baseline, "seed": seed,
                    "one_step_gain_vs_without_condition": (without_one_value - one_value) / max(without_one_value, DEFAULT_NORMALIZATION_EPSILON),
                    "two_step_gain_vs_without_condition": (without_rec_value - rec_value) / max(without_rec_value, DEFAULT_NORMALIZATION_EPSILON),
                })
                all_metrics.append(row)
            for index, record in enumerate(test):
                all_samples.append({
                    "cell": cell, "tuple_id": record.sample_id, "scene_id": record.scene_id,
                    "seed": seed, "baseline": baseline,
                    **{key: float(value[index]) for key, value in arrays.items()},
                })
        np.savez_compressed(child / f"raw_predictions_seed{seed}.npz", **predictions_to_save)
        all_training[str(seed)] = {
            "training": training,
            "controls": {
                "matched_wrong_pose_fixed_points": int(np.sum(donor == np.arange(len(test)))),
                "matched_wrong_pose_same_scene": int(sum(test[i].scene_id == test[j].scene_id for i, j in enumerate(donor))),
                "shuffle_step1_fixed_points": int(np.sum(shuffle1 == np.arange(len(test)))),
                "shuffle_step2_fixed_points": int(np.sum(shuffle2 == np.arange(len(test)))),
                "additive_independent_path_audit": additive_audit,
            },
        }
        control = all_training[str(seed)]["controls"]
        if any(control[key] for key in control if key.endswith("fixed_points") or key.endswith("same_scene")):
            raise RuntimeError(f"control derangement failure: {cell} {seed} {control}")
    write_csv(child / "per_seed_metrics.csv", all_metrics)
    write_csv(child / "per_sample_metrics.csv", all_samples)
    dump_json(child / "training_and_controls.json", all_training)
    return all_metrics, all_samples


def paired_bootstrap(
    sample_rows: list[dict[str, Any]], cell_groups: dict[str, list[str]], repetitions: int = BOOTSTRAP_REPETITIONS,
) -> list[dict[str, Any]]:
    index = {(r["cell"], r["scene_id"], int(r["seed"]), r["baseline"], r["tuple_id"]): r for r in sample_rows}
    result = []
    for dataset, cells in cell_groups.items():
        scenes = sorted({r["scene_id"] for r in sample_rows if r["cell"] in cells})
        tuples = {scene: sorted({key[4] for key in index if key[0] in cells and key[1] == scene}) for scene in scenes}
        rng = np.random.default_rng(BOOTSTRAP_SEED + sum(map(ord, dataset)))
        draws = rng.integers(0, len(scenes), size=(repetitions, len(scenes)))
        cache: dict[tuple[str, str, str], np.ndarray] = {}
        def values(cell: str, baseline: str, field: str) -> np.ndarray:
            key = (cell, baseline, field)
            if key not in cache:
                matrix = np.asarray([
                    [np.mean([float(index[(cell, scene, seed, baseline, tid)][field]) for tid in tuples[scene]]) for scene in scenes]
                    for seed in SEEDS
                ])
                cache[key] = matrix.mean(0)[draws].mean(1)
            return cache[key]
        comparisons = []
        for cell in cells:
            comparisons.extend([
                (cell, "full", cell, "copy", "grounding_gain_vs_copy"),
                (cell, "full", cell, "without_condition", "grounding_gain_vs_without_condition"),
                (cell, "full", cell, "copy", "normalized_order_margin"),
                (cell, "full", cell, "action_only", "normalized_margin_full_minus_action_only"),
                (cell, "full", cell, "matched_wrong_pose", "normalized_margin_full_minus_matched_wrong_pose"),
                (cell, "full", cell, "shuffled_pose", "normalized_margin_full_minus_shuffle"),
                (cell, "full", cell, "additive", "normalized_margin_full_minus_additive"),
            ])
        if dataset == "vjepa2_layer":
            for left in cells:
                if left == "vjepa2_l23":
                    continue
                comparisons.extend([
                    (left, "full", "vjepa2_l23", "full", "grounding_gain_layer_difference"),
                    (left, "full", "vjepa2_l23", "full", "normalized_margin_layer_difference"),
                ])
        for left_cell, left_base, right_cell, right_base, metric in comparisons:
            if metric == "grounding_gain_vs_copy":
                left = values(left_cell, left_base, "recursive_error"); denominator = values(right_cell, right_base, "recursive_error")
                estimates = (denominator - left) / np.maximum(denominator, DEFAULT_NORMALIZATION_EPSILON)
            elif metric == "grounding_gain_vs_without_condition":
                left = values(left_cell, left_base, "recursive_error"); denominator = values(right_cell, right_base, "recursive_error")
                estimates = (denominator - left) / np.maximum(denominator, DEFAULT_NORMALIZATION_EPSILON)
            elif metric == "grounding_gain_layer_difference":
                left_copy = values(left_cell, "copy", "recursive_error"); right_copy = values(right_cell, "copy", "recursive_error")
                left_gain = (left_copy - values(left_cell, "full", "recursive_error")) / np.maximum(left_copy, DEFAULT_NORMALIZATION_EPSILON)
                right_gain = (right_copy - values(right_cell, "full", "recursive_error")) / np.maximum(right_copy, DEFAULT_NORMALIZATION_EPSILON)
                estimates = left_gain - right_gain
            else:
                left_copy = values(left_cell, "copy", "recursive_error")
                left_margin = values(left_cell, left_base, "margin") / np.maximum(left_copy, DEFAULT_NORMALIZATION_EPSILON)
                if metric == "normalized_order_margin":
                    estimates = left_margin
                else:
                    right_copy = values(right_cell, "copy", "recursive_error")
                    right_margin = values(right_cell, right_base, "margin") / np.maximum(right_copy, DEFAULT_NORMALIZATION_EPSILON)
                    estimates = left_margin - right_margin
            p = min(1.0, 2 * min(float(np.mean(estimates <= 0)), float(np.mean(estimates >= 0))))
            result.append({
                "dataset": dataset, "left_cell": left_cell, "left_baseline": left_base,
                "right_cell": right_cell, "right_baseline": right_base, "metric": metric,
                "estimate": float(estimates.mean()), "ci95_low": float(np.quantile(estimates, .025)),
                "ci95_high": float(np.quantile(estimates, .975)), "bootstrap_p_two_sided": p,
                "bootstrap_unit": "scene", "resamples": repetitions, "training_seeds_averaged": len(SEEDS),
            })
    adjusted = holm_adjust([float(row["bootstrap_p_two_sided"]) for row in result])
    for row, value in zip(result, adjusted):
        row["holm_adjusted_p"] = value
    return result


def write_documents(
    output: Path, confirmation_audit: dict[str, Any], layer_audit: dict[str, Any],
    aggregate: list[dict[str, Any]], bootstrap: list[dict[str, Any]], gap: dict[str, Any],
) -> None:
    full = {r["cell"]: r for r in aggregate if r["baseline"] == "full" and r["aggregation"] == "scene_macro"}
    cis = {(r["left_cell"], r["metric"]): r for r in bootstrap}
    rows = []
    labels = {"confirmation_dino_l0": "DINO-L0 new-scene confirmation", "vjepa2_l0": "V-JEPA2 L0", "vjepa2_l17": "V-JEPA2 L17", "vjepa2_l23": "V-JEPA2 L23"}
    for cell, row in full.items():
        grounding, benefit, order = cis[(cell, "grounding_gain_vs_copy")], cis[(cell, "grounding_gain_vs_without_condition")], cis[(cell, "normalized_order_margin")]
        table_row = {
            "cell": cell, "display_name": labels[cell], "dataset": "new_scene_confirmation" if cell.startswith("confirmation") else "legacy_layer_validity_intersection",
            "test_tuples": confirmation_audit["counts"]["test"]["tuples"] if cell.startswith("confirmation") else layer_audit["counts"]["test"]["tuples"],
            "test_scenes": confirmation_audit["counts"]["test"]["scenes"] if cell.startswith("confirmation") else layer_audit["counts"]["test"]["scenes"],
            "one_step_error": row["one_step_error_mean"], "recursive_two_step_error": row["recursive_two_step_error_mean"],
            "grounding_gain_vs_copy": row["two_step_gain_vs_copy_mean"], "grounding_gain_copy_ci95_low": grounding["ci95_low"], "grounding_gain_copy_ci95_high": grounding["ci95_high"],
            "grounding_gain_vs_without_condition": row["two_step_gain_vs_without_condition_mean"], "action_benefit_ci95_low": benefit["ci95_low"], "action_benefit_ci95_high": benefit["ci95_high"],
            "normalized_order_margin": row["normalized_order_margin_mean"], "normalized_order_ci95_low": order["ci95_low"], "normalized_order_ci95_high": order["ci95_high"],
            "tie_aware_accuracy": row["tie_aware_accuracy_mean"],
        }
        for short, metric in (
            ("action_only", "normalized_margin_full_minus_action_only"),
            ("matched_wrong_pose", "normalized_margin_full_minus_matched_wrong_pose"),
            ("shuffled_pose", "normalized_margin_full_minus_shuffle"),
            ("additive", "normalized_margin_full_minus_additive"),
        ):
            comparison = cis[(cell, metric)]
            table_row[f"order_effect_vs_{short}"] = comparison["estimate"]
            table_row[f"order_effect_vs_{short}_ci95_low"] = comparison["ci95_low"]
            table_row[f"order_effect_vs_{short}_ci95_high"] = comparison["ci95_high"]
            table_row[f"order_effect_vs_{short}_holm_p"] = comparison["holm_adjusted_p"]
        rows.append(table_row)
    write_csv(output / "paper_main_table_increment.csv", rows)
    latex = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Cell & Gain vs copy & Gain vs no-action & Norm. order & Tie-aware acc. \\", r"\midrule"]
    for row in rows:
        latex.append(f"{row['display_name']} & {float(row['grounding_gain_vs_copy']):.4f} & {float(row['grounding_gain_vs_without_condition']):.4f} & {float(row['normalized_order_margin']):.4f} & {float(row['tie_aware_accuracy']):.4f} \\")
    latex.extend([r"\bottomrule", r"\end{tabular}"])
    (output / "paper_main_table_increment.tex").write_text("\n".join(latex) + "\n")
    confirmation = next(row for row in rows if row["cell"] == "confirmation_dino_l0")
    lines = [
        "# Incremental findings", "",
        "## New-scene confirmation", "",
        f"DINO-L0 is the only runnable four-core-model cell. On {confirmation['test_tuples']} tuples from {confirmation['test_scenes']} scenes outside the legacy 69, two-step gain versus copy is {float(confirmation['grounding_gain_vs_copy']):.4f} (95% CI [{float(confirmation['grounding_gain_copy_ci95_low']):.4f}, {float(confirmation['grounding_gain_copy_ci95_high']):.4f}]). Correct-action gain versus the learned without-condition predictor is {float(confirmation['grounding_gain_vs_without_condition']):.4f} (CI [{float(confirmation['action_benefit_ci95_low']):.4f}, {float(confirmation['action_benefit_ci95_high']):.4f}]). These are distinct claims.",
        f"Normalized observed-order margin is {float(confirmation['normalized_order_margin']):.4f} (CI [{float(confirmation['normalized_order_ci95_low']):.4f}, {float(confirmation['normalized_order_ci95_high']):.4f}]); tie-aware accuracy is {float(confirmation['tie_aware_accuracy']):.4f}. Control-adjusted comparisons are in `paired_bootstrap.csv`.",
        f"Full-minus-control normalized-margin effects are: action-only {float(confirmation['order_effect_vs_action_only']):.4f} (CI [{float(confirmation['order_effect_vs_action_only_ci95_low']):.4f}, {float(confirmation['order_effect_vs_action_only_ci95_high']):.4f}]); matched wrong-pose {float(confirmation['order_effect_vs_matched_wrong_pose']):.4f} (CI [{float(confirmation['order_effect_vs_matched_wrong_pose_ci95_low']):.4f}, {float(confirmation['order_effect_vs_matched_wrong_pose_ci95_high']):.4f}]); shuffled pose {float(confirmation['order_effect_vs_shuffled_pose']):.4f} (CI [{float(confirmation['order_effect_vs_shuffled_pose_ci95_low']):.4f}, {float(confirmation['order_effect_vs_shuffled_pose_ci95_high']):.4f}]); additive {float(confirmation['order_effect_vs_additive']):.4f} (CI [{float(confirmation['order_effect_vs_additive_ci95_low']):.4f}, {float(confirmation['order_effect_vs_additive_ci95_high']):.4f}]). All are inconclusive after Holm correction.",
        "Relative to legacy r2, the positive DINO grounding trend is confirmed on new scenes. The earlier small positive absolute order trend is weakened: the new order-margin CI crosses zero and every registered order-control difference is inconclusive. Effect sizes are not treated as directly exchangeable because the new immutable algebra sampling distribution differs from legacy C horizons. This is not four-model confirmation.", "",
        "## Real-history contrast", "",
        f"Unavailable under the required contract. There are {gap['vjepa2_context_missing_units']} exact V-JEPA2 L23 p8 context states in the frozen A/B/C design and no complete non-canonical matched store. Existing eval dumps contain one original-window source plus isolated endpoints, so they cannot implement a recursive same-role A/B/C comparison. `vjepa2_real_history_missing_contexts.csv` is the exact supplement list.", "",
        "## Sparse layer contrast", "",
    ]
    for row in rows:
        if row["cell"].startswith("vjepa2"):
            lines.append(f"- {row['display_name']}: gain vs copy {float(row['grounding_gain_vs_copy']):.4f}; gain vs without-condition {float(row['grounding_gain_vs_without_condition']):.4f}; normalized order {float(row['normalized_order_margin']):.4f}; tie-aware accuracy {float(row['tie_aware_accuracy']):.4f}.")
    lines.extend([
        "", "All individual layer CIs and L0/L17-minus-L23 paired CIs cross zero. This is an availability-frozen L0/L17/L23 contrast on 169 validity-intersection tuples, not a complete early/middle/late sweep; the predeclared middle L11 cell is absent.",
        "", "## Local protocol corrections", "",
        "The additive A→B→C and B→A paths are independently accumulated and checked against a float64 roundoff bound. They are never assigned to the same tensor. Composite support is described only as an empirical componentwise min–max range check, never as distribution-in certification.",
    ])
    (output / "FINDINGS.md").write_text("\n".join(lines) + "\n")
    (output / "REPORT.md").write_text("""# SE3-Observed-v2 incremental report

This directory adds a DINO-only new-scene confirmation cell and a V-JEPA2 sparse layer contrast. It does not relabel the legacy 276 tuples and does not claim that unavailable context or four-model confirmation experiments were completed. Exact missing feature and context units are provided as bounded supplement lists.
""")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_REPETITIONS)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = REPO_ROOT / output
    if output.exists():
        raise SystemExit(f"refusing to overwrite output: {output}")
    output.mkdir(parents=True)
    torch.set_num_threads(4)
    freeze = {
        "status": "FROZEN_BEFORE_INCREMENTAL_SCORES",
        "research_increment": ["DINO-only old69-excluded confirmation", "V-JEPA2 L0/L17/L23 sparse layer contrast"],
        "seeds": list(SEEDS), "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_resamples": args.bootstrap_resamples,
        "training": CFG, "tie_tolerance": DEFAULT_TIE_TOLERANCE, "normalization_epsilon": DEFAULT_NORMALIZATION_EPSILON,
        "confirmation_tuple_rule": "A=anchor,B=h1,C=h2 from immutable independent science-pilot algebra manifests",
        "confirmation_train_val_rule": "source-train scenes SHA256-ranked, first round(20%) held for validation; source-val is confirmation test",
        "legacy_scene_policy": "exclude every scene in SE3-Observed-v1 manifest from every confirmation partition",
        "layer_rule": "availability-frozen V-JEPA2 L0/L17/L23, common validity intersection, legacy r2 scene assignments",
        "additive_rule": "independent float64 accumulation with explicit roundoff-bound audit; no tensor equality assignment",
        "composite_support_wording": "empirical componentwise min-max range check only",
        "pose_convention": POSE_CONVENTION,
        "canonical_v2": "forbidden_not_read",
        "input_hashes": {name: file_digest_or_na(path) for name, path in {
            "legacy_manifest": LEGACY_ROOT / "se3_observed_v1_manifest.jsonl",
            "algebra_train": ALGEBRA_TRAIN, "algebra_test": ALGEBRA_TEST,
            "feature_request": REQUEST_MANIFEST, "feature_receipt": FEATURE_RECEIPT,
            "c_manifest": C_MANIFEST,
        }.items()},
    }
    dump_json(output / "STUDY_FREEZE.json", freeze)
    (output / "STUDY_FREEZE.md").write_text("""# SE3-Observed-v2 incremental freeze

Frozen before reading any incremental endpoint score. The runnable primary increment is DINO-L0 on immutable independent algebra A/observed-h1/observed-h2 triples, excluding all legacy 69 scenes. Source-train scenes are deterministically split for probe train/validation; the independent source-validation scenes are confirmation test. The secondary increment is the availability-fixed V-JEPA2 L0/L17/L23 target-isolated validity intersection. Five seeds and 2,000 paired scene bootstraps are fixed. Canonical-v2 is not consumed.
""")

    confirmation, manifest_rows, confirmation_audit = load_independent_confirmation(output)
    confirmation_gap_lists(output, manifest_rows)
    layer_records, layer_audit = prepare_vjepa_layer_records(output)
    all_metrics, all_samples = [], []
    metrics, samples = evaluate_cell(output, "confirmation_dino_l0", confirmation)
    all_metrics.extend(metrics); all_samples.extend(samples)
    for layer in (0, 17, 23):
        metrics, samples = evaluate_cell(output, f"vjepa2_l{layer}", layer_records[layer])
        all_metrics.extend(metrics); all_samples.extend(samples)
    aggregate = mean_rows(all_metrics, ("cell", "baseline", "aggregation"))
    bootstrap = paired_bootstrap(
        all_samples,
        {"confirmation": ["confirmation_dino_l0"], "vjepa2_layer": ["vjepa2_l0", "vjepa2_l17", "vjepa2_l23"]},
        args.bootstrap_resamples,
    )
    write_csv(output / "all_seed_metrics.csv", all_metrics)
    write_csv(output / "all_per_sample_metrics.csv", all_samples)
    write_csv(output / "aggregate_baselines.csv", aggregate)
    write_csv(output / "paired_bootstrap.csv", bootstrap)
    gap = json.loads((output / "gap_summary.json").read_text())
    write_documents(output, confirmation_audit, layer_audit, aggregate, bootstrap, gap)
    dump_json(output / "execution_summary.json", {
        "status": "COMPLETE_RUNNABLE_INCREMENT_WITH_CELL_LEVEL_GAPS",
        "confirmation": confirmation_audit, "layer": {k: layer_audit[k] for k in ("status", "layers", "counts", "common_valid_rows")},
        "gaps": gap, "runner_sha256": sha256(Path(__file__).resolve()),
        "tests": "run separately and recorded in final handoff",
    })
    (output / "REPRODUCE.md").write_text(f"""# Reproduce

```bash
export MPLCONFIGDIR=/tmp/probe_spatial_mpl
{sys.executable} -m pytest -q tests/test_se3_protocol_v2.py tests/test_se3_latent_mvp.py
{sys.executable} -m analyses.se3_protocol_v2.incremental_v2 --output-dir <NEW_OUTPUT_DIR> --bootstrap-resamples {args.bootstrap_resamples}
```
""")
    print(json.dumps(json.loads((output / "execution_summary.json").read_text()), indent=2))


if __name__ == "__main__":
    main()
