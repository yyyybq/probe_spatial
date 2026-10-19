#!/usr/bin/env python3
"""Evaluation-only P16 validity repair with fail-closed evidence publication.

This module is deliberately reachable only through run_probe_validity.py's
explicit ``--p16-repair`` entry.  It never trains a probe, fits a prior, or
extracts a feature.  A cell is published only after its historical replay gate
passes; failed attempts are preserved beside the requested cell directory.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import statistics
import time
import traceback
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

from analyses.probe_validity import run_probe_validity as base


C1_MANIFEST_SHA256 = "875e6f106b85dd6ca5e54707af98425772bf6596d198e9ae478f8777313e6aa0"
C1_REFERENCE_KEYS = (
    "mean_cos", "global_R@1", "global_R@5", "mean_rank",
    "no_action_mean_cos", "no_action_global_R@1", "no_action_global_R@5",
    "no_action_mean_rank", "shuffled_action_mean_cos",
    "shuffled_action_global_R@1", "shuffled_action_global_R@5",
    "shuffled_action_mean_rank", "last_observation_mean_cos",
    "last_observation_global_R@1", "last_observation_global_R@5",
    "last_observation_mean_rank",
)
B2_REFERENCE_KEYS = (
    "mean_ang_err_deg", "mean_az_err_deg", "mean_el_err_deg", "top1",
    "top3", "mean_log_dist_err",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _csv_gz(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            import io
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(text, fieldnames=fields)
                writer.writeheader()
                writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def _stable_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_probe(task: str, model: str, checkpoint: Path, expected_sha: str, device):
    import torch
    from vidfm3d.models.components.probe_action_dynamics import ActionDynamicsProbe
    from vidfm3d.models.components.probe_ego_belief_v2 import EgoBeliefProbeV2

    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"CHECKPOINT_NOT_FOUND: {checkpoint}")
    observed_sha = _sha256(checkpoint)
    if observed_sha != expected_sha.lower():
        raise RuntimeError(
            f"CHECKPOINT_SHA256_MISMATCH: expected={expected_sha} observed={observed_sha}"
        )
    spec = base.MODEL_SPECS[model]
    torch.manual_seed(base.PHASE2_SEED)
    if task == "c1":
        probe = ActionDynamicsProbe(
            in_channels=spec["channels"], action_dim=9, hidden_dim=512,
            num_layers=2, num_heads=8, max_seq_len=64,
        )
        allowed = {"constant_context", "constant_action"}
    else:
        probe = EgoBeliefProbeV2(
            in_channels=spec["channels"], decoder_type="transformer",
            num_frames=16, hidden_dim=512, num_layers=4, num_heads=8,
            max_seq_len=64, max_h=spec["b2_max_h"], max_w=spec["b2_max_w"],
            n_az_bins=16, n_el_bins=8, patch_pool=2,
        )
        allowed = {"constant_context"}
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "state_dict" not in payload:
        raise RuntimeError("CHECKPOINT_HAS_NO_STATE_DICT")
    state = {
        key[len("probe."):]: value for key, value in payload["state_dict"].items()
        if key.startswith("probe.")
    }
    if not state:
        raise RuntimeError("CHECKPOINT_HAS_NO_PROBE_KEYS")
    incompatible = probe.load_state_dict(state, strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if unexpected or not missing.issubset(allowed):
        raise RuntimeError(
            f"CHECKPOINT_CONTRACT_MISMATCH missing={sorted(missing)} "
            f"allowed={sorted(allowed)} unexpected={sorted(unexpected)}"
        )
    receipt = {
        "path": str(checkpoint), "sha256": observed_sha,
        "size_bytes": checkpoint.stat().st_size,
        "epoch": int(payload.get("epoch", -1)),
        "global_step": int(payload.get("global_step", -1)),
        "probe_state_key_count": len(state),
        "missing_keys": sorted(missing), "allowed_missing_keys": sorted(allowed),
        "unexpected_keys": sorted(unexpected),
        "strict_contract": "unexpected empty; missing subset of explicit unused-key allowlist",
    }
    return probe.float().to(device).eval(), receipt


def _unused_parameter_invariance_c1(probe, bundle, device) -> Dict[str, object]:
    import torch
    context = bundle["contexts"][0].unsqueeze(0).unsqueeze(2).unsqueeze(2).float().to(device)
    action = bundle["actions"][0:1].float().to(device)
    with torch.no_grad():
        before = probe(context, action)
        saved_context = probe.constant_context.detach().clone()
        saved_action = probe.constant_action.detach().clone()
        probe.constant_context.fill_(113.0)
        probe.constant_action.fill_(-127.0)
        after = probe(context, action)
        probe.constant_context.copy_(saved_context)
        probe.constant_action.copy_(saved_action)
    difference = float((before - after).abs().max().cpu())
    if probe.conditioning != "full" or difference != 0.0:
        raise RuntimeError(f"C1_UNUSED_PARAMETER_INVARIANCE_FAILED: {difference}")
    return {
        "conditioning": probe.conditioning,
        "mutated_parameters": ["constant_context", "constant_action"],
        "max_abs_output_difference": difference,
        "status": "PASS",
    }


def _unused_parameter_invariance_b2(probe, sample, device) -> Dict[str, object]:
    import torch
    with torch.no_grad():
        before = probe(
            sample["feature"].unsqueeze(0).float().to(device),
            sample["query"].unsqueeze(0).float().to(device),
        )
        saved = probe.constant_context.detach().clone()
        probe.constant_context.fill_(113.0)
        after = probe(
            sample["feature"].unsqueeze(0).float().to(device),
            sample["query"].unsqueeze(0).float().to(device),
        )
        probe.constant_context.copy_(saved)
    difference = max(
        float((before["logits"] - after["logits"]).abs().max().cpu()),
        float((before["log_dist"] - after["log_dist"]).abs().max().cpu()),
    )
    if probe.context_mode != "full" or difference != 0.0:
        raise RuntimeError(f"B2_UNUSED_PARAMETER_INVARIANCE_FAILED: {difference}")
    return {
        "context_mode": probe.context_mode,
        "mutated_parameters": ["constant_context"],
        "max_abs_output_difference": difference,
        "status": "PASS",
    }


def _reference_gate(reference: Dict[str, object], reproduced: Dict[str, object], keys, tol=1e-5):
    differences = {key: float(reproduced[key]) - float(reference[key]) for key in keys}
    failures = {key: value for key, value in differences.items() if abs(value) > tol}
    n_ok = int(reproduced["n"]) == int(reference["n"])
    result = {
        "status": "PASS" if n_ok and not failures else "FAIL",
        "tolerance": tol, "n_matches": n_ok,
        "reference": {"n": reference["n"], **{key: reference[key] for key in keys}},
        "reproduced": {"n": reproduced["n"], **{key: reproduced[key] for key in keys}},
        "signed_differences": differences,
        "max_absolute_difference": max(abs(value) for value in differences.values()),
        "failures": failures,
    }
    if result["status"] != "PASS":
        raise RuntimeError(f"REFERENCE_REPLAY_MISMATCH: {json.dumps(result, sort_keys=True)}")
    return result


def _global_sample_rows(bundle, predictions, physical) -> List[Dict[str, object]]:
    rows = []
    targets = bundle["targets"]
    metadata = bundle["metadata"]
    matched = physical["matched_wrong_indices"]
    roll = []
    for start in range(0, len(metadata), 4):
        group = list(range(start, min(start + 4, len(metadata))))
        roll.extend([group[-1], *group[:-1]])
    sources = {
        "FULL": list(range(len(metadata))),
        "ZERO_ACTION": [None] * len(metadata),
        "SHUFFLED_ACTION": roll,
        "MATCHED_WRONG": matched,
        "COPY": list(range(len(metadata))),
    }
    for index, item in enumerate(metadata):
        for condition, prediction in predictions.items():
            metrics = base._rank_metrics(prediction[index], targets, index)
            source = sources[condition][index]
            rows.append({
                "sample_id": item["manifest_sample_id"], "scene_id": item["scene_id"],
                "trajectory_id": item["trajectory_id"], "window_id": item["window_id"],
                "condition": condition, "candidate_pool": "GLOBAL_ALL_TARGETS",
                "candidate_count": len(metadata), "target_frame_id": item["target_frame_id"],
                "context_frame_ids": json.dumps(item["context_frame_ids"], separators=(",", ":")),
                "action_source_sample_id": (
                    metadata[source]["manifest_sample_id"] if source is not None else "ZERO_VECTOR"
                ),
                **metrics,
            })
    return rows


def _c1(stage: Path, model: str, checkpoint: Path, expected_sha: str,
        reference_path: Path, device) -> Dict[str, object]:
    import torch

    spec = base.MODEL_SPECS[model]
    reference = json.loads(reference_path.read_text())
    if Path(reference["ckpt_path"]).resolve() != checkpoint.resolve():
        raise RuntimeError("REFERENCE_CHECKPOINT_BINDING_MISMATCH")
    manifest = base.C1_MANIFESTS[16]
    manifest_sha = _sha256(manifest)
    if manifest_sha != C1_MANIFEST_SHA256:
        raise RuntimeError(f"C1_MANIFEST_SHA256_MISMATCH: {manifest_sha}")
    probe, checkpoint_receipt = _load_probe("c1", model, checkpoint, expected_sha, device)
    bundle = base._collect_c1_bundle(model, 16, 1)
    if bundle["manifest_sha256"] != manifest_sha:
        raise RuntimeError("C1_RUNTIME_MANIFEST_BINDING_MISMATCH")
    invariance = _unused_parameter_invariance_c1(probe, bundle, device)
    n = len(bundle["metadata"])
    batch_size = 4 if int(spec["channels"]) > 1024 else 16
    roll = []
    for start in range(0, n, 4):
        group = list(range(start, min(start + 4, n)))
        roll.extend([group[-1], *group[:-1]])
    condition_actions = {
        "FULL": bundle["actions"],
        "ZERO_ACTION": torch.zeros_like(bundle["actions"]),
        "SHUFFLED_ACTION": bundle["actions"][roll],
    }
    predictions = {
        name: base._predict_c1_probe(
            probe, bundle["contexts"], actions, device, batch_size=batch_size
        ) for name, actions in condition_actions.items()
    }
    predictions["COPY"] = bundle["copies"]
    reproduced = {"n": n, **base._feature_summary(predictions["FULL"], bundle["targets"])}
    reproduced.update({
        f"no_action_{key}": value
        for key, value in base._feature_summary(predictions["ZERO_ACTION"], bundle["targets"]).items()
    })
    reproduced.update({
        f"shuffled_action_{key}": value
        for key, value in base._feature_summary(predictions["SHUFFLED_ACTION"], bundle["targets"]).items()
    })
    reproduced.update({
        f"last_observation_{key}": value
        for key, value in base._feature_summary(predictions["COPY"], bundle["targets"]).items()
    })
    replay_gate = _reference_gate(reference, reproduced, C1_REFERENCE_KEYS)

    # All non-historical perturbations are intentionally downstream of the
    # untouched replay gate above.
    physical = base._c1_physical_index(bundle)
    predictions["MATCHED_WRONG"] = base._predict_c1_probe(
        probe, bundle["contexts"],
        bundle["actions"][physical["matched_wrong_indices"]], device,
        batch_size=batch_size,
    )
    global_rows = _global_sample_rows(bundle, predictions, physical)
    candidate_rows = []
    primary = []
    pool_name = "MATCHED_TOTAL_POSE_DISPLACEMENT"
    for index, item in enumerate(bundle["metadata"]):
        candidates = base._sample_physical_pool(
            physical["candidates"][pool_name][index], index, 8,
            base.PHASE2_SEED, pool_name,
        )
        candidate_targets = bundle["targets"][candidates]
        positive = candidates.index(index)
        row = {
            "sample_id": item["manifest_sample_id"], "scene_id": item["scene_id"],
            "trajectory_id": item["trajectory_id"], "window_id": item["window_id"],
            "candidate_pool": pool_name, "candidate_seed": base.PHASE2_SEED,
            "desired_candidate_count": 8, "candidate_count": len(candidates),
            "candidate_indices": json.dumps(candidates, separators=(",", ":")),
            "candidate_sample_ids": json.dumps(
                [bundle["metadata"][value]["manifest_sample_id"] for value in candidates],
                separators=(",", ":"),
            ),
            "positive_candidate_position": positive,
            "matched_wrong_index": physical["matched_wrong_indices"][index],
            "matched_wrong_source_sample_id": bundle["metadata"][
                physical["matched_wrong_indices"][index]
            ]["manifest_sample_id"],
            "translation_magnitude": float(physical["translation"][index]),
            "rotation_magnitude_rad": float(physical["rotation"][index]),
            "total_pose_displacement": float(physical["total"][index]),
            "target_frame_id": item["target_frame_id"],
            "expected_target_frame_id": item["expected_target_frame_id"],
        }
        for condition, prediction in predictions.items():
            metrics = base._rank_metrics(prediction[index], candidate_targets, positive)
            for key, value in metrics.items():
                row[f"{condition}_{key}"] = value
        row["FULL_minus_MATCHED_WRONG_cosine"] = (
            float(row["FULL_target_cosine"]) - float(row["MATCHED_WRONG_target_cosine"])
        )
        row["FULL_minus_MATCHED_WRONG_r1"] = (
            int(row["FULL_r1"]) - int(row["MATCHED_WRONG_r1"])
        )
        row["high_absolute_motion"] = int(
            float(row["translation_magnitude"]) >= base.PAPER_PHYSICAL_THRESHOLDS["translation_magnitude"]
            or float(row["rotation_magnitude_rad"]) >= base.PAPER_PHYSICAL_THRESHOLDS["rotation_magnitude_rad"]
            or float(row["total_pose_displacement"]) >= base.PAPER_PHYSICAL_THRESHOLDS["total_pose_displacement"]
        )
        candidate_rows.append(row)
        primary.append(row)

    def aggregate_subset(name, rows):
        result = {"subset": name, "n": len(rows)}
        if not rows:
            return result
        for condition in predictions:
            result[condition] = {
                "r1": statistics.fmean(float(row[f"{condition}_r1"]) for row in rows),
                "mrr": statistics.fmean(float(row[f"{condition}_reciprocal_rank"]) for row in rows),
                "mean_rank": statistics.fmean(float(row[f"{condition}_target_rank"]) for row in rows),
                "mean_cosine": statistics.fmean(float(row[f"{condition}_target_cosine"]) for row in rows),
            }
        r1_ci = base._cluster_bootstrap_delta(rows, "FULL_r1", "MATCHED_WRONG_r1", "scene_id")
        cos_ci = base._cluster_bootstrap_delta(
            rows, "FULL_target_cosine", "MATCHED_WRONG_target_cosine", "scene_id"
        )
        result["full_minus_matched_wrong"] = {
            "r1": statistics.fmean(float(row["FULL_minus_MATCHED_WRONG_r1"]) for row in rows),
            "r1_scene_bootstrap_95ci": list(r1_ci),
            "cosine": statistics.fmean(float(row["FULL_minus_MATCHED_WRONG_cosine"]) for row in rows),
            "cosine_scene_bootstrap_95ci": list(cos_ci),
        }
        return result

    aggregates = {
        "global": {name: base._feature_summary(value, bundle["targets"]) for name, value in predictions.items()},
        "matched_total_pose_k8_seed42": aggregate_subset("ALL", primary),
        "high_absolute_motion": aggregate_subset(
            "HIGH_ABSOLUTE_MOTION", [row for row in primary if row["high_absolute_motion"]]
        ),
    }
    _csv_gz(stage / "global_sample_diagnostics.csv.gz", global_rows)
    _csv_gz(stage / "candidate_sample_diagnostics.csv.gz", candidate_rows)
    torch.save({
        "predictions": predictions, "targets": bundle["targets"],
        "actions": bundle["actions"], "matched_wrong_indices": physical["matched_wrong_indices"],
        "batch_roll_indices": roll,
    }, stage / "predictions.pt")
    _json(stage / "aggregates.json", aggregates)
    return {
        "status": "PASS", "task": "C1", "model": model, "prefix": 16,
        "layer": spec["layer"], "seed": base.PHASE2_SEED,
        "checkpoint": checkpoint_receipt, "reference_summary": {
            "path": str(reference_path.resolve()), "sha256": _sha256(reference_path),
            "checkpoint_path_binding": reference["ckpt_path"],
        },
        "replay_gate": replay_gate, "unused_parameter_invariance": invariance,
        "data_contract": {
            "manifest": str(manifest.resolve()), "manifest_sha256": manifest_sha,
            "sample_count": n, "exact_target_coverage": bundle["coverage"],
            "streaming_feature_root": base._feature_root(model, prefix=16),
            "isolated_target_feature_root": base._feature_root(model, target=True, prefix=16),
            "target_loading": "isolated target cache; dyn_valid requires exact target present",
            "sample_universe_sha256": _stable_hash(bundle["metadata"]),
        },
        "protocol": {
            "evaluation_only": True, "trained_or_fitted_parameters": 0,
            "candidate_pool": pool_name, "desired_k": 8,
            "candidate_seed": base.PHASE2_SEED,
            "physical_thresholds": base.PAPER_PHYSICAL_THRESHOLDS,
        },
    }


def _b2_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return {
        "n": len(rows),
        "mean_ang_err_deg": statistics.fmean(float(row["angular_error_deg"]) for row in rows),
        "mean_az_err_deg": statistics.fmean(float(row["azimuth_error_deg"]) for row in rows),
        "mean_el_err_deg": statistics.fmean(float(row["elevation_error_deg"]) for row in rows),
        "top1": statistics.fmean(float(row["top1"]) for row in rows),
        "top3": statistics.fmean(float(row["top3"]) for row in rows),
        "mean_log_dist_err": statistics.fmean(float(row["log_distance_error"]) for row in rows),
    }


def _b2(stage: Path, model: str, checkpoint: Path, expected_sha: str,
        reference_path: Path, device) -> Dict[str, object]:
    import torch

    spec = base.MODEL_SPECS[model]
    reference = json.loads(reference_path.read_text())
    if Path(reference["ckpt_path"]).resolve() != checkpoint.resolve():
        raise RuntimeError("REFERENCE_CHECKPOINT_BINDING_MISMATCH")
    probe, checkpoint_receipt = _load_probe("b2", model, checkpoint, expected_sha, device)
    bundle = base._collect_b2_bundle(model, 16, "val")
    samples = bundle["samples"]
    invariance = _unused_parameter_invariance_b2(probe, samples[0], device)

    # Untouched evaluator path first.  No intervention is computed unless this
    # exact historical aggregate gate succeeds.
    full_rows = []
    full_predictions = []
    for index, sample in enumerate(samples):
        logits, distance = base._b2_prediction(probe, sample["feature"], sample["query"], device)
        metrics = base._b2_metrics(logits, distance, sample["polar"])
        row = {
            "sample_id": f"B2|{model}|P16|{sample['manifest_sample_id']}|obj{sample['hidden_id']}",
            "manifest_sample_id": sample["manifest_sample_id"], "scene_id": sample["scene_id"],
            "trajectory_id": sample["trajectory_id"], "window_id": sample["window_id"],
            "method": "FULL_HISTORY_PLUS_QUERY", "query_frame": sample["query_frame"],
            "query_raw_frame": sample["query_raw_frame"], "target_raw_frame": sample["target_raw_frame"],
            "target_object_id": sample["hidden_id"], "predicted_log_distance": distance,
            **metrics,
        }
        full_rows.append(row)
        full_predictions.append((logits, distance))
        if (index + 1) % 20 == 0:
            print(f"[P16 repair B2 {model} replay] {index + 1}/{len(samples)}", flush=True)
    reproduced = _b2_summary(full_rows)
    replay_gate = _reference_gate(reference, reproduced, B2_REFERENCE_KEYS)

    by_scene = defaultdict(list)
    for index, sample in enumerate(samples):
        by_scene[sample["scene_id"]].append(index)
    swaps = {}
    for indices in by_scene.values():
        if len(indices) > 1:
            for offset, index in enumerate(indices):
                swaps[index] = indices[(offset + 1) % len(indices)]

    all_rows = list(full_rows)
    method_rows = defaultdict(list)
    method_rows["FULL_HISTORY_PLUS_QUERY"] = full_rows
    saved_predictions = {
        "FULL_HISTORY_PLUS_QUERY": {
            "logits": torch.stack([value[0] for value in full_predictions]),
            "log_distance": torch.tensor([value[1] for value in full_predictions]),
        }
    }
    identity_max_difference = 0.0
    methods = (
        "HISTORY_WITHOUT_QUERY", "CURRENT_FRAME_PLUS_QUERY",
        "REMOVE_TARGET_VISIBLE_FRAMES", "CONTENT_ONLY_SHUFFLE",
        "FRAME_TOKEN_SHUFFLE", "REVERSE_CONTENT_ONLY", "REVERSE_FRAME_TOKENS",
        "SAME_SCENE_QUERY_SWAP",
    )
    prediction_lists = {name: [] for name in methods}
    for index, sample in enumerate(samples):
        frames = int(sample["feature"].shape[0])
        identity = list(range(frames))
        explicit_logits, explicit_distance = base._b2_prediction_frame_ids(
            probe, sample["feature"], sample["query"], identity, device
        )
        identity_max_difference = max(
            identity_max_difference,
            float((explicit_logits - full_predictions[index][0]).abs().max()),
            abs(explicit_distance - full_predictions[index][1]),
        )
        shuffle = base._b2_temporal_mapping(frames, base.PHASE2_SEED * 1000003 + index)
        reverse = base._b2_temporal_mapping(frames, base.PHASE2_SEED, reverse=True)
        visible = [frame for frame, count in enumerate(sample["pixel_counts"]) if count > 0]
        current = torch.zeros_like(sample["feature"])
        current[-1] = sample["feature"][-1]
        removed = sample["feature"].clone()
        if visible:
            removed[visible] = 0
        conditions = [
            ("HISTORY_WITHOUT_QUERY", sample["feature"], torch.zeros_like(sample["query"]), identity, identity, ""),
            ("CURRENT_FRAME_PLUS_QUERY", current, sample["query"], identity, identity, ""),
            ("REMOVE_TARGET_VISIBLE_FRAMES", removed, sample["query"], identity, identity, ""),
            ("CONTENT_ONLY_SHUFFLE", sample["feature"][shuffle], sample["query"], identity, shuffle, ""),
            ("FRAME_TOKEN_SHUFFLE", sample["feature"][shuffle], sample["query"], shuffle, shuffle, ""),
            ("REVERSE_CONTENT_ONLY", sample["feature"][reverse], sample["query"], identity, reverse, ""),
            ("REVERSE_FRAME_TOKENS", sample["feature"][reverse], sample["query"], reverse, reverse, ""),
        ]
        if index in swaps:
            conditions.append((
                "SAME_SCENE_QUERY_SWAP", sample["feature"], samples[swaps[index]]["query"],
                identity, identity, samples[swaps[index]]["manifest_sample_id"],
            ))
        for method, feature, query, frame_ids, mapping, swap_source in conditions:
            logits, distance = base._b2_prediction_frame_ids(probe, feature, query, frame_ids, device)
            metrics = base._b2_metrics(logits, distance, sample["polar"])
            row = {
                "sample_id": full_rows[index]["sample_id"],
                "manifest_sample_id": sample["manifest_sample_id"], "scene_id": sample["scene_id"],
                "trajectory_id": sample["trajectory_id"], "window_id": sample["window_id"],
                "method": method, "query_frame": sample["query_frame"],
                "query_raw_frame": sample["query_raw_frame"], "target_raw_frame": sample["target_raw_frame"],
                "target_object_id": sample["hidden_id"], "target_visible_frame_indices": json.dumps(visible),
                "history_mapping": json.dumps(mapping, separators=(",", ":")),
                "temporal_frame_ids": json.dumps(frame_ids, separators=(",", ":")),
                "final_frame_fixed": int(mapping[-1] == frames - 1),
                "query_unchanged": int(method != "HISTORY_WITHOUT_QUERY" and method != "SAME_SCENE_QUERY_SWAP"),
                "gt_unchanged": 1, "swap_source_sample_id": swap_source,
                "predicted_log_distance": distance, **metrics,
            }
            method_rows[method].append(row)
            all_rows.append(row)
            prediction_lists[method].append((logits, distance, index))
        if (index + 1) % 10 == 0:
            print(f"[P16 repair B2 {model} controls] {index + 1}/{len(samples)}", flush=True)
    if identity_max_difference > 1e-5:
        raise RuntimeError(f"B2_IDENTITY_FRAME_PATH_MISMATCH: {identity_max_difference}")

    base_by_id = {row["sample_id"]: row for row in full_rows}
    aggregates = {}
    for method, rows in method_rows.items():
        paired_base = [base_by_id[row["sample_id"]] for row in rows]
        paired_mean = statistics.fmean(float(row["angular_error_deg"]) for row in paired_base)
        stats = base._b2_method_stats(rows)
        low, high = base._paired_b2_bootstrap(rows, base_by_id, "scene_id")
        aggregates[method] = {
            **stats, "paired_full_mean_ang_err_deg": paired_mean,
            "delta_error_vs_paired_full": stats["mean"] - paired_mean,
            "paired_scene_bootstrap_95ci": [low, high],
            "sample_scope": "same-scene-swappable subset" if method == "SAME_SCENE_QUERY_SWAP" else "full eval set",
        }
    for method, values in prediction_lists.items():
        if values:
            saved_predictions[method] = {
                "sample_indices": torch.tensor([value[2] for value in values]),
                "logits": torch.stack([value[0] for value in values]),
                "log_distance": torch.tensor([value[1] for value in values]),
            }
    _csv_gz(stage / "sample_diagnostics.csv.gz", all_rows)
    torch.save(saved_predictions, stage / "predictions.pt")
    _json(stage / "aggregates.json", aggregates)
    universe = [{
        key: sample[key] for key in (
            "manifest_sample_id", "scene_id", "window_id", "prefix_indices",
            "query_frame", "query_raw_frame", "target_raw_frame", "hidden_id",
        )
    } for sample in samples]
    return {
        "status": "PASS", "task": "B2", "model": model, "prefix": 16,
        "layer": spec["layer"], "seed": base.PHASE2_SEED,
        "checkpoint": checkpoint_receipt, "reference_summary": {
            "path": str(reference_path.resolve()), "sha256": _sha256(reference_path),
            "checkpoint_path_binding": reference["ckpt_path"],
        },
        "replay_gate": replay_gate, "unused_parameter_invariance": invariance,
        "data_contract": {
            "dataset_n": bundle["dataset_n"], "valid_sample_count": len(samples),
            "streaming_feature_root": base._feature_root(model, prefix=16),
            "raw_data_root": base.SCANNET_LOCAL_ROOT,
            "sample_universe_sha256": _stable_hash(universe),
        },
        "protocol": {
            "evaluation_only": True, "trained_or_fitted_parameters": 0,
            "prior_refit": False, "synthetic_order_sanity": base._b2_order_synthetic_sanity(),
            "identity_frame_id_max_difference": identity_max_difference,
            "conditions": ["FULL_HISTORY_PLUS_QUERY", *methods],
        },
    }


def run_cell(task: str, model: str, output: Path, checkpoint: Path,
             checkpoint_sha256: str, reference_summary: Path) -> None:
    import importlib
    import torch

    if model not in base.CORE_MODELS:
        raise ValueError(f"P16 repair supports only {base.CORE_MODELS}, got {model}")
    if not torch.cuda.is_available():
        raise RuntimeError("P16_REPAIR_REQUIRES_CUDA_ON_ZOETROPE")
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"CELL_OUTPUT_ALREADY_EXISTS: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    attempt_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{uuid.uuid4().hex[:8]}"
    stage = output.parent / f".{output.name}.{attempt_id}.staging"
    stage.mkdir()
    started = time.time()
    try:
        receipt = (
            _c1(stage, model, checkpoint, checkpoint_sha256, reference_summary, torch.device("cuda"))
            if task == "c1" else
            _b2(stage, model, checkpoint, checkpoint_sha256, reference_summary, torch.device("cuda"))
        )
        receipt.update({
            "attempt_id": attempt_id,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_seconds": time.time() - started,
            "execution": {
                "hostname": os.uname().nodename,
                "cuda_device_name": torch.cuda.get_device_name(0),
                "torch_version": torch.__version__,
                "source_snapshot_root": str(base.EXECUTION_SOURCE_ROOT),
                "project_root": str(base.ROOT),
                "imported_module_paths": {
                    name: str(Path(importlib.import_module(name).__file__).resolve())
                    for name in (
                        "analyses.probe_validity.run_probe_validity",
                        "analyses.probe_validity.p16_repair",
                        "vidfm3d.data.components.inscene15k_dataset",
                        "vidfm3d.models.components.probe_action_dynamics",
                        "vidfm3d.models.components.probe_ego_belief_v2",
                        "vidfm3d.utils.spatial_diag",
                        "vidfm3d.utils.aoss_fs",
                    )
                },
            },
        })
        artifacts = {}
        for path in sorted(stage.iterdir()):
            if path.is_file() and path.name != "receipt.json":
                artifacts[path.name] = {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
        receipt["artifacts"] = artifacts
        _json(stage / "receipt.json", receipt)
        os.replace(stage, output)
        print(json.dumps({"status": "PASS", "cell": str(output), "attempt_id": attempt_id}, indent=2))
    except Exception as error:
        failure = {
            "status": "FAIL", "task": task, "model": model, "attempt_id": attempt_id,
            "reason": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(),
            "checkpoint": str(checkpoint), "expected_checkpoint_sha256": checkpoint_sha256,
            "reference_summary": str(reference_summary),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _json(stage / "failure_receipt.json", failure)
        failed = output.parent / f"{output.name}__FAILED__{attempt_id}"
        os.replace(stage, failed)
        print(json.dumps(failure, indent=2), flush=True)
        raise
