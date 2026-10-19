#!/usr/bin/env python3
"""Run formal Phase-1 interventions for the frozen Wan P12/L20 probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyses.probe_validity.run_probe_validity import MODEL_SPECS, _dataset, _load_probe
from scripts.prepare_wan_p12_l20_action_validity import (
    CHECKPOINT_SHA256,
    MANIFEST_SHA256,
    PREDICTIONS_SHA256,
    RUN_TAG,
)
from vidfm3d.models.components.probe_action_dynamics import ActionDynamicsProbe
from vidfm3d.utils.action_validity import aggregate_retrieval, retrieval_per_sample


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def outcomes(
    predictions: torch.Tensor,
    targets_normalized: torch.Tensor,
    positive_indices: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    similarity = torch.nn.functional.normalize(predictions.float(), dim=-1) @ targets_normalized.T
    ordered = similarity.argsort(dim=-1, descending=True)
    rank0 = (ordered == positive_indices[:, None]).to(torch.int64).argmax(dim=-1).float()
    rows = torch.arange(len(predictions), device=predictions.device)
    positive = similarity[rows, positive_indices]
    return {
        "cos": positive,
        "r1": (rank0 == 0).float(),
        "r5": (rank0 < 5).float(),
        "mrr": 1.0 / (rank0 + 1.0),
        "rank0": rank0,
        "rank1": rank0 + 1.0,
    }


@torch.inference_mode()
def predict_pairs(
    probe: torch.nn.Module,
    contexts: Sequence[torch.Tensor],
    actions: torch.Tensor,
    recipients: np.ndarray,
    donors: np.ndarray,
    targets: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    metric_names = ("cos", "r1", "r5", "mrr", "rank0", "rank1")
    result = {key: np.empty(len(recipients), dtype=np.float32) for key in metric_names}
    targets_normalized = torch.nn.functional.normalize(targets.to(device).float(), dim=-1)
    for start in range(0, len(recipients), batch_size):
        stop = min(start + batch_size, len(recipients))
        batch_recipients = recipients[start:stop]
        batch_donors = donors[start:stop]
        by_shape: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
        for offset, recipient in enumerate(batch_recipients):
            by_shape[tuple(contexts[int(recipient)].shape)].append(offset)
        for offsets in by_shape.values():
            recipient_indices = np.asarray([batch_recipients[offset] for offset in offsets], dtype=np.int64)
            donor_indices = np.asarray([batch_donors[offset] for offset in offsets], dtype=np.int64)
            context = torch.stack([contexts[int(index)] for index in recipient_indices]).to(device).float()
            action = actions[torch.from_numpy(donor_indices)].to(device).float()
            prediction = probe(context, action)
            positive = torch.from_numpy(recipient_indices).to(device).long()
            current = outcomes(prediction, targets_normalized, positive)
            destination = np.asarray([start + offset for offset in offsets], dtype=np.int64)
            for key, value in current.items():
                result[key][destination] = value.cpu().numpy()
        if stop % 6400 == 0 or stop == len(recipients):
            print(f"interventions {stop}/{len(recipients)}", flush=True)
    return result


def condition_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    per_sample_torch = retrieval_per_sample(predictions.float(), targets.float())
    per_sample = {
        key: value.cpu().numpy().astype(np.float32)
        for key, value in per_sample_torch.items()
    }
    return per_sample, aggregate_retrieval(per_sample_torch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=PROJECT_ROOT / "results/analysis/wan_p12_l20_action_validity",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    analysis = args.analysis_dir.resolve()
    output = analysis / "layers/layer20"
    output.mkdir(parents=True, exist_ok=False)
    provenance = json.loads((analysis / "input_provenance.json").read_text())
    checkpoint = Path(provenance["checkpoint"])
    manifest = Path(provenance["manifest"])
    existing_path = Path(provenance["predictions"])
    mapping_index_path = analysis / "permutations/index.json"
    if sha256(manifest) != MANIFEST_SHA256:
        raise RuntimeError("manifest hash gate failed")
    if sha256(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint hash gate failed")
    if sha256(existing_path) != PREDICTIONS_SHA256:
        raise RuntimeError("frozen prediction hash gate failed")

    mapping_index = json.loads(mapping_index_path.read_text())
    if len(mapping_index) != 100:
        raise RuntimeError("exactly 100 permutation mappings are required")
    mappings: List[np.ndarray] = []
    mapping_seeds: List[int] = []
    mapping_hashes: List[str] = []
    for entry in mapping_index:
        path = Path(entry["path"])
        if sha256(path) != entry["sha256"]:
            raise RuntimeError(f"permutation mapping hash failed: {path}")
        mapping = json.loads(path.read_text())
        if (
            not mapping["exact_marginal_preserved"]
            or mapping["fixed_points"] != 0
            or mapping["same_scene_donors"] != 0
        ):
            raise RuntimeError(f"invalid permutation mapping: {path}")
        mappings.append(np.asarray(mapping["recipient_to_donor_index"], dtype=np.int64))
        mapping_seeds.append(int(mapping["seed"]))
        mapping_hashes.append(entry["sha256"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("formal P12/L20 intervention run requires CUDA")
    print(f"device={torch.cuda.get_device_name(0)}", flush=True)

    dataset = _dataset("c1", "val", model="wan")
    contexts: List[torch.Tensor] = []
    actions_list: List[torch.Tensor] = []
    targets_list: List[torch.Tensor] = []
    sample_ids: List[str] = []
    scene_paths: List[str] = []
    target_frames: List[int] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        if not bool(sample["dyn_valid"].item()):
            raise RuntimeError(f"invalid P12 val sample at index {index}")
        manifest_row = dataset.scenes[index]
        contexts.append(sample["input_feat"].detach().cpu())
        actions_list.append(sample["action"].detach().cpu().float())
        targets_list.append(sample["target_feat"].mean(dim=(0, 1)).detach().cpu().float())
        sample_ids.append(str(manifest_row["sample_id"]))
        scene_paths.append(str(manifest_row["scene_dir"]))
        target_frames.append(int(sample["target_frame_idx"].item()))
        if (index + 1) % 20 == 0:
            print(f"dataset {index + 1}/{len(dataset)}", flush=True)
    if len(dataset) != 276 or len(set(sample_ids)) != 276 or len(set(scene_paths)) != 69:
        raise RuntimeError("Dataset inventory gate failed")

    actions = torch.stack(actions_list)
    dataset_targets = torch.stack(targets_list)
    existing = torch.load(existing_path, map_location="cpu", weights_only=False)
    existing_scenes = [str(record["scene_path"]) for record in existing]
    existing_target_frames = [int(record["target_frame_idx"]) for record in existing]
    existing_actions = torch.stack([record["action"].float() for record in existing])
    existing_targets = torch.stack([record["target"].float() for record in existing])
    existing_predictions = torch.stack([record["pred"].float() for record in existing])
    existing_copy = torch.stack([record["last_observation"].float() for record in existing])
    if scene_paths != existing_scenes or target_frames != existing_target_frames:
        raise RuntimeError("DataLoader ordering differs from frozen evaluator")
    dataset_action_max_abs_diff = float((actions - existing_actions).abs().max())
    if dataset_action_max_abs_diff > 5e-7:
        raise RuntimeError(
            "Dataset action differs materially from frozen evaluator: "
            f"{dataset_action_max_abs_diff}"
        )
    # JSON-to-float and torch serialization can differ by one FP32 ULP. Once
    # alignment is verified, use the frozen evaluator tensor for intervention.
    actions = existing_actions
    dataset_target_max_abs_diff = float((dataset_targets - existing_targets).abs().max())
    if dataset_target_max_abs_diff > 5e-6:
        raise RuntimeError(
            "Dataset target differs materially from frozen evaluator: "
            f"{dataset_target_max_abs_diff}"
        )
    targets = existing_targets

    spec = MODEL_SPECS["wan"]
    probe = _load_probe(
        checkpoint,
        ActionDynamicsProbe(
            in_channels=1536,
            action_dim=9,
            hidden_dim=512,
            num_layers=2,
            num_heads=8,
            max_seq_len=64,
        ),
        device,
        allowed_missing=("constant_context", "constant_action"),
    )

    recipients = np.arange(276, dtype=np.int64)
    recomputed_vectors: List[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, 276, 4):
            stop = min(start + 4, 276)
            x = torch.stack(contexts[start:stop]).to(device).float()
            a = actions[start:stop].to(device).float()
            recomputed_vectors.append(probe(x, a).cpu())
    recomputed_predictions = torch.cat(recomputed_vectors)
    frozen_outcome = retrieval_per_sample(existing_predictions, targets)
    recomputed_outcome = retrieval_per_sample(recomputed_predictions, targets)
    metric_crosscheck = {
        key: abs(float(recomputed_outcome[key].mean()) - float(frozen_outcome[key].mean()))
        for key in recomputed_outcome
    }
    correct_vector_max_abs_diff = float(
        (recomputed_predictions - existing_predictions).abs().max()
    )
    rank_delta = recomputed_outcome["rank0"] - frozen_outcome["rank0"]
    rank_mismatch = rank_delta != 0
    frozen_similarity = (
        torch.nn.functional.normalize(existing_predictions.float(), dim=-1)
        @ torch.nn.functional.normalize(targets.float(), dim=-1).T
    )
    recomputed_similarity = (
        torch.nn.functional.normalize(recomputed_predictions.float(), dim=-1)
        @ torch.nn.functional.normalize(targets.float(), dim=-1).T
    )
    rank_mismatch_indices = torch.nonzero(rank_mismatch, as_tuple=True)[0]
    flipped_frozen_margins: List[float] = []
    for index in rank_mismatch_indices.tolist():
        frozen_margin = frozen_similarity[index] - frozen_similarity[index, index]
        recomputed_margin = recomputed_similarity[index] - recomputed_similarity[index, index]
        flipped = (frozen_margin > 0) != (recomputed_margin > 0)
        flipped[index] = False
        flipped_frozen_margins.extend(frozen_margin[flipped].abs().tolist())
    replay_diagnostic = {
        "correct_prediction_max_abs_diff_from_frozen": correct_vector_max_abs_diff,
        "metric_crosscheck_abs_diff": metric_crosscheck,
        "rank_mismatch_count": int(rank_mismatch.sum()),
        "rank_mismatch_indices": rank_mismatch_indices.tolist(),
        "rank_delta": rank_delta[rank_mismatch].tolist(),
        "max_abs_frozen_cosine_margin_among_flipped_candidates": (
            max(flipped_frozen_margins) if flipped_frozen_margins else 0.0
        ),
        "flipped_candidate_count": len(flipped_frozen_margins),
    }
    write_json(output / "replay_diagnostic.json", replay_diagnostic)
    print(json.dumps({"replay_diagnostic": replay_diagnostic}, sort_keys=True), flush=True)
    if bool(rank_mismatch.any()):
        raise RuntimeError("recomputed correct prediction changes frozen retrieval ranks")
    if max(metric_crosscheck.values()) > 1e-6:
        raise RuntimeError(f"correct retrieval metric gate failed: {metric_crosscheck}")
    if correct_vector_max_abs_diff > 5e-5:
        raise RuntimeError(
            "recomputed correct prediction differs from frozen evaluator: "
            f"{correct_vector_max_abs_diff}"
        )

    zero_donors = recipients
    zero_actions = torch.zeros_like(actions)
    null_flat = predict_pairs(
        probe,
        contexts,
        zero_actions,
        recipients,
        zero_donors,
        targets,
        device,
        batch_size=args.batch_size,
    )
    null_predictions_outcome = {key: values for key, values in null_flat.items()}

    copy_last = torch.stack(
        [context[-1].float().mean(dim=(0, 1)) for context in contexts]
    )
    copy_last_max_abs_diff = float((copy_last - existing_copy).abs().max())
    if copy_last_max_abs_diff > 5e-6:
        raise RuntimeError(f"last-state copy differs from frozen evaluator: {copy_last_max_abs_diff}")
    copy_prefix = torch.stack(
        [context.float().mean(dim=(0, 1, 2)) for context in contexts]
    )

    condition_arrays: Dict[str, Dict[str, np.ndarray]] = {}
    condition_summary: Dict[str, Dict[str, float]] = {}
    condition_arrays["correct_action"], condition_summary["correct_action"] = condition_metrics(
        existing_predictions, targets
    )
    condition_arrays["eval_null_action"] = null_predictions_outcome
    condition_summary["eval_null_action"] = {
        key: float(values.mean()) for key, values in null_predictions_outcome.items()
    }
    condition_arrays["copy_last_streaming_state"], condition_summary[
        "copy_last_streaming_state"
    ] = condition_metrics(existing_copy, targets)
    condition_arrays["copy_prefix_pooled_state"], condition_summary[
        "copy_prefix_pooled_state"
    ] = condition_metrics(copy_prefix, targets)

    permutation_array = np.stack(mappings)
    permutation_recipients = np.tile(recipients, len(mappings))
    permutation_donors = permutation_array.reshape(-1)
    print(f"running {len(permutation_recipients)} formal shuffled-action pairs", flush=True)
    permutation_flat = predict_pairs(
        probe,
        contexts,
        actions,
        permutation_recipients,
        permutation_donors,
        targets,
        device,
        batch_size=args.batch_size,
    )
    permutation = {
        key: value.reshape(len(mappings), 276)
        for key, value in permutation_flat.items()
    }

    wrong_recipients = np.repeat(recipients, 275)
    wrong_donors = np.concatenate(
        [np.concatenate((recipients[:index], recipients[index + 1 :])) for index in recipients]
    )
    print(f"running {len(wrong_recipients)} exhaustive wrong-action pairs", flush=True)
    wrong_flat = predict_pairs(
        probe,
        contexts,
        actions,
        wrong_recipients,
        wrong_donors,
        targets,
        device,
        batch_size=args.batch_size,
    )
    wrong: Dict[str, np.ndarray] = {}
    for key, values in wrong_flat.items():
        matrix = np.full((276, 276), np.nan, dtype=np.float32)
        matrix[wrong_recipients, wrong_donors] = values
        wrong[key] = matrix

    arrays: Dict[str, Any] = {
        "sample_ids": np.asarray(sample_ids, dtype=object),
        "scene_paths": np.asarray(scene_paths, dtype=object),
        "target_frame_indices": np.asarray(target_frames, dtype=np.int64),
        "actions": actions.numpy(),
        "mapping_seeds": np.asarray(mapping_seeds, dtype=np.int64),
        "permutation_donor_indices": permutation_array,
    }
    for condition, metrics in condition_arrays.items():
        for metric, values in metrics.items():
            arrays[f"condition__{condition}__{metric}"] = values
    for metric, values in permutation.items():
        arrays[f"permutation__{metric}"] = values
    for metric, values in wrong.items():
        arrays[f"wrong__{metric}"] = values
    arrays_path = output / "interventions.npz"
    np.savez_compressed(arrays_path, **arrays)

    correct_cos = condition_arrays["correct_action"]["cos"]
    correct_rank1 = condition_arrays["correct_action"]["rank1"]
    wrong_margin = correct_cos[:, None] - wrong["cos"]
    finite_wrong = np.isfinite(wrong_margin)
    summary = {
        "ok": True,
        "run_tag": RUN_TAG,
        "model": "wan",
        "prefix": 12,
        "layer": 20,
        "n": 276,
        "n_scenes": 69,
        "device": torch.cuda.get_device_name(0),
        "correct_prediction_max_abs_diff_from_frozen": correct_vector_max_abs_diff,
        "correct_metric_crosscheck_abs_diff": metric_crosscheck,
        "dataset_action_max_abs_diff_from_frozen": dataset_action_max_abs_diff,
        "dataset_target_max_abs_diff_from_frozen": dataset_target_max_abs_diff,
        "copy_last_max_abs_diff_from_frozen": copy_last_max_abs_diff,
        "conditions": condition_summary,
        "permutation": {
            metric: {
                "mean_over_seeds": float(values.mean()),
                "seed_metric_mean": [float(value) for value in values.mean(axis=1)],
            }
            for metric, values in permutation.items()
        },
        "wrong_action": {
            "mean_correct_minus_wrong_cos": float(np.nanmean(wrong_margin)),
            "median_correct_minus_wrong_cos": float(np.nanmedian(wrong_margin)),
            "positive_margin_fraction": float(
                np.count_nonzero(wrong_margin[finite_wrong] > 0)
                / np.count_nonzero(finite_wrong)
            ),
            "mean_rank_improvement": float(
                np.nanmean(wrong["rank1"] - correct_rank1[:, None])
            ),
            "all_legal_wrong_actions_per_sample": 275,
        },
        "copy_definitions": {
            "copy_last_streaming_state": "spatial mean of final saved P12 streaming feature map",
            "copy_prefix_pooled_state": "mean over all 12 timesteps and spatial positions",
            "copy_anchor_isolated": "unavailable in frozen roots; no extraction performed",
        },
        "metric_definitions": {
            "candidate_pool": "all 276 frozen P12 val targets",
            "positive": "manifest sample-ID-paired target",
            "rank0": "zero-based historical rank",
            "rank1": "one-based report rank",
        },
    }
    write_json(output / "layer_summary.json", summary)
    run_provenance = {
        "ok": True,
        "run_tag": RUN_TAG,
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "streaming_root": "fj:s3://baiqiao/InsScene-15K/FEAT_STREAMING_BY_PREFIX/wan_prefix_12_m0p35_r0p5",
        "target_root": "fj:s3://baiqiao/InsScene-15K/FEAT_TARGET_BY_PREFIX/wan_prefix_12_m0p35_r0p5",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "existing_predictions": str(existing_path),
        "existing_predictions_sha256": sha256(existing_path),
        "mapping_index": str(mapping_index_path),
        "mapping_index_sha256": sha256(mapping_index_path),
        "mapping_hashes": mapping_hashes,
        "arrays": str(arrays_path),
        "arrays_sha256": sha256(arrays_path),
        "batch_size": args.batch_size,
        "fallback_count": 0,
    }
    write_json(output / "provenance.json", run_provenance)
    print(json.dumps({"ok": True, "output": str(output)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
