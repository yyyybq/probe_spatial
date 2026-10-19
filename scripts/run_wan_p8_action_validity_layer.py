#!/usr/bin/env python3
"""Run Phase-1 action interventions for one frozen Wan p8 probe layer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from vidfm3d.utils.action_validity import aggregate_retrieval, retrieval_per_sample


RUN_TAG = "wan_p8_action_dynamics_canonical_manifest_exact_20260821_200604_2075f3e"
MANIFEST_SHA256 = "772194f077c45cc85a942f17f46f50103125ae8f5b1c983c08881c1759607e0c"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: (
            value.to(device).float()
            if isinstance(value, torch.Tensor) and value.is_floating_point()
            else value.to(device)
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in batch.items()
    }


def _outcomes(
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
def _predict_pairs(
    probe: torch.nn.Module,
    contexts: torch.Tensor,
    actions: torch.Tensor,
    recipients: np.ndarray,
    donors: np.ndarray,
    targets: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    result = {key: np.empty(len(recipients), dtype=np.float32) for key in ("cos", "r1", "r5", "mrr", "rank0", "rank1")}
    targets_normalized = torch.nn.functional.normalize(targets.to(device).float(), dim=-1)
    for start in range(0, len(recipients), batch_size):
        stop = min(start + batch_size, len(recipients))
        recipient = torch.from_numpy(recipients[start:stop]).long()
        donor = torch.from_numpy(donors[start:stop]).long()
        pred = probe(
            contexts[recipient].to(device).float(),
            actions[donor].to(device).float(),
        )
        outcome = _outcomes(pred, targets_normalized, recipient.to(device))
        for key, value in outcome.items():
            result[key][start:stop] = value.cpu().numpy()
    return result


def _condition_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    per_sample_torch = retrieval_per_sample(predictions.float(), targets.float())
    per_sample = {key: value.cpu().numpy().astype(np.float32) for key, value in per_sample_torch.items()}
    return per_sample, aggregate_retrieval(per_sample_torch)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    required = [
        "analysis_output_dir",
        "analysis_mapping_index",
        "analysis_existing_predictions",
        "analysis_manifest",
        "analysis_checkpoint_sha256",
    ]
    missing = [name for name in required if not cfg.get(name)]
    if missing:
        raise ValueError(f"missing required analysis settings: {missing}")
    if not cfg.get("ckpt_path"):
        raise ValueError("ckpt_path is required")
    layer = int(cfg.feature_layer)
    if layer not in {0, 5, 10, 15, 25, 29}:
        raise ValueError(f"unexpected layer {layer}")

    output = Path(str(cfg.analysis_output_dir)).resolve() / "layers" / f"layer{layer}"
    output.mkdir(parents=True, exist_ok=False)
    manifest = Path(str(cfg.analysis_manifest)).resolve()
    checkpoint = Path(str(cfg.ckpt_path)).resolve()
    existing_path = Path(str(cfg.analysis_existing_predictions)).resolve()
    mapping_index_path = Path(str(cfg.analysis_mapping_index)).resolve()
    streaming_root = Path(str(cfg.streaming_feat_root)).resolve()
    target_root = Path(str(cfg.target_feat_root)).resolve()
    if sha256(manifest) != MANIFEST_SHA256:
        raise RuntimeError("manifest hash gate failed")
    if sha256(checkpoint) != str(cfg.analysis_checkpoint_sha256):
        raise RuntimeError("checkpoint hash gate failed")
    expected_feature_parent = Path(
        f"/mnt/umm/users/yinbaiqiao/InsScene-15K/FEATURES_VERSIONED/{RUN_TAG}"
    ).resolve()
    if streaming_root != expected_feature_parent / "streaming" or target_root != expected_feature_parent / "target":
        raise RuntimeError("feature root gate failed")
    if streaming_root.is_symlink() or target_root.is_symlink():
        raise RuntimeError("feature roots must not be aliases")

    mapping_index = json.loads(mapping_index_path.read_text())
    if len(mapping_index) != 100:
        raise RuntimeError("exactly 100 permutation mappings are required")
    mappings = []
    mapping_seeds = []
    mapping_hashes = []
    for entry in mapping_index:
        path = Path(entry["path"])
        if sha256(path) != entry["sha256"]:
            raise RuntimeError(f"permutation mapping hash failed: {path}")
        mapping = json.loads(path.read_text())
        if not mapping["exact_marginal_preserved"] or mapping["fixed_points"] != 0 or mapping["same_scene_donors"] != 0:
            raise RuntimeError(f"invalid permutation mapping: {path}")
        mappings.append(np.asarray(mapping["recipient_to_donor_index"], dtype=np.int64))
        mapping_seeds.append(int(mapping["seed"]))
        mapping_hashes.append(entry["sha256"])

    print(OmegaConf.to_yaml(cfg), flush=True)
    datamodule = hydra.utils.instantiate(cfg.data.data_module)
    datamodule.setup(stage="validate")
    datamodule.trainer = SimpleNamespace(
        strategy=None, world_size=1, global_rank=0, local_rank=0, num_nodes=1
    )
    loader = datamodule.val_dataloader()
    if isinstance(loader, list):
        loader = loader[0]

    model = hydra.utils.instantiate(cfg.model)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=True)
    if model.probe_type != "action_dynamics":
        raise RuntimeError("loaded model is not action_dynamics")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Phase-1 intervention job requires CUDA")
    model.float().to(device).eval()

    contexts: List[torch.Tensor] = []
    actions: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    sample_ids: List[str] = []
    scene_paths: List[str] = []
    anchor_valid: List[bool] = []
    for item in loader:
        batch = item[0] if isinstance(item, tuple) else item
        valid = batch.get("dyn_valid", torch.ones(len(batch["input_feat"]), dtype=torch.bool)).bool()
        if not bool(valid.all()):
            raise RuntimeError("invalid canonical val row encountered")
        contexts.append(batch["input_feat"].cpu())
        actions.append(batch["action"].float().cpu())
        targets.append(batch["target_feat"].float().mean(dim=(1, 2)).cpu())
        sample_ids.extend([str(value) for value in batch["sample_id"]])
        scene_paths.extend([str(value) for value in batch["scene_path"]])
        anchor_valid.extend(batch["anchor_isolated_valid"].bool().tolist())
    contexts_t = torch.cat(contexts, dim=0)
    actions_t = torch.cat(actions, dim=0)
    targets_t = torch.cat(targets, dim=0)
    if len(sample_ids) != 276 or len(set(sample_ids)) != 276 or len(set(scene_paths)) != 69:
        raise RuntimeError("Dataset/DataLoader inventory gate failed")
    if any(anchor_valid):
        raise RuntimeError("unexpected isolated anchor availability; frozen copy contract changed")

    existing = torch.load(existing_path, map_location="cpu", weights_only=False)
    existing_ids = [str(record["sample_id"]) for record in existing]
    existing_actions = torch.stack([record["action"].float() for record in existing])
    existing_targets = torch.stack([record["target"].float() for record in existing])
    existing_predictions = torch.stack([record["pred"].float() for record in existing])
    if sample_ids != existing_ids:
        raise RuntimeError("DataLoader order/sample IDs differ from frozen eval")
    torch.testing.assert_close(actions_t, existing_actions, rtol=0.0, atol=0.0)
    dataset_target_max_abs_diff = float((targets_t - existing_targets).abs().max())
    if dataset_target_max_abs_diff > 5e-6:
        raise RuntimeError(
            "Dataset target differs materially from the frozen candidate pool: "
            f"{dataset_target_max_abs_diff}"
        )
    # CPU and CUDA reductions over the same FP16 feature map differ by a few
    # ULPs. Use the frozen eval targets after the semantic/alignment gate so the
    # intervention candidate pool is bit-identical to canonical evaluation.
    targets_t = existing_targets

    batch_size = int(cfg.get("analysis_batch_size", 64))
    recipients = np.arange(276, dtype=np.int64)
    recomputed_correct_parts = []
    with torch.inference_mode():
        # Match the frozen canonical eval's batch size for this reproducibility
        # gate. Formal intervention batches may be larger, but the baseline is
        # always the frozen prediction tensor after this check passes.
        for start in range(0, 276, 4):
            stop = min(start + 4, 276)
            context = contexts_t[start:stop].to(device).float()
            action = actions_t[start:stop].to(device).float()
            recomputed_correct_parts.append(model.probe(context, action).cpu())
    null_pred_parts = []
    with torch.inference_mode():
        for start in range(0, 276, batch_size):
            stop = min(start + batch_size, 276)
            context = contexts_t[start:stop].to(device).float()
            action = actions_t[start:stop].to(device).float()
            null_pred_parts.append(model.probe(context, torch.zeros_like(action)).cpu())
    recomputed_correct = torch.cat(recomputed_correct_parts)
    null_predictions = torch.cat(null_pred_parts)
    max_correct_diff = float((recomputed_correct - existing_predictions).abs().max())
    recomputed_outcome = retrieval_per_sample(recomputed_correct, targets_t)
    frozen_outcome = retrieval_per_sample(existing_predictions, targets_t)
    if not torch.equal(recomputed_outcome["rank0"], frozen_outcome["rank0"]):
        raise RuntimeError("recomputed correct prediction changes frozen retrieval ranks")
    metric_crosscheck = {
        key: abs(float(recomputed_outcome[key].mean()) - float(frozen_outcome[key].mean()))
        for key in ("cos", "r1", "r5", "mrr", "rank0", "rank1")
    }
    if max(metric_crosscheck.values()) > 1e-6:
        raise RuntimeError(f"recomputed retrieval metric gate failed: {metric_crosscheck}")
    if max_correct_diff > 5e-5:
        raise RuntimeError(f"recomputed correct prediction differs from frozen eval: {max_correct_diff}")
    correct_predictions = existing_predictions

    condition_arrays: Dict[str, Dict[str, np.ndarray]] = {}
    condition_summary: Dict[str, Dict[str, float]] = {}
    condition_arrays["correct_action"], condition_summary["correct_action"] = _condition_metrics(
        correct_predictions, targets_t
    )
    condition_arrays["eval_null_action"], condition_summary["eval_null_action"] = _condition_metrics(
        null_predictions, targets_t
    )
    copy_last = contexts_t[:, -1].float().mean(dim=(1, 2))
    copy_prefix = contexts_t.float().mean(dim=(1, 2, 3))
    condition_arrays["copy_last_streaming_state"], condition_summary["copy_last_streaming_state"] = _condition_metrics(
        copy_last, targets_t
    )
    condition_arrays["copy_prefix_pooled_state"], condition_summary["copy_prefix_pooled_state"] = _condition_metrics(
        copy_prefix, targets_t
    )

    permutation_array = np.stack(mappings)
    permutation_recipients = np.tile(recipients, len(mappings))
    permutation_donors = permutation_array.reshape(-1)
    print(f"layer={layer} running {len(permutation_recipients)} permuted-action interventions", flush=True)
    permutation_flat = _predict_pairs(
        model.probe,
        contexts_t,
        actions_t,
        permutation_recipients,
        permutation_donors,
        targets_t,
        device,
        batch_size,
    )
    permutation = {key: value.reshape(len(mappings), 276) for key, value in permutation_flat.items()}

    wrong_recipients = np.repeat(recipients, 275)
    wrong_donors = np.concatenate([np.concatenate((recipients[:i], recipients[i + 1 :])) for i in recipients])
    print(f"layer={layer} running {len(wrong_recipients)} exhaustive wrong-action interventions", flush=True)
    wrong_flat = _predict_pairs(
        model.probe,
        contexts_t,
        actions_t,
        wrong_recipients,
        wrong_donors,
        targets_t,
        device,
        batch_size,
    )
    wrong = {}
    for key, values in wrong_flat.items():
        matrix = np.full((276, 276), np.nan, dtype=np.float32)
        matrix[wrong_recipients, wrong_donors] = values
        wrong[key] = matrix

    arrays: Dict[str, Any] = {
        "sample_ids": np.asarray(sample_ids, dtype=object),
        "scene_paths": np.asarray(scene_paths, dtype=object),
        "actions": actions_t.numpy(),
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

    permutation_summary = {
        metric: {
            "mean_over_seeds": float(values.mean()),
            "seed_metric_mean": [float(value) for value in values.mean(axis=1)],
        }
        for metric, values in permutation.items()
    }
    correct_cos = condition_arrays["correct_action"]["cos"]
    correct_rank1 = condition_arrays["correct_action"]["rank1"]
    wrong_margin = correct_cos[:, None] - wrong["cos"]
    finite_wrong = np.isfinite(wrong_margin)
    wrong_summary = {
        "mean_correct_minus_wrong_cos": float(np.nanmean(correct_cos[:, None] - wrong["cos"])),
        "median_correct_minus_wrong_cos": float(np.nanmedian(correct_cos[:, None] - wrong["cos"])),
        "positive_margin_fraction": float(np.count_nonzero(wrong_margin[finite_wrong] > 0) / np.count_nonzero(finite_wrong)),
        "mean_rank_improvement": float(np.nanmean(wrong["rank1"] - correct_rank1[:, None])),
        "all_legal_wrong_actions_per_sample": 275,
    }
    summary = {
        "ok": True,
        "run_tag": RUN_TAG,
        "layer": layer,
        "n": 276,
        "n_scenes": 69,
        "correct_prediction_max_abs_diff_from_frozen": max_correct_diff,
        "correct_metric_crosscheck_abs_diff": metric_crosscheck,
        "dataset_target_max_abs_diff_from_frozen": dataset_target_max_abs_diff,
        "conditions": condition_summary,
        "permutation": permutation_summary,
        "wrong_action": wrong_summary,
        "copy_definitions": {
            "copy_last_streaming_state": "spatial mean of final saved streaming-prefix feature map",
            "copy_last_frame_feature": "identical to copy_last_streaming_state because each saved streaming state is a per-frame feature map",
            "copy_prefix_pooled_state": "mean over all prefix timesteps and spatial positions",
            "copy_anchor_isolated": "unavailable in frozen roots; no extraction performed",
        },
        "metric_definitions": {
            "candidate_pool": "all 276 frozen canonical val targets",
            "positive": "sample-id paired target",
            "rank0": "zero-based historical canonical rank",
            "rank1": "one-based report rank",
        },
    }
    json_dump(output / "layer_summary.json", summary)
    provenance = {
        "ok": True,
        "layer": layer,
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "streaming_root": str(streaming_root),
        "target_root": str(target_root),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "existing_predictions": str(existing_path),
        "existing_predictions_sha256": sha256(existing_path),
        "mapping_index": str(mapping_index_path),
        "mapping_index_sha256": sha256(mapping_index_path),
        "mapping_hashes": mapping_hashes,
        "arrays": str(arrays_path),
        "arrays_sha256": sha256(arrays_path),
        "hydra_config": OmegaConf.to_container(cfg, resolve=True),
        "fallback_count": 0,
    }
    json_dump(output / "provenance.json", provenance)
    print(json.dumps({"ok": True, "layer": layer, "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
