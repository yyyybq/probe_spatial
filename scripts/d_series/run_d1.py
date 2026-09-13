#!/usr/bin/env python3
"""Train and evaluate one formal D1 seed; writes only auditable per-sample rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset

from scripts.d_series.feature_store import MANIFEST_SHA256, code_provenance, feature_paths, load_verified, stable_seed
from vidfm3d.models.components.probe_layout_readout import (
    LinearSpatialDepthReadout, PooledDepthReadout, SpatialDepthReadout, parameter_count,
)
from vidfm3d.utils.layout_metrics import (
    coarse_layout_mae, depth_metrics, detail_gradient_mae, normalize_depth_per_sample,
    resize_depth_validity_weighted, scene_macro,
    metric_depth_to_camera_pointmap,
)


REVISIONS = {
    "vjepa2_vitl256": "b3c1679b7c34d3255ef3547f27c7b226aefab26f",
    "wan21_t2v_1p3b": "0fad780a534b6463e45facd96134c9f345acfa5b",
    "dinov2_vitl14": "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c",
}


@lru_cache(maxsize=32)
def _scene_geometry(path_string):
    with np.load(path_string, allow_pickle=False) as values:
        return [str(v) for v in values["images"]], np.asarray(values["intrinsics"], dtype=np.float32)


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class D1Dataset(Dataset):
    def __init__(self, rows, scene_root, feature_root, model, reference, branch, report_branch=None):
        self.rows = rows
        self.scene_root = scene_root
        self.feature_root = feature_root
        self.model = model
        self.reference = reference
        self.branch = branch
        self.report_branch = report_branch or branch

    def __len__(self): return len(self.rows)

    def feature(self, sample_id, space=None, key=None):
        payload, _ = feature_paths(self.feature_root, self.model, space or self.reference, sample_id)
        values = load_file(payload)
        return values[key or self.branch].float()

    def label(self, row):
        depth = torch.from_numpy(np.asarray(Image.open(self.scene_root / row["depth_rel"]), dtype=np.float32).copy()) / 1000.0
        valid = torch.isfinite(depth) & (depth > 0)
        depth, valid = resize_depth_validity_weighted(depth, valid, (72, 128))
        normalized, scale = normalize_depth_per_sample(depth[None], valid[None])
        return normalized[0], valid, scale[0]

    def resized_intrinsic(self, row):
        depth_image = Image.open(self.scene_root / row["depth_rel"])
        width, height = depth_image.size
        names, intrinsics = _scene_geometry(str(self.scene_root / row["metadata_rel"]))
        index = names.index(row["target_frame_name"])
        intrinsic = torch.from_numpy(intrinsics[index].copy())
        intrinsic[0] *= 128.0 / width
        intrinsic[1] *= 72.0 / height
        return intrinsic

    def __getitem__(self, index):
        row = self.rows[index]
        depth, valid, scale = self.label(row)
        return {
            "feature": self.feature(row["sample_id"]), "depth": depth, "valid": valid,
            "scale": scale, "index": index,
        }


def masked_l1(pred, target, valid):
    return ((pred[:, 0] - target).abs() * valid).sum() / valid.sum().clamp_min(1)


def permutations(rows):
    within, across = {}, {}
    groups = defaultdict(list)
    by_horizon = defaultdict(list)
    for row in rows:
        groups[(row["scene_id"], row["horizon"])].append(row["sample_id"])
        by_horizon[row["horizon"]].append(row)
    for values in groups.values():
        ordered = sorted(values)
        for i, value in enumerate(ordered):
            within[value] = ordered[(i + 1) % len(ordered)]
    for horizon, values in by_horizon.items():
        ordered = sorted(values, key=lambda row: row["sample_id"])
        shift = 1
        while shift < len(ordered) and any(row["scene_id"] == ordered[(i + shift) % len(ordered)]["scene_id"] for i, row in enumerate(ordered)):
            shift += 1
        if shift == len(ordered):
            raise ValueError(f"cannot build cross-scene permutation for h{horizon}")
        for i, row in enumerate(ordered):
            across[row["sample_id"]] = ordered[(i + shift) % len(ordered)]["sample_id"]
    return within, across


def evaluate(readout, dataset, rows, prior, device, seed, source_run):
    readout.eval()
    within, across = permutations(rows)
    records = []
    generator = torch.Generator().manual_seed(seed)
    with torch.inference_mode():
        for index, row in enumerate(rows):
            target, valid, scale = dataset.label(row)
            target_b, valid_b = target[None].to(device), valid[None].to(device)
            intrinsic = dataset.resized_intrinsic(row).to(device)
            target_pointmap = metric_depth_to_camera_pointmap(target_b[0] * scale.to(device), intrinsic)
            true = dataset.feature(row["sample_id"]).to(device)
            spatial_for_shuffle = dataset.feature(row["sample_id"], key="spatial")
            flat_spatial = spatial_for_shuffle.reshape(-1, spatial_for_shuffle.shape[-1])
            shuffle_gen = torch.Generator().manual_seed(stable_seed(row["sample_id"], "spatial_token_shuffle"))
            shuffled_spatial = flat_spatial[torch.randperm(flat_spatial.shape[0], generator=shuffle_gen)].reshape(spatial_for_shuffle.shape).to(device)
            shuffled_input = shuffled_spatial if dataset.branch == "spatial" else shuffled_spatial.mean(tuple(range(shuffled_spatial.ndim - 1)))
            context_key = "context_spatial" if dataset.branch == "spatial" else None
            context_payload, _ = feature_paths(dataset.feature_root, dataset.model, "prefix", row["parent_sample_id"])
            context_values = load_file(context_payload)
            last = (context_values[context_key].float() if context_key else context_values["context_pooled_sequence"][-1].float()).to(device)
            conditions = {
                "true_target": readout(true[None]),
                "within_scene_wrong_target": readout(dataset.feature(within[row["sample_id"]]).to(device)[None]),
                "across_scene_shuffled_target": readout(dataset.feature(across[row["sample_id"]]).to(device)[None]),
                "last_context": readout(last[None]),
                "zero": readout(torch.zeros_like(true)[None]),
                "random": readout(torch.randn(true.shape, generator=generator).to(device)[None]),
                "spatial_token_shuffle": readout(shuffled_input[None]),
                "train_layout_prior": prior[None, None].to(device),
            }
            for condition, prediction in conditions.items():
                metric = depth_metrics(prediction, target_b, valid_b)[0]
                metric["coarse_mae"] = float(coarse_layout_mae(prediction, target_b, valid_b)[0])
                metric["detail_mae"] = float(detail_gradient_mae(prediction, target_b, valid_b)[0])
                oracle_pred_pointmap = metric_depth_to_camera_pointmap(prediction[0, 0] * scale.to(device), intrinsic)
                metric["oracle_gt_scale_pointmap_mae_m"] = float(
                    torch.linalg.vector_norm(oracle_pred_pointmap - target_pointmap, dim=-1)[valid_b[0]].mean()
                )
                records.append({
                    "schema_version": 1, "experiment": "D1", "protocol": "d_series_scannetpp_v1",
                    "evaluation_status": "development_only", "model": dataset.model,
                    "reference": dataset.reference, "branch": dataset.report_branch, "condition": condition,
                    "seed": seed, "split": row["split"], "manifest_sha256": MANIFEST_SHA256,
                    "sample_id": row["sample_id"], "parent_sample_id": row["parent_sample_id"],
                    "scene_id": row["scene_id"], "window_id": row["window_id"],
                    "horizon": row["horizon"], "target_raw_frame_id": row["target_raw_frame_id"],
                    "target_observation_index": row["target_observation_index"], "target_temporal_token": None,
                    "raw_frame_delta": row["raw_frame_delta"], "translation_m": row["translation_m"],
                    "rotation_deg": row["rotation_deg"], "directed_depth_overlap": row["directed_depth_overlap"],
                    "target_layout_scale_m": float(scale), "source_run": source_run, **metric,
                })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(REVISIONS), required=True)
    parser.add_argument("--reference", choices=["isolated_target", "native_target"], default="isolated_target")
    parser.add_argument("--branch", choices=["spatial", "pooled", "linear_spatial"], required=True)
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--manifest", type=Path, default=Path("experiments/d_series_scannetpp_v1/data/sample_manifest.jsonl"))
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if sha256(args.manifest) != MANIFEST_SHA256:
        raise ValueError("manifest hash mismatch")
    if args.reference == "native_target" and args.model != "vjepa2_vitl256":
        raise ValueError("native_target is V-JEPA2 only")
    branch = "spatial" if args.branch == "linear_spatial" else args.branch
    rows = [json.loads(line) for line in args.manifest.open()]
    train_rows = [row for row in rows if row["split"] == "train"]
    dev_rows = [row for row in rows if row["split"] == "dev"]

    # Fail closed on every feature/receipt before training, including dev prefix controls.
    for row in rows:
        load_verified(args.feature_root, args.model, args.reference, row["sample_id"], REVISIONS[args.model])
    for parent in sorted({row["parent_sample_id"] for row in rows}):
        load_verified(args.feature_root, args.model, "prefix", parent, REVISIONS[args.model])

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = D1Dataset(train_rows, args.scene_root, args.feature_root, args.model, args.reference, branch, args.branch)
    dev = D1Dataset(dev_rows, args.scene_root, args.feature_root, args.model, args.reference, branch, args.branch)
    example = train.feature(train_rows[0]["sample_id"])
    channels = example.shape[-1]
    if args.branch == "spatial": readout = SpatialDepthReadout(channels)
    elif args.branch == "linear_spatial": readout = LinearSpatialDepthReadout(channels)
    else: readout = PooledDepthReadout(channels)
    readout.to(device)
    optimizer = torch.optim.AdamW(readout.parameters(), lr=3e-4, weight_decay=1e-4)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda")

    prior_num = torch.zeros(72, 128); prior_den = torch.zeros(72, 128)
    for row in train_rows:
        depth, valid, _ = train.label(row)
        prior_num += depth * valid; prior_den += valid
    prior = prior_num / prior_den.clamp_min(1)
    best, patience_left, best_state, history = float("inf"), args.patience, None, []
    source_run = args.output_dir.name
    for epoch in range(args.max_epochs):
        readout.train()
        for batch in loader:
            prediction = readout(batch["feature"].to(device, non_blocking=True))
            loss = masked_l1(prediction, batch["depth"].to(device), batch["valid"].to(device))
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        dev_records = evaluate(readout, dev, dev_rows, prior, device, args.seed, source_run)
        score = scene_macro((row for row in dev_records if row["condition"] == "true_target"), "mae")
        history.append({"epoch": epoch, "dev_scene_macro_mae": score})
        print(json.dumps(history[-1]), flush=True)
        if score < best - 1e-6:
            best, patience_left = score, args.patience
            best_state = {key: value.detach().cpu().clone() for key, value in readout.state_dict().items()}
            best_epoch = epoch
        else:
            patience_left -= 1
            if patience_left == 0: break
    readout.load_state_dict(best_state)
    records = evaluate(readout, dev, dev_rows, prior, device, args.seed, source_run)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.save({"state_dict": best_state, "train_layout_prior": prior, "args": vars(args)}, args.output_dir / "checkpoint.pt")
    with (args.output_dir / "per_sample_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(records)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    receipt = {
        "schema_version": 1, "experiment": "D1", "status": "complete_development_evaluation",
        "source_run": source_run, "model": args.model, "model_revision": REVISIONS[args.model],
        "reference": args.reference, "branch": args.branch, "seed": args.seed,
        "manifest_sha256": MANIFEST_SHA256, "train_scenes_sha256": sha256(args.manifest.parent / "train_scenes.txt"),
        "dev_scenes_sha256": sha256(args.manifest.parent / "dev_scenes.txt"), "test_scenes": 0,
        "parameter_count": parameter_count(readout), "best_epoch": best_epoch,
        "dev_scene_macro_mae": best, "n_dev": len(dev_rows), "n_dev_scenes": len({r['scene_id'] for r in dev_rows}),
        "git_base_commit": commit, "git_dirty": bool(subprocess.check_output(["git", "status", "--short"], text=True).strip()),
        "history": history,
    } | code_provenance([
        Path(__file__), REPO_ROOT / "vidfm3d/models/components/probe_layout_readout.py",
        REPO_ROOT / "vidfm3d/utils/layout_metrics.py", REPO_ROOT / "scripts/d_series/feature_store.py",
    ])
    (args.output_dir / "run_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__": main()
