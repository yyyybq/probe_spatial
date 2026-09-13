#!/usr/bin/env python3
"""Train matched motion-conditioned/context-only C1 controls and evaluate D2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset

from scripts.d_series.feature_store import MANIFEST_SHA256, code_provenance, feature_paths, load_verified
from scripts.d_series.run_d1 import D1Dataset, REVISIONS, sha256
from vidfm3d.models.components.probe_layout_readout import ExternalPooledPredictor, PooledDepthReadout, parameter_count
from vidfm3d.utils.layout_metrics import coarse_layout_mae, depth_metrics, detail_gradient_mae, latent_diagnostics, metric_depth_to_camera_pointmap, scene_macro


IDENTITY_3X4 = torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float32)


class PredictorDataset(Dataset):
    def __init__(self, rows, feature_root, model): self.rows, self.root, self.model = rows, feature_root, model
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        prefix = load_file(feature_paths(self.root, self.model, "prefix", row["parent_sample_id"])[0])
        target = load_file(feature_paths(self.root, self.model, "isolated_target", row["sample_id"])[0])
        return {
            "context": prefix["context_pooled_sequence"].float(), "target": target["pooled"].float(),
            "motion": torch.tensor(row["motion_target_from_context_3x4"], dtype=torch.float32), "index": index,
        }


def train_stats(dataset):
    targets, motions = [], []
    for i in range(len(dataset)):
        item = dataset[i]; targets.append(item["target"]); motions.append(item["motion"])
    targets, motions = torch.stack(targets), torch.stack(motions)
    return targets.mean(0), targets.std(0, unbiased=False).clamp_min(1e-6), motions.mean(0), motions.std(0, unbiased=False).clamp_min(1e-6)


def shuffled_motion_map(train_rows, dev_rows):
    thresholds = {}
    for horizon in (2, 4):
        values = np.array([row["translation_m"] for row in train_rows if row["horizon"] == horizon])
        thresholds[horizon] = np.quantile(values, [1/3, 2/3]).tolist()
    groups = {}
    for row in dev_rows:
        a, b = thresholds[row["horizon"]]
        key = (row["horizon"], int(row["translation_m"] > a) + int(row["translation_m"] > b))
        groups.setdefault(key, []).append(row)
    result = {}
    rng = random.Random(42)
    for values in groups.values():
        ordered = sorted(values, key=lambda row: row["sample_id"])
        if len(ordered) == 1:
            candidates = [r for r in dev_rows if r["horizon"] == ordered[0]["horizon"] and r["sample_id"] != ordered[0]["sample_id"]]
            result[ordered[0]["sample_id"]] = min(candidates, key=lambda r: abs(r["translation_m"] - ordered[0]["translation_m"]))
        else:
            shift = rng.randrange(1, len(ordered))
            for i, row in enumerate(ordered): result[row["sample_id"]] = ordered[(i + shift) % len(ordered)]
    return result, thresholds


def predictor_latent(model, context, motion, tmean, tstd, mmean, mstd):
    normalized = model(context, (motion - mmean) / mstd)
    return normalized * tstd + tmean


def evaluate(models, readout, depth_dataset, pred_dataset, rows, prior, stats, shuffle, device, seed, source_run):
    tmean, tstd, mmean, mstd = [value.to(device) for value in stats]
    for module in [*models.values(), readout]: module.eval()
    records = []
    with torch.inference_mode():
        for i, row in enumerate(rows):
            item = pred_dataset[i]
            context, target = item["context"][None].to(device), item["target"][None].to(device)
            motion = item["motion"][None].to(device)
            shuffled = torch.tensor(shuffle[row["sample_id"]]["motion_target_from_context_3x4"], dtype=torch.float32, device=device)[None]
            predicted = {
                "true_motion": predictor_latent(models["motion"], context, motion, tmean, tstd, mmean, mstd),
                "identity_motion": predictor_latent(models["motion"], context, IDENTITY_3X4.to(device)[None], tmean, tstd, mmean, mstd),
                "matched_horizon_shuffled_motion": predictor_latent(models["motion"], context, shuffled, tmean, tstd, mmean, mstd),
                "independently_trained_context_only": predictor_latent(models["context_only"], context, motion, tmean, tstd, mmean, mstd),
            }
            outputs = {name: readout(value) for name, value in predicted.items()}
            outputs["last_context"] = readout(context[:, -1])
            outputs["true_pooled_target_reference"] = readout(target)
            outputs["train_layout_prior"] = prior[None, None].to(device)
            depth, valid, scale = depth_dataset.label(row)
            db, vb = depth[None].to(device), valid[None].to(device)
            intrinsic = depth_dataset.resized_intrinsic(row).to(device)
            target_pointmap = metric_depth_to_camera_pointmap(db[0] * scale.to(device), intrinsic)
            for condition, output in outputs.items():
                metric = depth_metrics(output, db, vb)[0]
                metric["coarse_mae"] = float(coarse_layout_mae(output, db, vb)[0])
                metric["detail_mae"] = float(detail_gradient_mae(output, db, vb)[0])
                oracle_pointmap = metric_depth_to_camera_pointmap(output[0, 0] * scale.to(device), intrinsic)
                metric["oracle_gt_scale_pointmap_mae_m"] = float(
                    torch.linalg.vector_norm(oracle_pointmap - target_pointmap, dim=-1)[vb[0]].mean()
                )
                diag = {"cosine": None, "pred_variance": None, "target_variance": float(target.var(unbiased=False))}
                if condition in predicted:
                    values = latent_diagnostics(predicted[condition], target)
                    diag = {key: float(value[0]) for key, value in values.items()}
                records.append({
                    "schema_version": 1, "experiment": "D2", "protocol": "d_series_scannetpp_v1",
                    "evaluation_status": "development_only", "model": pred_dataset.model, "branch": "pooled",
                    "condition": condition, "seed": seed, "split": row["split"], "manifest_sha256": MANIFEST_SHA256,
                    "sample_id": row["sample_id"], "parent_sample_id": row["parent_sample_id"], "scene_id": row["scene_id"],
                    "window_id": row["window_id"], "horizon": row["horizon"], "target_raw_frame_id": row["target_raw_frame_id"],
                    "translation_m": row["translation_m"], "rotation_deg": row["rotation_deg"],
                    "directed_depth_overlap": row["directed_depth_overlap"], "target_layout_scale_m": float(scale),
                    "source_run": source_run, **metric, **diag,
                })
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=sorted(REVISIONS), required=True)
    p.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    p.add_argument("--manifest", type=Path, default=Path("experiments/d_series_scannetpp_v1/data/sample_manifest.jsonl"))
    p.add_argument("--scene-root", type=Path, required=True); p.add_argument("--feature-root", type=Path, required=True)
    p.add_argument("--d1-checkpoint", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=32); p.add_argument("--max-epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=6); p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()
    if sha256(args.manifest) != MANIFEST_SHA256: raise ValueError("manifest hash mismatch")
    rows = [json.loads(line) for line in args.manifest.open()]
    train_rows, dev_rows = ([r for r in rows if r["split"] == split] for split in ("train", "dev"))
    for row in rows:
        load_verified(args.feature_root, args.model, "isolated_target", row["sample_id"], REVISIONS[args.model])
    for parent in sorted({row["parent_sample_id"] for row in rows}):
        load_verified(args.feature_root, args.model, "prefix", parent, REVISIONS[args.model])
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = PredictorDataset(train_rows, args.feature_root, args.model); dev = PredictorDataset(dev_rows, args.feature_root, args.model)
    stats = train_stats(train); channels = stats[0].numel()
    models = {
        name: ExternalPooledPredictor(channels, conditioning=name).to(device)
        for name in ("motion", "context_only")
    }
    optimizers = {name: torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4) for name, model in models.items()}
    checkpoint = torch.load(args.d1_checkpoint, map_location="cpu", weights_only=False)
    readout = PooledDepthReadout(channels).to(device); readout.load_state_dict(checkpoint["state_dict"]); readout.requires_grad_(False)
    prior = checkpoint["train_layout_prior"]
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda")
    tmean, tstd, mmean, mstd = [v.to(device) for v in stats]
    shuffle, thresholds = shuffled_motion_map(train_rows, dev_rows)
    best, remaining, states, history = float("inf"), args.patience, None, []
    source_run = args.output_dir.name
    for epoch in range(args.max_epochs):
        for model in models.values(): model.train()
        for batch in loader:
            context, target, motion = (batch[k].to(device) for k in ("context", "target", "motion"))
            target = (target - tmean) / tstd
            for name, model in models.items():
                loss = torch.nn.functional.mse_loss(model(context, (motion - mmean) / mstd), target)
                optimizers[name].zero_grad(set_to_none=True); loss.backward(); optimizers[name].step()
        records = evaluate(models, readout, D1Dataset(dev_rows, args.scene_root, args.feature_root, args.model, "isolated_target", "pooled"),
                           dev, dev_rows, prior, stats, shuffle, device, args.seed, source_run)
        score = scene_macro((r for r in records if r["condition"] == "true_motion"), "mae")
        history.append({"epoch": epoch, "dev_scene_macro_mae": score}); print(json.dumps(history[-1]), flush=True)
        if score < best - 1e-6:
            best, remaining, best_epoch = score, args.patience, epoch
            states = {name: {k: v.detach().cpu().clone() for k, v in model.state_dict().items()} for name, model in models.items()}
        else:
            remaining -= 1
            if remaining == 0: break
    for name in models: models[name].load_state_dict(states[name])
    depth_dataset = D1Dataset(dev_rows, args.scene_root, args.feature_root, args.model, "isolated_target", "pooled")
    records = evaluate(models, readout, depth_dataset, dev, dev_rows, prior, stats, shuffle, device, args.seed, source_run)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.save({"state_dict": states, "target_mean": stats[0], "target_std": stats[1], "motion_mean": stats[2],
                "motion_std": stats[3], "motion_bin_thresholds": thresholds}, args.output_dir / "checkpoint.pt")
    with (args.output_dir / "per_sample_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(records)
    receipt = {"schema_version": 1, "experiment": "D2", "status": "complete_development_evaluation",
               "source_run": source_run, "model": args.model, "model_revision": REVISIONS[args.model], "seed": args.seed,
               "manifest_sha256": MANIFEST_SHA256, "d1_checkpoint": str(args.d1_checkpoint.resolve()),
               "motion_predictor_parameters": parameter_count(models["motion"]),
               "context_only_parameters": parameter_count(models["context_only"]), "best_epoch": best_epoch,
               "dev_scene_macro_mae": best, "n_dev": len(dev_rows), "n_dev_scenes": len({r['scene_id'] for r in dev_rows}),
               "motion_shuffle_seed": 42, "motion_bin_thresholds": thresholds, "history": history} | code_provenance([
                   Path(__file__), REPO_ROOT / "scripts/d_series/run_d1.py",
                   REPO_ROOT / "vidfm3d/models/components/probe_layout_readout.py",
                   REPO_ROOT / "vidfm3d/utils/layout_metrics.py",
               ])
    (args.output_dir / "run_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__": main()
