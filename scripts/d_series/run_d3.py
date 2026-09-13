#!/usr/bin/env python3
"""Evaluate V-JEPA2 native-space frozen transfer, linear alignment, and recovery."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset

from scripts.d_series.feature_store import MANIFEST_SHA256, code_provenance, feature_paths, load_verified
from scripts.d_series.run_d1 import D1Dataset, REVISIONS, masked_l1, permutations, sha256
from vidfm3d.models.components.probe_layout_readout import (
    PooledDepthReadout, SpatialDepthReadout, TokenwiseLinearAdapter, parameter_count,
)
from vidfm3d.utils.layout_metrics import (
    coarse_layout_mae, depth_metrics, detail_gradient_mae, latent_diagnostics, metric_depth_to_camera_pointmap, scene_macro,
)


MODEL = "vjepa2_vitl256"


class D3TrainDataset(Dataset):
    def __init__(self, rows, depth_dataset, root, branch):
        self.rows, self.depth_dataset, self.root, self.branch = rows, depth_dataset, root, branch
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        pred = load_file(feature_paths(self.root, MODEL, "native_pred", row["sample_id"])[0])[self.branch].float()
        target = load_file(feature_paths(self.root, MODEL, "native_target", row["sample_id"])[0])[self.branch].float()
        depth, valid, _ = self.depth_dataset.label(row)
        return {"pred": pred, "target": target, "depth": depth, "valid": valid, "index": index}


def evaluate(target_readout, retrained, adapter, depth_dataset, rows, root, branch, prior, device, seed, source_run):
    for module in (target_readout, retrained, adapter): module.eval()
    within, _ = permutations(rows); records = []
    with torch.inference_mode():
        for row in rows:
            def feat(space, sample=row["sample_id"]):
                return load_file(feature_paths(root, MODEL, space, sample)[0])[branch].float().to(device)[None]
            target, pred, shuffled = feat("native_target"), feat("native_pred"), feat("native_pred_order_shuffle")
            prefix = load_file(feature_paths(root, MODEL, "prefix", row["parent_sample_id"])[0])
            last = prefix["context_spatial" if branch == "spatial" else "context_pooled_sequence"].float()
            if branch == "pooled": last = last[-1]
            last = last.to(device)[None]
            wrong = feat("native_target", within[row["sample_id"]])
            outputs = {
                "native_target_reference": target_readout(target),
                "frozen_readout_transfer": target_readout(pred),
                "tokenwise_linear_adapter": target_readout(adapter(pred)),
                "retrained_same_capacity_readout": retrained(pred),
                "last_context": target_readout(last),
                "train_layout_prior": prior[None, None].to(device),
                "within_scene_wrong_target": target_readout(wrong),
                "early_tubelet_order_shuffle": target_readout(shuffled),
            }
            depth, valid, scale = depth_dataset.label(row); db, vb = depth[None].to(device), valid[None].to(device)
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
                latent = pred if condition not in {"native_target_reference", "within_scene_wrong_target", "last_context"} else None
                diag = {"cosine": None, "pred_variance": None, "target_variance": float(target.var(unbiased=False))}
                if latent is not None:
                    value = adapter(latent) if condition == "tokenwise_linear_adapter" else (shuffled if condition == "early_tubelet_order_shuffle" else latent)
                    diag = {key: float(v[0]) for key, v in latent_diagnostics(value, target).items()}
                records.append({
                    "schema_version": 1, "experiment": "D3", "protocol": "d_series_scannetpp_v1",
                    "evaluation_status": "development_only", "model": MODEL, "reference": "native_target",
                    "branch": branch, "condition": condition, "seed": seed, "split": row["split"],
                    "manifest_sha256": MANIFEST_SHA256, "sample_id": row["sample_id"],
                    "parent_sample_id": row["parent_sample_id"], "scene_id": row["scene_id"],
                    "window_id": row["window_id"], "horizon": row["horizon"],
                    "target_raw_frame_id": row["target_raw_frame_id"], "target_layout_scale_m": float(scale),
                    "source_run": source_run, **metric, **diag,
                })
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--branch", choices=["spatial", "pooled"], required=True)
    p.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    p.add_argument("--manifest", type=Path, default=Path("experiments/d_series_scannetpp_v1/data/sample_manifest.jsonl"))
    p.add_argument("--scene-root", type=Path, required=True); p.add_argument("--feature-root", type=Path, required=True)
    p.add_argument("--native-d1-checkpoint", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=16); p.add_argument("--max-epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=6); p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()
    if sha256(args.manifest) != MANIFEST_SHA256: raise ValueError("manifest hash mismatch")
    rows = [json.loads(line) for line in args.manifest.open()]
    train_rows, dev_rows = ([r for r in rows if r["split"] == split] for split in ("train", "dev"))
    for row in rows:
        for space in ("native_target", "native_pred", "native_pred_order_shuffle"):
            load_verified(args.feature_root, MODEL, space, row["sample_id"], REVISIONS[MODEL])
    for parent in sorted({row["parent_sample_id"] for row in rows}):
        load_verified(args.feature_root, MODEL, "prefix", parent, REVISIONS[MODEL])
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depth_train = D1Dataset(train_rows, args.scene_root, args.feature_root, MODEL, "native_target", args.branch)
    depth_dev = D1Dataset(dev_rows, args.scene_root, args.feature_root, MODEL, "native_target", args.branch)
    train = D3TrainDataset(train_rows, depth_train, args.feature_root, args.branch)
    dev = D3TrainDataset(dev_rows, depth_dev, args.feature_root, args.branch)
    channels = train[0]["target"].shape[-1]
    make_readout = lambda: SpatialDepthReadout(channels) if args.branch == "spatial" else PooledDepthReadout(channels)
    checkpoint = torch.load(args.native_d1_checkpoint, map_location="cpu", weights_only=False)
    target_readout = make_readout().to(device); target_readout.load_state_dict(checkpoint["state_dict"]); target_readout.requires_grad_(False)
    prior = checkpoint["train_layout_prior"]
    retrained, adapter = make_readout().to(device), TokenwiseLinearAdapter(channels).to(device)
    optim_r = torch.optim.AdamW(retrained.parameters(), lr=3e-4, weight_decay=1e-4)
    optim_a = torch.optim.AdamW(adapter.parameters(), lr=1e-4, weight_decay=1e-4)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda")
    best_r = best_a = float("inf"); left_r = left_a = args.patience; state_r = state_a = None; history = []
    source_run = args.output_dir.name
    for epoch in range(args.max_epochs):
        retrained.train(); adapter.train()
        for batch in loader:
            pred, target = batch["pred"].to(device), batch["target"].to(device)
            loss_a = torch.nn.functional.mse_loss(adapter(pred), target)
            optim_a.zero_grad(set_to_none=True); loss_a.backward(); optim_a.step()
            output = retrained(pred)
            loss_r = masked_l1(output, batch["depth"].to(device), batch["valid"].to(device))
            optim_r.zero_grad(set_to_none=True); loss_r.backward(); optim_r.step()
        adapter.eval(); retrained.eval(); latent_losses = []
        with torch.inference_mode():
            for batch in DataLoader(dev, batch_size=args.batch_size, num_workers=args.num_workers):
                latent_losses.append(torch.nn.functional.mse_loss(adapter(batch["pred"].to(device)), batch["target"].to(device)).item())
        records = evaluate(target_readout, retrained, adapter, depth_dev, dev_rows, args.feature_root, args.branch,
                           prior, device, args.seed, source_run)
        score_r = scene_macro((r for r in records if r["condition"] == "retrained_same_capacity_readout"), "mae")
        score_a = float(np.mean(latent_losses)); history.append({"epoch": epoch, "adapter_dev_mse": score_a, "retrained_dev_scene_macro_mae": score_r})
        print(json.dumps(history[-1]), flush=True)
        if score_a < best_a - 1e-8:
            best_a, left_a, epoch_a = score_a, args.patience, epoch
            state_a = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
        else: left_a -= 1
        if score_r < best_r - 1e-6:
            best_r, left_r, epoch_r = score_r, args.patience, epoch
            state_r = {k: v.detach().cpu().clone() for k, v in retrained.state_dict().items()}
        else: left_r -= 1
        if left_a == 0 and left_r == 0: break
    adapter.load_state_dict(state_a); retrained.load_state_dict(state_r)
    records = evaluate(target_readout, retrained, adapter, depth_dev, dev_rows, args.feature_root, args.branch,
                       prior, device, args.seed, source_run)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.save({"adapter_state_dict": state_a, "retrained_readout_state_dict": state_r}, args.output_dir / "checkpoint.pt")
    with (args.output_dir / "per_sample_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(records)
    receipt = {"schema_version": 1, "experiment": "D3", "status": "complete_development_evaluation",
               "source_run": source_run, "model": MODEL, "model_revision": REVISIONS[MODEL], "branch": args.branch,
               "seed": args.seed, "manifest_sha256": MANIFEST_SHA256, "native_d1_checkpoint": str(args.native_d1_checkpoint.resolve()),
               "adapter_parameters": parameter_count(adapter), "retrained_readout_parameters": parameter_count(retrained),
               "adapter_best_epoch": epoch_a, "adapter_dev_mse": best_a, "retrained_best_epoch": epoch_r,
               "retrained_dev_scene_macro_mae": best_r, "n_dev": len(dev_rows),
               "n_dev_scenes": len({r['scene_id'] for r in dev_rows}), "history": history} | code_provenance([
                   Path(__file__), REPO_ROOT / "scripts/d_series/run_d1.py",
                   REPO_ROOT / "vidfm3d/models/components/probe_layout_readout.py",
                   REPO_ROOT / "vidfm3d/utils/layout_metrics.py",
               ])
    (args.output_dir / "run_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__": main()
