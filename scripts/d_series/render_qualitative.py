#!/usr/bin/env python3
"""Render predeclared best/median/worst V-JEPA2 D3 panels with fixed scales."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from scripts.d_series.feature_store import MANIFEST_SHA256, feature_paths
from scripts.d_series.run_d1 import D1Dataset
from vidfm3d.models.components.probe_layout_readout import PooledDepthReadout, SpatialDepthReadout


def find_d1(run_root, branch):
    matches = []
    for path in run_root.rglob("run_receipt.json"):
        row = json.loads(path.read_text())
        if row.get("experiment") == "D1" and row.get("model") == "vjepa2_vitl256" and row.get("reference") == "native_target" and row.get("branch") == branch and row.get("seed") == 0:
            matches.append(path.parent / "checkpoint.pt")
    if len(matches) != 1: raise RuntimeError(f"expected one seed-0 native D1 checkpoint for {branch}, got {matches}")
    return matches[0]


def main():
    p = argparse.ArgumentParser(); p.add_argument("--manifest", type=Path, default=Path("experiments/d_series_scannetpp_v1/data/sample_manifest.jsonl"))
    p.add_argument("--visual-index", type=Path, required=True); p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--feature-root", type=Path, required=True); p.add_argument("--scene-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True); args = p.parse_args()
    manifest = {row["sample_id"]: row for row in (json.loads(line) for line in args.manifest.open())}
    selections = list(csv.DictReader(args.visual_index.open())); args.output_dir.mkdir(parents=True, exist_ok=True)
    for branch in ("spatial", "pooled"):
        checkpoint = torch.load(find_d1(args.run_root, branch), map_location="cpu", weights_only=False)
        target = load_file(feature_paths(args.feature_root, "vjepa2_vitl256", "native_target", selections[0]["sample_id"])[0])
        channels = target[branch].shape[-1]
        readout = SpatialDepthReadout(channels) if branch == "spatial" else PooledDepthReadout(channels)
        readout.load_state_dict(checkpoint["state_dict"]); readout.eval(); prior = checkpoint["train_layout_prior"].numpy()
        subset = [item for item in selections if item["branch"] == branch]
        for item in subset:
            row = manifest[item["sample_id"]]
            dataset = D1Dataset([row], args.scene_root, args.feature_root, "vjepa2_vitl256", "native_target", branch)
            gt, valid, _ = dataset.label(row)
            zt = load_file(feature_paths(args.feature_root, "vjepa2_vitl256", "native_target", row["sample_id"])[0])[branch].float()
            zp = load_file(feature_paths(args.feature_root, "vjepa2_vitl256", "native_pred", row["sample_id"])[0])[branch].float()
            with torch.inference_mode(): target_depth = readout(zt[None])[0, 0].numpy(); pred_depth = readout(zp[None])[0, 0].numpy()
            gt_np, mask = gt.numpy(), valid.numpy(); error = np.where(mask, np.abs(pred_depth - gt_np), np.nan)
            rgb = np.asarray(Image.open(args.scene_root / row["rgb_rel"]).convert("RGB"))
            fig, axes = plt.subplots(1, 6, figsize=(15, 2.8))
            axes[0].imshow(rgb); axes[0].set_title("target RGB")
            for ax, value, title in zip(axes[1:5], (gt_np, prior, target_depth, pred_depth), ("GT depth", "train prior", "target readout", "pred readout")):
                ax.imshow(np.where(mask, value, np.nan), cmap="viridis", vmin=0, vmax=2.5); ax.set_title(title)
            axes[5].imshow(error, cmap="magma", vmin=0, vmax=1.0); axes[5].set_title("|pred-GT|")
            for ax in axes: ax.axis("off")
            fig.suptitle(f"{branch} h{row['horizon']} {item['selection_rule']} {row['sample_id']}", fontsize=9)
            fig.tight_layout(); stem = f"{branch}_h{row['horizon']}_{item['selection_rule']}_{row['sample_id']}"
            fig.savefig(args.output_dir / f"{stem}.png", dpi=180); fig.savefig(args.output_dir / f"{stem}.pdf"); plt.close(fig)


if __name__ == "__main__": main()
