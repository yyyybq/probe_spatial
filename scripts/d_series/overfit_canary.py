#!/usr/bin/env python3
"""Single-sample train gate for D-series readout/data/cache integration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

import torch

from scripts.d_series.run_d1 import D1Dataset, masked_l1
from vidfm3d.models.components.probe_layout_readout import PooledDepthReadout, SpatialDepthReadout


def main():
    p = argparse.ArgumentParser(); p.add_argument("--model", required=True); p.add_argument("--reference", default="isolated_target")
    p.add_argument("--branch", choices=["spatial", "pooled"], required=True); p.add_argument("--manifest", type=Path, default=Path("experiments/d_series_scannetpp_v1/data/sample_manifest.jsonl"))
    p.add_argument("--scene-root", type=Path, required=True); p.add_argument("--feature-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--steps", type=int, default=300); args = p.parse_args()
    rows = [json.loads(line) for line in args.manifest.open()]
    row = next(r for r in rows if r["split"] == "train")
    dataset = D1Dataset([row], args.scene_root, args.feature_root, args.model, args.reference, args.branch)
    item = dataset[0]; feature = item["feature"][None]; depth = item["depth"][None]; valid = item["valid"][None]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature, depth, valid = feature.to(device), depth.to(device), valid.to(device)
    readout = (SpatialDepthReadout(feature.shape[-1]) if args.branch == "spatial" else PooledDepthReadout(feature.shape[-1])).to(device)
    optimizer = torch.optim.AdamW(readout.parameters(), lr=1e-3, weight_decay=0)
    with torch.no_grad(): initial = float(masked_l1(readout(feature), depth, valid))
    for _ in range(args.steps):
        loss = masked_l1(readout(feature), depth, valid); optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    with torch.no_grad(): final = float(masked_l1(readout(feature), depth, valid))
    passed = final < 0.25 * initial
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"model": args.model, "reference": args.reference, "branch": args.branch,
                                      "sample_id": row["sample_id"], "steps": args.steps, "initial_mae": initial,
                                      "final_mae": final, "ratio": final / initial, "passed": passed}, indent=2) + "\n")
    if not passed: raise SystemExit(f"overfit gate failed: {final}/{initial}")


if __name__ == "__main__": main()
