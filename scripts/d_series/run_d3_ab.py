#!/usr/bin/env python3
"""Evaluate seed0 V-JEPA2 D3 A/B plus matched depth-only controls."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from safetensors.torch import load_file

from scripts.d_series.feature_store import MANIFEST_SHA256, code_provenance, feature_paths, load_verified
from scripts.d_series.run_d1 import D1Dataset, REVISIONS, permutations, sha256
from vidfm3d.models.components.probe_layout_readout import PooledDepthReadout, SpatialDepthReadout, parameter_count
from vidfm3d.utils.layout_metrics import coarse_layout_mae, depth_metrics, detail_gradient_mae, latent_diagnostics

MODEL = "vjepa2_vitl256"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=("spatial", "pooled"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--manifest", type=Path, default=Path("experiments/d_series_scannetpp_v1/data/sample_manifest.jsonl"))
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--native-d1-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-audit-receipt", type=Path)
    args = parser.parse_args()

    if sha256(args.manifest) != MANIFEST_SHA256:
        raise ValueError("manifest hash mismatch")
    rows = [json.loads(line) for line in args.manifest.open()]
    dev_rows = [row for row in rows if row["split"] == "dev"]
    if args.feature_audit_receipt is not None:
        audit = json.loads(args.feature_audit_receipt.read_text())
        expected_samples = {"expected": len(rows), "complete": len(rows)}
        expected_parents = {"expected": len({row['parent_sample_id'] for row in rows}),
                            "complete": len({row['parent_sample_id'] for row in rows})}
        if (audit.get("status") != "COMPLETE_VALID" or audit.get("manifest_sha256") != MANIFEST_SHA256
                or audit.get("model") != MODEL or audit.get("roles", {}).get("native_target") != expected_samples
                or audit.get("roles", {}).get("native_pred") != expected_samples
                or audit.get("roles", {}).get("prefix") != expected_parents):
            raise ValueError("feature audit receipt does not cover D3 A/B")
    else:
        for row in rows:
            for space in ("native_target", "native_pred"):
                load_verified(args.feature_root, MODEL, space, row["sample_id"], REVISIONS[MODEL])
        for parent in sorted({row["parent_sample_id"] for row in rows}):
            load_verified(args.feature_root, MODEL, "prefix", parent, REVISIONS[MODEL])

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depth_dataset = D1Dataset(dev_rows, args.scene_root, args.feature_root, MODEL, "native_target", args.branch)
    checkpoint = torch.load(args.native_d1_checkpoint, map_location="cpu", weights_only=False)
    example = load_file(feature_paths(args.feature_root, MODEL, "native_target", dev_rows[0]["sample_id"])[0])[args.branch]
    readout = (SpatialDepthReadout(example.shape[-1]) if args.branch == "spatial" else PooledDepthReadout(example.shape[-1])).to(device)
    readout.load_state_dict(checkpoint["state_dict"]); readout.requires_grad_(False); readout.eval()
    prior = checkpoint["train_layout_prior"].to(device)
    within, _ = permutations(dev_rows)
    records = []
    with torch.inference_mode():
        for row in dev_rows:
            def feat(space, sample_id=row["sample_id"]):
                return load_file(feature_paths(args.feature_root, MODEL, space, sample_id)[0])[args.branch].float().to(device)[None]
            target_feature, pred_feature = feat("native_target"), feat("native_pred")
            wrong_feature = feat("native_target", within[row["sample_id"]])
            prefix = load_file(feature_paths(args.feature_root, MODEL, "prefix", row["parent_sample_id"])[0])
            last = prefix["context_spatial" if args.branch == "spatial" else "context_pooled_sequence"].float()
            if args.branch == "pooled":
                last = last[-1]
            outputs = {
                "native_target_reference": readout(target_feature),
                "frozen_readout_transfer": readout(pred_feature),
                "last_context": readout(last.to(device)[None]),
                "train_layout_prior": prior[None, None],
                "within_scene_wrong_target": readout(wrong_feature),
            }
            depth, valid, scale = depth_dataset.label(row)
            depth, valid = depth[None].to(device), valid[None].to(device)
            for condition, output in outputs.items():
                metrics = depth_metrics(output, depth, valid)[0]
                metrics["coarse_mae"] = float(coarse_layout_mae(output, depth, valid)[0])
                metrics["detail_mae"] = float(detail_gradient_mae(output, depth, valid)[0])
                metrics["oracle_gt_scale_pointmap_mae_m"] = ""
                diagnostics = {"cosine": "", "pred_variance": "", "target_variance": float(target_feature.var(unbiased=False))}
                if condition == "frozen_readout_transfer":
                    diagnostics = {key: float(value[0]) for key, value in latent_diagnostics(pred_feature, target_feature).items()}
                records.append({
                    "schema_version": 1, "experiment": "D3", "phase": "AB", "protocol": "d_series_scannetpp_v1",
                    "evaluation_status": "development_only_seed0_diagnostic", "model": MODEL, "reference": "native_target",
                    "branch": args.branch, "condition": condition, "seed": args.seed, "split": "dev",
                    "manifest_sha256": MANIFEST_SHA256, "sample_id": row["sample_id"],
                    "parent_sample_id": row["parent_sample_id"], "scene_id": row["scene_id"], "window_id": row["window_id"],
                    "horizon": row["horizon"], "target_raw_frame_id": row["target_raw_frame_id"],
                    "target_observation_index": row["target_observation_index"],
                    "target_temporal_token": row["target_observation_index"] // 2,
                    "raw_frame_delta": row["raw_frame_delta"], "translation_m": row["translation_m"],
                    "rotation_deg": row["rotation_deg"], "directed_depth_overlap": row["directed_depth_overlap"],
                    "target_layout_scale_m": float(scale), "source_run": args.output_dir.name,
                    **metrics, **diagnostics,
                })

    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.save({"frozen_native_d1_checkpoint": str(args.native_d1_checkpoint.resolve()),
                "frozen_native_d1_checkpoint_sha256": sha256(args.native_d1_checkpoint)}, args.output_dir / "checkpoint.pt")
    with (args.output_dir / "per_sample_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(records)
    receipt = {
        "schema_version": 1, "experiment": "D3", "phase": "AB", "status": "complete_development_evaluation",
        "diagnostic_label": "development-only / seed0 / diagnostic", "source_run": args.output_dir.name,
        "model": MODEL, "model_revision": REVISIONS[MODEL], "branch": args.branch, "seed": args.seed,
        "manifest_sha256": MANIFEST_SHA256, "native_d1_checkpoint": str(args.native_d1_checkpoint.resolve()),
        "native_d1_checkpoint_sha256": sha256(args.native_d1_checkpoint), "depth_only": True,
        "conditions": sorted({record["condition"] for record in records}), "parameter_count": parameter_count(readout),
        "feature_audit_receipt": str(args.feature_audit_receipt.resolve()) if args.feature_audit_receipt else None,
        "feature_audit_receipt_sha256": sha256(args.feature_audit_receipt) if args.feature_audit_receipt else None,
        "n_dev": len(dev_rows), "n_dev_scenes": len({row["scene_id"] for row in dev_rows}), "test_scenes": 0,
    } | code_provenance([Path(__file__), REPO_ROOT / "scripts/d_series/run_d1.py",
                         REPO_ROOT / "vidfm3d/models/components/probe_layout_readout.py",
                         REPO_ROOT / "vidfm3d/utils/layout_metrics.py"])
    (args.output_dir / "run_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": receipt["status"], "branch": args.branch, "records": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
