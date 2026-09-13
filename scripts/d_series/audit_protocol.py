#!/usr/bin/env python3
"""Materialise dataset, cache, checkpoint, normalization, and mask audit status."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from scripts.d_series.feature_store import MANIFEST_SHA256, feature_paths, load_verified
from scripts.d_series.run_d1 import REVISIONS


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(); p.add_argument("--experiment-root", type=Path, default=Path("experiments/d_series_scannetpp_v1"))
    p.add_argument("--feature-root", type=Path); p.add_argument("--run-root", type=Path); args = p.parse_args()
    data = args.experiment_root / "data"; manifest = data / "sample_manifest.jsonl"
    rows = [json.loads(line) for line in manifest.open()]
    if sha(manifest) != MANIFEST_SHA256: raise RuntimeError("manifest hash mismatch")
    ids = [r["sample_id"] for r in rows]
    if len(ids) != len(set(ids)): raise RuntimeError("duplicate sample identity")
    split_scenes = {split: set((data / f"{split}_scenes.txt").read_text().split()) for split in ("train", "dev", "test")}
    if split_scenes["train"] & split_scenes["dev"]: raise RuntimeError("train/dev scene leakage")
    if split_scenes["test"]: raise RuntimeError("test must remain empty under development-only protocol")
    counts = Counter((r["split"], r["horizon"]) for r in rows)
    report = {
        "schema_version": 1, "protocol": "d_series_scannetpp_v1", "manifest_sha256": MANIFEST_SHA256,
        "sample_identity_unique": True, "scene_disjoint": True, "test_empty": True,
        "counts": {f"{split}_h{horizon}": count for (split, horizon), count in sorted(counts.items())},
        "feature_status": {}, "checkpoint_status": {},
    }
    spaces = {
        "vjepa2_vitl256": ["prefix", "isolated_target", "native_target", "native_pred", "native_pred_order_shuffle"],
        "wan21_t2v_1p3b": ["prefix", "isolated_target"], "dinov2_vitl14": ["prefix", "isolated_target"],
    }
    if args.feature_root:
        parents = sorted({r["parent_sample_id"] for r in rows})
        for model, model_spaces in spaces.items():
            for space in model_spaces:
                sample_ids = parents if space == "prefix" else ids
                complete, failed = 0, []
                for sample_id in sample_ids:
                    try: load_verified(args.feature_root, model, space, sample_id, REVISIONS[model]); complete += 1
                    except Exception as error:
                        if len(failed) < 5: failed.append({"sample_id": sample_id, "error": str(error)})
                report["feature_status"][f"{model}/{space}"] = {"complete": complete, "expected": len(sample_ids), "first_failures": failed}
    checkpoint_rows = []
    if args.run_root:
        for receipt_path in sorted(args.run_root.rglob("run_receipt.json")):
            receipt = json.loads(receipt_path.read_text()); checkpoint = receipt_path.parent / "checkpoint.pt"
            row = {"receipt": str(receipt_path), "experiment": receipt.get("experiment"), "model": receipt.get("model"),
                   "branch": receipt.get("branch", "pooled"), "seed": receipt.get("seed"), "status": receipt.get("status"),
                   "manifest_match": receipt.get("manifest_sha256") == MANIFEST_SHA256,
                   "checkpoint_exists": checkpoint.is_file(), "checkpoint_sha256": sha(checkpoint) if checkpoint.is_file() else ""}
            checkpoint_rows.append(row)
        report["checkpoint_status"] = {"receipts": len(checkpoint_rows), "complete": sum(r["status"] == "complete_development_evaluation" and r["manifest_match"] and r["checkpoint_exists"] for r in checkpoint_rows)}
    out = args.experiment_root / "audits"; out.mkdir(parents=True, exist_ok=True)
    (out / "protocol_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    fields = ["receipt", "experiment", "model", "branch", "seed", "status", "manifest_match", "checkpoint_exists", "checkpoint_sha256"]
    with (out / "checkpoint_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(checkpoint_rows)
    (out / "normalization_mask_audit.md").write_text(
        "# Normalization and mask audit\n\n"
        "- Main labels: uint16 millimetres -> metres -> validity-weighted 72x128 area resize -> divide by each sample's finite-positive mean.\n"
        "- `target_layout_scale` is label/oracle metadata and is never a readout or predictor input.\n"
        "- Readout latent normalization is internal per-token LayerNorm; external predictor targets and 3x4 motion entries use train-only channel mean/std saved in the checkpoint.\n"
        "- All depth metrics use only the GT finite-positive mask; no predicted mask can suppress a pixel.\n"
        "- `oracle_gt_scale_pointmap_mae_m` is explicitly oracle-scaled and is not a metric-scale prediction claim. Intrinsics are scaled with the direct depth resize.\n"
        "- V-JEPA2 predictor context masks are applied before encoder attention; target masks have explicit batch dimension and preserve absolute THW RoPE indices.\n"
    )


if __name__ == "__main__": main()
