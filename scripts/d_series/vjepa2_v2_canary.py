#!/usr/bin/env python3
"""Real V2 pooled-read/overfit and frozen-readout identity canary."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from vidfm3d.models.components.probe_layout_readout import PooledDepthReadout
from vidfm3d.utils.layout_metrics import depth_metrics, normalize_depth_per_sample, resize_depth_validity_weighted


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def masked_l1(prediction, target, valid):
    return ((prediction[:, 0] - target).abs() * valid).sum() / valid.sum().clamp_min(1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--coverage-receipt", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()

    coverage = json.loads(args.coverage_receipt.read_text())
    if coverage.get("status") != "COMPLETE_VALID" or coverage.get("verified") != 12358:
        raise ValueError("V2 full coverage receipt is not passing")
    with args.mapping.open() as handle:
        candidates = [row for row in csv.DictReader(handle) if row["d_series_role"] == "isolated_target"
                      and row["status"] == "adapted" and row["split"] == "train"]
    if not candidates:
        raise ValueError("no exact V2/D-series train target mapping")
    selected = sorted(candidates, key=lambda row: row["sample_id"])[0]
    rows = {row["sample_id"]: row for row in (json.loads(line) for line in args.manifest.open())}
    row = rows[selected["sample_id"]]
    payload = Path(selected["v2_payload"])
    feature = load_file(str(payload))["feat"].float()
    if list(feature.shape) != [1024] or not torch.isfinite(feature).all():
        raise ValueError("real V2 tensor identity/dimension check failed")
    if sha256(payload) != selected["v2_payload_sha256"]:
        raise ValueError("real V2 tensor SHA mismatch")

    encoded = np.asarray(Image.open(args.scene_root / row["depth_rel"]), dtype=np.float32).copy()
    depth_m = torch.from_numpy(encoded) / 1000.0
    valid = torch.isfinite(depth_m) & (depth_m > 0)
    target, valid = resize_depth_validity_weighted(depth_m, valid, (72, 128))
    target, scale = normalize_depth_per_sample(target[None], valid[None])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_b = feature[None].to(device)
    target, valid = target.to(device), valid[None].to(device)
    readout = PooledDepthReadout(1024).to(device)
    optimizer = torch.optim.AdamW(readout.parameters(), lr=1e-3, weight_decay=0)
    with torch.no_grad():
        initial = float(masked_l1(readout(feature_b), target, valid))
    for _ in range(args.steps):
        loss = masked_l1(readout(feature_b), target, valid)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    readout.eval()
    with torch.no_grad():
        direct = readout(feature_b)
        predictor_override = readout(feature_b.clone())
        final = float(masked_l1(direct, target, valid))
        identity_max_abs = float((direct - predictor_override).abs().max())
        direct_metrics = depth_metrics(direct, target, valid)[0]
        override_metrics = depth_metrics(predictor_override, target, valid)[0]
    metric_identity = max(abs(float(direct_metrics[key]) - float(override_metrics[key])) for key in direct_metrics)
    passed = final < 0.25 * initial and identity_max_abs == 0.0 and metric_identity == 0.0
    result = {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "real_v2_tensor_loaded": True,
        "v2_unit_id": selected["v2_unit_id"],
        "v2_role": selected["v2_role"],
        "v2_payload": str(payload),
        "v2_payload_sha256": selected["v2_payload_sha256"],
        "shape": list(feature.shape),
        "dtype": str(feature.dtype),
        "sample_id": row["sample_id"],
        "scene_id": row["scene_id"],
        "split": row["split"],
        "horizon": row["horizon"],
        "steps": args.steps,
        "initial_mae": initial,
        "final_mae": final,
        "ratio": final / initial,
        "evaluator_identity": {
            "definition": "native-target feature and prediction-override feature enter the same frozen readout",
            "prediction_max_abs_difference": identity_max_abs,
            "metric_max_abs_difference": metric_identity,
        },
        "depth_contract": {
            "source_encoding": "uint16 millimeters",
            "metric_conversion": "divide by 1000",
            "valid_mask": "finite and positive GT depth",
            "resize": "validity-weighted to 72x128",
            "target": "per-sample valid-mean normalized depth",
            "target_scale_label_only_m": float(scale[0]),
            "selection_uses_overlap_geometry": False,
        },
        "coverage_receipt": str(args.coverage_receipt),
        "coverage_receipt_sha256": sha256(args.coverage_receipt),
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
