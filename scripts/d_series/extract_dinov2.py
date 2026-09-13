#!/usr/bin/env python3
"""Extract DINOv2 target and prefix features for the frozen D-series rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from PIL import Image
from transformers import AutoImageProcessor, Dinov2Model

from scripts.d_series.feature_store import MANIFEST_SHA256, code_provenance, write_feature


MODEL = "dinov2_vitl14"
MODEL_ID = "facebook/dinov2-large"
REVISION = "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c"


def load_images(scene_root: Path, row, names):
    images = []
    for name in names:
        with Image.open(scene_root / row["scene_dir_rel"] / "images" / name) as image:
            images.append(image.convert("RGB").resize((728, 420), Image.Resampling.BICUBIC))
    return images


def encode(model, processor, images, device):
    inputs = processor(
        images=images, return_tensors="pt", do_center_crop=False, do_resize=False,
    ).to(device)
    tokens = model(**inputs).last_hidden_state[:, 1:]
    if tokens.shape[1:] != (30 * 52, 1024):
        raise ValueError(f"unexpected DINO shape {tuple(tokens.shape)}")
    return tokens.reshape(len(images), 30, 52, 1024)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("experiments/d_series_scannetpp_v1/data/sample_manifest.jsonl"))
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--window-start", type=int, default=0)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--canary-only", action="store_true")
    args = parser.parse_args()
    if hashlib.sha256(args.manifest.read_bytes()).hexdigest() != MANIFEST_SHA256:
        raise ValueError("manifest hash mismatch")
    windows = defaultdict(list)
    for line in args.manifest.open():
        row = json.loads(line)
        windows[row["parent_sample_id"]].append(row)
    selected = sorted(windows.items())[args.window_start:]
    if args.max_windows is not None:
        selected = selected[:args.max_windows]
    if args.canary_only:
        selected = selected[:1]

    processor = AutoImageProcessor.from_pretrained(MODEL_ID, revision=REVISION)
    model = Dinov2Model.from_pretrained(MODEL_ID, revision=REVISION).to(args.device).eval()
    receipt = {
        "model_id": MODEL_ID, "model_revision": REVISION, "layer_zero_based": 23,
        "final_layernorm": True, "register_tokens": False,
        "preprocess": "RGB bicubic resize 420x728 then checkpoint normalization; no crop",
    } | code_provenance([Path(__file__), REPO_ROOT / "scripts/d_series/feature_store.py"])
    with torch.inference_mode():
        for number, (parent_id, rows) in enumerate(selected, 1):
            rows.sort(key=lambda row: row["horizon"])
            first = rows[0]
            prefix = encode(model, processor, load_images(args.scene_root, first, first["prefix_frame_names"]), args.device)
            write_feature(args.output_root, MODEL, "prefix", parent_id, {
                "context_spatial": prefix[-1].half(),
                "context_pooled_sequence": prefix.mean((1, 2)).half(),
            }, receipt | {"raw_frame_ids": first["prefix_raw_frame_ids"], "future_rgb_access": False})
            for row in rows:
                target = encode(model, processor, load_images(args.scene_root, row, [row["target_frame_name"]]), args.device)[0]
                write_feature(args.output_root, MODEL, "isolated_target", row["sample_id"], {
                    "spatial": target.half(), "pooled": target.mean((0, 1)).half(),
                }, receipt | {"horizon": row["horizon"], "target_raw_frame_id": row["target_raw_frame_id"]})
            print(json.dumps({"completed": number, "parent_sample_id": parent_id}), flush=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / f"dinov2_audit_{args.window_start}_{len(selected)}.json").write_text(
        json.dumps({"windows_requested": len(selected), "windows_completed": len(selected)}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
