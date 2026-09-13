#!/usr/bin/env python3
"""Extract frozen V-JEPA2 D1/D2/D3 features with an end-to-end canary."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoVideoProcessor

from features.vjepa2.native_predictor import (
    _masked_encoder,
    make_prefix_safe_clip,
    native_target_reference,
    predict_future_tubelets,
    tubelet_token_indices,
)
from scripts.d_series.feature_store import MANIFEST_SHA256, code_provenance, stable_seed, write_feature


MODEL = "vjepa2_vitl256"
MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
REVISION = "b3c1679b7c34d3255ef3547f27c7b226aefab26f"


def load_rgb(scene_root: Path, row, names):
    return [Image.open(scene_root / row["scene_dir_rel"] / "images" / name).convert("RGB") for name in names]


def process(processor, frames, device):
    array = np.stack([np.asarray(frame) for frame in frames])
    return processor(array, return_tensors="pt")["pixel_values_videos"].to(device)


def pad64(frames):
    if not frames or len(frames) > 64:
        raise ValueError("teacher clip must contain 1..64 frames")
    return frames + [frames[-1]] * (64 - len(frames))


def encode_isolated(model, processor, frame, device):
    pixels = process(processor, [frame] * 64, device)
    encoded = model.encoder(pixels).last_hidden_state
    spatial = encoded.reshape(1, 32, 16, 16, 1024)[:, 16]
    return spatial[0]


def order_control(frames, sample_id):
    # Shuffle complete early two-frame tubelets and keep observations 6,7 fixed.
    pairs = [frames[0:2], frames[2:4], frames[4:6]]
    rng = np.random.default_rng(stable_seed(sample_id, "early_tubelet_order"))
    perm = rng.permutation(3)
    return [item for index in perm for item in pairs[index]] + frames[6:]


def assert_full_chain_future_invariance(model, processor, observed, device, atol, rtol):
    prefix = observed[:8]
    height, width = np.asarray(observed[0]).shape[:2]
    rng = np.random.default_rng(0)
    variants = [
        observed,
        prefix + [Image.fromarray(np.zeros((height, width, 3), np.uint8))] * (len(observed) - 8),
        prefix + [Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8)) for _ in observed[8:]],
    ]
    outputs = [predict_future_tubelets(model, process(processor, pad64(v), device)) for v in variants]
    maxima = {}
    for obs in (9, 11):
        maxima[str(obs)] = max(float((candidate[obs] - outputs[0][obs]).abs().max()) for candidate in outputs[1:])
        if not all(torch.allclose(candidate[obs], outputs[0][obs], atol=atol, rtol=rtol) for candidate in outputs[1:]):
            raise AssertionError(f"future-invariance failed at target observation {obs}: {maxima[str(obs)]}")
    return maxima


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("experiments/d_series_scannetpp_v1/data/sample_manifest.jsonl"))
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, default=Path("artifacts/tstep_canonical_v2/manifests/inscene_c_h124_prefix8.jsonl"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--window-start", type=int, default=0)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--canary-only", action="store_true")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.open()]
    if MANIFEST_SHA256 != __import__("hashlib").sha256(args.manifest.read_bytes()).hexdigest():
        raise ValueError("manifest hash mismatch")
    windows = defaultdict(list)
    for row in rows:
        windows[row["parent_sample_id"]].append(row)
    selected = sorted(windows.items())[args.window_start:]
    if args.max_windows is not None:
        selected = selected[: args.max_windows]
    if args.canary_only:
        selected = selected[:1]
    parents = {row["sample_id"]: row for row in (json.loads(line) for line in args.parent_manifest.open())}

    processor = AutoVideoProcessor.from_pretrained(MODEL_ID, revision=REVISION)
    model = AutoModel.from_pretrained(
        MODEL_ID, revision=REVISION, torch_dtype=torch.float16, attn_implementation="sdpa",
    ).to(args.device).eval()
    base_receipt = {
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "layer_zero_based": 23,
        "final_layernorm": True,
        "preprocess": "checkpoint resize-short-edge-292 center-crop-256 imagenet-normalize",
        "predictor_input": "prefix-only patch tokens masked before encoder attention",
    } | code_provenance([
        Path(__file__), REPO_ROOT / "scripts/d_series/feature_store.py",
        REPO_ROOT / "features/vjepa2/native_predictor.py",
    ])
    audit = {"windows_requested": len(selected), "windows_completed": 0, "future_invariance": None}

    with torch.inference_mode():
        for parent_id, horizon_rows in selected:
            horizon_rows.sort(key=lambda row: row["horizon"])
            first = horizon_rows[0]
            # Reconstruct observations 8..11 from the parent manifest's valid-frame indices.
            parent = parents[parent_id]
            observed_names = [parent["valid_frames"][i] for i in parent["streaming_window_indices"][:12]]
            observed = load_rgb(args.scene_root, first, observed_names)
            teacher_pixels = process(processor, pad64(observed), args.device)
            pred = predict_future_tubelets(model, teacher_pixels)
            reference = native_target_reference(model, teacher_pixels)
            shuffled_pixels = process(processor, pad64(order_control(observed, parent_id)), args.device)
            shuffled_pred = predict_future_tubelets(model, shuffled_pixels)
            safe = make_prefix_safe_clip(teacher_pixels, 8, 64)
            context_indices = tubelet_token_indices([0, 1, 2, 3], (16, 16), device=safe.device)
            context, _ = _masked_encoder(model, safe, context_indices)
            context = context.reshape(1, 4, 16, 16, 1024)[0]
            write_feature(args.output_root, MODEL, "prefix", parent_id, {
                "context_spatial": context[-1].half(),
                "context_pooled_sequence": context.mean((1, 2)).half(),
            }, base_receipt | {"raw_frame_ids": first["prefix_raw_frame_ids"]})

            for row in horizon_rows:
                obs = int(row["target_observation_index"])
                target_frame = load_rgb(args.scene_root, row, [row["target_frame_name"]])[0]
                isolated = encode_isolated(model, processor, target_frame, args.device)
                common = base_receipt | {
                    "horizon": row["horizon"], "target_raw_frame_id": row["target_raw_frame_id"],
                    "target_observation_index": obs,
                }
                write_feature(args.output_root, MODEL, "isolated_target", row["sample_id"], {
                    "spatial": isolated.half(), "pooled": isolated.mean((0, 1)).half(),
                }, common | {"reference": "isolated_repeated_target_auxiliary"})
                write_feature(args.output_root, MODEL, "native_target", row["sample_id"], {
                    "spatial": reference[obs][0].half(), "pooled": reference[obs][0].mean((0, 1)).half(),
                }, common | {"reference": "oracle_full_teacher_clip_final_encoder_space"})
                write_feature(args.output_root, MODEL, "native_pred", row["sample_id"], {
                    "spatial": pred[obs][0].half(), "pooled": pred[obs][0].mean((0, 1)).half(),
                }, common | {"future_rgb_access": False})
                write_feature(args.output_root, MODEL, "native_pred_order_shuffle", row["sample_id"], {
                    "spatial": shuffled_pred[obs][0].half(), "pooled": shuffled_pred[obs][0].mean((0, 1)).half(),
                }, common | {"last_tubelet_fixed": True})

            if audit["future_invariance"] is None:
                audit["future_invariance"] = assert_full_chain_future_invariance(
                    model, processor, observed, args.device, 1e-5, 1e-5,
                )
            audit["windows_completed"] += 1
            print(json.dumps({"completed": audit["windows_completed"], "parent_sample_id": parent_id}), flush=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / f"vjepa2_audit_{args.window_start}_{len(selected)}.json").write_text(
        json.dumps(audit, sort_keys=True, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
