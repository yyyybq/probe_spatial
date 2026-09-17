#!/usr/bin/env python3
"""Extract frozen Wan layer-20/index-749 features for D1 and external D2."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from PIL import Image

from features.wan.wan_feature import get_wan_featurizer
from scripts.d_series.feature_store import (
    MANIFEST_SHA256,
    code_provenance,
    load_verified,
    stable_seed,
    write_feature,
)


MODEL = "wan21_t2v_1p3b"
MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"
MODEL_LOAD_PATH = Path(
    "/mnt/umm/users/yinbaiqiao/.cache/huggingface/hub/"
    "models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/" + REVISION
)


def load_images(scene_root: Path, row, names):
    frames = []
    for name in names:
        with Image.open(scene_root / row["scene_dir_rel"] / "images" / name) as image:
            frames.append(image.convert("RGB").resize((832, 480), Image.Resampling.LANCZOS))
    return frames


def set_sample_seed(key: str):
    seed = stable_seed(42, key)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def encode(model, frames, key):
    if not frames or len(frames) > 81:
        raise ValueError("Wan input must contain 1..81 frames")
    padded = frames + [frames[-1]] * (81 - len(frames))
    set_sample_seed(key)
    raw = model.forward(
        video=padded, prompt="", t=749, output_layer_indices=[20], ensemble_size=1,
    )[20]
    if raw.shape[0] != 1 or raw.shape[1] != 21 * 30 * 52:
        raise ValueError(f"unexpected Wan feature shape: {tuple(raw.shape)}")
    return raw[0].reshape(21, 30, 52, raw.shape[-1])


def complete_feature(root: Path, space: str, sample_id: str, expected_tensors, expected_receipt):
    try:
        tensors, receipt = load_verified(root, MODEL, space, sample_id, REVISION)
    except (FileNotFoundError, ValueError):
        return False
    actual = {key: list(value.shape) for key, value in tensors.items()}
    if actual != expected_tensors:
        raise RuntimeError(f"existing {space}/{sample_id} tensor contract mismatch: {actual}")
    failures = {
        key: (receipt.get(key), value)
        for key, value in expected_receipt.items()
        if receipt.get(key) != value
    }
    if failures:
        raise RuntimeError(f"existing {space}/{sample_id} receipt mismatch: {failures}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("experiments/d_series_scannetpp_v1/data/sample_manifest.jsonl"))
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--window-start", type=int, default=0)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--selection-file", type=Path)
    parser.add_argument("--shard-id")
    parser.add_argument("--canary-only", action="store_true")
    args = parser.parse_args()
    if hashlib.sha256(args.manifest.read_bytes()).hexdigest() != MANIFEST_SHA256:
        raise ValueError("manifest hash mismatch")
    windows = defaultdict(list)
    for line in args.manifest.open():
        row = json.loads(line)
        windows[row["parent_sample_id"]].append(row)
    selected = sorted(windows.items())
    selection_ids = None
    if args.selection_file is not None:
        values = [line.strip() for line in args.selection_file.read_text().splitlines() if line.strip()]
        if not values or len(values) != len(set(values)):
            raise ValueError("selection file must contain unique non-empty parent sample IDs")
        selection_ids = set(values)
        unknown = selection_ids - set(windows)
        if unknown:
            raise ValueError(f"selection contains unknown parents: {sorted(unknown)[:5]}")
        selected = [item for item in selected if item[0] in selection_ids]
        if len(selected) != len(selection_ids):
            raise RuntimeError("selection coverage mismatch")
    selected = selected[args.window_start:]
    if args.max_windows is not None:
        selected = selected[:args.max_windows]
    if args.canary_only:
        selected = selected[:1]

    if not MODEL_LOAD_PATH.is_dir():
        raise FileNotFoundError(f"pinned Wan snapshot missing: {MODEL_LOAD_PATH}")
    model = get_wan_featurizer(str(MODEL_LOAD_PATH), null_prompt="")
    model.pipe.scheduler.set_timesteps(1000, device="cuda")
    actual_timestep = int(model.pipe.scheduler.timesteps[749].item())
    receipt = {
        "model_id": MODEL_ID, "model_revision": REVISION, "layer_zero_based": 20,
        "final_layernorm": False, "input_frames": 81, "resize": [480, 832],
        "prompt": "", "target_or_future_text_used": False,
        "timestep_argument": 749, "timestep_semantics": "scheduler_timesteps_index",
        "resolved_scheduler_timestep": actual_timestep, "scheduler": "UniPCMultistepScheduler",
        "flow_shift": 3.0, "noise_seed_scheme": "sha256(42|space_and_sample_id)_uint32_global_rng_reset_before_forward",
    } | code_provenance([
        Path(__file__), REPO_ROOT / "scripts/d_series/feature_store.py",
        REPO_ROOT / "features/wan/wan_feature.py", REPO_ROOT / "features/noise_seed.py",
    ])
    counters = Counter()
    with torch.inference_mode():
        for number, (parent_id, rows) in enumerate(selected, 1):
            rows.sort(key=lambda row: row["horizon"])
            first = rows[0]
            prefix_expected = receipt | {
                "raw_frame_ids": first["prefix_raw_frame_ids"], "future_rgb_access": False,
            }
            if complete_feature(
                args.output_root, "prefix", parent_id,
                {"context_pooled_sequence": [21, 1536], "context_spatial": [30, 52, 1536]},
                prefix_expected,
            ):
                counters["prefix_skipped_complete"] += 1
            else:
                prefix_frames = load_images(args.scene_root, first, first["prefix_frame_names"])
                prefix = encode(model, prefix_frames, f"prefix:{parent_id}")
                context_only = encode(model, [prefix_frames[-1]], f"last_context:{parent_id}")[10]
                write_feature(args.output_root, MODEL, "prefix", parent_id, {
                    "context_spatial": context_only.half(),
                    "context_pooled_sequence": prefix.mean((1, 2)).half(),
                }, prefix_expected)
                counters["prefix_extracted"] += 1
            for row in rows:
                target_expected = receipt | {
                    "horizon": row["horizon"], "target_raw_frame_id": row["target_raw_frame_id"],
                }
                if complete_feature(
                    args.output_root, "isolated_target", row["sample_id"],
                    {"pooled": [1536], "spatial": [30, 52, 1536]}, target_expected,
                ):
                    counters[f"h{row['horizon']}_skipped_complete"] += 1
                else:
                    target_frame = load_images(args.scene_root, row, [row["target_frame_name"]])
                    target = encode(model, target_frame, f"target:{row['sample_id']}")[10]
                    write_feature(args.output_root, MODEL, "isolated_target", row["sample_id"], {
                        "spatial": target.half(), "pooled": target.mean((0, 1)).half(),
                    }, target_expected)
                    counters[f"h{row['horizon']}_extracted"] += 1
            print(json.dumps({"completed": number, "parent_sample_id": parent_id}), flush=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    audit_key = args.shard_id or f"{args.window_start}_{len(selected)}"
    (args.output_root / f"wan_shard_{audit_key}.json").write_text(
        json.dumps({
            "windows_requested": len(selected), "windows_completed": len(selected),
            "resolved_scheduler_timestep": actual_timestep,
            "selection_file": str(args.selection_file.resolve()) if args.selection_file else None,
            "selection_sha256": hashlib.sha256(args.selection_file.read_bytes()).hexdigest() if args.selection_file else None,
            "counts": dict(counters),
        }, indent=2) + "\n"
    )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
