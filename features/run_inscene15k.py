#!/usr/bin/env python3
"""
Extract WAN features for InsScene-15K scenes.

Usage:
  CUDA_VISIBLE_DEVICES=2 python -m features.run_inscene15k \
      --data-root /nas/baiqiao/InsScene-15K/data \
      --out-root /nas/baiqiao/InsScene-15K/FEAT \
      --model-id Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
      --t 749 --output-layers 20 \
      --source all

Scenes are collected the same way InsScene15KDataset does, and for each one:
  1. Select 81 evenly-spaced frames from the available frames.
  2. Resize to 480x832 (WAN's expected input).
  3. Run one-step WAN forward to extract layer-20 features.
  4. Save to <out_root>/wan/<source>/<scene_name>/feature_t749_layer20.sft

Resume-safe: scenes whose output .sft already exist are skipped.
"""

import argparse
import logging
import os
import sys
import time
from datetime import timedelta
from glob import glob
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="{asctime}: [{levelname}] {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
)
log = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Scene collection (mirrors InsScene15KDataset)                      #
# ------------------------------------------------------------------ #

def collect_infinigen_scenes(source_path):
    """Collect infinigen scenes, supports flat and nested layouts."""
    scenes = []
    for scene_dir in sorted(glob(os.path.join(source_path, "scene_*"))):
        frames_dir = os.path.join(scene_dir, "frames", "Image", "camera_0")
        if os.path.isdir(frames_dir):
            imgs = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
            if len(imgs) >= 5:
                scenes.append({
                    "source": "infinigen",
                    "scene_dir": scene_dir,
                    "img_dir": frames_dir,
                    "img_files": imgs,
                    "ext": "png",
                })
            continue
        # Nested layout: scene_XXX/<subscene_hash>/frames/...
        for sub_dir in sorted(glob(os.path.join(scene_dir, "*"))):
            if not os.path.isdir(sub_dir):
                continue
            frames_dir = os.path.join(sub_dir, "frames", "Image", "camera_0")
            if os.path.isdir(frames_dir):
                imgs = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
                if len(imgs) >= 5:
                    scenes.append({
                        "source": "infinigen",
                        "scene_dir": sub_dir,
                        "img_dir": frames_dir,
                        "img_files": imgs,
                        "ext": "png",
                    })
    return scenes


def collect_scannetpp_scenes(source_path):
    """Collect scannetpp scenes, handles nested extraction layout."""
    # Handle nested directory from zip extraction
    nested = os.path.join(source_path, "processed_scannetpp_v2")
    if os.path.isdir(nested):
        source_path = nested

    scenes = []
    for scene_id in sorted(os.listdir(source_path)):
        scene_dir = os.path.join(source_path, scene_id)
        if not os.path.isdir(scene_dir):
            continue
        img_dir = os.path.join(scene_dir, "images")
        if not os.path.isdir(img_dir):
            continue
        imgs = sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))
        if len(imgs) >= 5:
            scenes.append({
                "source": "scannetpp",
                "scene_dir": scene_dir,
                "img_dir": img_dir,
                "img_files": imgs,
                "ext": "jpg",
            })
    return scenes


def select_frames(img_files, n=81):
    """Evenly sample n frames from img_files list."""
    total = len(img_files)
    if total <= n:
        return list(range(total))
    indices = np.linspace(0, total - 1, n).round().astype(int).tolist()
    return indices


def load_and_resize_frames(img_dir, img_files, indices, size=(480, 832)):
    """Load selected frames and resize to (height, width)."""
    h, w = size
    frames = []
    for idx in indices:
        path = os.path.join(img_dir, img_files[idx])
        img = Image.open(path).convert("RGB").resize((w, h), Image.LANCZOS)
        frames.append(img)
    return frames


def scene_name(scene_info):
    """Get a unique name for the scene (used as output dir name)."""
    scene_dir = scene_info["scene_dir"]
    source = scene_info["source"]
    if source == "infinigen":
        # e.g. scene_000/16255241 -> scene_000__16255241
        parts = Path(scene_dir).parts
        # Find scene_XXX part
        for i, p in enumerate(parts):
            if p.startswith("scene_"):
                rest = "__".join(parts[i:])
                return rest
        return Path(scene_dir).name
    else:
        return Path(scene_dir).name


# ------------------------------------------------------------------ #
# Main                                                               #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="WAN feature extraction for InsScene-15K",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", required=True, help="InsScene-15K data root")
    parser.add_argument("--out-root", required=True, help="Output root for features")
    parser.add_argument("--source", default="all",
                        choices=["all", "infinigen", "scannetpp"],
                        help="Which data source to process")
    parser.add_argument("--vfm", default="wan",
                        choices=["wan", "cogvideox", "vjepa", "vjepa2"],
                        help="Which VFM to extract features from")
    parser.add_argument("--model-id", default=None,
                        help="Model ID (default per VFM)")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--t", type=int, default=749)
    parser.add_argument("--output-layers", nargs="+", type=int, default=[20])
    parser.add_argument("--ensemble", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=None,
                        help="Number of frames to sample (default per VFM)")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--checkpoint", default=None,
                        help="Local checkpoint path (for V-JEPA)")
    parser.add_argument("--partition", default="spaced",
                        choices=["spaced", "chunked"],
                        help="V-JEPA partition mode")
    # ---------------- Spatial Diagnostic Suite extraction modes ----------------
    parser.add_argument(
        "--mode", default="normal",
        choices=["normal", "shuffled", "target_isolated"],
        help=(
            "normal: standard clip forward (default).\n"
            "shuffled: shuffle frame order before VFM forward (for A3 abnormal probe). "
            "Output features are re-ordered to match the original frame order so that "
            "shuffled[i] = feature of frame i extracted under scrambled temporal context.\n"
            "target_isolated: extract clip-isolated features for M target frames "
            "(replicate each target frame to fill the clip; for C1 action-dynamics probe)."
        ),
    )
    parser.add_argument("--shuffle-seed", type=int, default=42,
                        help="Seed used to permute frame order in --mode shuffled.")
    parser.add_argument("--num-targets", type=int, default=8,
                        help="Number of target frames per scene in --mode target_isolated.")
    args = parser.parse_args()

    # Per-VFM defaults
    VFM_DEFAULTS = {
        "wan":       {"model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "num_frames": 81, "size": (480, 832)},
        "cogvideox": {"model_id": "THUDM/CogVideoX-5b-I2V",          "num_frames": 97, "size": (480, 720)},
        "vjepa":     {"model_id": None,                                "num_frames": 76, "size": (480, 832)},
        "vjepa2":    {"model_id": "facebook/vjepa2-vitl-fpc64-256",   "num_frames": 64, "size": (256, 256)},
    }
    defaults = VFM_DEFAULTS[args.vfm]
    if args.model_id is None:
        args.model_id = defaults["model_id"]
    if args.num_frames is None:
        args.num_frames = defaults["num_frames"]
    args.resize = defaults["size"]

    # Collect scenes
    scenes = []
    if args.source in ("all", "infinigen"):
        inf_path = os.path.join(args.data_root, "processed_infinigen")
        if os.path.isdir(inf_path):
            inf_scenes = collect_infinigen_scenes(inf_path)
            log.info(f"Found {len(inf_scenes)} infinigen scenes")
            scenes.extend(inf_scenes)

    if args.source in ("all", "scannetpp"):
        spp_path = os.path.join(args.data_root, "processed_scannetpp_v2")
        if os.path.isdir(spp_path):
            spp_scenes = collect_scannetpp_scenes(spp_path)
            log.info(f"Found {len(spp_scenes)} scannetpp scenes")
            scenes.extend(spp_scenes)

    log.info(f"Total scenes to process: {len(scenes)}")

    # Slice
    scenes = scenes[args.start:args.end]
    log.info(f"Processing scenes [{args.start}:{args.end}] = {len(scenes)}")

    # Build output filename pattern per VFM
    if args.vfm == "wan":
        fname_prefix = "feature"
        if args.model_id and args.model_id.endswith("14B-Diffusers"):
            fname_prefix = "feature_t2v_14b"
        def out_fname(layer):
            return f"{fname_prefix}_t{args.t}_layer{layer}.sft"
    elif args.vfm == "cogvideox":
        fname_prefix = "feature"
        if args.model_id and not args.model_id.endswith("I2V"):
            model_size = args.model_id.split("-")[-1]
            fname_prefix = f"feature_t2v_{model_size}"
        def out_fname(layer):
            return f"{fname_prefix}_t{args.t}_layer{layer}.sft"
    elif args.vfm == "vjepa":
        def out_fname(layer):
            return "feature.sft" if args.partition == "spaced" else "feature_chunked.sft"
    elif args.vfm == "vjepa2":
        def out_fname(layer):
            return f"feature_layer{layer}.sft"

    # The mode (normal / shuffled / target_isolated) is differentiated by
    # `--out-root` (e.g. FEAT vs FEAT_SHUFFLED vs FEAT_TARGET) — the per-VFM
    # subdirectory keeps a clean, mode-agnostic name so InsScene15KDataset can
    # locate it via `vfm_name` regardless of mode.
    vfm_dir_name = args.vfm

    # Check how many are already done
    done = 0
    for s in scenes:
        name = scene_name(s)
        out_dir = os.path.join(args.out_root, vfm_dir_name, s["source"], name)
        if args.vfm == "vjepa":
            all_exist = os.path.exists(os.path.join(out_dir, out_fname(0)))
        else:
            all_exist = all(
                os.path.exists(os.path.join(out_dir, out_fname(l)))
                for l in args.output_layers
            )
        if all_exist:
            done += 1
    log.info(f"Already done: {done}/{len(scenes)}, remaining: {len(scenes) - done}")

    if done == len(scenes):
        log.info("All scenes already processed!")
        return

    # ------------------------------------------------------------------ #
    # Load model (once)                                                  #
    # ------------------------------------------------------------------ #
    from safetensors.torch import save_file

    if args.vfm == "wan":
        from features.wan.wan_feature import get_wan_featurizer
        from features.wan.extract_features import reshape_to_t_h_w_c
        log.info(f"Loading WAN model: {args.model_id}")
        model = get_wan_featurizer(model_id=args.model_id, null_prompt=args.prompt)
    elif args.vfm == "cogvideox":
        from features.cogvideox.cogvideox_feature import get_cogvideox_featurizer
        from features.cogvideox.cogvideox_feature_i2v import get_cogvideox_featurizer_i2v
        from features.cogvideox.extract_features import forward_cogvideox
        log.info(f"Loading CogVideoX model: {args.model_id}")
        if args.model_id.endswith("I2V"):
            model = get_cogvideox_featurizer_i2v(model_id=args.model_id)
        else:
            model = get_cogvideox_featurizer(model_id=args.model_id)
    elif args.vfm == "vjepa":
        from features.vjepa.extract_features import VJEPAFeaturizer_Spaced, VJEPAFeaturizer_Chunked
        log.info("Loading V-JEPA model...")
        if args.partition == "spaced":
            model = VJEPAFeaturizer_Spaced(args.checkpoint)
        else:
            model = VJEPAFeaturizer_Chunked(args.checkpoint)
    elif args.vfm == "vjepa2":
        from features.vjepa2.vjepa2_feature import get_vjepa2_featurizer
        log.info(f"Loading V-JEPA 2 model: {args.model_id}")
        model = get_vjepa2_featurizer(model_id=args.model_id)
        # Override output_layers default to last encoder layer if user kept the default [20]
        if args.output_layers == [20]:
            args.output_layers = [model.num_hidden_layers - 1]
            log.info(f"V-JEPA 2: auto-set output_layers to {args.output_layers}")

    log.info("Model loaded.")

    # ------------------------------------------------------------------ #
    # Process scenes                                                     #
    # ------------------------------------------------------------------ #
    total_time = 0
    processed = 0
    failed = 0

    for i, s in enumerate(scenes):
        name = scene_name(s)
        out_dir = os.path.join(args.out_root, vfm_dir_name, s["source"], name)

        # Check resume
        if args.vfm == "vjepa":
            if os.path.exists(os.path.join(out_dir, out_fname(0))):
                continue
        else:
            missing_layers = [
                l for l in args.output_layers
                if not os.path.exists(os.path.join(out_dir, out_fname(l)))
            ]
            if not missing_layers:
                continue

        t0 = time.time()
        try:
            # Select and load frames
            indices = select_frames(s["img_files"], n=args.num_frames)

            # Pad if fewer frames available
            while len(indices) < args.num_frames:
                indices.append(indices[-1])

            frames = load_and_resize_frames(
                s["img_dir"], s["img_files"], indices, size=args.resize
            )
            os.makedirs(out_dir, exist_ok=True)

            # ---------------- Mode dispatch ----------------
            if args.mode == "shuffled":
                # Shuffle in *latent-time chunks* so the output's compressed
                # temporal axis can be cleanly inverse-permuted.
                #   Wan / CogVideoX (VAE stride 4, first frame standalone):
                #       chunks = [[0], [1..4], [5..8], ...]
                #   V-JEPA2 (tubelet size 2): chunks = [[0,1], [2,3], ...]
                #   else: per-frame.
                if args.vfm in ("wan", "cogvideox"):
                    chunks = [[0]]
                    j = 1
                    while j < len(frames):
                        chunks.append(list(range(j, min(j + 4, len(frames)))))
                        j += 4
                elif args.vfm == "vjepa2":
                    chunks = [
                        list(range(k, min(k + 2, len(frames))))
                        for k in range(0, len(frames), 2)
                    ]
                else:
                    chunks = [[k] for k in range(len(frames))]

                rng = np.random.default_rng(seed=args.shuffle_seed + i)
                chunk_perm = rng.permutation(len(chunks))
                inv_chunk_perm = np.argsort(chunk_perm).tolist()
                flat_order = [fi for ci in chunk_perm for fi in chunks[ci]]
                frames_input = [frames[fi] for fi in flat_order]
            elif args.mode == "target_isolated":
                # M target frame indices spread across the original sequence
                M = max(2, min(args.num_targets, len(s["img_files"])))
                target_local_idx = np.linspace(
                    0, len(s["img_files"]) - 1, M
                ).round().astype(int).tolist()
                target_global = target_local_idx  # already global within the scene
                frames_input = None  # filled per-target inside the per-VFM block
            else:
                frames_input = frames

            if args.mode == "target_isolated":
                # Run M forwards per layer, collect into (M, H_f, W_f, C) per layer.
                M = len(target_global)
                target_imgs = [
                    Image.open(os.path.join(s["img_dir"], s["img_files"][gi]))
                    .convert("RGB").resize(
                        (args.resize[1], args.resize[0]), Image.LANCZOS
                    )
                    for gi in target_global
                ]
                per_layer_collect = {l: [] for l in (missing_layers if args.vfm != "vjepa" else [0])}

                for tgt_img in target_imgs:
                    rep_frames = [tgt_img] * args.num_frames
                    if args.vfm == "wan":
                        with torch.no_grad():
                            feats = model.forward(
                                video=rep_frames, prompt=args.prompt, t=args.t,
                                output_layer_indices=missing_layers,
                                ensemble_size=args.ensemble,
                            )
                        for layer_id, raw_feat in feats.items():
                            reshaped = reshape_to_t_h_w_c(raw_feat)  # (T, H_f, W_f, C)
                            per_layer_collect[layer_id].append(reshaped[reshaped.shape[0] // 2])
                    elif args.vfm == "cogvideox":
                        feats = forward_cogvideox(model, rep_frames, t=args.t, layer_ids=missing_layers)
                        for layer_id, feat in feats.items():
                            # feat shape (2, T_clip, H_f, W_f, C) -> (2*T_clip, ...) -> middle
                            f = feat.reshape(-1, *feat.shape[2:])
                            per_layer_collect[layer_id].append(f[f.shape[0] // 2])
                    elif args.vfm == "vjepa2":
                        feats = model(rep_frames, output_layers=missing_layers)
                        for layer_id, feat in feats.items():
                            per_layer_collect[layer_id].append(feat[feat.shape[0] // 2])
                    else:
                        raise NotImplementedError(
                            f"target_isolated mode not implemented for vfm={args.vfm}"
                        )

                for layer_id, lst in per_layer_collect.items():
                    stacked = torch.stack(lst, dim=0).half().contiguous()
                    out_path = os.path.join(out_dir, out_fname(layer_id))
                    save_file({"feat": stacked}, out_path)
                # Save target index metadata (one per scene, layer-agnostic)
                np.save(
                    os.path.join(out_dir, "target_indices.npy"),
                    np.array(target_global, dtype=np.int64),
                )

            elif args.vfm == "wan":
                with torch.no_grad():
                    feats = model.forward(
                        video=frames_input, prompt=args.prompt, t=args.t,
                        output_layer_indices=missing_layers,
                        ensemble_size=args.ensemble,
                    )
                for layer_id, raw_feat in feats.items():
                    reshaped = reshape_to_t_h_w_c(raw_feat)
                    if args.mode == "shuffled":
                        reshaped = reshaped[inv_chunk_perm]
                    out_path = os.path.join(out_dir, out_fname(layer_id))
                    save_file({"feat": reshaped.half()}, out_path)

            elif args.vfm == "cogvideox":
                feats = forward_cogvideox(model, frames_input, t=args.t, layer_ids=missing_layers)
                for layer_id, feat in feats.items():
                    if args.mode == "shuffled":
                        # feat shape (2, T_latent, H, W, C); leading 2 is CFG, not temporal.
                        feat = feat[:, inv_chunk_perm]
                    out_path = os.path.join(out_dir, out_fname(layer_id))
                    save_file({"feat": feat.half()}, out_path)

            elif args.vfm == "vjepa":
                feats = model(frames_input)  # (N_clips, 8, H, W, C)
                out_path = os.path.join(out_dir, out_fname(0))
                save_file({"feat": feats.half().contiguous()}, out_path)

            elif args.vfm == "vjepa2":
                feats = model(frames_input, output_layers=missing_layers)
                for layer_id, feat in feats.items():
                    if args.mode == "shuffled":
                        feat = feat[inv_chunk_perm]
                    out_path = os.path.join(out_dir, out_fname(layer_id))
                    save_file({"feat": feat.half().contiguous()}, out_path)

            elapsed = time.time() - t0
            total_time += elapsed
            processed += 1
            remaining = len(scenes) - done - processed - failed
            avg = total_time / processed
            eta = str(timedelta(seconds=int(avg * remaining)))
            log.info(
                f"[{done + processed + failed}/{len(scenes)}] "
                f"{s['source']}/{name}: {elapsed:.1f}s "
                f"({len(s['img_files'])} frames, sampled {len(indices)}) "
                f"ETA: {eta}"
            )

        except Exception as e:
            elapsed = time.time() - t0
            failed += 1
            log.error(f"[{done + processed + failed}/{len(scenes)}] FAILED {s['source']}/{name}: {e}")

    log.info(
        f"Done. Processed: {processed}, Failed: {failed}, Skipped: {done}, "
        f"Total time: {str(timedelta(seconds=int(total_time)))}"
    )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
