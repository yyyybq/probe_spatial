#!/usr/bin/env python3
"""
Extract WAN features for InsScene-15K scenes.

Usage:
  CUDA_VISIBLE_DEVICES=2 python -m features.run_inscene15k \
      --data-root ${INSCENE_DATA_ROOT} \
      --out-root ${INSCENE_FEAT_ROOT} \
      --model-id Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
      --t 749 --output-layers 20 \
      --source scannetpp

Scenes are collected the same way InsScene15KDataset does, and for each one:
  1. Select 81 evenly-spaced frames from the available frames.
  2. Resize to 480x832 (WAN's expected input).
  3. Run one-step WAN forward to extract layer-20 features.
  4. Save to <out_root>/wan/<source>/<scene_name>/feature_t749_layer20.sft

Resume-safe: scenes whose output .sft already exist are skipped.
"""

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import subprocess
import sys
import time
from datetime import timedelta
from glob import glob
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    from vidfm3d.utils.feature_layers import parse_layers_arg
    from vidfm3d.utils.temporal_sampling import (
        sort_frame_names,
        temporal_windows_from_poses,
    )
except Exception:
    _FEATURE_LAYERS_PATH = Path(__file__).resolve().parents[1] / "vidfm3d" / "utils" / "feature_layers.py"
    _SPEC = importlib.util.spec_from_file_location("feature_layers", _FEATURE_LAYERS_PATH)
    feature_layers = importlib.util.module_from_spec(_SPEC)
    assert _SPEC.loader is not None
    sys.modules[_SPEC.name] = feature_layers
    _SPEC.loader.exec_module(feature_layers)
    parse_layers_arg = feature_layers.parse_layers_arg
    _TEMPORAL_PATH = Path(__file__).resolve().parents[1] / "vidfm3d" / "utils" / "temporal_sampling.py"
    _TEMPORAL_SPEC = importlib.util.spec_from_file_location("temporal_sampling", _TEMPORAL_PATH)
    temporal_sampling = importlib.util.module_from_spec(_TEMPORAL_SPEC)
    assert _TEMPORAL_SPEC.loader is not None
    sys.modules[_TEMPORAL_SPEC.name] = temporal_sampling
    _TEMPORAL_SPEC.loader.exec_module(temporal_sampling)
    sort_frame_names = temporal_sampling.sort_frame_names
    temporal_windows_from_poses = temporal_sampling.temporal_windows_from_poses

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
            imgs = sort_frame_names(f for f in os.listdir(frames_dir) if f.endswith(".png"))
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
                imgs = sort_frame_names(f for f in os.listdir(frames_dir) if f.endswith(".png"))
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
        mask_dir = os.path.join(scene_dir, "refined_ins_ids")
        meta_path = os.path.join(scene_dir, "scene_iphone_metadata.npz")
        if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir) or not os.path.exists(meta_path):
            continue
        mask_files = set(os.listdir(mask_dir))
        image_files = set(f for f in os.listdir(img_dir) if f.endswith(".jpg"))
        meta = np.load(meta_path)
        meta_images = [str(f) for f in meta["images"].tolist()]
        imgs = []
        pose_indices = []
        for meta_idx, fname in enumerate(meta_images):
            if fname in image_files and f"{fname}.npy" in mask_files:
                imgs.append(fname)
                pose_indices.append(meta_idx)
        if len(imgs) >= 5:
            scenes.append({
                "source": "scannetpp",
                "scene_dir": scene_dir,
                "img_dir": img_dir,
                "img_files": imgs,
                "pose_indices": pose_indices,
                "poses_c2w": meta["trajectories"][pose_indices].astype(np.float32),
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


def parse_int_list(value):
    if value is None:
        return None
    items = str(value).replace(",", " ").split()
    parsed = sorted({int(item) for item in items if item.strip()})
    return parsed or None


def streaming_prefix_records(
    num_frames,
    min_len=1,
    max_len=None,
    stride=1,
    model_max_len=None,
    lengths=None,
):
    """Build online prefix records H_t = [I_0, ..., I_t]."""
    min_len = max(int(min_len), 1)
    stride = max(int(stride), 1)
    if max_len is None:
        max_len = num_frames
    max_len = min(int(max_len), num_frames)
    if model_max_len is not None:
        max_len = min(max_len, int(model_max_len))
    if max_len < min_len:
        return []
    if lengths is not None:
        candidate_lengths = [
            int(length) for length in lengths
            if min_len <= int(length) <= max_len
        ]
    else:
        candidate_lengths = range(min_len, max_len + 1, stride)
    records = []
    for length in candidate_lengths:
        records.append({
            "tail": length - 1,
            "indices": list(range(length)),
            "valid_length": length,
        })
    return records


def prefix_dir_name(record):
    if "window_id" in record:
        return f"window_{int(record['window_id']):04d}/prefix_{int(record['tail']):06d}"
    return f"prefix_{int(record['tail']):06d}"


def streaming_prefix_records_for_scene(scene, args):
    """Build streaming-prefix records for one scene.

    ScanNet++ defaults to motion-normalized temporal windows.  Legacy contiguous
    indexing is still available through ``--temporal-sampling none``.
    """
    num_frames = len(scene["img_files"])
    max_record_len = max(args.prefix_lengths or [args.prefix_max_len])
    max_record_len = min(int(max_record_len), int(args.prefix_max_len), int(args.num_frames), num_frames)
    if (
        args.mode == "streaming_prefix"
        and args.temporal_sampling == "motion_uniform"
        and scene.get("source") == "scannetpp"
        and "poses_c2w" in scene
    ):
        windows = temporal_windows_from_poses(
            scene["poses_c2w"],
            observations_per_window=max_record_len,
            motion_step=args.streaming_motion_step,
            rotation_weight=args.streaming_rotation_weight,
            window_stride=args.streaming_window_stride,
            max_windows_per_scene=args.streaming_max_windows_per_scene,
        )
        records = []
        for window in windows:
            indices = list(window["indices"])
            lengths = args.prefix_lengths or list(range(args.prefix_min_len, len(indices) + 1, args.prefix_stride))
            for length in lengths:
                length = int(length)
                if length < args.prefix_min_len or length > len(indices):
                    continue
                records.append({
                    "tail": length - 1,
                    "indices": indices[:length],
                    "window_indices": indices,
                    "window_id": int(window["window_id"]),
                    "obs_start": int(window["obs_start"]),
                    "valid_length": length,
                    "sampling": window["sampling"],
                    "motion_step": float(window["motion_step"]),
                    "rotation_weight": float(window["rotation_weight"]),
                })
        return records

    return streaming_prefix_records(
        num_frames,
        min_len=args.prefix_min_len,
        max_len=args.prefix_max_len,
        stride=args.prefix_stride,
        model_max_len=args.num_frames,
        lengths=args.prefix_lengths,
    )


def target_indices_for_scene(scene, args):
    """Return exact local frame ids for target_isolated extraction.

    In streaming C probes, the input prefix comes from a motion-normalized
    ScanNet++ window and the target frame is selected by prefix length +
    future horizon inside that same window.  The target cache still stores
    scene-local indices so InsScene15KDataset can exact-match target_indices.npy.
    """
    if getattr(args, "target_from_streaming_windows", False):
        prefixes = args.target_prefix_lengths or [8]
        horizons = args.target_horizons or [1, 2, 4]
        required_obs = max(
            [max(args.prefix_lengths or [0])]
            + [int(prefix) + int(horizon) for prefix in prefixes for horizon in horizons]
        )
        num_frames = len(scene["img_files"])
        required_obs = min(int(required_obs), num_frames)
        if (
            args.temporal_sampling == "motion_uniform"
            and scene.get("source") == "scannetpp"
            and "poses_c2w" in scene
        ):
            windows = temporal_windows_from_poses(
                scene["poses_c2w"],
                observations_per_window=required_obs,
                motion_step=args.streaming_motion_step,
                rotation_weight=args.streaming_rotation_weight,
                window_stride=args.streaming_window_stride,
                max_windows_per_scene=args.streaming_max_windows_per_scene,
            )
        else:
            windows = [{
                "window_id": 0,
                "indices": list(range(required_obs)),
            }]

        target_ids = set()
        for window in windows:
            indices = [int(i) for i in window["indices"]]
            for prefix_len in prefixes:
                tail = int(prefix_len) - 1
                if tail < 0 or tail >= len(indices):
                    continue
                for horizon in horizons:
                    pos = tail + int(horizon)
                    if 0 <= pos < len(indices):
                        target_ids.add(int(indices[pos]))
        return sorted(target_ids)

    if args.num_targets == 0:
        return list(range(len(scene["img_files"])))
    count = max(2, min(int(args.num_targets), len(scene["img_files"])))
    return np.linspace(0, len(scene["img_files"]) - 1, count).round().astype(int).tolist()


def context_segment_records(num_frames, context_len=76, stride=1, min_tail=1):
    """Build causal sliding context records [I_start, ..., I_tail]."""
    context_len = max(int(context_len), 1)
    stride = max(int(stride), 1)
    min_tail = max(int(min_tail), 0)
    records = []
    for tail in range(min_tail, num_frames, stride):
        start = max(0, tail - context_len + 1)
        records.append({
            "start": start,
            "tail": tail,
            "indices": list(range(start, tail + 1)),
            "valid_length": tail - start + 1,
        })
    return records


def context_dir_name(record):
    return f"context_{int(record['start']):06d}_{int(record['tail']):06d}"


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
    parser.add_argument("--source", default="scannetpp",
                        choices=["all", "infinigen", "scannetpp"],
                        help="Which data source to process. Temporal streaming defaults to ScanNet++ only.")
    parser.add_argument("--vfm", default="wan",
                        choices=["wan", "cogvideox", "vjepa", "vjepa2", "dino", "aether", "opensora", "f3r"],
                        help="Which VFM to extract features from")
    parser.add_argument("--model-id", default=None,
                        help="Model ID (default per VFM)")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--t", type=int, default=749)
    parser.add_argument(
        "--output-layers",
        nargs="+",
        default=None,
        help=(
            "Backbone layers to cache. Accepts integers plus aliases "
            "`default`, `last`, and `all`. Defaults preserve the historical "
            "probe layer per VFM."
        ),
    )
    parser.add_argument(
        "--all-layers",
        action="store_true",
        help="Cache every registered layer for this VFM.",
    )
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
        "--mode", default="streaming_prefix",
        choices=["normal", "shuffled", "target_isolated", "streaming_prefix", "context_segment", "streaming_target"],
        help=(
            "streaming_prefix: independently forward prefixes [I_0, ..., I_t] "
            "and save them under prefix_<tail> (default project setting).\n"
            "normal: legacy standard clip forward.\n"
            "shuffled: shuffle frame order before VFM forward (for A3 abnormal probe). "
            "Output features are re-ordered to match the original frame order so that "
            "shuffled[i] = feature of frame i extracted under scrambled temporal context.\n"
            "target_isolated: extract clip-isolated features for M target frames "
            "(replicate each target frame to fill the clip; for C1 action-dynamics probe).\n"
            "context_segment: independently forward sliding causal input segments "
            "[I_start, ..., I_tail] and save them under context_<start>_<tail>.\n"
            "streaming_target: for each prefix_len p, extract isolated features for frames "
            "{tail+h for h in target_horizons} and save under prefix_<tail>/; "
            "designed to provide per-prefix C1/C2/C3 targets."
        ),
    )
    parser.add_argument("--shuffle-seed", type=int, default=42,
                        help="Seed used to permute frame order in --mode shuffled.")
    parser.add_argument("--num-targets", type=int, default=8,
                        help="Target frames per scene in target_isolated mode; 0 means every frame.")
    parser.add_argument("--target-from-streaming-windows", action="store_true",
                        help="In target_isolated mode, extract only target frames used by streaming C probes.")
    parser.add_argument("--target-prefix-lengths", default="8,12,16,24",
                        help="Input prefix lengths for streaming C target extraction, e.g. '8'.")
    parser.add_argument("--target-horizons", default="1,2,4",
                        help="Future observation offsets for streaming C target extraction.")
    parser.add_argument("--no-cache-checksum", action="store_true",
                        help="Skip SHA-256 in cache sidecars (faster, less robust).")
    parser.add_argument("--allow-legacy-cache", action="store_true",
                        help="Treat caches without a current provenance sidecar as complete.")
    parser.add_argument("--prefix-stride", type=int, default=1,
                        help="Frame stride between streaming-prefix tails.")
    parser.add_argument("--prefix-min-len", type=int, default=1,
                        help="Minimum streaming-prefix length.")
    parser.add_argument("--prefix-max-len", type=int, default=24,
                        help="Maximum streaming-prefix length before model padding.")
    parser.add_argument("--prefix-lengths", default="8,12,16,24",
                        help="Exact streaming prefix lengths to cache, e.g. '8,12,16,24'.")
    parser.add_argument(
        "--temporal-sampling",
        default="motion_uniform",
        choices=["motion_uniform", "none"],
        help="For ScanNet++ streaming prefixes, sample observation windows uniformly in camera motion.",
    )
    parser.add_argument("--streaming-motion-step", type=float, default=0.35,
                        help="Approximate camera-motion distance between sampled observations.")
    parser.add_argument("--streaming-rotation-weight", type=float, default=0.5,
                        help="Meters-equivalent weight for one radian of camera rotation.")
    parser.add_argument("--streaming-window-stride", type=int, default=8,
                        help="Stride in sampled observation steps between temporal windows.")
    parser.add_argument("--streaming-max-windows-per-scene", type=int, default=4,
                        help="Cap temporal windows per ScanNet++ scene; <=0 keeps all windows.")
    parser.add_argument("--context-len", type=int, default=76,
                        help="Maximum causal input segment length for context_segment mode.")
    parser.add_argument("--context-stride", type=int, default=1,
                        help="Tail-frame stride for context_segment mode.")
    parser.add_argument("--opensora-config", default="features/opensora/configs/diffusion/inference/640px.py",
                        help="Open-Sora inference config path (only used when --vfm opensora).")
    parser.add_argument("--dino-batch-size", type=int, default=64,
                        help="Per-device batch size for DINO feature extraction.")
    args = parser.parse_args()
    args.prefix_lengths = parse_int_list(args.prefix_lengths)
    args.target_prefix_lengths = parse_int_list(args.target_prefix_lengths) or [8, 12, 16, 24]
    args.target_horizons = parse_int_list(args.target_horizons) or [1, 2, 4]
    if (
        args.mode in {"normal", "shuffled", "context_segment", "streaming_target"}
        and os.environ.get("ALLOW_NON_STREAMING") != "1"
    ):
        raise SystemExit(
            f"--mode {args.mode} is a legacy non-streaming extraction path. "
            "Streaming is the default; use --mode streaming_prefix, or set "
            "ALLOW_NON_STREAMING=1 for intentional legacy extraction."
        )
    if (
        args.mode in {"streaming_prefix", "target_isolated", "context_segment", "streaming_target"}
        and args.source != "scannetpp"
        and os.environ.get("ALLOW_INFINIGEN_TEMPORAL") != "1"
    ):
        raise SystemExit(
            "Temporal streaming/target extraction is ScanNet++ only. "
            "Use --source scannetpp, or set ALLOW_INFINIGEN_TEMPORAL=1 only for "
            "explicit legacy/debug reproduction."
        )

    # Per-VFM defaults
    VFM_DEFAULTS = {
        "wan":       {"model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "num_frames": 81, "size": (480, 832)},
        "cogvideox": {"model_id": "THUDM/CogVideoX-5b-I2V",          "num_frames": 97, "size": (480, 720)},
        "vjepa":     {"model_id": None,                                "num_frames": 76, "size": (480, 832)},
        "vjepa2":    {"model_id": "facebook/vjepa2-vitl-fpc64-256",   "num_frames": 64, "size": (256, 256)},
        "dino":      {"model_id": "facebook/dinov2-large",             "num_frames": 64, "size": (420, 728)},
        "aether":    {"model_id": None,                                "num_frames": 81, "size": (480, 720)},
        "opensora":  {"model_id": None,                                "num_frames": 81, "size": (480, 720)},
        "f3r":       {"model_id": None,                                "num_frames": 64, "size": (288, 512)},
    }
    defaults = VFM_DEFAULTS[args.vfm]
    if args.model_id is None:
        args.model_id = defaults["model_id"]
    if args.num_frames is None:
        args.num_frames = defaults["num_frames"]
    args.resize = defaults["size"]
    args.output_layers = parse_layers_arg(
        args.output_layers,
        vfm_name=args.vfm,
        model_id=args.model_id,
        all_layers=args.all_layers,
    )
    log.info(f"Using output layers for {args.vfm}: {args.output_layers}")

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
    elif args.vfm == "dino":
        def out_fname(layer):
            return "feature.sft"  # DINO has no layers, single last-hidden-state file
    elif args.vfm in ("aether", "opensora"):
        def out_fname(layer):
            return f"feature_t{args.t}_layer{layer}.sft"
    elif args.vfm == "f3r":
        def out_fname(layer):
            return f"feature_l{layer}.sft"

    # The mode is differentiated by
    # `--out-root` (e.g. FEAT vs FEAT_SHUFFLED vs FEAT_TARGET) — the per-VFM
    # subdirectory keeps a clean, mode-agnostic name so InsScene15KDataset can
    # locate it via `vfm_name` regardless of mode.
    vfm_dir_name = args.vfm

    def cache_complete(path):
        path = Path(path)
        if not path.exists():
            return False
        sidecar = Path(f"{path}.manifest.json")
        if not sidecar.exists():
            return args.allow_legacy_cache
        try:
            meta = json.loads(sidecar.read_text())
            return (
                meta.get("frame_index_schema") == "image_mask_intersection_v1"
                and meta.get("mode") == args.mode
                and meta.get("model_id") == args.model_id
                and path.stat().st_size == meta.get("size_bytes")
            )
        except Exception:
            return False

    def target_index_complete(out_dir, expected_indices):
        if not expected_indices:
            return True
        path = os.path.join(out_dir, "target_indices.npy")
        if not os.path.exists(path):
            return False
        try:
            cached = np.load(path).astype(np.int64).tolist()
        except Exception:
            return False
        return cached == [int(v) for v in expected_indices]

    # Check how many are already done
    done = 0
    for s in scenes:
        name = scene_name(s)
        out_dir = os.path.join(args.out_root, vfm_dir_name, s["source"], name)
        if args.mode == "streaming_target":
            all_exist = True
            for p in (args.prefix_lengths or []):
                tail = p - 1
                if tail >= len(s["img_files"]):
                    continue
                prefix_out_dir = os.path.join(out_dir, f"prefix_{tail:06d}")
                if args.vfm == "vjepa":
                    if not cache_complete(os.path.join(prefix_out_dir, out_fname(0))):
                        all_exist = False
                        break
                else:
                    if not all(cache_complete(os.path.join(prefix_out_dir, out_fname(l)))
                               for l in args.output_layers):
                        all_exist = False
                        break
        elif args.mode in ("streaming_prefix", "context_segment"):
            records = (
                streaming_prefix_records_for_scene(s, args)
                if args.mode == "streaming_prefix"
                else context_segment_records(
                    len(s["img_files"]),
                    context_len=min(args.context_len, args.num_frames),
                    stride=args.context_stride,
                    min_tail=0,
                )
            )
            all_exist = True
            for record in records:
                prefix_out_dir = os.path.join(
                    out_dir,
                    prefix_dir_name(record) if args.mode == "streaming_prefix" else context_dir_name(record),
                )
                if args.vfm == "vjepa":
                    prefix_exists = cache_complete(os.path.join(prefix_out_dir, out_fname(0)))
                else:
                    prefix_exists = all(
                        cache_complete(os.path.join(prefix_out_dir, out_fname(l)))
                        for l in args.output_layers
                    )
                if not prefix_exists:
                    all_exist = False
                    break
        elif args.mode == "target_isolated":
            expected_targets = target_indices_for_scene(s, args)
            layer_complete = (
                cache_complete(os.path.join(out_dir, out_fname(0)))
                if args.vfm == "vjepa"
                else all(
                    cache_complete(os.path.join(out_dir, out_fname(l)))
                    for l in args.output_layers
                )
            )
            all_exist = target_index_complete(out_dir, expected_targets) and layer_complete
        elif args.vfm == "vjepa":
            all_exist = cache_complete(os.path.join(out_dir, out_fname(0)))
        else:
            all_exist = all(
                cache_complete(os.path.join(out_dir, out_fname(l)))
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
    from safetensors.torch import save_file as _save_file

    try:
        extractor_git_commit = subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL, timeout=15,
        ).strip()
        extractor_git_dirty = bool(subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "status", "--short"],
            text=True, stderr=subprocess.DEVNULL, timeout=15,
        ).strip())
    except Exception:
        extractor_git_commit = "unknown"
        extractor_git_dirty = None

    def save_file(tensors, out_path):
        """Atomically publish a cache and its provenance sidecar."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_name(f".{out_path.name}.tmp-{os.getpid()}")
        _save_file(tensors, str(tmp))
        digest = None
        if not args.no_cache_checksum:
            sha = hashlib.sha256()
            with open(tmp, "rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    sha.update(chunk)
            digest = sha.hexdigest()
        os.replace(tmp, out_path)
        manifest = {
            "schema_version": 1,
            "frame_index_schema": "image_mask_intersection_v1",
            "extractor_git_commit": extractor_git_commit,
            "extractor_git_dirty": extractor_git_dirty,
            "vfm": args.vfm,
            "model_id": args.model_id,
            "mode": args.mode,
            "timestep": args.t,
            "prompt": args.prompt,
            "resize": list(args.resize),
            "num_frames": args.num_frames,
            "file": out_path.name,
            "size_bytes": out_path.stat().st_size,
            "sha256": digest,
            "tensors": {
                key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in tensors.items()
            },
        }
        sidecar = Path(f"{out_path}.manifest.json")
        sidecar_tmp = sidecar.with_name(f".{sidecar.name}.tmp-{os.getpid()}")
        sidecar_tmp.write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(sidecar_tmp, sidecar)

    def atomic_save_npy(path, array):
        path = Path(path)
        tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        with open(tmp, "wb") as handle:
            np.save(handle, array)
        digest = None
        if not args.no_cache_checksum:
            digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
        os.replace(tmp, path)
        meta = {
            "schema_version": 1,
            "file": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": digest,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
        sidecar = Path(f"{path}.manifest.json")
        sidecar_tmp = sidecar.with_name(f".{sidecar.name}.tmp-{os.getpid()}")
        sidecar_tmp.write_text(json.dumps(meta, indent=2) + "\n")
        os.replace(sidecar_tmp, sidecar)

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
    elif args.vfm == "dino":
        from features.dino.extract_features import dino_forward as _dino_forward, reshape_tokens as _dino_reshape
        from transformers import AutoImageProcessor, Dinov2Model
        log.info(f"Loading DINOv2 model: {args.model_id}")
        _dino_proc = AutoImageProcessor.from_pretrained(args.model_id)
        _dino_model = Dinov2Model.from_pretrained(args.model_id).cuda().eval()
        model = (_dino_model, _dino_proc)  # stored as a tuple
    elif args.vfm == "aether":
        from features.aether.aether_feature import get_aether_featurizer
        from features.aether.extract_features import forward_aether
        log.info("Loading Aether featurizer (may take several minutes)...")
        model = get_aether_featurizer()
    elif args.vfm == "opensora":
        from features.opensora.opensora_features import extract_feature as opensora_extract
        log.info("OpenSora: will initialise per-scene via config_path=%s", args.opensora_config)
        model = None  # opensora builds its pipeline internally on each call
    elif args.vfm == "f3r":
        from features.f3r.extract_features import (
            vidfm3d_forward as f3r_forward,
            load_model as f3r_load_model,
            HTOK as F3R_HTOK, WTOK as F3R_WTOK, C as F3R_C,
        )
        _f3r_model_id = args.model_id or "jedyang97/Fast3R_ViT_Large_512"
        log.info(f"Loading Fast3R model: {_f3r_model_id}")
        _f3r_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = f3r_load_model(_f3r_model_id, str(_f3r_device))

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
        if args.mode == "streaming_target":
            scene_complete = True
            for p in (args.prefix_lengths or []):
                tail = p - 1
                if tail >= len(s["img_files"]):
                    continue
                prefix_out_dir = os.path.join(out_dir, f"prefix_{tail:06d}")
                if args.vfm == "vjepa":
                    if not cache_complete(os.path.join(prefix_out_dir, out_fname(0))):
                        scene_complete = False
                        break
                else:
                    if not all(cache_complete(os.path.join(prefix_out_dir, out_fname(l)))
                               for l in args.output_layers):
                        scene_complete = False
                        break
            if scene_complete:
                continue
        elif args.mode in ("streaming_prefix", "context_segment"):
            records = (
                streaming_prefix_records_for_scene(s, args)
                if args.mode == "streaming_prefix"
                else context_segment_records(
                    len(s["img_files"]),
                    context_len=min(args.context_len, args.num_frames),
                    stride=args.context_stride,
                    min_tail=0,
                )
            )
            scene_complete = True
            for record in records:
                prefix_out_dir = os.path.join(
                    out_dir,
                    prefix_dir_name(record) if args.mode == "streaming_prefix" else context_dir_name(record),
                )
                if args.vfm == "vjepa":
                    prefix_complete = cache_complete(os.path.join(prefix_out_dir, out_fname(0)))
                else:
                    prefix_complete = all(
                        cache_complete(os.path.join(prefix_out_dir, out_fname(l)))
                        for l in args.output_layers
                    )
                if not prefix_complete:
                    scene_complete = False
                    break
            if scene_complete:
                continue
        elif args.mode == "target_isolated":
            expected_targets = target_indices_for_scene(s, args)
            if not expected_targets:
                log.warning(f"{s['source']}/{name}: no streaming C target frames selected; skip target_isolated")
                continue
            target_ok = target_index_complete(out_dir, expected_targets)
            if args.vfm == "vjepa":
                missing_layers = [] if target_ok and cache_complete(os.path.join(out_dir, out_fname(0))) else [0]
            else:
                missing_layers = [
                    l for l in args.output_layers
                    if not (target_ok and cache_complete(os.path.join(out_dir, out_fname(l))))
                ]
            if not missing_layers:
                continue
        elif args.vfm == "vjepa":
            if cache_complete(os.path.join(out_dir, out_fname(0))):
                continue
        else:
            missing_layers = [
                l for l in args.output_layers
                if not cache_complete(os.path.join(out_dir, out_fname(l)))
            ]
            if not missing_layers:
                continue

        t0 = time.time()
        try:
            if args.mode in ("streaming_prefix", "context_segment"):
                os.makedirs(out_dir, exist_ok=True)
                records = (
                    streaming_prefix_records_for_scene(s, args)
                    if args.mode == "streaming_prefix"
                    else context_segment_records(
                        len(s["img_files"]),
                        context_len=min(args.context_len, args.num_frames),
                        stride=args.context_stride,
                        min_tail=0,
                    )
                )
                if not records:
                    log.warning(f"{s['source']}/{name}: no {args.mode} records to process")
                    continue

                prefix_meta = []
                processed_prefixes = 0
                skipped_prefixes = 0
                for record in records:
                    prefix_out_dir = os.path.join(
                        out_dir,
                        prefix_dir_name(record) if args.mode == "streaming_prefix" else context_dir_name(record),
                    )
                    if args.vfm == "vjepa":
                        missing_layers = [0] if not cache_complete(
                            os.path.join(prefix_out_dir, out_fname(0))
                        ) else []
                    else:
                        missing_layers = [
                            l for l in args.output_layers
                            if not cache_complete(os.path.join(prefix_out_dir, out_fname(l)))
                        ]
                    if not missing_layers:
                        skipped_prefixes += 1
                        meta_record = dict(record)
                        meta_record["input_length"] = args.num_frames
                        meta_record["pad_mode"] = "repeat_tail"
                        prefix_meta.append(meta_record)
                        continue

                    frames_input = load_and_resize_frames(
                        s["img_dir"],
                        s["img_files"],
                        record["indices"],
                        size=args.resize,
                    )
                    valid_length = len(frames_input)
                    while len(frames_input) < args.num_frames:
                        frames_input.append(frames_input[-1])
                    if len(frames_input) > args.num_frames:
                        raise ValueError(
                            f"{args.mode} length {len(frames_input)} exceeds "
                            f"model input length {args.num_frames}; lower the segment length"
                        )
                    os.makedirs(prefix_out_dir, exist_ok=True)

                    if args.vfm == "wan":
                        with torch.no_grad():
                            feats = model.forward(
                                video=frames_input, prompt=args.prompt, t=args.t,
                                output_layer_indices=missing_layers,
                                ensemble_size=args.ensemble,
                            )
                        for layer_id, raw_feat in feats.items():
                            reshaped = reshape_to_t_h_w_c(raw_feat)
                            out_path = os.path.join(prefix_out_dir, out_fname(layer_id))
                            save_file({"feat": reshaped.half()}, out_path)

                    elif args.vfm == "cogvideox":
                        feats = forward_cogvideox(model, frames_input, t=args.t, layer_ids=missing_layers)
                        for layer_id, feat in feats.items():
                            out_path = os.path.join(prefix_out_dir, out_fname(layer_id))
                            save_file({"feat": feat.half()}, out_path)

                    elif args.vfm == "vjepa":
                        feats = model(frames_input)
                        out_path = os.path.join(prefix_out_dir, out_fname(0))
                        save_file({"feat": feats.half().contiguous()}, out_path)

                    elif args.vfm == "vjepa2":
                        feats = model(frames_input, output_layers=missing_layers)
                        for layer_id, feat in feats.items():
                            out_path = os.path.join(prefix_out_dir, out_fname(layer_id))
                            save_file({"feat": feat.half().contiguous()}, out_path)

                    elif args.vfm == "dino":
                        _dino_model, _dino_proc = model
                        _dev = next(_dino_model.parameters()).device
                        with torch.no_grad():
                            all_feats = []
                            for bi in range(0, len(frames_input), args.dino_batch_size):
                                batch = frames_input[bi:bi + args.dino_batch_size]
                                tok = _dino_forward(_dino_model, _dino_proc, batch, _dev)
                                for b in range(tok.shape[0]):
                                    all_feats.append(_dino_reshape(tok[b]))
                        feat = torch.stack(all_feats, dim=0)
                        out_path = os.path.join(prefix_out_dir, out_fname(0))
                        save_file({"feat": feat.half().contiguous()}, out_path)

                    elif args.vfm == "aether":
                        feats = model.forward(
                            frames_input, t=args.t, output_layer_indices=missing_layers
                        )
                        for layer_id, feat in feats.items():
                            h_lat = args.resize[0] // 16
                            w_lat = args.resize[1] // 16
                            t_lat = feat.shape[1] // (h_lat * w_lat)
                            reshaped = feat[0].reshape(t_lat, h_lat, w_lat, -1)
                            out_path = os.path.join(prefix_out_dir, out_fname(layer_id))
                            save_file({"feat": reshaped.half().contiguous()}, out_path)

                    elif args.vfm == "opensora":
                        feats = opensora_extract(
                            frames_input, layer_indices=missing_layers,
                            config_path=args.opensora_config,
                        )
                        for layer_id, feat in feats.items():
                            out_path = os.path.join(prefix_out_dir, out_fname(layer_id))
                            save_file({"feat": feat.half().contiguous()}, out_path)

                    elif args.vfm == "f3r":
                        f3r_indices = list(record["indices"])
                        while len(f3r_indices) < args.num_frames:
                            f3r_indices.append(f3r_indices[-1])
                        f3r_filelist = [
                            os.path.join(s["img_dir"], s["img_files"][idx])
                            for idx in f3r_indices
                        ]
                        _dev = next(model.parameters()).device
                        raw_feats = f3r_forward(f3r_filelist, model, _dev)
                        if isinstance(raw_feats, torch.Tensor):
                            raw_feats = [raw_feats]
                        for layer_id in missing_layers:
                            raw = raw_feats[layer_id]
                            feat_spatial = raw.reshape(
                                len(f3r_filelist), F3R_HTOK, F3R_WTOK, F3R_C
                            ).contiguous()
                            out_path = os.path.join(prefix_out_dir, out_fname(layer_id))
                            save_file({"feat": feat_spatial.half()}, out_path)

                    meta_record = dict(record)
                    meta_record["input_length"] = len(frames_input)
                    meta_record["valid_length"] = valid_length
                    meta_record["pad_mode"] = "repeat_tail"
                    prefix_meta.append(meta_record)
                    processed_prefixes += 1

                atomic_save_npy(
                    os.path.join(
                        out_dir,
                        "prefix_index.npy" if args.mode == "streaming_prefix" else "context_index.npy",
                    ),
                    np.array(prefix_meta, dtype=object),
                )

                elapsed = time.time() - t0
                total_time += elapsed
                processed += 1
                remaining = len(scenes) - done - processed - failed
                avg = total_time / processed
                eta = str(timedelta(seconds=int(avg * remaining)))
                log.info(
                    f"[{done + processed + failed}/{len(scenes)}] "
                    f"{s['source']}/{name}: {elapsed:.1f}s "
                    f"({len(s['img_files'])} frames, {args.mode} +{processed_prefixes}/skip {skipped_prefixes}) "
                    f"ETA: {eta}"
                )
                continue

            elif args.mode == "streaming_target":
                if args.prefix_lengths is None:
                    raise ValueError("--prefix-lengths required for streaming_target mode")
                horizons = args.target_horizons or [1, 2, 4]
                num_scene_frames = len(s["img_files"])
                os.makedirs(out_dir, exist_ok=True)
                processed_prefixes = 0
                skipped_prefixes = 0
                for p in args.prefix_lengths:
                    tail = p - 1
                    if tail >= num_scene_frames:
                        log.warning(f"{s['source']}/{name}: prefix {p} (tail={tail}) >= scene frames {num_scene_frames}, skipping")
                        continue
                    prefix_out_dir = os.path.join(out_dir, f"prefix_{tail:06d}")
                    if args.vfm == "vjepa":
                        missing_lyrs = [] if cache_complete(os.path.join(prefix_out_dir, out_fname(0))) else [0]
                    else:
                        missing_lyrs = [l for l in args.output_layers
                                        if not cache_complete(os.path.join(prefix_out_dir, out_fname(l)))]
                    if not missing_lyrs:
                        skipped_prefixes += 1
                        continue
                    target_frame_ids = [tail + h for h in horizons if (tail + h) < num_scene_frames]
                    if not target_frame_ids:
                        log.warning(f"{s['source']}/{name} prefix {p}: no valid target frames, skipping")
                        continue
                    target_imgs = [
                        Image.open(os.path.join(s["img_dir"], s["img_files"][gi]))
                        .convert("RGB").resize((args.resize[1], args.resize[0]), Image.LANCZOS)
                        for gi in target_frame_ids
                    ]
                    per_layer_collect = {l: [] for l in missing_lyrs}
                    for tgt_img in target_imgs:
                        rep_frames = [tgt_img] * args.num_frames
                        if args.vfm == "wan":
                            with torch.no_grad():
                                feats = model.forward(
                                    video=rep_frames, prompt=args.prompt, t=args.t,
                                    output_layer_indices=missing_lyrs,
                                    ensemble_size=args.ensemble,
                                )
                            for layer_id, raw_feat in feats.items():
                                reshaped = reshape_to_t_h_w_c(raw_feat)
                                per_layer_collect[layer_id].append(reshaped[reshaped.shape[0] // 2])
                        elif args.vfm == "cogvideox":
                            feats = forward_cogvideox(model, rep_frames, t=args.t, layer_ids=missing_lyrs)
                            for layer_id, feat in feats.items():
                                f = feat.reshape(-1, *feat.shape[2:])
                                per_layer_collect[layer_id].append(f[f.shape[0] // 2])
                        elif args.vfm == "vjepa2":
                            feats = model(rep_frames, output_layers=missing_lyrs)
                            for layer_id, feat in feats.items():
                                per_layer_collect[layer_id].append(feat[feat.shape[0] // 2])
                        else:
                            raise NotImplementedError(
                                f"streaming_target mode not implemented for vfm={args.vfm}")
                    os.makedirs(prefix_out_dir, exist_ok=True)
                    for layer_id, lst in per_layer_collect.items():
                        stacked = torch.stack(lst, dim=0).half().contiguous()
                        out_path = os.path.join(prefix_out_dir, out_fname(layer_id))
                        save_file({"feat": stacked}, out_path)
                    atomic_save_npy(
                        os.path.join(prefix_out_dir, "target_indices.npy"),
                        np.array(target_frame_ids, dtype=np.int64),
                    )
                    processed_prefixes += 1

                elapsed = time.time() - t0
                total_time += elapsed
                processed += 1
                remaining = len(scenes) - done - processed - failed
                avg = total_time / processed
                eta = str(timedelta(seconds=int(avg * remaining)))
                log.info(
                    f"[{done + processed + failed}/{len(scenes)}] "
                    f"{s['source']}/{name}: {elapsed:.1f}s "
                    f"(streaming_target +{processed_prefixes}/skip {skipped_prefixes}) "
                    f"ETA: {eta}"
                )
                continue

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
                target_global = target_indices_for_scene(s, args)
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

                for gi, tgt_img in zip(target_global, target_imgs):
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
                    elif args.vfm == "dino":
                        _dino_model, _dino_proc = model
                        _dev = next(_dino_model.parameters()).device
                        with torch.no_grad():
                            tok = _dino_forward(_dino_model, _dino_proc, [tgt_img], _dev)
                        per_layer_collect[0].append(_dino_reshape(tok[0]))
                    elif args.vfm == "aether":
                        with torch.no_grad():
                            feats = model.forward(
                                rep_frames, t=args.t, output_layer_indices=missing_layers
                            )
                        for layer_id, feat in feats.items():
                            h_lat = args.resize[0] // 16
                            w_lat = args.resize[1] // 16
                            t_lat = feat.shape[1] // (h_lat * w_lat)
                            reshaped = feat[0].reshape(t_lat, h_lat, w_lat, -1)
                            per_layer_collect[layer_id].append(reshaped[reshaped.shape[0] // 2])
                    elif args.vfm == "f3r":
                        target_path = os.path.join(s["img_dir"], s["img_files"][int(gi)])
                        f3r_filelist = [target_path] * args.num_frames
                        _dev = next(model.parameters()).device
                        with torch.no_grad():
                            raw_feats = f3r_forward(f3r_filelist, model, _dev)
                        if isinstance(raw_feats, torch.Tensor):
                            raw_feats = [raw_feats]
                        for layer_id in missing_layers:
                            raw = raw_feats[layer_id]
                            feat_spatial = raw.reshape(
                                len(f3r_filelist), F3R_HTOK, F3R_WTOK, F3R_C
                            ).contiguous()
                            per_layer_collect[layer_id].append(feat_spatial[feat_spatial.shape[0] // 2])
                    else:
                        raise NotImplementedError(
                            f"target_isolated mode not implemented for vfm={args.vfm}"
                        )

                for layer_id, lst in per_layer_collect.items():
                    stacked = torch.stack(lst, dim=0).half().contiguous()
                    out_path = os.path.join(out_dir, out_fname(layer_id))
                    save_file({"feat": stacked}, out_path)
                # Save target index metadata (one per scene, layer-agnostic)
                atomic_save_npy(
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

            elif args.vfm == "dino":
                _dino_model, _dino_proc = model
                _dev = next(_dino_model.parameters()).device
                with torch.no_grad():
                    all_feats = []
                    for bi in range(0, len(frames_input), args.dino_batch_size):
                        batch = frames_input[bi:bi + args.dino_batch_size]
                        tok = _dino_forward(_dino_model, _dino_proc, batch, _dev)
                        for b in range(tok.shape[0]):
                            all_feats.append(_dino_reshape(tok[b]))
                feat = torch.stack(all_feats, dim=0)
                out_path = os.path.join(out_dir, out_fname(0))
                save_file({"feat": feat.half().contiguous()}, out_path)

            elif args.vfm == "aether":
                feats = model.forward(
                    frames_input, t=args.t, output_layer_indices=missing_layers
                )
                for layer_id, feat in feats.items():
                    h_lat = args.resize[0] // 16
                    w_lat = args.resize[1] // 16
                    t_lat = feat.shape[1] // (h_lat * w_lat)
                    reshaped = feat[0].reshape(t_lat, h_lat, w_lat, -1)
                    out_path = os.path.join(out_dir, out_fname(layer_id))
                    save_file({"feat": reshaped.half().contiguous()}, out_path)

            elif args.vfm == "opensora":
                feats = opensora_extract(
                    frames_input, layer_indices=missing_layers,
                    config_path=args.opensora_config,
                )
                for layer_id, feat in feats.items():
                    out_path = os.path.join(out_dir, out_fname(layer_id))
                    save_file({"feat": feat.half().contiguous()}, out_path)

            elif args.vfm == "f3r":
                f3r_filelist = [
                    os.path.join(s["img_dir"], s["img_files"][idx]) for idx in indices
                ]
                _dev = next(model.parameters()).device
                raw_feats = f3r_forward(f3r_filelist, model, _dev)
                if isinstance(raw_feats, torch.Tensor):
                    raw_feats = [raw_feats]
                for layer_id in missing_layers:
                    raw = raw_feats[layer_id]
                    feat_spatial = raw.reshape(
                        len(f3r_filelist), F3R_HTOK, F3R_WTOK, F3R_C
                    ).contiguous()
                    out_path = os.path.join(out_dir, out_fname(layer_id))
                    save_file({"feat": feat_spatial.half()}, out_path)

            elapsed = time.time() - t0
            total_time += elapsed
            processed += 1
            remaining = len(scenes) - done - processed - failed
            avg = total_time / processed
            sampled_count = len(target_global) if args.mode == "target_isolated" else len(indices)
            eta = str(timedelta(seconds=int(avg * remaining)))
            log.info(
                f"[{done + processed + failed}/{len(scenes)}] "
                f"{s['source']}/{name}: {elapsed:.1f}s "
                f"({len(s['img_files'])} frames, sampled {sampled_count}) "
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
