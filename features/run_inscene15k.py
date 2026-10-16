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

from features.noise_seed import (
    DETERMINISTIC_NOISE_CONTRACT,
    MINIMAX_FIXED_NOISE_SEED,
    MINIMAX_NOISE_SEED_SCHEME,
    NO_NOISE_SEED_SCHEME,
    NOISE_SEED_SCHEME,
    canonical_noise_metadata,
    canonical_noise_sample,
    deterministic_no_noise_metadata,
    make_noise_generator,
    noise_metadata_matches,
)
from features.offline_hf import resolve_offline_snapshot

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


CANONICAL_STREAMING_PROTOCOL = "inscene_streaming_canonical_v2_obs28_h124"
CANONICAL_STREAMING_OBS_LEN = 28
CANONICAL_GAP_NAMESPACES = {
    "FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1",
    "FEAT_TARGET_CANONICAL_V2_GAP_DELTA_V1",
}
CANONICAL_STREAMING_PROTOCOL_SHA256 = (
    "215c720c8537f4dfae6243bedcc7339fbb7e3ba8a6ca2dcaf04f49cd4b738e4f"
)
SEEDED_FEATURE_NAMESPACES = frozenset({
    "FEAT_STREAMING_CANONICAL_V2_SEEDED_V1",
    "FEAT_TARGET_CANONICAL_V2_SEEDED_V1",
    # bf16 re-extraction of the MiniMax-H3 seeded-v1 caches whose fp16 files
    # overflow at layers 20/25 (audit 2026-10-15/16). Same protocol, same
    # deterministic-noise contract; only the storage dtype differs.
    "FEAT_STREAMING_CANONICAL_V2_SEEDED_V1_MINIMAX_BF16_V1",
    "FEAT_TARGET_CANONICAL_V2_SEEDED_V1_MINIMAX_BF16_V1",
})
# Canonical MiniMax-H3 namespaces produced with the bf16 storage contract.
MINIMAX_BF16_FEATURE_NAMESPACES = frozenset({
    "FEAT_STREAMING_CANONICAL_V2_MINIMAX_BF16_V1",
    "FEAT_TARGET_CANONICAL_V2_MINIMAX_BF16_V1",
    "FEAT_STREAMING_CANONICAL_V2_MINIMAX_BF16_V2",
    "FEAT_TARGET_CANONICAL_V2_MINIMAX_BF16_V2",
    "FEAT_STREAMING_CANONICAL_V2_SEEDED_V1_MINIMAX_BF16_V1",
    "FEAT_TARGET_CANONICAL_V2_SEEDED_V1_MINIMAX_BF16_V1",
})
SAMPLE_LOCAL_NOISE_VFMS = frozenset({
    "wan",
    "cogvideox",
    "aether",
    "f3r",
    "lingbot",
})
DETERMINISTIC_NO_NOISE_VFMS = frozenset({"dino", "vjepa2", "videomae"})
SEEDED_SUPPORTED_VFMS = frozenset(
    SAMPLE_LOCAL_NOISE_VFMS | DETERMINISTIC_NO_NOISE_VFMS | {"minimax_h3"}
)

# MiniMax-H3 activations reach |x| ~ 4e6 in deep blocks; fp16 (max 65,504)
# overflows to inf, which was observed in the fp16 timestep caches at layers
# 20/25 and in the legacy t=250 caches. MiniMax caches are therefore stored as
# bf16 and every tensor is checked for finiteness before and after conversion.
# This contract was first used by the frozen extractor behind the canonical
# MINIMAX_BF16_V2 namespaces (2026-09-23) and is merged here unchanged.
MINIMAX_STORAGE_CONTRACT = "MINIMAX_H3_FINITE_BF16_V1"


def validate_finite_feature(tensor, *, context):
    """Fail closed on non-finite model features without altering values."""
    if not isinstance(tensor, torch.Tensor) or not torch.is_floating_point(tensor):
        raise TypeError(
            f"{context}: expected a floating torch.Tensor, got {type(tensor)!r}"
        )
    finite = torch.isfinite(tensor)
    if not bool(finite.all().item()):
        nonfinite = int((~finite).sum().item())
        positive_inf = int(torch.isposinf(tensor).sum().item())
        negative_inf = int(torch.isneginf(tensor).sum().item())
        nan = int(torch.isnan(tensor).sum().item())
        raise FloatingPointError(
            f"{context}: refusing non-finite feature: total={nonfinite} "
            f"nan={nan} posinf={positive_inf} neginf={negative_inf} "
            f"dtype={tensor.dtype} shape={tuple(tensor.shape)}"
        )
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "max_abs": float(tensor.abs().amax().item()) if tensor.numel() else 0.0,
    }


def prepare_feature_for_storage(tensor, *, vfm, context):
    """Apply the versioned storage policy while preserving feature semantics."""
    if vfm == "minimax_h3":
        validate_finite_feature(tensor, context=f"{context}:pre_bf16")
        stored = tensor.to(dtype=torch.bfloat16).contiguous()
        validate_finite_feature(stored, context=f"{context}:post_bf16")
        return stored
    return tensor.half().contiguous()

# These are deliberately small structural checks, not a replacement for the
# loader's own weight validation.  Their purpose is to ensure that an offline
# job resolves a revision-pinned local snapshot instead of passing a Hub ID to
# a Diffusers/Transformers loader that may call ``model_info``.
OFFLINE_HF_REQUIRED_PATHS = {
    "dino": (
        "config.json",
        "preprocessor_config.json",
        "model.safetensors",
    ),
    "vjepa2": (
        "config.json",
        "video_preprocessor_config.json",
        "model.safetensors",
    ),
    "videomae": (
        "config.json",
        "preprocessor_config.json",
        "model.safetensors",
    ),
    "wan": (
        "model_index.json",
        "scheduler/scheduler_config.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
        "text_encoder/config.json",
        "text_encoder/model.safetensors.index.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/special_tokens_map.json",
        "tokenizer/spiece.model",
    ),
    "cogvideox": (
        "configuration.json",
        "model_index.json",
        "scheduler/scheduler_config.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
        "text_encoder/config.json",
        "text_encoder/model.safetensors.index.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/added_tokens.json",
        "tokenizer/special_tokens_map.json",
        "tokenizer/spiece.model",
    ),
}


def resolve_model_load_id(vfm, model_id):
    """Resolve a physical local checkpoint without changing logical identity."""
    if vfm not in OFFLINE_HF_REQUIRED_PATHS:
        return model_id, None
    return resolve_offline_snapshot(
        model_id,
        required_paths=OFFLINE_HF_REQUIRED_PATHS[vfm],
    )


def merge_extra_metadata(manifest, extra_metadata):
    """Merge producer metadata without permitting an identity override.

    Canonical gap helpers deliberately repeat immutable fields such as
    ``canonical_protocol_sha256`` that the generic sidecar writer already
    establishes.  A duplicate is safe only when it is an exact value match;
    a differing value remains a fail-closed provenance error.
    """
    if not extra_metadata:
        return
    conflicting = sorted(
        key
        for key, value in extra_metadata.items()
        if key in manifest and manifest[key] != value
    )
    if conflicting:
        raise RuntimeError(
            "feature metadata attempts to replace reserved fields: "
            + ",".join(conflicting)
        )
    for key, value in extra_metadata.items():
        if key not in manifest:
            manifest[key] = value


def is_seeded_feature_namespace(namespace):
    return str(namespace or "").strip() in SEEDED_FEATURE_NAMESPACES


def stable_file_sha256(path, cache=None):
    """Hash a stable regular file and optionally memoize by full stat identity."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        return None
    before = path.stat(follow_symlinks=False)
    signature = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    cache_key = str(path.absolute())
    cached = cache.get(cache_key) if cache is not None else None
    if cached is not None and cached[0] == signature:
        return cached[1]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat(follow_symlinks=False)
    after_signature = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if after_signature != signature:
        return None
    value = digest.hexdigest()
    if cache is not None:
        cache[cache_key] = (signature, value)
    return value


def validate_seeded_cache_policy(args, feature_namespace):
    """Fail closed before cache discovery for the seeded production namespace."""
    if not is_seeded_feature_namespace(feature_namespace):
        return
    if not is_canonical_feature_mode(args):
        raise ValueError(
            "seeded-v1 feature namespaces are reserved for canonical "
            "streaming_prefix/streaming-window target_isolated extraction"
        )
    if args.vfm not in SEEDED_SUPPORTED_VFMS:
        raise ValueError(f"seeded-v1 extraction does not support vfm={args.vfm!r}")
    if args.no_cache_checksum:
        raise ValueError(
            "seeded-v1 extraction forbids --no-cache-checksum; every local "
            "feature cache must be verified against its sidecar SHA-256"
        )
    protocol = canonical_protocol_metadata(args)
    if protocol["protocol_sha256"] != CANONICAL_STREAMING_PROTOCOL_SHA256:
        raise ValueError(
            "seeded-v1 runtime protocol does not exactly match the frozen "
            f"canonical protocol (expected {CANONICAL_STREAMING_PROTOCOL_SHA256}, "
            f"got {protocol['protocol_sha256']})"
        )


def seeded_feature_metadata_matches(
    path,
    metadata,
    *,
    vfm,
    model_id,
    mode,
    feature_namespace,
    timestep,
    prompt,
    resize,
    num_frames,
    expected_noise_metadata,
    actual_sha256,
):
    """Validate every identity-bearing field of a seeded local feature cache.

    ``actual_sha256`` is computed by the caller from a stable file snapshot so
    repeated cache probes can memoize the expensive read without weakening the
    contract checked here.
    """
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        return False
    if not isinstance(metadata, dict) or not isinstance(expected_noise_metadata, dict):
        return False
    if not isinstance(actual_sha256, str) or len(actual_sha256) != 64:
        return False
    try:
        int(actual_sha256, 16)
    except ValueError:
        return False
    expected = {
        "schema_version": 1,
        "frame_index_schema": "image_mask_intersection_v1",
        "vfm": str(vfm),
        "model_id": model_id,
        "mode": str(mode),
        "feature_namespace": str(feature_namespace),
        "timestep": int(timestep),
        "prompt": str(prompt),
        "resize": [int(value) for value in resize],
        "num_frames": int(num_frames),
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": actual_sha256,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        return False
    return noise_metadata_matches(metadata, expected_noise_metadata)


def is_canonical_feature_mode(args):
    return args.mode == "streaming_prefix" or (
        args.mode == "target_isolated" and args.target_from_streaming_windows
    )


def requires_sample_local_noise(args):
    return is_canonical_feature_mode(args) and args.vfm in SAMPLE_LOCAL_NOISE_VFMS


def backend_generation_contract(args):
    """Return the strict backend-level deterministic generation contract."""

    if args.vfm in SAMPLE_LOCAL_NOISE_VFMS:
        return {
            "feature_generation_contract": DETERMINISTIC_NOISE_CONTRACT,
            "noise_seed_scheme": NOISE_SEED_SCHEME,
        }
    if args.vfm == "minimax_h3":
        return {
            "feature_generation_contract": DETERMINISTIC_NOISE_CONTRACT,
            "noise_seed_scheme": MINIMAX_NOISE_SEED_SCHEME,
            "noise_seed": MINIMAX_FIXED_NOISE_SEED,
            "noise_seed_scope": "reset_per_forward",
        }
    if args.vfm in DETERMINISTIC_NO_NOISE_VFMS:
        return deterministic_no_noise_metadata()
    return None


def streaming_noise_metadata(args, scene, record):
    """Return strict sidecar metadata for one canonical streaming prefix."""
    protocol = canonical_protocol_metadata(args)
    sample = canonical_noise_sample(
        protocol=protocol,
        vfm=args.vfm,
        model_id=args.model_id,
        scene=scene,
        sample_kind="streaming_prefix",
        window_id=int(record["window_id"]),
        prefix_len=int(record["valid_length"]),
    )
    return canonical_noise_metadata([sample])


def target_noise_metadata(args, scene, frame_indices):
    """Return ordered per-row provenance for a canonical isolated-target cache."""
    protocol = canonical_protocol_metadata(args)
    samples = [
        canonical_noise_sample(
            protocol=protocol,
            vfm=args.vfm,
            model_id=args.model_id,
            scene=scene,
            sample_kind="target_isolated",
            frame_index=int(frame_index),
        )
        for frame_index in frame_indices
    ]
    return canonical_noise_metadata(samples)


def canonical_protocol_metadata(args):
    """Return the stable temporal protocol identity stored with every cache.

    The named v2 protocol is the production 28-observation layout.  Keep a
    deterministic hash as well, so intentional non-default/debug variants
    cannot be confused with that production layout.
    """
    prefixes = [int(v) for v in (args.prefix_lengths or [])]
    target_prefixes = [int(v) for v in (args.target_prefix_lengths or [])]
    horizons = [int(v) for v in (args.target_horizons or [])]
    obs_len = int(args.streaming_window_obs_len)
    payload = {
        "schema_version": 2,
        "protocol": CANONICAL_STREAMING_PROTOCOL,
        "sampling": str(args.temporal_sampling),
        "canonical_obs_len": obs_len,
        "prefixes": prefixes,
        "target_prefixes": target_prefixes,
        "horizons": horizons,
        "include_anchors": True,
        "motion_step": float(args.streaming_motion_step),
        "rotation_weight": float(args.streaming_rotation_weight),
        "window_stride": int(args.streaming_window_stride),
        "max_windows_per_scene": int(args.streaming_max_windows_per_scene),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["protocol_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def validate_canonical_window_args(args):
    """Reject temporal settings that cannot share one canonical window."""
    prefixes = [int(v) for v in (args.prefix_lengths or [])]
    target_prefixes = [int(v) for v in (args.target_prefix_lengths or [])]
    horizons = [int(v) for v in (args.target_horizons or [])]
    if any(v <= 0 for v in prefixes + target_prefixes):
        raise ValueError("Streaming prefix lengths must all be positive")
    if any(v <= 0 for v in horizons):
        raise ValueError("Target horizons must all be positive")

    required = max(
        prefixes
        + target_prefixes
        + [prefix + horizon for prefix in target_prefixes for horizon in horizons]
        + [1]
    )
    obs_len = int(args.streaming_window_obs_len)
    if obs_len < required:
        raise ValueError(
            "--streaming-window-obs-len must cover every streaming prefix, "
            f"anchor, and target (need >= {required}, got {obs_len})"
        )
    return required


def canonical_windows_for_scene(scene, args):
    """Build the one window layout shared by streaming inputs and targets."""
    observations_per_window = int(args.streaming_window_obs_len)
    num_frames = len(scene["img_files"])
    if num_frames < observations_per_window:
        return []
    if (
        args.temporal_sampling == "motion_uniform"
        and scene.get("source") == "scannetpp"
        and "poses_c2w" in scene
    ):
        return temporal_windows_from_poses(
            scene["poses_c2w"],
            observations_per_window=observations_per_window,
            motion_step=args.streaming_motion_step,
            rotation_weight=args.streaming_rotation_weight,
            window_stride=args.streaming_window_stride,
            max_windows_per_scene=args.streaming_max_windows_per_scene,
        )
    return [{
        "window_id": 0,
        "obs_start": 0,
        "indices": list(range(observations_per_window)),
        "sampling": "contiguous",
        "motion_step": 0.0,
        "rotation_weight": 0.0,
    }]


def videomae_forward(model, processor, frames, device, layer_ids):
    """Extract VideoMAE hidden states as (T_tubelet, H, W, C) feature grids."""
    model_num_frames = int(getattr(model.config, "num_frames", 16))
    if len(frames) != model_num_frames:
        sample_idx = np.round(
            np.linspace(0, len(frames) - 1, model_num_frames)
        ).astype(int).tolist()
        frames = [frames[i] for i in sample_idx]
    video_np = np.stack([np.array(frame) for frame in frames])
    inputs = processor(video_np, return_tensors="pt")
    model_dtype = next(model.parameters()).dtype
    inputs = {
        key: (
            value.to(device=device, dtype=model_dtype)
            if torch.is_floating_point(value)
            else value.to(device)
        )
        for key, value in inputs.items()
    }
    outputs = model(**inputs, output_hidden_states=True)

    cfg = model.config
    patch_size = cfg.patch_size if isinstance(cfg.patch_size, int) else cfg.patch_size[0]
    tubelet_size = int(getattr(cfg, "tubelet_size", 2))
    image_size = getattr(cfg, "image_size", frames[0].size[0])
    if isinstance(image_size, (tuple, list)):
        image_size = image_size[0]
    grid_h = int(image_size) // int(patch_size)
    grid_w = int(image_size) // int(patch_size)
    grid_t = len(frames) // tubelet_size
    expected_tokens = grid_t * grid_h * grid_w

    result = {}
    for layer_id in layer_ids:
        tokens = outputs.hidden_states[int(layer_id) + 1][0]
        if tokens.shape[0] == expected_tokens + 1:
            tokens = tokens[1:]
        if tokens.shape[0] != expected_tokens:
            raise ValueError(
                f"VideoMAE layer {layer_id} produced {tokens.shape[0]} tokens; "
                f"expected {expected_tokens} for ({grid_t}, {grid_h}, {grid_w})"
            )
        result[int(layer_id)] = tokens.reshape(grid_t, grid_h, grid_w, -1).contiguous()
    return result


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
    if (
        args.mode == "streaming_prefix"
        and args.temporal_sampling in {"motion_uniform", "none"}
    ):
        windows = canonical_windows_for_scene(scene, args)
        protocol = canonical_protocol_metadata(args)
        records = []
        for window in windows:
            indices = list(window["indices"])
            lengths = args.prefix_lengths or list(range(
                args.prefix_min_len,
                min(int(args.prefix_max_len), len(indices)) + 1,
                args.prefix_stride,
            ))
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
                    "canonical_protocol": protocol["protocol"],
                    "canonical_protocol_sha256": protocol["protocol_sha256"],
                    "canonical_obs_len": int(args.streaming_window_obs_len),
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


def target_selection_for_scene(scene, args):
    """Return the flat target union plus sample-level canonical provenance."""
    prefixes = [int(v) for v in (args.target_prefix_lengths or [8, 12, 16, 24])]
    horizons = [int(v) for v in (args.target_horizons or [1, 2, 4])]
    windows = canonical_windows_for_scene(scene, args)
    protocol = canonical_protocol_metadata(args)
    target_ids = set()
    provenance = []
    for window in windows:
        indices = [int(i) for i in window["indices"]]
        for prefix_len in prefixes:
            anchor = prefix_len - 1
            positions = [("anchor", 0, anchor)] + [
                ("future", horizon, anchor + horizon) for horizon in horizons
            ]
            for role, horizon, pos in positions:
                if not 0 <= pos < len(indices):
                    continue
                frame_index = int(indices[pos])
                target_ids.add(frame_index)
                provenance.append({
                    "window_id": int(window["window_id"]),
                    "obs_start": int(window.get("obs_start", 0)),
                    "prefix_length": int(prefix_len),
                    "role": role,
                    "horizon": int(horizon),
                    "position": int(pos),
                    "frame_index": frame_index,
                    "canonical_protocol": protocol["protocol"],
                    "canonical_obs_len": int(args.streaming_window_obs_len),
                })
    return sorted(target_ids), provenance


def target_indices_for_scene(scene, args):
    """Return exact local frame ids for target_isolated extraction.

    In streaming probes, the input prefix comes from a canonical
    motion-normalized ScanNet++ window.  The flat cache includes each prefix's
    anchor at ``p-1`` and future targets at ``p-1+h`` inside that same window.
    It stores scene-local indices so InsScene15KDataset can exact-match
    target_indices.npy.
    """
    if getattr(args, "target_from_streaming_windows", False):
        target_ids, _ = target_selection_for_scene(scene, args)
        return target_ids

    if args.num_targets == 0:
        return list(range(len(scene["img_files"])))
    count = max(2, min(int(args.num_targets), len(scene["img_files"])))
    return np.linspace(0, len(scene["img_files"]) - 1, count).round().astype(int).tolist()


def collect_cogvideox_target_rows_cpu(feats, per_layer_collect):
    """Retain only the selected target row, backed by independent CPU storage.

    CogVideoX target extraction performs one large two-clip forward per target
    frame.  Keeping a view of that output on CUDA until the whole scene is
    saved makes memory grow once per target and eventually OOMs an H800.  The
    target cache itself is written on CPU, so materialize the selected row
    immediately and release the full forward output before the next target.
    """
    for layer_id, feat in feats.items():
        # feat: (2, T_clip, H_f, W_f, C).  Flatten the two temporal clips and
        # select the same middle row as the historical implementation.
        flattened = feat.reshape(-1, *feat.shape[2:])
        row = flattened[flattened.shape[0] // 2]
        per_layer_collect[layer_id].append(
            row.detach().to(device="cpu", copy=True).contiguous()
        )


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


def is_cogvideox_i2v(model_id):
    """Recognize both Hub I2V ids and their local snapshot paths."""
    if not model_id:
        return False
    text = str(model_id)
    return text.endswith("I2V") or "CogVideoX-5b-I2V" in text


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
                        choices=["wan", "cogvideox", "vjepa", "vjepa2", "dino", "videomae", "aether", "opensora", "f3r", "minimax_h3", "lingbot"],
                        help="Which VFM to extract features from")
    parser.add_argument("--model-id", default=None,
                        help="Model ID (default per VFM)")
    parser.add_argument("--prompt", default="")
    parser.add_argument(
        "--t",
        type=int,
        default=None,
        help="Diffusion/flow timestep code. Wan/CogVideoX: scheduler index (default 749). "
             "MiniMax-H3: continuous t*1000 (default 250 ⇒ t=0.250). "
             "LingBot: flow σ*1000 (default 250 ⇒ σ=0.250).",
    )
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
    parser.add_argument(
        "--canonical-gap-worklist",
        default=None,
        help=(
            "Immutable CANONICAL_V2_GAP_WORKLIST_V1 positive allow-list. "
            "Only listed scene/window/prefix/layer or scene/frame/layer items "
            "may be generated."
        ),
    )
    parser.add_argument(
        "--canonical-gap-worklist-sha256",
        default=None,
        help="Required file SHA-256 binding for --canonical-gap-worklist.",
    )
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
    parser.add_argument(
        "--streaming-window-obs-len",
        type=int,
        default=CANONICAL_STREAMING_OBS_LEN,
        help=(
            "Canonical observations per motion-uniform window, shared by every "
            "streaming prefix and isolated target selection. Production v2 uses "
            "28 = max prefix 24 + max horizon 4."
        ),
    )
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
    try:
        validate_canonical_window_args(args)
    except ValueError as exc:
        parser.error(str(exc))
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
        "wan":       {"model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "num_frames": 81, "size": (480, 832), "t": 749},
        "cogvideox": {"model_id": "THUDM/CogVideoX-5b-I2V",          "num_frames": 97, "size": (480, 720), "t": 749},
        "vjepa":     {"model_id": None,                                "num_frames": 76, "size": (480, 832), "t": 0},
        "vjepa2":    {"model_id": "facebook/vjepa2-vitl-fpc64-256",   "num_frames": 64, "size": (256, 256), "t": 0},
        "dino":      {"model_id": "facebook/dinov2-large",             "num_frames": 64, "size": (420, 728), "t": 0},
        "videomae":  {"model_id": "MCG-NJU/videomae-large",            "num_frames": 16, "size": (224, 224), "t": 0},
        "aether":    {"model_id": None,                                "num_frames": 81, "size": (480, 720), "t": 749},
        "opensora":  {"model_id": None,                                "num_frames": 81, "size": (480, 720), "t": 1},
        "f3r":       {"model_id": None,                                "num_frames": 64, "size": (288, 512), "t": 0},
        "minimax_h3": {
            "model_id": "/mnt/umm/users/yinbaiqiao/Consistency/minimax-h3/models/MiniMax-H3",
            "num_frames": 73,
            "size": (480, 832),
            "t": 250,
        },
        "lingbot": {
            "model_id": "/mnt/umm/users/yinbaiqiao/Consistency/lingbot-video/models/lingbot-video-moe-30b-a3b",
            "num_frames": 49,
            "size": (480, 832),
            "t": 250,
        },
    }
    defaults = VFM_DEFAULTS[args.vfm]
    if args.model_id is None:
        args.model_id = defaults["model_id"]
    if args.num_frames is None:
        args.num_frames = defaults["num_frames"]
    if args.t is None:
        args.t = defaults["t"]
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
        if args.model_id and not is_cogvideox_i2v(args.model_id):
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
    elif args.vfm == "videomae":
        def out_fname(layer):
            return f"feature_layer{layer}.sft"
    elif args.vfm in ("aether", "opensora", "minimax_h3", "lingbot"):
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

    feature_namespace = os.environ.get("FEATURE_NAMESPACE", "").strip()
    canonical_feature_mode = is_canonical_feature_mode(args)
    if canonical_feature_mode and not feature_namespace:
        raise SystemExit(
            "canonical-v2 extraction requires explicit FEATURE_NAMESPACE provenance"
        )
    if (
        canonical_feature_mode
        and Path(args.out_root).resolve().name != feature_namespace
    ):
        raise SystemExit(
            "canonical-v2 --out-root basename must exactly match FEATURE_NAMESPACE "
            f"({Path(args.out_root).resolve().name!r} != {feature_namespace!r})"
        )
    if (
        canonical_feature_mode
        and args.vfm in SEEDED_SUPPORTED_VFMS
        and not is_seeded_feature_namespace(feature_namespace)
        and not (
            args.canonical_gap_worklist
            and feature_namespace in CANONICAL_GAP_NAMESPACES
        )
    ):
        raise SystemExit(
            "canonical seeded backend extraction requires an isolated "
            "seeded-v1 FEATURE_NAMESPACE"
        )
    try:
        validate_seeded_cache_policy(args, feature_namespace)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.canonical_gap_worklist and args.no_cache_checksum:
        raise SystemExit("canonical gap extraction forbids --no-cache-checksum")

    if bool(args.canonical_gap_worklist) != bool(args.canonical_gap_worklist_sha256):
        raise SystemExit(
            "--canonical-gap-worklist and --canonical-gap-worklist-sha256 "
            "must be supplied together"
        )
    gap_worklist = None
    if args.canonical_gap_worklist:
        if not canonical_feature_mode:
            raise SystemExit("canonical gap worklists are valid only for canonical feature modes")
        from features.canonical_gap_worklist import (
            CanonicalGapWorklist,
            GapWorklistError,
            static_prefix_index_metadata,
            streaming_worklist_preprocess_identity,
            stream_unit_producer_metadata,
            stream_unit_semantic_matches,
            stream_unit_semantic_metadata,
        )

        protocol = canonical_protocol_metadata(args)
        try:
            gap_worklist = CanonicalGapWorklist.load(
                args.canonical_gap_worklist,
                expected_sha256=args.canonical_gap_worklist_sha256,
                mode=args.mode,
                vfm=args.vfm,
                model_id=str(args.model_id),
                timestep=int(args.t),
                feature_namespace=feature_namespace,
                output_layers=args.output_layers,
                protocol_sha256=protocol["protocol_sha256"],
                actual_prompt=str(args.prompt),
                preprocess_identity=(
                    streaming_worklist_preprocess_identity(
                        args.vfm, args.resize, int(args.num_frames)
                    )
                    if args.mode == "streaming_prefix"
                    else {
                        "resize": [int(value) for value in args.resize],
                        "num_frames": int(args.num_frames),
                    }
                ),
            )
        except GapWorklistError as exc:
            raise SystemExit(f"invalid canonical gap worklist: {exc}") from exc
        scenes = [s for s in scenes if scene_name(s) in gap_worklist.scenes]
        log.info(
            "Bound gap worklist %s (%d records, %d layer-items, %d selected scenes)",
            gap_worklist.file_sha256,
            gap_worklist.entry_count,
            gap_worklist.layer_item_count,
            len(scenes),
        )

    def requested_stream_layers(scene, record):
        if gap_worklist is None:
            return [0] if args.vfm == "vjepa" else list(args.output_layers)
        return sorted(
            gap_worklist.stream_layers(
                scene,
                int(record["window_id"]),
                int(record["valid_length"]),
            )
        )

    def requested_target_indices_by_layer(scene):
        if gap_worklist is None:
            return {}
        by_layer = {int(layer): [] for layer in args.output_layers}
        for frame_index in gap_worklist.target_frame_indices(scene):
            for layer in gap_worklist.target_layers(scene, frame_index):
                by_layer[int(layer)].append(int(frame_index))
        return {layer: values for layer, values in by_layer.items() if values}

    def target_delta_dir(out_dir):
        if gap_worklist is None:
            raise RuntimeError("target_delta_dir requires a bound gap worklist")
        return (
            Path(out_dir)
            / ".canonical_v2_target_delta"
            / gap_worklist.file_sha256[:16]
        )

    def target_delta_feature_path(out_dir, layer):
        return target_delta_dir(out_dir) / f"feature_t{int(args.t)}_layer{int(layer)}.delta.sft"

    def target_delta_index_path(out_dir, layer):
        return target_delta_dir(out_dir) / f"target_indices_layer{int(layer)}.npy"

    feature_cache_root = (
        Path(args.out_root).resolve() / vfm_dir_name / args.source
    )

    done_index_path = os.environ.get("EXTRACT_DONE_INDEX", "").strip()
    done_index_entries = set()
    done_index_mtime_ns = None
    if done_index_path and args.source == "all":
        raise SystemExit("EXTRACT_DONE_INDEX requires one concrete --source")
    done_index_root = feature_cache_root if done_index_path else None

    def refresh_done_index(force=False):
        """Reload the local AOSS-key snapshot when the sync loop replaces it."""
        nonlocal done_index_entries, done_index_mtime_ns
        if not done_index_path:
            return
        index_path = Path(done_index_path)
        try:
            stat = index_path.stat()
        except FileNotFoundError:
            if force:
                raise SystemExit(f"EXTRACT_DONE_INDEX does not exist: {index_path}")
            return
        if not force and done_index_mtime_ns == stat.st_mtime_ns:
            return
        entries = set()
        for raw_line in index_path.read_text().splitlines():
            rel = raw_line.strip().replace("\\", "/").lstrip("./")
            if not rel or rel.startswith("/") or ".." in Path(rel).parts:
                continue
            if rel.endswith(".sft"):
                entries.add(rel)
        old_count = len(done_index_entries)
        done_index_entries = entries
        done_index_mtime_ns = stat.st_mtime_ns
        if force or len(entries) != old_count:
            log.info(f"Loaded EXTRACT_DONE_INDEX: {len(entries)} keys from {index_path}")

    def done_index_contains(path):
        if not done_index_path:
            return False
        refresh_done_index()
        try:
            rel = Path(path).resolve().relative_to(done_index_root).as_posix()
        except (OSError, ValueError):
            return False
        return rel in done_index_entries

    def expected_noise_metadata_for_path(path):
        """Reconstruct canonical noise provenance from the cache identity."""
        if not canonical_feature_mode:
            return None
        static_contract = backend_generation_contract(args)
        if args.vfm == "minimax_h3" or args.vfm in DETERMINISTIC_NO_NOISE_VFMS:
            return static_contract
        if args.vfm not in SAMPLE_LOCAL_NOISE_VFMS:
            return False
        try:
            rel = Path(path).resolve().relative_to(feature_cache_root)
            parts = rel.parts
            scene = parts[0]
            if args.mode == "streaming_prefix":
                if (
                    len(parts) != 4
                    or not parts[1].startswith("window_")
                    or not parts[2].startswith("prefix_")
                ):
                    return False
                window_id = int(parts[1].removeprefix("window_"))
                prefix_len = int(parts[2].removeprefix("prefix_")) + 1
                record = {
                    "window_id": window_id,
                    "valid_length": prefix_len,
                }
                return streaming_noise_metadata(args, scene, record)
            if args.mode == "target_isolated":
                if len(parts) != 2:
                    return False
                target_indices = np.load(
                    Path(path).parent / "target_indices.npy"
                ).astype(np.int64).tolist()
                return target_noise_metadata(args, scene, target_indices)
        except (OSError, ValueError, IndexError, KeyError):
            return False
        return False

    local_feature_digest_cache = {}

    def stable_local_feature_sha256(path):
        return stable_file_sha256(path, cache=local_feature_digest_cache)

    def local_cache_complete(path):
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            return False
        sidecar = Path(f"{path}.manifest.json")
        if sidecar.is_symlink() or not sidecar.is_file():
            return args.allow_legacy_cache and not canonical_feature_mode
        try:
            meta = json.loads(sidecar.read_text())
            complete = (
                isinstance(meta, dict)
                and meta.get("schema_version") == 1
                and meta.get("frame_index_schema") == "image_mask_intersection_v1"
                and meta.get("vfm") == args.vfm
                and meta.get("mode") == args.mode
                and meta.get("model_id") == args.model_id
                and meta.get("timestep") == int(args.t)
                and meta.get("file") == path.name
                and path.stat().st_size == meta.get("size_bytes")
            )
            if args.mode == "streaming_prefix" or (
                args.mode == "target_isolated" and args.target_from_streaming_windows
            ):
                protocol = canonical_protocol_metadata(args)
                complete = complete and (
                    meta.get("feature_namespace") == feature_namespace
                    and meta.get("canonical_protocol") == protocol["protocol"]
                    and meta.get("canonical_protocol_sha256") == protocol["protocol_sha256"]
                    and meta.get("canonical_obs_len") == int(args.streaming_window_obs_len)
                )
                expected_noise = expected_noise_metadata_for_path(path)
                if expected_noise is False:
                    return False
                if expected_noise is not None:
                    complete = complete and noise_metadata_matches(
                        meta, expected_noise
                    )
                if is_seeded_feature_namespace(feature_namespace):
                    actual_sha256 = stable_local_feature_sha256(path)
                    complete = complete and seeded_feature_metadata_matches(
                        path,
                        meta,
                        vfm=args.vfm,
                        model_id=args.model_id,
                        mode=args.mode,
                        feature_namespace=feature_namespace,
                        timestep=args.t,
                        prompt=args.prompt,
                        resize=args.resize,
                        num_frames=args.num_frames,
                        expected_noise_metadata=expected_noise,
                        actual_sha256=actual_sha256,
                    )
            return complete
        except Exception:
            return False

    def cache_complete(path):
        return local_cache_complete(path) or done_index_contains(path)

    def stream_gap_metadata(scene, record, layer, _requested_layers):
        metadata = {
            **stream_unit_semantic_metadata(
                gap_worklist, scene, record, layer
            ),
            **stream_unit_producer_metadata(gap_worklist),
        }
        # These fields are already written by ``save_file``.  Keep them in the
        # stable semantic expectation, but do not attempt to replace reserved
        # manifest keys while publishing.
        for key in ("frame_index_schema", "resize", "num_frames"):
            metadata.pop(key)
        return metadata

    def stream_gap_complete(path, scene, record, layer, requested_layers):
        if gap_worklist is None:
            return cache_complete(path)
        path = Path(path)
        sidecar = Path(f"{path}.manifest.json")
        if path.is_symlink() or sidecar.is_symlink() or not path.is_file() or not sidecar.is_file():
            return False
        try:
            metadata = json.loads(sidecar.read_text())
            expected = stream_unit_semantic_metadata(
                gap_worklist, scene, record, layer
            )
            digest = stable_local_feature_sha256(path)
            return (
                digest is not None
                and local_cache_complete(path)
                and metadata.get("sha256") == digest
                and metadata.get("size_bytes") == path.stat().st_size
                and stream_unit_semantic_matches(metadata, expected)
                and metadata.get("vfm") == args.vfm
                and metadata.get("model_id") == args.model_id
                and metadata.get("timestep") == int(args.t)
                and metadata.get("prompt") == args.prompt
                and metadata.get("resize") == list(args.resize)
                and metadata.get("num_frames") == int(args.num_frames)
            )
        except Exception:
            return False

    def target_delta_complete(out_dir, layer, expected_indices):
        """Validate one immutable per-layer target delta for worklist resume."""
        if gap_worklist is None:
            return False
        feature_path = target_delta_feature_path(out_dir, layer)
        index_path = target_delta_index_path(out_dir, layer)
        feature_sidecar = Path(f"{feature_path}.manifest.json")
        index_sidecar = Path(f"{index_path}.manifest.json")
        paths = (feature_path, index_path, feature_sidecar, index_sidecar)
        if any(path.is_symlink() or not path.is_file() for path in paths):
            return False
        try:
            actual_indices = np.load(index_path).astype(np.int64).tolist()
            if actual_indices != [int(value) for value in expected_indices]:
                return False
            feature_meta = json.loads(feature_sidecar.read_text())
            index_meta = json.loads(index_sidecar.read_text())
            feature_sha = stable_local_feature_sha256(feature_path)
            index_sha = stable_file_sha256(index_path)
            shared = {
                "gap_worklist_file_sha256": gap_worklist.file_sha256,
                "gap_worklist_content_sha256": gap_worklist.content_sha256,
                "canonical_source_sha256": gap_worklist.canonical_source_sha256,
                "vfm": args.vfm,
                "model_id": args.model_id,
                "timestep": int(args.t),
                "layer": int(layer),
                "prompt": str(args.prompt),
                "resize": list(args.resize),
                "num_frames": int(args.num_frames),
                "target_indices_sha256": index_sha,
            }
            expected_generation = (
                target_noise_metadata(args, str(Path(out_dir).name), expected_indices)
                if requires_sample_local_noise(args)
                else backend_generation_contract(args)
            )
            return (
                all(feature_meta.get(key) == value for key, value in shared.items())
                and all(index_meta.get(key) == value for key, value in shared.items())
                and feature_meta.get("sha256") == feature_sha
                and feature_meta.get("size_bytes") == feature_path.stat().st_size
                and index_meta.get("sha256") == index_sha
                and index_meta.get("size_bytes") == index_path.stat().st_size
                and feature_meta.get("tensors", {}).get("feat", {}).get("shape", [None])[0]
                == len(actual_indices)
                and isinstance(expected_generation, dict)
                and noise_metadata_matches(feature_meta, expected_generation)
            )
        except Exception:
            return False

    refresh_done_index(force=bool(done_index_path))

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
        if cached != [int(v) for v in expected_indices]:
            return False
        if not args.target_from_streaming_windows:
            return True
        meta_path = os.path.join(out_dir, "target_indices.meta.json")
        try:
            meta = json.loads(Path(meta_path).read_text())
            index_sidecar = json.loads(Path(f"{path}.manifest.json").read_text())
            protocol = canonical_protocol_metadata(args)
            digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            expected_noise_contract = backend_generation_contract(args)
            if expected_noise_contract is None:
                return False
            return (
                meta.get("feature_namespace") == feature_namespace
                and meta.get("protocol") == protocol["protocol"]
                and meta.get("protocol_sha256") == protocol["protocol_sha256"]
                and meta.get("obs_len") == int(args.streaming_window_obs_len)
                and meta.get("prefixes") == [int(v) for v in args.target_prefix_lengths]
                and meta.get("horizons") == [int(v) for v in args.target_horizons]
                and meta.get("include_anchors") is True
                and meta.get("target_indices_sha256") == digest
                and noise_metadata_matches(meta, expected_noise_contract)
                and index_sidecar.get("feature_namespace") == feature_namespace
                and index_sidecar.get("canonical_protocol") == protocol["protocol"]
                and index_sidecar.get("canonical_protocol_sha256")
                == protocol["protocol_sha256"]
                and noise_metadata_matches(index_sidecar, expected_noise_contract)
            )
        except Exception:
            return False

    def target_cache_complete(path, target_indices_ok):
        # AOSS completion is defined by the feature key itself. Local target
        # caches still require the matching target_indices.npy provenance.
        return done_index_contains(path) or (
            target_indices_ok and local_cache_complete(path)
        )

    def _structural_prefix_record(record):
        """Normalize only fields that identify the canonical raw-frame window."""
        return {
            "tail": int(record["tail"]),
            "indices": [int(value) for value in record["indices"]],
            "window_indices": [int(value) for value in record["window_indices"]],
            "window_id": int(record["window_id"]),
            "obs_start": int(record["obs_start"]),
            "valid_length": int(record["valid_length"]),
            "sampling": str(record["sampling"]),
            "motion_step": float(record["motion_step"]),
            "rotation_weight": float(record["rotation_weight"]),
            "canonical_protocol": str(record["canonical_protocol"]),
            "canonical_protocol_sha256": str(record["canonical_protocol_sha256"]),
            "canonical_obs_len": int(record["canonical_obs_len"]),
        }

    def prefix_index_complete(out_dir, expected_records):
        """Require exact raw-ID records and byte-bound seeded-v1 provenance."""
        if not canonical_feature_mode or args.mode != "streaming_prefix":
            return True
        path = Path(out_dir) / "prefix_index.npy"
        sidecar_path = Path(f"{path}.manifest.json")
        if (
            path.is_symlink()
            or sidecar_path.is_symlink()
            or not path.is_file()
            or not sidecar_path.is_file()
        ):
            return False
        try:
            cached_records = np.load(path, allow_pickle=True).tolist()
            if not isinstance(cached_records, list):
                return False
            cached = [_structural_prefix_record(record) for record in cached_records]
            expected = [_structural_prefix_record(record) for record in expected_records]
            sidecar = json.loads(sidecar_path.read_text())
            digest = stable_file_sha256(path)
            protocol = canonical_protocol_metadata(args)
            generation_contract = backend_generation_contract(args)
            return (
                cached == expected
                and isinstance(digest, str)
                and sidecar.get("schema_version") == 1
                and sidecar.get("file") == path.name
                and sidecar.get("feature_namespace") == feature_namespace
                and sidecar.get("size_bytes") == path.stat().st_size
                and sidecar.get("sha256") == digest
                and sidecar.get("shape") == [len(expected)]
                and sidecar.get("dtype") == "object"
                and sidecar.get("canonical_protocol") == protocol["protocol"]
                and sidecar.get("canonical_protocol_sha256") == protocol["protocol_sha256"]
                and sidecar.get("canonical_obs_len") == int(args.streaming_window_obs_len)
                and isinstance(generation_contract, dict)
                and noise_metadata_matches(sidecar, generation_contract)
            )
        except Exception:
            return False

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
            # In gap mode the structural index is materialized/audited
            # separately from the reference overlay.  Completion here concerns
            # only positively allow-listed tensor records.
            all_exist = True if gap_worklist is not None else prefix_index_complete(out_dir, records)
            for record in records:
                requested_layers = requested_stream_layers(name, record)
                if not requested_layers:
                    continue
                prefix_out_dir = os.path.join(
                    out_dir,
                    prefix_dir_name(record) if args.mode == "streaming_prefix" else context_dir_name(record),
                )
                prefix_exists = all(
                    stream_gap_complete(
                        os.path.join(prefix_out_dir, out_fname(l)),
                        name,
                        record,
                        l,
                        requested_layers,
                    )
                    for l in requested_layers
                )
                if not prefix_exists:
                    all_exist = False
                    break
        elif args.mode == "target_isolated":
            expected_targets = target_indices_for_scene(s, args)
            if gap_worklist is not None:
                requested = requested_target_indices_by_layer(name)
                invalid = sorted(
                    set().union(*(set(values) for values in requested.values()), set())
                    - set(expected_targets)
                )
                if invalid:
                    raise RuntimeError(
                        f"{name}: target gap worklist escapes canonical target union: {invalid[:8]}"
                    )
                all_exist = bool(requested) and all(
                    target_delta_complete(out_dir, layer, indices)
                    for layer, indices in requested.items()
                )
            else:
                target_ok = target_index_complete(out_dir, expected_targets)
                all_exist = (
                    target_cache_complete(os.path.join(out_dir, out_fname(0)), target_ok)
                    if args.vfm == "vjepa"
                    else all(
                        target_cache_complete(os.path.join(out_dir, out_fname(l)), target_ok)
                        for l in args.output_layers
                    )
                )
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
    if os.environ.get("EXTRACT_CACHE_CHECK_ONLY") == "1":
        log.info("EXTRACT_CACHE_CHECK_ONLY=1; exiting before model load")
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

    # Keep ``args.model_id`` as the canonical logical identifier used by the
    # worklist and every output sidecar.  Only the loading call below receives
    # the revision-pinned physical snapshot path when this is an offline Hub
    # model.  This prevents a Diffusers/Transformers ``model_info`` call while
    # preserving a stable cross-run cache identity.
    model_load_id, model_load_provenance = resolve_model_load_id(
        args.vfm, args.model_id
    )
    if model_load_id != args.model_id:
        log.info(
            "Resolved offline model load path for %s: logical=%s revision=%s",
            args.vfm,
            args.model_id,
            model_load_provenance.get("resolved_revision"),
        )

    def save_file(tensors, out_path, noise_metadata=None, extra_metadata=None):
        """Atomically publish a cache and its provenance sidecar."""
        if canonical_feature_mode and noise_metadata is None:
            static_contract = backend_generation_contract(args)
            if args.vfm == "minimax_h3" or args.vfm in DETERMINISTIC_NO_NOISE_VFMS:
                noise_metadata = static_contract
        if (
            canonical_feature_mode
            and args.vfm in SEEDED_SUPPORTED_VFMS
            and noise_metadata is None
        ):
            raise RuntimeError(
                f"canonical {args.vfm} cache is missing deterministic noise provenance"
            )
        if args.vfm == "minimax_h3":
            for key, value in tensors.items():
                if value.dtype != torch.bfloat16:
                    raise TypeError(
                        f"MiniMax storage requires bfloat16, got {value.dtype} "
                        f"for tensor {key!r} at {out_path}"
                    )
                validate_finite_feature(
                    value,
                    context=f"minimax_publish:{out_path}:{key}",
                )
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
            "feature_namespace": feature_namespace or None,
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
        if args.vfm == "minimax_h3":
            manifest.update({
                "feature_storage_contract": MINIMAX_STORAGE_CONTRACT,
                "feature_storage_dtype": "torch.bfloat16",
                "finite_check": "pre_conversion_post_conversion_pre_publish",
            })
        if model_load_provenance is not None:
            manifest["model_load_provenance"] = model_load_provenance
        if args.mode == "streaming_prefix" or (
            args.mode == "target_isolated" and args.target_from_streaming_windows
        ):
            protocol = canonical_protocol_metadata(args)
            manifest.update({
                "canonical_protocol": protocol["protocol"],
                "canonical_protocol_sha256": protocol["protocol_sha256"],
                "canonical_obs_len": int(args.streaming_window_obs_len),
            })
            if args.vfm in SEEDED_SUPPORTED_VFMS:
                manifest.update(noise_metadata)
        merge_extra_metadata(manifest, extra_metadata)
        sidecar = Path(f"{out_path}.manifest.json")
        sidecar_tmp = sidecar.with_name(f".{sidecar.name}.tmp-{os.getpid()}")
        sidecar_tmp.write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(sidecar_tmp, sidecar)

    def atomic_write_json(path, value):
        path = Path(path)
        tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, path)

    def atomic_save_npy(path, array, provenance=None):
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
            "feature_namespace": feature_namespace or None,
            "size_bytes": path.stat().st_size,
            "sha256": digest,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
        if provenance:
            meta.update(provenance)
        sidecar = Path(f"{path}.manifest.json")
        atomic_write_json(sidecar, meta)

    def write_target_index_metadata(path, target_indices, target_provenance):
        protocol = canonical_protocol_metadata(args)
        metadata = {
            "schema_version": 2,
            "feature_namespace": feature_namespace,
            "protocol": protocol["protocol"],
            "protocol_sha256": protocol["protocol_sha256"],
            "obs_len": int(args.streaming_window_obs_len),
            "prefixes": [int(v) for v in args.target_prefix_lengths],
            "horizons": [int(v) for v in args.target_horizons],
            "include_anchors": True,
            "target_count": int(len(target_indices)),
            "target_indices_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "provenance": target_provenance,
        }
        generation_contract = backend_generation_contract(args)
        if generation_contract is not None:
            metadata.update(generation_contract)
        atomic_write_json(Path(path).with_name("target_indices.meta.json"), metadata)

    if args.vfm == "wan":
        from features.wan.wan_feature import get_wan_featurizer
        from features.wan.extract_features import reshape_to_t_h_w_c
        log.info(f"Loading WAN model: {model_load_id}")
        model = get_wan_featurizer(model_id=model_load_id, null_prompt=args.prompt)
    elif args.vfm == "cogvideox":
        from features.cogvideox.cogvideox_feature import get_cogvideox_featurizer
        from features.cogvideox.cogvideox_feature_i2v import get_cogvideox_featurizer_i2v
        from features.cogvideox.extract_features import forward_cogvideox
        log.info(f"Loading CogVideoX model: {model_load_id}")
        if is_cogvideox_i2v(args.model_id):
            model = get_cogvideox_featurizer_i2v(model_id=model_load_id)
        else:
            model = get_cogvideox_featurizer(model_id=model_load_id)
    elif args.vfm == "vjepa":
        from features.vjepa.extract_features import VJEPAFeaturizer_Spaced, VJEPAFeaturizer_Chunked
        log.info("Loading V-JEPA model...")
        if args.partition == "spaced":
            model = VJEPAFeaturizer_Spaced(args.checkpoint)
        else:
            model = VJEPAFeaturizer_Chunked(args.checkpoint)
    elif args.vfm == "vjepa2":
        from features.vjepa2.vjepa2_feature import get_vjepa2_featurizer
        log.info(f"Loading V-JEPA 2 model: {model_load_id}")
        model = get_vjepa2_featurizer(model_id=model_load_id)
    elif args.vfm == "dino":
        from features.dino.extract_features import dino_forward as _dino_forward, reshape_tokens as _dino_reshape
        from transformers import AutoImageProcessor, Dinov2Model
        log.info(f"Loading DINOv2 model: {model_load_id}")
        _dino_proc = AutoImageProcessor.from_pretrained(model_load_id)
        _dino_model = Dinov2Model.from_pretrained(model_load_id).cuda().eval()
        model = (_dino_model, _dino_proc)  # stored as a tuple
    elif args.vfm == "videomae":
        from transformers import AutoImageProcessor, VideoMAEModel
        log.info(f"Loading VideoMAE model: {model_load_id}")
        _videomae_proc = AutoImageProcessor.from_pretrained(model_load_id)
        _videomae_model = VideoMAEModel.from_pretrained(
            model_load_id, torch_dtype=torch.float16
        ).cuda().eval()
        model = (_videomae_model, _videomae_proc)
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
    elif args.vfm == "minimax_h3":
        from features.minimax_h3.minimax_h3_feature import get_minimax_h3_featurizer
        from features.minimax_h3.extract_features import reshape_to_t_h_w_c as reshape_minimax_h3
        # Empty prompts yield zero text tokens and break MiniMax packed layout.
        if not str(args.prompt).strip():
            args.prompt = "."
        # The canonical unit retains its historical logical ``model_id`` in
        # feature provenance.  GAP workers may instead hydrate an immutable
        # AOSS model mirror into node-local cache; only this physical load path
        # is substituted.  Passing the cache path through ``--model-id`` would
        # corrupt the worklist/sidecar identity contract.
        minimax_load_id = os.environ.get("MINIMAX_H3_MODEL_LOAD_ID", args.model_id)
        if minimax_load_id != args.model_id:
            model_load_provenance = {
                "logical_model_id": args.model_id,
                "load_kind": "aoss_model_mirror_hydrated_local_cache",
                "runtime_model_id": minimax_load_id,
                "mirror_content_sha256": os.environ.get("MINIMAX_H3_MIRROR_SHA256"),
                "mirror_remote_base": os.environ.get("MINIMAX_H3_MIRROR_REMOTE_BASE"),
            }
        log.info(f"Loading MiniMax-H3 model: {minimax_load_id}")
        model = get_minimax_h3_featurizer(
            model_id=minimax_load_id,
            null_prompt=args.prompt,
            height=args.resize[0],
            width=args.resize[1],
            num_frames=args.num_frames,
        )
    elif args.vfm == "lingbot":
        from features.lingbot.lingbot_feature import get_lingbot_featurizer
        from features.lingbot.extract_features import reshape_to_t_h_w_c as reshape_lingbot
        if not str(args.prompt).strip():
            args.prompt = "."
        log.info(f"Loading LingBot-Video model: {args.model_id}")
        model = get_lingbot_featurizer(
            model_id=args.model_id,
            null_prompt=args.prompt,
            height=args.resize[0],
            width=args.resize[1],
            num_frames=args.num_frames,
        )

    log.info("Model loaded.")
    if (
        canonical_feature_mode
        and args.vfm == "minimax_h3"
        and int(getattr(model, "seed", -1)) != MINIMAX_FIXED_NOISE_SEED
    ):
        raise RuntimeError(
            "canonical MiniMax requires fixed per-forward seed "
            f"{MINIMAX_FIXED_NOISE_SEED}"
        )

    def generator_for_noise_sample(sample):
        """Create one model-device generator for one logical sample."""
        if args.vfm not in SAMPLE_LOCAL_NOISE_VFMS:
            return None
        if args.vfm in {"f3r", "aether"}:
            # Aether latents are bf16 CUDA.  A CUDA generator plus
            # in-place ``normal_``, or DPM ``randn_tensor(..., dtype=bf16)``,
            # can ignore the local generator and advance the process-global
            # CUDA RNG, which is exactly the cross-order canary failure.
            # CPU generators are consumed by diffusers ``randn_tensor`` on
            # CPU and then moved, and by ``randn_like_on_generator_device``.
            return make_noise_generator(int(sample["noise_seed"]), device="cpu")
        if args.vfm == "lingbot":
            return make_noise_generator(
                int(sample["noise_seed"]), device=getattr(model, "device", "cuda")
            )
        pipe = getattr(model, "pipe", None)
        device = getattr(pipe, "_execution_device", None)
        if device is None:
            transformer = getattr(pipe, "transformer", None)
            try:
                device = next(transformer.parameters()).device
            except (AttributeError, StopIteration):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return make_noise_generator(int(sample["noise_seed"]), device=device)

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
                requested_layers = requested_stream_layers(name, record)
                if not requested_layers:
                    continue
                prefix_out_dir = os.path.join(
                    out_dir,
                    prefix_dir_name(record) if args.mode == "streaming_prefix" else context_dir_name(record),
                )
                prefix_complete = all(
                    stream_gap_complete(
                        os.path.join(prefix_out_dir, out_fname(l)),
                        name,
                        record,
                        l,
                        requested_layers,
                    )
                    for l in requested_layers
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
            if gap_worklist is not None:
                requested_target_by_layer = requested_target_indices_by_layer(name)
                invalid = sorted(
                    set().union(
                        *(set(values) for values in requested_target_by_layer.values()),
                        set(),
                    )
                    - set(expected_targets)
                )
                if invalid:
                    raise RuntimeError(
                        f"{name}: target gap worklist escapes canonical target union: {invalid[:8]}"
                    )
                missing_layers = [
                    layer
                    for layer, indices in requested_target_by_layer.items()
                    if not target_delta_complete(out_dir, layer, indices)
                ]
            else:
                target_ok = target_index_complete(out_dir, expected_targets)
                if args.vfm == "vjepa":
                    missing_layers = [] if target_cache_complete(
                        os.path.join(out_dir, out_fname(0)), target_ok
                    ) else [0]
                else:
                    missing_layers = [
                        l for l in args.output_layers
                        if not target_cache_complete(
                            os.path.join(out_dir, out_fname(l)), target_ok
                        )
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
                    requested_layers = requested_stream_layers(name, record)
                    missing_layers = [
                        layer
                        for layer in requested_layers
                        if not stream_gap_complete(
                            os.path.join(prefix_out_dir, out_fname(layer)),
                            name,
                            record,
                            layer,
                            requested_layers,
                        )
                    ]
                    if not missing_layers:
                        skipped_prefixes += 1
                        meta_record = dict(record)
                        meta_record["input_length"] = args.num_frames
                        meta_record["pad_mode"] = "repeat_tail"
                        prefix_meta.append(meta_record)
                        continue

                    record_noise_metadata = None
                    record_noise_generator = None
                    if requires_sample_local_noise(args):
                        record_noise_metadata = streaming_noise_metadata(
                            args, name, record
                        )
                        record_noise_generator = generator_for_noise_sample(
                            record_noise_metadata["noise_seed_samples"][0]
                        )

                    frames_input = load_and_resize_frames(
                        s["img_dir"],
                        s["img_files"],
                        record["indices"],
                        size=args.resize,
                    )
                    valid_length = len(frames_input)
                    while len(frames_input) < args.num_frames:
                        frames_input.append(frames_input[-1])
                    if len(frames_input) > args.num_frames and args.vfm != "videomae":
                        raise ValueError(
                            f"{args.mode} length {len(frames_input)} exceeds "
                            f"model input length {args.num_frames}; lower the segment length"
                        )
                    os.makedirs(prefix_out_dir, exist_ok=True)
                    record_extra_metadata = lambda layer: (
                        stream_gap_metadata(
                            name, record, layer, requested_layers
                        )
                        if gap_worklist is not None
                        else None
                    )

                    if args.vfm == "wan":
                        with torch.no_grad():
                            feats = model.forward(
                                video=frames_input, prompt=args.prompt, t=args.t,
                                output_layer_indices=missing_layers,
                                ensemble_size=args.ensemble,
                                noise_generator=record_noise_generator,
                            )
                        for layer_id, raw_feat in feats.items():
                            reshaped = reshape_to_t_h_w_c(raw_feat)
                            out_path = os.path.join(prefix_out_dir, out_fname(layer_id))
                            save_file(
                                {"feat": reshaped.half()},
                                out_path,
                                noise_metadata=record_noise_metadata,
                                extra_metadata=record_extra_metadata(layer_id),
                            )

                    elif args.vfm == "cogvideox":
                        feats = forward_cogvideox(
                            model,
                            frames_input,
                            t=args.t,
                            layer_ids=missing_layers,
                            noise_generator=record_noise_generator,
                        )
                        for layer_id, feat in feats.items():
                            out_path = os.path.join(prefix_out_dir, out_fname(layer_id))
                            save_file(
                                {"feat": feat.half()},
                                out_path,
                                noise_metadata=record_noise_metadata,
                                extra_metadata=record_extra_metadata(layer_id),
                            )

                    elif args.vfm == "vjepa":
                        feats = model(frames_input)
                        out_path = os.path.join(prefix_out_dir, out_fname(0))
                        save_file(
                            {"feat": feats.half().contiguous()}, out_path,
                            extra_metadata=record_extra_metadata(0),
                        )

                    elif args.vfm == "vjepa2":
                        feats = model(frames_input, output_layers=missing_layers)
                        for layer_id, feat in feats.items():
                            out_path = os.path.join(prefix_out_dir, out_fname(layer_id))
                            save_file(
                                {"feat": feat.half().contiguous()}, out_path,
                                extra_metadata=record_extra_metadata(layer_id),
                            )

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
                        save_file(
                            {"feat": feat.half().contiguous()}, out_path,
                            extra_metadata=record_extra_metadata(0),
                        )

                    elif args.vfm == "videomae":
                        _videomae_model, _videomae_proc = model
                        _dev = next(_videomae_model.parameters()).device
                        with torch.no_grad():
                            feats = videomae_forward(
                                _videomae_model,
                                _videomae_proc,
                                frames_input,
                                _dev,
                                missing_layers,
                            )
                        for layer_id, feat in feats.items():
                            out_path = os.path.join(prefix_out_dir, out_fname(layer_id))
                            save_file(
                                {"feat": feat.half().contiguous()}, out_path,
                                extra_metadata=record_extra_metadata(layer_id),
                            )

                    elif args.vfm == "aether":
                        feats = forward_aether(
                            model,
                            frames_input,
                            t=args.t,
                            layer_ids=missing_layers,
                            noise_generator=record_noise_generator,
                        )
                        for layer_id, feat in feats.items():
                            expected_prefix = (2, 11, 30, 45)
                            if feat.ndim != 5 or tuple(feat.shape[:4]) != expected_prefix:
                                raise RuntimeError(
                                    "Aether streaming feature shape contract changed: "
                                    f"expected (2,11,30,45,C), found {tuple(feat.shape)}"
                                )
                            out_path = os.path.join(prefix_out_dir, out_fname(layer_id))
                            save_file(
                                {"feat": feat.half().contiguous()}, out_path,
                                noise_metadata=record_noise_metadata,
                                extra_metadata=record_extra_metadata(layer_id),
                            )

                    elif args.vfm == "opensora":
                        feats = opensora_extract(
                            frames_input, layer_indices=missing_layers,
                            config_path=args.opensora_config,
                        )
                        for layer_id, feat in feats.items():
                            out_path = os.path.join(prefix_out_dir, out_fname(layer_id))
                            save_file(
                                {"feat": feat.half().contiguous()}, out_path,
                                extra_metadata=record_extra_metadata(layer_id),
                            )

                    elif args.vfm == "f3r":
                        f3r_indices = list(record["indices"])
                        while len(f3r_indices) < args.num_frames:
                            f3r_indices.append(f3r_indices[-1])
                        f3r_filelist = [
                            os.path.join(s["img_dir"], s["img_files"][idx])
                            for idx in f3r_indices
                        ]
                        _dev = next(model.parameters()).device
                        raw_feats = f3r_forward(
                            f3r_filelist,
                            model,
                            _dev,
                            random_generator=record_noise_generator,
                        )
                        if isinstance(raw_feats, torch.Tensor):
                            raw_feats = [raw_feats]
                        for layer_id in missing_layers:
                            raw = raw_feats[layer_id]
                            feat_spatial = raw.reshape(
                                len(f3r_filelist), F3R_HTOK, F3R_WTOK, F3R_C
                            ).contiguous()
                            out_path = os.path.join(prefix_out_dir, out_fname(layer_id))
                            save_file(
                                {"feat": feat_spatial.half()}, out_path,
                                noise_metadata=record_noise_metadata,
                                extra_metadata=record_extra_metadata(layer_id),
                            )

                    elif args.vfm == "minimax_h3":
                        with torch.no_grad():
                            feats = model.forward(
                                video=frames_input,
                                prompt=args.prompt,
                                t=args.t,
                                output_layer_indices=missing_layers,
                            )
                        for layer_id, raw_feat in feats.items():
                            reshaped = reshape_minimax_h3(
                                raw_feat,
                                t_tokens=model.feat_t,
                                h_tokens=model.feat_h,
                                w_tokens=model.feat_w,
                            )
                            out_path = os.path.join(prefix_out_dir, out_fname(layer_id))
                            save_file(
                                {
                                    "feat": prepare_feature_for_storage(
                                        reshaped,
                                        vfm=args.vfm,
                                        context=(
                                            f"streaming_prefix:{name}:"
                                            f"window{record.get('window_id')}:"
                                            f"prefix{record.get('valid_length')}:layer{layer_id}"
                                        ),
                                    )
                                },
                                out_path,
                                noise_metadata=record_noise_metadata,
                                extra_metadata=record_extra_metadata(layer_id),
                            )

                    elif args.vfm == "lingbot":
                        with torch.no_grad():
                            feats = model.forward(
                                video=frames_input,
                                prompt=args.prompt,
                                t=args.t,
                                output_layer_indices=missing_layers,
                                noise_generator=record_noise_generator,
                            )
                        for layer_id, raw_feat in feats.items():
                            reshaped = reshape_lingbot(
                                raw_feat,
                                t_tokens=model.feat_t,
                                h_tokens=model.feat_h,
                                w_tokens=model.feat_w,
                            )
                            out_path = os.path.join(prefix_out_dir, out_fname(layer_id))
                            save_file(
                                {"feat": reshaped.half()}, out_path,
                                noise_metadata=record_noise_metadata,
                                extra_metadata=record_extra_metadata(layer_id),
                            )

                    meta_record = dict(record)
                    meta_record["input_length"] = len(frames_input)
                    meta_record["valid_length"] = valid_length
                    meta_record["pad_mode"] = "repeat_tail"
                    prefix_meta.append(meta_record)
                    processed_prefixes += 1

                index_provenance = None
                if args.mode == "streaming_prefix":
                    protocol = canonical_protocol_metadata(args)
                    index_provenance = {
                        "canonical_protocol": protocol["protocol"],
                        "canonical_protocol_sha256": protocol["protocol_sha256"],
                        "canonical_obs_len": int(args.streaming_window_obs_len),
                    }
                    generation_contract = backend_generation_contract(args)
                    if generation_contract is not None:
                        index_provenance.update(generation_contract)
                    if gap_worklist is not None:
                        index_provenance.update(
                            static_prefix_index_metadata(gap_worklist)
                        )
                atomic_save_npy(
                    os.path.join(
                        out_dir,
                        "prefix_index.npy" if args.mode == "streaming_prefix" else "context_index.npy",
                    ),
                    np.array(prefix_meta, dtype=object),
                    provenance=index_provenance,
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
                        elif args.vfm == "minimax_h3":
                            with torch.no_grad():
                                feats = model.forward(
                                    video=rep_frames,
                                    prompt=args.prompt,
                                    t=args.t,
                                    output_layer_indices=missing_lyrs,
                                )
                            for layer_id, raw_feat in feats.items():
                                reshaped = reshape_minimax_h3(
                                    raw_feat,
                                    t_tokens=model.feat_t,
                                    h_tokens=model.feat_h,
                                    w_tokens=model.feat_w,
                                )
                                per_layer_collect[layer_id].append(
                                    reshaped[reshaped.shape[0] // 2]
                                )
                        elif args.vfm == "lingbot":
                            with torch.no_grad():
                                feats = model.forward(
                                    video=rep_frames,
                                    prompt=args.prompt,
                                    t=args.t,
                                    output_layer_indices=missing_lyrs,
                                )
                            for layer_id, raw_feat in feats.items():
                                reshaped = reshape_lingbot(
                                    raw_feat,
                                    t_tokens=model.feat_t,
                                    h_tokens=model.feat_h,
                                    w_tokens=model.feat_w,
                                )
                                per_layer_collect[layer_id].append(
                                    reshaped[reshaped.shape[0] // 2]
                                )
                        else:
                            raise NotImplementedError(
                                f"streaming_target mode not implemented for vfm={args.vfm}")
                    os.makedirs(prefix_out_dir, exist_ok=True)
                    for layer_id, lst in per_layer_collect.items():
                        stacked = prepare_feature_for_storage(
                            torch.stack(lst, dim=0),
                            vfm=args.vfm,
                            context=(
                                f"streaming_target:{name}:"
                                f"prefix{prefix_len}:layer{layer_id}"
                            ),
                        )
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
                if gap_worklist is not None:
                    target_global = sorted(
                        {
                            frame_index
                            for layer in missing_layers
                            for frame_index in requested_target_by_layer[layer]
                        }
                    )
                    target_provenance = []
                elif args.target_from_streaming_windows:
                    target_global, target_provenance = target_selection_for_scene(s, args)
                else:
                    target_global = target_indices_for_scene(s, args)
                    target_provenance = []
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
                per_layer_collect = {
                    layer: [] for layer in (missing_layers if args.vfm != "vjepa" else [0])
                }

                target_cache_noise_metadata = None
                target_noise_samples = []
                if requires_sample_local_noise(args):
                    target_cache_noise_metadata = target_noise_metadata(
                        args, name, target_global
                    )
                    target_noise_samples = target_cache_noise_metadata[
                        "noise_seed_samples"
                    ]

                for target_pos, (gi, tgt_img) in enumerate(zip(target_global, target_imgs)):
                    active_layers = (
                        sorted(
                            set(gap_worklist.target_layers(name, int(gi)))
                            & set(missing_layers)
                        )
                        if gap_worklist is not None
                        else list(missing_layers if args.vfm != "vjepa" else [0])
                    )
                    if not active_layers:
                        continue
                    rep_frames = [tgt_img] * args.num_frames
                    target_noise_generator = (
                        generator_for_noise_sample(target_noise_samples[target_pos])
                        if target_noise_samples
                        else None
                    )
                    if args.vfm == "wan":
                        with torch.no_grad():
                            feats = model.forward(
                                video=rep_frames, prompt=args.prompt, t=args.t,
                                output_layer_indices=active_layers,
                                ensemble_size=args.ensemble,
                                noise_generator=target_noise_generator,
                            )
                        for layer_id, raw_feat in feats.items():
                            reshaped = reshape_to_t_h_w_c(raw_feat)  # (T, H_f, W_f, C)
                            per_layer_collect[layer_id].append(reshaped[reshaped.shape[0] // 2])
                    elif args.vfm == "cogvideox":
                        feats = forward_cogvideox(
                            model,
                            rep_frames,
                            t=args.t,
                            layer_ids=active_layers,
                            noise_generator=target_noise_generator,
                        )
                        collect_cogvideox_target_rows_cpu(
                            feats, per_layer_collect
                        )
                        # Do not let loop-local references retain the previous
                        # target's multi-layer CUDA outputs into the next
                        # two-clip forward.
                        del feats
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    elif args.vfm == "vjepa2":
                        feats = model(rep_frames, output_layers=active_layers)
                        for layer_id, feat in feats.items():
                            per_layer_collect[layer_id].append(feat[feat.shape[0] // 2])
                    elif args.vfm == "dino":
                        _dino_model, _dino_proc = model
                        _dev = next(_dino_model.parameters()).device
                        with torch.no_grad():
                            tok = _dino_forward(_dino_model, _dino_proc, [tgt_img], _dev)
                        per_layer_collect[0].append(_dino_reshape(tok[0]))
                    elif args.vfm == "videomae":
                        _videomae_model, _videomae_proc = model
                        _dev = next(_videomae_model.parameters()).device
                        with torch.no_grad():
                            feats = videomae_forward(
                                _videomae_model,
                                _videomae_proc,
                                rep_frames,
                                _dev,
                                active_layers,
                            )
                        for layer_id, feat in feats.items():
                            per_layer_collect[layer_id].append(feat[feat.shape[0] // 2])
                    elif args.vfm == "aether":
                        with torch.no_grad():
                            feats = forward_aether(
                                model,
                                rep_frames,
                                t=args.t,
                                layer_ids=active_layers,
                                noise_generator=target_noise_generator,
                            )
                        for layer_id, feat in feats.items():
                            expected_prefix = (2, 11, 30, 45)
                            if feat.ndim != 5 or tuple(feat.shape[:4]) != expected_prefix:
                                raise RuntimeError(
                                    "Aether target feature shape contract changed: "
                                    f"expected (2,11,30,45,C), found {tuple(feat.shape)}"
                                )
                            flattened = feat.reshape(-1, *feat.shape[2:])
                            per_layer_collect[layer_id].append(
                                flattened[flattened.shape[0] // 2]
                            )
                    elif args.vfm == "f3r":
                        target_path = os.path.join(s["img_dir"], s["img_files"][int(gi)])
                        f3r_filelist = [target_path] * args.num_frames
                        _dev = next(model.parameters()).device
                        with torch.no_grad():
                            raw_feats = f3r_forward(
                                f3r_filelist,
                                model,
                                _dev,
                                random_generator=target_noise_generator,
                            )
                        if isinstance(raw_feats, torch.Tensor):
                            raw_feats = [raw_feats]
                        for layer_id in active_layers:
                            raw = raw_feats[layer_id]
                            feat_spatial = raw.reshape(
                                len(f3r_filelist), F3R_HTOK, F3R_WTOK, F3R_C
                            ).contiguous()
                            per_layer_collect[layer_id].append(feat_spatial[feat_spatial.shape[0] // 2])
                    elif args.vfm == "minimax_h3":
                        with torch.no_grad():
                            feats = model.forward(
                                video=rep_frames,
                                prompt=args.prompt,
                                t=args.t,
                                output_layer_indices=active_layers,
                            )
                        for layer_id, raw_feat in feats.items():
                            reshaped = reshape_minimax_h3(
                                raw_feat,
                                t_tokens=model.feat_t,
                                h_tokens=model.feat_h,
                                w_tokens=model.feat_w,
                            )
                            per_layer_collect[layer_id].append(
                                reshaped[reshaped.shape[0] // 2]
                            )
                    elif args.vfm == "lingbot":
                        with torch.no_grad():
                            feats = model.forward(
                                video=rep_frames,
                                prompt=args.prompt,
                                t=args.t,
                                output_layer_indices=active_layers,
                                noise_generator=target_noise_generator,
                            )
                        for layer_id, raw_feat in feats.items():
                            reshaped = reshape_lingbot(
                                raw_feat,
                                t_tokens=model.feat_t,
                                h_tokens=model.feat_h,
                                w_tokens=model.feat_w,
                            )
                            per_layer_collect[layer_id].append(
                                reshaped[reshaped.shape[0] // 2]
                            )
                    else:
                        raise NotImplementedError(
                            f"target_isolated mode not implemented for vfm={args.vfm}"
                        )

                if gap_worklist is not None:
                    delta_root = target_delta_dir(out_dir)
                    delta_root.mkdir(parents=True, exist_ok=True)
                    for layer_id, lst in per_layer_collect.items():
                        layer_indices = requested_target_by_layer[layer_id]
                        if len(lst) != len(layer_indices):
                            raise RuntimeError(
                                f"{name}/layer{layer_id}: target delta row count "
                                f"{len(lst)} != worklist {len(layer_indices)}"
                            )
                        index_path = target_delta_index_path(out_dir, layer_id)
                        target_array = np.array(layer_indices, dtype=np.int64)
                        # The index is written before the feature so both
                        # sidecars can bind the identical index digest.
                        atomic_save_npy(index_path, target_array)
                        index_sha = stable_file_sha256(index_path)
                        shared_delta_metadata = {
                            "gap_contract": "CANONICAL_V2_GAP_WORKLIST_V1",
                            "gap_worklist_file_sha256": gap_worklist.file_sha256,
                            "gap_worklist_content_sha256": gap_worklist.content_sha256,
                            "canonical_source_sha256": gap_worklist.canonical_source_sha256,
                            "vfm": args.vfm,
                            "model_id": args.model_id,
                            "timestep": int(args.t),
                            "layer": int(layer_id),
                            "prompt": str(args.prompt),
                            "resize": list(args.resize),
                            "num_frames": int(args.num_frames),
                            "target_indices_sha256": index_sha,
                        }
                        # Extend the already-written generic index sidecar with
                        # the immutable worklist/model binding.
                        index_sidecar = Path(f"{index_path}.manifest.json")
                        index_meta = json.loads(index_sidecar.read_text())
                        index_meta.update(shared_delta_metadata)
                        atomic_write_json(index_sidecar, index_meta)
                        stacked = prepare_feature_for_storage(
                            torch.stack(lst, dim=0),
                            vfm=args.vfm,
                            context=f"target_delta:{name}:layer{layer_id}",
                        )
                        layer_noise_metadata = (
                            target_noise_metadata(args, name, layer_indices)
                            if requires_sample_local_noise(args)
                            else target_cache_noise_metadata
                        )
                        save_file(
                            {"feat": stacked},
                            target_delta_feature_path(out_dir, layer_id),
                            noise_metadata=layer_noise_metadata,
                            extra_metadata={
                                key: value
                                for key, value in shared_delta_metadata.items()
                                if key
                                not in {
                                    "vfm",
                                    "model_id",
                                    "timestep",
                                    "prompt",
                                    "resize",
                                    "num_frames",
                                }
                            },
                        )
                else:
                    for layer_id, lst in per_layer_collect.items():
                        stacked = prepare_feature_for_storage(
                            torch.stack(lst, dim=0),
                            vfm=args.vfm,
                            context=f"target_isolated:{name}:layer{layer_id}",
                        )
                        out_path = os.path.join(out_dir, out_fname(layer_id))
                        save_file(
                            {"feat": stacked},
                            out_path,
                            noise_metadata=target_cache_noise_metadata,
                        )
                    # Save target index metadata (one per scene, layer-agnostic)
                    target_index_path = os.path.join(out_dir, "target_indices.npy")
                    target_array = np.array(target_global, dtype=np.int64)
                    target_index_provenance = None
                    if args.target_from_streaming_windows:
                        protocol = canonical_protocol_metadata(args)
                        target_index_provenance = {
                            "canonical_protocol": protocol["protocol"],
                            "canonical_protocol_sha256": protocol["protocol_sha256"],
                            "canonical_obs_len": int(args.streaming_window_obs_len),
                        }
                        generation_contract = backend_generation_contract(args)
                        if generation_contract is not None:
                            target_index_provenance.update(generation_contract)
                    atomic_save_npy(
                        target_index_path,
                        target_array,
                        provenance=target_index_provenance,
                    )
                    if args.target_from_streaming_windows:
                        write_target_index_metadata(
                            target_index_path,
                            target_array,
                            target_provenance,
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

            elif args.vfm == "videomae":
                _videomae_model, _videomae_proc = model
                _dev = next(_videomae_model.parameters()).device
                with torch.no_grad():
                    feats = videomae_forward(
                        _videomae_model,
                        _videomae_proc,
                        frames_input,
                        _dev,
                        missing_layers,
                    )
                for layer_id, feat in feats.items():
                    out_path = os.path.join(out_dir, out_fname(layer_id))
                    save_file({"feat": feat.half().contiguous()}, out_path)

            elif args.vfm == "aether":
                feats = forward_aether(
                    model,
                    frames_input,
                    t=args.t,
                    layer_ids=missing_layers,
                    # Non-canonical legacy modes do not carry a per-sample
                    # noise contract.  Canonical streaming/target extraction
                    # is handled by the two branches above and always passes
                    # its SHA-bound deterministic generator.
                    noise_generator=None,
                )
                for layer_id, feat in feats.items():
                    expected_prefix = (2, 11, 30, 45)
                    if feat.ndim != 5 or tuple(feat.shape[:4]) != expected_prefix:
                        raise RuntimeError(
                            "Aether feature shape contract changed: "
                            f"expected (2,11,30,45,C), found {tuple(feat.shape)}"
                        )
                    out_path = os.path.join(out_dir, out_fname(layer_id))
                    save_file({"feat": feat.half().contiguous()}, out_path)

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

            elif args.vfm == "minimax_h3":
                with torch.no_grad():
                    feats = model.forward(
                        video=frames_input,
                        prompt=args.prompt,
                        t=args.t,
                        output_layer_indices=missing_layers,
                    )
                for layer_id, raw_feat in feats.items():
                    reshaped = reshape_minimax_h3(
                        raw_feat,
                        t_tokens=model.feat_t,
                        h_tokens=model.feat_h,
                        w_tokens=model.feat_w,
                    )
                    out_path = os.path.join(out_dir, out_fname(layer_id))
                    save_file(
                        {
                            "feat": prepare_feature_for_storage(
                                reshaped,
                                vfm=args.vfm,
                                context=f"legacy:{args.mode}:{name}:layer{layer_id}",
                            )
                        },
                        out_path,
                    )

            elif args.vfm == "lingbot":
                with torch.no_grad():
                    feats = model.forward(
                        video=frames_input,
                        prompt=args.prompt,
                        t=args.t,
                        output_layer_indices=missing_layers,
                    )
                for layer_id, raw_feat in feats.items():
                    reshaped = reshape_lingbot(
                        raw_feat,
                        t_tokens=model.feat_t,
                        h_tokens=model.feat_h,
                        w_tokens=model.feat_w,
                    )
                    out_path = os.path.join(out_dir, out_fname(layer_id))
                    save_file({"feat": reshaped.half()}, out_path)

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
    if failed:
        raise SystemExit(f"{failed} scenes failed")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
