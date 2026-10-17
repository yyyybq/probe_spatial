#!/usr/bin/env python3
"""Extract frame-aligned MLLM/VLM activations for InsScene-15K.

This script mirrors ``features/run_inscene15k.py`` but targets models whose
internal states are token sequences instead of diffusion/video grids.  It saves
the selected visual token activations as ``feat: (S, H_t, W_t, C)`` safetensors
so the existing InsScene15KDataset and probe heads can consume them unchanged.

The default path is Qwen2.5-VL through Transformers.  BAGEL is kept as a generic
``trust_remote_code`` Hugging Face backend because the model card points users to
the upstream BAGEL library rather than exposing a stable Transformers snippet.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import sys
import time
from datetime import timedelta
from glob import glob
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
from features.noise_seed import (
    DETERMINISTIC_NOISE_CONTRACT,
    NO_NOISE_SEED_SCHEME,
    deterministic_no_noise_metadata,
    noise_metadata_matches,
)
from features.run_inscene15k import (
    CANONICAL_STREAMING_OBS_LEN,
    CANONICAL_STREAMING_PROTOCOL,
    CANONICAL_STREAMING_PROTOCOL_SHA256,
    canonical_protocol_metadata,
    streaming_prefix_records_for_scene as canonical_streaming_prefix_records_for_scene,
    target_selection_for_scene as canonical_target_selection_for_scene,
    validate_canonical_window_args,
)
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


SEEDED_STREAM_NAMESPACE = "FEAT_STREAMING_CANONICAL_V2_SEEDED_V1"
SEEDED_TARGET_NAMESPACE = "FEAT_TARGET_CANONICAL_V2_SEEDED_V1"
SEEDED_NAMESPACES = {SEEDED_STREAM_NAMESPACE, SEEDED_TARGET_NAMESPACE}
GAP_NAMESPACES = {
    "FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1",
    "FEAT_TARGET_CANONICAL_V2_GAP_DELTA_V1",
}


def merge_extra_metadata(manifest: dict, extra_metadata: dict | None) -> None:
    """Merge gap metadata without allowing identity fields to be changed."""
    if not extra_metadata:
        return
    conflicts = sorted(
        key
        for key, value in extra_metadata.items()
        if key in manifest and manifest[key] != value
    )
    if conflicts:
        raise RuntimeError(
            "target delta metadata attempts to replace reserved fields: "
            + ",".join(conflicts)
        )
    for key, value in extra_metadata.items():
        if key not in manifest:
            manifest[key] = value


def collect_infinigen_scenes(source_path: str) -> list[dict]:
    scenes = []
    for scene_dir in sorted(glob(os.path.join(source_path, "scene_*"))):
        candidates = [scene_dir] + [
            p for p in sorted(glob(os.path.join(scene_dir, "*"))) if os.path.isdir(p)
        ]
        for candidate in candidates:
            img_dir = os.path.join(candidate, "frames", "Image", "camera_0")
            if not os.path.isdir(img_dir):
                continue
            imgs = sort_frame_names(f for f in os.listdir(img_dir) if f.endswith(".png"))
            if len(imgs) >= 2:
                scenes.append(
                    {
                        "source": "infinigen",
                        "scene_dir": candidate,
                        "img_dir": img_dir,
                        "img_files": imgs,
                    }
                )
                if candidate == scene_dir:
                    break
    return scenes


def collect_scannetpp_scenes(source_path: str) -> list[dict]:
    nested = os.path.join(source_path, "processed_scannetpp_v2")
    if os.path.isdir(nested):
        source_path = nested

    scenes = []
    for scene_id in sorted(os.listdir(source_path)):
        scene_dir = os.path.join(source_path, scene_id)
        img_dir = os.path.join(scene_dir, "images")
        mask_dir = os.path.join(scene_dir, "refined_ins_ids")
        meta_path = os.path.join(scene_dir, "scene_iphone_metadata.npz")
        if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir) or not os.path.exists(meta_path):
            continue
        image_files = set(f for f in os.listdir(img_dir) if f.endswith(".jpg"))
        mask_files = set(os.listdir(mask_dir))
        meta = np.load(meta_path)
        meta_images = [str(f) for f in meta["images"].tolist()]
        imgs = []
        pose_indices = []
        for meta_idx, fname in enumerate(meta_images):
            if fname in image_files and f"{fname}.npy" in mask_files:
                imgs.append(fname)
                pose_indices.append(meta_idx)
        if len(imgs) >= 2:
            scenes.append(
                {
                    "source": "scannetpp",
                    "scene_dir": scene_dir,
                    "img_dir": img_dir,
                    "img_files": imgs,
                    "pose_indices": pose_indices,
                    "poses_c2w": meta["trajectories"][pose_indices].astype(np.float32),
                }
            )
    return scenes


def scene_name(scene_info: dict) -> str:
    scene_dir = scene_info["scene_dir"]
    if scene_info["source"] == "infinigen":
        parts = Path(scene_dir).parts
        for i, part in enumerate(parts):
            if part.startswith("scene_"):
                return "__".join(parts[i:])
    return Path(scene_dir).name


def _parse_int_list(value):
    """Parse comma/space-separated integers into a sorted list, or None."""
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
    return f"prefix_{int(record['tail']):06d}"


def select_frames(img_files: Sequence[str], n: int) -> list[int]:
    total = len(img_files)
    if total <= n:
        return list(range(total))
    return np.linspace(0, total - 1, n).round().astype(int).tolist()


def parse_int_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    items = str(value).replace(",", " ").split()
    parsed = sorted({int(item) for item in items if item.strip()})
    return parsed or None


def streaming_prefix_records(
    num_frames: int,
    min_len: int = 1,
    max_len: int | None = None,
    stride: int = 1,
    lengths: Sequence[int] | None = None,
) -> list[dict]:
    min_len = max(int(min_len), 1)
    stride = max(int(stride), 1)
    max_len = num_frames if max_len is None else min(int(max_len), num_frames)
    if max_len < min_len:
        return []
    if lengths is not None:
        candidate_lengths = [
            int(length) for length in lengths
            if min_len <= int(length) <= max_len
        ]
    else:
        candidate_lengths = range(min_len, max_len + 1, stride)
    return [
        {
            "tail": int(length) - 1,
            "indices": list(range(int(length))),
            "valid_length": int(length),
        }
        for length in candidate_lengths
    ]


def prefix_dir_name(record: dict) -> str:
    if "window_id" in record:
        return f"window_{int(record['window_id']):04d}/prefix_{int(record['tail']):06d}"
    return f"prefix_{int(record['tail']):06d}"


def streaming_prefix_records_for_scene(scene: dict, args) -> list[dict]:
    if args.mode == "streaming_prefix":
        return canonical_streaming_prefix_records_for_scene(scene, args)
    num_frames = len(scene["img_files"])
    return streaming_prefix_records(
        num_frames,
        min_len=args.prefix_min_len,
        max_len=args.prefix_max_len,
        stride=args.prefix_stride,
        lengths=args.prefix_lengths,
    )


def streaming_target_indices_for_scene(scene: dict, args) -> list[int]:
    if args.target_from_streaming_windows:
        indices, _ = canonical_target_selection_for_scene(scene, args)
        return indices
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
        windows = [{"window_id": 0, "indices": list(range(required_obs))}]

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


def context_segment_records(
    num_frames: int,
    context_len: int = 76,
    stride: int = 1,
    min_tail: int = 0,
) -> list[dict]:
    context_len = max(int(context_len), 1)
    stride = max(int(stride), 1)
    min_tail = max(int(min_tail), 0)
    records = []
    for tail in range(min_tail, num_frames, stride):
        start = max(0, tail - context_len + 1)
        records.append(
            {
                "start": start,
                "tail": tail,
                "indices": list(range(start, tail + 1)),
                "valid_length": tail - start + 1,
            }
        )
    return records


def context_dir_name(record: dict) -> str:
    return f"context_{int(record['start']):06d}_{int(record['tail']):06d}"


def load_frames(
    img_dir: str,
    img_files: Sequence[str],
    indices: Sequence[int],
    resize: Tuple[int, int] | None,
) -> list[Image.Image]:
    frames = []
    for idx in indices:
        img = Image.open(os.path.join(img_dir, img_files[idx])).convert("RGB")
        if resize is not None:
            h, w = resize
            img = img.resize((w, h), Image.LANCZOS)
        frames.append(img)
    return frames


def _to_device(inputs, device):
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in inputs.items()
    }


def _find_visual_token_id(model, processor, token_kind: str) -> int | None:
    names = [f"{token_kind}_token_id"]
    if token_kind == "image":
        names += ["vision_token_id"]
    for obj in (getattr(model, "config", None), getattr(processor, "tokenizer", None)):
        if obj is None:
            continue
        for name in names:
            value = getattr(obj, name, None)
            if isinstance(value, int):
                return value
    token = "<|image_pad|>" if token_kind == "image" else "<|video_pad|>"
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        try:
            value = tokenizer.convert_tokens_to_ids(token)
            if isinstance(value, int) and value >= 0:
                return value
        except Exception:
            pass
    return None


def _split_visual_tokens(
    hidden: torch.Tensor,
    input_ids: torch.Tensor,
    grids: torch.Tensor,
    visual_token_id: int,
) -> torch.Tensor:
    """Split visual token hidden states into ``(S, H, W, C)``.

    ``grids`` is the Qwen-style ``(S, 3)`` tensor of ``(T, H, W)`` token grids.
    For multi-image extraction each frame has T=1; if a backend emits T>1 we
    flatten temporal groups into the frame axis.
    """
    positions = (input_ids[0] == visual_token_id).nonzero(as_tuple=False).flatten()
    visual = hidden[0, positions]
    chunks = []
    cursor = 0
    for grid in grids.cpu().tolist():
        t, h, w = [int(x) for x in grid]
        n = t * h * w
        chunk = visual[cursor : cursor + n]
        cursor += n
        if chunk.shape[0] != n:
            raise RuntimeError(
                f"Expected {n} visual tokens for grid {(t, h, w)}, got {chunk.shape[0]}"
            )
        chunks.append(chunk.reshape(t, h, w, -1))
    if cursor != visual.shape[0]:
        log.warning("Unused visual tokens after grid split: %d", visual.shape[0] - cursor)
    return torch.cat(chunks, dim=0).contiguous()


def _square_token_grid(tokens: torch.Tensor) -> torch.Tensor:
    n, c = tokens.shape
    side = int(round(n ** 0.5))
    if side * side == n:
        return tokens.reshape(1, side, side, c)
    return tokens.reshape(1, n, 1, c)


def _split_merged_visual_tokens(
    tokens: torch.Tensor,
    grids: torch.Tensor,
    spatial_merge_unit: int,
) -> torch.Tensor:
    """Split Qwen visual-merger tokens into ``(S, Hm, Wm, C)``.

    Qwen2.5-VL's vision tower first builds a patch grid ``(T,H,W)`` and then
    merges ``spatial_merge_unit`` neighboring spatial patches before feeding
    visual embeddings to the LLM.  The pooler output length is therefore
    ``T * (H/merge) * (W/merge)`` per image/video.
    """
    merge = int(round(spatial_merge_unit ** 0.5))
    chunks = []
    cursor = 0
    for grid in grids.cpu().tolist():
        t, h, w = [int(x) for x in grid]
        hm = max(h // merge, 1)
        wm = max(w // merge, 1)
        n = t * hm * wm
        chunk = tokens[cursor : cursor + n]
        cursor += n
        if chunk.shape[0] != n:
            raise RuntimeError(
                f"Expected {n} merged visual tokens for grid {(t, h, w)}, got {chunk.shape[0]}"
            )
        chunks.append(chunk.reshape(t, hm, wm, -1))
    if cursor != tokens.shape[0]:
        log.warning("Unused merged visual tokens after grid split: %d", tokens.shape[0] - cursor)
    return torch.cat(chunks, dim=0).contiguous()


def _split_visual_tokens_from_grid(tokens: torch.Tensor, grids: torch.Tensor) -> torch.Tensor:
    """Split unmerged vision-tower tokens into ``(S, H, W, C)``."""

    if tokens.ndim == 3 and tokens.shape[0] == 1:
        tokens = tokens[0]
    if tokens.ndim != 2:
        tokens = tokens.reshape(-1, tokens.shape[-1])

    chunks = []
    cursor = 0
    for grid in grids.cpu().tolist():
        t, h, w = [int(x) for x in grid]
        n = t * h * w
        chunk = tokens[cursor : cursor + n]
        cursor += n
        if chunk.shape[0] != n:
            raise RuntimeError(
                f"Expected {n} visual tokens for grid {(t, h, w)}, got {chunk.shape[0]}"
            )
        chunks.append(chunk.reshape(t, h, w, -1))
    if cursor != tokens.shape[0]:
        log.warning("Unused unmerged visual tokens after grid split: %d", tokens.shape[0] - cursor)
    return torch.cat(chunks, dim=0).contiguous()


class MLLMActivationExtractor:
    def __init__(
        self,
        backend: str,
        model_id: str,
        device: str,
        torch_dtype: str,
        prompt: str,
        attn_implementation: str | None,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.backend = backend
        self.prompt = prompt
        self.device = torch.device(device)
        dtype = {
            "auto": "auto",
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[torch_dtype]

        model_kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        if device == "auto":
            model_kwargs["device_map"] = "auto"
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )

        if backend == "qwen2_5_vl":
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration

                model_cls = Qwen2_5_VLForConditionalGeneration
            except Exception:
                model_cls = AutoModelForCausalLM
        else:
            model_cls = AutoModelForCausalLM

        self.model = model_cls.from_pretrained(model_id, **model_kwargs).eval()
        if device != "auto":
            self.model.to(self.device)

    def _qwen_visual_module(self):
        """Return the Qwen2.5-VL vision tower if this model exposes one."""
        visual = getattr(self.model, "visual", None)
        if visual is not None:
            return visual
        inner = getattr(self.model, "model", None)
        return getattr(inner, "visual", None)

    @torch.no_grad()
    def _forward_qwen_visual(self, inputs, layer_ids: Iterable[int]) -> dict[int, torch.Tensor] | None:
        visual = self._qwen_visual_module()
        if visual is None or "pixel_values" not in inputs or "image_grid_thw" not in inputs:
            return None

        layer_ids = [int(layer_id) for layer_id in layer_ids]
        pixel_values = inputs["pixel_values"]
        image_grid = inputs["image_grid_thw"]
        dtype = next(visual.parameters()).dtype

        # layer -1 is the historical default: Qwen visual-merger tokens in the
        # LLM hidden dimension. Non-negative layer ids are raw vision-tower block
        # outputs, captured before the merger so layer-wise probes can inspect
        # where spatial information peaks.
        requested_blocks = [l for l in layer_ids if l >= 0]
        captured: dict[int, torch.Tensor] = {}
        hooks = []
        blocks = getattr(visual, "blocks", None)
        if blocks is not None:
            for layer_id in requested_blocks:
                if layer_id >= len(blocks):
                    log.warning("Qwen visual layer %d out of range [0, %d)", layer_id, len(blocks))
                    continue

                def _make_hook(idx):
                    def _hook(_module, _inputs, output):
                        tensor = output[0] if isinstance(output, (tuple, list)) else output
                        captured[idx] = tensor.detach().float().cpu()
                    return _hook

                hooks.append(blocks[layer_id].register_forward_hook(_make_hook(layer_id)))

        visual_out = visual(pixel_values.to(dtype=dtype), grid_thw=image_grid)
        for hook in hooks:
            hook.remove()

        feats: dict[int, torch.Tensor] = {}
        for layer_id, tokens in captured.items():
            feats[layer_id] = _split_visual_tokens_from_grid(
                tokens,
                image_grid.detach().cpu(),
            )
        if -1 not in layer_ids:
            return feats

        if hasattr(visual_out, "pooler_output"):
            tokens = visual_out.pooler_output
        elif isinstance(visual_out, (tuple, list)) and len(visual_out) > 1:
            tokens = visual_out[1]
        else:
            tokens = visual_out

        spatial_merge_unit = int(getattr(visual, "spatial_merge_unit", 4))
        feat = _split_merged_visual_tokens(
            tokens.detach().float().cpu(),
            image_grid.detach().cpu(),
            spatial_merge_unit=spatial_merge_unit,
        )
        feats[-1] = feat
        return feats

    @torch.no_grad()
    def __call__(self, frames: Sequence[Image.Image], layer_ids: Iterable[int]) -> dict[int, torch.Tensor]:
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": frame} for frame in frames],
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=list(frames),
            padding=True,
            return_tensors="pt",
        )
        inputs = _to_device(inputs, self.device)

        layer_ids = [int(layer_id) for layer_id in layer_ids]

        # Historical Qwen default: layer -1 stores final visual-merger tokens.
        # For explicit layer sweeps, use the model hidden states below so each
        # requested layer corresponds to a distinct VLM activation.
        if self.backend == "qwen2_5_vl" and set(layer_ids) == {-1}:
            visual_feats = self._forward_qwen_visual(inputs, layer_ids)
            if visual_feats is not None:
                return visual_feats

        outputs = self.model(
            **inputs,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        image_token_id = _find_visual_token_id(self.model, self.processor, "image")
        grids = inputs.get("image_grid_thw")

        feats = {}
        for layer_id in layer_ids:
            hidden = hidden_states[layer_id].detach().float().cpu()
            if image_token_id is not None and grids is not None:
                feat = _split_visual_tokens(
                    hidden,
                    inputs["input_ids"].detach().cpu(),
                    grids.detach().cpu(),
                    image_token_id,
                )
            else:
                if image_token_id is None:
                    log.warning("No image token id found; saving all non-batch tokens as a 1D grid.")
                feat = _square_token_grid(hidden[0].cpu())
            feats[int(layer_id)] = feat
        return feats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Qwen2.5-VL/BAGEL visual-token activations for InsScene-15K",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--source", default="scannetpp", choices=["all", "infinigen", "scannetpp"])
    parser.add_argument("--backend", default="qwen2_5_vl", choices=["qwen2_5_vl", "bagel_hf", "generic_hf"])
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--t", type=int, default=749, help="Provenance timestep identity")
    parser.add_argument("--vfm-name", default=None, help="Output subdir name; defaults to backend.")
    parser.add_argument("--prompt", default="Describe the spatial layout of this scene.")
    parser.add_argument(
        "--output-layers",
        nargs="+",
        default=None,
        help="Layer ids to cache. Accepts integers plus aliases default, last, all.",
    )
    parser.add_argument("--all-layers", action="store_true")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--resize", nargs=2, type=int, default=[336, 336], metavar=("H", "W"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument(
        "--mode",
        default="streaming_prefix",
        choices=["normal", "shuffled", "target_isolated", "streaming_prefix", "context_segment"],
        help=(
            "streaming_prefix: independently forward online prefixes [I_0..I_t] "
            "(default project setting). normal: legacy sampled clip forward. "
            "shuffled: per-frame order shuffle, "
            "then inverse-order saved features for A3. target_isolated: each target "
            "frame is forwarded alone for C probes. context_segment: independently "
            "forward causal segments [I_start..I_tail]."
        ),
    )
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--num-targets", type=int, default=8, help="0 means every frame")
    parser.add_argument("--prefix-stride", type=int, default=1)
    parser.add_argument("--prefix-min-len", type=int, default=1)
    parser.add_argument("--prefix-max-len", type=int, default=24)
    parser.add_argument("--prefix-lengths", default="8,12,16,24")
    parser.add_argument("--temporal-sampling", default="motion_uniform", choices=["motion_uniform", "none"])
    parser.add_argument("--streaming-motion-step", type=float, default=0.35)
    parser.add_argument("--streaming-rotation-weight", type=float, default=0.5)
    parser.add_argument("--streaming-window-stride", type=int, default=8)
    parser.add_argument("--streaming-max-windows-per-scene", type=int, default=4)
    parser.add_argument(
        "--streaming-window-obs-len",
        type=int,
        default=CANONICAL_STREAMING_OBS_LEN,
    )
    parser.add_argument("--context-len", type=int, default=76)
    parser.add_argument("--context-stride", type=int, default=1)
    parser.add_argument("--target-from-streaming-windows", action="store_true")
    parser.add_argument("--target-prefix-lengths", default="8,12,16,24")
    parser.add_argument("--target-horizons", default="1,2,4")
    parser.add_argument("--no-cache-checksum", action="store_true")
    parser.add_argument("--canonical-gap-worklist", default=None)
    parser.add_argument("--canonical-gap-worklist-sha256", default=None)
    args = parser.parse_args()
    args.prefix_lengths = parse_int_list(args.prefix_lengths)
    args.target_prefix_lengths = parse_int_list(args.target_prefix_lengths) or [8, 12, 16, 24]
    args.target_horizons = parse_int_list(args.target_horizons) or [1, 2, 4]
    canonical_feature_mode = args.mode == "streaming_prefix" or (
        args.mode == "target_isolated" and args.target_from_streaming_windows
    )
    feature_namespace = os.environ.get("FEATURE_NAMESPACE", "").strip()
    if canonical_feature_mode:
        if args.backend != "qwen2_5_vl":
            raise SystemExit("canonical-v2 MLLM extraction currently supports qwen2_5_vl only")
        if feature_namespace not in SEEDED_NAMESPACES and not (
            args.canonical_gap_worklist and feature_namespace in GAP_NAMESPACES
        ):
            raise SystemExit(
                "canonical-v2 Qwen extraction requires a seeded-v1 FEATURE_NAMESPACE"
            )
        if Path(args.out_root).resolve().name != feature_namespace:
            raise SystemExit(
                "canonical-v2 --out-root basename must exactly match FEATURE_NAMESPACE"
            )
        if args.no_cache_checksum:
            raise SystemExit("canonical-v2 forbids --no-cache-checksum")
        if os.environ.get("EXTRACT_DONE_INDEX", "").strip():
            raise SystemExit(
                "canonical-v2 forbids EXTRACT_DONE_INDEX key-only cache completion"
            )
        try:
            validate_canonical_window_args(args)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        protocol = canonical_protocol_metadata(args)
        if (
            protocol["protocol"] != CANONICAL_STREAMING_PROTOCOL
            or protocol["protocol_sha256"] != CANONICAL_STREAMING_PROTOCOL_SHA256
            or int(args.streaming_window_obs_len) != CANONICAL_STREAMING_OBS_LEN
        ):
            raise SystemExit("Qwen runtime does not match the frozen canonical-v2 protocol")
    if (
        args.mode in {"normal", "shuffled", "context_segment"}
        and os.environ.get("ALLOW_NON_STREAMING") != "1"
    ):
        raise SystemExit(
            f"--mode {args.mode} is a legacy non-streaming extraction path. "
            "Streaming is the default; use --mode streaming_prefix, or set "
            "ALLOW_NON_STREAMING=1 for intentional legacy extraction."
        )
    if (
        args.mode in {"streaming_prefix", "target_isolated", "context_segment"}
        and args.source != "scannetpp"
        and os.environ.get("ALLOW_INFINIGEN_TEMPORAL") != "1"
    ):
        raise SystemExit(
            "Temporal streaming/target extraction is ScanNet++ only. "
            "Use --source scannetpp, or set ALLOW_INFINIGEN_TEMPORAL=1 only for "
            "explicit legacy/debug reproduction."
        )

    default_model = {
        "qwen2_5_vl": "Qwen/Qwen2.5-VL-7B-Instruct",
        "bagel_hf": "ByteDance-Seed/BAGEL-7B-MoT",
        "generic_hf": None,
    }[args.backend]
    if args.model_id is None:
        if default_model is None:
            raise ValueError("--model-id is required for generic_hf")
        args.model_id = default_model
    vfm_name = args.vfm_name or ("bagel" if args.backend == "bagel_hf" else args.backend)
    args.output_layers = parse_layers_arg(
        args.output_layers,
        vfm_name=vfm_name,
        model_id=args.model_id,
        all_layers=args.all_layers,
    )
    log.info("Using output layers for %s: %s", vfm_name, args.output_layers)

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
                vfm=vfm_name,
                model_id=str(args.model_id),
                timestep=int(args.t),
                feature_namespace=feature_namespace,
                output_layers=args.output_layers,
                protocol_sha256=protocol["protocol_sha256"],
                actual_prompt=str(args.prompt),
                preprocess_identity=(
                    streaming_worklist_preprocess_identity(
                        vfm_name, args.resize, int(args.num_frames)
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

    scenes = []
    if args.source in ("all", "infinigen"):
        path = os.path.join(args.data_root, "processed_infinigen")
        if os.path.isdir(path):
            scenes.extend(collect_infinigen_scenes(path))
    if args.source in ("all", "scannetpp"):
        path = os.path.join(args.data_root, "processed_scannetpp_v2")
        if os.path.isdir(path):
            scenes.extend(collect_scannetpp_scenes(path))

    scenes = scenes[args.start : args.end]
    if gap_worklist is not None:
        scenes = [scene for scene in scenes if scene_name(scene) in gap_worklist.scenes]
    log.info("Processing %d scenes", len(scenes))

    def out_fname(layer: int) -> str:
        return f"feature_layer{layer}.sft"

    def requested_stream_layers(scene: str, record: dict) -> list[int]:
        if gap_worklist is None:
            return list(args.output_layers)
        return sorted(
            gap_worklist.stream_layers(
                scene, int(record["window_id"]), int(record["valid_length"])
            )
        )

    def requested_target_indices_by_layer(scene: str) -> dict[int, list[int]]:
        if gap_worklist is None:
            return {}
        result = {int(layer): [] for layer in args.output_layers}
        for frame_index in gap_worklist.target_frame_indices(scene):
            for layer in gap_worklist.target_layers(scene, frame_index):
                result[int(layer)].append(int(frame_index))
        return {layer: values for layer, values in result.items() if values}

    def target_delta_dir(base_dir: str) -> Path:
        if gap_worklist is None:
            raise RuntimeError("target_delta_dir requires a bound gap worklist")
        return Path(base_dir) / ".canonical_v2_target_delta" / gap_worklist.file_sha256[:16]

    def target_delta_feature_path(base_dir: str, layer: int) -> Path:
        return target_delta_dir(base_dir) / f"feature_t{int(args.t)}_layer{int(layer)}.delta.sft"

    def target_delta_index_path(base_dir: str, layer: int) -> Path:
        return target_delta_dir(base_dir) / f"target_indices_layer{int(layer)}.npy"

    extractor_source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    no_noise_metadata = deterministic_no_noise_metadata()

    def cache_complete(path: str) -> bool:
        path_obj = Path(path)
        if not canonical_feature_mode:
            return path_obj.is_file()
        sidecar = Path(f"{path_obj}.manifest.json")
        if (
            path_obj.is_symlink()
            or sidecar.is_symlink()
            or not path_obj.is_file()
            or not sidecar.is_file()
        ):
            return False
        try:
            metadata = json.loads(sidecar.read_text())
            digest = hashlib.sha256(path_obj.read_bytes()).hexdigest()
            protocol = canonical_protocol_metadata(args)
            expected = {
                "schema_version": 1,
                "frame_index_schema": "image_mask_intersection_v1",
                "backend": args.backend,
                "vfm": vfm_name,
                "model_id": args.model_id,
                "mode": args.mode,
                "feature_namespace": feature_namespace,
                "timestep": int(args.t),
                "prompt": args.prompt,
                "resize": list(args.resize) if args.resize else None,
                "file": path_obj.name,
                "size_bytes": path_obj.stat().st_size,
                "sha256": digest,
                "extractor_source_sha256": extractor_source_sha256,
                "canonical_protocol": protocol["protocol"],
                "canonical_protocol_sha256": protocol["protocol_sha256"],
                "canonical_obs_len": int(args.streaming_window_obs_len),
            }
            return (
                isinstance(metadata, dict)
                and all(metadata.get(key) == value for key, value in expected.items())
                and noise_metadata_matches(metadata, no_noise_metadata)
            )
        except Exception:
            return False

    def stream_gap_metadata(
        scene: str,
        record: dict,
        layer: int,
        _requested_layers: Sequence[int],
    ) -> dict:
        metadata = {
            **stream_unit_semantic_metadata(
                gap_worklist, scene, record, layer
            ),
            **stream_unit_producer_metadata(gap_worklist),
        }
        # ``save_feat`` owns these immutable base fields.  Keep them in the
        # stable semantic expectation, but do not attempt to replace reserved
        # manifest keys while publishing.  ``num_frames`` is *not* owned by
        # ``save_feat``: Qwen is no-pad, so each prefix must keep its own
        # frame count from ``stream_unit_semantic_metadata``.
        for key in (
            "frame_index_schema",
            "resize",
            "canonical_protocol",
            "canonical_protocol_sha256",
            "canonical_obs_len",
        ):
            metadata.pop(key, None)
        return metadata

    def stream_gap_complete(
        path: str, scene: str, record: dict, layer: int, requested_layers: Sequence[int]
    ) -> bool:
        if gap_worklist is None:
            return cache_complete(path)
        path_obj = Path(path)
        sidecar = Path(f"{path_obj}.manifest.json")
        if path_obj.is_symlink() or sidecar.is_symlink() or not path_obj.is_file() or not sidecar.is_file():
            return False
        try:
            metadata = json.loads(sidecar.read_text())
            digest = hashlib.sha256(path_obj.read_bytes()).hexdigest()
            expected = stream_unit_semantic_metadata(
                gap_worklist, scene, record, layer
            )
            return (
                cache_complete(str(path_obj))
                and
                metadata.get("sha256") == digest
                and metadata.get("size_bytes") == path_obj.stat().st_size
                and stream_unit_semantic_matches(metadata, expected)
                and metadata.get("vfm") == vfm_name
                and metadata.get("model_id") == args.model_id
                and metadata.get("timestep") == int(args.t)
                and metadata.get("prompt") == args.prompt
                and metadata.get("resize") == list(args.resize)
            )
        except Exception:
            return False

    def target_indices_for_scene(scene: dict) -> list[int]:
        if args.target_from_streaming_windows:
            return streaming_target_indices_for_scene(scene, args)
        if args.num_targets == 0:
            return list(range(len(scene["img_files"])))
        count = max(2, min(int(args.num_targets), len(scene["img_files"])))
        return np.linspace(0, len(scene["img_files"]) - 1, count).round().astype(int).tolist()

    def target_provenance_for_scene(scene: dict) -> list[dict]:
        if not args.target_from_streaming_windows:
            return []
        _, provenance = canonical_target_selection_for_scene(scene, args)
        return provenance

    def index_provenance() -> dict:
        """Return the immutable canonical identity for structural indexes."""
        protocol = canonical_protocol_metadata(args)
        result = {
            "canonical_protocol": protocol["protocol"],
            "canonical_protocol_sha256": protocol["protocol_sha256"],
            "canonical_obs_len": int(args.streaming_window_obs_len),
            "extractor_source_sha256": extractor_source_sha256,
            **no_noise_metadata,
        }
        if gap_worklist is not None:
            result.update(static_prefix_index_metadata(gap_worklist))
        return result

    def _jsonable(value):
        if isinstance(value, np.ndarray):
            return [_jsonable(item) for item in value.tolist()]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(key): _jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonable(item) for item in value]
        return value

    def prefix_index_complete(base_dir: str, expected_records: Sequence[dict]) -> bool:
        """Require a byte-bound index whose records equal the canonical windows."""
        path = Path(base_dir) / "prefix_index.npy"
        sidecar_path = Path(f"{path}.manifest.json")
        if (
            path.is_symlink()
            or sidecar_path.is_symlink()
            or not path.is_file()
            or not sidecar_path.is_file()
        ):
            return False
        try:
            cached = _jsonable(np.load(path, allow_pickle=True).tolist())
            expected = _jsonable(list(expected_records))
            sidecar = json.loads(sidecar_path.read_text())
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            provenance = index_provenance()
            return (
                cached == expected
                and sidecar.get("schema_version") == 1
                and sidecar.get("file") == path.name
                and sidecar.get("feature_namespace") == feature_namespace
                and sidecar.get("size_bytes") == path.stat().st_size
                and sidecar.get("sha256") == digest
                and sidecar.get("shape") == [len(expected)]
                and sidecar.get("dtype") == "object"
                and all(sidecar.get(key) == value for key, value in provenance.items())
                and noise_metadata_matches(sidecar, no_noise_metadata)
            )
        except Exception:
            return False

    def target_index_complete(
        base_dir: str,
        expected_indices: Sequence[int],
        expected_provenance: Sequence[dict] | None = None,
    ) -> bool:
        if not expected_indices:
            return True
        path_obj = Path(base_dir) / "target_indices.npy"
        sidecar_path = Path(f"{path_obj}.manifest.json")
        target_meta_path = Path(base_dir) / "target_indices.meta.json"
        if (
            path_obj.is_symlink()
            or sidecar_path.is_symlink()
            or target_meta_path.is_symlink()
            or not path_obj.is_file()
            or not sidecar_path.is_file()
            or not target_meta_path.is_file()
        ):
            return False
        try:
            cached = np.load(path_obj).astype(np.int64).tolist()
        except Exception:
            return False
        if cached != [int(v) for v in expected_indices]:
            return False
        if not canonical_feature_mode:
            return True
        try:
            sidecar = json.loads(sidecar_path.read_text())
            target_meta = json.loads(target_meta_path.read_text())
            protocol = canonical_protocol_metadata(args)
            digest = hashlib.sha256(path_obj.read_bytes()).hexdigest()
            return (
                sidecar.get("schema_version") == 1
                and sidecar.get("file") == path_obj.name
                and sidecar.get("sha256") == digest
                and sidecar.get("size_bytes") == path_obj.stat().st_size
                and sidecar.get("feature_namespace") == feature_namespace
                and sidecar.get("canonical_protocol") == protocol["protocol"]
                and sidecar.get("canonical_protocol_sha256") == protocol["protocol_sha256"]
                and sidecar.get("canonical_obs_len") == int(args.streaming_window_obs_len)
                and sidecar.get("extractor_source_sha256") == extractor_source_sha256
                and noise_metadata_matches(sidecar, no_noise_metadata)
                and target_meta.get("schema_version") == 2
                and target_meta.get("feature_namespace") == feature_namespace
                and target_meta.get("protocol") == protocol["protocol"]
                and target_meta.get("protocol_sha256") == protocol["protocol_sha256"]
                and target_meta.get("obs_len") == int(args.streaming_window_obs_len)
                and target_meta.get("prefixes") == [int(v) for v in args.target_prefix_lengths]
                and target_meta.get("horizons") == [int(v) for v in args.target_horizons]
                and target_meta.get("include_anchors") is True
                and target_meta.get("target_count") == len(expected_indices)
                and target_meta.get("target_indices_sha256") == digest
                and target_meta.get("extractor_source_sha256") == extractor_source_sha256
                and target_meta.get("provenance") == list(expected_provenance or [])
                and noise_metadata_matches(target_meta, no_noise_metadata)
            )
        except Exception:
            return False

    def write_target_index_metadata(
        path: str,
        target_indices: Sequence[int],
        target_provenance: Sequence[dict],
    ) -> None:
        path_obj = Path(path)
        protocol = canonical_protocol_metadata(args)
        metadata = {
            "schema_version": 2,
            "feature_namespace": feature_namespace,
            "protocol": protocol["protocol"],
            "protocol_sha256": protocol["protocol_sha256"],
            "obs_len": int(args.streaming_window_obs_len),
            "prefixes": [int(value) for value in args.target_prefix_lengths],
            "horizons": [int(value) for value in args.target_horizons],
            "include_anchors": True,
            "target_count": len(target_indices),
            "target_indices_sha256": hashlib.sha256(path_obj.read_bytes()).hexdigest(),
            "extractor_source_sha256": extractor_source_sha256,
            "provenance": list(target_provenance),
            **no_noise_metadata,
        }
        target_meta = path_obj.with_name("target_indices.meta.json")
        temporary = target_meta.with_name(f".{target_meta.name}.tmp-{os.getpid()}")
        temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, target_meta)

    def layer_paths_complete(base_dir: str, layers: Sequence[int]) -> bool:
        return all(cache_complete(os.path.join(base_dir, out_fname(l))) for l in layers)

    def stored_prefix_record(record: dict) -> dict:
        stored = dict(record)
        stored["input_length"] = len(record["indices"])
        stored["valid_length"] = len(record["indices"])
        stored["pad_mode"] = "none"
        return stored

    def save_index(path: str, array, provenance: dict | None = None) -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        tmp = path_obj.with_name(f".{path_obj.name}.tmp-{os.getpid()}")
        with open(tmp, "wb") as handle:
            np.save(handle, array)
        digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
        os.replace(tmp, path_obj)
        metadata = {
            "schema_version": 1,
            "file": path_obj.name,
            "feature_namespace": feature_namespace or None,
            "size_bytes": path_obj.stat().st_size,
            "sha256": digest,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
        if provenance:
            metadata.update(provenance)
        sidecar = Path(f"{path_obj}.manifest.json")
        sidecar_tmp = sidecar.with_name(f".{sidecar.name}.tmp-{os.getpid()}")
        sidecar_tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        os.replace(sidecar_tmp, sidecar)

    def save_feat(path: str | Path, feat: torch.Tensor, extra_metadata: dict | None = None) -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        tmp = path_obj.with_name(f".{path_obj.name}.tmp-{os.getpid()}")
        saved_feat = feat.half().contiguous()
        save_file({"feat": saved_feat}, str(tmp))
        digest = None
        if not args.no_cache_checksum:
            sha = hashlib.sha256()
            with open(tmp, "rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    sha.update(chunk)
            digest = sha.hexdigest()
        os.replace(tmp, path_obj)
        manifest = {
            "schema_version": 1,
            "frame_index_schema": "image_mask_intersection_v1",
            "backend": args.backend,
            "vfm": vfm_name,
            "model_id": args.model_id,
            "mode": args.mode,
            "feature_namespace": feature_namespace or None,
            "timestep": int(args.t),
            "prompt": args.prompt,
            "resize": list(args.resize) if args.resize else None,
            "extractor_source_sha256": extractor_source_sha256,
            "file": path_obj.name,
            "size_bytes": path_obj.stat().st_size,
            "sha256": digest,
            "tensors": {
                "feat": {"shape": list(saved_feat.shape), "dtype": str(saved_feat.dtype)}
            },
        }
        if canonical_feature_mode:
            protocol = canonical_protocol_metadata(args)
            manifest.update(
                {
                    "canonical_protocol": protocol["protocol"],
                    "canonical_protocol_sha256": protocol["protocol_sha256"],
                    "canonical_obs_len": int(args.streaming_window_obs_len),
                    **no_noise_metadata,
                }
            )
        merge_extra_metadata(manifest, extra_metadata)
        os.chmod(path_obj, 0o644)
        sidecar = Path(f"{path_obj}.manifest.json")
        sidecar_tmp = sidecar.with_name(f".{sidecar.name}.tmp-{os.getpid()}")
        sidecar_tmp.write_text(json.dumps(manifest, indent=2) + "\n")
        os.chmod(sidecar_tmp, 0o644)
        os.replace(sidecar_tmp, sidecar)
        os.chmod(sidecar, 0o644)

    def target_delta_complete(base_dir: str, layer: int, expected_indices: Sequence[int]) -> bool:
        if gap_worklist is None:
            return False
        feature_path = target_delta_feature_path(base_dir, layer)
        index_path = target_delta_index_path(base_dir, layer)
        feature_sidecar = Path(f"{feature_path}.manifest.json")
        index_sidecar = Path(f"{index_path}.manifest.json")
        if any(
            path.is_symlink() or not path.is_file()
            for path in (feature_path, index_path, feature_sidecar, index_sidecar)
        ):
            return False
        try:
            indices = np.load(index_path).astype(np.int64).tolist()
            if indices != [int(value) for value in expected_indices]:
                return False
            feature_meta = json.loads(feature_sidecar.read_text())
            index_meta = json.loads(index_sidecar.read_text())
            feature_sha = hashlib.sha256(feature_path.read_bytes()).hexdigest()
            index_sha = hashlib.sha256(index_path.read_bytes()).hexdigest()
            shared = {
                "gap_worklist_file_sha256": gap_worklist.file_sha256,
                "gap_worklist_content_sha256": gap_worklist.content_sha256,
                "canonical_source_sha256": gap_worklist.canonical_source_sha256,
                "vfm": vfm_name,
                "model_id": args.model_id,
                "timestep": int(args.t),
                "layer": int(layer),
                "prompt": str(args.prompt),
                "resize": list(args.resize),
                "num_frames": int(args.num_frames),
                "target_indices_sha256": index_sha,
            }
            return (
                all(feature_meta.get(key) == value for key, value in shared.items())
                and all(index_meta.get(key) == value for key, value in shared.items())
                and feature_meta.get("sha256") == feature_sha
                and index_meta.get("sha256") == index_sha
                and feature_meta.get("tensors", {}).get("feat", {}).get("shape", [None])[0]
                == len(indices)
                and noise_metadata_matches(feature_meta, no_noise_metadata)
            )
        except Exception:
            return False

    done = 0
    for scene in scenes:
        out_dir = os.path.join(args.out_root, vfm_name, scene["source"], scene_name(scene))
        if args.mode in ("streaming_prefix", "context_segment"):
            records = (
                streaming_prefix_records_for_scene(scene, args)
                if args.mode == "streaming_prefix"
                else context_segment_records(
                    len(scene["img_files"]),
                    context_len=args.context_len,
                    stride=args.context_stride,
                    min_tail=0,
                )
            )
            all_exist = (
                True
                if gap_worklist is not None
                else
                prefix_index_complete(
                    out_dir,
                    [stored_prefix_record(record) for record in records],
                )
                if args.mode == "streaming_prefix"
                else True
            )
            for record in records:
                requested_layers = requested_stream_layers(scene_name(scene), record)
                if not requested_layers:
                    continue
                subdir = prefix_dir_name(record) if args.mode == "streaming_prefix" else context_dir_name(record)
                record_dir = os.path.join(out_dir, subdir)
                if not all(
                    stream_gap_complete(
                        os.path.join(record_dir, out_fname(layer)),
                        scene_name(scene), record, layer, requested_layers,
                    )
                    for layer in requested_layers
                ):
                    all_exist = False
                    break
        elif args.mode == "target_isolated":
            expected_targets = target_indices_for_scene(scene)
            if gap_worklist is not None:
                requested = requested_target_indices_by_layer(scene_name(scene))
                invalid = sorted(
                    set().union(*(set(values) for values in requested.values()), set())
                    - set(expected_targets)
                )
                if invalid:
                    raise RuntimeError(
                        f"{scene_name(scene)}: target gap worklist escapes canonical union: {invalid[:8]}"
                    )
                all_exist = bool(requested) and all(
                    target_delta_complete(out_dir, layer, indices)
                    for layer, indices in requested.items()
                )
            else:
                all_exist = (
                    target_index_complete(
                        out_dir,
                        expected_targets,
                        target_provenance_for_scene(scene),
                    )
                    and layer_paths_complete(out_dir, args.output_layers)
                )
        else:
            all_exist = layer_paths_complete(out_dir, args.output_layers)
        if all_exist:
            done += 1
    log.info("Already done: %d/%d", done, len(scenes))
    if done == len(scenes):
        return
    if os.environ.get("EXTRACT_CACHE_CHECK_ONLY") == "1":
        log.info("EXTRACT_CACHE_CHECK_ONLY=1; exiting before model load")
        return

    extractor = MLLMActivationExtractor(
        backend=args.backend,
        model_id=args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
        prompt=args.prompt,
        attn_implementation=args.attn_implementation,
    )

    processed = 0
    failed = 0
    total_time = 0.0
    resize = tuple(args.resize) if args.resize else None
    for i, scene in enumerate(scenes):
        name = scene_name(scene)
        out_dir = os.path.join(args.out_root, vfm_name, scene["source"], name)

        t0 = time.time()
        try:
            if args.mode in ("streaming_prefix", "context_segment"):
                records = (
                    streaming_prefix_records_for_scene(scene, args)
                    if args.mode == "streaming_prefix"
                    else context_segment_records(
                        len(scene["img_files"]),
                        context_len=args.context_len,
                        stride=args.context_stride,
                        min_tail=0,
                    )
                )
                if not records:
                    log.warning("%s/%s: no %s records to process", scene["source"], name, args.mode)
                    continue

                meta = []
                processed_records = 0
                skipped_records = 0
                for record in records:
                    subdir = prefix_dir_name(record) if args.mode == "streaming_prefix" else context_dir_name(record)
                    record_dir = os.path.join(out_dir, subdir)
                    requested_layers = requested_stream_layers(name, record)
                    missing_layers = [
                        layer for layer in requested_layers
                        if not stream_gap_complete(
                            os.path.join(record_dir, out_fname(layer)),
                            name, record, layer, requested_layers,
                        )
                    ]
                    record_meta = stored_prefix_record(record)
                    meta.append(record_meta)
                    if not missing_layers:
                        skipped_records += 1
                        continue
                    frames = load_frames(scene["img_dir"], scene["img_files"], record["indices"], resize)
                    feats = extractor(frames, missing_layers)
                    for layer_id, feat in feats.items():
                        save_feat(
                            os.path.join(record_dir, out_fname(layer_id)),
                            feat,
                            extra_metadata=(
                                stream_gap_metadata(
                                    name, record, layer_id, requested_layers
                                )
                                if gap_worklist is not None
                                else None
                            ),
                        )
                    processed_records += 1

                save_index(
                    os.path.join(
                        out_dir,
                        "prefix_index.npy" if args.mode == "streaming_prefix" else "context_index.npy",
                    ),
                    np.array(meta, dtype=object),
                    provenance=(index_provenance() if args.mode == "streaming_prefix" else None),
                )

                elapsed = time.time() - t0
                total_time += elapsed
                processed += 1
                remaining = len(scenes) - done - processed - failed
                eta = str(timedelta(seconds=int(total_time / max(processed, 1) * remaining)))
                log.info(
                    "[%d/%d] %s/%s %.1fs (%s +%d/skip %d) ETA %s",
                    done + processed + failed,
                    len(scenes),
                    scene["source"],
                    name,
                    elapsed,
                    args.mode,
                    processed_records,
                    skipped_records,
                    eta,
                )
                continue

            if args.mode == "target_isolated":
                expected_targets = target_indices_for_scene(scene)
                if not expected_targets:
                    log.warning("%s/%s: no streaming C target frames selected; skip target_isolated", scene["source"], name)
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
                            f"{name}: target gap worklist escapes canonical union: {invalid[:8]}"
                        )
                    missing_layers = [
                        layer
                        for layer, indices in requested_target_by_layer.items()
                        if not target_delta_complete(out_dir, layer, indices)
                    ]
                else:
                    target_ok = target_index_complete(
                        out_dir,
                        expected_targets,
                        target_provenance_for_scene(scene),
                    )
                    missing_layers = [
                        layer for layer in args.output_layers
                        if not (target_ok and cache_complete(os.path.join(out_dir, out_fname(layer))))
                    ]
            else:
                missing_layers = [
                    layer for layer in args.output_layers
                    if not cache_complete(os.path.join(out_dir, out_fname(layer)))
                ]
            if not missing_layers:
                continue

            if args.mode == "target_isolated":
                target_indices = (
                    sorted(
                        {
                            frame_index
                            for layer in missing_layers
                            for frame_index in requested_target_by_layer[layer]
                        }
                    )
                    if gap_worklist is not None
                    else target_indices_for_scene(scene)
                )
                per_layer_collect = {layer: [] for layer in missing_layers}
                for target_idx in target_indices:
                    active_layers = (
                        sorted(
                            set(gap_worklist.target_layers(name, int(target_idx)))
                            & set(missing_layers)
                        )
                        if gap_worklist is not None
                        else list(missing_layers)
                    )
                    if not active_layers:
                        continue
                    frames = load_frames(scene["img_dir"], scene["img_files"], [target_idx], resize)
                    feats = extractor(frames, active_layers)
                    for layer_id, feat in feats.items():
                        per_layer_collect[layer_id].append(feat[0])
                os.makedirs(out_dir, exist_ok=True)
                if gap_worklist is not None:
                    target_delta_dir(out_dir).mkdir(parents=True, exist_ok=True)
                    for layer_id, values in per_layer_collect.items():
                        layer_indices = requested_target_by_layer[layer_id]
                        if len(values) != len(layer_indices):
                            raise RuntimeError(
                                f"{name}/layer{layer_id}: target delta row count "
                                f"{len(values)} != worklist {len(layer_indices)}"
                            )
                        index_path = target_delta_index_path(out_dir, layer_id)
                        target_array = np.array(layer_indices, dtype=np.int64)
                        save_index(str(index_path), target_array)
                        index_sha = hashlib.sha256(index_path.read_bytes()).hexdigest()
                        shared = {
                            "gap_contract": "CANONICAL_V2_GAP_WORKLIST_V1",
                            "gap_worklist_file_sha256": gap_worklist.file_sha256,
                            "gap_worklist_content_sha256": gap_worklist.content_sha256,
                            "canonical_source_sha256": gap_worklist.canonical_source_sha256,
                            "vfm": vfm_name,
                            "model_id": args.model_id,
                            "timestep": int(args.t),
                            "layer": int(layer_id),
                            "prompt": str(args.prompt),
                            "resize": list(args.resize),
                            "num_frames": int(args.num_frames),
                            "target_indices_sha256": index_sha,
                        }
                        index_sidecar = Path(f"{index_path}.manifest.json")
                        index_meta = json.loads(index_sidecar.read_text())
                        index_meta.update(shared)
                        index_tmp = index_sidecar.with_name(
                            f".{index_sidecar.name}.tmp-{os.getpid()}"
                        )
                        index_tmp.write_text(json.dumps(index_meta, indent=2, sort_keys=True) + "\n")
                        os.replace(index_tmp, index_sidecar)
                        save_feat(
                            target_delta_feature_path(out_dir, layer_id),
                            torch.stack(values, dim=0),
                            extra_metadata={
                                key: value
                                for key, value in shared.items()
                                if key
                                not in {
                                    "vfm",
                                    "model_id",
                                    "timestep",
                                    "prompt",
                                    "resize",
                                }
                            },
                        )
                else:
                    for layer_id, values in per_layer_collect.items():
                        save_feat(os.path.join(out_dir, out_fname(layer_id)), torch.stack(values, dim=0))
                    target_index_path = os.path.join(out_dir, "target_indices.npy")
                    target_array = np.array(target_indices, dtype=np.int64)
                    target_provenance = target_provenance_for_scene(scene)
                    save_index(
                        target_index_path,
                        target_array,
                        provenance=index_provenance(),
                    )
                    write_target_index_metadata(
                        target_index_path,
                        target_array,
                        target_provenance,
                    )
            else:
                indices = select_frames(scene["img_files"], args.num_frames)
                while len(indices) < args.num_frames:
                    indices.append(indices[-1])
                frames = load_frames(scene["img_dir"], scene["img_files"], indices, resize)
                if args.mode == "shuffled":
                    rng = np.random.default_rng(seed=args.shuffle_seed + i)
                    perm = rng.permutation(len(frames))
                    inv_perm = np.argsort(perm).tolist()
                    frames = [frames[int(j)] for j in perm]
                os.makedirs(out_dir, exist_ok=True)

                feats = extractor(frames, missing_layers)
                for layer_id, feat in feats.items():
                    if args.mode == "shuffled":
                        feat = feat[inv_perm]
                    save_feat(os.path.join(out_dir, out_fname(layer_id)), feat)
                save_index(os.path.join(out_dir, "source_indices.npy"), np.array(indices, dtype=np.int64))

        except Exception as exc:
            failed += 1
            log.error("[%d/%d] FAILED %s/%s: %s", done + processed + failed, len(scenes), scene["source"], name, exc)

    log.info("Done. Processed=%d Failed=%d Skipped=%d", processed, failed, done)
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
