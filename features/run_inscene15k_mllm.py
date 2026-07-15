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
    num_frames = len(scene["img_files"])
    max_record_len = max(args.prefix_lengths or [args.prefix_max_len])
    max_record_len = min(int(max_record_len), int(args.prefix_max_len), num_frames)
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
        lengths=args.prefix_lengths,
    )


def streaming_target_indices_for_scene(scene: dict, args) -> list[int]:
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
    parser.add_argument("--context-len", type=int, default=76)
    parser.add_argument("--context-stride", type=int, default=1)
    parser.add_argument("--target-from-streaming-windows", action="store_true")
    parser.add_argument("--target-prefix-lengths", default="8,12,16,24")
    parser.add_argument("--target-horizons", default="1,2,4")
    parser.add_argument("--no-cache-checksum", action="store_true")
    args = parser.parse_args()
    args.prefix_lengths = parse_int_list(args.prefix_lengths)
    args.target_prefix_lengths = parse_int_list(args.target_prefix_lengths) or [8, 12, 16, 24]
    args.target_horizons = parse_int_list(args.target_horizons) or [1, 2, 4]
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
    log.info("Processing %d scenes", len(scenes))

    def out_fname(layer: int) -> str:
        return f"feature_layer{layer}.sft"

    def cache_complete(path: str) -> bool:
        return os.path.exists(path)

    def target_indices_for_scene(scene: dict) -> list[int]:
        if args.target_from_streaming_windows:
            return streaming_target_indices_for_scene(scene, args)
        if args.num_targets == 0:
            return list(range(len(scene["img_files"])))
        count = max(2, min(int(args.num_targets), len(scene["img_files"])))
        return np.linspace(0, len(scene["img_files"]) - 1, count).round().astype(int).tolist()

    def target_index_complete(base_dir: str, expected_indices: Sequence[int]) -> bool:
        if not expected_indices:
            return True
        path = os.path.join(base_dir, "target_indices.npy")
        if not os.path.exists(path):
            return False
        try:
            cached = np.load(path).astype(np.int64).tolist()
        except Exception:
            return False
        return cached == [int(v) for v in expected_indices]

    def layer_paths_complete(base_dir: str, layers: Sequence[int]) -> bool:
        return all(cache_complete(os.path.join(base_dir, out_fname(l))) for l in layers)

    def save_index(path: str, array) -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        tmp = path_obj.with_name(f".{path_obj.name}.tmp-{os.getpid()}")
        with open(tmp, "wb") as handle:
            np.save(handle, array)
        os.replace(tmp, path_obj)

    def save_feat(path: str, feat: torch.Tensor) -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        tmp = path_obj.with_name(f".{path_obj.name}.tmp-{os.getpid()}")
        save_file({"feat": feat.half().contiguous()}, str(tmp))
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
            "prompt": args.prompt,
            "resize": list(args.resize) if args.resize else None,
            "file": path_obj.name,
            "size_bytes": path_obj.stat().st_size,
            "sha256": digest,
            "tensors": {"feat": {"shape": list(feat.shape), "dtype": str(feat.dtype)}},
        }
        sidecar = Path(f"{path_obj}.manifest.json")
        sidecar_tmp = sidecar.with_name(f".{sidecar.name}.tmp-{os.getpid()}")
        sidecar_tmp.write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(sidecar_tmp, sidecar)

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
            all_exist = True
            for record in records:
                subdir = prefix_dir_name(record) if args.mode == "streaming_prefix" else context_dir_name(record)
                if not layer_paths_complete(os.path.join(out_dir, subdir), args.output_layers):
                    all_exist = False
                    break
        elif args.mode == "target_isolated":
            expected_targets = target_indices_for_scene(scene)
            all_exist = (
                target_index_complete(out_dir, expected_targets)
                and layer_paths_complete(out_dir, args.output_layers)
            )
        else:
            all_exist = layer_paths_complete(out_dir, args.output_layers)
        if all_exist:
            done += 1
    log.info("Already done: %d/%d", done, len(scenes))
    if done == len(scenes):
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
                    missing_layers = [
                        layer for layer in args.output_layers
                        if not cache_complete(os.path.join(record_dir, out_fname(layer)))
                    ]
                    record_meta = dict(record)
                    record_meta["input_length"] = len(record["indices"])
                    record_meta["valid_length"] = len(record["indices"])
                    record_meta["pad_mode"] = "none"
                    meta.append(record_meta)
                    if not missing_layers:
                        skipped_records += 1
                        continue
                    frames = load_frames(scene["img_dir"], scene["img_files"], record["indices"], resize)
                    feats = extractor(frames, missing_layers)
                    for layer_id, feat in feats.items():
                        save_feat(os.path.join(record_dir, out_fname(layer_id)), feat)
                    processed_records += 1

                save_index(
                    os.path.join(
                        out_dir,
                        "prefix_index.npy" if args.mode == "streaming_prefix" else "context_index.npy",
                    ),
                    np.array(meta, dtype=object),
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
                target_ok = target_index_complete(out_dir, expected_targets)
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
                target_indices = target_indices_for_scene(scene)
                per_layer_collect = {layer: [] for layer in missing_layers}
                for target_idx in target_indices:
                    frames = load_frames(scene["img_dir"], scene["img_files"], [target_idx], resize)
                    feats = extractor(frames, missing_layers)
                    for layer_id, feat in feats.items():
                        per_layer_collect[layer_id].append(feat[0])
                os.makedirs(out_dir, exist_ok=True)
                for layer_id, values in per_layer_collect.items():
                    save_feat(os.path.join(out_dir, out_fname(layer_id)), torch.stack(values, dim=0))
                save_index(os.path.join(out_dir, "target_indices.npy"), np.array(target_indices, dtype=np.int64))
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


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
