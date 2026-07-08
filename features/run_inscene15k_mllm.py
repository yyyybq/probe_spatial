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
import importlib.util
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
except Exception:
    _FEATURE_LAYERS_PATH = Path(__file__).resolve().parents[1] / "vidfm3d" / "utils" / "feature_layers.py"
    _SPEC = importlib.util.spec_from_file_location("feature_layers", _FEATURE_LAYERS_PATH)
    feature_layers = importlib.util.module_from_spec(_SPEC)
    assert _SPEC.loader is not None
    sys.modules[_SPEC.name] = feature_layers
    _SPEC.loader.exec_module(feature_layers)
    parse_layers_arg = feature_layers.parse_layers_arg

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
            imgs = sorted(f for f in os.listdir(img_dir) if f.endswith(".png"))
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
        if not os.path.isdir(img_dir):
            continue
        imgs = sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))
        if len(imgs) >= 2:
            scenes.append(
                {
                    "source": "scannetpp",
                    "scene_dir": scene_dir,
                    "img_dir": img_dir,
                    "img_files": imgs,
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
    parser.add_argument("--source", default="all", choices=["all", "infinigen", "scannetpp"])
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
    # Streaming prefix mode
    parser.add_argument(
        "--mode", default="normal", choices=["normal", "streaming_prefix"],
        help=(
            "normal: select num-frames uniformly from the full scene (default).\n"
            "streaming_prefix: for each prefix length p, run the model on frames [0..tail] "
            "(subsampled to num-frames if needed) and save under prefix_<tail>/."
        ),
    )
    parser.add_argument("--prefix-stride", type=int, default=1,
                        help="Frame stride between streaming-prefix tails.")
    parser.add_argument("--prefix-min-len", type=int, default=1,
                        help="Minimum streaming-prefix length.")
    parser.add_argument("--prefix-max-len", type=int, default=None,
                        help="Maximum streaming-prefix length (defaults to scene length).")
    parser.add_argument("--prefix-lengths", default=None,
                        help="Exact streaming prefix lengths to cache, e.g. '4,8,16,32,64'.")
    args = parser.parse_args()
    args.prefix_lengths = _parse_int_list(args.prefix_lengths)

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

    # Pre-count already-done scenes for normal mode
    done = 0
    if args.mode == "normal":
        for scene in scenes:
            out_dir = os.path.join(args.out_root, vfm_name, scene["source"], scene_name(scene))
            if all(os.path.exists(os.path.join(out_dir, out_fname(l))) for l in args.output_layers):
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
            if args.mode == "streaming_prefix":
                records = streaming_prefix_records(
                    len(scene["img_files"]),
                    min_len=args.prefix_min_len,
                    max_len=args.prefix_max_len,
                    stride=args.prefix_stride,
                    model_max_len=None,
                    lengths=args.prefix_lengths,
                )
                if not records:
                    log.warning("[%d/%d] %s/%s: no streaming_prefix records, skipping",
                                done + processed + failed, len(scenes), scene["source"], name)
                    continue

                os.makedirs(out_dir, exist_ok=True)
                processed_prefixes = 0
                skipped_prefixes = 0
                prefix_meta = []
                for record in records:
                    pdir = os.path.join(out_dir, prefix_dir_name(record))
                    missing_layers = [
                        l for l in args.output_layers
                        if not os.path.exists(os.path.join(pdir, out_fname(l)))
                    ]
                    if not missing_layers:
                        skipped_prefixes += 1
                        meta_record = dict(record)
                        meta_record["input_length"] = args.num_frames
                        meta_record["pad_mode"] = "repeat_tail"
                        prefix_meta.append(meta_record)
                        continue

                    # Sub-sample from the prefix to num_frames; pad with repeat_tail if short
                    prefix_indices = record["indices"]
                    if len(prefix_indices) > args.num_frames:
                        sub = select_frames(prefix_indices, args.num_frames)
                        sample_indices = [prefix_indices[k] for k in sub]
                    else:
                        sample_indices = list(prefix_indices)
                    while len(sample_indices) < args.num_frames:
                        sample_indices.append(sample_indices[-1])

                    frames = load_frames(scene["img_dir"], scene["img_files"], sample_indices, resize)
                    os.makedirs(pdir, exist_ok=True)

                    feats = extractor(frames, missing_layers)
                    for layer_id, feat in feats.items():
                        save_file({"feat": feat.half().contiguous()}, os.path.join(pdir, out_fname(layer_id)))
                    np.save(os.path.join(pdir, "prefix_index.npy"), np.array(sample_indices, dtype=np.int64))

                    meta_record = dict(record)
                    meta_record["input_length"] = len(sample_indices)
                    meta_record["valid_length"] = record["valid_length"]
                    meta_record["pad_mode"] = "repeat_tail"
                    prefix_meta.append(meta_record)
                    processed_prefixes += 1

                np.save(os.path.join(out_dir, "prefix_index.npy"), np.array(prefix_meta, dtype=object))

                elapsed = time.time() - t0
                total_time += elapsed
                processed += 1
                remaining = len(scenes) - done - processed - failed
                avg = total_time / processed
                eta = str(timedelta(seconds=int(avg * remaining)))
                log.info(
                    "[%d/%d] %s/%s: %.1fs (%d frames, streaming_prefix +%d/skip %d) ETA %s",
                    done + processed + failed, len(scenes), scene["source"], name,
                    elapsed, len(scene["img_files"]), processed_prefixes, skipped_prefixes, eta,
                )

            else:  # normal mode
                missing_layers = [
                    layer for layer in args.output_layers
                    if not os.path.exists(os.path.join(out_dir, out_fname(layer)))
                ]
                if not missing_layers:
                    done += 1
                    continue

                indices = select_frames(scene["img_files"], args.num_frames)
                while len(indices) < args.num_frames:
                    indices.append(indices[-1])
                frames = load_frames(scene["img_dir"], scene["img_files"], indices, resize)
                os.makedirs(out_dir, exist_ok=True)

                feats = extractor(frames, missing_layers)
                for layer_id, feat in feats.items():
                    save_file({"feat": feat.half().contiguous()}, os.path.join(out_dir, out_fname(layer_id)))
                np.save(os.path.join(out_dir, "source_indices.npy"), np.array(indices, dtype=np.int64))

                elapsed = time.time() - t0
                total_time += elapsed
                processed += 1
                remaining = len(scenes) - done - processed - failed
                eta = str(timedelta(seconds=int(total_time / max(processed, 1) * remaining)))
                log.info("[%d/%d] %s/%s %.1fs ETA %s", done + processed + failed, len(scenes), scene["source"], name, elapsed, eta)

        except Exception as exc:
            failed += 1
            log.error("[%d/%d] FAILED %s/%s: %s", done + processed + failed, len(scenes), scene["source"], name, exc)

    log.info("Done. Processed=%d Failed=%d Skipped=%d", processed, failed, done)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
