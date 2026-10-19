#!/usr/bin/env python3
"""Extract the three frozen Active Spatial taps without generation.

This command is intentionally conservative.  It requires an exact registered
checkpoint path and a validated state bank, uses `eval()` and inference mode,
and writes sample ids next to every tensor.  No fallback model id, old cache,
or answer generation path is available.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from .io_utils import read_jsonl, stable_hash, write_json
from .protocol import validate_state_bank


UNIFIED_SYSTEM_PROMPT = (
    "You are a spatial navigation agent in a 3D indoor environment. "
    "Use the three consecutive camera observations and the task to choose a future navigation action. "
    "Do not assume access to camera pose, distance readouts, or object metadata beyond the task text."
)


def _requirements() -> tuple[Any, Any, Any]:
    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError("BLOCKED: feature extraction needs torch, Pillow, and transformers in the selected GPU environment") from exc
    return torch, Image, (AutoModelForCausalLM, AutoProcessor, AutoTokenizer)


def _task_text(sample: dict[str, Any]) -> str:
    task = sample.get("task", {})
    text = task.get("task_description") if isinstance(task, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError(f"BLOCKED: sample {sample.get('sample_id')} has no task_description")
    return text.strip()


def _load_images(sample: dict[str, Any], state_bank: Path, image_cls: Any) -> list[Any]:
    images = []
    for entry in sample["history"]:
        p = state_bank.parent / entry["image_path"]
        if not p.is_file():
            raise RuntimeError(f"BLOCKED: state-bank image missing: {p}")
        images.append(image_cls.open(p).convert("RGB"))
    return images


def _last_valid(hidden: Any, attention_mask: Any) -> Any:
    # State bank extraction is batch size 1, so the last non-padding position is
    # explicit rather than inferred from the raw tensor length.
    pos = int(attention_mask[0].long().sum().item()) - 1
    if pos < 0:
        raise RuntimeError("empty attention mask")
    return hidden[0, pos].detach().float().cpu()


class QwenExtractor:
    backend = "qwen2_5_vl"

    def __init__(self, checkpoint: Path, device: str, dtype: str, input_mode: str):
        torch, _, classes = _requirements()
        _, processor_cls, _ = classes
        from transformers import Qwen2_5_VLForConditionalGeneration
        self.torch = torch
        self.device = torch.device(device)
        self.processor = processor_cls.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            checkpoint, torch_dtype=getattr(torch, dtype), local_files_only=True,
        ).to(self.device).eval()
        # Qwen VL checkpoints differ across transformers revisions: this
        # export places the decoder beneath ``model.language_model`` rather
        # than directly at ``model.layers``.  Resolve the actual loaded
        # module, never silently substitute another layer family.
        backbone = self.model.model
        self.language_model = getattr(backbone, "language_model", backbone)
        self.layers = getattr(self.language_model, "layers", None)
        self.norm = getattr(self.language_model, "norm", None)
        if self.layers is None or self.norm is None:
            raise RuntimeError("BLOCKED: loaded Qwen checkpoint lacks decoder layers/norm probe boundaries")
        self.mid_layer = len(self.layers) // 2
        self.mid_module_path = f"model.model.{'language_model.' if self.language_model is not backbone else ''}layers[{self.mid_layer}]"
        self.norm_module_path = f"model.model.{'language_model.' if self.language_model is not backbone else ''}norm"
        self.input_mode = input_mode

    def _inputs(self, images: list[Any], text: str):
        content = []
        if self.input_mode == "multimodal":
            content.extend({"type": "image", "image": image} for image in images)
        content.append({"type": "text", "text": f"Task: {text}"})
        messages = [{"role": "system", "content": UNIFIED_SYSTEM_PROMPT}, {"role": "user", "content": content}]
        rendered = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        kwargs = {"text": [rendered], "padding": True, "return_tensors": "pt"}
        if self.input_mode == "multimodal":
            kwargs["images"] = images
        inputs = self.processor(**kwargs)
        return {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}

    def extract(self, images: list[Any], text: str) -> dict[str, Any]:
        torch = self.torch
        inputs = self._inputs(images, text)
        self.last_prompt_token_position = int(inputs["attention_mask"][0].long().sum().item()) - 1
        visual_capture: list[Any] = []
        mid_capture: list[Any] = []
        final_capture: list[Any] = []

        def visual_hook(_module, _args, output):
            value = output.pooler_output if hasattr(output, "pooler_output") else (output[1] if isinstance(output, (tuple, list)) and len(output) > 1 else output)
            visual_capture.append(value)

        def mid_hook(_module, _args, output):
            mid_capture.append(output[0] if isinstance(output, (tuple, list)) else output)

        def final_hook(_module, _args, output):
            final_capture.append(output)

        handles = [self.layers[self.mid_layer].register_forward_hook(mid_hook),
            self.norm.register_forward_hook(final_hook),
        ]
        if self.input_mode == "multimodal":
            handles.insert(0, self.model.visual.register_forward_hook(visual_hook))
        try:
            with torch.inference_mode():
                self.model(**inputs, use_cache=False, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
        expected_visual = 1 if self.input_mode == "multimodal" else 0
        if len(visual_capture) != expected_visual or len(mid_capture) != 1 or len(final_capture) != 1:
            raise RuntimeError("BLOCKED: Qwen tap hooks did not each fire exactly once")
        # Visual output is a token matrix after Qwen's merger.  Mean pooling is
        # explicitly over all three current input images, documented in manifest.
        result = {
            "llm_mid_last_prompt": _last_valid(mid_capture[0], inputs["attention_mask"]),
            "llm_final_norm_last_prompt": _last_valid(final_capture[0], inputs["attention_mask"]),
        }
        if self.input_mode == "multimodal":
            visual = visual_capture[0]
            if visual.ndim == 3:
                visual = visual[0]
            if visual.ndim != 2:
                raise RuntimeError(f"BLOCKED: unexpected Qwen visual terminal shape {tuple(visual.shape)}")
            result["vision_terminal"] = visual.detach().float().mean(dim=0).cpu()
        return result


class CambrianExtractor:
    backend = "cambrians"

    def __init__(self, checkpoint: Path, device: str, dtype: str, vagen_root: Path, input_mode: str):
        torch, _, classes = _requirements()
        auto_model, _, auto_tokenizer = classes
        if not vagen_root.is_dir():
            raise RuntimeError(f"BLOCKED: VAGEN root missing for Cambrian adapter: {vagen_root}")
        os.environ.setdefault("CAMBRIAN_SRC", str(vagen_root.parent / "cambrian-s"))
        sys.path.insert(0, str(vagen_root))
        importlib.import_module("vagen.models.cambrian_register")
        wrapper = importlib.import_module("vagen.models.cambrian_processor").CambrianProcessorWrapper
        self.torch = torch
        self.device = torch.device(device)
        tokenizer = auto_tokenizer.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
        self.processor = wrapper(tokenizer)
        self.model = auto_model.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True, torch_dtype=getattr(torch, dtype)).to(self.device).eval()
        inner = getattr(self.model, "model", None)
        self.layers = getattr(inner, "layers", None)
        self.norm = getattr(inner, "norm", None)
        if self.layers is None or self.norm is None:
            raise RuntimeError("BLOCKED: Cambrian formal adapter lacks model.layers/model.norm boundaries")
        self.mid_layer = len(self.layers) // 2
        self.vision = next((module for name, module in self.model.named_modules() if "vision_tower" in name), None)
        if self.vision is None and input_mode == "multimodal":
            raise RuntimeError("BLOCKED: Cambrian vision-terminal module could not be identified; no substitute tap is used")
        self.input_mode = input_mode

    def _inputs(self, images: list[Any], text: str) -> dict[str, Any]:
        content = []
        if self.input_mode == "multimodal":
            content.extend({"type": "image", "image": image} for image in images)
        content.append({"type": "text", "text": f"Task: {text}"})
        messages = [{"role": "system", "content": UNIFIED_SYSTEM_PROMPT}, {"role": "user", "content": content}]
        rendered = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = self.processor(rendered, images=images if self.input_mode == "multimodal" else None, return_tensors="pt")
        output = {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device),
        }
        if self.input_mode == "multimodal":
            pixel_values = self.processor.preprocess_images(images)["pixel_values"]
            output["pixel_values"] = pixel_values.to(self.device, dtype=getattr(self.torch, "bfloat16"))
        return output

    def extract(self, images: list[Any], text: str) -> dict[str, Any]:
        torch = self.torch
        inputs = self._inputs(images, text)
        captures: dict[str, list[Any]] = {"vision": [], "mid": [], "final": []}
        def capture(name: str):
            def hook(_module, _args, output):
                captures[name].append(output[0] if isinstance(output, (tuple, list)) else output)
            return hook
        handles = [self.layers[self.mid_layer].register_forward_hook(capture("mid")), self.norm.register_forward_hook(capture("final"))]
        if self.input_mode == "multimodal":
            handles.insert(0, self.vision.register_forward_hook(capture("vision")))
        try:
            with torch.inference_mode():
                self.model(**inputs, use_cache=False, output_hidden_states=False, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
        expected_vision = 1 if self.input_mode == "multimodal" else 0
        if len(captures["vision"]) != expected_vision or len(captures["mid"]) != 1 or len(captures["final"]) != 1:
            raise RuntimeError("BLOCKED: Cambrian formal-adapter taps did not each fire exactly once")
        result = {
            "llm_mid_last_prompt": _last_valid(captures["mid"][0], inputs["attention_mask"]),
            "llm_final_norm_last_prompt": _last_valid(captures["final"][0], inputs["attention_mask"]),
        }
        if self.input_mode == "multimodal":
            visual = captures["vision"][0]
            if visual.ndim == 3:
                visual = visual[0]
            if visual.ndim != 2:
                raise RuntimeError(f"BLOCKED: unexpected Cambrian vision-terminal shape {tuple(visual.shape)}")
            result["vision_terminal"] = visual.detach().float().mean(dim=0).cpu()
        return result


def _extractor(args: argparse.Namespace):
    if args.backend == "qwen2_5_vl":
        return QwenExtractor(args.checkpoint, args.device, args.dtype, args.input_mode)
    if args.backend == "cambrians":
        return CambrianExtractor(args.checkpoint, args.device, args.dtype, args.vagen_root, args.input_mode)
    raise ValueError(f"Unsupported backend: {args.backend}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-bank", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backend", choices=("qwen2_5_vl", "cambrians"), required=True)
    parser.add_argument("--vagen-root", type=Path, default=Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--input-mode", choices=("multimodal", "text_only"), default="multimodal")
    args = parser.parse_args()
    check = validate_state_bank(args.state_bank)
    if not check["valid"]:
        raise SystemExit(f"BLOCKED: invalid state bank: {check['errors'][:3]}")
    if not args.checkpoint.is_dir():
        raise SystemExit(f"BLOCKED: exact checkpoint path unavailable: {args.checkpoint}")
    torch, image_cls, _ = _requirements()
    extractor = _extractor(args)
    rows = list(read_jsonl(args.state_bank))
    if args.limit is not None:
        rows = rows[:args.limit]
    sample_ids: list[str] = []
    data: dict[str, list[Any]] = {}
    for index, row in enumerate(rows):
        result = extractor.extract(_load_images(row, args.state_bank, image_cls), _task_text(row))
        sample_ids.append(str(row["sample_id"]))
        for tap, value in result.items():
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"BLOCKED: NaN/Inf in {tap} for {row['sample_id']}")
            data.setdefault(tap, []).append(value)
        print(json.dumps({"status": "OK", "index": index, "sample_id": row["sample_id"]}))
    payload = {"sample_ids": sample_ids, "taps": {tap: torch.stack(values) for tap, values in data.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    manifest = {
        "status": "COMPLETE", "checkpoint": str(args.checkpoint), "backend": args.backend, "input_state_bank": str(args.state_bank),
        "input_state_bank_validation": check, "sample_count": len(sample_ids), "sample_id_hash": stable_hash(sample_ids),
        "tap_dimensions": {tap: list(tensor.shape) for tap, tensor in payload["taps"].items()},
        "unified_system_prompt": UNIFIED_SYSTEM_PROMPT,
        "unified_input_hash": stable_hash({"system": UNIFIED_SYSTEM_PROMPT, "history_window": 3, "include_actions": False}),
        "input_mode": args.input_mode,
        "generation": "disabled", "gradients": "disabled", "output": str(args.output),
    }
    write_json(args.output.with_suffix(args.output.suffix + ".manifest.json"), manifest)


if __name__ == "__main__":
    main()
