#!/usr/bin/env python3
"""Extract pooled, exact-unit features for the frozen independent P8 protocol.

This is intentionally separate from the historical C/canonical-v2 stores. Each
unit is content-identified, fail-closed, and stored as one FP32 pooled vector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

MODEL_CONTRACTS = {
    "dino": {"layer": 0, "timestep": None, "input_frames": None, "width": 1024},
    "f3r": {"layer": 24, "timestep": None, "input_frames": 64, "width": 1024},
    "vjepa2": {"layer": 23, "timestep": None, "input_frames": 64, "width": 1024},
    "wan": {"layer": 20, "timestep": 749, "input_frames": 81, "width": 1536},
}


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_seed(model: str, revision: str, unit: dict[str, Any]) -> int:
    identity = {
        "scheme": "independent-p8-feature-seed-v1",
        "model": model,
        "revision": revision,
        "scene_id": unit["scene_id"],
        "raw_frame_id": unit["raw_frame_id"],
        "context_raw_ids": unit["context_raw_ids"],
    }
    return int.from_bytes(hashlib.sha256(canonical(identity).encode()).digest()[:8], "big") & ((1 << 63) - 1)


def parse_raw_id(value: str) -> tuple[str, str]:
    prefix = "scannetpp:"
    if not value.startswith(prefix):
        raise ValueError(f"not a ScanNet++ raw ID: {value}")
    scene, separator, frame = value[len(prefix):].partition("/")
    if not separator or not scene or not frame or "/" in frame:
        raise ValueError(f"malformed raw ID: {value}")
    return scene, frame


def requested_raw_ids(unit: dict[str, Any]) -> list[str]:
    if unit["feature_role"] == "streaming_context_p8":
        values = unit["context_raw_ids"]
        if not isinstance(values, list) or len(values) != 8 or values[-1] != unit["raw_frame_id"]:
            raise ValueError(f"invalid ordered p8 unit: {unit['unit_id']}")
        return [str(value) for value in values]
    if unit["feature_role"] == "isolated_frame" and unit["context_raw_ids"] is None:
        return [str(unit["raw_frame_id"])]
    raise ValueError(f"unsupported feature role: {unit['feature_role']}")


def expanded_raw_ids(model: str, unit: dict[str, Any]) -> list[str]:
    values = requested_raw_ids(unit)
    target = MODEL_CONTRACTS[model]["input_frames"]
    if target is None:
        return values
    if len(values) > int(target):
        raise ValueError("logical input exceeds fixed model input")
    return values + [values[-1]] * (int(target) - len(values))


def image_path(data_root: Path, raw_id: str) -> Path:
    scene, frame = parse_raw_id(raw_id)
    path = data_root / scene / "images" / frame
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    return path


def load_pil(path: Path, resize: tuple[int, int] | None = None):
    from PIL import Image

    with Image.open(path) as image:
        value = image.convert("RGB")
        if resize is not None:
            value = value.resize(resize, Image.Resampling.BICUBIC)
        return value


def output_path(root: Path, model: str, unit: dict[str, Any]) -> Path:
    return root / model / "scannetpp" / unit["scene_id"] / f"{unit['unit_id']}.safetensors"


def atomic_json_create(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o644)
        os.link(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def extraction_contract(args: argparse.Namespace, model_tree_sha: str) -> dict[str, Any]:
    contract = MODEL_CONTRACTS[args.model]
    return {
        "schema_version": 1,
        "study_protocol_sha256": args.protocol_sha256,
        "model": args.model,
        "logical_model_id": args.model_id,
        "model_revision": args.model_revision,
        "snapshot_tree_sha256": model_tree_sha,
        "layer": contract["layer"],
        "timestep": contract["timestep"],
        "input_frames": contract["input_frames"],
        "context_padding": "none" if args.model == "dino" else "repeat_tail",
        "isolated_padding": "none" if args.model == "dino" else "repeat_single_frame",
        "pooling": "FP32 mean over every returned temporal and spatial token",
        "stored_dtype": "torch.float32",
        "stored_width": contract["width"],
        "dino_source_resize_wh": [728, 420] if args.model == "dino" else None,
        "f3r_source_resize_wh": [512, 288] if args.model == "f3r" else None,
        "vjepa2_processor_size": [256, 256] if args.model == "vjepa2" else None,
        "wan_source_resize_wh": [832, 480] if args.model == "wan" else None,
        "noise_seed_scheme": "independent-p8-feature-seed-v1" if args.model in {"f3r", "wan"} else "none",
    }


def tree_sha256(root: Path) -> str:
    """Hash snapshot paths, resolved blob identities and sizes without rereading LFS blobs."""
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        resolved = path.resolve()
        rows.append({
            "path": str(path.relative_to(root)),
            "resolved_name": resolved.name,
            "size": resolved.stat().st_size,
        })
    if not rows:
        raise ValueError(f"empty model snapshot: {root}")
    return hashlib.sha256(canonical(rows).encode()).hexdigest()


def select_units(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    contract = MODEL_CONTRACTS[args.model]
    values = [
        row for row in rows
        if row.get("model") == args.model
        and row.get("layer") == contract["layer"]
        and row.get("timestep") == contract["timestep"]
    ]
    values.sort(key=lambda row: row["unit_id"])
    if len({row["unit_id"] for row in values}) != len(values):
        raise ValueError("duplicate unit ID")
    if args.smoke_balanced:
        isolated = [row for row in values if row["feature_role"] == "isolated_frame"][:2]
        contexts = [row for row in values if row["feature_role"] == "streaming_context_p8"][:2]
        values = isolated + contexts
    if args.num_shards > 1:
        values = [row for index, row in enumerate(values) if index % args.num_shards == args.shard_index]
    if args.max_units is not None:
        values = values[: args.max_units]
    if not values:
        raise ValueError("no requested units after filtering")
    return values


class Backend:
    def __init__(self, args: argparse.Namespace) -> None:
        import torch

        self.args = args
        self.torch = torch
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA required for backbone extraction but unavailable")
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        source = str(args.model_source.resolve())
        if args.model == "dino":
            from transformers import AutoImageProcessor, Dinov2Model

            self.processor = AutoImageProcessor.from_pretrained(source, local_files_only=True)
            self.backend = Dinov2Model.from_pretrained(source, local_files_only=True).to(args.device).eval()
        elif args.model == "f3r":
            from features.f3r.extract_features import load_model

            self.backend = load_model(source, args.device)
        elif args.model == "vjepa2":
            from transformers import AutoModel, AutoVideoProcessor

            self.processor = AutoVideoProcessor.from_pretrained(source, local_files_only=True)
            self.backend = AutoModel.from_pretrained(
                source, local_files_only=True, torch_dtype=torch.float16, attn_implementation="sdpa"
            ).to(args.device).eval()
        elif args.model == "wan":
            from features.wan.wan_feature import get_wan_featurizer

            self.backend = get_wan_featurizer(model_id=source, null_prompt="")
        else:  # pragma: no cover
            raise ValueError(args.model)

    def extract(self, paths: list[Path], seed: int) -> tuple[Any, list[int], dict[str, Any]]:
        torch = self.torch
        model = self.args.model
        if model == "dino":
            from features.dino.extract_features import dino_forward, reshape_tokens

            images = [load_pil(path, (728, 420)) for path in paths]
            pieces = []
            with torch.no_grad():
                for start in range(0, len(images), self.args.batch_size):
                    output = dino_forward(self.backend, self.processor, images[start:start + self.args.batch_size], self.args.device)
                    pieces.extend(reshape_tokens(row).float().cpu() for row in output)
            raw = torch.stack(pieces)
            return raw.mean(dim=(0, 1, 2)), list(raw.shape), {"processor_config": self.processor.to_dict()}

        if model == "f3r":
            from features.f3r.extract_features import C, HTOK, WTOK, vidfm3d_forward

            generator = torch.Generator(device="cpu").manual_seed(seed)
            device = next(self.backend.parameters()).device
            with torch.no_grad():
                output = vidfm3d_forward([str(path) for path in paths], self.backend, device, random_generator=generator)
            if isinstance(output, torch.Tensor):
                output = [output]
            raw = output[24].reshape(len(paths), HTOK, WTOK, C).float().cpu()
            return raw.mean(dim=(0, 1, 2)), list(raw.shape), {"random_generator_seed": seed}

        if model == "vjepa2":
            import numpy as np

            images = [load_pil(path) for path in paths]
            array = np.stack([np.asarray(image) for image in images])
            inputs = self.processor(array, return_tensors="pt")
            inputs = {key: value.to(self.args.device) for key, value in inputs.items()}
            with torch.no_grad():
                output = self.backend(**inputs, output_hidden_states=True, skip_predictor=True)
            raw = output.hidden_states[24][0].float().cpu()
            cfg = self.backend.config
            shape = [
                int(cfg.frames_per_clip // cfg.tubelet_size),
                int(cfg.crop_size // cfg.patch_size),
                int(cfg.crop_size // cfg.patch_size),
                int(cfg.hidden_size),
            ]
            if raw.numel() != int(torch.tensor(shape).prod().item()):
                raise ValueError(f"V-JEPA2 token shape mismatch: {list(raw.shape)} vs {shape}")
            return raw.mean(dim=0), shape, {"processor_config": self.processor.to_dict()}

        if model == "wan":
            from features.wan.extract_features import reshape_to_t_h_w_c

            images = [load_pil(path, (832, 480)) for path in paths]
            generator = torch.Generator(device="cpu").manual_seed(seed)
            with torch.no_grad():
                output = self.backend.forward(
                    video=images, prompt="", t=749, output_layer_indices=[20],
                    ensemble_size=1, noise_generator=generator,
                )
            raw = reshape_to_t_h_w_c(output[20]).float().cpu()
            return raw.mean(dim=(0, 1, 2)), list(raw.shape), {"noise_seed": seed, "prompt": "", "ensemble": 1}
        raise ValueError(model)  # pragma: no cover


def reuse_dino(unit: dict[str, Any], root: Path):
    from safetensors.torch import load_file

    old = output_path(root, "dino", unit)
    sidecar = Path(f"{old}.manifest.json")
    if not old.is_file() or not sidecar.is_file() or old.is_symlink() or sidecar.is_symlink():
        return None
    meta = json.loads(sidecar.read_text())
    required = {
        "unit_id": unit["unit_id"], "model": "dino", "layer": 0,
        "timestep": None, "feature_role": unit["feature_role"],
        "raw_frame_id": unit["raw_frame_id"], "context_raw_ids": unit["context_raw_ids"],
        "source_manifest_sha256": unit["source_manifest_sha256"], "fallback": False,
    }
    if any(meta.get(key) != value for key, value in required.items()):
        return None
    if old.stat().st_size != meta.get("size_bytes") or sha256(old) != meta.get("feature_sha256"):
        return None
    tensor = load_file(str(old))["feat"].float()
    vector = tensor.reshape(-1, tensor.shape[-1]).mean(0).contiguous()
    return vector, list(tensor.shape), {
        "reused_verified_payload": str(old),
        "reused_payload_sha256": meta["feature_sha256"],
        "reused_sidecar_sha256": sha256(sidecar),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--science-tag", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--request-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", choices=tuple(MODEL_CONTRACTS), required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--reuse-dino-root", type=Path)
    parser.add_argument("--smoke-balanced", action="store_true")
    parser.add_argument("--max-units", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.science_tag.startswith("independent_actionability_algebra_p8_"):
        raise SystemExit("unexpected science tag")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("invalid shard")
    request = args.request_manifest.resolve()
    rows = [json.loads(line) for line in request.read_text().splitlines() if line.strip()]
    units = select_units(rows, args)
    request_sha = sha256(request)
    source = args.model_source.resolve()
    if not source.is_dir():
        raise SystemExit(f"local model snapshot missing: {source}")
    tree_sha = tree_sha256(source)
    contract = extraction_contract(args, tree_sha)
    contract_sha = hashlib.sha256(canonical(contract).encode()).hexdigest()
    extractor_sha = sha256(Path(__file__).resolve())
    dependencies = {
        str(path.relative_to(REPO)): sha256(path)
        for path in [
            REPO / "features/dino/extract_features.py",
            REPO / "features/f3r/extract_features.py",
            REPO / "features/f3r/f3r/dust3r/inference_multiview.py",
            REPO / "features/vjepa2/vjepa2_feature.py",
            REPO / "features/wan/wan_feature.py",
            REPO / "features/wan/extract_features.py",
        ]
        if path.is_file()
    }
    for unit in units:
        if unit.get("exact_raw_id_required") is not True or unit.get("fallback_forbidden") is not True:
            raise SystemExit(f"weakened identity policy: {unit['unit_id']}")
        logical = requested_raw_ids(unit)
        if any(parse_raw_id(raw)[0] != unit["scene_id"] for raw in logical):
            raise SystemExit(f"cross-scene unit: {unit['unit_id']}")
        for raw in set(logical):
            image_path(args.data_root, raw)
    if args.dry_run:
        print(json.dumps({"status": "COMPLETE_VALID", "units": len(units), "request_sha256": request_sha,
                          "contract_sha256": contract_sha, "snapshot_tree_sha256": tree_sha}, indent=2))
        return 0

    import torch
    from safetensors.torch import load_file, save_file

    backend = None
    created = reused = skipped = 0
    started = time.monotonic()
    for position, unit in enumerate(units, 1):
        destination = output_path(args.output_root.resolve(), args.model, unit)
        sidecar = Path(f"{destination}.manifest.json")
        if destination.exists() or sidecar.exists():
            if not destination.is_file() or not sidecar.is_file() or destination.is_symlink() or sidecar.is_symlink():
                raise RuntimeError(f"partial or symlink output: {destination}")
            meta = json.loads(sidecar.read_text())
            expected = {
                "unit_id": unit["unit_id"], "model": args.model,
                "request_manifest_sha256": request_sha, "extraction_contract_sha256": contract_sha,
                "extractor_sha256": extractor_sha, "feature_role": unit["feature_role"],
                "raw_frame_id": unit["raw_frame_id"], "context_raw_ids": unit["context_raw_ids"],
                "fallback": False,
            }
            if any(meta.get(key) != value for key, value in expected.items()):
                raise RuntimeError(f"mismatched existing output: {destination}")
            tensor = load_file(str(destination))["feat"]
            if destination.stat().st_size != meta["size_bytes"] or sha256(destination) != meta["feature_sha256"]:
                raise RuntimeError(f"corrupt existing output: {destination}")
            if list(tensor.shape) != [MODEL_CONTRACTS[args.model]["width"]] or not torch.isfinite(tensor).all():
                raise RuntimeError(f"invalid existing tensor: {destination}")
            skipped += 1
            continue

        result = None
        if args.model == "dino" and args.reuse_dino_root is not None:
            result = reuse_dino(unit, args.reuse_dino_root.resolve())
        if result is None:
            if backend is None:
                backend = Backend(args)
            expanded = expanded_raw_ids(args.model, unit)
            seed = content_seed(args.model, args.model_revision, unit)
            result = backend.extract([image_path(args.data_root, raw) for raw in expanded], seed)
            provenance = result[2]
            provenance["computed_in_final_extractor"] = True
            if args.device.startswith("cuda"):
                provenance["peak_cuda_memory_allocated_gib"] = backend.torch.cuda.max_memory_allocated() / (1024 ** 3)
            created += 1
        else:
            expanded = requested_raw_ids(unit)
            seed = None
            provenance = result[2]
            provenance["computed_in_final_extractor"] = False
            reused += 1
        vector, raw_shape, extra = result[0], result[1], provenance
        vector = vector.detach().float().cpu().contiguous()
        expected_width = MODEL_CONTRACTS[args.model]["width"]
        if list(vector.shape) != [expected_width] or not torch.isfinite(vector).all():
            raise RuntimeError(f"invalid pooled tensor for {unit['unit_id']}: {list(vector.shape)}")
        if float(vector.std().item()) <= 0.0 or float(vector.abs().max().item()) <= 0.0:
            raise RuntimeError(f"constant/zero pooled tensor for {unit['unit_id']}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        save_file({"feat": vector}, str(temporary))
        payload_sha = sha256(temporary)
        os.chmod(temporary, 0o644)
        os.link(temporary, destination)
        temporary.unlink()
        atomic_json_create(sidecar, {
            "schema_version": 1,
            "science_tag": args.science_tag,
            "study_protocol_sha256": args.protocol_sha256,
            "unit_id": unit["unit_id"],
            "model": args.model,
            "logical_model_id": args.model_id,
            "model_source": str(source),
            "model_revision": args.model_revision,
            "snapshot_tree_sha256": tree_sha,
            "layer": unit["layer"],
            "timestep": unit["timestep"],
            "feature_role": unit["feature_role"],
            "requested_roles": unit["requested_roles"],
            "scene_id": unit["scene_id"],
            "raw_frame_id": unit["raw_frame_id"],
            "context_raw_ids": unit["context_raw_ids"],
            "logical_input_raw_ids": requested_raw_ids(unit),
            "expanded_input_count": len(expanded_raw_ids(args.model, unit)),
            "padding_mode": contract["context_padding"] if unit["context_raw_ids"] else contract["isolated_padding"],
            "source_manifest_sha256": unit["source_manifest_sha256"],
            "request_manifest_sha256": request_sha,
            "extractor": str(Path(__file__).resolve()),
            "extractor_sha256": extractor_sha,
            "dependency_sha256": dependencies,
            "extraction_contract": contract,
            "extraction_contract_sha256": contract_sha,
            "raw_feature_shape": raw_shape,
            "pooling": contract["pooling"],
            "shape": list(vector.shape),
            "dtype": str(vector.dtype),
            "finite": True,
            "std": float(vector.std().item()),
            "max_abs": float(vector.abs().max().item()),
            "feature_sha256": payload_sha,
            "size_bytes": destination.stat().st_size,
            "fallback": False,
            "content_seed": seed,
            "provenance": extra,
        })
        if position % 25 == 0 or position == len(units):
            print(json.dumps({"progress": position, "units": len(units), "created": created,
                              "reused": reused, "skipped": skipped,
                              "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)
    print(json.dumps({"status": "COMPLETE_VALID", "model": args.model, "units": len(units),
                      "created": created, "reused": reused, "skipped": skipped,
                      "request_sha256": request_sha, "contract_sha256": contract_sha,
                      "snapshot_tree_sha256": tree_sha}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
