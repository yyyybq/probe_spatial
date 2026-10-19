#!/usr/bin/env python3
"""Extract raw-ID exact DINO L0 units for an independent p8 study.

The unit manifest distinguishes an ordered p8 context from an isolated frame.
Every output is written once with an adjacent provenance sidecar; a partial
feature without a valid sidecar is retained and causes fail-closed resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_raw_id(value: str) -> tuple[str, str]:
    if not value.startswith("scannetpp:"):
        raise ValueError(f"not a ScanNet++ raw ID: {value}")
    scene_and_frame = value.removeprefix("scannetpp:")
    scene, slash, frame = scene_and_frame.partition("/")
    if not slash or not scene or not frame or "/" in frame:
        raise ValueError(f"malformed raw ID: {value}")
    return scene, frame


def load_image(data_root: Path, raw: str) -> Any:
    # Kept local so a request-only preflight does not import model runtimes.
    from PIL import Image

    scene, frame = parse_raw_id(raw)
    path = data_root / scene / "images" / frame
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        return image.convert("RGB").resize((728, 420), Image.Resampling.BICUBIC)


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
            # SCO workers map to a squashed NFS identity.  Receipts must remain
            # readable by the submitting user for resume and audit.
            os.fchmod(handle.fileno(), 0o644)
        os.link(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def sidecar_matches(feature: Path, unit: dict[str, Any]) -> bool:
    sidecar = Path(f"{feature}.manifest.json")
    if feature.is_symlink() or sidecar.is_symlink() or not feature.is_file() or not sidecar.is_file():
        return False
    try:
        meta = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    expected = {
        "schema_version": 1,
        "unit_id": unit["unit_id"],
        "model": "dino",
        "layer": 0,
        "timestep": None,
        "feature_role": unit["feature_role"],
        "raw_frame_id": unit["raw_frame_id"],
        "context_raw_ids": unit["context_raw_ids"],
        "source_manifest_sha256": unit["source_manifest_sha256"],
    }
    if any(meta.get(key) != value for key, value in expected.items()):
        return False
    return feature.stat().st_size == meta.get("size_bytes") and sha256(feature) == meta.get("feature_sha256")


def output_path(root: Path, unit: dict[str, Any]) -> Path:
    scene, _ = parse_raw_id(str(unit["raw_frame_id"]))
    return root / "dino" / "scannetpp" / scene / f"{unit['unit_id']}.safetensors"


def unit_inputs(unit: dict[str, Any]) -> list[str]:
    if unit["feature_role"] == "streaming_context_p8":
        values = unit["context_raw_ids"]
        if not isinstance(values, list) or len(values) != 8 or values[-1] != unit["raw_frame_id"]:
            raise ValueError(f"invalid p8 unit {unit['unit_id']}")
        return [str(value) for value in values]
    if unit["feature_role"] == "isolated_frame" and unit["context_raw_ids"] is None:
        return [str(unit["raw_frame_id"])]
    raise ValueError(f"unsupported unit role {unit['feature_role']}")


def read_units(path: Path, max_units: int | None) -> list[dict[str, Any]]:
    values = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    units = [row for row in values if row.get("model") == "dino" and int(row.get("layer", -1)) == 0 and row.get("timestep") is None]
    if not units:
        raise ValueError("no DINO L0 units in request manifest")
    if len({row["unit_id"] for row in units}) != len(units):
        raise ValueError("duplicate DINO unit ID")
    return units[:max_units] if max_units is not None else units


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--science-tag", required=True)
    parser.add_argument("--request-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-id", default="facebook/dinov2-large")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--model-source", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-units", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.science_tag.startswith("independent_actionability_algebra_p8_"):
        raise SystemExit("unexpected science tag")
    if args.batch_size < 1 or args.max_units is not None and args.max_units < 1:
        raise SystemExit("batch-size/max-units must be positive")
    request = args.request_manifest.resolve()
    output_root = args.output_root.resolve()
    units = read_units(request, args.max_units)
    request_sha = sha256(request)
    source_sha = sha256(Path(__file__).resolve())
    for unit in units:
        if unit.get("exact_raw_id_required") is not True or unit.get("fallback_forbidden") is not True:
            raise SystemExit(f"unit {unit['unit_id']} has weakened identity policy")
        inputs = unit_inputs(unit)
        if parse_raw_id(str(unit["raw_frame_id"]))[0] != unit["scene_id"]:
            raise SystemExit(f"unit {unit['unit_id']} scene/raw-ID mismatch")
        for raw in inputs:
            if parse_raw_id(raw)[0] != unit["scene_id"]:
                raise SystemExit(f"unit {unit['unit_id']} cross-scene context")
            if not (args.data_root / unit["scene_id"] / "images" / parse_raw_id(raw)[1]).is_file():
                raise SystemExit(f"unit {unit['unit_id']} missing raw image: {raw}")
    if args.dry_run:
        print(json.dumps({"status": "COMPLETE_VALID", "mode": "dry_run", "units": len(units), "request_sha256": request_sha}, indent=2))
        return 0
    import torch
    from safetensors.torch import save_file
    from transformers import AutoImageProcessor, Dinov2Model
    from features.dino.extract_features import dino_forward, reshape_tokens

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for DINO extraction but unavailable")
    model_source = str(args.model_source.resolve()) if args.model_source is not None else args.model_id
    if args.model_source is not None and not args.model_source.is_dir():
        raise SystemExit(f"model source is not a directory: {args.model_source}")
    load_options = {"local_files_only": args.local_files_only}
    processor = AutoImageProcessor.from_pretrained(model_source, revision=args.model_revision, **load_options)
    model = Dinov2Model.from_pretrained(model_source, revision=args.model_revision, **load_options).to(args.device).eval()
    resolved_revision = getattr(model.config, "_commit_hash", None) or args.model_revision
    processor_config = processor.to_dict() if hasattr(processor, "to_dict") else {"class": type(processor).__name__}
    created = skipped = 0
    for unit in units:
        destination = output_path(output_root, unit)
        sidecar = Path(f"{destination}.manifest.json")
        if destination.exists() or sidecar.exists():
            if sidecar_matches(destination, unit):
                skipped += 1
                continue
            raise RuntimeError(f"incomplete or mismatched existing output; preserve for review: {destination}")
        raw_inputs = unit_inputs(unit)
        images = [load_image(args.data_root, raw) for raw in raw_inputs]
        tokens = []
        with torch.no_grad():
            for start in range(0, len(images), args.batch_size):
                tokens.append(dino_forward(model, processor, images[start : start + args.batch_size], args.device))
        feature = torch.stack([reshape_tokens(item) for item in torch.cat(tokens, dim=0)], dim=0).half().contiguous().cpu()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        save_file({"feat": feature}, str(temporary))
        feature_sha = sha256(temporary)
        os.link(temporary, destination)
        temporary.unlink()
        atomic_json_create(
            sidecar,
            {
                "schema_version": 1,
                "science_tag": args.science_tag,
                "unit_id": unit["unit_id"],
                "model": "dino",
                "model_id": args.model_id,
                "model_source": model_source,
                "local_files_only": args.local_files_only,
                "model_revision": resolved_revision,
                "processor_config": processor_config,
                "layer": 0,
                "timestep": None,
                "feature_role": unit["feature_role"],
                "requested_roles": unit["requested_roles"],
                "scene_id": unit["scene_id"],
                "raw_frame_id": unit["raw_frame_id"],
                "context_raw_ids": unit["context_raw_ids"],
                "actual_input_raw_ids": raw_inputs,
                "source_manifest_sha256": unit["source_manifest_sha256"],
                "request_manifest_sha256": request_sha,
                "extractor": str(Path(__file__).resolve()),
                "extractor_sha256": source_sha,
                "file": destination.name,
                "shape": list(feature.shape),
                "dtype": str(feature.dtype),
                "size_bytes": destination.stat().st_size,
                "feature_sha256": feature_sha,
                "fallback": False,
            },
        )
        created += 1
    print(json.dumps({"status": "COMPLETE_VALID", "created": created, "skipped": skipped, "units": len(units), "output_root": str(output_root), "request_sha256": request_sha}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
