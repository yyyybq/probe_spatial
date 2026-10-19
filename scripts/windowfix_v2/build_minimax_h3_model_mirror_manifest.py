#!/usr/bin/env python3
"""Build a content-addressed manifest for a recovered MiniMax-H3 checkpoint.

The manifest is deliberately separate from feature provenance.  It makes a
model mirror immutable and lets an SCO worker verify its local hydrated copy
without trusting a directory name or ``exists()`` alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


CONTRACT = "MINIMAX_H3_AOSS_MODEL_MIRROR_V1"
REPOSITORY = "MiniMaxAI/MiniMax-H3"
EXCLUDED_TOP_LEVEL = {".cache", ".git"}
REQUIRED = (
    "modular_model_index.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors.index.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors.index.json",
    "tokenizer/tokenizer.json",
    "scheduler/scheduler_config.json",
    "audio_scheduler/scheduler_config.json",
    "audio_vae/diffusion_pytorch_model.safetensors",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def require_regular(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"{label} must be a non-empty regular file: {path}")


def index_shards(root: Path, relative: str) -> list[str]:
    payload = json.loads((root / relative).read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise SystemExit(f"invalid safetensors index: {relative}")
    return sorted({str(Path(relative).parent / str(value)) for value in weight_map.values()})


def manifest_for(root: Path) -> dict[str, Any]:
    for relative in REQUIRED:
        require_regular(root / relative, f"required MiniMax-H3 asset {relative}")
    required_shards: set[str] = set()
    for relative in REQUIRED:
        if relative.endswith(".safetensors.index.json"):
            required_shards.update(index_shards(root, relative))
    for relative in sorted(required_shards):
        require_regular(root / relative, f"required indexed shard {relative}")

    assets: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] in EXCLUDED_TOP_LEVEL:
            continue
        if relative.name == "model_mirror_manifest.json":
            continue
        assets.append(
            {
                "path": relative.as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    if not assets:
        raise SystemExit("MiniMax-H3 mirror has no payload assets")
    result: dict[str, Any] = {
        "contract": CONTRACT,
        "repository": REPOSITORY,
        "assets": assets,
        "asset_count": len(assets),
        "total_bytes": sum(int(item["size"]) for item in assets),
        "required_paths": [*REQUIRED, *sorted(required_shards)],
    }
    result["content_sha256"] = canonical_sha(result)
    return result


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o444)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.model_root).resolve()
    if root.is_symlink() or not root.is_dir():
        raise SystemExit(f"model root must be a regular directory: {root}")
    output = Path(args.output)
    if output.is_symlink():
        raise SystemExit(f"refusing symlink manifest output: {output}")
    value = manifest_for(root)
    atomic_json(output, value)
    print(json.dumps({"output": str(output), **value}, sort_keys=True))


if __name__ == "__main__":
    main()
