#!/usr/bin/env python3
"""Extract Wan streaming/target features from exact manifest frame ids.

This recovery extractor deliberately has no temporal-window generator. Every
frame id comes from the frozen canonical JSONL and is persisted in output
metadata so alignment can be audited independently of file names.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file as save_safetensors

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CANONICAL_SHA256 = "772194f077c45cc85a942f17f46f50103125ae8f5b1c983c08881c1759607e0c"
DEFAULT_LAYERS = [0, 5, 10, 15, 25, 29]
INPUT_FRAMES = 81
RESIZE = (480, 832)
TARGET_FIELDS = ("target_raw_id", "target_raw_index", "target_frame_idx", "target_index")

logging.basicConfig(
    level=logging.INFO,
    format="{asctime}: [{levelname}] {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
)
LOG = logging.getLogger("wan_manifest_exact")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reshape_to_t_h_w_c(raw: torch.Tensor) -> torch.Tensor:
    expected = 21 * 30 * 52
    if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[1] != expected:
        raise ValueError(f"Unexpected Wan feature shape: {tuple(raw.shape)}")
    return raw.squeeze(0).reshape(21, 30, 52, raw.shape[-1]).contiguous()


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_selection(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    selected = {
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not selected:
        raise ValueError(f"Selection file is empty: {path}")
    return selected


def target_raw_id(row: dict[str, Any], horizon: int) -> tuple[int, str]:
    for field in TARGET_FIELDS:
        if row.get(field) is not None:
            return int(row[field]), field
    tail = int(row["streaming_tail"])
    indices = [int(value) for value in row["streaming_window_indices"]]
    position = tail + int(horizon)
    if position >= len(indices):
        raise IndexError(
            f"{row.get('sample_id')}: target position {position} outside "
            f"streaming_window_indices length {len(indices)}"
        )
    return int(indices[position]), f"streaming_window_indices[{position}]"


def scene_id(row: dict[str, Any]) -> str:
    return str(row.get("scene_id") or Path(str(row["scene_dir"])).name)


def frame_names(row: dict[str, Any]) -> list[str]:
    names = row.get("valid_frames")
    if not isinstance(names, list) or not names:
        raise ValueError(f"{row.get('sample_id')}: canonical valid_frames is missing")
    return [str(name) for name in names]


def local_scene_dir(data_root: Path, row: dict[str, Any]) -> Path:
    return data_root / "processed_scannetpp_v2" / "processed_scannetpp_v2" / scene_id(row)


def load_frame(data_root: Path, row: dict[str, Any], raw_id: int) -> Image.Image:
    names = frame_names(row)
    if raw_id < 0 or raw_id >= len(names):
        raise IndexError(f"{row.get('sample_id')}: raw frame {raw_id} outside 0..{len(names)-1}")
    path = local_scene_dir(data_root, row) / "images" / names[raw_id]
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        return image.convert("RGB").resize((RESIZE[1], RESIZE[0]), Image.Resampling.LANCZOS)


def deterministic_seed(base_seed: int, key: str) -> int:
    payload = hashlib.sha256(f"{base_seed}:{key}".encode()).digest()
    return int.from_bytes(payload[:4], "big") & 0x7FFFFFFF


def set_seed(base_seed: int, key: str) -> int:
    seed = deterministic_seed(base_seed, key)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def git_state(repo: Path) -> tuple[str, bool | None]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, timeout=15
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(repo), "status", "--short"], text=True, timeout=15
            ).strip()
        )
        return commit, dirty
    except Exception:
        return "unknown", None


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def atomic_npy(path: Path, array: np.ndarray, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("wb") as handle:
        np.save(handle, array)
    digest = sha256(tmp)
    os.replace(tmp, path)
    atomic_json(
        Path(f"{path}.manifest.json"),
        {
            **metadata,
            "file": path.name,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "size_bytes": path.stat().st_size,
            "sha256": digest,
        },
    )


def atomic_safetensors(path: Path, tensor: torch.Tensor, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    value = tensor.half().contiguous().cpu()
    save_safetensors({"feat": value}, str(tmp))
    digest = sha256(tmp)
    os.replace(tmp, path)
    atomic_json(
        Path(f"{path}.manifest.json"),
        {
            **metadata,
            "file": path.name,
            "tensors": {"feat": {"shape": list(value.shape), "dtype": str(value.dtype)}},
            "size_bytes": path.stat().st_size,
            "sha256": digest,
        },
    )


def feature_path(root: Path, scene: str, window: int | None, tail: int | None, layer: int) -> Path:
    base = root / "wan" / "scannetpp" / scene
    if window is not None and tail is not None:
        base = base / f"window_{window:04d}" / f"prefix_{tail:06d}"
    return base / f"feature_t749_layer{layer}.sft"


def sidecar_matches(path: Path, expected: dict[str, Any]) -> bool:
    sidecar = Path(f"{path}.manifest.json")
    if not path.is_file() or not sidecar.is_file():
        return False
    try:
        meta = json.loads(sidecar.read_text())
    except Exception:
        return False
    return path.stat().st_size == meta.get("size_bytes") and all(meta.get(k) == v for k, v in expected.items())


def forward_streaming(model: Any, data_root: Path, row: dict[str, Any], layers: list[int], seed: int) -> dict[int, torch.Tensor]:
    actual = [int(value) for value in row["streaming_indices"]]
    canonical_prefix = [int(value) for value in row["streaming_window_indices"]][: int(row["streaming_tail"]) + 1]
    if actual != canonical_prefix:
        raise ValueError(f"{row['sample_id']}: streaming_indices is not the canonical window prefix")
    frames = [load_frame(data_root, row, raw_id) for raw_id in actual]
    valid_length = len(frames)
    if not frames or valid_length > INPUT_FRAMES:
        raise ValueError(f"{row['sample_id']}: invalid streaming input length {valid_length}")
    frames.extend([frames[-1]] * (INPUT_FRAMES - valid_length))
    set_seed(seed, f"streaming:{row['sample_id']}")
    with torch.no_grad():
        raw = model.forward(
            video=frames,
            prompt=".",
            t=749,
            output_layer_indices=layers,
            ensemble_size=1,
        )
    return {layer: reshape_to_t_h_w_c(raw[layer]) for layer in layers}


def forward_target(model: Any, image: Image.Image, layers: list[int], seed: int, key: str) -> dict[int, torch.Tensor]:
    set_seed(seed, key)
    with torch.no_grad():
        raw = model.forward(
            video=[image] * INPUT_FRAMES,
            prompt=".",
            t=749,
            output_layer_indices=layers,
            ensemble_size=1,
        )
    result = {}
    for layer in layers:
        temporal = reshape_to_t_h_w_c(raw[layer])
        result[layer] = temporal[temporal.shape[0] // 2]
    return result


def common_metadata(args: argparse.Namespace, manifest_sha: str, commit: str, dirty: bool | None) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "frame_index_schema": "canonical_manifest_exact_v1",
        "mode": f"manifest_exact_{args.mode}",
        "canonical_manifest": str(Path(args.manifest).resolve()),
        "canonical_manifest_sha256": manifest_sha,
        "model_id": str(Path(args.model_id).resolve()),
        "prompt": ".",
        "timestep": 749,
        "layers": args.layers,
        "num_frames": INPUT_FRAMES,
        "resize": list(RESIZE),
        "base_seed": args.seed,
        "seed_scheme": "sha256(base_seed:mode:key)[:31bit]",
        "extractor": str(Path(__file__).resolve()),
        "extractor_sha256": sha256(Path(__file__).resolve()),
        "git_commit": commit,
        "git_dirty": dirty,
        "motion_window_generator_used": False,
    }


def process_streaming(
    args: argparse.Namespace,
    model: Any,
    rows_by_scene: dict[str, list[dict[str, Any]]],
    common: dict[str, Any],
) -> None:
    for scene_number, (scene, rows) in enumerate(rows_by_scene.items(), 1):
        started = time.monotonic()
        records = []
        for row in rows:
            window = int(row["streaming_window_id"])
            tail = int(row["streaming_tail"])
            actual = [int(value) for value in row["streaming_indices"]]
            expected = {
                "canonical_manifest_sha256": common["canonical_manifest_sha256"],
                "sample_id": row["sample_id"],
                "actual_input_raw_ids": actual,
            }
            paths = {layer: feature_path(Path(args.out_root), scene, window, tail, layer) for layer in args.layers}
            missing = [layer for layer, path in paths.items() if not sidecar_matches(path, expected)]
            if missing:
                if any(path.exists() or Path(f"{path}.manifest.json").exists() for path in paths.values()):
                    raise RuntimeError(f"Refusing to overwrite incomplete streaming cache for {row['sample_id']}")
                outputs = forward_streaming(model, Path(args.data_root), row, args.layers, args.seed)
                for layer in args.layers:
                    atomic_safetensors(
                        paths[layer],
                        outputs[layer],
                        {
                            **common,
                            "sample_id": row["sample_id"],
                            "split": row["split"],
                            "scene_id": scene,
                            "window_id": window,
                            "tail": tail,
                            "actual_input_raw_ids": actual,
                            "canonical_window_raw_ids": [int(value) for value in row["streaming_window_indices"]],
                            "actual_input_frame_names": [frame_names(row)[idx] for idx in actual],
                            "sample_seed": deterministic_seed(args.seed, f"streaming:{row['sample_id']}"),
                        },
                    )
            records.append(
                {
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "window_id": window,
                    "tail": tail,
                    "indices": actual,
                    "actual_input_raw_ids": actual,
                    "window_indices": [int(value) for value in row["streaming_window_indices"]],
                    "canonical_window_raw_ids": [int(value) for value in row["streaming_window_indices"]],
                    "input_length": INPUT_FRAMES,
                    "valid_length": len(actual),
                    "pad_mode": "repeat_tail",
                    "sampling": "canonical_manifest_exact",
                    "motion_window_generator_used": False,
                }
            )
        meta_path = Path(args.out_root) / "wan" / "scannetpp" / scene / "prefix_index.npy"
        atomic_npy(meta_path, np.array(records, dtype=object), common)
        LOG.info("[%d/%d] streaming %s rows=%d %.1fs", scene_number, len(rows_by_scene), scene, len(rows), time.monotonic() - started)


def process_target(
    args: argparse.Namespace,
    model: Any,
    rows_by_scene: dict[str, list[dict[str, Any]]],
    common: dict[str, Any],
) -> None:
    for scene_number, (scene, rows) in enumerate(rows_by_scene.items(), 1):
        started = time.monotonic()
        target_to_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
        sources: dict[int, str] = {}
        for row in rows:
            raw_id, source = target_raw_id(row, args.target_horizon)
            target_to_rows[raw_id].append(row)
            sources[raw_id] = source
        target_ids = sorted(target_to_rows)
        scene_root = Path(args.out_root) / "wan" / "scannetpp" / scene
        index_path = scene_root / "target_indices.npy"
        if index_path.exists():
            cached = np.load(index_path).astype(np.int64).tolist()
            if cached != target_ids:
                raise RuntimeError(f"Refusing to overwrite target index mismatch for {scene}: {cached} != {target_ids}")
        expected = {
            "canonical_manifest_sha256": common["canonical_manifest_sha256"],
            "target_raw_ids": target_ids,
        }
        paths = {layer: feature_path(Path(args.out_root), scene, None, None, layer) for layer in args.layers}
        missing = [layer for layer, path in paths.items() if not sidecar_matches(path, expected)]
        if missing:
            if any(path.exists() or Path(f"{path}.manifest.json").exists() for path in paths.values()):
                raise RuntimeError(f"Refusing to overwrite incomplete target cache for {scene}")
            collected: dict[int, list[torch.Tensor]] = {layer: [] for layer in args.layers}
            for raw_id in target_ids:
                representative = target_to_rows[raw_id][0]
                image = load_frame(Path(args.data_root), representative, raw_id)
                outputs = forward_target(model, image, args.layers, args.seed, f"target:{scene}:{raw_id}")
                for layer in args.layers:
                    collected[layer].append(outputs[layer])
            sample_ids = sorted({row["sample_id"] for row in rows})
            for layer in args.layers:
                atomic_safetensors(
                    paths[layer],
                    torch.stack(collected[layer]),
                    {
                        **common,
                        "scene_id": scene,
                        "sample_ids": sample_ids,
                        "target_raw_ids": target_ids,
                        "target_frame_names": [frame_names(target_to_rows[idx][0])[idx] for idx in target_ids],
                        "target_id_sources": {str(idx): sources[idx] for idx in target_ids},
                        "sample_seeds": {
                            str(idx): deterministic_seed(args.seed, f"target:{scene}:{idx}")
                            for idx in target_ids
                        },
                    },
                )
        atomic_npy(
            index_path,
            np.array(target_ids, dtype=np.int64),
            {
                **common,
                "scene_id": scene,
                "target_raw_ids": target_ids,
                "sample_ids": sorted({row["sample_id"] for row in rows}),
            },
        )
        LOG.info("[%d/%d] target %s targets=%d %.1fs", scene_number, len(rows_by_scene), scene, len(target_ids), time.monotonic() - started)


def ordered_scene_groups(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[scene_id(row)].append(row)
    return {scene: grouped[scene] for scene in sorted(grouped)}


def validate_rows(rows: list[dict[str, Any]], target_horizon: int) -> None:
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Manifest selection contains duplicate sample IDs")
    for row in rows:
        tail = int(row["streaming_tail"])
        window = [int(value) for value in row["streaming_window_indices"]]
        prefix = [int(value) for value in row["streaming_indices"]]
        names = frame_names(row)
        if prefix != window[: tail + 1]:
            raise ValueError(f"{row['sample_id']}: streaming_indices does not match canonical window prefix")
        if any(raw_id < 0 or raw_id >= len(names) for raw_id in window):
            raise ValueError(f"{row['sample_id']}: canonical window raw id is out of range")
        target_id, _ = target_raw_id(row, target_horizon)
        if target_id < 0 or target_id >= len(names):
            raise ValueError(f"{row['sample_id']}: target raw id {target_id} is out of range")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("streaming", "target"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--selection-file")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--layers", nargs="+", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--target-horizon", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    manifest = Path(args.manifest).resolve()
    manifest_sha = sha256(manifest)
    if manifest_sha != CANONICAL_SHA256:
        raise SystemExit(f"Canonical manifest hash mismatch: {manifest_sha}")
    if sorted(set(args.layers)) != DEFAULT_LAYERS:
        raise SystemExit(f"Layers must be exactly {DEFAULT_LAYERS}, got {args.layers}")
    args.layers = DEFAULT_LAYERS

    rows = read_rows(manifest)
    if len(rows) != 2724:
        raise SystemExit(f"Expected 2724 canonical rows, got {len(rows)}")
    selected = read_selection(Path(args.selection_file).resolve() if args.selection_file else None)
    if selected is not None:
        known = {row["sample_id"] for row in rows}
        missing = selected - known
        if missing:
            raise SystemExit(f"Selection contains unknown sample IDs: {sorted(missing)[:3]}")
        rows = [row for row in rows if row["sample_id"] in selected]
        if len(rows) != len(selected):
            raise SystemExit("Selection contains duplicate or unresolved sample IDs")

    validate_rows(rows, args.target_horizon)

    groups = ordered_scene_groups(rows)
    items = list(groups.items())[args.start : args.end]
    groups = dict(items)
    if not groups:
        raise SystemExit("Scene shard is empty")

    repo = Path(__file__).resolve().parents[1]
    commit, dirty = git_state(repo)
    common = common_metadata(args, manifest_sha, commit, dirty)
    LOG.info(
        "mode=%s rows=%d scenes=%d shard=[%s:%s] layers=%s out=%s",
        args.mode,
        sum(len(value) for value in groups.values()),
        len(groups),
        args.start,
        args.end,
        args.layers,
        args.out_root,
    )
    if args.validate_only:
        LOG.info("Validation-only complete; no model loaded and no output written")
        return
    from features.wan.wan_feature import get_wan_featurizer

    model = get_wan_featurizer(model_id=args.model_id, null_prompt=".")
    LOG.info("Wan model loaded")
    if args.mode == "streaming":
        process_streaming(args, model, groups, common)
    else:
        process_target(args, model, groups, common)
    LOG.info("Done mode=%s scenes=%d", args.mode, len(groups))


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
