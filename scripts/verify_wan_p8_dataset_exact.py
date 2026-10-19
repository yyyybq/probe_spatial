#!/usr/bin/env python3
"""Run the real action_dynamics Dataset/DataLoader and verify target raw ids."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import numpy as np
from safetensors.torch import load_file
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vidfm3d.data.components.inscene15k_dataset import InsScene15KDataset


CANONICAL_SHA256 = "772194f077c45cc85a942f17f46f50103125ae8f5b1c983c08881c1759607e0c"
LAYERS = [0, 5, 10, 15, 25, 29]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_raw(row: dict[str, Any]) -> int:
    for field in ("target_raw_id", "target_raw_index", "target_frame_idx", "target_index"):
        if row.get(field) is not None:
            return int(row[field])
    return int(row["streaming_window_indices"][int(row["streaming_tail"]) + 1])


def scene_id(row: dict[str, Any]) -> str:
    return str(row.get("scene_id") or Path(str(row["scene_dir"])).name)


def verify_feature_mapping(
    row: dict[str, Any],
    layer: int,
    streaming_root: Path,
    target_root: Path,
    batch: dict[str, Any],
) -> list[dict[str, Any]]:
    problems = []
    scene = scene_id(row)
    window = int(row["streaming_window_id"])
    tail = int(row["streaming_tail"])
    stream_path = (
        streaming_root
        / "wan"
        / "scannetpp"
        / scene
        / f"window_{window:04d}"
        / f"prefix_{tail:06d}"
        / f"feature_t749_layer{layer}.sft"
    )
    stream_sidecar = Path(f"{stream_path}.manifest.json")
    try:
        stream_meta = json.loads(stream_sidecar.read_text())
        actual_input = [int(value) for value in stream_meta["actual_input_raw_ids"]]
    except Exception as exc:
        problems.append({"error": "streaming_sidecar", "path": str(stream_sidecar), "detail": repr(exc)})
    else:
        if actual_input != [int(value) for value in row["streaming_indices"]]:
            problems.append(
                {
                    "error": "streaming_actual_input_mismatch",
                    "expected": row["streaming_indices"],
                    "actual": actual_input,
                }
            )

    target_path = target_root / "wan" / "scannetpp" / scene / f"feature_t749_layer{layer}.sft"
    index_path = target_path.parent / "target_indices.npy"
    expected_raw = target_raw(row)
    try:
        indices = np.load(index_path).astype(np.int64)
        positions = np.flatnonzero(indices == expected_raw)
        if len(positions) != 1:
            raise ValueError(f"target raw id occurs {len(positions)} times")
        direct = load_file(str(target_path))["feat"][int(positions[0])].float()
        loaded = batch["target_feat"][0].float()
        if not torch.equal(direct, loaded):
            problems.append(
                {
                    "error": "dataset_target_tensor_mismatch",
                    "path": str(target_path),
                    "target_raw": expected_raw,
                    "max_abs_diff": float((direct - loaded).abs().max().item()),
                }
            )
    except Exception as exc:
        problems.append(
            {
                "error": "target_mapping_read",
                "path": str(target_path),
                "target_raw": expected_raw,
                "detail": repr(exc),
            }
        )
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--selection-file")
    parser.add_argument("--filtered-manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--streaming-root", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    manifest = Path(args.manifest).resolve()
    if sha256(manifest) != CANONICAL_SHA256:
        raise SystemExit("Canonical manifest hash mismatch")
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    selected = None
    if args.selection_file:
        selected = {
            line.strip()
            for line in Path(args.selection_file).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        rows = [row for row in rows if row["sample_id"] in selected]
        if len(rows) != len(selected):
            raise SystemExit("Canary selection did not resolve one-to-one")

    filtered = Path(args.filtered_manifest).resolve()
    filtered.parent.mkdir(parents=True, exist_ok=True)
    tmp = filtered.with_name(f".{filtered.name}.tmp-{os.getpid()}")
    if selected is None:
        shutil.copyfile(manifest, tmp)
    else:
        tmp.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))
    os.replace(tmp, filtered)

    result: dict[str, Any] = {
        "ok": True,
        "canonical_manifest": str(manifest),
        "canonical_manifest_sha256": sha256(manifest),
        "filtered_manifest": str(filtered),
        "filtered_manifest_sha256": sha256(filtered),
        "selected_rows": len(rows),
        "streaming_root": str(Path(args.streaming_root).resolve()),
        "target_root": str(Path(args.target_root).resolve()),
        "layers": {},
    }

    for layer in LAYERS:
        layer_problems = []
        checked = 0
        for split in ("train", "val"):
            expected = [row for row in rows if row["split"] == split]
            if not expected:
                continue
            dataset = InsScene15KDataset(
                root=args.data_root,
                root_vfm=args.streaming_root,
                streaming_feat_root=args.streaming_root,
                target_feat_root=args.target_root,
                scene_manifest=str(filtered),
                sources=["processed_scannetpp_v2"],
                split=split,
                vfm_name="wan",
                feature_layer=layer,
                feature_timestep=749,
                feat_pixalign=True,
                num_views=8,
                min_view_interval=5,
                context_len=8,
                query_idx_divisor=None,
                target_h=288,
                target_w=512,
                window_size=0,
                include_pmaps=False,
                streaming_prefix=True,
                prefix_min_len=8,
                prefix_max_len=8,
                diag_action=True,
                action_horizons=[1],
                allow_missing_vfm=False,
            )
            if len(dataset) != len(expected):
                layer_problems.append(
                    {"split": split, "error": "dataset_length", "expected": len(expected), "actual": len(dataset)}
                )
                continue
            loader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers,
                drop_last=False,
            )
            for row, batch in zip(expected, loader, strict=True):
                actual_target = int(batch["target_frame_idx"].item())
                expected_target = target_raw(row)
                valid = bool(batch["dyn_valid"].item())
                if actual_target != expected_target or not valid:
                    layer_problems.append(
                        {
                            "sample_id": row["sample_id"],
                            "split": split,
                            "expected_target_raw": expected_target,
                            "actual_target_raw": actual_target,
                            "dyn_valid": valid,
                        }
                    )
                mapping_problems = verify_feature_mapping(
                    row,
                    layer,
                    Path(args.streaming_root).resolve(),
                    Path(args.target_root).resolve(),
                    batch,
                )
                for problem in mapping_problems:
                    layer_problems.append(
                        {"sample_id": row["sample_id"], "split": split, **problem}
                    )
                checked += 1
        layer_ok = not layer_problems and checked == len(rows)
        result["layers"][str(layer)] = {
            "ok": layer_ok,
            "checked": checked,
            "expected": len(rows),
            "problems": layer_problems[:100],
        }
        result["ok"] = result["ok"] and layer_ok

    out = Path(args.out_json).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, out)
    print(json.dumps({"ok": result["ok"], "selected_rows": len(rows), "layers": result["layers"]}))
    raise SystemExit(0 if result["ok"] else 2)


if __name__ == "__main__":
    main()
