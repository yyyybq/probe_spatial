#!/usr/bin/env python3
"""Audit B2 common-history object sampling.

This script checks the data/task construction before training: frame order,
valid object rate, object area, border distance, current-tail visibility,
future hidden tails, tail camera motion, target angular distribution, and a few
mask contact sheets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vidfm3d.data.components.inscene15k_dataset import InsScene15KDataset


def parse_ints(value: str) -> list[int]:
    return [int(v) for v in str(value).replace(",", " ").split() if v.strip()]


def camera_center_from_w2c(extr: torch.Tensor) -> torch.Tensor:
    R = extr[:3, :3]
    t = extr[:3, 3]
    return -R.T @ t


def overlay_mask(img: torch.Tensor, mask: torch.Tensor) -> Image.Image:
    arr = (img.permute(1, 2, 0).detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    m = mask.detach().cpu().numpy().astype(bool)
    out = arr.copy()
    out[m] = (0.55 * out[m] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)
    return Image.fromarray(out)


def save_contact_sheet(sample: dict, path: Path, max_frames: int = 8) -> None:
    images = sample["image"]
    masks = sample["hidden_obj_mask"]
    raw_indices = sample["prefix_indices"].tolist()
    frames = min(images.shape[0], max_frames)
    tiles = []
    for i in range(frames):
        tile = overlay_mask(images[i], masks[i])
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 0, 150, 20), fill=(0, 0, 0))
        draw.text((4, 4), f"obs {i} raw {raw_indices[i]}", fill=(255, 255, 255))
        tiles.append(tile)
    w, h = tiles[0].size
    sheet = Image.new("RGB", (w * frames, h), (0, 0, 0))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, (i * w, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def summarize(values: list[float]) -> dict:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-root", default=os.environ.get("INSCENE_DATA_ROOT", "/nas/baiqiao/InsScene-15K/data"))
    parser.add_argument("--streaming-feat-root", default=os.environ.get("INSCENE_STREAMING_FEAT_ROOT", ""))
    parser.add_argument("--split", default="val", choices=["train", "val", "test", "all"])
    parser.add_argument("--split-manifest", default=os.environ.get("INSCENE_SPLIT_MANIFEST"))
    parser.add_argument("--prefix-lengths", default="8,12,16,24")
    parser.add_argument("--history-len", type=int, default=8)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--out-dir", default="logs/b2_sampling_audit")
    parser.add_argument("--contact-sheets", type=int, default=16)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    contact_saved = 0

    for prefix_len in parse_ints(args.prefix_lengths):
        ds = InsScene15KDataset(
            root=args.data_root,
            root_vfm=args.streaming_feat_root or None,
            streaming_feat_root=args.streaming_feat_root or None,
            sources=["processed_scannetpp_v2"],
            split=args.split,
            split_manifest=args.split_manifest,
            vfm_name="wan",
            feat_pixalign=True,
            num_views=prefix_len,
            target_h=288,
            target_w=512,
            include_pmaps=True,
            streaming_prefix=True,
            prefix_min_len=prefix_len,
            prefix_max_len=prefix_len,
            diag_hidden_obj=True,
            streaming_shared_hidden_obj=True,
            streaming_hidden_seed_prefix_len=args.history_len,
            streaming_hidden_prefix_lengths=parse_ints(args.prefix_lengths),
            allow_missing_vfm=True,
        )
        n = min(len(ds), args.limit)
        for idx in range(n):
            sample = ds[idx]
            valid = bool(sample["hidden_obj_valid"].item())
            row = {
                "prefix_len": prefix_len,
                "dataset_idx": idx,
                "scene_path": sample["scene_path"],
                "window_id": int(ds.scenes[idx].get("streaming_window_id", 0)),
                "raw_indices": " ".join(str(int(v)) for v in sample["prefix_indices"].tolist()),
                "valid": int(valid),
            }
            if valid:
                mask = sample["hidden_obj_mask"].bool()
                counts = mask.flatten(1).sum(dim=1)
                visible = torch.nonzero(counts >= ds.hidden_obj_min_visible_pixels, as_tuple=False).flatten()
                best = int(counts.argmax().item())
                ys, xs = torch.nonzero(mask[best], as_tuple=True)
                border = min(
                    int(xs.min().item()),
                    int(ys.min().item()),
                    int(mask.shape[-1] - 1 - xs.max().item()),
                    int(mask.shape[-2] - 1 - ys.max().item()),
                ) if xs.numel() else -1
                last_visible = int(visible.max().item()) if visible.numel() else -1
                history_tail = max(int(args.history_len) - 1, 0)
                history_tail_area = (
                    int(counts[history_tail].item())
                    if history_tail < counts.numel() else -1
                )
                centers = torch.stack([camera_center_from_w2c(e) for e in sample["extrinsics"]])
                tail_motion = float((centers[-1] - centers[0]).norm().item())
                polar = sample["hidden_obj_polar"]
                row.update({
                    "obj_id": int(sample["hidden_obj_id"].item()),
                    "query_frame": best,
                    "query_area": int(counts[best].item()),
                    "border_px": border,
                    "visible_frames": int(visible.numel()),
                    "history_tail_area": history_tail_area,
                    "history_tail_visible": int(history_tail_area >= ds.hidden_obj_min_visible_pixels),
                    "last_visible": last_visible,
                    "last_visible_gap": int(mask.shape[0] - 1 - last_visible) if last_visible >= 0 else -1,
                    "tail_motion": tail_motion,
                    "azimuth": float(polar[0].item()),
                    "elevation": float(polar[1].item()),
                    "log_dist": float(polar[2].item()),
                })
                if contact_saved < args.contact_sheets:
                    save_contact_sheet(
                        sample,
                        out_dir / "contact_sheets" / f"p{prefix_len}_idx{idx:04d}.jpg",
                    )
                    contact_saved += 1
            rows.append(row)

    csv_path = out_dir / "b2_sampling_audit.csv"
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for prefix_len in parse_ints(args.prefix_lengths):
        subset = [r for r in rows if r["prefix_len"] == prefix_len]
        valid = [r for r in subset if r["valid"]]
        summary[str(prefix_len)] = {
            "n": len(subset),
            "valid": len(valid),
            "valid_rate": len(valid) / max(len(subset), 1),
            "query_area": summarize([r["query_area"] for r in valid]),
            "border_px": summarize([r["border_px"] for r in valid]),
            "history_tail_visible_rate": (
                sum(int(r.get("history_tail_visible", 0)) for r in valid) / max(len(valid), 1)
            ),
            "history_tail_area": summarize([r["history_tail_area"] for r in valid]),
            "last_visible_gap": summarize([r["last_visible_gap"] for r in valid]),
            "tail_motion": summarize([r["tail_motion"] for r in valid]),
            "azimuth": summarize([r["azimuth"] for r in valid]),
            "elevation": summarize([r["elevation"] for r in valid]),
            "log_dist": summarize([r["log_dist"] for r in valid]),
        }
    (out_dir / "b2_sampling_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
