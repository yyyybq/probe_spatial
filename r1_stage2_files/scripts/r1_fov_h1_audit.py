#!/usr/bin/env python3
"""R1 H1 audit for all v46 fov_inclusion samples."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vagen.envs.active_spatial.canonical_camera import (  # noqa: E402
    CANONICAL_CAMERA_H1_RESIZE_V1,
    build_canonical_camera,
    build_historical_v46_camera,
    camera_params_for_visual_metrics,
    camera_pose_from_forward,
)
from vagen.envs.active_spatial.visual_bbox_metrics import (  # noqa: E402
    _extract_objects,
    _match_object,
    _project_object,
    compute_visual_bbox_metrics,
)


def load_items(path: Path, task_type: str) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip() and json.loads(line).get("task_type") == task_type]


def target_c2w(item: dict[str, Any]) -> np.ndarray:
    return camera_pose_from_forward(item["sample_target"], item["camera_params"]["forward"])


def detailed_pair_projections(item: dict[str, Any], camera: dict[str, Any]) -> list[dict[str, Any]]:
    params = item.get("target_region", {}).get("params", {})
    objects = _extract_objects({"_target_object": item.get("target_object")})
    obj_a = _match_object(objects, center=params.get("object_a_center"), fallback_index=0)
    obj_b = _match_object(objects, center=params.get("object_b_center"), fallback_index=1)
    out = []
    for obj in (obj_a, obj_b):
        if obj is not None:
            out.append(
                _project_object(
                    obj,
                    np.asarray(camera["c2w"], dtype=float),
                    np.asarray(camera["K_effective"], dtype=float),
                    int(camera["render_resolution"][0]),
                    int(camera["render_resolution"][1]),
                )
            )
    return out


def _score(item: dict[str, Any], c2w: np.ndarray, camera_model: str, render_size: tuple[int, int]) -> dict[str, Any]:
    if camera_model == CANONICAL_CAMERA_H1_RESIZE_V1:
        cam = build_canonical_camera(
            K_native=item["init_camera"]["intrinsics"],
            render_size=render_size,
            transform="resize",
            c2w=c2w,
            item=item,
        )
    elif camera_model == "historical_v46_unscaled_env256":
        cam = build_historical_v46_camera(
            K_native=item["init_camera"]["intrinsics"],
            render_size=render_size,
            c2w=c2w,
        )
    else:
        raise ValueError(camera_model)
    vm = compute_visual_bbox_metrics(
        np.asarray(c2w[:3, 3], dtype=float),
        np.asarray(c2w[:3, 2], dtype=float),
        item["task_type"],
        camera_params_for_visual_metrics(item, cam),
        item["target_region"],
    )
    detailed = detailed_pair_projections(item, cam.to_dict())
    if detailed:
        vm = dict(vm)
        vm["objects"] = detailed
    return {"camera": cam.to_dict(), "visual_metrics": vm}


def raw_area(box: Any) -> float:
    if not box:
        return 0.0
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def projection_features(vm: dict[str, Any], success_threshold: float) -> dict[str, Any]:
    objects = vm.get("objects") or []
    visible_objects = [bool(o.get("visible")) for o in objects]
    clipped_areas = [raw_area(o.get("bbox")) for o in objects]
    raw_areas = [raw_area(o.get("bbox_raw")) for o in objects]
    inside_fractions = [
        (clipped / raw) if raw > 1e-6 else 0.0
        for clipped, raw in zip(clipped_areas, raw_areas)
    ]
    center_front = [bool(o.get("center_in_front")) for o in objects]
    area_ratios = [float(o.get("area_ratio", 0.0) or 0.0) for o in objects]
    return {
        "bbox_valid": bool(objects and all(o.get("bbox") for o in objects)),
        "center_in_front": bool(objects and all(center_front)),
        "target_visible": bool(objects and all(visible_objects)),
        "inside_frame_fraction_min": min(inside_fractions) if inside_fractions else 0.0,
        "truncated": bool(inside_fractions and min(inside_fractions) < 0.95),
        "out_of_frame": bool(not objects or not all(visible_objects)),
        "tiny_object": bool(area_ratios and min(area_ratios) < 1e-4),
        "canonical_score": float(vm.get("visual_score", 0.0) or 0.0),
        "success": bool(float(vm.get("visual_score", 0.0) or 0.0) >= success_threshold),
        "objects": objects,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    target_scores = []
    init_scores = []
    h0_target_scores = []
    h1_target_scores = []
    for rec in records:
        counts["total"] += 1
        h1_target = rec["h1_target"]
        h1_init = rec["h1_initial"]
        h0_target = rec["h0_target"]
        target_scores.append(h1_target["canonical_score"])
        init_scores.append(h1_init["canonical_score"])
        h0_target_scores.append(h0_target["canonical_score"])
        h1_target_scores.append(h1_target["canonical_score"])
        for key in ("bbox_valid", "target_visible", "success", "truncated", "out_of_frame", "tiny_object"):
            counts[f"h1_target_{key}"] += int(bool(h1_target[key]))
        counts["h1_initial_success"] += int(bool(h1_init["success"]))
        counts["h0_h1_target_success_disagreement"] += int(bool(h0_target["success"]) != bool(h1_target["success"]))
        counts["degenerate_or_unreachable_proxy"] += int(not h1_target["bbox_valid"] or not h1_target["center_in_front"])
    def stats(vals: list[float]) -> dict[str, float | None]:
        if not vals:
            return {"min": None, "max": None, "mean": None, "median": None}
        return {
            "min": min(vals),
            "max": max(vals),
            "mean": statistics.mean(vals),
            "median": statistics.median(vals),
        }
    summary = {
        "counts": dict(counts),
        "h1_target_score": stats(target_scores),
        "h1_initial_score": stats(init_scores),
        "h0_target_score": stats(h0_target_scores),
        "h1_target_score_again": stats(h1_target_scores),
    }
    total = max(counts["total"], 1)
    summary["rates"] = {k: v / total for k, v in counts.items() if k != "total"}
    if summary["rates"].get("h1_initial_success", 0.0) > 0.25:
        decision = "FOV-B"
        reason = "H1 target poses are mostly valid, but initial states are frequently already successful; treat this as a degenerate subset before training use."
    elif summary["rates"].get("h1_target_success", 0.0) >= 0.95 and summary["rates"].get("h1_target_out_of_frame", 1.0) <= 0.05:
        decision = "FOV-A"
        reason = "H1 target success is high and invalid/out-of-frame rate is low; data appears correct once camera/scorer is fixed."
    elif summary["rates"].get("h1_target_success", 0.0) >= 0.75:
        decision = "FOV-B"
        reason = "Most H1 targets pass but a non-trivial tail remains; inspect/filter/regenerate only the degenerate subset."
    else:
        decision = "FOV-C"
        reason = "H1 target success is too low for a camera-only repair conclusion."
    summary["decision"] = decision
    summary["decision_reason"] = reason
    summary["visibility_note"] = "projection/bbox proxy only; no instance-mask true visible-fraction oracle is claimed"
    return summary


def select_overlay_records(records: list[dict[str, Any]], limit: int, seed: int, overlay_scenes: str | None = None) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    if overlay_scenes:
        allowed = set(overlay_scenes.split(","))
        records = [r for r in records if r.get("scene_id") in allowed]
    buckets: list[list[dict[str, Any]]] = [
        [r for r in records if r["h1_target"]["success"] and not r["h1_target"]["truncated"]],
        [r for r in records if r["h1_target"]["truncated"]],
        [r for r in records if not r["h1_target"]["success"]],
        [r for r in records if r["h0_target"]["success"] != r["h1_target"]["success"]],
    ]
    picked: list[dict[str, Any]] = []
    seen: set[int] = set()
    quota = max(1, limit // max(len(buckets), 1))
    for bucket in buckets:
        rng.shuffle(bucket)
        for rec in bucket[:quota]:
            if rec["dataset_index"] not in seen:
                picked.append(rec)
                seen.add(rec["dataset_index"])
    remaining = [r for r in records if r["dataset_index"] not in seen]
    rng.shuffle(remaining)
    picked.extend(remaining[: max(0, limit - len(picked))])
    return picked[:limit]


def draw_overlay(img: Image.Image, rec: dict[str, Any]) -> Image.Image:
    im = img.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    vm = rec["h1_target"]["raw_visual_metrics"]
    for obj in vm.get("objects") or []:
        box = obj.get("bbox")
        if box:
            color = (255, 40, 40) if obj.get("visible") else (128, 128, 128)
            d.rectangle([float(x) for x in box], outline=color, width=2)
        center = obj.get("center_uv")
        if center:
            u, v = float(center[0]), float(center[1])
            d.ellipse((u - 3, v - 3, u + 3, v + 3), fill=(40, 255, 40))
        label = str(obj.get("label") or obj.get("id") or "obj")
        anchor = box[:2] if box else (2, 18)
        d.text((float(anchor[0]), max(0.0, float(anchor[1]) - 12.0)), label, fill=(255, 255, 0))
    d.rectangle((0, 0, 255, 255), outline=(0, 200, 255), width=1)
    d.text((3, 3), f"{rec['dataset_index']} {rec['scene_id']} score={rec['h1_target']['canonical_score']:.3f}", fill=(255, 255, 0))
    return im


async def render_overlays(records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    from vagen.envs.active_spatial.render.unified_renderer import UnifiedRenderGS

    out_dir = Path(args.output_dir) / "fov_h1_overlays"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    thumbs = []
    for scene_id in sorted({r["scene_id"] for r in records}):
        scene_records = [r for r in records if r["scene_id"] == scene_id]
        renderer = UnifiedRenderGS(render_backend="http", client_url=args.renderer_url, scene_id=scene_id)
        tasks = []
        for rec in scene_records:
            cam = rec["h1_target"]["camera"]
            tasks.append({"mode": "cam_param", "intrinsics": cam["K_effective"], "extrinsics": cam["w2c"], "size": [args.render_width, args.render_height]})
        images = await renderer.render_tasks(tasks)
        for rec, img in zip(scene_records, images):
            overlay = draw_overlay(img, rec)
            name = f"{rec['dataset_index']:05d}_{scene_id}_fov_h1.png"
            overlay.save(out_dir / name)
            manifest.append({"dataset_index": rec["dataset_index"], "scene_id": scene_id, "image": name, "h1_target": rec["h1_target"]})
            thumbs.append(overlay.resize((128, 128)))
    cols = 10
    rows = int(math.ceil(len(thumbs) / cols)) if thumbs else 1
    sheet = Image.new("RGB", (cols * 128, rows * 128), "white")
    for idx, im in enumerate(thumbs):
        sheet.paste(im, ((idx % cols) * 128, (idx // cols) * 128))
    sheet.save(out_dir / "contact_sheet.png")
    with (out_dir / "manifest.jsonl").open("w") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")


def audit(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items = load_items(Path(args.dataset), "fov_inclusion")
    records: list[dict[str, Any]] = []
    render_size = (args.render_width, args.render_height)
    for dataset_index, item in enumerate(items):
        init_c2w = np.asarray(item["init_camera"]["extrinsics"], dtype=float)
        tgt_c2w = target_c2w(item)
        h0_init = _score(item, init_c2w, "historical_v46_unscaled_env256", render_size)
        h0_tgt = _score(item, tgt_c2w, "historical_v46_unscaled_env256", render_size)
        h1_init = _score(item, init_c2w, CANONICAL_CAMERA_H1_RESIZE_V1, render_size)
        h1_tgt = _score(item, tgt_c2w, CANONICAL_CAMERA_H1_RESIZE_V1, render_size)
        rec = {
            "dataset_index": dataset_index,
            "task_id": item.get("task_id") or f"legacy_fov_index_{dataset_index:05d}",
            "scene_id": item.get("scene_id"),
            "task_type": item.get("task_type"),
            "object_ids": [obj.get("id") for obj in (item.get("target_object", {}).get("objects") or [])],
            "object_labels": [obj.get("label") for obj in (item.get("target_object", {}).get("objects") or [])],
            "initial_pose_c2w": init_c2w.tolist(),
            "sample_target_pose_c2w": tgt_c2w.tolist(),
            "sample_target": item.get("sample_target"),
            "K_native": item.get("init_camera", {}).get("intrinsics"),
            "K_effective_h1": h1_tgt["camera"]["K_effective"],
            "native_resolution": h1_tgt["camera"]["native_resolution"],
            "render_resolution": h1_tgt["camera"]["render_resolution"],
            "h0_initial": projection_features(h0_init["visual_metrics"], args.success_threshold),
            "h0_target": projection_features(h0_tgt["visual_metrics"], args.success_threshold),
            "h1_initial": projection_features(h1_init["visual_metrics"], args.success_threshold),
            "h1_target": projection_features(h1_tgt["visual_metrics"], args.success_threshold),
        }
        rec["h1_initial"]["camera"] = h1_init["camera"]
        rec["h1_target"]["camera"] = h1_tgt["camera"]
        rec["h1_target"]["raw_visual_metrics"] = h1_tgt["visual_metrics"]
        rec["h0_target"]["raw_visual_metrics"] = h0_tgt["visual_metrics"]
        records.append(rec)
    summary = summarize(records)
    return records, summary


def write_outputs(records: list[dict[str, Any]], summary: dict[str, Any], args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "r1_fov_h1_audit_manifest.jsonl"
    with manifest_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    with (out / "r1_fov_h1_audit_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    report = [
        "# R1 FOV H1 Audit",
        "",
        f"Dataset: `{args.dataset}`",
        f"Camera model: `{CANONICAL_CAMERA_H1_RESIZE_V1}`",
        f"Render resolution: `{args.render_width}x{args.render_height}`",
        "",
        "## Summary",
        "",
        f"Decision: **{summary['decision']}**",
        f"Reason: {summary['decision_reason']}",
        "",
        "Counts:",
        "",
        "```json",
        json.dumps(summary["counts"], indent=2),
        "```",
        "",
        "Rates:",
        "",
        "```json",
        json.dumps(summary["rates"], indent=2),
        "```",
        "",
        "Visibility note: projection/bbox proxy only; no true instance-mask visible fraction is claimed.",
    ]
    (out / "r1_fov_h1_audit_report.md").write_text("\n".join(report) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/v46_baseline_qwen25vl_7b/train_filtered.jsonl")
    parser.add_argument("--output-dir", default="/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/r1_observation_aligned_reward_audit/r1_fov_h1_audit_20260902")
    parser.add_argument("--render-width", type=int, default=256)
    parser.add_argument("--render-height", type=int, default=256)
    parser.add_argument("--success-threshold", type=float, default=0.65)
    parser.add_argument("--overlay-count", type=int, default=50)
    parser.add_argument("--overlay-scenes", default=None)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--renderer-url", default=None)
    args = parser.parse_args()

    records, summary = audit(args)
    write_outputs(records, summary, args)
    if args.renderer_url:
        selected = select_overlay_records(records, args.overlay_count, args.seed, args.overlay_scenes)
        asyncio.run(render_overlays(selected, args))
    print(json.dumps({"records": len(records), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
