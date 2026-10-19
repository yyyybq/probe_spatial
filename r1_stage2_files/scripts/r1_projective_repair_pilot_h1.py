#!/usr/bin/env python3
"""Generate and render a versioned R1 projective repair pilot under H1 camera."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vagen.envs.active_spatial.canonical_camera import (  # noqa: E402
    CANONICAL_CAMERA_H1_RESIZE_V1,
    build_canonical_camera,
    camera_params_for_visual_metrics,
    camera_pose_from_forward,
    normalize_vector,
)
from vagen.envs.active_spatial.visual_bbox_metrics import (  # noqa: E402
    _extract_objects,
    _match_object,
    _project_object,
    compute_visual_bbox_metrics,
)

GENERATOR_VERSION = "projective_canonical_h1_v1"
REPAIR_REASON = "legacy_projective_target_pose_wrong_side"


def load_projective(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows = []
    with path.open() as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("task_type") == "projective_relations":
                rows.append((idx, item))
    return rows


def canonical_normal_from_prompt(item: dict[str, Any]) -> np.ndarray:
    params = item["target_region"]["params"]
    a = np.asarray(params["object_a_center"], dtype=np.float64)
    b = np.asarray(params["object_b_center"], dtype=np.float64)
    ab = b[:2] - a[:2]
    ab = ab / max(float(np.linalg.norm(ab)), 1e-12)
    relation = params.get("relation", "left")
    if relation == "left":
        return np.array([ab[1], -ab[0]], dtype=np.float64)
    return np.array([-ab[1], ab[0]], dtype=np.float64)


def forward_to_pair_midpoint(item: dict[str, Any], point: np.ndarray) -> np.ndarray:
    params = item["target_region"]["params"]
    a = np.asarray(params["object_a_center"], dtype=np.float64)
    b = np.asarray(params["object_b_center"], dtype=np.float64)
    look_at = np.array([(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, (a[2] + b[2]) / 2.0], dtype=np.float64)
    return normalize_vector(look_at - point)


def build_h1_score(item: dict[str, Any], c2w: np.ndarray, render_size: tuple[int, int]) -> dict[str, Any]:
    cam = build_canonical_camera(
        K_native=item["init_camera"]["intrinsics"],
        render_size=render_size,
        transform="resize",
        c2w=c2w,
        item=item,
    )
    vm = compute_visual_bbox_metrics(
        np.asarray(c2w[:3, 3], dtype=float),
        np.asarray(c2w[:3, 2], dtype=float),
        item["task_type"],
        camera_params_for_visual_metrics(item, cam),
        item["target_region"],
    )
    params = item.get("target_region", {}).get("params", {})
    objects = _extract_objects({"_target_object": item.get("target_object")})
    obj_a = _match_object(objects, center=params.get("object_a_center"), fallback_index=0)
    obj_b = _match_object(objects, center=params.get("object_b_center"), fallback_index=1)
    detailed = []
    for obj in (obj_a, obj_b):
        if obj is not None:
            detailed.append(
                _project_object(
                    obj,
                    np.asarray(cam.c2w, dtype=float),
                    np.asarray(cam.K_effective, dtype=float),
                    int(cam.render_resolution[0]),
                    int(cam.render_resolution[1]),
                )
            )
    if detailed:
        vm = dict(vm)
        vm["objects"] = detailed
    return {"camera": cam.to_dict(), "visual_metrics": vm}


def _area(box: Any) -> float:
    if not box:
        return 0.0
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def _inside_fraction(obj: dict[str, Any]) -> float:
    raw = _area(obj.get("bbox_raw"))
    clipped = _area(obj.get("bbox"))
    return clipped / raw if raw > 1e-6 else 0.0


def validates(vm: dict[str, Any], args: argparse.Namespace) -> tuple[bool, list[str]]:
    reasons = []
    objects = vm.get("objects") or []
    if len(objects) < 2 or not vm.get("available"):
        reasons.append("projection_unavailable")
    if not all(o.get("center_in_front") for o in objects):
        reasons.append("behind_camera")
    if not all(o.get("visible") for o in objects):
        reasons.append("object_not_visible")
    if any(float(o.get("area_ratio", 0.0) or 0.0) < args.min_area_ratio for o in objects):
        reasons.append("tiny_object")
    if any(_inside_fraction(o) < args.min_inside_fraction for o in objects):
        reasons.append("extreme_edge_or_truncated")
    if not bool(vm.get("visual_relation_satisfied")):
        reasons.append("relation_false")
    if float(vm.get("visual_relation_margin_px", -1e9) or -1e9) < args.min_margin_px:
        reasons.append("margin_too_small")
    if float(vm.get("visual_score", 0.0) or 0.0) < args.success_threshold:
        reasons.append("score_below_threshold")
    return (not reasons), reasons


def candidate_points(item: dict[str, Any]) -> list[np.ndarray]:
    params = item["target_region"]["params"]
    boundary = np.asarray(params["boundary_point"], dtype=np.float64)[:2]
    ab = np.asarray(params["boundary_direction"], dtype=np.float64)[:2]
    ab = ab / max(float(np.linalg.norm(ab)), 1e-12)
    old_point = np.asarray(item["sample_target"], dtype=np.float64)
    old_normal = np.asarray(params["normal"], dtype=np.float64)[:2]
    old_normal = old_normal / max(float(np.linalg.norm(old_normal)), 1e-12)
    canonical = canonical_normal_from_prompt(item)
    old_offset = old_point[:2] - boundary
    offset_along = float(np.dot(old_offset, ab))
    offset_normal = float(abs(np.dot(old_offset, old_normal)))
    min_distance = float(params.get("min_distance", 2.0) or 2.0)
    base_dist = max(offset_normal, min_distance, 2.0)
    height = float(old_point[2])

    points: list[np.ndarray] = []
    for scale in (1.0, 0.9, 1.1, 0.75, 1.25, 1.5, 2.0):
        for along_delta in (0.0, -0.5, 0.5, -1.0, 1.0, -2.0, 2.0, -3.0, 3.0):
            xy = boundary + (offset_along + along_delta) * ab + base_dist * scale * canonical
            points.append(np.array([xy[0], xy[1], height], dtype=np.float64))
    return points


def repair_one(old_dataset_index: int, item: dict[str, Any], local_index: int, args: argparse.Namespace) -> Optional[dict[str, Any]]:
    render_size = (args.render_width, args.render_height)
    old_pose = camera_pose_from_forward(item["sample_target"], item["camera_params"]["forward"])
    old_score = build_h1_score(item, old_pose, render_size)
    old_normal = np.asarray(item["target_region"]["params"]["normal"], dtype=np.float64)[:2]
    old_normal = old_normal / max(float(np.linalg.norm(old_normal)), 1e-12)
    new_normal = canonical_normal_from_prompt(item)

    last_reasons: list[str] = []
    for point in candidate_points(item):
        repaired = copy.deepcopy(item)
        forward = forward_to_pair_midpoint(repaired, point)
        pose = camera_pose_from_forward(point, forward)
        params = repaired["target_region"]["params"]
        params["normal"] = new_normal.tolist()
        params["sample_distance"] = float(np.linalg.norm(point[:2] - np.asarray(params["boundary_point"], dtype=float)[:2]))
        repaired["target_region"]["sample_point"] = point.tolist()
        repaired["target_region"]["sample_forward"] = forward.tolist()
        repaired["sample_target"] = point.tolist()
        repaired["camera_params"] = dict(repaired.get("camera_params") or {})
        repaired["camera_params"]["forward"] = forward.tolist()
        repaired["task_id"] = f"projective_canonical_h1_v1_{old_dataset_index:05d}"
        repaired["camera_model_version"] = CANONICAL_CAMERA_H1_RESIZE_V1
        repaired["generator_version"] = GENERATOR_VERSION
        repaired["repair_lineage"] = {
            "old_dataset_index": old_dataset_index,
            "old_task_id": item.get("task_id"),
            "repair_reason": REPAIR_REASON,
        }
        new_score = build_h1_score(repaired, pose, render_size)
        ok, reasons = validates(new_score["visual_metrics"], args)
        last_reasons = reasons
        if ok:
            init_pose = np.asarray(item["init_camera"]["extrinsics"], dtype=np.float64)
            init_score = build_h1_score(repaired, init_pose, render_size)
            mapping = {
                "old_dataset_index": old_dataset_index,
                "old_task_id": item.get("task_id") or f"legacy_projective_index_{old_dataset_index:05d}",
                "new_task_id": repaired["task_id"],
                "scene_id": item["scene_id"],
                "object_a": item["target_region"]["params"].get("object_a_center"),
                "object_b": item["target_region"]["params"].get("object_b_center"),
                "relation": item["target_region"]["params"].get("relation"),
                "old_normal": old_normal.tolist(),
                "new_normal": new_normal.tolist(),
                "old_sample_target": item.get("sample_target"),
                "new_sample_target": repaired["sample_target"],
                "camera_model_version": CANONICAL_CAMERA_H1_RESIZE_V1,
                "generator_version": GENERATOR_VERSION,
                "repair_reason": REPAIR_REASON,
            }
            validation = {
                "pilot_index": local_index,
                "old_dataset_index": old_dataset_index,
                "scene_id": item["scene_id"],
                "relation": mapping["relation"],
                "old_h1_target": old_score,
                "new_h1_target": new_score,
                "new_h1_initial": init_score,
                "old_pose_c2w": old_pose.tolist(),
                "new_pose_c2w": pose.tolist(),
            }
            return {"repaired": repaired, "mapping": mapping, "validation": validation}
    return {
        "failed": {
            "old_dataset_index": old_dataset_index,
            "scene_id": item.get("scene_id"),
            "relation": item.get("target_region", {}).get("params", {}).get("relation"),
            "last_reasons": last_reasons,
        }
    }


def select_rows(rows: list[tuple[int, dict[str, Any]]], args: argparse.Namespace) -> list[tuple[int, dict[str, Any]]]:
    rng = random.Random(args.seed)
    if args.scenes:
        allowed = set(args.scenes.split(","))
        rows = [(idx, row) for idx, row in rows if row.get("scene_id") in allowed]
    scene_counts = Counter(row.get("scene_id") for _, row in rows)
    scenes = [scene for scene, _ in scene_counts.most_common(args.max_scenes)]
    rows = [(idx, row) for idx, row in rows if row.get("scene_id") in scenes]
    by_rel_scene: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for idx, row in rows:
        rel = row["target_region"]["params"].get("relation", "left")
        by_rel_scene[(rel, row["scene_id"])].append((idx, row))
    for bucket in by_rel_scene.values():
        rng.shuffle(bucket)
    selected: list[tuple[int, dict[str, Any]]] = []
    keys = sorted(by_rel_scene)
    cursor = 0
    while len(selected) < args.pilot_count and keys:
        key = keys[cursor % len(keys)]
        bucket = by_rel_scene[key]
        if bucket:
            selected.append(bucket.pop())
        keys = [k for k in keys if by_rel_scene[k]]
        cursor += 1
    return selected


def metric_for_difficulty(validation: dict[str, Any], repaired: dict[str, Any], old: dict[str, Any]) -> dict[str, Any]:
    init_pos = np.asarray(old["init_camera"]["extrinsics"], dtype=float)[:3, 3]
    old_tgt = np.asarray(old["sample_target"], dtype=float)
    new_tgt = np.asarray(repaired["sample_target"], dtype=float)
    init_fwd = np.asarray(old["init_camera"]["extrinsics"], dtype=float)[:3, 2]
    old_fwd = np.asarray(old["camera_params"]["forward"], dtype=float)
    new_fwd = np.asarray(repaired["camera_params"]["forward"], dtype=float)
    def yaw_delta(a: np.ndarray, b: np.ndarray) -> float:
        aa = a[:2] / max(float(np.linalg.norm(a[:2])), 1e-12)
        bb = b[:2] / max(float(np.linalg.norm(b[:2])), 1e-12)
        return float(math.degrees(math.acos(float(np.clip(np.dot(aa, bb), -1.0, 1.0)))))
    def bbox_area_min(vm: dict[str, Any]) -> float:
        areas = [_area(o.get("bbox")) for o in (vm.get("objects") or [])]
        return min(areas) if areas else 0.0
    old_vm = validation["old_h1_target"]["visual_metrics"]
    new_vm = validation["new_h1_target"]["visual_metrics"]
    return {
        "old_translation_distance": float(np.linalg.norm(old_tgt - init_pos)),
        "new_translation_distance": float(np.linalg.norm(new_tgt - init_pos)),
        "old_yaw_delta_deg": yaw_delta(init_fwd, old_fwd),
        "new_yaw_delta_deg": yaw_delta(init_fwd, new_fwd),
        "old_relation_margin_px": old_vm.get("visual_relation_margin_px"),
        "new_relation_margin_px": new_vm.get("visual_relation_margin_px"),
        "old_min_bbox_area": bbox_area_min(old_vm),
        "new_min_bbox_area": bbox_area_min(new_vm),
        "new_initial_score": validation["new_h1_initial"]["visual_metrics"].get("visual_score"),
        "old_target_score": old_vm.get("visual_score"),
        "new_target_score": new_vm.get("visual_score"),
    }


def summarize_difficulty(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "old_translation_distance",
        "new_translation_distance",
        "old_yaw_delta_deg",
        "new_yaw_delta_deg",
        "old_relation_margin_px",
        "new_relation_margin_px",
        "old_min_bbox_area",
        "new_min_bbox_area",
        "new_initial_score",
        "old_target_score",
        "new_target_score",
    ]
    out: dict[str, Any] = {}
    for key in keys:
        vals = [float(r[key]) for r in rows if r.get(key) is not None and math.isfinite(float(r[key]))]
        out[key] = {
            "min": min(vals) if vals else None,
            "max": max(vals) if vals else None,
            "mean": statistics.mean(vals) if vals else None,
            "median": statistics.median(vals) if vals else None,
        }
    out["scene_distribution"] = dict(Counter(r["scene_id"] for r in rows))
    out["relation_distribution"] = dict(Counter(r["relation"] for r in rows))
    return out


def draw_pair(old_img: Image.Image, new_img: Image.Image, rec: dict[str, Any]) -> Image.Image:
    def draw(img: Image.Image, vm: dict[str, Any], title: str) -> Image.Image:
        im = img.convert("RGB").copy()
        d = ImageDraw.Draw(im)
        for obj in vm.get("objects") or []:
            box = obj.get("bbox")
            if box:
                d.rectangle([float(x) for x in box], outline=(255, 40, 40), width=2)
                d.text((float(box[0]), max(0.0, float(box[1]) - 12.0)), str(obj.get("label") or obj.get("id")), fill=(255, 255, 0))
            center = obj.get("center_uv")
            if center:
                u, v = float(center[0]), float(center[1])
                d.ellipse((u - 3, v - 3, u + 3, v + 3), fill=(40, 255, 40))
        d.text((3, 3), title, fill=(255, 255, 0))
        return im
    old_vm = rec["old_h1_target"]["visual_metrics"]
    new_vm = rec["new_h1_target"]["visual_metrics"]
    left = draw(old_img, old_vm, f"old ok={old_vm.get('visual_relation_satisfied')} m={old_vm.get('visual_relation_margin_px'):.1f}")
    right = draw(new_img, new_vm, f"new ok={new_vm.get('visual_relation_satisfied')} m={new_vm.get('visual_relation_margin_px'):.1f}")
    pair = Image.new("RGB", (left.width + right.width, max(left.height, right.height)), "white")
    pair.paste(left, (0, 0))
    pair.paste(right, (left.width, 0))
    return pair


async def render_pairs(validations: list[dict[str, Any]], args: argparse.Namespace) -> None:
    from vagen.envs.active_spatial.render.unified_renderer import UnifiedRenderGS

    out = Path(args.output_dir) / "projective_pilot_paired_render"
    out.mkdir(parents=True, exist_ok=True)
    thumbs = []
    manifest = []
    for scene_id in sorted({v["scene_id"] for v in validations}):
        scene_vals = [v for v in validations if v["scene_id"] == scene_id]
        renderer = UnifiedRenderGS(render_backend="http", client_url=args.renderer_url, scene_id=scene_id)
        tasks = []
        for rec in scene_vals:
            for key in ("old_h1_target", "new_h1_target"):
                cam = rec[key]["camera"]
                tasks.append({"mode": "cam_param", "intrinsics": cam["K_effective"], "extrinsics": cam["w2c"], "size": [args.render_width, args.render_height]})
        imgs = await renderer.render_tasks(tasks)
        for i, rec in enumerate(scene_vals):
            old_img, new_img = imgs[2 * i], imgs[2 * i + 1]
            pair = draw_pair(old_img, new_img, rec)
            name = f"{rec['pilot_index']:03d}_{scene_id}_old_new.png"
            pair.save(out / name)
            manifest.append({"pilot_index": rec["pilot_index"], "scene_id": scene_id, "image": name, "relation": rec["relation"], "old_h1_target": rec["old_h1_target"], "new_h1_target": rec["new_h1_target"]})
            thumbs.append(pair.resize((256, 128)))
    cols = 5
    rows = int(math.ceil(len(thumbs) / cols)) if thumbs else 1
    sheet = Image.new("RGB", (cols * 256, rows * 128), "white")
    for idx, im in enumerate(thumbs):
        sheet.paste(im, ((idx % cols) * 256, (idx // cols) * 128))
    sheet.save(out / "paired_contact_sheet.png")
    with (out / "paired_render_manifest.jsonl").open("w") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")


def write_outputs(valid: list[dict[str, Any]], failed: list[dict[str, Any]], old_by_index: dict[int, dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    repaired_rows = [v["repaired"] for v in valid]
    mappings = [v["mapping"] for v in valid]
    validations = [v["validation"] for v in valid]
    with (out / "r1_projective_repaired_pilot_h1_v1.jsonl").open("w") as f:
        for row in repaired_rows:
            f.write(json.dumps(row) + "\n")
    with (out / "r1_projective_repaired_pilot_mapping_h1_v1.jsonl").open("w") as f:
        for row in mappings:
            f.write(json.dumps(row) + "\n")
    with (out / "r1_projective_repaired_pilot_validation_h1_v1.jsonl").open("w") as f:
        for row in validations:
            f.write(json.dumps(row) + "\n")
    if failed:
        with (out / "r1_projective_repaired_pilot_failures_h1_v1.jsonl").open("w") as f:
            for row in failed:
                f.write(json.dumps(row) + "\n")

    difficulty_rows = []
    for v in valid:
        old = old_by_index[v["mapping"]["old_dataset_index"]]
        m = metric_for_difficulty(v["validation"], v["repaired"], old)
        m.update({"pilot_index": v["validation"]["pilot_index"], "scene_id": v["mapping"]["scene_id"], "relation": v["mapping"]["relation"]})
        difficulty_rows.append(m)
    difficulty = summarize_difficulty(difficulty_rows)
    with (out / "r1_projective_repaired_pilot_difficulty_h1_v1.json").open("w") as f:
        json.dump({"summary": difficulty, "rows": difficulty_rows}, f, indent=2)

    success = sum(bool(v["validation"]["new_h1_target"]["visual_metrics"].get("visual_relation_satisfied")) for v in valid)
    old_success = sum(bool(v["validation"]["old_h1_target"]["visual_metrics"].get("visual_relation_satisfied")) for v in valid)
    pilot_summary = {
        "generator_version": GENERATOR_VERSION,
        "camera_model_version": CANONICAL_CAMERA_H1_RESIZE_V1,
        "requested": args.pilot_count,
        "generated": len(valid),
        "failed_generation": len(failed),
        "old_relation_success": old_success,
        "new_relation_success": success,
        "new_relation_success_rate": success / max(len(valid), 1),
        "scene_distribution": dict(Counter(v["mapping"]["scene_id"] for v in valid)),
        "relation_distribution": dict(Counter(v["mapping"]["relation"] for v in valid)),
        "difficulty_summary": difficulty,
        "visibility_note": "bbox projection proxy only; true instance-mask visibility is not claimed",
    }
    with (out / "r1_projective_repaired_pilot_summary_h1_v1.json").open("w") as f:
        json.dump(pilot_summary, f, indent=2)
    report = [
        "# R1 Projective Repaired Pilot H1 V1",
        "",
        f"Generator version: `{GENERATOR_VERSION}`",
        f"Camera model: `{CANONICAL_CAMERA_H1_RESIZE_V1}`",
        f"Generated: `{len(valid)}` / requested `{args.pilot_count}`",
        f"Failed generation: `{len(failed)}`",
        f"Old target relation success: `{old_success}/{len(valid)}`",
        f"New target relation success: `{success}/{len(valid)}`",
        "",
        "Scene distribution:",
        "",
        "```json",
        json.dumps(pilot_summary["scene_distribution"], indent=2),
        "```",
        "",
        "Relation distribution:",
        "",
        "```json",
        json.dumps(pilot_summary["relation_distribution"], indent=2),
        "```",
        "",
        "Difficulty summary:",
        "",
        "```json",
        json.dumps(difficulty, indent=2),
        "```",
    ]
    (out / "r1_projective_repaired_pilot_report_h1_v1.md").write_text("\n".join(report) + "\n")
    return pilot_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/v46_baseline_qwen25vl_7b/train_filtered.jsonl")
    parser.add_argument("--output-dir", default="/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/r1_observation_aligned_reward_audit/r1_projective_repair_pilot_h1_v1")
    parser.add_argument("--pilot-count", type=int, default=120)
    parser.add_argument("--max-scenes", type=int, default=5)
    parser.add_argument("--scenes", default=None)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--render-width", type=int, default=256)
    parser.add_argument("--render-height", type=int, default=256)
    parser.add_argument("--success-threshold", type=float, default=0.65)
    parser.add_argument("--min-margin-px", type=float, default=12.0)
    parser.add_argument("--min-area-ratio", type=float, default=1e-4)
    parser.add_argument("--min-inside-fraction", type=float, default=0.50)
    parser.add_argument("--renderer-url", default=None)
    args = parser.parse_args()

    all_rows = load_projective(Path(args.dataset))
    selected = select_rows(all_rows, args)
    old_by_index = {idx: row for idx, row in all_rows}
    valid: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for old_dataset_index, item in selected:
        result = repair_one(old_dataset_index, item, len(valid), args)
        if result and "repaired" in result:
            valid.append(result)
        elif result and "failed" in result:
            failed.append(result["failed"])
        if len(valid) >= args.pilot_count:
            break
    summary = write_outputs(valid, failed, old_by_index, args)
    if args.renderer_url:
        asyncio.run(render_pairs([v["validation"] for v in valid], args))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
