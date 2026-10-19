#!/usr/bin/env python3
"""Deterministic tests for the R1 canonical camera utility."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vagen.envs.active_spatial.canonical_camera import (  # noqa: E402
    CANONICAL_CAMERA_H1_RESIZE_V1,
    build_canonical_camera,
    camera_params_for_visual_metrics,
    camera_pose_from_forward,
)
from vagen.envs.active_spatial.visual_bbox_metrics import compute_visual_bbox_metrics  # noqa: E402


def _assert_close(name: str, actual: Any, expected: Any, atol: float = 1e-6) -> None:
    if not np.allclose(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float), atol=atol):
        raise AssertionError(f"{name} mismatch: actual={actual}, expected={expected}")


def _toy_item() -> dict[str, Any]:
    objects = [
        {
            "id": "A",
            "label": "A",
            "bbox_min": [-0.1, -0.1, -0.1],
            "bbox_max": [0.1, 0.1, 0.1],
            "center": [0.0, 0.0, 0.0],
        },
        {
            "id": "B",
            "label": "B",
            "bbox_min": [0.9, -0.1, -0.1],
            "bbox_max": [1.1, 0.1, 0.1],
            "center": [1.0, 0.0, 0.0],
        },
    ]
    return {
        "target_object": {"objects": objects, "primary": objects[0]},
        "target_region": {
            "params": {
                "object_a_center": [0.0, 0.0, 0.0],
                "object_b_center": [1.0, 0.0, 0.0],
                "relation": "left",
            }
        },
    }


def run_deterministic_tests() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    K_native = [[320.0, 0.0, 320.0], [0.0, 320.0, 240.0], [0.0, 0.0, 1.0]]
    cam = build_canonical_camera(
        K_native=K_native,
        native_size=(640, 480),
        render_size=(256, 256),
        transform="resize",
        camera_position=[0.0, -3.0, 0.0],
        camera_forward=[0.0, 1.0, 0.0],
    )
    expected_K = [[128.0, 0.0, 128.0], [0.0, 170.6666666667, 128.0], [0.0, 0.0, 1.0]]
    _assert_close("K_native_to_effective", cam.K_effective, expected_K, atol=1e-5)
    results.append({"name": "K_native_to_effective_nonuniform", "passed": True, "K_effective": cam.K_effective})

    inferred = build_canonical_camera(
        K_native=K_native,
        render_size=(256, 256),
        transform="resize",
        camera_position=[0.0, -3.0, 0.0],
        camera_forward=[0.0, 1.0, 0.0],
    )
    if inferred.native_resolution != (640, 480):
        raise AssertionError(f"inferred native resolution failed: {inferred.native_resolution}")
    results.append({"name": "native_resolution_inference_from_principal_point", "passed": True})

    c2w = np.asarray(cam.c2w)
    w2c = np.asarray(cam.w2c)
    _assert_close("c2w_w2c_round_trip", w2c @ c2w, np.eye(4), atol=1e-6)
    results.append({"name": "c2w_w2c_round_trip", "passed": True})

    item = _toy_item()
    for name, pos, fwd, relation, expected in [
        ("A_left_B", [0.5, -3.0, 0.0], [0.0, 1.0, 0.0], "left", True),
        ("A_right_B", [0.5, -3.0, 0.0], [0.0, 1.0, 0.0], "right", False),
        ("yaw_reversal", [0.5, -3.0, 0.0], [0.0, -1.0, 0.0], "left", False),
        ("behind_camera", [0.5, 3.0, 0.0], [0.0, 1.0, 0.0], "left", False),
    ]:
        local_item = json.loads(json.dumps(item))
        local_item["target_region"]["params"]["relation"] = relation
        c = build_canonical_camera(
            K_native=[[100.0, 0.0, 128.0], [0.0, 100.0, 128.0], [0.0, 0.0, 1.0]],
            native_size=(256, 256),
            render_size=(256, 256),
            transform="resize",
            camera_position=pos,
            camera_forward=fwd,
        )
        vm = compute_visual_bbox_metrics(
            np.asarray(pos, dtype=float),
            np.asarray(fwd, dtype=float),
            "projective_relations",
            camera_params_for_visual_metrics(local_item, c),
            local_item["target_region"],
        )
        actual = bool(vm.get("visual_relation_satisfied"))
        if actual != expected:
            raise AssertionError(f"{name} failed: expected {expected}, vm={vm}")
        results.append({"name": name, "passed": True, "margin_px": vm.get("visual_relation_margin_px")})

    partial_item = {
        "target_object": {
            "objects": [
                {
                    "id": "P",
                    "label": "partial",
                    "bbox_min": [0.8, -0.1, -0.1],
                    "bbox_max": [1.1, 0.1, 0.1],
                    "center": [0.95, 0.0, 0.0],
                }
            ]
        },
        "target_region": {"params": {"object_center": [0.95, 0.0, 0.0]}},
    }
    c = build_canonical_camera(
        K_native=[[120.0, 0.0, 128.0], [0.0, 120.0, 128.0], [0.0, 0.0, 1.0]],
        native_size=(256, 256),
        render_size=(256, 256),
        transform="resize",
        camera_position=[0.0, -2.0, 0.0],
        camera_forward=[0.0, 1.0, 0.0],
    )
    vm = compute_visual_bbox_metrics(
        np.array([0.0, -2.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        "screen_occupancy",
        camera_params_for_visual_metrics(partial_item, c),
        partial_item["target_region"],
    )
    if not vm.get("available"):
        raise AssertionError(f"partial object projection unavailable: {vm}")
    results.append({"name": "partial_out_of_frame_projection_available", "passed": True})

    non_square = build_canonical_camera(
        K_native=[[400.0, 0.0, 400.0], [0.0, 400.0, 300.0], [0.0, 0.0, 1.0]],
        native_size=(800, 600),
        render_size=(256, 256),
        transform="resize",
        camera_position=[0.0, -3.0, 0.0],
        camera_forward=[0.0, 1.0, 0.0],
    )
    _assert_close("non_square_to_square", non_square.K_effective, expected_K, atol=1e-5)
    results.append({"name": "non_square_native_to_square_render", "passed": True})

    overlay_regression = build_canonical_camera(
        K_native=K_native,
        native_size=(640, 480),
        render_size=(256, 256),
        transform="resize",
        c2w=camera_pose_from_forward([3.3599685426918207, 5.000738149544905, 1.5], [0.6731475503184378, -0.7387745843503218, -0.03293158086541385]),
    )
    if overlay_regression.camera_model_version != CANONICAL_CAMERA_H1_RESIZE_V1:
        raise AssertionError("overlay regression did not use the canonical H1 version")
    results.append({"name": "known_overlay_sample_camera_model_version", "passed": True})

    return results


async def run_renderer_equivalence(args: argparse.Namespace) -> dict[str, Any]:
    from vagen.envs.active_spatial.render.unified_renderer import UnifiedRenderGS

    with open(args.dataset) as f:
        item = next(json.loads(line) for line in f if line.strip() and json.loads(line).get("scene_id") == args.scene_id)

    K_native = np.asarray(item["init_camera"]["intrinsics"], dtype=np.float32)
    c2w = np.asarray(item["init_camera"]["extrinsics"], dtype=np.float32)
    cam = build_canonical_camera(
        K_native=K_native,
        native_size=(640, 480),
        render_size=(256, 256),
        transform="resize",
        c2w=c2w,
    )
    renderer = UnifiedRenderGS(render_backend="http", client_url=args.renderer_url, scene_id=args.scene_id)
    tasks = [
        {"mode": "cam_param", "intrinsics": K_native.tolist(), "extrinsics": np.linalg.inv(c2w).tolist(), "size": [640, 480]},
        {"mode": "cam_param", "intrinsics": cam.K_effective, "extrinsics": cam.w2c, "size": [256, 256]},
    ]
    native_img, direct_img = await renderer.render_tasks(tasks)
    ref_img = native_img.resize((256, 256), Image.Resampling.BILINEAR)
    ref = np.asarray(ref_img, dtype=np.float32)
    direct = np.asarray(direct_img, dtype=np.float32)
    mae = float(np.mean(np.abs(ref - direct)))
    rmse = float(np.sqrt(np.mean((ref - direct) ** 2)))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_img.save(out_dir / "reference_native_resize_256.png")
    direct_img.save(out_dir / "direct_h1_256.png")
    return {"name": "native_resize_vs_direct_h1_real_renderer", "passed": mae <= args.mae_threshold, "mae": mae, "rmse": rmse}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--renderer-url", default=None)
    parser.add_argument("--scene-id", default="0003_839989")
    parser.add_argument("--dataset", default="/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/v46_baseline_qwen25vl_7b/train_filtered.jsonl")
    parser.add_argument("--output-dir", default="/tmp/r1_camera_utility_tests")
    parser.add_argument("--mae-threshold", type=float, default=6.0)
    args = parser.parse_args()

    results = run_deterministic_tests()
    if args.renderer_url:
        results.append(asyncio.run(run_renderer_equivalence(args)))
    passed = all(bool(x.get("passed")) for x in results)
    payload = {"passed": passed, "results": results}
    print(json.dumps(payload, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
