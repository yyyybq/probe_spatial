#!/usr/bin/env python3
"""Build a model-independent Active Spatial state bank.

The builder restores a branch by resetting the exact episode seed and replaying
the recorded action prefix.  Candidate-action labels are therefore derived from
the real environment transition (including collision invalidation), never from
an idealized pose update.  RGB images are persisted once; feature extractors
consume their paths and never invoke the environment.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import LABEL_VERSION, UNAVAILABLE, geometric_labels, rotation_delta_deg, target_center
from .io_utils import stable_hash, write_json, write_jsonl
from .protocol import freeze_protocol, validate_state_bank


ACTION_TEXT = "<action>{}</action>"


def _scalar(text: str) -> Any:
    text = text.strip().strip('"').strip("'")
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    if text.lower() in {"null", "none"}:
        return None
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def read_env_yaml(path: Path) -> dict[str, Any]:
    """Read only the scalar ``env1.env_config`` mapping used by VAGEN configs."""
    result: dict[str, Any] = {}
    inside = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "env_config:":
            inside = True
            continue
        if inside and not raw.startswith("        "):
            if raw and not raw.startswith(" "):
                break
            continue
        if inside and ":" in stripped:
            key, value = stripped.split(":", 1)
            if value.strip() and not value.lstrip().startswith("#"):
                result[key.strip()] = _scalar(value.split("#", 1)[0])
    if not result:
        raise ValueError(f"Could not read scalar env1.env_config fields from {path}")
    return result


def scene_split(scene_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{scene_id}".encode()).digest()[0]
    return "train" if digest < 180 else "val" if digest < 218 else "test"


def _image(obs: dict[str, Any], placeholder: str):
    images = obs.get("multi_modal_data", {}).get(placeholder, [])
    if not images:
        raise RuntimeError("Environment observation has no current RGB image")
    return images[-1]


def _save_image(image: Any, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _visibility_label(env: Any) -> dict[str, Any]:
    """Use the environment checker only when it is instantiated and scene-backed."""
    checker = getattr(env, "visibility_checker", None)
    center = env._get_target_center() if hasattr(env, "_get_target_center") else None
    if checker is None or center is None:
        return {"status": UNAVAILABLE, "reason": "visibility checker or target center unavailable"}
    pose = env.view_engine.get_pose()
    bmin, bmax = env._get_target_bbox()
    result = checker.check_visibility(
        camera_position=pose[:3, 3], camera_forward=pose[:3, 2], camera_up=pose[:3, 1],
        target_center=center, target_bbox_min=bmin, target_bbox_max=bmax,
        target_label=env.current_item.get("object_label", ""),
    )
    # ``VisibilityResult`` deliberately stores the continuous ratio but not
    # the checker’s categorical cause.  Obtain that cause from the same formal
    # checker rather than inventing it from the ratio.
    _, occlusion_type = checker.check_occlusion(
        camera_position=pose[:3, 3], target_center=center,
        target_label=env.current_item.get("object_label", ""), target_id="",
        target_bbox_min=bmin, target_bbox_max=bmax, num_samples=9,
    )
    return {
        "status": "VERIFIED", "in_fov": bool(result.in_fov), "is_visible": bool(result.is_visible),
        "occlusion_ratio": float(result.occlusion_ratio), "occlusion_type": str(occlusion_type),
        "screen_coverage": float(result.screen_coverage),
    }


def _task_relation_label(env: Any) -> dict[str, Any]:
    metric = env._calculate_canonical_metric() if hasattr(env, "_calculate_canonical_metric") else None
    if not isinstance(metric, dict):
        return {"status": UNAVAILABLE, "reason": "canonical task metric unavailable for this row"}
    relation = metric.get("visual_relation_satisfied")
    if relation is None:
        return {"status": UNAVAILABLE, "reason": "canonical metric does not expose visual relation"}
    return {"status": "VERIFIED", "current_relation_satisfied": bool(relation), "metric_version": metric.get("metric_version")}


class VagenReplay:
    def __init__(self, vagen_root: Path, config: dict[str, Any]):
        sys.path.insert(0, str(vagen_root))
        env_config_module = importlib.import_module("vagen.envs.active_spatial.env_config")
        env_module = importlib.import_module("vagen.envs.active_spatial.env")
        allowed = {field.name for field in dataclasses.fields(env_config_module.ActiveSpatialEnvConfig)}
        self.config = env_config_module.ActiveSpatialEnvConfig(**{k: v for k, v in config.items() if k in allowed})
        self.env_cls = env_module.ActiveSpatialEnv

    def reset_and_replay(self, episode_seed: int, actions: list[str]):
        env = self.env_cls(self.config)
        obs, info = env.reset(seed=episode_seed)
        for action in actions:
            obs, _, done, info = env.step(ACTION_TEXT.format(action))
            if done:
                raise RuntimeError(f"Recorded action prefix terminates early at action {action}")
        return env, obs, info


def _counterfactuals(replay: VagenReplay, episode_seed: int, prefix: list[str], actions: list[str], current_score: float, current_distance: float | str, expected_pose: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for action in actions:
        env, _, _ = replay.reset_and_replay(episode_seed, prefix)
        before = env.view_engine.get_pose().tolist()
        # This is an environment-state assertion, not a geometry-only check:
        # every candidate branch must first reconstruct the exact rendered
        # state at which it is evaluated.
        restored = bool(np.allclose(np.asarray(before), np.asarray(expected_pose), atol=1e-6, rtol=0.0))
        _, _, done, info = env.step(ACTION_TEXT.format(action))
        after = env.view_engine.get_pose().tolist()
        next_center = target_center(env.current_item)
        next_distance: float | str = UNAVAILABLE
        if next_center is not None:
            from .geometry import distance_m
            next_distance = distance_m(after, next_center)
        next_score = env._calculate_current_score()
        rows.append({
            "requested_action": action,
            "state_restored_exact": restored,
            "actual_pose_changed": before != after,
            "done": bool(done),
            "collision_count": int(info.get("collision_count", 0)),
            "distance_before_m": current_distance,
            "distance_after_m": next_distance,
            "distance_decrease": (next_distance < current_distance - 0.05) if isinstance(next_distance, float) and isinstance(current_distance, float) else UNAVAILABLE,
            "task_score_before": current_score,
            "task_score_after": float(next_score),
            "task_score_improve": bool(next_score > current_score + 1e-6),
        })
    return rows


def build(args: argparse.Namespace) -> dict[str, Any]:
    config = read_env_yaml(args.env_config)
    # These are execution bindings, not a second protocol.  They let an SCO
    # worker reuse the checked-in training config while resolving the renderer
    # and task file to its verified mounted/AOSS locations.
    if args.jsonl_path is not None:
        config["jsonl_path"] = str(args.jsonl_path)
    if args.gs_root is not None:
        config["gs_root"] = str(args.gs_root)
    if args.render_backend is not None:
        config["render_backend"] = str(args.render_backend)
    if args.gpu_device is not None:
        config["gpu_device"] = int(args.gpu_device)
    # State-bank labels require evidence; this only affects reward accounting,
    # not rendered observations or actions.  The exact overrides are frozen.
    if args.enable_visibility_labels:
        config["enable_visibility_check"] = True
    if args.enable_collision_labels:
        config["enable_collision_detection"] = True
        config["collision_invalidate_action"] = True
    config["enable_distance_in_obs"] = False  # prevent direct GT distance leakage
    replay = VagenReplay(args.vagen_root, config)
    rng = random.Random(args.seed)
    # ``ACTION_LOOKUP`` is a global superset.  Selecting from it made the
    # previous canary emit pitch actions in a ``strafe`` environment, which
    # the native parser correctly rejects and can auto-terminate.  We obtain
    # the effective set from the first real reset below.
    actions: list[str] | None = None
    output = args.output_dir
    image_root = output / "images"
    rows: list[dict[str, Any]] = []
    reset_failures: list[dict[str, Any]] = []
    per_scene: dict[str, int] = {}
    if args.episode_seeds_file is not None:
        episode_seeds = [int(line.strip()) for line in args.episode_seeds_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        episode_seeds = list(range(args.start_seed, args.start_seed + args.episodes))
    for episode_seed in episode_seeds:
        prefix: list[str] = []
        history: list[dict[str, Any]] = []
        pose_history: list[Any] = []
        try:
            env, obs, _ = replay.reset_and_replay(episode_seed, prefix)
        except Exception as exc:
            rows.append({"sample_id": f"failed_seed_{episode_seed}", "build_status": "BLOCKED", "reason": repr(exc)})
            reset_failures.append({"episode_seed": episode_seed, "exception": repr(exc)})
            continue
        if actions is None:
            actions = sorted(set(getattr(env, "_allowed_actions", set())) - {"done"})
            if not actions:
                raise RuntimeError("Environment exposed no effective non-terminal actions")
        for env_step in range(args.steps_per_episode):
            scene_id = str(env.current_item.get("scene_id", ""))
            if per_scene.get(scene_id, 0) >= args.max_per_scene:
                break
            image_path = image_root / f"seed{episode_seed:06d}_step{env_step:03d}.png"
            image_hash = _save_image(_image(obs, config.get("image_placeholder", "<image>")), image_path)
            # Capture the checker output at every real observation so the
            # three required observability groups are evidence-backed.
            entry = {
                "env_step": env_step,
                "image_path": str(image_path.relative_to(output)),
                "image_sha256": image_hash,
                "visibility": _visibility_label(env),
            }
            history.append(entry)
            pose_history.append(env.view_engine.get_pose().tolist())
            if len(history) >= 3:
                c2w = pose_history[-1]
                labels = geometric_labels(env.current_item, c2w)
                score = float(env._calculate_current_score())
                labels["visibility"] = _visibility_label(env)
                labels["target_relation"] = _task_relation_label(env)
                labels["task_score"] = score
                labels["camera_rotation_from_previous_deg"] = rotation_delta_deg(pose_history[-2], c2w)
                sample_id = stable_hash({"episode_seed": episode_seed, "prefix": prefix, "history": history[-3:]})[:24]
                visible_history = [h["visibility"].get("is_visible") for h in history[-3:] if h["visibility"].get("status") == "VERIFIED"]
                current_visible = visible_history[-1] if visible_history else UNAVAILABLE
                previous_visible = any(bool(value) for value in visible_history[:-1]) if len(visible_history) >= 2 else UNAVAILABLE
                never_visible = (not any(bool(value) for value in visible_history)) if len(visible_history) == 3 else UNAVAILABLE
                rows.append({
                    "sample_id": sample_id, "scene_id": scene_id, "episode_id": f"seed_{episode_seed}",
                    "split": scene_split(scene_id, args.split_seed), "history": list(history[-3:]),
                    "task": {k: env.current_item.get(k) for k in ("task_type", "task_description", "object_label", "preset", "target_object")},
                    "labels": labels,
                    "observability": {
                        "current_visible": current_visible,
                        "previously_visible": previous_visible,
                        "never_visible": never_visible,
                        "status": "VERIFIED" if len(visible_history) == 3 else UNAVAILABLE,
                    },
                    "counterfactual_actions": _counterfactuals(replay, episode_seed, prefix, actions, score, labels.get("distance_m", UNAVAILABLE), c2w),
                    "label_version": LABEL_VERSION,
                })
                per_scene[scene_id] = per_scene.get(scene_id, 0) + 1
                if args.target_rows is not None and len(rows) >= args.target_rows:
                    break
            action = rng.choice(actions)
            prefix.append(action)
            obs, _, done, _ = env.step(ACTION_TEXT.format(action))
            if done:
                break
        if args.target_rows is not None and len(rows) >= args.target_rows:
            break
    complete = [row for row in rows if row.get("build_status") != "BLOCKED"]
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "states.jsonl", complete)
    metadata = {
        "status": "COMPLETE" if complete else "BLOCKED", "seed": args.seed, "source_env_config": str(args.env_config),
        "effective_env_config": config, "rows": len(complete), "failed_episode_resets": len(rows) - len(complete),
        "scene_counts": per_scene, "state_bank_hash": stable_hash(complete), "reset_failures": reset_failures,
        "effective_actions": actions,
    }
    write_json(output / "state_bank_manifest.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vagen-root", type=Path, required=True)
    parser.add_argument("--env-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--steps-per-episode", type=int, default=6)
    parser.add_argument("--target-rows", type=int, help="Stop after exactly this many complete state rows.")
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--episode-seeds-file", type=Path)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--split-seed", type=int, default=20260906)
    parser.add_argument("--max-per-scene", type=int, default=64)
    parser.add_argument("--enable-visibility-labels", action="store_true")
    parser.add_argument("--enable-collision-labels", action="store_true")
    parser.add_argument("--jsonl-path", type=Path)
    parser.add_argument("--gs-root")
    parser.add_argument("--render-backend", choices=("local", "client", "http", "none"))
    parser.add_argument("--gpu-device", type=int)
    args = parser.parse_args()
    metadata = build(args)
    freeze_protocol(args.output_dir, {"state_bank_hash": metadata["state_bank_hash"], "effective_env_config_hash": stable_hash(metadata["effective_env_config"])})
    check = validate_state_bank(args.output_dir / "states.jsonl")
    write_json(args.output_dir / "state_bank_validation.json", check)
    if not check["valid"]:
        raise SystemExit(f"BLOCKED: state-bank validation failed: {check['errors'][:3]}")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
