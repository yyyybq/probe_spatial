"""Frozen MVP protocol and strict state-bank validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import PROTOCOL_VERSION
from .io_utils import read_jsonl, stable_hash, write_json


DEFAULT_PROTOCOL: dict[str, Any] = {
    "protocol_version": PROTOCOL_VERSION,
    "input": {
        "history_window": 3,
        "include_history_actions": False,
        "include_pose_or_gt": False,
        "generate_answers": False,
        "prompt": "Active Spatial task text plus exactly three contiguous RGB observations",
    },
    "taps": ["vision_terminal", "llm_mid_last_prompt", "llm_final_norm_last_prompt"],
    "probes": ["linear", "mlp_d256"],
    "seeds": [41, 42, 43],
    "labels": [
        "relative_direction", "distance_bin", "visibility", "occlusion",
        "camera_rotation", "target_relation", "action_gain",
    ],
    "checks": [
        "majority", "label_shuffle", "task_text_only", "paired_sample_id",
        "scene_disjoint_split", "no_silent_checkpoint_fallback",
    ],
}


REQUIRED_SAMPLE_KEYS = {
    "sample_id", "scene_id", "episode_id", "split", "history", "task", "labels", "observability", "counterfactual_actions",
}


def freeze_protocol(output_dir: Path, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    protocol = {**DEFAULT_PROTOCOL}
    if extra:
        protocol.update(extra)
    protocol["protocol_hash"] = stable_hash(protocol)
    write_json(output_dir / "protocol.json", protocol)
    return protocol


def validate_state_bank(path: Path) -> dict[str, Any]:
    seen: set[str] = set()
    scene_split: dict[str, str] = {}
    rows = 0
    errors: list[str] = []
    for row in read_jsonl(path):
        rows += 1
        missing = REQUIRED_SAMPLE_KEYS - set(row)
        if missing:
            errors.append(f"{row.get('sample_id', '<missing>')}: missing {sorted(missing)}")
            continue
        sample_id = str(row["sample_id"])
        if sample_id in seen:
            errors.append(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        scene = str(row["scene_id"])
        split = str(row["split"])
        if scene in scene_split and scene_split[scene] != split:
            errors.append(f"scene split leakage: {scene} occurs in {scene_split[scene]} and {split}")
        scene_split[scene] = split
        history = row["history"]
        if not isinstance(history, list) or len(history) != 3:
            errors.append(f"{sample_id}: expected exactly three history observations")
            continue
        steps = [entry.get("env_step") for entry in history if isinstance(entry, dict)]
        if len(steps) != 3 or any(not isinstance(step, int) for step in steps):
            errors.append(f"{sample_id}: missing integer contiguous history env_step values")
        elif steps != list(range(steps[0], steps[0] + 3)):
            errors.append(f"{sample_id}: history is not contiguous: {steps}")
    return {"rows": rows, "scenes": len(scene_split), "errors": errors, "valid": not errors}
