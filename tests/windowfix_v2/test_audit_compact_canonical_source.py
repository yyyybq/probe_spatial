from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "analyses/windowfix_v2/audit_compact_canonical_source.py"
)
SPEC = importlib.util.spec_from_file_location("audit_compact_canonical_source", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def _fixture(tmp_path: Path, *, bad_window: bool = False):
    raw_root = tmp_path / "raw"
    scene = raw_root / "scene0"
    for name in ("images", "refined_ins_ids", "depth"):
        (scene / name).mkdir(parents=True, exist_ok=True)
    frames = [f"frame_{index:06d}.jpg" for index in range(30)]
    for frame in frames:
        Image.new("RGB", (2, 2)).save(scene / "images" / frame)
        np.save(scene / "refined_ins_ids" / f"{frame}.npy", np.zeros((2, 2), np.int64))
        Image.fromarray(np.ones((2, 2), np.uint16)).save(
            scene / "depth" / f"{Path(frame).stem}.png"
        )
    np.savez(
        scene / "scene_iphone_metadata.npz",
        images=np.array(frames, dtype=object),
        intrinsics=np.repeat(np.eye(3)[None], 30, axis=0),
        trajectories=np.repeat(np.eye(4)[None], 30, axis=0),
    )
    protocol = {
        "canonical_protocol": "inscene_streaming_canonical_v2_obs28_h124",
        "source": "scannetpp",
        "canonical_window": {"observations": 28, "sampling": "motion_uniform"},
        "streaming": {"prefix_lengths": [8, 12, 16, 24]},
        "target": {
            "prefix_lengths": [8, 12, 16, 24],
            "future_horizons": [1, 2, 4],
            "include_anchors": True,
        },
        "feature_namespaces": {"streaming": "STREAM_V2", "target": "TARGET_V2"},
        "expected_universe": {"canonical_windows": 1, "eligible_scenes": 1},
        "acceptance": {"legacy_features_valid": False, "audit_gate": "fail_closed"},
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    window = list(range(28))
    if bad_window:
        window[-1] = window[-2]
    source = tmp_path / "p24.jsonl"
    row = {
        "sample_id": "scene0_w000_p24",
        "scene_id": "scene0",
        "split": "train",
        "source": "scannetpp",
        "streaming_sampling": "motion_uniform",
        "streaming_window_id": 0,
        "streaming_window_indices": window,
        "streaming_indices": window[:24],
        "streaming_tail": 23,
        "streaming_tail_raw": window[23],
        "valid_frames": frames,
        "valid_pose_indices": list(range(30)),
        "streaming_feature_root": "legacy/ignored",
        "target_feature_root": "legacy/ignored",
    }
    source.write_text(json.dumps(row) + "\n")
    Path(f"{source}.meta.json").write_text(
        json.dumps(
            {
                "sha256": module.sha256_file(source),
                "prefix_lengths": [24],
                "action_horizons": [1, 2, 4],
                "num_samples": 2721,
            }
        )
    )
    return source, protocol_path, raw_root


def test_compact_source_derives_all_prefixes_and_raw_labels(tmp_path: Path):
    source, protocol, raw_root = _fixture(tmp_path)
    result = module.audit(
        source_manifest=source,
        protocol_path=protocol,
        prefixes=[8, 12, 16, 24],
        raw_scene_root=raw_root,
        namespace_lock_path=None,
    )
    assert result["status"] == "pass"
    assert result["counts"]["derived_prefix_rows"] == 4
    assert result["counts"]["input_frame_references"] == 60
    assert result["counts"]["future_target_references"] == 12
    assert result["counts"]["raw_frame_identities_checked"] == 28
    assert set(result["per_prefix_mapping_sha256"]) == {"8", "12", "16", "24"}


def test_compact_source_fails_non_strict_window(tmp_path: Path):
    source, protocol, raw_root = _fixture(tmp_path, bad_window=True)
    result = module.audit(
        source_manifest=source,
        protocol_path=protocol,
        prefixes=[8, 12, 16, 24],
        raw_scene_root=raw_root,
        namespace_lock_path=None,
    )
    assert result["status"] == "fail"
    assert result["problem_counts"]["source_contract"] == 1
