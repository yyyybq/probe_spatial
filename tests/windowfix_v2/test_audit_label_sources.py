from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image


MODULE_PATH = Path(__file__).resolve().parents[2] / "analyses/windowfix_v2/audit_label_sources.py"
SPEC = importlib.util.spec_from_file_location("audit_label_sources", MODULE_PATH)
audit_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit_module)


def _write_fixture(tmp_path: Path, mismatch: bool = False):
    manifest_dir = tmp_path / "manifests"
    raw_root = tmp_path / "raw"
    manifest_dir.mkdir()
    scene = raw_root / "scene0"
    for name in ("images", "refined_ins_ids", "depth"):
        (scene / name).mkdir(parents=True, exist_ok=True)
    frames = [f"frame_{index:06d}.jpg" for index in range(30)]
    for frame in frames:
        Image.new("RGB", (2, 2)).save(scene / "images" / frame)
        np.save(scene / "refined_ins_ids" / f"{frame}.npy", np.zeros((2, 2), np.int64))
        Image.fromarray(np.ones((2, 2), np.uint16)).save(scene / "depth" / f"{Path(frame).stem}.png")
    np.savez(
        scene / "scene_iphone_metadata.npz",
        images=np.array(frames, dtype=object),
        intrinsics=np.repeat(np.eye(3)[None], 30, axis=0),
        trajectories=np.repeat(np.eye(4)[None], 30, axis=0),
    )
    protocol = {
        "canonical_protocol": "test_v2",
        "canonical_window": {"observations": 28},
        "streaming": {"prefix_lengths": [8, 12, 16, 24]},
        "target": {
            "prefix_lengths": [8, 12, 16, 24],
            "future_horizons": [1, 2, 4],
        },
        "feature_namespaces": {"streaming": "STREAM_V2", "target": "TARGET_V2"},
        "expected_universe": {"canonical_windows": 1, "eligible_scenes": 1},
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    window = list(range(28))
    for prefix in (8, 12, 16, 24):
        row_window = list(window)
        if mismatch and prefix == 12:
            row_window[-1] = 29
        row = {
            "canonical_protocol": "test_v2",
            "canonical_obs_len": 28,
            "prefix_length": prefix,
            "sample_id": f"scene0_w000_p{prefix}",
            "scene_id": "scene0",
            "split": "train",
            "streaming_window_id": 0,
            "streaming_window_indices": row_window,
            "streaming_indices": row_window[:prefix],
            "streaming_tail": prefix - 1,
            "streaming_tail_raw": row_window[prefix - 1],
            "target_horizons": [1, 2, 4],
            "include_anchors": True,
            "streaming_feature_root": "aoss://bucket/STREAM_V2",
            "target_feature_root": "aoss://bucket/TARGET_V2",
            "valid_frames": frames,
            "valid_pose_indices": list(range(30)),
        }
        path = manifest_dir / f"inscene_c_h124_prefix{prefix}.jsonl"
        path.write_text(json.dumps(row) + "\n")
        meta = {
            "canonical_protocol": "test_v2",
            "canonical_obs_len": 28,
            "prefix_length": prefix,
            "action_horizons": [1, 2, 4],
            "sha256": audit_module.sha256_file(path),
        }
        Path(f"{path}.meta.json").write_text(json.dumps(meta))
    return manifest_dir, protocol_path, raw_root


def test_passes_exact_raw_label_lineage(tmp_path: Path):
    manifest_dir, protocol, raw_root = _write_fixture(tmp_path)
    result = audit_module.audit(
        manifest_dir=manifest_dir,
        protocol_path=protocol,
        prefixes=[8, 12, 16, 24],
        raw_scene_root=raw_root,
    )
    assert result["status"] == "pass"
    assert result["counts"]["manifest_rows"] == 4
    assert result["counts"]["future_target_references"] == 12
    assert result["counts"]["raw_frame_identities_checked"] == 28


def test_fails_cross_prefix_window_mismatch(tmp_path: Path):
    manifest_dir, protocol, raw_root = _write_fixture(tmp_path, mismatch=True)
    result = audit_module.audit(
        manifest_dir=manifest_dir,
        protocol_path=protocol,
        prefixes=[8, 12, 16, 24],
        raw_scene_root=raw_root,
    )
    assert result["status"] == "fail"
    assert result["problem_counts"]["cross_prefix_window_mismatch"] == 1
