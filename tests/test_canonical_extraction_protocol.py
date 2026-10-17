import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from features.canonical_gap_worklist import CONTRACT, canonical_json_sha
from features.run_inscene15k import (
    CANONICAL_STREAMING_PROTOCOL,
    canonical_protocol_metadata,
    canonical_windows_for_scene,
    main,
    streaming_prefix_records_for_scene,
    target_selection_for_scene,
    validate_canonical_window_args,
)


def _args(**overrides):
    values = dict(
        mode="streaming_prefix",
        temporal_sampling="motion_uniform",
        streaming_window_obs_len=28,
        streaming_motion_step=1.0,
        streaming_rotation_weight=0.5,
        streaming_window_stride=8,
        streaming_max_windows_per_scene=4,
        prefix_lengths=[8, 12, 16, 24],
        target_prefix_lengths=[8, 12, 16, 24],
        target_horizons=[1, 2, 4],
        prefix_min_len=1,
        prefix_max_len=24,
        prefix_stride=1,
        num_frames=81,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _scene(num_frames=100):
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], num_frames, axis=0)
    poses[:, 0, 3] = np.arange(num_frames, dtype=np.float32)
    return {
        "source": "scannetpp",
        "img_files": [f"frame_{i:06d}.jpg" for i in range(num_frames)],
        "poses_c2w": poses,
    }


def test_streaming_and_target_share_the_same_28_observation_windows():
    args = _args()
    scene = _scene()
    windows = canonical_windows_for_scene(scene, args)
    records = streaming_prefix_records_for_scene(scene, args)
    targets, provenance = target_selection_for_scene(scene, args)

    assert len(windows) == 4
    assert len(records) == 4 * 4
    by_window = {window["window_id"]: window for window in windows}
    for record in records:
        window = by_window[record["window_id"]]
        assert len(record["window_indices"]) == 28
        assert record["window_indices"] == window["indices"]
        assert record["indices"] == window["indices"][: record["valid_length"]]
        assert record["canonical_protocol"] == CANONICAL_STREAMING_PROTOCOL
        assert record["canonical_obs_len"] == 28

    expected_targets = set()
    for item in provenance:
        window = by_window[item["window_id"]]
        assert item["frame_index"] == window["indices"][item["position"]]
        expected_targets.add(item["frame_index"])
    assert set(targets) == expected_targets

    roles = {
        (item["window_id"], item["prefix_length"], item["role"], item["horizon"])
        for item in provenance
    }
    for window in windows:
        for prefix in (8, 12, 16, 24):
            assert (window["window_id"], prefix, "anchor", 0) in roles
            for horizon in (1, 2, 4):
                assert (window["window_id"], prefix, "future", horizon) in roles


def test_protocol_rejects_a_window_shorter_than_prefix_plus_horizon():
    args = _args(streaming_window_obs_len=27)
    with pytest.raises(ValueError, match="need >= 28"):
        validate_canonical_window_args(args)


def test_protocol_hash_is_stable_and_identifies_v2():
    first = canonical_protocol_metadata(_args(streaming_motion_step=0.35))
    second = canonical_protocol_metadata(_args(streaming_motion_step=0.35))
    assert first == second
    assert first["protocol"] == "inscene_streaming_canonical_v2_obs28_h124"
    assert first["protocol_sha256"] == (
        "215c720c8537f4dfae6243bedcc7339fbb7e3ba8a6ca2dcaf04f49cd4b738e4f"
    )


def test_target_gap_precheck_does_not_close_over_unbound_target_ok(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    scene_dir = data_root / "processed_scannetpp_v2" / "scene-a"
    scene_dir.mkdir(parents=True)
    out_root = tmp_path / "FEAT_TARGET_CANONICAL_V2_GAP_DELTA_V1"
    out_root.mkdir()

    poses = np.repeat(np.eye(4, dtype=np.float32)[None], 100, axis=0)
    poses[:, 0, 3] = np.arange(100, dtype=np.float32)
    scene = {
        "source": "scannetpp",
        "scene_dir": str(scene_dir),
        "img_files": [f"frame_{i:06d}.jpg" for i in range(100)],
        "poses_c2w": poses,
    }
    args = _args(
        mode="target_isolated",
        streaming_motion_step=0.35,
        num_frames=64,
        target_from_streaming_windows=True,
    )
    targets, _ = target_selection_for_scene(scene, args)
    assert targets
    protocol = canonical_protocol_metadata(args)
    worklist_path = tmp_path / "target_gap.json"
    payload = {
        "schema_version": 1,
        "contract": CONTRACT,
        "canonical_protocol": CANONICAL_STREAMING_PROTOCOL,
        "protocol_sha256": protocol["protocol_sha256"],
        "protocol_file_sha256": "c" * 64,
        "canonical_source_sha256": "d" * 64,
        "mode": "target_isolated",
        "vfm": "dino",
        "model_id": "facebook/dinov2-large",
        "timestep": 749,
        "feature_namespace": "FEAT_TARGET_CANONICAL_V2_GAP_DELTA_V1",
        "prompt_contract": {"mode": "not_applicable", "value": None},
        "preprocess_identity": {"resize": [420, 728], "num_frames": 64},
        "output_layers": [0],
        "entries": [{
            "scene": "scene-a",
            "frame_index": int(targets[0]),
            "layers": [0],
            "reason": "CANONICAL_TARGET_RAW_ID_UNROUTED",
        }],
        "counts": {"entries": 1, "layer_items": 1, "scenes": 1},
    }
    payload["content_sha256"] = canonical_json_sha(payload)
    worklist_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    digest = hashlib.sha256(worklist_path.read_bytes()).hexdigest()

    monkeypatch.setattr(
        "features.run_inscene15k.collect_scannetpp_scenes",
        lambda _source_path: [scene],
    )
    monkeypatch.setenv("FEATURE_NAMESPACE", "FEAT_TARGET_CANONICAL_V2_GAP_DELTA_V1")
    monkeypatch.setenv("EXTRACT_CACHE_CHECK_ONLY", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_inscene15k",
            "--vfm", "dino",
            "--mode", "target_isolated",
            "--target-from-streaming-windows",
            "--data-root", str(data_root),
            "--out-root", str(out_root),
            "--source", "scannetpp",
            "--t", "749",
            "--model-id", "facebook/dinov2-large",
            "--output-layers", "0",
            "--num-frames", "64",
            "--streaming-window-obs-len", "28",
            "--streaming-motion-step", "0.35",
            "--canonical-gap-worklist", str(worklist_path),
            "--canonical-gap-worklist-sha256", digest,
        ],
    )

    main()
