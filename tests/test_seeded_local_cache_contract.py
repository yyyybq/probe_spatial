import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from features.noise_seed import canonical_noise_metadata, canonical_noise_sample
from features.run_inscene15k import (
    CANONICAL_STREAMING_PROTOCOL,
    CANONICAL_STREAMING_PROTOCOL_SHA256,
    canonical_protocol_metadata,
    seeded_feature_metadata_matches,
    stable_file_sha256,
    validate_seeded_cache_policy,
)


ROOT = Path(__file__).resolve().parents[1]
STREAM_NAMESPACE = "FEAT_STREAMING_CANONICAL_V2_SEEDED_V1"
MODEL_ID = "/models/wan-snapshot"


def _production_args(**overrides):
    values = {
        "no_cache_checksum": False,
        "mode": "streaming_prefix",
        "target_from_streaming_windows": False,
        "vfm": "wan",
        "prefix_lengths": [8, 12, 16, 24],
        "target_prefix_lengths": [8, 12, 16, 24],
        "target_horizons": [1, 2, 4],
        "streaming_window_obs_len": 28,
        "temporal_sampling": "motion_uniform",
        "streaming_motion_step": 0.35,
        "streaming_rotation_weight": 0.5,
        "streaming_window_stride": 8,
        "streaming_max_windows_per_scene": 4,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _noise_metadata():
    protocol = canonical_protocol_metadata(_production_args())
    sample = canonical_noise_sample(
        protocol=protocol,
        vfm="wan",
        model_id=MODEL_ID,
        scene="scene-a",
        sample_kind="streaming_prefix",
        window_id=2,
        prefix_len=8,
    )
    return canonical_noise_metadata([sample])


def _feature_metadata(path, digest):
    metadata = {
        "schema_version": 1,
        "frame_index_schema": "image_mask_intersection_v1",
        "vfm": "wan",
        "model_id": MODEL_ID,
        "mode": "streaming_prefix",
        "feature_namespace": STREAM_NAMESPACE,
        "timestep": 499,
        "prompt": ".",
        "resize": [480, 832],
        "num_frames": 81,
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": digest,
    }
    metadata.update(_noise_metadata())
    return metadata


def _matches(path, metadata, digest):
    return seeded_feature_metadata_matches(
        path,
        metadata,
        vfm="wan",
        model_id=MODEL_ID,
        mode="streaming_prefix",
        feature_namespace=STREAM_NAMESPACE,
        timestep=499,
        prompt=".",
        resize=(480, 832),
        num_frames=81,
        expected_noise_metadata=_noise_metadata(),
        actual_sha256=digest,
    )


def test_seeded_local_feature_requires_real_sha_and_exact_identity(tmp_path):
    path = tmp_path / "feature_t499_layer15.sft"
    path.write_bytes(b"seeded-feature-payload")
    digest = stable_file_sha256(path)
    metadata = _feature_metadata(path, digest)
    assert _matches(path, metadata, digest)

    mutations = {
        "schema_version": 0,
        "vfm": "cogvideox",
        "model_id": "/models/other",
        "mode": "target_isolated",
        "feature_namespace": "FEAT_STREAMING_CANONICAL_V2",
        "timestep": 999,
        "prompt": "different",
        "resize": [480, 720],
        "num_frames": 97,
        "file": "renamed.sft",
        "size_bytes": path.stat().st_size + 1,
        "sha256": "0" * 64,
        "feature_generation_contract": "legacy",
        "noise_seed_scheme": "legacy",
    }
    for field, value in mutations.items():
        changed = copy.deepcopy(metadata)
        changed[field] = value
        assert not _matches(path, changed, digest), field

    changed = copy.deepcopy(metadata)
    changed["noise_seed"] += 1
    assert not _matches(path, changed, digest)


def test_stable_sha_cache_does_not_accept_same_size_replacement(tmp_path):
    path = tmp_path / "feature_t499_layer15.sft"
    path.write_bytes(b"first-payload")
    cache = {}
    first = stable_file_sha256(path, cache=cache)
    assert first == hashlib.sha256(b"first-payload").hexdigest()

    replacement = tmp_path / "replacement.sft"
    replacement.write_bytes(b"other-payload")
    assert replacement.stat().st_size == path.stat().st_size
    os.replace(replacement, path)

    second = stable_file_sha256(path, cache=cache)
    assert second == hashlib.sha256(b"other-payload").hexdigest()
    assert second != first


def test_seeded_namespace_forbids_checksum_opt_out_and_protocol_drift():
    validate_seeded_cache_policy(_production_args(), STREAM_NAMESPACE)
    with pytest.raises(ValueError, match="reserved for canonical"):
        validate_seeded_cache_policy(
            _production_args(mode="normal"), STREAM_NAMESPACE
        )
    with pytest.raises(ValueError, match="does not support"):
        validate_seeded_cache_policy(
            _production_args(vfm="unknown_vfm"), STREAM_NAMESPACE
        )
    with pytest.raises(ValueError, match="forbids --no-cache-checksum"):
        validate_seeded_cache_policy(
            _production_args(no_cache_checksum=True), STREAM_NAMESPACE
        )
    with pytest.raises(ValueError, match="does not exactly match"):
        validate_seeded_cache_policy(
            _production_args(streaming_motion_step=0.36), STREAM_NAMESPACE
        )


def test_runtime_protocol_sha_is_exactly_derived_from_frozen_spec():
    spec = json.loads(
        (ROOT / "protocols/inscene_streaming_canonical_v2.json").read_text()
    )
    window = spec["canonical_window"]
    args = _production_args(
        prefix_lengths=spec["streaming"]["prefix_lengths"],
        target_prefix_lengths=spec["target"]["prefix_lengths"],
        target_horizons=spec["target"]["future_horizons"],
        streaming_window_obs_len=window["observations"],
        temporal_sampling=window["sampling"],
        streaming_motion_step=window["motion_step"],
        streaming_rotation_weight=window["rotation_weight"],
        streaming_window_stride=window["window_stride"],
        streaming_max_windows_per_scene=window["max_windows_per_scene"],
    )
    protocol = canonical_protocol_metadata(args)
    assert spec["canonical_protocol"] == CANONICAL_STREAMING_PROTOCOL
    assert protocol["protocol"] == CANONICAL_STREAMING_PROTOCOL
    assert protocol["protocol_sha256"] == CANONICAL_STREAMING_PROTOCOL_SHA256


def test_seeded_resume_keeps_authoritative_done_index_as_remote_skip_layer():
    source = (ROOT / "features/run_inscene15k.py").read_text(encoding="utf-8")
    assert "seeded-v1 extraction forbids EXTRACT_DONE_INDEX" not in source
    assert "return local_cache_complete(path) or done_index_contains(path)" in source
    assert "return done_index_contains(path) or (" in source
