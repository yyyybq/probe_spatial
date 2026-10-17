from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from features.canonical_gap_worklist import (
    CONTRACT,
    CanonicalGapWorklist,
    GapWorklistError,
    canonical_json_sha,
    static_prefix_index_metadata,
    streaming_worklist_preprocess_identity,
    stream_unit_producer_metadata,
    stream_unit_semantic_matches,
    stream_unit_semantic_metadata,
)


SHA = "a" * 64


def _write(
    path: Path,
    *,
    mode: str,
    entries: list[dict],
    layers=(5, 10),
    vfm: str = "wan",
    model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    resize=(480, 832),
    num_frames: int = 81,
) -> str:
    payload = {
        "schema_version": 1,
        "contract": CONTRACT,
        "canonical_protocol": "inscene_streaming_canonical_v2_obs28_h124",
        "protocol_sha256": "b" * 64,
        "protocol_file_sha256": "c" * 64,
        "canonical_source_sha256": "d" * 64,
        "mode": mode,
        "vfm": vfm,
        "model_id": model_id,
        "timestep": 749,
        "feature_namespace": (
            "FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1"
            if mode == "streaming_prefix"
            else "FEAT_TARGET_CANONICAL_V2_GAP_DELTA_V1"
        ),
        "prompt_contract": {"mode": "exact", "value": "."},
        "preprocess_identity": (
            streaming_worklist_preprocess_identity(vfm, resize, num_frames)
            if mode == "streaming_prefix"
            else {"resize": list(resize), "num_frames": num_frames}
        ),
        "output_layers": list(layers),
        "entries": entries,
        "counts": {
            "entries": len(entries),
            "layer_items": sum(len(entry["layers"]) for entry in entries),
            "scenes": len({entry["scene"] for entry in entries}),
        },
    }
    payload["content_sha256"] = canonical_json_sha(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(
    path: Path,
    digest: str,
    mode: str,
    layers=(5, 10),
    vfm: str = "wan",
    model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    resize=(480, 832),
    num_frames: int = 81,
) -> CanonicalGapWorklist:
    return CanonicalGapWorklist.load(
        path,
        expected_sha256=digest,
        mode=mode,
        vfm=vfm,
        model_id=model_id,
        timestep=749,
        feature_namespace=(
            "FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1"
            if mode == "streaming_prefix"
            else "FEAT_TARGET_CANONICAL_V2_GAP_DELTA_V1"
        ),
        output_layers=layers,
        protocol_sha256="b" * 64,
        actual_prompt=".",
        resize=resize,
        num_frames=num_frames,
    )


def test_streaming_worklist_is_a_positive_layer_allowlist(tmp_path: Path) -> None:
    path = tmp_path / "stream.json"
    digest = _write(
        path,
        mode="streaming_prefix",
        entries=[
            {
                "scene": "scene-a",
                "window_id": 2,
                "prefix": 8,
                "layers": [5],
                "reason": "RAW_ID_MISMATCH",
            },
            {
                "scene": "scene-a",
                "window_id": 2,
                "prefix": 12,
                "layers": [5, 10],
                "reason": "PAYLOAD_GATE_FAILED",
            },
        ],
    )
    worklist = _load(path, digest, "streaming_prefix")

    assert worklist.stream_layers("scene-a", 2, 8) == {5}
    assert worklist.stream_layers("scene-a", 2, 12) == {5, 10}
    assert worklist.stream_layers("scene-a", 2, 16) == set()
    assert worklist.layer_item_count == 3
    assert worklist.reason_counts == {"PAYLOAD_GATE_FAILED": 2, "RAW_ID_MISMATCH": 1}


def test_target_worklist_selects_rows_per_layer(tmp_path: Path) -> None:
    path = tmp_path / "target.json"
    digest = _write(
        path,
        mode="target_isolated",
        entries=[
            {"scene": "scene-a", "frame_index": 7, "layers": [5], "reason": "ANCHOR_MISSING"},
            {
                "scene": "scene-a",
                "frame_index": 11,
                "layers": [5, 10],
                "reason": "PAYLOAD_GATE_FAILED",
            },
        ],
    )
    worklist = _load(path, digest, "target_isolated")

    assert worklist.target_frame_indices("scene-a") == (7, 11)
    assert worklist.target_layers("scene-a", 7) == {5}
    assert worklist.target_layers("scene-a", 11) == {5, 10}
    assert worklist.target_layers("scene-a", 99) == set()


def test_stream_payload_resume_identity_survives_later_worklist_union(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "phase1.json"
    first_sha = _write(
        first_path,
        mode="streaming_prefix",
        layers=(5,),
        entries=[{
            "scene": "scene-a", "window_id": 2, "prefix": 8,
            "layers": [5], "reason": "RAW_ID_MISMATCH",
        }],
    )
    first = _load(first_path, first_sha, "streaming_prefix", layers=(5,))
    second_path = tmp_path / "phase2.json"
    second_sha = _write(
        second_path,
        mode="streaming_prefix",
        layers=(5, 10),
        entries=[{
            "scene": "scene-a", "window_id": 2, "prefix": 8,
            "layers": [5, 10], "reason": "REMOTE_GATE_FAILED",
        }],
    )
    second = _load(second_path, second_sha, "streaming_prefix")
    record = {"window_id": 2, "valid_length": 8, "indices": list(range(8))}

    first_semantic = stream_unit_semantic_metadata(first, "scene-a", record, 5)
    second_semantic = stream_unit_semantic_metadata(second, "scene-a", record, 5)
    assert first.file_sha256 != second.file_sha256
    assert first_semantic == second_semantic
    sidecar = {**first_semantic, **stream_unit_producer_metadata(first)}
    assert stream_unit_semantic_matches(sidecar, second_semantic)
    assert sidecar["produced_by_gap_worklist_file_sha256"] == first.file_sha256


def test_full_prefix_index_identity_does_not_depend_on_campaign_worklist(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "phase1.json"
    first_sha = _write(
        first_path,
        mode="streaming_prefix",
        layers=(5,),
        entries=[{
            "scene": "scene-a", "window_id": 0, "prefix": 8,
            "layers": [5], "reason": "RAW_ID_MISMATCH",
        }],
    )
    second_path = tmp_path / "phase2.json"
    second_sha = _write(
        second_path,
        mode="streaming_prefix",
        layers=(10,),
        entries=[{
            "scene": "scene-a", "window_id": 3, "prefix": 24,
            "layers": [10], "reason": "REMOTE_GATE_FAILED",
        }],
    )
    first = _load(first_path, first_sha, "streaming_prefix", layers=(5,))
    second = _load(second_path, second_sha, "streaming_prefix", layers=(10,))

    assert first.file_sha256 != second.file_sha256
    assert static_prefix_index_metadata(first) == static_prefix_index_metadata(second)
    assert "worklist" not in json.dumps(static_prefix_index_metadata(first))


def test_qwen_no_pad_unit_resume_uses_numeric_prefix_frame_count(
    tmp_path: Path,
) -> None:
    path = tmp_path / "qwen.json"
    digest = _write(
        path,
        mode="streaming_prefix",
        layers=(-1,),
        vfm="qwen2_5_vl",
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        resize=(336, 336),
        num_frames=8,
        entries=[{
            "scene": "scene-a", "window_id": 2, "prefix": 16,
            "layers": [-1], "reason": "REMOTE_GATE_FAILED",
        }],
    )
    worklist = _load(
        path,
        digest,
        "streaming_prefix",
        layers=(-1,),
        vfm="qwen2_5_vl",
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        resize=(336, 336),
        num_frames=24,  # CLI maximum is intentionally not the unit identity.
    )
    record = {"window_id": 2, "valid_length": 16, "indices": list(range(16))}
    expected = stream_unit_semantic_metadata(worklist, "scene-a", record, -1)

    assert expected["frame_index_schema"] == "image_mask_intersection_v1"
    assert expected["num_frames"] == 16
    assert expected["input_num_frames"] == 16
    assert expected["temporal_padding"] == "none"
    # A sidecar from the first pass is accepted unchanged on the identical
    # worklist rerun, so the extraction loop will skip rather than overwrite.
    first_pass_sidecar = {**expected, **stream_unit_producer_metadata(worklist)}
    assert stream_unit_semantic_matches(first_pass_sidecar, expected)
    stale_fixed_eight = dict(first_pass_sidecar, num_frames=8, input_num_frames=8)
    assert not stream_unit_semantic_matches(stale_fixed_eight, expected)


@pytest.mark.parametrize("mutation", ["file_sha", "content_sha", "prompt", "duplicate"])
def test_worklist_fails_closed(tmp_path: Path, mutation: str) -> None:
    path = tmp_path / "bad.json"
    entry = {
        "scene": "scene-a",
        "window_id": 0,
        "prefix": 8,
        "layers": [5],
        "reason": "RAW_ID_MISMATCH",
    }
    digest = _write(path, mode="streaming_prefix", entries=[entry])
    expected = digest
    if mutation == "file_sha":
        expected = SHA
    else:
        payload = json.loads(path.read_text())
        if mutation == "content_sha":
            payload["content_sha256"] = SHA
        elif mutation == "prompt":
            payload["prompt_contract"] = {"mode": "exact", "value": ""}
            payload["content_sha256"] = canonical_json_sha(
                {key: value for key, value in payload.items() if key != "content_sha256"}
            )
        else:
            payload["entries"].append(dict(entry))
            payload["counts"]["entries"] = 2
            payload["counts"]["layer_items"] = 2
            payload["content_sha256"] = canonical_json_sha(
                {key: value for key, value in payload.items() if key != "content_sha256"}
            )
        path.write_text(json.dumps(payload, sort_keys=True) + "\n")
        expected = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(GapWorklistError):
        _load(path, expected, "streaming_prefix")
