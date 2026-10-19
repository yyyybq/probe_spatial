from __future__ import annotations

import hashlib
import json

import pytest

from analyses.windowfix_v2.build_streaming_gap_worklists import (
    CATALOG_SCHEMA,
    GapBuildError,
    validate_route,
)


def _route(status: str = "REEXTRACT_RAW_ID_MISMATCH") -> dict:
    raw_ids = [0, 2, 4, 6, 8, 10, 12, 14]
    return {
        "schema": CATALOG_SCHEMA,
        "status": status,
        "action": "REEXTRACT" if status.startswith("REEXTRACT_") else "WAIT_FOR_REMOTE_GATE",
        "scene_id": "scene-a",
        "window_id": 0,
        "prefix": 8,
        "feature_layer": 5,
        "vfm": "wan",
        "model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "timestep": 749,
        "canonical_raw_ids": raw_ids,
        "canonical_raw_ids_sha256": hashlib.sha256(
            json.dumps(raw_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "reason_codes": ["NO_EXACT_LEGACY_RAW_ID_SOURCE"],
        "unit_id": "unit-a",
    }


def test_explicit_reextract_route_is_independently_bound_to_canonical_window() -> None:
    window = tuple(range(0, 56, 2))
    value = validate_route(_route(), {("scene-a", 0): window})
    assert value["scene"] == "scene-a"
    assert value["prefix"] == 8
    assert value["layer"] == 5


def test_pending_remote_gate_is_never_scheduled() -> None:
    with pytest.raises(GapBuildError, match="non-explicit reextract"):
        validate_route(_route("PENDING_REMOTE_GATE"), {("scene-a", 0): tuple(range(0, 56, 2))})


def test_route_raw_ids_must_equal_frozen_source() -> None:
    route = _route()
    route["canonical_raw_ids"][0] = 1
    with pytest.raises(GapBuildError, match="disagree with source"):
        validate_route(route, {("scene-a", 0): tuple(range(0, 56, 2))})
