from __future__ import annotations

import pytest

from features.run_inscene15k import merge_extra_metadata


def test_identical_canonical_metadata_can_be_repeated_by_gap_provenance():
    manifest = {
        "canonical_protocol_sha256": "a" * 64,
        "canonical_obs_len": 28,
    }

    merge_extra_metadata(
        manifest,
        {
            "canonical_protocol_sha256": "a" * 64,
            "canonical_obs_len": 28,
            "gap_contract": "CANONICAL_V2_GAP_WORKLIST_V1",
        },
    )

    assert manifest == {
        "canonical_protocol_sha256": "a" * 64,
        "canonical_obs_len": 28,
        "gap_contract": "CANONICAL_V2_GAP_WORKLIST_V1",
    }


def test_conflicting_metadata_remains_fail_closed():
    with pytest.raises(RuntimeError, match="canonical_obs_len"):
        merge_extra_metadata(
            {"canonical_obs_len": 28},
            {"canonical_obs_len": 24},
        )
