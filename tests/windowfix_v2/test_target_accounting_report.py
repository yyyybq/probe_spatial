from __future__ import annotations

import json
from pathlib import Path

import pytest

from analyses.windowfix_v2 import build_target_accounting_report as accounting


@pytest.fixture(scope="module")
def report() -> dict:
    return accounting.build_report(
        plan_path=accounting.DEFAULT_PLAN.resolve(),
        summary_path=accounting.DEFAULT_SUMMARY.resolve(),
        audit_dir=accounting.DEFAULT_AUDIT_DIR.resolve(),
    )


def test_frozen_target_accounting_is_set_exact(report: dict) -> None:
    current = report["current_exhaustive_status_partition"]
    assert current["row_layer_universe"] == 1_205_664
    assert (
        current["certified_local_rows"]
        + current["pending_remote_candidate_rows"]
        + current["definite_gap_rows"]
        == current["row_layer_universe"]
    )
    assert current["remote_rows_reusable_after_full_audit"] is None
    assert current["duplicate_owner_rows"] == 0
    assert current["conflicting_owner_rows"] == 0

    difference = report["difference_477620_minus_360426"]
    assert difference["value"] == 117_194
    assert difference["raw_id_rows_per_layer"] == 5_327
    assert difference["remote_layer_count"] == 22
    assert len(difference["by_layer"]) == 22
    assert sum(item["row_layer_items"] for item in difference["by_layer"]) == 117_194

    compatibility = report["compatibility_accounting"]
    roles = compatibility["anchor_role_membership"]
    assert roles["union_rows"] == 5_327
    assert roles["p8_anchor_rows"] + roles["p24_anchor_rows"] - roles[
        "p8_and_p24_anchor_overlap_rows"
    ] == roles["union_rows"]
    assert compatibility["anchor_gap_contains_future_role_rows"] == 0
    assert compatibility["known_shape_incompatible_rows"] == 0

    lower = report["best_case_reextract_lower_bound"]
    assert lower["value"] == 477_620
    assert sum(lower["components"].values()) == lower["value"]
    assert lower["authoritative_as_current_definite_gap"] is False
    assert report["legacy_field_disposition"]["stale"] is False
    assert report["actual_index_evidence"]["equalities_proved"] == {
        "flat_minimax_equals_canonical_future_only_every_scene": True,
        "flat_wan_equals_flat_minimax_every_scene": True,
        "wan_dot_is_subset_of_flat_wan_every_scene": True,
    }


def test_target_accounting_markdown_preserves_pending_semantics(report: dict) -> None:
    rendered = accounting.render_markdown(report)
    assert "PENDING remote" not in rendered  # keep the table's human wording stable
    assert "Pending remote" in rendered
    assert "not FAIL" in rendered
    assert "not stale" in rendered
    assert "117194 = 5327 * 22" in rendered


def test_immutable_target_accounting_artifact_is_self_and_input_bound() -> None:
    path = accounting.DEFAULT_OUTPUT_JSON.resolve()
    artifact = json.loads(path.read_text(encoding="utf-8"))
    recorded = artifact.pop("content_sha256")
    assert accounting.canonical_sha(artifact) == recorded
    for receipt in artifact["input_receipts"].values():
        input_path = Path(receipt["path"])
        assert input_path.is_file() and not input_path.is_symlink()
        assert accounting.sha256_file(input_path) == receipt["file_sha256"]
    assert artifact["report_id"] == "target_accounting_r6_20260831"
    assert artifact["scope"] == {
        "aoss_contacted": False,
        "feature_payload_written": False,
        "old_artifacts_modified": False,
        "target_materialized": False,
    }
