"""Keep the public test suite independent of historical campaign artifacts."""

from pathlib import Path

import pytest


TEST_ROOT = Path(__file__).resolve().parent
# Release tests need no local campaign artifacts (2026-10-18).
#
# Internal campaign tests (cluster submission, upload and recovery, extraction
# campaigns, the 816 overlay audit and unreleased analyses) were untracked on
# 2026-10-18 and tracked again by 8a8948d (2026-10-19) as provenance. Several
# import frozen runtime copies, read untracked artifacts or call the cluster
# Python, so they are excluded before import unless --run-artifact-tests is
# given: marker deselection alone cannot keep missing local artifacts from
# breaking collection on a fresh checkout. This is the same list as the
# "Internal campaign tests" block of .gitignore.
INTERNAL_MODULES = frozenset({
    "test_a1_movement_task_registry.py",
    "test_active_repr_probe.py",
    "test_audit_wan_target_h124_canary.py",
    "test_cog_upload_backpressure_v1.py",
    "test_f3r_recovery_logging.py",
    "test_feature_readiness_10models.py",
    "test_minimax_production_handoff.py",
    "test_probe_v2_816_audit.py",
    "test_se3_latent_mvp.py",
    "test_se3_protocol_v2.py",
    "test_submit_tstep_probe_canonical_v2.py",
    "test_tstep_canonical_alignment.py",
    "test_tstep_canonical_v2_canary_receipt.py",
    "test_tstep_extract_campaign_v2_hardening.py",
    "test_tstep_sync_loop.py",
    "test_upload_recovery_20260923.py",
    "test_wan_p12_l20_c1_b2_study.py",
    "test_wan_target_h124_extend_exact.py",
    "test_wan_target_h124_sco_execution.py",
    "test_wan_vjepa2_p1.py",
    "test_wan_vjepa2_v2_data_routes.py",
    "test_wan_vjepa2_v2_structural.py",
    "windowfix_v2/test_aoss_reference_gate_shard_plan.py",
    "windowfix_v2/test_aoss_streaming_reference_gate.py",
    "windowfix_v2/test_atomic_temp_rename_race.py",
    "windowfix_v2/test_audit_compact_canonical_source.py",
    "windowfix_v2/test_audit_label_sources.py",
    "windowfix_v2/test_backend_determinism_canary.py",
    "windowfix_v2/test_build_reextract_plan.py",
    "windowfix_v2/test_canary_first_scene_order.py",
    "windowfix_v2/test_canonical_gap_aoss_lifecycle.py",
    "windowfix_v2/test_canonical_gap_lifecycle_smoke.py",
    "windowfix_v2/test_classify_feature_reuse.py",
    "windowfix_v2/test_compact_namespace_lock.py",
    "windowfix_v2/test_feature_inventory.py",
    "windowfix_v2/test_finalize_aoss_streaming_gate.py",
    "windowfix_v2/test_gap_backend_canary_target_subset.py",
    "windowfix_v2/test_gap_receipt_permissions.py",
    "windowfix_v2/test_hybrid_reuse_plan.py",
    "windowfix_v2/test_namespace_lock.py",
    "windowfix_v2/test_qwen_canonical_extractor.py",
    "windowfix_v2/test_reference_overlay_plan.py",
    "windowfix_v2/test_sco_aoss_reference_gate_jobs.py",
    "windowfix_v2/test_sco_feature_job_preparation.py",
    "windowfix_v2/test_sco_gap_jobs.py",
    "windowfix_v2/test_streaming_gap_worklists.py",
    "windowfix_v2/test_target_accounting_report.py",
    "windowfix_v2/test_target_route_audit_merge.py",
    "windowfix_v2/test_target_route_audit_worker.py",
    "windowfix_v2/test_target_row_merge.py",
    "test_probe_validity_b2_order.py",
})
ARTIFACT_MODULES: set[str] = set(INTERNAL_MODULES)
# An internal checkout may list further modules (one path per line, relative to
# tests/) in this untracked file.
_INTERNAL_LIST = TEST_ROOT / "internal_artifact_modules.txt"
if _INTERNAL_LIST.is_file():
    ARTIFACT_MODULES |= {
        line.strip() for line in _INTERNAL_LIST.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }


def pytest_addoption(parser):
    parser.addoption(
        "--run-artifact-tests",
        action="store_true",
        default=False,
        help="Include historical campaign tests; requires their original local artifacts.",
    )


def pytest_ignore_collect(collection_path, config):
    if _artifact_module(collection_path) and not config.getoption("--run-artifact-tests"):
        return True
    return None


def _artifact_module(path):
    return (
        path.is_relative_to(TEST_ROOT)
        and path.relative_to(TEST_ROOT).as_posix() in ARTIFACT_MODULES
    )


def pytest_collection_modifyitems(config, items):
    selected, deselected = [], []
    for item in items:
        if _artifact_module(item.path):
            item.add_marker(pytest.mark.artifact)
        if item.get_closest_marker("artifact") and not config.getoption("--run-artifact-tests"):
            deselected.append(item)
        else:
            selected.append(item)
    if deselected:
        items[:] = selected
        config.hook.pytest_deselected(items=deselected)
