"""Keep the public test suite independent of historical campaign artifacts."""

from pathlib import Path

import pytest


TEST_ROOT = Path(__file__).resolve().parent
# These modules import frozen runtime copies or assert against completed runs.
# Exclude them before import: marker deselection alone cannot prevent missing
# local artifacts from breaking collection on a fresh checkout.
ARTIFACT_MODULES = {
    "test_aether_streaming_temporal_mapping.py",
    "test_cog_upload_backpressure_v1.py",
    "test_f3r_recovery_logging.py",
    "test_upload_recovery_20260923.py",
    "test_wan_p12_l20_c1_b2_study.py",
    "windowfix_v2/test_canonical_gap_lifecycle_smoke.py",
    "windowfix_v2/test_target_accounting_report.py",
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
