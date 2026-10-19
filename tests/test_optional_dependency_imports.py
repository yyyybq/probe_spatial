"""The streaming evaluators must import with the pinned CPU test environment.

`pytorch3d` and `trimesh` are only needed by the point-cloud metrics in
`vidfm3d.utils.eval_utils`; they are not in `requirements-dev.txt`. Before
2026-10-19 the module imported them at import time, so `vidfm3d.eval_a1_streaming`
(and `tests/test_nonfinite_handling.py`) failed to import on a clean checkout
even though the A1 evaluator only uses `build_mask` and `align_pmaps`.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Setting ``sys.modules[name] = None`` makes ``import name`` raise ImportError,
# which simulates the package being absent even when it is installed.
_IMPORT_SCRIPT = """
import sys
for name in ("pytorch3d", "pytorch3d.ops", "pytorch3d.ops.knn",
             "pytorch3d.structures", "pytorch3d.structures.pointclouds", "trimesh"):
    sys.modules[name] = None
import vidfm3d.utils.eval_utils as eval_utils
import vidfm3d.eval_a1_streaming
assert callable(eval_utils.build_mask) and callable(eval_utils.align_pmaps)
print("OK")
"""


@pytest.mark.parametrize("module", ["vidfm3d.utils.eval_utils", "vidfm3d.eval_a1_streaming"])
def test_streaming_evaluator_imports_without_point_cloud_extras(module):
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_SCRIPT],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("OK")


def test_point_cloud_metrics_still_require_pytorch3d_only_when_called():
    import torch

    import vidfm3d.utils.eval_utils as eval_utils

    pytest.importorskip("pytorch3d")  # skip the positive path when not installed
    x = torch.zeros(1, 4, 3)
    cham_x, cham_y, _, _ = eval_utils.chamfer_dist(x, x)
    assert cham_x.shape == (1, 4) and cham_y.shape == (1, 4)
