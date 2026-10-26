"""The Hydra entry points must compose their configs from the command line.

Regression (2026-10-19): a helper inserted between ``@hydra.main`` and ``main`` in
vidfm3d/eval_a1_streaming.py took the decorator, so every CLI call failed with
``main() missing 1 required positional argument``. Unit tests that call helpers
directly cannot see that; ``--cfg job`` composes the config through the real
decorated entry point without loading data or checkpoints.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "entry, experiment",
    [
        ("vidfm3d/train.py", "inscene15k_streaming/view_consistency_dino_v1"),
        ("vidfm3d/eval_diag.py", "inscene15k_streaming/view_consistency_dino_v1"),
        ("vidfm3d/eval_a1_streaming.py", "inscene15k_streaming/streaming_depth_dino_v1"),
    ],
)
def test_entry_point_composes_config(entry, experiment):
    env = dict(os.environ, PROJECT_ROOT=str(REPO_ROOT))
    result = subprocess.run(
        [sys.executable, entry, f"experiment={experiment}", "--cfg", "job"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "experiment" in result.stdout or "model" in result.stdout
