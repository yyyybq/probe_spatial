from argparse import Namespace
from pathlib import Path
from unittest.mock import patch
import sys
import types

from scripts.windowfix_v2.run_backend_determinism_canary import run_once


def test_first_scene_is_compared_after_another_scene(tmp_path):
    calls = []
    fake = types.ModuleType("canary_fake_extractor")
    fake.collect_scannetpp_scenes = lambda path: ["missing", "other", "third"]
    fake.main = lambda: calls.append(fake.collect_scannetpp_scenes(None))
    args = Namespace(project_root=tmp_path)
    argv = [sys.executable, "-m", fake.__name__, "--start", "0", "--end", "2"]
    with patch.dict(sys.modules, {fake.__name__: fake}), patch(
        "scripts.windowfix_v2.run_backend_determinism_canary.extractor_command",
        return_value=argv,
    ), patch(
        "scripts.windowfix_v2.run_backend_determinism_canary.subprocess.run",
        return_value=Namespace(returncode=0, stdout=""),
    ) as proc:
        result = run_once(args, mode="streaming_prefix", out_root=tmp_path / "out",
                          start=0, end=2, swap_first_two_scenes=True)
        cmd = proc.call_args.args[0]
        with patch.object(sys, "argv", ["-c", *cmd[3:]]):
            exec(cmd[2], {})
    assert calls == [["other", "missing", "third"]]
    assert result["swap_first_two_scenes"] is True
    assert result["range"] == [0, 2]
