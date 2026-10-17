from pathlib import Path
import runpy
import pytest

from scripts.windowfix_v2.canonical_gap_aoss_adapter_atomicrace_v2 import (
    atomic_temps_for_worklist, sha256_file, GapSyncError,
)

def setup_case(tmp_path):
    helper = runpy.run_path(str(Path(__file__).with_name("test_canonical_gap_aoss_lifecycle.py")))
    worklist = tmp_path / "worklist.json"
    helper["_worklist"](worklist)
    root = tmp_path / "payload"
    candidate = root / "scene-a/window_0000/prefix_000007/.feature_t749_layer5.sft.tmp-1"
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(b"partial")
    return worklist, root, candidate

def test_atomic_rename_after_directory_listing(tmp_path, monkeypatch):
    worklist, root, candidate = setup_case(tmp_path)
    original = Path.lstat
    def race(path, *a, **kw):
        if path == candidate:
            raise FileNotFoundError(str(path))
        return original(path, *a, **kw)
    monkeypatch.setattr(Path, "lstat", race)
    assert atomic_temps_for_worklist(root=root, worklist_path=worklist,
                                    worklist_sha256=sha256_file(worklist)) == []

def test_existing_symlink_still_rejected(tmp_path):
    worklist, root, candidate = setup_case(tmp_path)
    candidate.unlink()
    candidate.symlink_to(tmp_path / "absent")
    with pytest.raises(GapSyncError, match="unsafe"):
        atomic_temps_for_worklist(root=root, worklist_path=worklist,
                                 worklist_sha256=sha256_file(worklist))
