from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[2] / "analyses/windowfix_v2/build_compact_namespace_lock.py"
SPEC = importlib.util.spec_from_file_location("build_compact_namespace_lock", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def test_rejects_noncanonical_protocol(tmp_path: Path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"canonical_protocol": "wrong"}))
    source = tmp_path / "source.jsonl"
    source.write_text("")
    try:
        module.build(protocol, source, "aoss://bucket")
    except ValueError as exc:
        assert "unexpected canonical protocol" in str(exc)
    else:
        raise AssertionError("noncanonical protocol must fail")
